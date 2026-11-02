import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import format_table
from src.benchmarks import bs_delta, leland_delta
from src.loss import EntropicRiskLoss, empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths
from src.studies import aggregate, format_pm

SEEDS = (42, 43, 44, 45, 46)
GBM_METRICS = ("mean_pnl", "std_pnl", "cvar_95", "cvar_99", "entropic", "mean_cost")


def decision_taus(config: Config) -> torch.Tensor:
    steps = torch.arange(config.n_steps, dtype=torch.float32)
    return (config.n_steps - steps) * config.dt


def metrics(pnl: torch.Tensor, costs: torch.Tensor, lam: float) -> dict[str, float]:
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(unbiased=True)),
        "cvar_95": float(empirical_cvar(pnl, 0.95)),
        "cvar_99": float(empirical_cvar(pnl, 0.99)),
        "entropic": float(EntropicRiskLoss(lam)(pnl)),
        "mean_cost": float(costs.mean()),
    }


def gbm_records(config: Config) -> dict[str, list[dict[str, float]]]:
    # Leland under both cost conventions, and both benchmarks with a liquidation charge at T
    taus = decision_taus(config)
    out: dict[str, list[dict[str, float]]] = {
        "Leland, one-way c (repo)": [],
        "Leland, round-trip 2c": [],
        "Black-Scholes + liquidation at T": [],
        "Leland c + liquidation at T": [],
    }
    for seed in SEEDS:
        spots, _ = simulate_paths(config, config.n_eval_paths, seed=seed + 1)
        S = spots[:, :-1]
        bs = bs_delta(S, config.K, config.r, config.sigma, taus)
        le_c = leland_delta(S, config.K, config.r, config.sigma, taus, config.cost_rate, config.dt)
        le_2c = leland_delta(S, config.K, config.r, config.sigma, taus, 2.0 * config.cost_rate, config.dt)
        for name, deltas, liquidate in (
            ("Leland, one-way c (repo)", le_c, False),
            ("Leland, round-trip 2c", le_2c, False),
            ("Black-Scholes + liquidation at T", bs, True),
            ("Leland c + liquidation at T", le_c, True),
        ):
            result = hedge_pnl(spots, deltas, config.K, config.cost_rate, return_path=False)
            pnl, costs = result["pnl"], result["costs"]
            if liquidate:
                closing = config.cost_rate * spots[:, -1] * deltas[:, -1].abs()
                pnl, costs = pnl - closing, costs + closing
            out[name].append(metrics(pnl, costs, config.lambda_entropic))
    return out


def heston_records(config: Config) -> dict[str, dict[str, list[dict[str, float]]]]:
    # Black-Scholes at the fixed sqrt(v0) against Black-Scholes at the current sqrt(v_k)
    taus = decision_taus(config)
    out: dict[str, dict[str, list[dict[str, float]]]] = {}
    paths = {seed: simulate_paths(config, config.n_eval_paths, seed=seed + 1) for seed in SEEDS}
    for cost in config.cost_levels:
        level = {"fixed": [], "instantaneous": []}
        for seed in SEEDS:
            spots, variances = paths[seed]
            S = spots[:, :-1]
            vol_now = torch.sqrt(torch.clamp(variances[:, :-1], min=1e-8))
            for name, sigma in (("fixed", config.benchmark_sigma), ("instantaneous", vol_now)):
                deltas = bs_delta(S, config.K, config.r, sigma, taus)
                pnl = hedge_pnl(spots, deltas, config.K, float(cost), return_path=False)["pnl"]
                level[name].append(
                    {
                        "cvar_95": float(empirical_cvar(pnl, 0.95)),
                        "cvar_99": float(empirical_cvar(pnl, 0.99)),
                    }
                )
        out[f"{cost:.4f}"] = level
    return out


def main() -> int:
    output = sys.argv[1] if len(sys.argv) > 1 else "results/extra_benchmarks.json"
    gbm = Config()
    heston = replace(Config(), simulator="heston")

    print("GBM benchmark variants at 1 percent cost, mean +/- std over 5 evaluation sets")
    gbm_runs = gbm_records(gbm)
    gbm_summary = {name: aggregate(runs, GBM_METRICS) for name, runs in gbm_runs.items()}
    print(
        format_table(
            ("Strategy", "Mean PnL", "Std", "CVaR 95", "CVaR 99", "Entropic", "Mean cost"),
            [
                (name,) + tuple(format_pm(s[m]["mean"], s[m]["std"]) for m in GBM_METRICS)
                for name, s in gbm_summary.items()
            ],
        )
    )

    deep_heston = json.loads(Path("results/heston_multiseed.json").read_text(encoding="utf-8"))
    heston_runs = heston_records(heston)
    heston_summary: dict[str, dict] = {}
    rows = []
    for key, level in heston_runs.items():
        fixed = aggregate(level["fixed"], ("cvar_95", "cvar_99"))
        inst = aggregate(level["instantaneous"], ("cvar_95", "cvar_99"))
        deep = deep_heston["levels"][key]["summary"]
        gaps = {}
        for q in ("95", "99"):
            # Paired by seed: deep and benchmark share each evaluation path set
            diffs = [
                b[f"cvar_{q}"] - d[f"deep_cvar_{q}"]
                for b, d in zip(level["instantaneous"], deep_heston["levels"][key]["records"])
            ]
            mean = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1))
            gaps[q] = {"gap": mean, "t": mean / (sd / math.sqrt(len(diffs)))}
        heston_summary[key] = {"fixed": fixed, "instantaneous": inst, "gap_vs_deep": gaps}
        rows.append(
            (
                f"{100.0 * float(key):5.2f}%",
                format_pm(deep["deep_cvar_95"]["mean"], deep["deep_cvar_95"]["std"]),
                format_pm(fixed["cvar_95"]["mean"], fixed["cvar_95"]["std"]),
                format_pm(inst["cvar_95"]["mean"], inst["cvar_95"]["std"]),
                f"{gaps['95']['gap']:+.3f} (t {gaps['95']['t']:.1f})",
                format_pm(deep["deep_cvar_99"]["mean"], deep["deep_cvar_99"]["std"]),
                format_pm(inst["cvar_99"]["mean"], inst["cvar_99"]["std"]),
                f"{gaps['99']['gap']:+.3f} (t {gaps['99']['t']:.1f})",
            )
        )
    print()
    print("Heston: Black-Scholes at fixed sqrt(v0) against current sqrt(v_k), 5 evaluation sets")
    print(
        format_table(
            (
                "Cost",
                "Deep CVaR95",
                "BS fixed CVaR95",
                "BS current CVaR95",
                "Gap 95 (paired t)",
                "Deep CVaR99",
                "BS current CVaR99",
                "Gap 99 (paired t)",
            ),
            rows,
        )
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "eval_seed_offset": 1,
                "n_eval_paths": gbm.n_eval_paths,
                "gbm_cost_rate": gbm.cost_rate,
                "gbm": {"summary": gbm_summary, "per_seed": gbm_runs},
                "heston": {"summary": heston_summary, "per_seed": heston_runs},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"extra benchmarks written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
