import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from config import Config
from src.heston import HestonParams, black_scholes_call_price, heston_call_price
from src.loss import empirical_cvar
from src.simulators import GBMSimulator, HestonSimulator, PaperHestonSimulator
from src.visualize import plot_eda_market, plot_eda_moments, plot_eda_results

PAPER_Q = 1.69
BS_Q = 2.2872


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory data analysis")
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--results-dir", default="results/paper")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def describe(name: str, values: np.ndarray) -> dict[str, float]:
    # Distributional summary of one simulated quantity
    centred = values - values.mean()
    std = float(values.std(ddof=1))
    skew = float(np.mean(centred**3) / max(std**3, 1e-18))
    kurt = float(np.mean(centred**4) / max(std**4, 1e-18))
    return {
        "name": name,
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": std,
        "se_mean": std / math.sqrt(values.size),
        "skew": skew,
        "excess_kurtosis": kurt - 3.0,
        "min": float(values.min()),
        "q01": float(np.percentile(values, 1.0)),
        "q05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "q95": float(np.percentile(values, 95.0)),
        "q99": float(np.percentile(values, 99.0)),
        "max": float(values.max()),
    }


def eda_market(paths: int) -> dict[str, object]:
    # Terminal distribution and martingale sanity for each simulator
    config = Config()
    config.validate()
    out: dict[str, object] = {}

    gbm = GBMSimulator(
        S0=config.S0, r=config.r, sigma=config.sigma, T=config.T,
        n_steps=config.n_steps, seed=11,
    )
    spots = gbm.simulate(paths, seed=11).numpy().astype(np.float64)
    terminal = spots[:, -1]
    log_ret = np.log(spots[:, -1] / spots[:, 0])
    out["gbm"] = {
        "terminal": describe("GBM terminal spot", terminal),
        "log_return": describe("GBM total log return", log_ret),
        "payoff": describe("GBM call payoff", np.maximum(terminal - config.K, 0.0)),
        "martingale": {
            "mean_spot": float(terminal.mean()),
            "se": float(terminal.std(ddof=1) / math.sqrt(paths)),
            "target": config.S0,
            "theoretical_call": BS_Q,
        },
    }

    heston = HestonSimulator(
        S0=config.S0, r=config.r, v0=config.heston_v0, kappa=config.heston_kappa,
        theta=config.heston_theta, xi=config.heston_xi, rho=config.heston_rho,
        T=config.T, n_steps=config.n_steps, seed=11,
    )
    hs, hv = heston.simulate(paths, seed=11)
    hs = hs.numpy().astype(np.float64)
    hv = hv.numpy().astype(np.float64)
    ht = hs[:, -1]
    out["heston"] = {
        "terminal": describe("Heston terminal spot", ht),
        "variance": describe("Heston variance path mean", hv.mean(axis=1)),
        "variance_terminal": describe("Heston terminal variance", hv[:, -1]),
        "payoff": describe("Heston call payoff", np.maximum(ht - config.K, 0.0)),
        "zero_fraction": float((hv < 1e-8).mean()),
        "martingale": {
            "mean_spot": float(ht.mean()),
            "se": float(ht.std(ddof=1) / math.sqrt(paths)),
            "target": config.S0,
        },
    }

    paper = PaperHestonSimulator(S0=config.S0, T=config.T, n_steps=config.n_steps, seed=11)
    ps, pv = paper.simulate(paths, seed=11)
    pt = ps[:, -1]
    swap = paper.variance_swap(pv)
    params = HestonParams()
    out["paper_heston"] = {
        "terminal": describe("Paper Heston terminal spot", pt),
        "variance": describe("Paper Heston variance path mean", pv.mean(axis=1)),
        "variance_terminal": describe("Paper Heston terminal variance", pv[:, -1]),
        "payoff": describe("Paper Heston call payoff", np.maximum(pt - config.K, 0.0)),
        "zero_fraction": float((pv < 1e-8).mean()),
        "feller": float(paper.feller),
        "fair_strike": float(swap[:, -1].mean()),
        "martingale": {
            "mean_spot": float(pt.mean()),
            "se": float(pt.std(ddof=1) / math.sqrt(paths)),
            "target": config.S0,
            "theoretical_call": float(
                heston_call_price(config.T, config.S0, params.v0, config.K, params)
            ),
        },
    }

    # Integrated variance against the analytic fair strike
    analytic = params.theta * config.T + (params.v0 - params.theta) * (
        1.0 - math.exp(-params.kappa * config.T)
    ) / params.kappa
    out["paper_heston"]["fair_strike_analytic"] = float(analytic)
    # Subsample the terminal spots so the figures can show real densities
    rng = np.random.default_rng(5)
    keep = rng.choice(paths, size=min(paths, 40_000), replace=False)
    out["samples"] = {
        "gbm": terminal[keep].tolist(),
        "heston": ht[keep].tolist(),
        "paper_heston": pt[keep].tolist(),
        "variance_gbm": None,
        "variance_heston": hv[keep][:, -1].tolist(),
        "variance_paper": pv[keep][:, -1].tolist(),
    }
    out["paper_heston"]["fair_strike_error"] = float(
        abs(out["paper_heston"]["fair_strike"] - analytic)
    )

    # Discretisation bias of the Euler scheme against the exact scheme
    euler_mean_payoff = float(np.maximum(ht - config.K, 0.0).mean())
    exact_mean_payoff = float(np.maximum(pt - config.K, 0.0).mean())
    out["scheme_comparison"] = {
        "euler_mean_payoff": euler_mean_payoff,
        "exact_mean_payoff": exact_mean_payoff,
        "euler_variance_zero_fraction": out["heston"]["zero_fraction"],
        "exact_variance_zero_fraction": out["paper_heston"]["zero_fraction"],
    }
    return out


def eda_results(results_dir: Path) -> dict[str, object]:
    # Summarise every stored study so the write-up can quote it defensibly
    out: dict[str, object] = {}
    multiseed = results_dir / "multiseed_layernorm.json"
    if multiseed.exists():
        payload = json.loads(multiseed.read_text(encoding="utf-8"))
        summary = payload["summary"]
        errors = {}
        for strategy, block in summary.items():
            errors[strategy] = {
                metric: {
                    "mean": block[metric]["mean"],
                    "std": block[metric]["std"],
                    "se": block[metric]["std"] / math.sqrt(max(block[metric]["n"], 1)),
                    "cv": abs(block[metric]["std"] / block[metric]["mean"])
                    if block[metric]["mean"] else None,
                }
                for metric in ("mean_pnl", "std_pnl", "cvar_95", "cvar_99", "entropic", "mean_cost")
            }
        out["multiseed"] = {
            "seeds": payload["seeds"],
            "n_seeds": len(payload["seeds"]),
            "config": payload.get("config", {}),
            "metrics": errors,
        }
        # Paired per-seed gaps, the statistic the write-up quotes
        per_seed = payload.get("per_seed", [])
        if per_seed:
            paired = {}
            for metric in ("mean_pnl", "std_pnl", "cvar_95", "cvar_99", "entropic", "mean_cost"):
                gaps = [
                    float(row["Leland"][metric]) - float(row["Deep Hedging"][metric])
                    for row in per_seed
                ]
                mean_gap = sum(gaps) / len(gaps)
                if len(gaps) > 1:
                    var = sum((g - mean_gap) ** 2 for g in gaps) / (len(gaps) - 1)
                    se_gap = math.sqrt(var / len(gaps))
                else:
                    se_gap = 0.0
                paired[metric] = {
                    "gap": mean_gap,
                    "se": se_gap,
                    "t": mean_gap / se_gap if se_gap > 0.0 else None,
                }
            out["multiseed"]["paired_vs_leland"] = paired
        # Relative seed noise, the quantity that governs how many seeds are needed
        noise = {
            strategy: {
                "cvar_95_rel_std": errors[strategy]["cvar_95"]["std"] / errors[strategy]["cvar_95"]["mean"],
                "entropic_rel_std": errors[strategy]["entropic"]["std"] / errors[strategy]["entropic"]["mean"],
            }
            for strategy in errors
        }
        out["multiseed"]["relative_noise"] = noise

    lambda_path = results_dir / "lambda_multiseed.json"
    if lambda_path.exists():
        payload = json.loads(lambda_path.read_text(encoding="utf-8"))
        rows = payload.get("levels", payload.get("summary", {}))
        frontier = []
        if isinstance(rows, list):
            for row in rows:
                lam = float(row.get("lambda", row.get("lambda_entropic", 0.0)))
                block = row.get("summary", row)
                frontier.append(
                    {
                        "lambda": lam,
                        "mean_pnl": block["deep"]["mean"]["mean"] if "deep" in block else None,
                        "cvar_95": block["deep"]["cvar_95"]["mean"] if "deep" in block else None,
                    }
                )
        out["lambda"] = frontier

    for name in ("heston_multiseed", "price_asymptotics_bs", "heston_nocost",
                 "recurrent_ablation", "paper_architecture", "call_spread",
                 "price_asymptotics_heston"):
        for candidate in (results_dir / f"{name}.json", results_dir / "paper" / f"{name}.json"):
            if candidate.exists():
                out[name] = {"path": str(candidate.name), "present": True}
                break
    return out


def eda_paper(results_dir: Path) -> dict[str, object]:
    # Distributional summary of the reproduced paper experiments
    out: dict[str, object] = {}
    bs_path = results_dir / "paper" / "price_asymptotics_bs.json"
    if bs_path.exists():
        payload = json.loads(bs_path.read_text(encoding="utf-8"))
        records = payload["records"]
        q = float(payload["q"])
        costs = sorted({float(r["cost"]) for r in records})
        levels = []
        for cost in costs:
            subset = [r for r in records if abs(float(r["cost"]) - cost) < 1e-12]
            ent = np.array([float(r["entropic"]) for r in subset])
            levels.append(
                {
                    "cost": cost,
                    "n": len(subset),
                    "p_eps_mean": float(ent.mean()),
                    "p_eps_std": float(ent.std(ddof=1)) if ent.size > 1 else None,
                    "excess_over_q": float(ent.mean() - q),
                    "cvar_95_mean": float(np.mean([float(r["cvar_95"]) for r in subset])),
                    "mean_cost_mean": float(np.mean([float(r["mean_cost"]) for r in subset])),
                }
            )
        out["price_asymptotics_bs"] = {
            "q": q,
            "p_0": float(payload["p_0"]),
            "levels": levels,
            "fit": payload["fit"],
        }
    return out


def print_market(block: dict[str, object]) -> None:
    print()
    print("Simulator EDA: terminal distributions")
    header = f"{'quantity':<32} {'mean':>10} {'se':>8} {'std':>9} {'skew':>8} {'exc kurt':>9} {'q05':>9} {'q95':>9}"
    print(header)
    print("-" * len(header))
    for key in ("gbm", "heston", "paper_heston"):
        for field in ("terminal", "payoff"):
            entry = block[key][field]
            print(
                f"{entry['name']:<32} {entry['mean']:10.4f} {entry['se_mean']:8.4f} "
                f"{entry['std']:9.4f} {entry['skew']:8.3f} {entry['excess_kurtosis']:9.3f} "
                f"{entry['q05']:9.4f} {entry['q95']:9.4f}"
            )
    print()
    print("Martingale and price consistency")
    for key in ("gbm", "heston", "paper_heston"):
        m = block[key]["martingale"]
        line = (
            f"  {key:<14} mean spot {m['mean_spot']:8.4f} +/- {m['se']:.4f} "
            f"(target {m['target']:.4f})"
        )
        if "theoretical_call" in m:
            line += f"   payoff {block[key]['payoff']['mean']:.4f} vs model {m['theoretical_call']:.4f}"
        print(line)
    print(
        f"  variance swap fair strike {block['paper_heston']['fair_strike']:.6f} "
        f"against analytic {block['paper_heston']['fair_strike_analytic']:.6f}"
    )
    print(
        f"  Euler variance zero fraction {block['heston']['zero_fraction']:.4f}, "
        f"exact CIR {block['paper_heston']['zero_fraction']:.4f}"
    )
    print(f"  Feller 2 kappa theta - xi^2 = {block['paper_heston']['feller']:.4f}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    results = root / args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    paths = 20_000 if args.smoke else args.paths
    if args.smoke:
        # Smoke runs must never touch the real results directory
        results = root / "_scratch" / "paper"
        results.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Exploratory data analysis")
    print("=" * 78)
    market = eda_market(paths)
    print_market(market)

    stored = eda_results(root / "results")
    paper = eda_paper(root / "results")
    payload = {
        "n_paths": paths,
        "market": market,
        "stored_studies": stored,
        "paper": paper,
        "config": {"paths": paths, "commit": "uncommitted"},
    }
    target = results / "eda.json"
    target.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten {target}")

    if not args.smoke:
        print(plot_eda_market(payload, results))
        print(plot_eda_moments(payload, results))
        if paper:
            print(plot_eda_results(payload, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
