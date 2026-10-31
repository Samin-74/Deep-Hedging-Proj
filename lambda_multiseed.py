import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import (
    benchmark_results,
    format_table,
    run_policy,
    seed_everything,
    train_policy,
)
from src.loss import EntropicRiskLoss, empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths
from src.studies import aggregate, format_pm
from src.visualize import plot_lambda_frontier

SEEDS = (42, 43, 44, 45, 46)
METRICS = ("mean_pnl", "std_pnl", "cvar_95", "cvar_99", "entropic", "mean_cost", "turnover")


def run_lambda(config: Config, lam: float, seeds, verbose: bool = True) -> list[dict[str, float]]:
    # One lambda value, one record per seed
    level = replace(config, lambda_entropic=float(lam))
    level.validate()
    records: list[dict[str, float]] = []
    for seed in seeds:
        started = time.perf_counter()
        train_spots, train_var = simulate_paths(level, level.n_paths, seed=seed)
        eval_spots, eval_var = simulate_paths(level, level.n_eval_paths, seed=seed + 1)
        model, history, _ = train_policy(level, train_spots, train_var, seed=seed)
        pnl, _ = run_policy(model, eval_spots, eval_var, level, return_path=False)
        with torch.no_grad():
            deltas = model(eval_spots, eval_var)
        detail = hedge_pnl(eval_spots, deltas, level.K, level.cost_rate, return_path=False)
        record = {
            "lambda": float(lam),
            "seed": int(seed),
            "mean_pnl": float(pnl.mean()),
            "std_pnl": float(pnl.std(unbiased=True)),
            "cvar_95": float(empirical_cvar(pnl, 0.95)),
            "cvar_99": float(empirical_cvar(pnl, 0.99)),
            "entropic": float(EntropicRiskLoss(lam)(pnl)),
            "mean_cost": float(detail["costs"].mean()),
            "turnover": float(detail["turnover"].mean()),
        }
        records.append(record)
        if verbose:
            print(
                f"    seed {seed}: mean {record['mean_pnl']:7.3f}  std {record['std_pnl']:6.3f}  "
                f"cvar95 {record['cvar_95']:6.3f}  cost {record['mean_cost']:5.3f}  "
                f"{time.perf_counter() - started:6.1f}s",
                flush=True,
            )
    return records


def main() -> int:
    lambdas = [float(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.5", "1.0", "2.0"])]
    paths = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    output = sys.argv[4] if len(sys.argv) > 4 else "results/lambda_multiseed.json"
    shard = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    shards = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    raw_out = sys.argv[7] if len(sys.argv) > 7 else None

    config = replace(Config(), n_paths=paths, epochs=epochs)
    config.validate()
    seed_everything(config.seed, deterministic=config.deterministic)

    print("=" * 78)
    print("Multi-seed entropic risk aversion sweep")
    print("=" * 78)
    print(f"lambdas {lambdas}  paths {paths:,}  epochs {epochs}  seeds {list(SEEDS)}")
    print(f"shard {shard} of {shards}")
    print("=" * 78)

    started = time.perf_counter()
    per_lambda: dict[str, dict] = {}
    for lam in lambdas:
        # Shard by seed rank so every lambda keeps the same seed coverage
        assigned = [s for i, s in enumerate(SEEDS) if i % shards == shard]
        if not assigned:
            continue
        print(f"  lambda {lam:4.2f}  seeds {assigned}")
        records = run_lambda(config, lam, assigned)
        per_lambda[f"{lam:.4f}"] = {"records": records, "summary": aggregate(records, METRICS)}

    if raw_out is not None:
        Path(raw_out).write_text(json.dumps(per_lambda, indent=2), encoding="utf-8")
        print(f"shard written to {raw_out}")
        return 0

    rows = []
    for lam in lambdas:
        s = per_lambda[f"{lam:.4f}"]["summary"]
        rows.append(
            (
                f"{lam:5.2f}",
                format_pm(s["mean_pnl"]["mean"], s["mean_pnl"]["std"]),
                format_pm(s["std_pnl"]["mean"], s["std_pnl"]["std"]),
                format_pm(s["cvar_95"]["mean"], s["cvar_95"]["std"]),
                format_pm(s["cvar_99"]["mean"], s["cvar_99"]["std"]),
                format_pm(s["mean_cost"]["mean"], s["mean_cost"]["std"]),
            )
        )
    print()
    print(f"Lambda sweep, mean +/- std over {len(SEEDS)} seeds")
    print(
        format_table(
            ("Lambda", "Mean PnL", "Std", "CVaR 95", "CVaR 99", "Mean cost"),
            rows,
        )
    )

    # Benchmarks on the shared evaluation paths
    eval_spots, eval_var = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    benches = benchmark_results(eval_spots, config, return_path=False)
    bench_rows = []
    for name, payload in benches.items():
        p = payload["pnl"]
        bench_rows.append(
            (
                name,
                f"{float(p.mean()):8.3f}",
                f"{float(p.std(unbiased=True)):8.3f}",
                f"{float(empirical_cvar(p, 0.95)):8.3f}",
                f"{float(empirical_cvar(p, 0.99)):8.3f}",
                f"{float(payload['costs'].mean()):8.3f}",
            )
        )
    print()
    print("Benchmarks on the same evaluation paths")
    print(format_table(("Strategy", "Mean PnL", "Std", "CVaR 95", "CVaR 99", "Mean cost"), bench_rows))

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "paths": paths,
                "epochs": epochs,
                "lambdas": per_lambda,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"lambda multi-seed written to {target}")

    figure = plot_lambda_frontier(per_lambda, lambdas, Path("results"))
    print(f"saved {figure}")
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
