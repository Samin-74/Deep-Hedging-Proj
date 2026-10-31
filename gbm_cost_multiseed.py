import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import benchmark_results, format_table, run_policy, seed_everything, train_policy
from src.loss import empirical_cvar
from src.simulators import simulate_paths
from src.studies import aggregate, format_pm
from src.visualize import plot_cost_sensitivity_multiseed

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
METRICS = (
    "deep_mean",
    "deep_cvar_95",
    "deep_cvar_99",
    "bs_cvar_95",
    "bs_cvar_99",
    "leland_cvar_95",
    "leland_cvar_99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GBM multi-seed cost sensitivity")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--eval-paths", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--costs", type=float, nargs="+", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--aggregate", type=int, default=0)
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        "n_paths": args.paths,
        "n_eval_paths": args.eval_paths,
        "epochs": args.epochs,
        "run_cost_sensitivity": False,
        "results_dir": args.results_dir,
    }
    config = replace(config, **overrides)
    config.validate()
    return config


def run_level(config: Config, cost: float, seed: int) -> dict[str, float]:
    # One training at one cost level, each seed draws its own train and eval paths
    level = replace(config, cost_rate=float(cost), seed=int(seed))
    train_spots, train_var = simulate_paths(level, level.n_paths, seed=seed)
    eval_spots, eval_var = simulate_paths(level, level.n_eval_paths, seed=seed + 1)
    model, _, _ = train_policy(level, train_spots, train_var, seed=seed)
    deep_pnl, _ = run_policy(model, eval_spots, eval_var, level, return_path=False)
    benches = benchmark_results(eval_spots, level, return_path=False)
    record = {
        "cost": float(cost),
        "seed": int(seed),
        "deep_mean": float(deep_pnl.mean()),
        "deep_cvar_95": float(empirical_cvar(deep_pnl, 0.95)),
        "deep_cvar_99": float(empirical_cvar(deep_pnl, 0.99)),
    }
    for name, payload in benches.items():
        prefix = "bs" if name == "Black-Scholes" else "leland"
        record[f"{prefix}_cvar_95"] = float(empirical_cvar(payload["pnl"], 0.95))
        record[f"{prefix}_cvar_99"] = float(empirical_cvar(payload["pnl"], 0.99))
        record[f"{prefix}_mean"] = float(payload["pnl"].mean())
    return record


def run_shard(args: argparse.Namespace, config: Config) -> int:
    # Worker mode, shard by seed rank so every shard gets a balanced share
    seed_everything(config.seed, deterministic=config.deterministic)
    costs = list(args.costs or config.cost_levels)
    cells = [(c, s) for c in costs for s in args.seeds]
    seed_rank = {seed: i for i, seed in enumerate(args.seeds)}
    mine = [cell for cell in cells if seed_rank[cell[1]] % args.shards == args.shard]
    if not mine:
        raise SystemExit(f"shard {args.shard} has no cells, reduce --shards")
    print(
        f"shard {args.shard}/{args.shards}: {len(mine)} of {len(cells)} cells, "
        f"paths {config.n_paths:,}, epochs {config.epochs}",
        flush=True,
    )
    records: list[dict[str, float]] = []
    for cost, seed in mine:
        tick = time.perf_counter()
        record = run_level(config, cost, seed)
        records.append(record)
        print(
            f"  cost {cost:8.6f} seed {seed}  cvar95 {record['deep_cvar_95']:8.4f}  "
            f"cvar99 {record['deep_cvar_99']:8.4f}  mean {record['deep_mean']:8.4f}  "
            f"{time.perf_counter() - tick:6.1f}s",
            flush=True,
        )
    target = Path(args.raw_out or f"raw_gbm_cost_shard{args.shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"shard {args.shard} wrote {target}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    config = base_config(args)
    seed_everything(config.seed, deterministic=config.deterministic)
    results = Path(config.results_path)
    results.mkdir(parents=True, exist_ok=True)
    costs = list(args.costs or config.cost_levels)

    if args.shards > 1:
        return run_shard(args, config)

    print("=" * 78)
    print("GBM multi-seed cost sensitivity")
    print("=" * 78)
    print(
        f"paths {config.n_paths:,}  eval {config.n_eval_paths:,}  epochs {config.epochs}  "
        f"seeds {list(args.seeds)}  levels {costs}"
    )
    print("=" * 78)

    started = time.perf_counter()
    if args.aggregate:
        # Aggregation only, reuse the shard records and retrain nothing
        records: list[dict[str, float]] = []
        for path in sorted(results.glob("raw_gbm_cost_shard*.json")):
            records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}  {len(records)} records so far")
        if not records:
            raise SystemExit("no raw shard files found, run the shards first")
    else:
        records = []
        for cost in costs:
            for seed in args.seeds:
                tick = time.perf_counter()
                record = run_level(config, cost, seed)
                records.append(record)
                print(
                    f"  cost {cost:8.6f} seed {seed}  cvar95 {record['deep_cvar_95']:8.4f}  "
                    f"cvar99 {record['deep_cvar_99']:8.4f}  {time.perf_counter() - tick:6.1f}s",
                    flush=True,
                )

    summary: dict[str, dict] = {}
    for cost in costs:
        subset = [r for r in records if abs(float(r["cost"]) - cost) < 1e-12]
        if not subset:
            continue
        summary[f"{cost:.6f}"] = aggregate(subset, METRICS)

    rows = []
    for cost in costs:
        block = summary.get(f"{cost:.6f}")
        if block is None:
            continue
        rows.append(
            (
                f"{100.0 * cost:5.2f}%",
                format_pm(block["deep_mean"]["mean"], block["deep_mean"]["std"]),
                format_pm(block["deep_cvar_95"]["mean"], block["deep_cvar_95"]["std"]),
                format_pm(block["deep_cvar_99"]["mean"], block["deep_cvar_99"]["std"]),
                format_pm(block["bs_cvar_95"]["mean"], block["bs_cvar_95"]["std"]),
                format_pm(block["leland_cvar_95"]["mean"], block["leland_cvar_95"]["std"]),
            )
        )
    print()
    print(f"GBM multi-seed cost sensitivity, mean +/- std over {len(args.seeds)} seeds")
    print(
        format_table(
            ("Cost", "Deep mean", "Deep CVaR95", "Deep CVaR99", "BS CVaR95", "Leland CVaR95"),
            rows,
        )
    )

    payload = {
        "seeds": list(args.seeds),
        "paths": config.n_paths,
        "eval_paths": config.n_eval_paths,
        "epochs": config.epochs,
        "costs": costs,
        "records": records,
        "summary": summary,
        "config": {
            "simulator": "gbm",
            "loss_type": config.loss_type,
            "lambda_entropic": config.lambda_entropic,
            "sigma": config.benchmark_sigma,
            "norm_type": config.norm_type,
            "time_dependent": config.time_dependent,
        },
    }
    target = results / "gbm_cost_multiseed.json"
    target.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten {target}")

    print(" ", plot_cost_sensitivity_multiseed(payload, results))
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
