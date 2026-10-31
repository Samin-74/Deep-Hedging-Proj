import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import benchmark_results, format_table, run_policy, seed_everything, train_policy
from src.loss import EntropicRiskLoss, empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths
from src.studies import aggregate, format_pm

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
METRICS = ("entropic", "cvar_95", "cvar_99", "mean_pnl", "mean_cost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared against time-dependent weights")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--eval-paths", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--aggregate", type=int, default=0)
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        "simulator": "gbm",
        "loss_type": "entropic",
        "cost_rate": 0.01,
        "n_paths": args.paths,
        "n_eval_paths": args.eval_paths,
        "epochs": args.epochs,
        "run_cost_sensitivity": False,
        "results_dir": args.results_dir,
    }
    config = replace(config, **overrides)
    config.validate()
    return config


def run_cell(config: Config, time_dependent: bool, seed: int) -> dict[str, float]:
    # One training of one weight-sharing mode at one seed
    level = replace(config, time_dependent=bool(time_dependent), seed=int(seed))
    train_spots, train_var = simulate_paths(level, level.n_paths, seed=seed)
    eval_spots, eval_var = simulate_paths(level, level.n_eval_paths, seed=seed + 1)
    model, _, _ = train_policy(level, train_spots, train_var, seed=seed)
    deep_pnl, _ = run_policy(model, eval_spots, eval_var, level, return_path=False)
    with torch.no_grad():
        deltas = model(eval_spots, eval_var)
    detail = hedge_pnl(eval_spots, deltas, level.K, level.cost_rate)
    record = {
        "variant": "time_dependent" if time_dependent else "shared",
        "seed": int(seed),
        "entropic": float(EntropicRiskLoss(level.lambda_entropic)(deep_pnl)),
        "cvar_95": float(empirical_cvar(deep_pnl, 0.95)),
        "cvar_99": float(empirical_cvar(deep_pnl, 0.99)),
        "mean_pnl": float(deep_pnl.mean()),
        "mean_cost": float(detail["costs"].mean()),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    benches = benchmark_results(eval_spots, level, return_path=False)
    for name, payload in benches.items():
        prefix = "bs" if name == "Black-Scholes" else "leland"
        record[f"{prefix}_entropic"] = float(
            EntropicRiskLoss(level.lambda_entropic)(payload["pnl"])
        )
        record[f"{prefix}_cvar_95"] = float(empirical_cvar(payload["pnl"], 0.95))
        record[f"{prefix}_cvar_99"] = float(empirical_cvar(payload["pnl"], 0.99))
    return record


def run_shard(args: argparse.Namespace, config: Config) -> int:
    # Worker mode, shard by seed rank so both variants of a seed stay together
    seed_everything(config.seed, deterministic=config.deterministic)
    cells = [(v, s) for v in (False, True) for s in args.seeds]
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
    for time_dependent, seed in mine:
        tick = time.perf_counter()
        record = run_cell(config, time_dependent, seed)
        records.append(record)
        print(
            f"  {record['variant']:14s} seed {seed}  entropic {record['entropic']:8.4f}  "
            f"cvar95 {record['cvar_95']:8.4f}  params {record['n_params']:,}  "
            f"{time.perf_counter() - tick:6.1f}s",
            flush=True,
        )
    target = Path(args.raw_out or f"raw_shared_shard{args.shard}.json")
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

    if args.shards > 1:
        return run_shard(args, config)

    print("=" * 78)
    print("Shared against time-dependent weights, like for like")
    print("=" * 78)
    print(
        f"paths {config.n_paths:,}  eval {config.n_eval_paths:,}  epochs {config.epochs}  "
        f"seeds {list(args.seeds)}  cost {config.cost_rate}"
    )
    print("=" * 78)

    started = time.perf_counter()
    if args.aggregate:
        # Aggregation only, reuse the shard records and retrain nothing
        records: list[dict[str, float]] = []
        for path in sorted(results.glob("raw_shared_shard*.json")):
            records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}  {len(records)} records so far")
        if not records:
            raise SystemExit("no raw shard files found, run the shards first")
    else:
        records = []
        for time_dependent in (False, True):
            for seed in args.seeds:
                tick = time.perf_counter()
                record = run_cell(config, time_dependent, seed)
                records.append(record)
                print(
                    f"  {record['variant']:14s} seed {seed}  entropic {record['entropic']:8.4f}  "
                    f"cvar95 {record['cvar_95']:8.4f}  {time.perf_counter() - tick:6.1f}s",
                    flush=True,
                )

    summary: dict[str, dict] = {}
    for variant in ("shared", "time_dependent"):
        subset = [r for r in records if r["variant"] == variant]
        if subset:
            summary[variant] = aggregate(subset, METRICS)
            summary[variant]["n_params"] = {
                "mean": float(subset[0]["n_params"]),
                "std": 0.0,
                "n": len(subset),
            }

    rows = []
    for variant, label in (("shared", "Shared"), ("time_dependent", "Time-dependent")):
        block = summary.get(variant)
        if block is None:
            continue
        rows.append(
            (
                label,
                format_pm(block["entropic"]["mean"], block["entropic"]["std"]),
                format_pm(block["cvar_95"]["mean"], block["cvar_95"]["std"]),
                format_pm(block["cvar_99"]["mean"], block["cvar_99"]["std"]),
                f"{int(block['n_params']['mean']):,}",
            )
        )
    print()
    print(f"Shared against time-dependent, mean +/- std over {len(args.seeds)} seeds")
    print(
        format_table(
            ("Variant", "Entropic", "CVaR 95", "CVaR 99", "Parameters"), rows
        )
    )

    payload = {
        "seeds": list(args.seeds),
        "paths": config.n_paths,
        "eval_paths": config.n_eval_paths,
        "epochs": config.epochs,
        "cost_rate": config.cost_rate,
        "records": records,
        "summary": summary,
        "config": {
            "simulator": "gbm",
            "loss_type": config.loss_type,
            "lambda_entropic": config.lambda_entropic,
            "norm_type": config.norm_type,
        },
    }
    target = results / "shared_vs_timedep.json"
    target.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten {target}")
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
