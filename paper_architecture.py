import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import Config
from main import run_policy, seed_everything, train_policy
from src.loss import empirical_cvar
from src.network import DeepHedgingNetwork
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths
from src.visualize import plot_architecture_comparison

DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Section 5.1 architecture comparison")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--eval-paths", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--results-dir", default="results/paper")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--aggregate", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


# The project default against the paper Section 5.1 description
VARIANTS = (
    {
        "key": "project",
        "label": "Project default",
        "time_dependent": False,
        "hidden_dim": 64,
        "n_layers": 3,
        "lr": 0.001,
        "norm_type": "layernorm",
        "activation": "leaky_relu",
        "feature_mode": "engineered",
        "use_prev_delta": True,
    },
    {
        "key": "paper",
        "label": "Paper Section 5.1",
        "time_dependent": True,
        "hidden_dim": 17,
        "n_layers": 2,
        "lr": 0.005,
        "norm_type": "batchnorm",
        "activation": "relu",
        "feature_mode": "paper",
        "use_prev_delta": True,
    },
)


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


def evaluate_variant(
    config: Config,
    seed: int,
    variant: dict,
    eval_spots: torch.Tensor,
) -> dict[str, object]:
    # Train one architecture and score it on the shared evaluation set
    level = replace(
        config,
        time_dependent=variant["time_dependent"],
        hidden_dim=int(variant["hidden_dim"]),
        n_layers=int(variant["n_layers"]),
        lr=float(variant["lr"]),
        norm_type=str(variant["norm_type"]),
        use_prev_delta=bool(variant["use_prev_delta"]),
        seed=int(seed),
    )
    train_spots, _ = simulate_paths(level, level.n_paths, seed=int(seed))
    model, history, _ = train_policy(level, train_spots, None, seed=int(seed))
    pnl, _, _ = run_policy(model, eval_spots, None, level)
    with torch.no_grad():
        deltas = model(eval_spots)
    detail = hedge_pnl(eval_spots, deltas, level.K, level.cost_rate)
    return {
        "variant": variant["key"],
        "seed": int(seed),
        "mean": float(pnl.mean()),
        "std": float(pnl.std(unbiased=True)),
        "cvar_95": float(empirical_cvar(pnl, 0.95)),
        "cvar_99": float(empirical_cvar(pnl, 0.99)),
        "mean_cost": float(detail["costs"].mean()),
        "turnover": float(detail["turnover"].mean()),
        "final_train_risk": float(history[-1]),
        "n_params": int(model.count_parameters()),
        "history": list(history),
    }


def build_variant_model(config: Config, variant: dict) -> DeepHedgingNetwork:
    # Parameter count only, used for the summary table
    return DeepHedgingNetwork(
        n_steps=config.n_steps,
        K=config.K,
        T=config.T,
        base_variance=config.base_variance,
        dt=config.dt,
        max_position=config.max_position,
        hidden_dim=int(variant["hidden_dim"]),
        n_layers=int(variant["n_layers"]),
        time_dependent=bool(variant["time_dependent"]),
        norm_type=str(variant["norm_type"]),
        use_prev_delta=bool(variant["use_prev_delta"]),
        feature_mode=str(variant["feature_mode"]),
        activation=str(variant["activation"]),
    )


def run_shard(args: argparse.Namespace, config: Config) -> int:
    seed_everything(config.seed, deterministic=config.deterministic)
    cells = [(v, s) for v in VARIANTS for s in args.seeds]
    seed_rank = {seed: i for i, seed in enumerate(args.seeds)}
    mine = [cell for cell in cells if seed_rank[cell[1]] % args.shards == args.shard]
    if not mine:
        raise SystemExit(f"shard {args.shard} has no cells, reduce --shards")
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(cells)} cells", flush=True)
    records: list[dict] = []
    for variant, seed in mine:
        eval_spots, _ = simulate_paths(config, config.n_eval_paths, seed=seed + 1)
        tick = time.perf_counter()
        record = evaluate_variant(config, seed, variant, eval_spots)
        record["seconds"] = time.perf_counter() - tick
        records.append(record)
        print(
            f"  {variant['key']:<8} seed {seed}  cvar95 {record['cvar_95']:7.4f}  "
            f"mean {record['mean']:7.4f}  params {record['n_params']:,}  "
            f"{record['seconds']:6.1f}s",
            flush=True,
        )
        if args.smoke:
            break
    target = Path(args.raw_out or f"r6_shard{args.shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"shard {args.shard} wrote {target}", flush=True)
    return 0


def aggregate_records(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for variant in VARIANTS:
        subset = [r for r in records if r["variant"] == variant["key"]]
        if not subset:
            continue
        out[variant["key"]] = {}
        for metric in ("mean", "std", "cvar_95", "cvar_99", "mean_cost", "turnover"):
            values = np.array([float(r[metric]) for r in subset])
            out[variant["key"]][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                "n": int(values.size),
            }
        out[variant["key"]]["n_params"] = {"mean": float(subset[0]["n_params"]), "std": 0.0,
                                           "n": len(subset)}
    return out


def print_table(summary: dict, records: list[dict], config: Config) -> None:
    print()
    print("Paper Section 5.1: project default against the paper architecture, 5 seed mean")
    header = f"{'variant':<20} {'Mean PnL':>16} {'Std':>14} {'CVaR 95':>16} {'Entropic':>16} {'Cost':>14} {'Params':>9}"
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        block = summary.get(variant["key"])
        if not block:
            continue
        ent = np.array([float(r["final_train_risk"]) for r in records if r["variant"] == variant["key"]])
        print(
            f"{variant['label']:<20} "
            f"{block['mean']['mean']:8.3f}+/-{block['mean']['std']:6.3f} "
            f"{block['std']['mean']:7.3f}+/-{block['std']['std']:5.3f} "
            f"{block['cvar_95']['mean']:8.3f}+/-{block['cvar_95']['std']:6.3f} "
            f"{ent.mean():8.3f}+/-{ent.std(ddof=1) if ent.size>1 else 0.0:6.3f} "
            f"{block['mean_cost']['mean']:6.3f}+/-{block['mean_cost']['std']:5.3f} "
            f"{int(block['n_params']['mean']):9,d}"
        )


def main() -> int:
    args = parse_args()
    config = base_config(args)
    results = Path(config.results_path)
    results.mkdir(parents=True, exist_ok=True)
    if args.shards > 1:
        return run_shard(args, config)

    print("=" * 78)
    print("Paper Section 5.1: architecture fidelity check")
    print("=" * 78)
    for variant in VARIANTS:
        model = build_variant_model(config, variant)
        print(f"  {variant['label']:<22} params {model.count_parameters():,}")
    started = time.perf_counter()

    if args.aggregate:
        records: list[dict] = []
        for path in sorted(results.glob("raw_r6_shard*.json")):
            records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}  {len(records)} records so far")
    else:
        seed_everything(config.seed, deterministic=config.deterministic)
        records = []
        for variant in VARIANTS:
            for seed in args.seeds:
                eval_spots, _ = simulate_paths(config, config.n_eval_paths, seed=seed + 1)
                tick = time.perf_counter()
                record = evaluate_variant(config, seed, variant, eval_spots)
                record["seconds"] = time.perf_counter() - tick
                records.append(record)
                print(
                    f"  {variant['key']:<8} seed {seed}  cvar95 {record['cvar_95']:7.4f}  "
                    f"mean {record['mean']:7.4f}  {record['seconds']:6.1f}s",
                    flush=True,
                )

    summary = aggregate_records(records)
    print_table(summary, records, config)
    payload = {
        "variants": [{k: v for k, v in variant.items()} for variant in VARIANTS],
        "seeds": list(args.seeds),
        "summary": summary,
        "records": records,
        "config": {
            "simulator": config.simulator,
            "loss_type": config.loss_type,
            "cost_rate": config.cost_rate,
            "n_paths": config.n_paths,
            "n_eval_paths": config.n_eval_paths,
            "epochs": config.epochs,
            "seeds": list(args.seeds),
            "commit": "uncommitted",
        },
    }
    target = results / "paper_architecture.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {target}")
    if not args.smoke:
        plot_architecture_comparison(payload, results)
    print(f"total runtime {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
