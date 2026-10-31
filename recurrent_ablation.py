import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import Config
from main import run_policy, seed_everything, train_policy
from src.loss import empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths
from src.visualize import plot_recurrent_ablation

DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Figures 4 and 5: recurrent vs simple")
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


# Scenario (a) no cost at alpha 0.5, scenario (b) one percent cost at alpha 0.99
SCENARIOS = (
    {"key": "a", "label": "no cost, CVaR 0.50", "cost_rate": 0.0, "cvar_alpha": 0.5},
    {"key": "b", "label": "1 percent cost, CVaR 0.99", "cost_rate": 0.01, "cvar_alpha": 0.99},
)


def base_config(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        "simulator": "gbm",
        "loss_type": "cvar",
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
    use_prev_delta: bool,
    eval_spots: torch.Tensor,
) -> dict[str, float]:
    # Train one variant and score its hedging error on the shared test set
    level = replace(config, use_prev_delta=bool(use_prev_delta), seed=int(seed))
    train_spots, _ = simulate_paths(level, level.n_paths, seed=int(seed))
    model, history, _ = train_policy(level, train_spots, None, seed=int(seed))
    pnl, _, _ = run_policy(model, eval_spots, None, level)
    with torch.no_grad():
        deltas = model(eval_spots)
    detail = hedge_pnl(eval_spots, deltas, level.K, level.cost_rate)
    return {
        "seed": int(seed),
        "recurrent": bool(use_prev_delta),
        "risk": float(empirical_cvar(pnl, level.cvar_alpha)),
        "mean": float(pnl.mean()),
        "std": float(pnl.std(unbiased=True)),
        "cvar_95": float(empirical_cvar(pnl, 0.95)),
        "cvar_99": float(empirical_cvar(pnl, 0.99)),
        "mean_cost": float(detail["costs"].mean()),
        "turnover": float(detail["turnover"].mean()),
        "final_train_risk": float(history[-1]),
        "n_params": int(model.count_parameters()),
        "pnl": pnl.numpy().astype(np.float32),
    }


def run_shard(args: argparse.Namespace, config: Config) -> int:
    # Worker mode over (scenario, seed, variant) cells
    seed_everything(config.seed, deterministic=config.deterministic)
    cells = [
        (scenario, seed, recurrent)
        for scenario in SCENARIOS
        for seed in args.seeds
        for recurrent in (True, False)
    ]
    mine = [cell for i, cell in enumerate(cells) if i % args.shards == args.shard]
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(cells)} cells", flush=True)
    records: list[dict[str, object]] = []
    for scenario, seed, recurrent in mine:
        level = replace(
            config,
            cost_rate=float(scenario["cost_rate"]),
            cvar_alpha=float(scenario["cvar_alpha"]),
        )
        eval_spots, _ = simulate_paths(level, level.n_eval_paths, seed=seed + 1)
        tick = time.perf_counter()
        record = evaluate_variant(level, seed, recurrent, eval_spots)
        record["scenario"] = scenario["key"]
        record["cost_rate"] = float(scenario["cost_rate"])
        record["cvar_alpha"] = float(scenario["cvar_alpha"])
        # The PnL sample is not used by any figure, so it is dropped to keep the JSON small
        record.pop("pnl", None)
        records.append(record)
        print(
            f"  scenario {scenario['key']} seed {seed} recurrent {recurrent}  "
            f"risk {record['risk']:8.4f}  cost {record['mean_cost']:6.4f}  "
            f"{time.perf_counter() - tick:6.1f}s",
            flush=True,
        )
        if args.smoke:
            break
    target = Path(args.raw_out or f"r3_shard{args.shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"shard {args.shard} wrote {target}", flush=True)
    return 0


def aggregate_records(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    # Mean and sample standard deviation per scenario and variant
    out: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIOS:
        key = scenario["key"]
        out[key] = {}
        for recurrent in (True, False):
            subset = [
                r for r in records
                if r["scenario"] == key and bool(r["recurrent"]) == recurrent
            ]
            if not subset:
                continue
            name = "recurrent" if recurrent else "simple"
            out[key][name] = {}
            for metric in ("risk", "mean", "std", "cvar_95", "cvar_99", "mean_cost", "turnover"):
                values = np.array([float(r[metric]) for r in subset])
                out[key][name][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                    "n": int(values.size),
                }
    return out


def print_table(summary: dict, records: list[dict]) -> None:
    print()
    print("Paper Figures 4 and 5: recurrent against simple, 5 seed mean")
    header = f"{'scenario':<26} {'variant':<11} {'risk':>14} {'mean cost':>16} {'turnover':>16}"
    print(header)
    print("-" * len(header))
    for scenario in SCENARIOS:
        block = summary.get(scenario["key"], {})
        for name in ("recurrent", "simple"):
            if name not in block:
                continue
            entry = block[name]
            print(
                f"{scenario['label']:<26} {name:<11} "
                f"{entry['risk']['mean']:7.4f}+/-{entry['risk']['std']:6.4f} "
                f"{entry['mean_cost']['mean']:8.4f}+/-{entry['mean_cost']['std']:6.4f} "
                f"{entry['turnover']['mean']:8.4f}+/-{entry['turnover']['std']:6.4f}"
            )
    print()
    print("Paired difference within seed, recurrent minus simple")
    for scenario in SCENARIOS:
        key = scenario["key"]
        rec = {r["seed"]: r for r in records if r["scenario"] == key and r["recurrent"]}
        simp = {r["seed"]: r for r in records if r["scenario"] == key and not r["recurrent"]}
        shared = sorted(set(rec) & set(simp))
        if len(shared) < 2:
            continue
        diffs = np.array([float(rec[s]["risk"]) - float(simp[s]["risk"]) for s in shared])
        se = float(diffs.std(ddof=1)) / math.sqrt(diffs.size)
        print(
            f"  {scenario['label']:<26} risk gap {diffs.mean():+8.4f} +/- {se:6.4f} (se), "
            f"t {abs(diffs.mean()) / se if se > 0 else float('inf'):5.2f}, n {diffs.size}"
        )


def main() -> int:
    args = parse_args()
    config = base_config(args)
    results = Path(config.results_path)
    results.mkdir(parents=True, exist_ok=True)

    if args.shards > 1:
        return run_shard(args, config)

    print("=" * 78)
    print("Paper Figures 4 and 5: does the previous position matter")
    print("=" * 78)
    started = time.perf_counter()
    if args.aggregate:
        records: list[dict] = []
        for path in sorted(results.glob("raw_r3_shard*.json")):
            records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}  {len(records)} records so far")
    else:
        seed_everything(config.seed, deterministic=config.deterministic)
        records = []
        for scenario in SCENARIOS:
            level = replace(
                config,
                cost_rate=float(scenario["cost_rate"]),
                cvar_alpha=float(scenario["cvar_alpha"]),
            )
            for seed in args.seeds:
                eval_spots, _ = simulate_paths(level, level.n_eval_paths, seed=seed + 1)
                for recurrent in (True, False):
                    tick = time.perf_counter()
                    record = evaluate_variant(level, seed, recurrent, eval_spots)
                    record["scenario"] = scenario["key"]
                    record["cost_rate"] = float(scenario["cost_rate"])
                    record["cvar_alpha"] = float(scenario["cvar_alpha"])
                    # The PnL sample is not used by any figure, so it is dropped to keep the JSON small
                    record.pop("pnl", None)
                    records.append(record)
                    print(
                        f"  scenario {scenario['key']} seed {seed} recurrent {recurrent}  "
                        f"risk {record['risk']:8.4f}  cost {record['mean_cost']:6.4f}  "
                        f"{time.perf_counter() - tick:6.1f}s",
                        flush=True,
                    )

    summary = aggregate_records(records)
    print_table(summary, records)

    payload = {
        "scenarios": [
            {
                "key": s["key"],
                "label": s["label"],
                "cost_rate": float(s["cost_rate"]),
                "cvar_alpha": float(s["cvar_alpha"]),
            }
            for s in SCENARIOS
        ],
        "seeds": list(args.seeds),
        "summary": summary,
        "records": records,
        "config": {
            "simulator": config.simulator,
            "loss_type": config.loss_type,
            "n_paths": config.n_paths,
            "n_eval_paths": config.n_eval_paths,
            "epochs": config.epochs,
            "norm_type": config.norm_type,
            "hidden_dim": config.hidden_dim,
            "n_layers": config.n_layers,
            "lr": config.lr,
            "seeds": list(args.seeds),
            "commit": "uncommitted",
        },
    }
    target = results / "recurrent_ablation.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {target}")

    if not args.smoke:
        plot_recurrent_ablation(records, SCENARIOS, results)
    print(f"total runtime {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
