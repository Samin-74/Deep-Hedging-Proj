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
from paper_heston import build_model_hedge, model_hedge_pnl
from src.heston import HestonParams, heston_call_price
from src.loss import EntropicRiskLoss
from src.simulators import simulate_paths
from src.visualize import plot_price_asymptotics, plot_heston_asymptotics

PAPER_COST_GRID = (0.000625, 0.00125, 0.0025, 0.005, 0.01)
DEFAULT_COSTS = (0.0,) + PAPER_COST_GRID
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Figure 11 Heston price asymptotics")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--eval-paths", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--costs", type=float, nargs="+", default=list(DEFAULT_COSTS))
    parser.add_argument("--results-dir", default="results/paper")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--aggregate", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        "simulator": "paper_heston",
        "paper_heston": True,
        "loss_type": "entropic",
        "lambda_entropic": 1.0,
        "n_paths": args.paths,
        "n_eval_paths": args.eval_paths,
        "epochs": args.epochs,
        "run_cost_sensitivity": False,
        "results_dir": args.results_dir,
        "hidden_dim": 30,
        "n_layers": 2,
    }
    config = replace(config, **overrides)
    config.validate()
    return config


def run_level(config: Config, cost: float, seed: int, eval_spots_t: torch.Tensor,
              eval_var_t: torch.Tensor) -> dict[str, float]:
    # One training at one cost level with the two instrument policy
    level = replace(config, cost_rate=float(cost), seed=int(seed))
    train_spots, train_var = simulate_paths(level, level.n_paths, seed=int(seed))
    model, history, _ = train_policy(level, train_spots, train_var, seed=int(seed))
    pnl, _ = run_policy(model, eval_spots_t, eval_var_t, level, return_path=False)
    entropic = float(EntropicRiskLoss(config.lambda_entropic)(pnl))
    return {
        "cost": float(cost),
        "seed": int(seed),
        "entropic": entropic,
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(unbiased=True)),
        "final_train_risk": float(history[-1]),
    }


def summarise(records: list[dict[str, float]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[float, list[dict[str, float]]] = {}
    for record in records:
        grouped.setdefault(float(record["cost"]), []).append(record)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for cost, rows in sorted(grouped.items()):
        out[f"{cost:.6f}"] = {}
        for key in ("entropic", "mean_pnl", "std_pnl", "final_train_risk"):
            values = np.array([float(r[key]) for r in rows])
            out[f"{cost:.6f}"][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                "n": int(values.size),
            }
    return out


def fit_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"slope": float(coef[1]), "intercept": float(coef[0])}


def run_shard(args: argparse.Namespace, config: Config) -> int:
    seed_everything(config.seed, deterministic=config.deterministic)
    cells = [(c, s) for c in args.costs for s in args.seeds]
    seed_rank = {seed: i for i, seed in enumerate(args.seeds)}
    mine = [cell for cell in cells if seed_rank[cell[1]] % args.shards == args.shard]
    if not mine:
        raise SystemExit(f"shard {args.shard} has no cells, reduce --shards")
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(cells)} cells", flush=True)
    eval_spots_t, eval_var_t = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    records: list[dict[str, float]] = []
    for cost, seed in mine:
        tick = time.perf_counter()
        record = run_level(config, cost, seed, eval_spots_t, eval_var_t)
        records.append(record)
        print(
            f"  cost {cost:8.6f} seed {seed}  p_eps {record['entropic']:8.4f}  "
            f"{time.perf_counter() - tick:6.1f}s",
            flush=True,
        )
        if args.smoke:
            break
    target = Path(args.raw_out or f"r5_shard{args.shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"shard {args.shard} wrote {target}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    config = base_config(args)
    results = Path(config.results_path)
    results.mkdir(parents=True, exist_ok=True)
    if args.shards > 1:
        return run_shard(args, config)

    params = HestonParams(
        kappa=config.paper_heston_kappa,
        theta=config.paper_heston_theta,
        xi=config.paper_heston_xi,
        rho=config.paper_heston_rho,
        v0=config.paper_heston_v0,
    )
    q = float(heston_call_price(config.T, config.S0, params.v0, config.K, params))
    print("=" * 78)
    print("Paper Figure 11: Heston price asymptotics with two instruments")
    print("=" * 78)
    print(f"model price q = {q:.4f}   cost grid {args.costs}   seeds {args.seeds}")

    started = time.perf_counter()
    if args.aggregate:
        records: list[dict[str, float]] = []
        for path in sorted(results.glob("raw_r5_shard*.json")):
            records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}  {len(records)} records so far")
        eval_spots_t, eval_var_t = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    else:
        seed_everything(config.seed, deterministic=config.deterministic)
        eval_spots_t, eval_var_t = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
        records = []
        for cost in args.costs:
            for seed in args.seeds:
                tick = time.perf_counter()
                record = run_level(config, cost, seed, eval_spots_t, eval_var_t)
                records.append(record)
                print(
                    f"  cost {cost:8.6f} seed {seed}  p_eps {record['entropic']:8.4f}  "
                    f"{time.perf_counter() - tick:6.1f}s",
                    flush=True,
                )

    summary = summarise(records)
    print()
    print("Heston entropic price against cost, 5 seed mean")
    header = f"{'cost':>9} {'p_eps':>9} {'p-q':>9} {'ln(p-q)':>9} {'mean PnL':>10}"
    print(header)
    print("-" * len(header))
    for key, block in summary.items():
        p = block["entropic"]["mean"]
        excess = p - q
        print(
            f"{float(key):9.6f} {p:9.4f} {excess:9.4f} "
            f"{math.log(excess) if excess > 0 else float('nan'):9.4f} "
            f"{block['mean_pnl']['mean']:10.4f}"
        )

    p0 = summary[min(summary, key=lambda k: float(k))]["entropic"]["mean"]
    grid = sorted(float(k) for k in summary if float(k) > 0.0)
    fit_out: dict[str, dict[str, float]] = {}
    if len(grid) >= 2:
        x = np.log(np.array(grid))
        y_q = np.log(np.array([summary[f"{c:.6f}"]["entropic"]["mean"] - q for c in grid]))
        y_p0 = np.log(np.array([summary[f"{c:.6f}"]["entropic"]["mean"] - p0 for c in grid]))
        fit_out["free_vs_q"] = fit_slope(x, y_q)
        fit_out["free_vs_p0"] = fit_slope(x, y_p0)
        fit_out["fixed_two_thirds_vs_q"] = {
            "fixed_slope": 2.0 / 3.0,
            "intercept": float(np.mean(y_q - (2.0 / 3.0) * x)),
        }
        fit_out["fixed_two_thirds_vs_p0"] = {
            "fixed_slope": 2.0 / 3.0,
            "intercept": float(np.mean(y_p0 - (2.0 / 3.0) * x)),
        }
        print()
        print("Fitted log-log slopes, the paper predicts 2/3")
        print(f"  against q    {fit_out['free_vs_q']['slope']:+.4f}")
        print(f"  against p_0  {fit_out['free_vs_p0']['slope']:+.4f}")

    payload = {
        "q": q,
        "p_0": p0,
        "cost_grid": list(args.costs),
        "seeds": list(args.seeds),
        "records": records,
        "summary": summary,
        "fit": fit_out,
        "config": {
            "simulator": "paper_heston",
            "loss_type": "entropic",
            "n_paths": config.n_paths,
            "n_eval_paths": config.n_eval_paths,
            "epochs": config.epochs,
            "hidden_dim": config.hidden_dim,
            "n_layers": config.n_layers,
            "swap_scale": config.swap_scale,
            "seeds": list(args.seeds),
            "commit": "uncommitted",
        },
    }
    target = results / "price_asymptotics_heston.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {target}")

    if not args.smoke and len(grid) >= 2:
        plot_heston_asymptotics(payload, results)
    print(f"total runtime {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
