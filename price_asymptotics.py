import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import Config
from main import (
    benchmark_results,
    build_model,
    extra_summary,
    run_policy,
    seed_everything,
    train_policy,
)
from src.benchmarks import bs_call_price
from src.heston import black_scholes_call_price
from src.loss import EntropicRiskLoss, empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths
from src.visualize import plot_price_asymptotics, plot_cost_sensitivity_5seed

# Paper Figure 10 cost grid, printed as eps_i = 2^(-i+5) in the source
PAPER_COST_GRID = (0.000625, 0.00125, 0.0025, 0.005, 0.01)
DEFAULT_COSTS = (0.0,) + PAPER_COST_GRID
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Figure 10 price asymptotics")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--eval-paths", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--costs", type=float, nargs="+", default=list(DEFAULT_COSTS))
    parser.add_argument("--results-dir", default="results/paper")
    # Sharding lets several processes cover disjoint (cost, seed) cells
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--aggregate", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
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


def entropy(pnl: torch.Tensor, lam: float) -> float:
    return float(EntropicRiskLoss(lam)(pnl))


def run_level(
    config: Config,
    cost: float,
    seed: int,
    eval_spots: torch.Tensor,
) -> dict[str, float]:
    # One training at one cost level, scored on a shared evaluation set
    level = replace(config, cost_rate=float(cost), seed=int(seed))
    train_spots, train_var = simulate_paths(level, level.n_paths, seed=seed)
    model, history, _ = train_policy(level, train_spots, train_var, seed=seed)
    deep_pnl, _, _ = run_policy(model, eval_spots, None, level)
    with torch.no_grad():
        deltas = model(eval_spots)
    detail = hedge_pnl(eval_spots, deltas, level.K, level.cost_rate)
    benches = benchmark_results(eval_spots, level)
    record = {
        "cost": float(cost),
        "seed": int(seed),
        "q": float(bs_call_price(level.S0, level.K, level.r, level.benchmark_sigma, level.T)),
        "entropic": entropy(deep_pnl, config.lambda_entropic),
        "mean_pnl": float(deep_pnl.mean()),
        "std_pnl": float(deep_pnl.std(unbiased=True)),
        "cvar_95": float(empirical_cvar(deep_pnl, 0.95)),
        "cvar_99": float(empirical_cvar(deep_pnl, 0.99)),
        "mean_cost": float(detail["costs"].mean()),
        "turnover": float(detail["turnover"].mean()),
        "final_train_risk": float(history[-1]),
        "mean_delta": float(deltas.mean()),
    }
    for name, prefix in (("Black-Scholes", "bs"), ("Leland", "leland")):
        payload = benches[name]
        record[f"{prefix}_mean"] = float(payload["pnl"].mean())
        record[f"{prefix}_cvar_95"] = float(empirical_cvar(payload["pnl"], 0.95))
        record[f"{prefix}_cvar_99"] = float(empirical_cvar(payload["pnl"], 0.99))
        record[f"{prefix}_entropic"] = entropy(payload["pnl"], config.lambda_entropic)
        record[f"{prefix}_mean_cost"] = float(payload["costs"].mean())
    return record


def fit_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    # Least squares slope with a normal approximation standard error
    if x.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "se": float("nan")}
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = max(x.size - 2, 1)
    sigma2 = float(residual @ residual) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return {
        "slope": float(coef[1]),
        "intercept": float(coef[0]),
        "se": float(math.sqrt(max(cov[1, 1], 0.0))),
    }


def bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    # Percentile interval over resampled cost levels
    rng = np.random.default_rng(seed)
    slopes = []
    n = x.size
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        if np.unique(x[index]).size < 2:
            continue
        fit = fit_slope(x[index], y[index])
        if math.isfinite(fit["slope"]):
            slopes.append(fit["slope"])
    if not slopes:
        return float("nan"), float("nan")
    low, high = np.percentile(slopes, [2.5, 97.5])
    return float(low), float(high)


def intercept_only_fit(
    x: np.ndarray,
    y: np.ndarray,
    fixed_slope: float,
) -> dict[str, float]:
    # Paper style fit: the slope is imposed and only the level is estimated
    intercept = float(np.mean(y - fixed_slope * x))
    return {"fixed_slope": float(fixed_slope), "intercept": intercept}


def summarise(records: list[dict[str, float]]) -> dict[str, dict[str, dict[str, float]]]:
    # Mean and sample standard deviation across seeds per cost level
    grouped: dict[float, list[dict[str, float]]] = {}
    for record in records:
        grouped.setdefault(record["cost"], []).append(record)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for cost, rows in sorted(grouped.items()):
        keys = [k for k in rows[0] if k not in {"cost", "seed"}]
        out[f"{cost:.6f}"] = {}
        for key in keys:
            values = [float(row[key]) for row in rows]
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            out[f"{cost:.6f}"][key] = {"mean": mean, "std": std, "n": len(values)}
    return out


def print_table(summary: dict, q: float) -> None:
    print()
    print("Paper Figure 10 price asymptotics: mean over seeds")
    header = (
        f"{'cost':>9} {'p_eps':>9} {'se':>7} {'p-q':>9} {'ln(p-q)':>9} "
        f"{'CVaR95':>8} {'CVaR99':>8} {'mean':>8} {'cost':>7}"
    )
    print(header)
    print("-" * len(header))
    for key, block in summary.items():
        cost = float(key)
        p = block["entropic"]["mean"]
        se = block["entropic"]["std"] / math.sqrt(max(block["entropic"]["n"], 1))
        excess = p - q
        log_excess = math.log(excess) if excess > 0.0 else float("nan")
        print(
            f"{cost:9.6f} {p:9.4f} {se:7.4f} {excess:9.4f} {log_excess:9.4f} "
            f"{block['cvar_95']['mean']:8.4f} {block['cvar_99']['mean']:8.4f} "
            f"{block['mean_pnl']['mean']:8.4f} {block['mean_cost']['mean']:7.4f}"
        )


def run_shard(args: argparse.Namespace, config: Config) -> int:
    # Worker mode: train the (cost, seed) cells assigned to this shard and dump raw records
    seed_everything(config.seed, deterministic=config.deterministic)
    cells = [(c, s) for c in args.costs for s in args.seeds]
    # Shard by position in the seed list so splits stay balanced when the cost list is short
    seed_rank = {seed: i for i, seed in enumerate(args.seeds)}
    mine = [cell for cell in cells if seed_rank[cell[1]] % args.shards == args.shard]
    if not mine:
        raise SystemExit(f"shard {args.shard} has no cells, reduce --shards")
    print(
        f"shard {args.shard}/{args.shards}: {len(mine)} of {len(cells)} cells, "
        f"paths {config.n_paths:,}, epochs {config.epochs}",
        flush=True,
    )
    eval_spots, _ = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    records: list[dict[str, float]] = []
    for cost, seed in mine:
        tick = time.perf_counter()
        record = run_level(config, cost, seed, eval_spots)
        records.append(record)
        print(
            f"  cost {cost:8.6f} seed {seed}  p_eps {record['entropic']:8.4f}  "
            f"cvar95 {record['cvar_95']:8.4f}  mean {record['mean_pnl']:8.4f}  "
            f"{time.perf_counter() - tick:6.1f}s",
            flush=True,
        )
        if args.smoke:
            break
    target = Path(args.raw_out or f"r1_shard{args.shard}.json")
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

    q = float(bs_call_price(config.S0, config.K, config.r, config.benchmark_sigma, config.T))
    q_heston = float(
        black_scholes_call_price(config.T, config.S0, config.benchmark_sigma**2, config.K)
    )
    print("=" * 78)
    print("Paper Section 5.3.1 and Figure 10: Black-Scholes price asymptotics")
    print("=" * 78)
    print(
        f"cost grid {args.costs}  seeds {args.seeds}  "
        f"paths {config.n_paths:,}  epochs {config.epochs}"
    )
    print(f"Black-Scholes price q = {q:.4f} (torch)  {q_heston:.4f} (numpy)")
    print(f"martingale check E[Z] = {q:.4f}, so pi(0) = 0 and p_eps equals the entropic risk")

    started = time.perf_counter()
    if args.aggregate:
        # Aggregation only, no training
        raw = sorted(results.glob("raw_r1_shard*.json"))
        if not raw:
            raise SystemExit("no raw shard files found, run the shards first")
        records = []
        for path in raw:
            records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}")
    else:
        eval_spots, _ = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
        print(f"evaluation set {config.n_eval_paths:,} paths, mean payoff {float(torch.relu(eval_spots[:, -1] - config.K).mean()):.4f}")
        records = []
        for cost in args.costs:
            for seed in args.seeds:
                tick = time.perf_counter()
                record = run_level(config, cost, seed, eval_spots)
                records.append(record)
                print(
                    f"  cost {cost:8.6f} seed {seed}  p_eps {record['entropic']:8.4f}  "
                    f"cvar95 {record['cvar_95']:8.4f}  mean {record['mean_pnl']:8.4f}  "
                    f"{time.perf_counter() - tick:6.1f}s",
                    flush=True,
                )
                if args.smoke:
                    break
            if args.smoke:
                break

    summary = summarise(records)
    print_table(summary, q)

    p0 = summary[min(summary, key=lambda k: float(k))]["entropic"]["mean"]
    positive_costs = [float(k) for k in summary if float(k) > 0.0]
    grid = sorted(positive_costs)
    x = np.log(np.array(grid))
    y_q = np.log(np.array([summary[f"{c:.6f}"]["entropic"]["mean"] - q for c in grid]))
    y_p0 = np.log(np.array([summary[f"{c:.6f}"]["entropic"]["mean"] - p0 for c in grid]))
    free_q = fit_slope(x, y_q)
    free_p0 = fit_slope(x, y_p0)
    ci_q = bootstrap_slope(x, y_q)
    ci_p0 = bootstrap_slope(x, y_p0)
    fixed_q = intercept_only_fit(x, y_q, 2.0 / 3.0)
    fixed_p0 = intercept_only_fit(x, y_p0, 2.0 / 3.0)

    print()
    print("Fitted log-log slopes, the paper predicts 2/3")
    print(f"  against q      free slope {free_q['slope']:.4f} +/- {free_q['se']:.4f} (se)  95% CI [{ci_q[0]:.4f}, {ci_q[1]:.4f}]")
    print(f"  against p_0    free slope {free_p0['slope']:.4f} +/- {free_p0['se']:.4f} (se)  95% CI [{ci_p0[0]:.4f}, {ci_p0[1]:.4f}]")
    print(f"  2/3 fixed intercept: against q {fixed_q['intercept']:.4f}, against p_0 {fixed_p0['intercept']:.4f}")

    payload = {
        "q": q,
        "q_numpy": q_heston,
        "p_0": p0,
        "cost_grid": list(args.costs),
        "seeds": list(args.seeds),
        "records": records,
        "summary": summary,
        "fit": {
            "free_vs_q": free_q,
            "free_vs_p0": free_p0,
            "ci_vs_q": {"low": ci_q[0], "high": ci_q[1]},
            "ci_vs_p0": {"low": ci_p0[0], "high": ci_p0[1]},
            "fixed_two_thirds_vs_q": fixed_q,
            "fixed_two_thirds_vs_p0": fixed_p0,
        },
        "config": {
            "simulator": config.simulator,
            "loss_type": config.loss_type,
            "lambda_entropic": config.lambda_entropic,
            "n_paths": config.n_paths,
            "n_eval_paths": config.n_eval_paths,
            "epochs": config.epochs,
            "n_steps": config.n_steps,
            "T": config.T,
            "sigma": config.benchmark_sigma,
            "norm_type": config.norm_type,
            "hidden_dim": config.hidden_dim,
            "n_layers": config.n_layers,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "seeds": list(args.seeds),
            "commit": "uncommitted",
        },
    }
    target = results / "price_asymptotics_bs.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {target}")

    if not args.smoke:
        plot_price_asymptotics(payload, results)
        plot_cost_sensitivity_5seed(payload, results)
    print(f"total runtime {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
