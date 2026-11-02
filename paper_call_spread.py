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
from src.heston import HestonParams, ModelHedge, heston_call_price
from src.loss import empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import PaperHestonSimulator, simulate_paths
from src.visualize import plot_call_spread

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
PAPER_Q = 1.69


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Figures 8 and 9: call spread")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--eval-paths", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.95, 0.99])
    parser.add_argument("--k2", type=float, default=101.0)
    parser.add_argument("--results-dir", default="results/paper")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--aggregate", type=int, default=0)
    parser.add_argument("--simple", action="store_true", help="drop the previous position input")
    parser.add_argument("--warm-start", action="store_true", help="start the higher alpha from the lower one")
    parser.add_argument("--warm-epochs", type=int, default=20)
    parser.add_argument("--warm-lr", type=float, default=0.0005)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def paper_config(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        "simulator": "paper_heston",
        "paper_heston": True,
        "loss_type": "cvar",
        "cost_rate": 0.0,
        "payoff_second_strike": float(args.k2),
        "n_paths": args.paths,
        "n_eval_paths": args.eval_paths,
        "epochs": args.epochs,
        "run_cost_sensitivity": False,
        "results_dir": args.results_dir,
        "hidden_dim": 30,
        "n_layers": 2,
        "use_prev_delta": not bool(args.simple),
    }
    config = replace(config, **overrides)
    config.validate()
    return config


def spread_payoff(spot: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    # Long the 100 call and short the 101 call, one unit of notional
    return torch.clamp(spot - k1, min=0.0) - torch.clamp(spot - k2, min=0.0)


def spread_pnl(spots: torch.Tensor, deltas: torch.Tensor, config: Config) -> torch.Tensor:
    # Same PnL engine as the main study, with the call spread liability
    return hedge_pnl(
        spots, deltas, config.K, config.cost_rate,
        return_path=False, second_strike=config.payoff_second_strike,
    )["pnl"]


def model_spread_delta(
    params: HestonParams,
    config: Config,
    strikes: tuple[float, float],
    s_grid: np.ndarray,
    v_grid: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    # Analytic spread delta: long the low strike call, short the high strike call
    n_s, n_v = s_grid.size, v_grid.size
    out = np.zeros((n_steps, n_s, n_v))
    d_s = s_grid[1] - s_grid[0]
    signs = (1.0, -1.0)
    for k in range(n_steps):
        tau = config.T - k * config.dt
        total = np.zeros((n_s, n_v))
        for strike, sign in zip(strikes, signs):
            price = heston_call_price(
                tau,
                np.broadcast_to(s_grid[:, None], (n_s, n_v)),
                np.broadcast_to(v_grid[None, :], (n_s, n_v)),
                float(strike),
                params,
            )
            grad = np.gradient(price, d_s, axis=0)
            total += sign * grad
        out[k] = total
    return out


def evaluate_alpha(
    config: Config,
    seed: int,
    alpha: float,
    eval_spots_t: torch.Tensor,
    eval_var_t: torch.Tensor,
    initial_model=None,
    epochs: int | None = None,
    lr: float | None = None,
) -> dict[str, object]:
    level = replace(config, cvar_alpha=float(alpha), seed=int(seed))
    if epochs is not None:
        level = replace(level, epochs=int(epochs))
    if lr is not None:
        level = replace(level, lr=float(lr))
    train_spots, train_var = simulate_paths(level, level.n_paths, seed=int(seed))
    model, history, _ = train_policy(
        level, train_spots, train_var, seed=int(seed), initial_model=initial_model
    )
    pnl, _ = run_policy(model, eval_spots_t, eval_var_t, level, return_path=False)
    record = {
        "alpha": float(alpha),
        "seed": int(seed),
        "cvar": float(empirical_cvar(pnl, alpha)),
        "mean": float(pnl.mean()),
        "std": float(pnl.std(unbiased=True)),
        "cvar_95": float(empirical_cvar(pnl, 0.95)),
        "cvar_99": float(empirical_cvar(pnl, 0.99)),
        "final_train_risk": float(history[-1]),
        "warm_started": initial_model is not None,
    }
    # The model is returned separately so it never reaches the JSON records
    return record, model


def run_shard(args: argparse.Namespace, config: Config) -> int:
    seed_everything(config.seed, deterministic=config.deterministic)
    cells = [(a, s) for a in args.alphas for s in args.seeds]
    seed_rank = {seed: i for i, seed in enumerate(args.seeds)}
    mine = [cell for cell in cells if seed_rank[cell[1]] % args.shards == args.shard]
    if not mine:
        raise SystemExit(f"shard {args.shard} has no cells, reduce --shards")
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(cells)} cells", flush=True)
    eval_spots_t, eval_var_t = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    records: list[dict] = []
    # Warm start trains the lowest alpha first, then reuses its weights for the rest
    base_alpha = min(args.alphas)
    warm_models: dict[int, object] = {}
    seen: set[tuple[float, int]] = set()
    for alpha, seed in mine:
        tick = time.perf_counter()
        initial = None
        epochs = None
        lr = None
        if args.warm_start and abs(alpha - base_alpha) > 1e-9:
            if seed not in warm_models:
                base, base_model = evaluate_alpha(config, seed, base_alpha, eval_spots_t, eval_var_t)
                warm_models[seed] = base_model
                # The base cell may also be in this shard, so record it only once
                if (base_alpha, seed) not in seen:
                    seen.add((base_alpha, seed))
                    records.append(base)
            initial = warm_models[seed]
            epochs = args.warm_epochs
            lr = args.warm_lr
        record, _ = evaluate_alpha(
            config, seed, alpha, eval_spots_t, eval_var_t,
            initial_model=initial, epochs=epochs, lr=lr,
        )
        if (alpha, seed) not in seen:
            seen.add((alpha, seed))
            records.append(record)
        print(
            f"  alpha {alpha:.2f} seed {seed}  CVaR {record['cvar']:8.4f}  "
            f"mean {record['mean']:8.4f}  {time.perf_counter() - tick:6.1f}s",
            flush=True,
        )
        if args.smoke:
            break
    target = Path(args.raw_out or f"r4_shard{args.shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"shard {args.shard} wrote {target}", flush=True)
    return 0


def aggregate_records(records: list[dict], alphas: list[float]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for alpha in alphas:
        subset = [r for r in records if abs(float(r["alpha"]) - alpha) < 1e-9]
        if not subset:
            continue
        entry: dict[str, object] = {"n": len(subset)}
        for metric in ("cvar", "mean", "std", "cvar_95", "cvar_99"):
            values = np.array([float(r[metric]) for r in subset])
            entry[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            }
        out[f"{alpha:.2f}"] = entry
    return out


def main() -> int:
    args = parse_args()
    config = paper_config(args)
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
    strikes = (config.K, float(args.k2))
    q_spread = float(
        heston_call_price(config.T, config.S0, params.v0, strikes[0], params)
        - heston_call_price(config.T, config.S0, params.v0, strikes[1], params)
    )
    print("=" * 78)
    print("Paper Figures 8 and 9: call spread, risk aversion flattens the hedge")
    print("=" * 78)
    print(f"strikes {strikes}  model spread price {q_spread:.4f}")

    started = time.perf_counter()
    if args.aggregate:
        records: list[dict] = []
        for path in sorted(results.glob("raw_r4_shard*.json")):
            records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}  {len(records)} records so far")
        eval_spots_t, eval_var_t = simulate_paths(
            config, config.n_eval_paths, seed=config.seed + 1
        )
    else:
        seed_everything(config.seed, deterministic=config.deterministic)
        eval_spots_t, eval_var_t = simulate_paths(
            config, config.n_eval_paths, seed=config.seed + 1
        )
        records = []
        for alpha in args.alphas:
            for seed in args.seeds:
                tick = time.perf_counter()
                record, _ = evaluate_alpha(config, seed, alpha, eval_spots_t, eval_var_t)
                records.append(record)
                print(
                    f"  alpha {alpha:.2f} seed {seed}  CVaR {record['cvar']:8.4f}  "
                    f"mean {record['mean']:8.4f}  {time.perf_counter() - tick:6.1f}s",
                    flush=True,
                )

    summary = aggregate_records(records, list(args.alphas))
    print()
    print("Call spread hedging error, 5 seed mean")
    for key, entry in summary.items():
        print(
            f"  alpha {key}  CVaR {entry['cvar']['mean']:7.4f}+/-{entry['cvar']['std']:6.4f}  "
            f"mean {entry['mean']['mean']:7.4f}  std {entry['std']['mean']:6.4f}  n {entry['n']}"
        )

    # Holdings surface at mid horizon for each alpha
    lvl = replace(config, cvar_alpha=float(args.alphas[0]))
    eval_spots = eval_spots_t[:, :, 0].numpy().astype(np.float64)
    eval_var = eval_var_t.numpy().astype(np.float64)
    step = config.n_steps // 2
    s_grid = np.linspace(97.0, 104.0, 43)
    v_grid = np.linspace(0.02, 0.10, 9)
    s_mesh, v_mesh = np.meshgrid(s_grid, v_grid, indexing="ij")
    holdings: dict[str, np.ndarray] = {}
    for alpha in args.alphas:
        level = replace(config, cvar_alpha=float(alpha))
        model = None
        for seed in args.seeds[:1] if args.smoke else args.seeds[:3]:
            train_spots, train_var = simulate_paths(level, level.n_paths, seed=int(seed))
            model, _, _ = train_policy(level, train_spots, train_var, seed=int(seed))
        if model is None:
            continue
        with torch.no_grad():
            flat_s = torch.as_tensor(s_mesh.reshape(-1), dtype=torch.float32)
            flat_v = torch.as_tensor(v_mesh.reshape(-1), dtype=torch.float32)
            prev = model.policy_at_step(flat_s, step, 0.0, variance=flat_v)
        stock = prev[:, 0] if prev.dim() == 2 else prev
        holdings[f"{alpha:.2f}"] = stock.numpy().reshape(s_mesh.shape).astype(np.float64)

    model_grid = model_spread_delta(params, config, strikes, s_grid, v_grid, config.n_steps)

    payload = {
        "strikes": list(strikes),
        "q_spread": q_spread,
        "alphas": list(args.alphas),
        "seeds": list(args.seeds),
        "summary": summary,
        "records": records,
        "holdings": {k: v.tolist() for k, v in holdings.items()},
        "holdings_grid": {
            "s": s_grid.tolist(),
            "v": v_grid.tolist(),
            "step": step,
            "model": model_grid[step].tolist(),
        },
        "config": {
            "simulator": "paper_heston",
            "loss_type": "cvar",
            "cost_rate": 0.0,
            "n_paths": config.n_paths,
            "n_eval_paths": config.n_eval_paths,
            "epochs": config.epochs,
            "seeds": list(args.seeds),
            "simple": bool(args.simple),
            "warm_start": bool(args.warm_start),
            "warm_epochs": int(args.warm_epochs) if args.warm_start else 0,
            "warm_lr": float(args.warm_lr) if args.warm_start else 0.0,
            "commit": "uncommitted",
        },
    }
    suffix = "_warmstart" if args.warm_start else ""
    target = results / f"call_spread{suffix}.json"
    target.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten {target}")
    if not args.smoke:
        plot_call_spread(payload, results, suffix=suffix)
    print(f"total runtime {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
