import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import Config
from main import build_model, run_policy, seed_everything, train_policy
from src.heston import HestonParams, ModelHedge, heston_call_price
from src.loss import empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import PaperHestonSimulator, simulate_paths
from src.visualize import (
    plot_heston_fig2,
    plot_heston_fig3,
    plot_heston_fig6,
    plot_heston_fig7,
)

PAPER_Q = 1.69
PAPER_CVAR_50 = 1.94
PAPER_CVAR_99 = 3.49
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Section 5.2 Heston, two instruments")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--eval-paths", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.99])
    parser.add_argument("--results-dir", default="results/paper")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--aggregate", type=int, default=0)
    parser.add_argument("--skip-fig3", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run_shard(args: argparse.Namespace, config: Config) -> int:
    # Worker mode: train the (alpha, seed) cells assigned to this shard
    seed_everything(config.seed, deterministic=config.deterministic)
    cells = [(a, s) for a in args.alphas for s in args.seeds]
    seed_rank = {seed: i for i, seed in enumerate(args.seeds)}
    mine = [cell for cell in cells if seed_rank[cell[1]] % args.shards == args.shard]
    if not mine:
        raise SystemExit(f"shard {args.shard} has no cells, reduce --shards")
    print(
        f"shard {args.shard}/{args.shards}: {len(mine)} of {len(cells)} cells, "
        f"paths {config.n_paths:,}, epochs {config.epochs}",
        flush=True,
    )
    eval_spots_t, eval_var_t = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    records: list[dict[str, float]] = []
    for alpha, seed in mine:
        level = replace(config, cvar_alpha=float(alpha))
        tick = time.perf_counter()
        train_spots, train_var = simulate_paths(level, level.n_paths, seed=int(seed))
        model, history, _ = train_policy(level, train_spots, train_var, seed=int(seed))
        pnl, _ = run_policy(model, eval_spots_t, eval_var_t, level, return_path=False)
        values = pnl.numpy().astype(np.float64)
        # A subsample of PnL is stored so the paper figures can be redrawn from JSON
        keep = np.random.default_rng(seed).choice(
            values.size, size=min(values.size, 40_000), replace=False
        )
        records.append(
            {
                "alpha": float(alpha),
                "seed": int(seed),
                "cvar": cvar(values, alpha),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "final_train_risk": float(history[-1]),
                "sample": values[keep].tolist(),
            }
        )
        print(
            f"  alpha {alpha:.2f} seed {seed}  CVaR {records[-1]['cvar']:8.4f}  "
            f"mean {records[-1]['mean']:8.4f}  {time.perf_counter() - tick:6.1f}s",
            flush=True,
        )
        if args.smoke:
            break
    target = Path(args.raw_out or f"r2_shard{args.shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"shard {args.shard} wrote {target}", flush=True)
    return 0


def paper_config(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        "simulator": "paper_heston",
        "paper_heston": True,
        "loss_type": "cvar",
        "cost_rate": 0.0,
        "n_paths": args.paths,
        "n_eval_paths": args.eval_paths,
        "epochs": args.epochs,
        "run_cost_sensitivity": False,
        "results_dir": args.results_dir,
        # Paper Section 5.1 shape: two hidden layers of d + 15 units, ReLU
        "hidden_dim": 30,
        "n_layers": 2,
        "lr": 0.001,
    }
    config = replace(config, **overrides)
    config.validate()
    return config


def heston_vector(config: Config) -> np.ndarray:
    # Grid used for the analytic model hedge
    params = HestonParams(
        kappa=config.paper_heston_kappa,
        theta=config.paper_heston_theta,
        xi=config.paper_heston_xi,
        rho=config.paper_heston_rho,
        v0=config.paper_heston_v0,
    )
    return params


def marginal_vol(params: HestonParams, maturity: float) -> float:
    # Unconditional variance of log S_T, includes the vol of vol contribution
    var_t = (
        params.v0 * (1.0 - math.exp(-params.kappa * maturity)) / params.kappa
        + params.theta * (maturity - (1.0 - math.exp(-params.kappa * maturity)) / params.kappa)
    )
    return math.sqrt(max(var_t, 1e-12))


def build_model_hedge(config: Config) -> ModelHedge:
    # Stock grid spans roughly six marginal standard deviations of the terminal spot
    params = heston_vector(config)
    sigma = marginal_vol(params, config.T)
    horizon_sigma = config.S0 * sigma
    s_grid = np.linspace(config.S0 - 6.0 * horizon_sigma, config.S0 + 6.0 * horizon_sigma, 161)
    # The lower bound keeps dq/dv finite, near zero variance it explodes and is never visited
    v_grid = np.linspace(0.2 * config.paper_heston_theta, 4.0 * config.paper_heston_theta, 77)
    return ModelHedge(params, config.K, config.T, config.n_steps, s_grid, v_grid)


def model_hedge_pnl(
    config: Config,
    hedge: ModelHedge,
    spots: np.ndarray,
    variances: np.ndarray,
    swap_scale: float,
) -> dict[str, np.ndarray]:
    # Analytic two instrument hedge, costs are zero in this study
    simulator = PaperHestonSimulator(
        S0=config.S0,
        T=config.T,
        n_steps=config.n_steps,
        kappa=config.paper_heston_kappa,
        theta=config.paper_heston_theta,
        xi=config.paper_heston_xi,
        rho=config.paper_heston_rho,
        v0=config.paper_heston_v0,
    )
    swap = simulator.variance_swap(variances)
    d1, d2 = hedge.deltas(spots, variances)
    spots_t = torch.as_tensor(np.stack([spots, swap_scale * swap], axis=-1), dtype=torch.float32)
    deltas_t = torch.as_tensor(
        np.stack([d1, d2 / swap_scale], axis=-1), dtype=torch.float32
    )
    result = hedge_pnl(spots_t, deltas_t, config.K, 0.0)
    payoff = torch.relu(spots_t[:, -1, 0] - config.K)
    gains = result["gains"].numpy().astype(np.float64)
    return {
        "pnl": gains - payoff.numpy().astype(np.float64),
        "payoff": payoff.numpy().astype(np.float64),
        "d1": d1,
        "d2": d2,
        "gains": gains,
    }


def cvar(values: np.ndarray, alpha: float) -> float:
    return float(empirical_cvar(torch.as_tensor(values, dtype=torch.float32), alpha))


def summarize(values: np.ndarray, alphas: list[float]) -> dict[str, float]:
    out = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "q05": float(np.percentile(values, 5.0)),
        "q95": float(np.percentile(values, 95.0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for alpha in alphas:
        out[f"cvar_{alpha:.2f}"] = cvar(values, alpha)
    return out


def main() -> int:
    args = parse_args()
    config = paper_config(args)
    seed_everything(config.seed, deterministic=config.deterministic)
    results = Path(config.results_path)
    results.mkdir(parents=True, exist_ok=True)
    if args.shards > 1:
        return run_shard(args, config)

    params = heston_vector(config)
    q = float(heston_call_price(config.T, config.S0, config.paper_heston_v0, config.K, params))
    print("=" * 78)
    print("Paper Section 5.2: Heston, no costs, stock plus variance swap")
    print("=" * 78)
    print(
        f"kappa {params.kappa}  theta {params.theta}  xi {params.xi}  rho {params.rho}  "
        f"v0 {params.v0}"
    )
    print(f"Feller 2 kappa theta - xi^2 = {params.feller:.4f} (negative, exact CIR step needed)")
    print(f"model price q = {q:.4f}   paper reports {PAPER_Q:.2f}")
    if abs(q - PAPER_Q) > 0.02:
        raise SystemExit(f"pricer gate failed: q = {q:.6f} does not match the paper")

    print("building the analytic model hedge grid")
    hedge = build_model_hedge(config)

    print(f"simulating {config.n_paths:,} training and {config.n_eval_paths:,} test paths")
    eval_spots_t, eval_var_t = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    eval_spots = eval_spots_t[:, :, 0].numpy().astype(np.float64)
    eval_var = eval_var_t.numpy().astype(np.float64)
    model_hedge = model_hedge_pnl(config, hedge, eval_spots, eval_var, config.swap_scale)
    print(
        f"model hedge delta2 range {model_hedge['d2'].min():.2f} to {model_hedge['d2'].max():.2f}, "
        f"scaled by 1/{config.swap_scale:.0f} for the network"
    )

    records: dict[str, object] = {
        "q": q,
        "paper": {"q": PAPER_Q, "cvar_0.50": PAPER_CVAR_50, "cvar_0.99": PAPER_CVAR_99},
        "alphas": list(args.alphas),
        "seeds": list(args.seeds),
        "model_hedge": summarize(model_hedge["pnl"], list(args.alphas)),
        "model_hedge_delta2_range": [
            float(model_hedge["d2"].min()),
            float(model_hedge["d2"].max()),
        ],
        "deep": {},
        "history": {},
    }
    print()
    print("Analytic model hedge on the test set")
    print(
        f"  mean {records['model_hedge']['mean']:.4f}  std {records['model_hedge']['std']:.4f}  "
        + "  ".join(
            f"CVaR{a:.2f} {records['model_hedge'][f'cvar_{a:.2f}']:.4f}" for a in args.alphas
        )
    )

    overall_started = time.perf_counter()

    if args.aggregate:
        # Aggregation only, reuse the shard records and retrain nothing
        shard_records: list[dict] = []
        for path in sorted(results.glob("raw_r2_shard*.json")):
            shard_records.extend(json.loads(path.read_text(encoding="utf-8")))
            print(f"loaded {path.name}  {len(shard_records)} records so far")
        for alpha in args.alphas:
            subset = [r for r in shard_records if abs(float(r["alpha"]) - alpha) < 1e-9]
            if not subset:
                continue
            block = {
                "mean": float(np.mean([r["mean"] for r in subset])),
                "std": float(np.mean([r["std"] for r in subset])),
                "per_seed_cvar": [float(r["cvar"]) for r in subset],
                "per_seed_mean": [float(r["mean"]) for r in subset],
                "n": len(subset),
            }
            block[f"cvar_{alpha:.2f}"] = float(np.mean([r["cvar"] for r in subset]))
            # Any other alpha key the figure code asks for is filled from the same records
            for other in args.alphas:
                block[f"cvar_{other:.2f}"] = float(np.mean([r["cvar"] for r in subset]))
            records["deep"][f"{alpha:.2f}"] = block
            # Per seed PnL samples feed figures 2, 6 and 7
            samples = [
                np.asarray(r["sample"], dtype=np.float64).tolist()
                for r in subset
                if "sample" in r
            ]
            if samples:
                records.setdefault("samples", {})
                records["samples"][f"{alpha:.2f}"] = samples
            print(
                f"  alpha {alpha:.2f}: mean CVaR {block[f'cvar_{alpha:.2f}']:.4f}  "
                f"target {PAPER_CVAR_50 if alpha < 0.75 else PAPER_CVAR_99:.2f}"
            )
        # The analytic model hedge is needed for the figure overlays
        hedge = build_model_hedge(config)
        eval_spots_np = eval_spots_t[:, :, 0].numpy().astype(np.float64)
        eval_var_np = eval_var_t.numpy().astype(np.float64)
        mh = model_hedge_pnl(config, hedge, eval_spots_np, eval_var_np, config.swap_scale)
        keep = np.random.default_rng(99).choice(
            mh["pnl"].size, size=min(mh["pnl"].size, 40_000), replace=False
        )
        records["model_sample"] = mh["pnl"][keep].tolist()
        per_seed_pnl = {alpha: [] for alpha in args.alphas}
        records["simple"] = records.get("simple", {})
        payload = {
            **records,
            "config": {
                "simulator": "paper_heston",
                "loss_type": "cvar",
                "cost_rate": 0.0,
                "n_paths": config.n_paths,
                "n_eval_paths": config.n_eval_paths,
                "epochs": config.epochs,
                "hidden_dim": config.hidden_dim,
                "n_layers": config.n_layers,
                "swap_scale": config.swap_scale,
                "seeds": list(args.seeds),
                "aggregated": True,
                "commit": "uncommitted",
            },
        }
        target = results / "heston_nocost.json"
        target.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"\nwritten {target}")

        sample_block = records.get("samples", {})
        model_sample = np.asarray(records.get("model_sample", []), dtype=np.float64)
        # Convert back to arrays for the plotting calls
        sample_block = {
            key: [np.asarray(item, dtype=np.float64) for item in value]
            for key, value in sample_block.items()
        }
        if args.smoke or not sample_block or model_sample.size == 0:
            return 0
        print("building figures 2, 6 and 7")
        main_alpha = f"{float(args.alphas[0]):.2f}"
        plot_heston_fig2(
            sample_block[main_alpha], model_sample, q,
            records["deep"][main_alpha], results,
        )
        plot_heston_fig6(
            {float(k): v for k, v in sample_block.items()}, records["deep"], q, results,
        )
        plot_heston_fig7(
            {float(k): v for k, v in sample_block.items()}, model_sample,
            records["deep"], q, results,
        )
        if args.skip_fig3:
            return 0
        # Figure 3 uses the non recurrent network, as the paper does
        print("training the non recurrent variant for Figure 3")
        fig_level = replace(
            config, cvar_alpha=float(args.alphas[0]), use_prev_delta=False
        )
        fig_model = None
        for seed in list(args.seeds)[:3]:
            train_spots, train_var = simulate_paths(fig_level, fig_level.n_paths, seed=int(seed))
            fig_model, _, _ = train_policy(fig_level, train_spots, train_var, seed=int(seed))
            print(f"  non recurrent seed {seed} trained", flush=True)
        hedge = build_model_hedge(config)
        plot_heston_fig3(
            fig_model, hedge, config, eval_spots_np, eval_var_np, results,
        )
        return 0

    per_seed_pnl: dict[float, list[np.ndarray]] = {alpha: [] for alpha in args.alphas}
    for alpha in args.alphas:
        level = replace(config, cvar_alpha=float(alpha))
        per_seed_pnl[alpha] = []
        seeds = args.seeds[:1] if args.smoke else args.seeds
        for seed in seeds:
            tick = time.perf_counter()
            train_spots, train_var = simulate_paths(level, level.n_paths, seed=int(seed))
            model, history, _ = train_policy(level, train_spots, train_var, seed=int(seed))
            deep_pnl, _ = run_policy(model, eval_spots_t, eval_var_t, level, return_path=False)
            values = deep_pnl.numpy().astype(np.float64)
            per_seed_pnl[alpha].append(values)
            records["history"][f"{alpha:.2f}_{seed}"] = list(history)
            print(
                f"  alpha {alpha:.2f} seed {seed}  CVaR {cvar(values, alpha):8.4f}  "
                f"mean {values.mean():8.4f}  std {values.std(ddof=1):7.4f}  "
                f"{time.perf_counter() - tick:6.1f}s",
                flush=True,
            )
        pooled = np.concatenate(per_seed_pnl[alpha])
        block = summarize(pooled, list(args.alphas))
        block["per_seed_cvar"] = [cvar(v, alpha) for v in per_seed_pnl[alpha]]
        block["per_seed_mean"] = [float(v.mean()) for v in per_seed_pnl[alpha]]
        records["deep"][f"{alpha:.2f}"] = block
        print(
            f"  alpha {alpha:.2f} pooled: CVaR {block[f'cvar_{alpha:.2f}']:.4f}  "
            f"paper {PAPER_CVAR_50 if alpha < 0.75 else PAPER_CVAR_99:.2f}"
        )

    # Figure 3 needs the non recurrent network, retrain once without the previous position
    print("\ntraining the non recurrent variant for Figure 3")
    alpha_main = float(args.alphas[0])
    level = replace(config, cvar_alpha=alpha_main, use_prev_delta=False)
    fig_seeds = args.seeds[:1] if args.smoke else args.seeds[:3]
    fig_pnls: list[np.ndarray] = []
    fig_model = None
    for seed in fig_seeds:
        train_spots, train_var = simulate_paths(level, level.n_paths, seed=int(seed))
        fig_model, _, _ = train_policy(level, train_spots, train_var, seed=int(seed))
        pnl, _ = run_policy(fig_model, eval_spots_t, eval_var_t, level, return_path=False)
        fig_pnls.append(pnl.numpy().astype(np.float64))
        print(f"  non recurrent seed {seed}  CVaR {cvar(fig_pnls[-1], alpha_main):.4f}", flush=True)
    simple_pnl = np.concatenate(fig_pnls)
    records["simple"] = summarize(simple_pnl, list(args.alphas))
    records["simple"]["per_seed_cvar"] = [cvar(v, alpha_main) for v in fig_pnls]

    payload = {
        **records,
        "config": {
            "simulator": "paper_heston",
            "loss_type": "cvar",
            "cost_rate": 0.0,
            "n_paths": config.n_paths,
            "n_eval_paths": config.n_eval_paths,
            "epochs": config.epochs,
            "n_steps": config.n_steps,
            "T": config.T,
            "hidden_dim": config.hidden_dim,
            "n_layers": config.n_layers,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "swap_scale": config.swap_scale,
            "seeds": list(args.seeds),
            "commit": "uncommitted",
        },
    }
    target = results / "heston_nocost.json"
    target.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten {target}")

    if not args.smoke:
        print("building figures 2, 3, 6 and 7")
        plot_heston_fig2(
            per_seed_pnl[alpha_main], model_hedge["pnl"], q,
            records["deep"][f"{alpha_main:.2f}"], results,
        )
        plot_heston_fig6(
            per_seed_pnl, records["deep"], q, results,
        )
        plot_heston_fig7(
            per_seed_pnl, model_hedge["pnl"], records["deep"], q, results,
        )
        plot_heston_fig3(
            fig_model, hedge, config, eval_spots, eval_var, results,
        )
    print(f"total runtime {time.perf_counter() - overall_started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
