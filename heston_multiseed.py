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
from src.loss import empirical_cvar
from src.simulators import simulate_paths
from src.studies import aggregate, format_pm, markdown_pm
from src.visualize import plot_heston_cost_sweep, plot_heston_diagnostics, plot_heston_policy

SEEDS = (42, 43, 44, 45, 46)
METRICS = ("deep_mean", "deep_cvar_95", "deep_cvar_99", "bs_cvar_95", "bs_cvar_99", "leland_cvar_95", "leland_cvar_99")


def run_level(config: Config, cost: float, seeds, verbose: bool = True) -> list[dict[str, float]]:
    # One cost level, one record per seed
    level = replace(config, cost_rate=float(cost))
    level.validate()
    records: list[dict[str, float]] = []
    for seed in seeds:
        started = time.perf_counter()
        train_spots, train_var = simulate_paths(level, level.n_paths, seed=seed)
        eval_spots, eval_var = simulate_paths(level, level.n_eval_paths, seed=seed + 1)
        model, history, _ = train_policy(level, train_spots, train_var, seed=seed)
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
        records.append(record)
        if verbose:
            print(
                f"    seed {seed}: deep cvar95 {record['deep_cvar_95']:7.4f}  "
                f"cvar99 {record['deep_cvar_99']:7.4f}  {time.perf_counter() - started:6.1f}s",
                flush=True,
            )
    return records


def main() -> int:
    paths = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    output = sys.argv[3] if len(sys.argv) > 3 else "results/heston_multiseed.json"
    shard = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    shards = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    raw_out = sys.argv[6] if len(sys.argv) > 6 else None

    config = replace(Config(), simulator="heston", n_paths=paths, epochs=epochs)
    config.validate()
    seed_everything(config.seed, deterministic=config.deterministic)

    print("=" * 78)
    print("Heston multi-seed cost sweep")
    print("=" * 78)
    print(
        f"v0 {config.heston_v0:.4f}  kappa {config.heston_kappa:.2f}  theta {config.heston_theta:.4f}  "
        f"xi {config.heston_xi:.2f}  rho {config.heston_rho:.2f}"
    )
    print(f"paths {paths:,}  epochs {epochs}  seeds {list(SEEDS)}  levels {config.cost_levels}")
    print(f"shard {shard} of {shards}")
    print("=" * 78)

    started = time.perf_counter()
    per_level: dict[str, dict] = {}
    for cost in config.cost_levels:
        # Shard by seed rank so every cost level keeps the same seed coverage
        assigned = [s for i, s in enumerate(SEEDS) if i % shards == shard]
        if not assigned:
            continue
        print(f"  cost level {100.0 * cost:5.2f} percent  seeds {assigned}")
        records = run_level(config, cost, assigned)
        summary = aggregate(records, METRICS)
        per_level[f"{cost:.4f}"] = {"records": records, "summary": summary}

    if raw_out is not None:
        Path(raw_out).write_text(json.dumps(per_level, indent=2), encoding="utf-8")
        print(f"shard written to {raw_out}")
        return 0

    rows = []
    for cost in config.cost_levels:
        s = per_level[f"{cost:.4f}"]["summary"]
        rows.append(
            (
                f"{100.0 * cost:5.2f}%",
                format_pm(s["deep_mean"]["mean"], s["deep_mean"]["std"]),
                format_pm(s["deep_cvar_95"]["mean"], s["deep_cvar_95"]["std"]),
                format_pm(s["deep_cvar_99"]["mean"], s["deep_cvar_99"]["std"]),
                format_pm(s["bs_cvar_95"]["mean"], s["bs_cvar_95"]["std"]),
                format_pm(s["leland_cvar_95"]["mean"], s["leland_cvar_95"]["std"]),
            )
        )
    print()
    print(f"Heston multi-seed cost sweep, mean +/- std over {len(SEEDS)} seeds")
    print(
        format_table(
            ("Cost", "Deep mean", "Deep CVaR95", "Deep CVaR99", "BS CVaR95", "Leland CVaR95"),
            rows,
        )
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "paths": paths,
                "epochs": epochs,
                "levels": per_level,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"heston multi-seed written to {target}")

    # Figures from the seed 42 run at the main cost level
    main_cfg = replace(config, cost_rate=0.01)
    train_spots, train_var = simulate_paths(main_cfg, main_cfg.n_paths, seed=main_cfg.seed)
    eval_spots, eval_var = simulate_paths(main_cfg, main_cfg.n_eval_paths, seed=main_cfg.seed + 1)
    model, _, _ = train_policy(main_cfg, train_spots, train_var, seed=main_cfg.seed)
    model.eval()
    with torch.no_grad():
        deltas = model(eval_spots, eval_var)

    results = Path("results")
    figures = [
        plot_heston_cost_sweep({"levels": per_level}, results),
        plot_heston_policy(model, main_cfg, eval_spots, eval_var, results),
        plot_heston_diagnostics(eval_spots, eval_var, deltas, results),
    ]
    print()
    print("Saved Heston figures")
    for figure in figures:
        print(f"  {figure}")
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
