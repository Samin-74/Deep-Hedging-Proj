import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import (
    benchmark_results,
    format_table,
    print_cost_study,
    run_cost_study,
    run_policy,
    seed_everything,
    train_policy,
)
from src.loss import empirical_cvar
from src.simulators import simulate_paths
from src.visualize import plot_heston_cost_sweep, plot_heston_diagnostics


def main() -> int:
    # Heston transaction cost sweep with figures
    paths = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    output = sys.argv[3] if len(sys.argv) > 3 else "results/heston_cost_sweep.json"

    config = replace(
        Config(),
        simulator="heston",
        sensitivity_paths=paths,
        sensitivity_epochs=epochs,
        run_cost_sensitivity=True,
    )
    config.validate()
    seed_everything(config.seed, deterministic=config.deterministic)

    print("=" * 78)
    print("Heston transaction cost sweep")
    print("=" * 78)
    print(
        f"v0 {config.heston_v0:.4f}  kappa {config.heston_kappa:.2f}  "
        f"theta {config.heston_theta:.4f}  xi {config.heston_xi:.2f}  rho {config.heston_rho:.2f}"
    )
    print(
        f"benchmark sigma {config.benchmark_sigma:.4f}  paths {paths:,}  epochs {epochs}  "
        f"cost levels {config.cost_levels}"
    )
    print("=" * 78)

    records = run_cost_study(config)
    print_cost_study(records)

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"heston sweep written to {target}")

    # Diagnostics at the main cost level
    main_cfg = replace(config, cost_rate=0.01, n_paths=paths, epochs=epochs)
    train_spots, train_var = simulate_paths(main_cfg, main_cfg.n_paths, seed=main_cfg.seed + 7)
    eval_spots, eval_var = simulate_paths(main_cfg, main_cfg.n_eval_paths, seed=main_cfg.seed + 1)
    model, history, _ = train_policy(main_cfg, train_spots, train_var, seed=main_cfg.seed + 11)
    model.eval()
    with torch.no_grad():
        deltas = model(eval_spots, eval_var)
    pnl, _ = run_policy(model, eval_spots, eval_var, main_cfg, return_path=False)
    benches = benchmark_results(eval_spots, main_cfg, return_path=False)

    rows = []
    for name, payload in (("Deep Hedging", {"pnl": pnl}), *benches.items()):
        p = payload["pnl"]
        rows.append(
            (
                name,
                f"{float(p.mean()):8.3f}",
                f"{float(p.std(unbiased=True)):8.3f}",
                f"{float(empirical_cvar(p, 0.95)):8.3f}",
                f"{float(empirical_cvar(p, 0.99)):8.3f}",
            )
        )
    print()
    print("Heston main configuration at one percent cost")
    print(format_table(("Strategy", "Mean PnL", "Std", "CVaR 95", "CVaR 99"), rows))

    results = Path("results")
    figures = [
        plot_heston_cost_sweep(records, results),
        plot_heston_diagnostics(eval_spots, eval_var, deltas, results),
    ]
    print()
    print("Saved Heston figures")
    for figure in figures:
        print(f"  {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
