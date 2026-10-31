import json
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import seed_everything, train_policy
from src.simulators import simulate_paths
from src.visualize import plot_heston_cost_sweep, plot_heston_diagnostics, plot_heston_policy


def main() -> int:
    # Redraw the three Heston figures from one seed 42 training
    config = replace(Config(), simulator="heston", n_paths=100_000, epochs=20, cost_rate=0.01)
    config.validate()
    seed_everything(config.seed, deterministic=config.deterministic)

    train_spots, train_var = simulate_paths(config, config.n_paths, seed=config.seed)
    eval_spots, eval_var = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    model, _, _ = train_policy(config, train_spots, train_var, seed=config.seed)
    model.eval()
    with torch.no_grad():
        deltas = model(eval_spots, eval_var)

    results = Path("results")
    payload = json.loads((results / "heston_multiseed.json").read_text(encoding="utf-8"))
    figures = [
        plot_heston_cost_sweep(payload, results),
        plot_heston_policy(model, config, eval_spots, eval_var, results),
        plot_heston_diagnostics(eval_spots, eval_var, deltas, results),
    ]
    for figure in figures:
        print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
