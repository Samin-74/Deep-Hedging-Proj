import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import format_table, print_cost_study, run_cost_study, seed_everything


def main() -> int:
    # Heston transaction cost sweep, reuses the existing cost study
    paths = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
