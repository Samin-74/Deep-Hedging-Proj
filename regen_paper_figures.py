import argparse
import json
import math
from pathlib import Path

import numpy as np

from config import Config
from src.heston import HestonParams, heston_call_price
from src.visualize import (
    plot_call_spread,
    plot_cost_sensitivity_5seed,
    plot_eda_market,
    plot_eda_moments,
    plot_eda_results,
    plot_heston_asymptotics,
    plot_heston_fig2,
    plot_heston_fig6,
    plot_heston_fig7,
    plot_price_asymptotics,
    plot_recurrent_ablation,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def regen_r1(results: Path) -> None:
    target = results / "price_asymptotics_bs.json"
    if not target.exists():
        print("  missing price_asymptotics_bs.json")
        return
    payload = load(target)
    print(" ", plot_price_asymptotics(payload, results))
    print(" ", plot_cost_sensitivity_5seed(payload, results))


def regen_r2(results: Path) -> None:
    target = results / "heston_nocost.json"
    if not target.exists():
        print("  missing heston_nocost.json")
        return
    payload = load(target)
    samples = {
        float(key): [np.asarray(item, dtype=np.float64) for item in value]
        for key, value in payload.get("samples", {}).items()
    }
    model_sample = np.asarray(payload.get("model_sample", []), dtype=np.float64)
    q = float(payload["q"])
    if samples and model_sample.size:
        main = sorted(samples)[0]
        print(" ", plot_heston_fig2(samples[main], model_sample, q,
                                    payload["deep"][f"{main:.2f}"], results))
        print(" ", plot_heston_fig6(samples, payload["deep"], q, results))
        print(" ", plot_heston_fig7(samples, model_sample, payload["deep"], q, results))


def regen_r3(results: Path) -> None:
    target = results / "recurrent_ablation.json"
    if not target.exists():
        print("  missing recurrent_ablation.json")
        return
    payload = load(target)
    print(" ", plot_recurrent_ablation(payload["records"], payload["scenarios"], results))


def regen_r4(results: Path) -> None:
    target = results / "call_spread.json"
    if not target.exists():
        print("  missing call_spread.json")
        return
    print(" ", plot_call_spread(load(target), results))


def regen_r5(results: Path) -> None:
    target = results / "price_asymptotics_heston.json"
    if not target.exists():
        print("  missing price_asymptotics_heston.json")
        return
    print(" ", plot_heston_asymptotics(load(target), results))


def regen_eda(results: Path) -> None:
    target = results / "eda.json"
    if not target.exists():
        print("  missing eda.json")
        return
    payload = load(target)
    print(" ", plot_eda_market(payload, results))
    print(" ", plot_eda_moments(payload, results))
    if payload.get("paper"):
        print(" ", plot_eda_results(payload, results))


def main() -> int:
    parser = argparse.ArgumentParser(description="Redraw paper figures from stored JSON")
    parser.add_argument("--results-dir", default="results/paper")
    args = parser.parse_args()
    results = Path(__file__).resolve().parent / args.results_dir
    if not results.exists():
        raise SystemExit(f"missing {results}")
    print(f"Redrawing figures from {results}")
    for name, fn in (
        ("R1 price asymptotics", regen_r1),
        ("R2 Heston no cost", regen_r2),
        ("R3 recurrent ablation", regen_r3),
        ("R4 call spread", regen_r4),
        ("R5 Heston asymptotics", regen_r5),
        ("EDA", regen_eda),
    ):
        print(f"{name}:")
        fn(results)
    params = HestonParams()
    print(
        f"\nreference prices: Black-Scholes q = 2.2872, "
        f"Heston q = {float(heston_call_price(Config().T, 100.0, params.v0, 100.0, params)):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
