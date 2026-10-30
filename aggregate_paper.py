import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.heston import HestonParams, heston_call_price


def load_records(results: Path, pattern: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted(results.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend(payload)
        else:
            records.append(payload)
        print(f"  loaded {path.name}  {len(payload)} entries")
    return records


def pm(values: list[float]) -> tuple[float, float]:
    array = np.array(values, dtype=float)
    std = float(array.std(ddof=1)) if array.size > 1 else float("nan")
    return float(array.mean()), std


def print_r1(results: Path) -> None:
    records = load_records(results, "raw_r1_shard*.json")
    if not records:
        return
    print()
    print("R1 price asymptotics, full budget")
    header = f"{'cost':>9} {'n':>3} {'p_eps':>16} {'p - q':>9} {'CVaR95':>9} {'CVaR99':>9} {'cost':>7}"
    print(header)
    print("-" * len(header))
    costs = sorted({float(r["cost"]) for r in records})
    for cost in costs:
        subset = [r for r in records if abs(float(r["cost"]) - cost) < 1e-12]
        ent = pm([float(r["entropic"]) for r in subset])
        c95 = pm([float(r["cvar_95"]) for r in subset])
        c99 = pm([float(r["cvar_99"]) for r in subset])
        mc = pm([float(r["mean_cost"]) for r in subset])
        q = float(subset[0].get("q", math.nan))
        print(
            f"{cost:9.6f} {len(subset):3d} {ent[0]:8.4f}+/-{ent[1]:6.4f} "
            f"{ent[0] - q if math.isfinite(q) else float('nan'):9.4f} "
            f"{c95[0]:9.4f} {c99[0]:9.4f} {mc[0]:7.4f}"
        )


def print_r2(results: Path) -> None:
    target = results / "heston_nocost.json"
    if not target.exists():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    print()
    print("R2 Heston no cost, two instruments")
    print(f"  q = {payload['q']:.4f}   paper 1.69")
    model = payload.get("model_hedge", {})
    if model:
        print(
            f"  model hedge  CVaR0.50 {model.get('cvar_0.50', float('nan')):7.4f}  "
            f"CVaR0.99 {model.get('cvar_0.99', float('nan')):7.4f}"
        )
    for key, block in payload.get("deep", {}).items():
        alpha = float(key)
        value = block.get(f"cvar_{alpha:.2f}", float("nan"))
        paper = 1.94 if alpha < 0.75 else 3.49
        print(
            f"  deep alpha {key}  CVaR {value:7.4f}  paper {paper:4.2f}  "
            f"gap {value - paper:+.4f}  n {block.get('n', 0)}"
        )


def print_r3(results: Path) -> None:
    target = results / "recurrent_ablation.json"
    if not target.exists():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    print()
    print("R3 recurrent against simple")
    for key, block in payload["summary"].items():
        for name in ("recurrent", "simple"):
            if name not in block:
                continue
            entry = block[name]
            print(
                f"  scenario {key} {name:<10} risk {entry['risk']['mean']:7.4f}+/-"
                f"{entry['risk']['std']:6.4f}  cost {entry['mean_cost']['mean']:6.4f}"
            )


def print_r5(results: Path) -> None:
    target = results / "price_asymptotics_heston.json"
    if not target.exists():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    print()
    print("R5 Heston price asymptotics")
    print(f"  q = {payload['q']:.4f}   p_0 = {payload['p_0']:.4f}")
    for key, block in payload["summary"].items():
        print(
            f"  cost {float(key):9.6f}  p_eps {block['entropic']['mean']:8.4f}"
            f"+/-{block['entropic']['std']:6.4f}"
        )
    fit = payload.get("fit", {})
    if fit:
        print(f"  free slope against q {fit['free_vs_q']['slope']:+.4f}")


def print_r6(results: Path) -> None:
    target = results / "paper_architecture.json"
    if not target.exists():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    print()
    print("R6 architecture comparison")
    for key, block in payload["summary"].items():
        print(
            f"  {key:<9} mean {block['mean']['mean']:7.4f}+/-{block['mean']['std']:6.4f}  "
            f"CVaR95 {block['cvar_95']['mean']:7.4f}+/-{block['cvar_95']['std']:6.4f}  "
            f"params {int(block['n_params']['mean']):,}"
        )


def print_r4(results: Path) -> None:
    target = results / "call_spread.json"
    if not target.exists():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    print()
    print("R4 call spread")
    print(f"  strikes {payload['strikes']}  model price {payload['q_spread']:.4f}")
    for key, block in payload["summary"].items():
        print(f"  alpha {key}  CVaR {block['cvar']['mean']:7.4f}+/-{block['cvar']['std']:6.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate paper replication outputs")
    parser.add_argument("--results-dir", default="results/paper")
    args = parser.parse_args()
    results = Path(__file__).resolve().parent / args.results_dir
    print("=" * 78)
    print(f"Aggregating {results}")
    print("=" * 78)
    print_r1(results)
    print_r2(results)
    print_r3(results)
    print_r4(results)
    print_r5(results)
    print_r6(results)
    params = HestonParams()
    print()
    print(f"Heston reference price q = {float(heston_call_price(30.0 / 365.0, 100.0, params.v0, 100.0, params)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
