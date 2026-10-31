import json
import sys
from pathlib import Path

from src.studies import aggregate


def merge(raw_paths, metrics, key_order=None):
    # Shards hold disjoint seeds per key, so records concatenate and summaries recompute
    merged = {}
    for raw in raw_paths:
        payload = json.loads(Path(raw).read_text(encoding="utf-8"))
        for key, block in payload.items():
            slot = merged.setdefault(key, {"records": []})
            slot["records"].extend(block["records"])
    for key, block in merged.items():
        block["records"].sort(key=lambda r: int(r["seed"]))
        block["summary"] = aggregate(block["records"], metrics)
    if key_order is not None:
        merged = {k: merged[k] for k in key_order if k in merged}
    return merged


def main() -> int:
    kind = sys.argv[1]
    output = sys.argv[2]
    raws = sys.argv[3:]

    if kind == "lambda":
        metrics = ("mean_pnl", "std_pnl", "cvar_95", "cvar_99", "entropic", "mean_cost", "turnover")
        merged = merge(raws, metrics)
        order = sorted(merged, key=float)
        merged = {k: merged[k] for k in order}
        payload = {
            "seeds": [42, 43, 44, 45, 46],
            "paths": 100_000,
            "epochs": 20,
            "lambdas": merged,
        }
    elif kind == "heston":
        metrics = (
            "deep_mean",
            "deep_cvar_95",
            "deep_cvar_99",
            "bs_cvar_95",
            "bs_cvar_99",
            "leland_cvar_95",
            "leland_cvar_99",
        )
        merged = merge(raws, metrics)
        order = sorted(merged, key=float)
        merged = {k: merged[k] for k in order}
        payload = {
            "seeds": [42, 43, 44, 45, 46],
            "paths": 100_000,
            "epochs": 20,
            "levels": merged,
        }
    elif kind == "gbm_cost":
        metrics = (
            "deep_mean",
            "deep_cvar_95",
            "deep_cvar_99",
            "bs_cvar_95",
            "bs_cvar_99",
            "leland_cvar_95",
            "leland_cvar_99",
        )
        records = []
        for raw in raws:
            records.extend(json.loads(Path(raw).read_text(encoding="utf-8")))
        merged = {}
        for cost in sorted({float(r["cost"]) for r in records}):
            subset = [r for r in records if float(r["cost"]) == cost]
            subset.sort(key=lambda r: int(r["seed"]))
            merged[f"{cost:.4f}"] = {
                "records": subset,
                "summary": aggregate(subset, metrics),
            }
        payload = {
            "seeds": [42, 43, 44, 45, 46],
            "paths": 100_000,
            "epochs": 20,
            "levels": merged,
        }
    elif kind == "shared":
        metrics = ("entropic", "cvar_95", "cvar_99", "mean_pnl", "mean_cost")
        records = []
        for raw in raws:
            records.extend(json.loads(Path(raw).read_text(encoding="utf-8")))
        records.sort(key=lambda r: (r["variant"], int(r["seed"])))
        merged = {}
        for variant in ("shared", "time_dependent"):
            subset = [r for r in records if r["variant"] == variant]
            if not subset:
                continue
            merged[variant] = {
                "records": subset,
                "summary": aggregate(subset, metrics),
                "n_params": float(subset[0]["n_params"]),
            }
        payload = {"seeds": [42, 43, 44, 45, 46], "variants": merged}
    else:
        raise SystemExit(f"unknown kind {kind}")

    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"merged {len(raws)} shards into {output}")
    for key, block in (payload.get("lambdas") or payload.get("levels") or payload.get("variants")).items():
        n = len(block["records"])
        print(f"  {key}: {n} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
