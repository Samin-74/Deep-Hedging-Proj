import math
from typing import Mapping, Sequence


def aggregate(
    records: Sequence[Mapping[str, float]],
    keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    # Mean and sample standard deviation across seeds for each key
    if not records:
        raise ValueError("records must not be empty")
    if not keys:
        raise ValueError("keys must not be empty")
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(record[key]) for record in records]
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(var)
        else:
            std = float("nan")
        summary[key] = {"mean": mean, "std": std, "n": n}
    return summary


def format_pm(mean: float, std: float, digits: int = 3) -> str:
    # ASCII mean plus or minus standard deviation
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def markdown_pm(mean: float, std: float, digits: int = 3) -> str:
    # Markdown rendering with the proper plus or minus sign
    return f"{mean:.{digits}f} \u00b1 {std:.{digits}f}"


def significance(
    summary: Mapping[str, Mapping[str, float]],
    key: str,
    reference: str,
    baseline: str,
) -> dict[str, float]:
    # Gap against a baseline with its standard error and t ratio
    ref = summary[reference][key]
    base = summary[baseline][key]
    gap = base["mean"] - ref["mean"]
    se = ref["std"] / math.sqrt(max(ref["n"], 1))
    return {
        "gap": gap,
        "se": se,
        "t": abs(gap) / se if se > 0.0 else float("inf"),
    }
