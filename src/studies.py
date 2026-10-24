import math
from typing import Mapping, Sequence


def _pchip_slopes(x: Sequence[float], y: Sequence[float]) -> list[float]:
    # Fritsch and Carlson slopes, monotone by construction
    n = len(x)
    if n < 2:
        raise ValueError("at least two points are required")
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    if any(step <= 0.0 for step in h):
        raise ValueError("x must be strictly increasing")
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [delta[0], delta[0]]
    d = [0.0] * n
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
        slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if slope * d0 <= 0.0:
            return 0.0
        if d0 * d1 <= 0.0 and abs(slope) > abs(3.0 * d0):
            return 3.0 * d0
        return slope

    d[0] = endpoint(h[0], h[1], delta[0], delta[1])
    d[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return d


def pchip(x: Sequence[float], y: Sequence[float], grid: Sequence[float]) -> list[float]:
    # Monotone piecewise cubic Hermite interpolation, numpy free
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    d = _pchip_slopes(x, y)
    out: list[float] = []
    n = len(x)
    for value in grid:
        if value <= x[0]:
            out.append(float(y[0]))
            continue
        if value >= x[-1]:
            out.append(float(y[-1]))
            continue
        index = n - 2
        for i in range(n - 1):
            if x[i] <= value <= x[i + 1]:
                index = i
                break
        h = x[index + 1] - x[index]
        s = (value - x[index]) / h
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        out.append(
            h00 * y[index]
            + h10 * h * d[index]
            + h01 * y[index + 1]
            + h11 * h * d[index + 1]
        )
    return out


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
