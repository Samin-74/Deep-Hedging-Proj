import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .benchmarks import BenchmarkRunner
from .loss import EntropicRiskLoss, empirical_cvar
from .portfolio import hedge_pnl

SEEDS = (42, 43, 44, 45, 46)
STRATEGIES = ("Deep Hedging", "Black-Scholes", "Leland")
METRICS = ("mean_pnl", "std_pnl", "cvar_95", "cvar_99", "entropic", "mean_cost")
METRIC_LABELS = {
    "mean_pnl": "Mean PnL",
    "std_pnl": "Std",
    "cvar_95": "CVaR 95",
    "cvar_99": "CVaR 99",
    "entropic": "Entropic",
    "mean_cost": "Mean cost",
}


def strategy_metrics(
    pnl: torch.Tensor,
    costs: torch.Tensor,
    lam: float,
) -> dict[str, float]:
    # Six headline metrics for one strategy on one path set
    if pnl.shape != costs.shape:
        raise ValueError("pnl and costs must share shape")
    if pnl.numel() < 2:
        raise ValueError("at least two paths are required")
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(unbiased=True)),
        "cvar_95": float(empirical_cvar(pnl, 0.95)),
        "cvar_99": float(empirical_cvar(pnl, 0.99)),
        "entropic": float(EntropicRiskLoss(lam)(pnl)),
        "mean_cost": float(costs.mean()),
    }


def run_single_seed(
    config,
    seed: int,
    simulate: Callable,
    train: Callable,
    evaluate: Callable,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    # One full train and evaluate cycle at a given seed
    started = time.perf_counter()
    train_spots, train_var = simulate(config, config.n_paths, seed=seed)
    eval_spots, eval_var = simulate(config, config.n_eval_paths, seed=seed + 1)
    model, history, _ = train(config, train_spots, train_var, seed=seed)
    deep_pnl, _ = evaluate(model, eval_spots, eval_var, config, return_path=False)

    runner = BenchmarkRunner(
        spots=eval_spots,
        K=config.K,
        r=config.r,
        sigma=config.benchmark_sigma,
        cost_rate=config.cost_rate,
        dt=config.dt,
        n_steps=config.n_steps,
    )
    with torch.no_grad():
        deep_deltas = model(eval_spots, eval_var)
    deep_costs = hedge_pnl(eval_spots, deep_deltas, config.K, config.cost_rate)["costs"]
    bs = runner.run_bs()
    leland = runner.run_leland()

    result = {
        "Deep Hedging": strategy_metrics(deep_pnl, deep_costs, config.lambda_entropic),
        "Black-Scholes": strategy_metrics(bs["pnl"], bs["costs"], config.lambda_entropic),
        "Leland": strategy_metrics(leland["pnl"], leland["costs"], config.lambda_entropic),
    }
    if verbose:
        print(
            f"  seed {seed}: deep entropic {result['Deep Hedging']['entropic']:7.4f}  "
            f"cvar95 {result['Deep Hedging']['cvar_95']:7.4f}  "
            f"final train risk {history[-1]:7.4f}  "
            f"{time.perf_counter() - started:6.1f}s",
            flush=True,
        )
    return result


def aggregate(
    per_seed: Sequence[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    # Mean and sample standard deviation across seeds
    if not per_seed:
        raise ValueError("per_seed must not be empty")
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for strategy in STRATEGIES:
        summary[strategy] = {}
        for metric in METRICS:
            values = [float(run[strategy][metric]) for run in per_seed]
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                var = sum((v - mean) ** 2 for v in values) / (n - 1)
                std = math.sqrt(var)
            else:
                std = float("nan")
            summary[strategy][metric] = {"mean": mean, "std": std, "n": n}
    return summary


def format_pm(mean: float, std: float, digits: int = 3) -> str:
    # ASCII mean plus or minus standard deviation
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def summary_rows(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    digits: int = 3,
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for strategy in STRATEGIES:
        cells = [strategy]
        for metric in METRICS:
            entry = summary[strategy][metric]
            cells.append(format_pm(entry["mean"], entry["std"], digits))
        rows.append(tuple(cells))
    return rows


def print_summary(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    seeds: Sequence[int],
    format_table: Callable,
    digits: int = 3,
) -> None:
    headers = ("Strategy",) + tuple(METRIC_LABELS[m] for m in METRICS)
    print()
    print(f"Multi-seed study: mean +/- standard deviation over {len(seeds)} seeds")
    print(f"seeds {', '.join(str(s) for s in seeds)}")
    print(format_table(headers, summary_rows(summary, digits)))


def save_summary(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    per_seed: Sequence[Mapping[str, Mapping[str, float]]],
    seeds: Sequence[int],
    path: str | Path,
    config=None,
) -> Path:
    # Persist the study so the write-up can be regenerated
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": list(seeds),
        "metrics": list(METRICS),
        "summary": {
            strategy: {
                metric: dict(summary[strategy][metric]) for metric in METRICS
            }
            for strategy in STRATEGIES
        },
        "per_seed": [
            {strategy: dict(run[strategy]) for strategy in STRATEGIES}
            for run in per_seed
        ],
    }
    if config is not None:
        # Run metadata guards against quoting a reduced budget by mistake
        payload["config"] = {
            "simulator": config.simulator,
            "loss_type": config.loss_type,
            "cost_rate": float(config.cost_rate),
            "n_paths": int(config.n_paths),
            "n_eval_paths": int(config.n_eval_paths),
            "epochs": int(config.epochs),
            "lambda_entropic": float(config.lambda_entropic),
            "norm_type": config.norm_type,
            "time_dependent": bool(config.time_dependent),
        }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def markdown_table(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    digits: int = 3,
    plus_minus: str = "\u00b1",
) -> str:
    # Markdown rendering, the separator is configurable for console safety
    headers = ["Strategy"] + [METRIC_LABELS[m] for m in METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for strategy in STRATEGIES:
        cells = [strategy]
        for metric in METRICS:
            entry = summary[strategy][metric]
            cells.append(
                f"{entry['mean']:.{digits}f} {plus_minus} {entry['std']:.{digits}f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_study(
    config,
    seeds: Sequence[int],
    simulate: Callable,
    train: Callable,
    evaluate: Callable,
    format_table: Callable,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    # Full multi-seed driver
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    started = time.perf_counter()
    per_seed: list[dict[str, dict[str, float]]] = []
    for seed in seeds:
        per_seed.append(
            run_single_seed(config, int(seed), simulate, train, evaluate, verbose=verbose)
        )
    summary = aggregate(per_seed)

    print_summary(summary, seeds, format_table)
    if output_path is not None:
        saved = save_summary(summary, per_seed, seeds, output_path, config)
        print(f"multi-seed summary written to {saved}")
    print(f"multi-seed study finished in {time.perf_counter() - started:.1f}s")
    return {"summary": summary, "per_seed": per_seed, "seeds": list(seeds)}
