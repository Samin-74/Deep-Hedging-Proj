import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .benchmarks import bs_delta, leland_delta
from .loss import empirical_cvar
from .studies import pchip

DPI = 220
COLORS = {
    "Deep Hedging": "#2b6cb0",
    "Black-Scholes": "#c53030",
    "Leland": "#2f855a",
}
CURVE_COLORS = ("#2b6cb0", "#dd6b20", "#805ad5", "#2f855a", "#b83280")
TEXT_COLOR = "#1a202c"
MUTED_COLOR = "#4a5568"
FALLBACK_COLOR = "#4a5568"


def _style(ax: Axes) -> None:
    # Shared axis styling
    ax.grid(True, which="major", color="#a0aec0", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#718096")
    ax.tick_params(colors=MUTED_COLOR, labelsize=9.5)
    ax.xaxis.label.set_color(MUTED_COLOR)
    ax.yaxis.label.set_color(MUTED_COLOR)
    ax.title.set_color(TEXT_COLOR)


def _output(results_path: str | Path, name: str) -> Path:
    directory = Path(results_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _save(fig: Figure, path: Path) -> Path:
    # High resolution export
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_pnl_distributions(
    pnls: Mapping[str, torch.Tensor],
    results_path: str | Path,
    bins: int = 160,
    alpha: float = 0.95,
) -> Path:
    # Terminal PnL histograms for all strategies
    if not pnls:
        raise ValueError("pnls must not be empty")
    if bins < 10:
        raise ValueError("bins must be at least 10")
    path = _output(results_path, "pnl_distributions.png")
    cleaned = {
        name: torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
        for name, values in pnls.items()
    }
    stacked = torch.cat(list(cleaned.values()))
    low = float(torch.quantile(stacked, 0.002))
    high = float(torch.quantile(stacked, 0.998))
    pad = 0.06 * max(high - low, 1e-6)
    edges = torch.linspace(low - pad, high + pad, bins + 1).numpy()

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    for name, values in cleaned.items():
        color = COLORS.get(name, FALLBACK_COLOR)
        mean = float(values.mean())
        tail = float(empirical_cvar(values, alpha))
        label = f"{name}: mean {mean:.2f}, CVaR {int(alpha * 100)} {tail:.2f}"
        ax.hist(
            values.numpy(),
            bins=edges,
            density=True,
            histtype="stepfilled",
            color=color,
            alpha=0.32,
            edgecolor=color,
            linewidth=1.3,
            label=label,
        )
        ax.axvline(mean, color=color, linestyle="--", linewidth=1.0, alpha=0.9)

    ax.axvline(0.0, color="#2d3748", linewidth=0.8, alpha=0.5)
    ax.set_xlim(low - pad, high + pad)
    ax.set_xlabel("Terminal profit and loss")
    ax.set_ylabel("Probability density")
    ax.set_title("Terminal PnL distribution under proportional transaction costs")
    ax.legend(frameon=False, fontsize=9.0, loc="upper left")
    ax.text(
        0.99,
        0.02,
        f"{next(iter(cleaned.values())).numel():,} evaluation paths",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=MUTED_COLOR,
    )
    _style(ax)
    return _save(fig, path)


def plot_policy_curves(
    model,
    config,
    results_path: str | Path,
    step: int | None = None,
    prev_deltas: Sequence[float] = (-0.4, 0.0, 0.4),
    spot_grid: int = 401,
    span: float = 0.3,
    no_trade_tol: float = 0.01,
    eval_spots: torch.Tensor | None = None,
    eval_variances: torch.Tensor | None = None,
    n_bins: int = 14,
) -> Path:
    # Learned hedge position, on-path response and no-transaction region
    path = _output(results_path, "policy_curves.png")
    step = model.n_steps // 2 if step is None else int(step)
    if not 0 <= step < model.n_steps:
        raise ValueError("step must lie in [0, n_steps)")
    if not prev_deltas:
        raise ValueError("prev_deltas must not be empty")
    model.eval()
    spots = torch.linspace(config.K * (1.0 - span), config.K * (1.0 + span), spot_grid)
    tau = (config.n_steps - step) * config.dt
    with torch.no_grad():
        curves = [model.policy_at_step(spots, step, float(prev)) for prev in prev_deltas]
    reference_bs = bs_delta(spots, config.K, config.r, config.benchmark_sigma, tau)
    reference_leland = leland_delta(
        spots,
        config.K,
        config.r,
        config.benchmark_sigma,
        tau,
        config.cost_rate,
        config.dt,
    )

    x = spots.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
    on_path, probe, trade = axes[0], axes[1], axes[2]

    if eval_spots is not None:
        sample = eval_spots[: min(eval_spots.shape[0], 40_000), : step + 2]
        with torch.no_grad():
            path_deltas = model(
                sample,
                None if eval_variances is None else eval_variances[: sample.shape[0]],
                n_steps=step + 1,
            )
        spot_now = sample[:, step].cpu()
        delta_now = path_deltas[:, step].cpu()
        edges = torch.linspace(
            float(torch.quantile(spot_now, 0.02)),
            float(torch.quantile(spot_now, 0.98)),
            n_bins + 1,
        )
        centers, means, lows, highs = [], [], [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (spot_now >= low) & (spot_now < high)
            if int(mask.sum()) < 30:
                continue
            values = delta_now[mask]
            centers.append(float(spot_now[mask].mean()))
            means.append(float(values.mean()))
            lows.append(float(torch.quantile(values, 0.1)))
            highs.append(float(torch.quantile(values, 0.9)))
        on_path.fill_between(centers, lows, highs, color=COLORS["Deep Hedging"], alpha=0.2, label="10 to 90 percent band")
        on_path.plot(centers, means, color=COLORS["Deep Hedging"], linewidth=2.2, marker="o", markersize=4.5, label="Deep Hedging mean")
        grid = torch.as_tensor(centers)
        on_path.plot(
            centers,
            bs_delta(grid, config.K, config.r, config.benchmark_sigma, tau).numpy(),
            color=TEXT_COLOR,
            linestyle="--",
            linewidth=1.3,
            label="Black-Scholes delta",
        )
        on_path.plot(
            centers,
            leland_delta(
                grid, config.K, config.r, config.benchmark_sigma, tau, config.cost_rate, config.dt
            ).numpy(),
            color="#718096",
            linestyle=":",
            linewidth=1.7,
            label="Leland delta",
        )
        on_path.set_title(f"On-path hedge position at step {step} of {model.n_steps}")
        on_path.legend(frameon=False, fontsize=8.2, loc="upper left")
    else:
        on_path.axis("off")
        on_path.text(0.5, 0.5, "No evaluation paths supplied", ha="center", va="center", fontsize=10, color=MUTED_COLOR)

    for color, prev, curve in zip(CURVE_COLORS, prev_deltas, curves):
        probe.plot(
            x,
            curve.numpy(),
            color=color,
            linewidth=1.9,
            label=f"Previous delta {prev:.2f}",
        )
    probe.plot(x, reference_bs.numpy(), color=TEXT_COLOR, linestyle="--", linewidth=1.2, label="Black-Scholes delta")
    probe.plot(x, reference_leland.numpy(), color="#718096", linestyle=":", linewidth=1.7, label="Leland delta")
    probe.set_title("Off-path probe across previous positions")
    probe.legend(frameon=False, fontsize=7.8, loc="upper left")

    trade.axhspan(-no_trade_tol, no_trade_tol, color="#cbd5e0", alpha=0.55, zorder=0)
    for color, prev, curve in zip(CURVE_COLORS, prev_deltas, curves):
        trade.plot(
            x,
            (curve - float(prev)).numpy(),
            color=color,
            linewidth=1.9,
            label=f"Previous delta {prev:.2f}",
        )
    trade.axhline(0.0, color=TEXT_COLOR, linewidth=0.9)
    trade.set_title(f"No-trade band at tolerance {no_trade_tol:.2f}")
    trade.set_ylabel("Trade size")
    trade.legend(frameon=False, fontsize=8.2, loc="upper left")

    for ax in axes:
        ax.set_xlabel("Spot price at the decision date")
        _style(ax)
    on_path.set_ylabel("Hedge position")
    fig.tight_layout()
    # Sidecar so the no-trade band and probe claims can be checked without retraining
    sidecar = {
        "step": int(step),
        "spot_grid": x.tolist(),
        "prev_deltas": [float(p) for p in prev_deltas],
        "curves": [c.numpy().tolist() for c in curves],
        "no_trade_tol": float(no_trade_tol),
        "bs_delta": reference_bs.numpy().tolist(),
        "leland_delta": reference_leland.numpy().tolist(),
    }
    Path(_output(results_path, "policy_curves.json")).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return _save(fig, path)


def plot_cost_sensitivity(
    records: Sequence[Mapping[str, float]],
    results_path: str | Path,
) -> Path:
    # CVaR of each strategy across proportional cost levels
    if not records:
        raise ValueError("records must not be empty")
    required = {
        "cost",
        "deep_cvar_95",
        "deep_cvar_99",
        "bs_cvar_95",
        "bs_cvar_99",
        "leland_cvar_95",
        "leland_cvar_99",
    }
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"record is missing keys: {sorted(missing)}")

    path = _output(results_path, "cost_sensitivity.png")
    costs = [100.0 * float(record["cost"]) for record in records]
    strategies = (
        ("Deep Hedging", "deep"),
        ("Black-Scholes", "bs"),
        ("Leland", "leland"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    for ax, quantile in zip(axes, (95, 99)):
        for label, prefix in strategies:
            values = [float(record[f"{prefix}_cvar_{quantile}"]) for record in records]
            ax.plot(
                costs,
                values,
                marker="o",
                markersize=5.5,
                linewidth=1.9,
                color=COLORS[label],
                label=label,
            )
        ax.set_title(f"CVaR at {quantile} percent confidence")
        ax.set_xlabel("Proportional transaction cost (percent)")
        ax.set_ylabel("CVaR of terminal PnL, loss basis")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9.0, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_learning_curves(
    deep_pnl: torch.Tensor,
    benchmark_pnls: Mapping[str, torch.Tensor],
    results_path: str | Path,
    alphas: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> Path:
    # Loss profile of each strategy across confidence levels
    path = _output(results_path, "learning_curves.png")
    series = {"Deep Hedging": torch.as_tensor(deep_pnl).reshape(-1).cpu()}
    for name, values in benchmark_pnls.items():
        series[name] = torch.as_tensor(values).reshape(-1).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    left, right = axes[0], axes[1]
    for name, values in series.items():
        quantiles = torch.linspace(0.01, 0.99, 99)
        points = torch.quantile(values, quantiles).numpy()
        left.plot(
            points,
            quantiles.numpy(),
            color=COLORS.get(name, FALLBACK_COLOR),
            linewidth=1.9,
            label=name,
        )
    left.axvline(0.0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    left.set_xlabel("Terminal PnL quantile")
    left.set_ylabel("Cumulative probability")
    left.set_title("Empirical PnL distribution function")
    left.legend(frameon=False, fontsize=9.0, loc="upper left")

    for name, values in series.items():
        curves = [float(empirical_cvar(values, float(alpha))) for alpha in alphas]
        right.plot(
            [100.0 * float(alpha) for alpha in alphas],
            curves,
            marker="o",
            markersize=5.5,
            linewidth=1.9,
            color=COLORS.get(name, FALLBACK_COLOR),
            label=name,
        )
    right.set_xlabel("Confidence level (percent)")
    right.set_ylabel("Expected shortfall of terminal PnL")
    right.set_title("Risk profile across confidence levels")
    right.legend(frameon=False, fontsize=9.0, loc="upper left")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
    # Sidecar so the expected shortfall ladder can be checked without retraining
    sidecar = {
        "alphas": [float(a) for a in alphas],
        "expected_shortfall": {
            name: [float(empirical_cvar(values, float(a))) for a in alphas]
            for name, values in series.items()
        },
    }
    Path(_output(results_path, "learning_curves.json")).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return _save(fig, path)


def plot_position_dynamics(
    deep_deltas: torch.Tensor,
    benchmark_deltas: Mapping[str, torch.Tensor],
    spots: torch.Tensor,
    results_path: str | Path,
    n_sample: int = 400,
) -> Path:
    # Delta trajectories and position dispersion through time
    path = _output(results_path, "position_dynamics.png")
    deltas = torch.as_tensor(deep_deltas).detach().cpu()
    spots_cpu = torch.as_tensor(spots).detach().cpu()
    if deltas.dim() != 2:
        raise ValueError("deep_deltas must be a 2D tensor")
    if spots_cpu.shape[0] != deltas.shape[0]:
        raise ValueError("spots and deltas must share the path dimension")
    n_steps = deltas.shape[1]
    steps = np.arange(n_steps)
    sample = torch.randperm(deltas.shape[0])[: min(n_sample, deltas.shape[0])].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    left, right = axes[0], axes[1]
    for row in sample[:120]:
        left.plot(
            steps,
            deltas[row].numpy(),
            color=COLORS["Deep Hedging"],
            alpha=0.14,
            linewidth=0.8,
        )
    for name, values in benchmark_deltas.items():
        curve = torch.as_tensor(values).detach().cpu()
        left.plot(
            steps,
            curve.mean(dim=0).numpy(),
            color=COLORS.get(name, FALLBACK_COLOR),
            linewidth=2.0,
            linestyle="--",
            label=f"{name} mean",
        )
    left.plot(
        steps,
        deltas.mean(dim=0).numpy(),
        color=TEXT_COLOR,
        linewidth=2.1,
        label="Deep Hedging mean",
    )
    left.set_xlabel("Decision step")
    left.set_ylabel("Hedge position")
    left.set_title("Position trajectories on a path sample")
    left.legend(frameon=False, fontsize=8.6, loc="upper left")

    moneyness = spots_cpu[:, :-1] / float(spots_cpu[:, :1].mean()) - 1.0
    scatter_sample = sample[: min(1500, sample.size)]
    scatter = right.scatter(
        moneyness[scatter_sample].reshape(-1).numpy(),
        deltas[scatter_sample].reshape(-1).numpy(),
        c=np.tile(steps, scatter_sample.size),
        cmap="viridis",
        s=6,
        alpha=0.55,
        linewidths=0,
    )
    bar = fig.colorbar(scatter, ax=right, pad=0.02)
    bar.set_label("Decision step")
    bar.ax.tick_params(labelsize=8.5, colors=MUTED_COLOR)
    right.set_xlabel("Spot relative to initial level")
    right.set_ylabel("Hedge position")
    right.set_title("Position against moneyness, colored by time")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
    # Sidecar so the trajectory claims can be checked without retraining
    sidecar = {
        "deep_mean": deltas.mean(dim=0).tolist(),
        "benchmark_mean": {
            name: torch.as_tensor(values).detach().cpu().mean(dim=0).tolist()
            for name, values in benchmark_deltas.items()
        },
    }
    Path(_output(results_path, "position_dynamics.json")).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return _save(fig, path)


def plot_pnl_paths(
    deep_paths: torch.Tensor,
    benchmark_paths: Mapping[str, torch.Tensor],
    results_path: str | Path,
    n_sample: int = 60,
) -> Path:
    # Cumulative PnL paths with the terminal payoff included at maturity
    path = _output(results_path, "pnl_paths.png")
    series = {"Deep Hedging": torch.as_tensor(deep_paths).detach().cpu()}
    for name, values in benchmark_paths.items():
        series[name] = torch.as_tensor(values).detach().cpu()
    steps = np.arange(next(iter(series.values())).shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    left, right = axes[0], axes[1]
    deep = series["Deep Hedging"]
    sample = torch.randperm(deep.shape[0])[: min(n_sample, deep.shape[0])].numpy()
    for row in sample:
        left.plot(steps, deep[row].numpy(), color=COLORS["Deep Hedging"], alpha=0.2, linewidth=0.8)
    left.plot(steps, deep.mean(dim=0).numpy(), color=TEXT_COLOR, linewidth=2.1, label="Mean path")

    for name, values in series.items():
        right.plot(
            steps,
            values.mean(dim=0).numpy(),
            color=COLORS.get(name, FALLBACK_COLOR),
            linewidth=2.0,
            label=f"{name} mean",
        )
        lower = torch.quantile(values, 0.05, dim=0).numpy()
        upper = torch.quantile(values, 0.95, dim=0).numpy()
        right.fill_between(
            steps,
            lower,
            upper,
            color=COLORS.get(name, FALLBACK_COLOR),
            alpha=0.12,
        )
    left.set_title("Cumulative PnL of the learned policy, payoff at maturity")
    right.set_title("Mean and 5 to 95 percent band, payoff at maturity")
    for ax in axes:
        ax.axhline(0.0, color=TEXT_COLOR, linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Decision step")
        _style(ax)
    left.set_ylabel("Cumulative PnL")
    left.legend(frameon=False, fontsize=9.0, loc="lower left")
    right.legend(frameon=False, fontsize=9.0, loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


def plot_lambda_frontier(
    per_lambda: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    lambdas: Sequence[float],
    results_path: str | Path,
) -> Path:
    # Risk return frontier across entropic risk aversion levels
    if not lambdas:
        raise ValueError("lambdas must not be empty")
    path = _output(results_path, "lambda_frontier.png")
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.4))
    left, middle, right = axes[0], axes[1], axes[2]

    def series(key: str) -> tuple[list[float], list[float]]:
        means, stds = [], []
        for lam in lambdas:
            entry = per_lambda[f"{lam:.4f}"]["summary"][key]
            means.append(float(entry["mean"]))
            stds.append(float(entry["std"]))
        return means, stds

    mean_pnl, mean_pnl_sd = series("mean_pnl")
    cvar95, cvar95_sd = series("cvar_95")
    cvar99, cvar99_sd = series("cvar_99")
    cost, cost_sd = series("mean_cost")
    std_pnl, std_pnl_sd = series("std_pnl")

    # Monotone cubic fit in log lambda, a visualization aid only
    log_lambdas = [math.log(float(lam)) for lam in lambdas]
    dense = [log_lambdas[0] + (log_lambdas[-1] - log_lambdas[0]) * i / 200 for i in range(201)]
    fit_pnl = pchip(log_lambdas, mean_pnl, dense)
    fit_cvar95 = pchip(log_lambdas, cvar95, dense)
    fit_cost = pchip(log_lambdas, cost, dense)

    left.plot(
        fit_pnl,
        fit_cvar95,
        color=COLORS["Deep Hedging"],
        linewidth=1.4,
        alpha=0.55,
        linestyle="-",
        label="Fitted frontier",
        zorder=1,
    )
    left.errorbar(
        mean_pnl,
        cvar95,
        xerr=mean_pnl_sd,
        yerr=cvar95_sd,
        color=COLORS["Deep Hedging"],
        marker="o",
        markersize=6,
        linewidth=0,
        elinewidth=1.4,
        capsize=3,
        label="Computed points",
        zorder=3,
    )
    for lam, x, y in zip(lambdas, mean_pnl, cvar95):
        left.annotate(f"{lam:.2f}", xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=TEXT_COLOR)
    left.set_xlabel("Mean PnL")
    left.set_ylabel("CVaR 95, loss basis")
    left.set_title("Risk return frontier, labels are lambda")
    left.legend(frameon=False, fontsize=8.6, loc="lower right")

    middle.plot(
        fit_cost,
        fit_cvar95,
        color=COLORS["Deep Hedging"],
        linewidth=1.4,
        alpha=0.55,
        label="Fitted frontier",
        zorder=1,
    )
    middle.errorbar(
        cost,
        cvar95,
        xerr=cost_sd,
        yerr=cvar95_sd,
        color=COLORS["Deep Hedging"],
        marker="o",
        markersize=6,
        linewidth=0,
        elinewidth=1.4,
        capsize=3,
        label="Computed points",
        zorder=3,
    )
    for lam, x, y in zip(lambdas, cost, cvar95):
        middle.annotate(f"{lam:.2f}", xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=TEXT_COLOR)
    middle.set_xlabel("Mean transaction cost")
    middle.set_ylabel("CVaR 95, loss basis")
    middle.set_title("Cost against tail risk, labels are lambda")
    middle.legend(frameon=False, fontsize=8.6, loc="lower left")

    right.errorbar(
        lambdas,
        std_pnl,
        yerr=std_pnl_sd,
        color=COLORS["Deep Hedging"],
        marker="o",
        markersize=6,
        linewidth=1.9,
        capsize=3,
        label="Std of PnL",
    )
    right.errorbar(
        lambdas,
        cvar99,
        yerr=cvar99_sd,
        color=COLORS["Leland"],
        marker="s",
        markersize=6,
        linewidth=1.9,
        capsize=3,
        label="CVaR 99",
    )
    right.set_xlabel("Entropic risk aversion lambda")
    right.set_ylabel("Risk measure value")
    right.set_title("Dispersion and tail risk against lambda")
    right.legend(frameon=False, fontsize=9.0, loc="upper right")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_heston_policy(
    model,
    config,
    eval_spots: torch.Tensor,
    eval_variances: torch.Tensor,
    results_path: str | Path,
    step: int | None = None,
    vol_levels: Sequence[float] = (0.12, 0.20, 0.30),
    prev_deltas: Sequence[float] = (-0.3, 0.0, 0.3),
    spot_grid: int = 301,
    span: float = 0.25,
    no_trade_tol: float = 0.01,
) -> Path:
    # Heston policy surface: position against spot at fixed variance levels
    path = _output(results_path, "heston_policy_curves.png")
    step = model.n_steps // 2 if step is None else int(step)
    if not 0 <= step < model.n_steps:
        raise ValueError("step must lie in [0, n_steps)")
    if not vol_levels:
        raise ValueError("vol_levels must not be empty")
    model.eval()
    spots = torch.linspace(config.K * (1.0 - span), config.K * (1.0 + span), spot_grid)
    x = spots.numpy()
    tau = (config.n_steps - step) * config.dt
    reference_bs = bs_delta(spots, config.K, config.r, config.benchmark_sigma, tau)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    left, delta_panel = axes[0]
    middle, right = axes[1]

    # Heatmap makes the variance dependence readable where curves overlap
    vol_grid = np.linspace(0.08, 0.36, 57)
    surface = np.zeros((vol_grid.size, spots.numel()))
    for i, vol in enumerate(vol_grid):
        variance = torch.full_like(spots, float(vol) ** 2)
        with torch.no_grad():
            surface[i] = model.policy_at_step(spots, step, 0.0, variance=variance).numpy()
    surface_artist = left.pcolormesh(
        spots.numpy(),
        vol_grid,
        surface,
        cmap="viridis",
        shading="auto",
    )
    bar = fig.colorbar(surface_artist, ax=left, pad=0.02)
    bar.set_label("Hedge position")
    bar.ax.tick_params(labelsize=8.5, colors=MUTED_COLOR)
    for vol in vol_levels:
        left.axhline(float(vol), color="white", linestyle=":", linewidth=1.2, alpha=0.85)
    left.set_xlabel("Spot price at the decision date")
    left.set_ylabel("Instantaneous volatility")
    left.set_title(f"Heston position surface at step {step}, previous position 0")

    # Difference against the baseline volatility makes the modest effect legible
    baseline_vol = float(vol_levels[len(vol_levels) // 2])
    baseline = torch.full_like(spots, baseline_vol**2)
    with torch.no_grad():
        base_curve = model.policy_at_step(spots, step, 0.0, variance=baseline).numpy()
    for color, vol in zip(CURVE_COLORS, vol_levels):
        variance = torch.full_like(spots, float(vol) ** 2)
        with torch.no_grad():
            curve = model.policy_at_step(spots, step, 0.0, variance=variance).numpy()
        delta_panel.plot(
            x,
            curve - base_curve,
            color=color,
            linewidth=2.0,
            label=f"Volatility {vol:.2f} minus {baseline_vol:.2f}",
        )
    delta_panel.axhline(0.0, color=TEXT_COLOR, linewidth=0.9)
    delta_panel.set_xlabel("Spot price at the decision date")
    delta_panel.set_ylabel("Position difference")
    delta_panel.set_title(f"Variance effect relative to volatility {baseline_vol:.2f}")
    delta_panel.legend(frameon=False, fontsize=8.6, loc="upper right")

    for color, prev in zip(CURVE_COLORS, prev_deltas):
        variance = torch.full_like(spots, float(vol_levels[len(vol_levels) // 2]) ** 2)
        with torch.no_grad():
            curve = model.policy_at_step(spots, step, float(prev), variance=variance)
        middle.plot(
            x,
            curve.numpy(),
            color=color,
            linewidth=2.0,
            label=f"Previous position {prev:.2f}",
        )
    middle.plot(x, reference_bs.numpy(), color=TEXT_COLOR, linestyle="--", linewidth=1.3, label="Black-Scholes delta")
    middle.set_xlabel("Spot price at the decision date")
    middle.set_ylabel("Hedge position")
    middle.set_title(f"Previous position effect at volatility {vol_levels[len(vol_levels) // 2]:.2f}")
    middle.legend(frameon=False, fontsize=8.6, loc="upper left")

    right.axhspan(-no_trade_tol, no_trade_tol, color="#cbd5e0", alpha=0.55, zorder=0)
    trade_curves = {}
    for color, prev in zip(CURVE_COLORS, prev_deltas):
        variance = torch.full_like(spots, float(vol_levels[len(vol_levels) // 2]) ** 2)
        with torch.no_grad():
            curve = model.policy_at_step(spots, step, float(prev), variance=variance)
        trade = (curve - float(prev)).numpy()
        trade_curves[f"{float(prev):.2f}"] = trade.tolist()
        right.plot(
            x,
            trade,
            color=color,
            linewidth=2.0,
            label=f"Previous position {prev:.2f}",
        )
    right.axhline(0.0, color=TEXT_COLOR, linewidth=0.9)
    right.set_xlabel("Spot price at the decision date")
    right.set_ylabel("Trade size")
    right.set_title(f"No-trade band at tolerance {no_trade_tol:.2f}")
    right.legend(frameon=False, fontsize=8.6, loc="upper left")

    for ax in (left, delta_panel, middle, right):
        _style(ax)
    fig.tight_layout()
    # Sidecar so the numeric claims about the surface can be checked without retraining
    sidecar = {
        "step": int(step),
        "spot_grid": x.tolist(),
        "vol_grid": vol_grid.tolist(),
        "surface": surface.tolist(),
        "vol_levels": [float(v) for v in vol_levels],
        "prev_deltas": [float(p) for p in prev_deltas],
        "baseline_vol": baseline_vol,
        "no_trade_tol": float(no_trade_tol),
        "trade_curves": trade_curves,
    }
    Path(_output(results_path, "heston_policy_curves.json")).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return _save(fig, path)


def plot_heston_diagnostics(
    spots: torch.Tensor,
    variances: torch.Tensor,
    deep_deltas: torch.Tensor,
    results_path: str | Path,
    n_sample: int = 400,
) -> Path:
    # Heston variance dynamics and the policy response to the variance state
    path = _output(results_path, "heston_diagnostics.png")
    spots_cpu = torch.as_tensor(spots).detach().cpu()
    var_cpu = torch.as_tensor(variances).detach().cpu()
    deltas = torch.as_tensor(deep_deltas).detach().cpu()
    if spots_cpu.shape != var_cpu.shape:
        raise ValueError("spots and variances must share shape")
    if deltas.shape[0] != spots_cpu.shape[0]:
        raise ValueError("spots and deltas must share the path dimension")
    steps = np.arange(spots_cpu.shape[1])
    sample = torch.randperm(spots_cpu.shape[0])[: min(n_sample, spots_cpu.shape[0])].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.4))
    left, middle, right = axes[0], axes[1], axes[2]

    for row in sample[:150]:
        left.plot(steps, np.sqrt(var_cpu[row].numpy()), color=COLORS["Deep Hedging"], alpha=0.12, linewidth=0.8)
    left.plot(steps, np.sqrt(var_cpu.mean(dim=0).numpy()), color=TEXT_COLOR, linewidth=2.2, label="Mean")
    left.axhline(float(np.sqrt(var_cpu[:, 0].mean())), color="#718096", linestyle=":", linewidth=1.6, label="Initial level")
    left.set_xlabel("Decision step")
    left.set_ylabel("Instantaneous volatility")
    left.set_title("Heston variance paths, square root scale")
    left.legend(frameon=False, fontsize=9.0, loc="upper right")

    vol_now = np.sqrt(var_cpu[:, :-1].numpy()).reshape(-1)
    delta_now = deltas.numpy().reshape(-1)
    keep = np.random.default_rng(0).choice(vol_now.size, size=min(6000, vol_now.size), replace=False)
    scatter = middle.scatter(
        vol_now[keep],
        delta_now[keep],
        c=np.tile(steps[:-1], deltas.shape[0])[keep],
        cmap="viridis",
        s=6,
        alpha=0.5,
        linewidths=0,
    )
    bar = fig.colorbar(scatter, ax=middle, pad=0.02)
    bar.set_label("Decision step")
    bar.ax.tick_params(labelsize=8.5, colors=MUTED_COLOR)
    middle.set_xlabel("Instantaneous volatility")
    middle.set_ylabel("Hedge position")
    middle.set_title("Position against volatility, colored by time")

    moneyness = (spots_cpu[:, :-1] / float(spots_cpu[:, :1].mean()) - 1.0).numpy().reshape(-1)
    right.scatter(
        moneyness[keep],
        delta_now[keep],
        c=vol_now[keep],
        cmap="magma",
        s=6,
        alpha=0.5,
        linewidths=0,
    )
    right.set_xlabel("Spot relative to initial level")
    right.set_ylabel("Hedge position")
    right.set_title("Position against moneyness, colored by volatility")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
    # Sidecar so the volatility and moneyness ordering claims can be checked
    sidecar = {
        "vol_now": vol_now[keep].tolist(),
        "moneyness": moneyness[keep].tolist(),
        "delta_now": delta_now[keep].tolist(),
        "step_now": np.tile(steps[:-1], deltas.shape[0])[keep].tolist(),
    }
    Path(_output(results_path, "heston_diagnostics.json")).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return _save(fig, path)


def plot_heston_cost_sweep(
    payload: Mapping[str, object],
    results_path: str | Path,
) -> Path:
    # Heston CVaR across cost levels, 5 seed means with error bars
    levels = payload["levels"]
    if not levels:
        raise ValueError("levels must not be empty")
    path = _output(results_path, "heston_cost_sensitivity.png")
    costs = sorted(float(k) for k in levels)
    strategies = (
        ("Deep Hedging", "deep", COLORS["Deep Hedging"]),
        ("Black-Scholes", "bs", COLORS["Black-Scholes"]),
        ("Leland", "leland", COLORS["Leland"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    for ax, quantile in zip(axes, (95, 99)):
        for label, prefix, color in strategies:
            field = f"deep_cvar_{quantile}" if prefix == "deep" else f"{prefix}_cvar_{quantile}"
            means = np.array([levels[f"{c:.4f}"]["summary"][field]["mean"] for c in costs])
            stds = np.array([levels[f"{c:.4f}"]["summary"][field]["std"] for c in costs])
            ax.errorbar(np.array(costs) * 100.0, means, yerr=np.nan_to_num(stds, nan=0.0),
                        fmt="o-", color=color, markersize=5.5, capsize=3.0, linewidth=1.9,
                        label=label)
        ax.set_title(f"Heston CVaR at {quantile} percent confidence")
        ax.set_xlabel("Proportional transaction cost (percent)")
        ax.set_ylabel("CVaR of terminal PnL, loss basis")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9.0, loc="upper left")
    fig.suptitle("Heston cost sensitivity: 5 seed mean with one standard deviation",
                 fontsize=11.5, color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_loss(
    history: Sequence[float],
    results_path: str | Path,
    loss_name: str = "Entropic risk",
    window: int = 3,
) -> Path:
    # Training risk measure by epoch with a moving average
    if not history:
        raise ValueError("history must not be empty")
    values = np.asarray(list(history), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain finite values")
    path = _output(results_path, "training_loss.png")
    epochs = np.arange(1, values.size + 1)

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.plot(
        epochs,
        values,
        color=COLORS["Deep Hedging"],
        linewidth=1.1,
        alpha=0.45,
        marker="o",
        markersize=3.2,
        label="Epoch risk",
    )
    if values.size >= window:
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(
            epochs[window - 1 :],
            smooth,
            color=COLORS["Deep Hedging"],
            linewidth=2.3,
            label=f"{window} epoch moving average",
        )
    if values.min() > 0.0 and values.max() / max(values.min(), 1e-12) > 50.0:
        ax.set_yscale("log")
    elif values.min() < 0.0 < values.max():
        ax.axhline(0.0, color=TEXT_COLOR, linewidth=0.8, alpha=0.6)
    ax.annotate(
        f"final {values[-1]:.4f}",
        xy=(epochs[-1], values[-1]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9.0,
        color=TEXT_COLOR,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_name)
    ax.set_title("Training convergence of the hedging policy")
    ax.legend(frameon=False, fontsize=9.0, loc="upper right")
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)

def plot_price_asymptotics(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # Paper Figure 10: log price excess against log cost with a 2/3 reference slope
    summary = payload["summary"]
    costs = sorted(float(k) for k in summary)
    q = float(payload["q"])
    p0 = float(payload["p_0"])
    positive = [c for c in costs if c > 0.0]
    x = np.log(np.array(positive))
    series = {
        "vs q": (np.array([summary[f"{c:.6f}"]["entropic"]["mean"] - q for c in positive]),
                 np.array([summary[f"{c:.6f}"]["entropic"]["std"] for c in positive])),
        "vs p_0": (np.array([summary[f"{c:.6f}"]["entropic"]["mean"] - p0 for c in positive]),
                   np.array([summary[f"{c:.6f}"]["entropic"]["std"] for c in positive])),
    }
    fit = payload["fit"]
    if len(positive) < 2:
        raise ValueError("at least two positive cost levels are required to fit a slope")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True)
    for ax, (label, (excess, err)) in zip(axes, series.items()):
        y = np.log(np.clip(excess, 1e-12, None))
        rel = err / np.clip(excess, 1e-12, None)
        n_seeds = int(summary[f"{positive[0]:.6f}"]["entropic"]["n"])
        ax.errorbar(x, y, yerr=rel, fmt="o", color="#2b6cb0", markersize=6.5,
                    capsize=3.5, linewidth=1.3, label=f"Deep Hedging ({n_seeds} seed mean)")
        grid = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
        key_free = "free_vs_q" if label == "vs q" else "free_vs_p0"
        free = fit[key_free]
        ax.plot(grid, free["intercept"] + free["slope"] * grid, color="#c53030", linewidth=2.0,
                label=f"OLS slope {free['slope']:.3f}")
        key_fixed = "fixed_two_thirds_vs_q" if label == "vs q" else "fixed_two_thirds_vs_p0"
        fixed = fit[key_fixed]
        ax.plot(grid, fixed["intercept"] + (2.0 / 3.0) * grid, color="#2f855a",
                linewidth=1.8, linestyle="--", label="slope fixed at 2/3")
        ax.set_xlabel("log(cost rate)")
        ax.set_ylabel(f"log(p_eps - {'q' if label == 'vs q' else 'p_0'})")
        ax.set_title(f"Price excess {label}")
        ax.legend(frameon=False, fontsize=8.8, loc="upper left")
        _style(ax)
    ci_q = fit["ci_vs_q"]
    ci_p0 = fit["ci_vs_p0"]
    fig.suptitle(
        f"Paper Figure 10 replication: q = {q:.4f}, p_0 = {p0:.4f}, "
        f"slope {fit['free_vs_q']['slope']:.3f} [{ci_q['low']:.3f}, {ci_q['high']:.3f}]",
        fontsize=11.5, color=TEXT_COLOR,
    )
    fig.tight_layout()
    return _save(fig, _output(results_path, "price_asymptotics_bs.png"))


def plot_cost_sensitivity_5seed(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # CVaR against cost for the three strategies, five seed means with error bars
    summary = payload["summary"]
    costs = sorted(float(k) for k in summary)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    strategies = (("Deep Hedging", "deep", COLORS["Deep Hedging"]),
                  ("Black-Scholes", "bs", COLORS["Black-Scholes"]),
                  ("Leland", "leland", COLORS["Leland"]))
    labels = (("CVaR95", "cvar_95"), ("CVaR99", "cvar_99"))
    for ax, (title, key) in zip(axes, labels):
        for name, prefix, color in strategies:
            # Deep Hedging records use bare metric names, benchmarks are prefixed
            field = key if prefix == "deep" else f"{prefix}_{key}"
            means = np.array([summary[f"{c:.6f}"][field]["mean"] for c in costs])
            stds = np.array([summary[f"{c:.6f}"][field]["std"] for c in costs])
            stds = np.nan_to_num(stds, nan=0.0)
            ax.errorbar(np.array(costs) * 100.0, means, yerr=stds, fmt="o-", color=color,
                        markersize=5.5, capsize=3.0, linewidth=1.6, label=name)
        ax.set_xlabel("Proportional cost rate (percent)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} against transaction cost")
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
    fig.suptitle("Paper Section 5.3.1 cost grid: 5 seed mean with one standard deviation",
                 fontsize=11.5, color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "cost_sensitivity_5seed.png"))

def plot_cost_sensitivity_multiseed(
    payload: Mapping[str, object],
    results_path: str | Path,
) -> Path:
    # CVaR against cost for the three strategies, five seed means with error bars
    summary = payload["summary"]
    costs = sorted(float(k) for k in summary)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    strategies = (("Deep Hedging", "deep", COLORS["Deep Hedging"]),
                  ("Black-Scholes", "bs", COLORS["Black-Scholes"]),
                  ("Leland", "leland", COLORS["Leland"]))
    labels = (("CVaR 95", "cvar_95"), ("CVaR 99", "cvar_99"))
    for ax, (title, key) in zip(axes, labels):
        for name, prefix, color in strategies:
            # Deep Hedging records use bare metric names, benchmarks are prefixed
            field = f"deep_{key}" if prefix == "deep" else f"{prefix}_{key}"
            means = np.array([summary[f"{c:.6f}"][field]["mean"] for c in costs])
            stds = np.array([summary[f"{c:.6f}"][field]["std"] for c in costs])
            stds = np.nan_to_num(stds, nan=0.0)
            ax.errorbar(np.array(costs) * 100.0, means, yerr=stds, fmt="o-", color=color,
                        markersize=5.5, capsize=3.0, linewidth=1.6, label=name)
        ax.set_xlabel("Proportional cost rate (percent)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} against transaction cost")
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
    fig.suptitle("GBM cost sensitivity: 5 seed mean with one standard deviation",
                 fontsize=11.5, color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "cost_sensitivity_multiseed.png"))


def _hist_common(ax, values, color, label, bins=140, lo=None, hi=None):
    lo = float(np.percentile(values, 0.1)) if lo is None else lo
    hi = float(np.percentile(values, 99.9)) if hi is None else hi
    ax.hist(values, bins=bins, range=(lo, hi), density=True, color=color, alpha=0.55, label=label)


def plot_heston_fig2(
    deep_pnls: Sequence[np.ndarray],
    model_pnl: np.ndarray,
    q: float,
    deep_block: Mapping[str, float],
    results_path: str | Path,
) -> Path:
    # Paper Figure 2: model hedge against the deep hedge at alpha 0.5
    deep = np.concatenate(list(deep_pnls))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    _hist_common(ax, model_pnl, "#c53030", f"Model hedge (mean {model_pnl.mean():.3f})")
    _hist_common(ax, deep, "#2b6cb0", f"Deep hedge (mean {deep.mean():.3f})")
    ax.axvline(-q, color=TEXT_COLOR, linewidth=1.4, linestyle=":", label=f"-q = {-q:.3f}")
    ax.set_xlabel("Hedging error q - Z + (delta . S)_T")
    ax.set_ylabel("Density")
    ax.set_title("Paper Figure 2: hedging error at alpha 0.5")
    ax.legend(frameon=False, fontsize=8.6)
    _style(ax)
    ax = axes[1]
    # CVaR is the mean of the worst half of the loss, so negate the worst half of the PnL
    alpha = 0.5
    k = max(1, int(np.ceil((1.0 - alpha) * deep.size)))
    deep_cvar = -float(np.sort(deep)[:k].mean())
    k_m = max(1, int(np.ceil((1.0 - alpha) * model_pnl.size)))
    model_cvar = -float(np.sort(model_pnl)[:k_m].mean())
    lo = float(np.percentile(deep, 100.0 * alpha))
    ax.hist(deep[deep <= lo], bins=70, density=True, color="#2b6cb0", alpha=0.85,
            label=f"deep worst half, CVaR {deep_cvar:.3f}")
    lo_m = float(np.percentile(model_pnl, 100.0 * alpha))
    ax.hist(model_pnl[model_pnl <= lo_m], bins=70, density=True, color="#c53030", alpha=0.5,
            label=f"model worst half, CVaR {model_cvar:.3f}")
    ax.set_xlabel("Hedging error, worst half of the loss distribution")
    ax.set_ylabel("Density within the tail")
    ax.set_title("Worst half of the loss distribution")
    ax.legend(frameon=False, fontsize=8.6)
    _style(ax)
    fig.suptitle("Heston, no transaction costs, stock and variance swap", fontsize=11.5,
                 color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "heston_fig2.png"))


def plot_heston_fig6(
    per_seed_pnl: Mapping[float, Sequence[np.ndarray]],
    deep_blocks: Mapping[str, Mapping[str, float]],
    q: float,
    results_path: str | Path,
) -> Path:
    # Paper Figure 6: alpha 0.5 and alpha 0.99 centred at their own risk measure
    alphas = sorted(per_seed_pnl)
    fig, axes = plt.subplots(1, len(alphas), figsize=(6.4 * len(alphas), 5.0), squeeze=False)
    for ax, alpha in zip(axes[0], alphas):
        values = np.concatenate(list(per_seed_pnl[alpha]))
        block = deep_blocks[f"{alpha:.2f}"]
        centre = block[f"cvar_{alpha:.2f}"]
        _hist_common(ax, values, COLORS["Deep Hedging"],
                     f"Deep hedge, CVaR {centre:.3f}")
        ax.axvline(-centre, color=TEXT_COLOR, linewidth=1.5, linestyle="--",
                   label=f"centred at -p = {-centre:.3f}")
        ax.set_xlabel("Hedging error q - Z + (delta . S)_T")
        ax.set_ylabel("Density")
        ax.set_title(f"alpha = {alpha:.2f}   realised CVaR {centre:.3f}")
        ax.legend(frameon=False, fontsize=8.8)
        _style(ax)
    fig.suptitle("Paper Figure 6: risk aversion widens the deep hedge tail", fontsize=11.5,
                 color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "heston_fig6.png"))


def plot_heston_fig7(
    per_seed_pnl: Mapping[float, Sequence[np.ndarray]],
    model_pnl: np.ndarray,
    deep_blocks: Mapping[str, Mapping[str, float]],
    q: float,
    results_path: str | Path,
) -> Path:
    # Paper Figure 7: both policies centred at the model price q
    alphas = sorted(per_seed_pnl)
    fig, axes = plt.subplots(1, len(alphas), figsize=(6.4 * len(alphas), 5.0), squeeze=False)
    for ax, alpha in zip(axes[0], alphas):
        deep = np.concatenate(list(per_seed_pnl[alpha]))
        _hist_common(ax, model_pnl, "#c53030", "Model hedge")
        _hist_common(ax, deep, COLORS["Deep Hedging"], "Deep hedge")
        ax.axvline(-q, color=TEXT_COLOR, linewidth=1.5, linestyle="--", label=f"-q = {-q:.3f}")
        ax.set_xlabel("Hedging error q - Z + (delta . S)_T")
        ax.set_ylabel("Density")
        ax.set_title(
            f"alpha = {alpha:.2f}: deep CVaR {deep_blocks[f'{alpha:.2f}'][f'cvar_{alpha:.2f}']:.3f}"
        )
        ax.legend(frameon=False, fontsize=8.8)
        _style(ax)
    fig.suptitle("Paper Figure 7: both policies centred at the model price", fontsize=11.5,
                 color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "heston_fig7.png"))


def plot_heston_fig3(
    model,
    hedge,
    config,
    spots: np.ndarray,
    variances: np.ndarray,
    results_path: str | Path,
    step: int | None = None,
    dump: bool = True,
) -> Path:
    # Paper Figure 3: stock holding at mid horizon across spot and variance grids
    step = config.n_steps // 2 if step is None else int(step)
    s_grid = np.linspace(95.0, 103.0, 41)
    v_grid = np.linspace(0.03, 0.13, 11)
    s_mesh, v_mesh = np.meshgrid(s_grid, v_grid, indexing="ij")
    model_delta = np.zeros_like(s_mesh)
    with torch.no_grad():
        flat_s = torch.as_tensor(s_mesh.reshape(-1), dtype=torch.float32)
        flat_v = torch.as_tensor(v_mesh.reshape(-1), dtype=torch.float32)
        prev = model.policy_at_step(flat_s, step, 0.0, variance=flat_v) if model is not None else None
        if prev is not None:
            # Two instrument policies expose the stock leg in the first column
            stock = prev[:, 0] if prev.dim() == 2 else prev
            deep_delta = stock.numpy().reshape(s_mesh.shape).astype(np.float64)
        else:
            deep_delta = np.zeros_like(s_mesh)
    for i in range(s_grid.size):
        for j in range(v_grid.size):
            table = hedge.delta_stock[step]
            model_delta[i, j] = hedge._bilinear(
                table, np.array([s_grid[i]]), np.array([v_grid[j]])
            )[0]
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.9))
    panels = (
        (model_delta, "Model hedge stock holding", "viridis"),
        (deep_delta, "Deep hedge stock holding", "viridis"),
        (deep_delta - model_delta, "Deep minus model", "coolwarm"),
    )
    for ax, (values, title, cmap) in zip(axes, panels):
        if title.startswith("Deep minus"):
            lim = float(np.max(np.abs(values))) or 1.0
            image = ax.pcolormesh(s_grid, v_grid, values.T, cmap=cmap, shading="auto",
                                 vmin=-lim, vmax=lim)
        else:
            image = ax.pcolormesh(s_grid, v_grid, values.T, cmap=cmap, shading="auto")
        fig.colorbar(image, ax=ax, label="Stock position")
        ax.set_xlabel("Spot")
        ax.set_ylabel("Variance")
        ax.set_title(f"{title} at step {step}")
        _style(ax)
    fig.suptitle("Paper Figure 3: stock holding on a spot and variance grid", fontsize=11.5,
                 color=TEXT_COLOR)
    fig.tight_layout()
    if dump:
        # Sidecar so the figure can be redrawn without retraining
        sidecar = {
            "step": int(step),
            "s_grid": s_grid.tolist(),
            "v_grid": v_grid.tolist(),
            "model_delta": model_delta.tolist(),
            "deep_delta": deep_delta.tolist(),
        }
        Path(_output(results_path, "heston_fig3.json")).write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )
    return _save(fig, _output(results_path, "heston_fig3.png"))

def plot_heston_fig3_from_file(results_path: str | Path) -> Path | None:
    # Redraw Figure 3 from the sidecar, no retraining
    source = Path(_output(results_path, "heston_fig3.json"))
    if not source.exists():
        return None
    data = json.loads(source.read_text(encoding="utf-8"))
    s_grid = np.asarray(data["s_grid"], dtype=np.float64)
    v_grid = np.asarray(data["v_grid"], dtype=np.float64)
    model_delta = np.asarray(data["model_delta"], dtype=np.float64)
    deep_delta = np.asarray(data["deep_delta"], dtype=np.float64)
    step = int(data["step"])
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.9))
    panels = (
        (model_delta, "Model hedge stock holding", "viridis"),
        (deep_delta, "Deep hedge stock holding", "viridis"),
        (deep_delta - model_delta, "Deep minus model", "coolwarm"),
    )
    for ax, (values, title, cmap) in zip(axes[:3], panels):
        if title.startswith("Deep minus"):
            lim = float(np.max(np.abs(values))) or 1.0
            image = ax.pcolormesh(s_grid, v_grid, values.T, cmap=cmap, shading="auto",
                                  vmin=-lim, vmax=lim)
        else:
            image = ax.pcolormesh(s_grid, v_grid, values.T, cmap=cmap, shading="auto")
        fig.colorbar(image, ax=ax, label="Stock position")
        ax.set_xlabel("Spot")
        ax.set_ylabel("Variance")
        ax.set_title(f"{title} at step {step}")
        _style(ax)
    # Small multiples make the modest variance effect readable on the heatmap scale
    ax = axes[3]
    levels = (0.03, 0.08, 0.13)
    colours = ("#0d6efd", "#d63384", "#fd7e14")
    j_at = [int(np.argmin(np.abs(v_grid - level))) for level in levels]
    for level, colour, j in zip(levels, colours, j_at):
        ax.plot(s_grid, model_delta[:, j], color=colour, linewidth=1.6, linestyle="--",
                label=f"model, v = {v_grid[j]:.2f}")
        ax.plot(s_grid, deep_delta[:, j], color=colour, linewidth=1.8,
                label=f"deep, v = {v_grid[j]:.2f}")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Stock position")
    ax.set_title(f"Variance slices at step {step}")
    ax.legend(fontsize=6.6, frameon=False, ncol=2)
    _style(ax)
    fig.suptitle("Paper Figure 3: stock holding on a spot and variance grid", fontsize=11.5,
                 color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "heston_fig3.png"))


def plot_recurrent_ablation(
    records: Sequence[Mapping[str, object]],
    scenarios: Sequence[Mapping[str, object]],
    results_path: str | Path,
) -> Path:
    # Paper Figures 4 and 5: recurrent against simple, absolute and paired difference
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.0))
    colors = {"recurrent": COLORS["Deep Hedging"], "simple": "#dd6b20"}
    labels = {"recurrent": "with previous position", "simple": "memoryless"}

    for ax, (metric, title) in zip(
        axes[:2],
        (("risk", "Risk measure of the hedging error"),
         ("mean_cost", "Mean transaction cost")),
    ):
        positions = np.arange(len(scenarios))
        width = 0.36
        for offset, name in ((-width / 2, "recurrent"), (width / 2, "simple")):
            means, errs = [], []
            for scenario in scenarios:
                subset = [
                    r for r in records
                    if r["scenario"] == scenario["key"]
                    and (bool(r["recurrent"]) == (name == "recurrent"))
                ]
                values = np.array([float(r[metric]) for r in subset])
                means.append(values.mean() if values.size else np.nan)
                errs.append(values.std(ddof=1) if values.size > 1 else 0.0)
            ax.bar(positions + offset, means, width, yerr=errs, capsize=4.0,
                   color=colors[name], alpha=0.9, label=labels[name])
        ax.set_xticks(positions)
        ax.set_xticklabels([str(s["label"]) for s in scenarios], fontsize=9.0)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8.6)
        _style(ax)

    # Paired difference within seed, the only way the small gaps are visible
    ax = axes[2]
    positions = np.arange(len(scenarios))
    gaps, errs = [], []
    for scenario in scenarios:
        rec = {r["seed"]: r for r in records
               if r["scenario"] == scenario["key"] and r["recurrent"]}
        simp = {r["seed"]: r for r in records
                if r["scenario"] == scenario["key"] and not r["recurrent"]}
        shared = sorted(set(rec) & set(simp))
        diffs = np.array([float(rec[k]["risk"]) - float(simp[k]["risk"]) for k in shared])
        gaps.append(diffs.mean() if diffs.size else np.nan)
        errs.append(diffs.std(ddof=1) / np.sqrt(diffs.size) if diffs.size > 1 else 0.0)
    ax.bar(positions, gaps, 0.5, yerr=errs, capsize=5.0, color="#805ad5", alpha=0.9)
    ax.axhline(0.0, color=TEXT_COLOR, linewidth=1.3)
    for i, (gap, err) in enumerate(zip(gaps, errs)):
        ax.annotate(f"{gap:+.4f}\nse {err:.4f}", (i, gap), ha="center",
                    va="bottom" if gap >= 0 else "top", fontsize=8.6, color=TEXT_COLOR)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(s["label"]) for s in scenarios], fontsize=9.0)
    ax.set_ylabel("Risk gap, recurrent minus memoryless")
    ax.set_title("Paired within seed: negative favours recurrent")
    _style(ax)
    fig.suptitle("Paper Figures 4 and 5: recurrent against simple network, 5 seed mean",
                 fontsize=11.5, color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "recurrent_ablation.png"))


def plot_architecture_comparison(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # Paper Section 5.1 fidelity: the project default against the paper architecture
    summary = payload["summary"]
    variants = payload["variants"]
    metrics = (("mean", "Mean PnL"), ("std", "Std"), ("cvar_95", "CVaR 95"),
               ("cvar_99", "CVaR 99"), ("mean_cost", "Mean cost"))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    ax = axes[0]
    positions = np.arange(len(metrics))
    width = 0.38
    for offset, variant, color in (
        (-width / 2, variants[0], COLORS["Deep Hedging"]),
        (width / 2, variants[1], "#dd6b20"),
    ):
        block = summary.get(variant["key"])
        if not block:
            continue
        means = [block[m]["mean"] for m, _ in metrics]
        errs = [np.nan_to_num(block[m]["std"]) for m, _ in metrics]
        ax.bar(positions + offset, means, width, yerr=errs, capsize=3.5, color=color,
               alpha=0.9, label=f"{variant['label']} ({int(block['n_params']['mean']):,} params)")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9.0, rotation=12)
    ax.set_ylabel("Value")
    ax.set_title("Risk measures under one percent cost")
    ax.legend(frameon=False, fontsize=8.6)
    _style(ax)

    ax = axes[1]
    for variant, color in ((variants[0], COLORS["Deep Hedging"]), (variants[1], "#dd6b20")):
        histories = [
            r["history"] for r in payload["records"] if r["variant"] == variant["key"]
        ]
        if not histories:
            continue
        length = min(len(h) for h in histories)
        stacked = np.array([h[:length] for h in histories])
        grid = np.arange(1, length + 1)
        ax.plot(grid, stacked.mean(axis=0), color=color, linewidth=1.9,
                label=str(variant["label"]))
        ax.fill_between(grid, stacked.mean(axis=0) - stacked.std(axis=0),
                        stacked.mean(axis=0) + stacked.std(axis=0), color=color, alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training entropic risk")
    ax.set_title("Learning curves, mean with one standard deviation")
    ax.legend(frameon=False, fontsize=8.8)
    _style(ax)
    fig.suptitle("Paper Section 5.1: architecture fidelity check", fontsize=11.5,
                 color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "paper_architecture.png"))

def plot_call_spread(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # Paper Figures 8 and 9: holdings flatten as risk aversion rises
    grid = payload["holdings_grid"]
    s_grid = np.array(grid["s"], dtype=float)
    v_grid = np.array(grid["v"], dtype=float)
    holdings = payload["holdings"]
    model = np.array(grid["model"], dtype=float)
    alphas = sorted(holdings)
    fig, axes = plt.subplots(1, len(alphas) + 2, figsize=(5.2 * (len(alphas) + 2), 4.7))
    j = int(np.argmin(np.abs(v_grid - 0.04)))
    ax = axes[0]
    ax.plot(s_grid, model[:, j], color="#c53030", linewidth=2.1, label="Model spread delta")
    for key, color in zip(alphas, CURVE_COLORS):
        ax.plot(s_grid, np.array(holdings[key], dtype=float)[:, j], color=color,
                linewidth=1.8, label=f"Deep hedge alpha {key}")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Spread holding")
    ax.set_title("Holdings at variance 0.04")
    ax.legend(frameon=False, fontsize=8.4)
    _style(ax)

    for ax, key in zip(axes[1:], alphas):
        values = np.array(holdings[key], dtype=float)
        vmin = float(np.min([values.min(), model.min()]))
        vmax = float(np.max([values.max(), model.max()]))
        image = ax.pcolormesh(s_grid, v_grid, values.T, cmap="viridis",
                              shading="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(image, ax=ax, label="Spread holding")
        ax.set_xlabel("Spot")
        ax.set_ylabel("Variance")
        ax.set_title(f"Deep hedge alpha {key} at step {grid['step']}")
        _style(ax)

    ax = axes[-1]
    spread = np.array(holdings[alphas[-1]], dtype=float) - np.array(holdings[alphas[0]], dtype=float)
    lim = float(np.max(np.abs(spread))) or 1.0
    image = ax.pcolormesh(s_grid, v_grid, spread.T, cmap="coolwarm", shading="auto",
                          vmin=-lim, vmax=lim)
    fig.colorbar(image, ax=ax, label="Holding difference")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Variance")
    ax.set_title(f"alpha {alphas[-1]} minus alpha {alphas[0]}")
    _style(ax)
    fig.suptitle("Paper Figures 8 and 9: call spread, higher risk aversion flattens the hedge",
                 fontsize=11.5, color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "call_spread.png"))

def plot_heston_asymptotics(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # Paper Figure 11: same 2/3 scaling under Heston with two instruments
    summary = payload["summary"]
    costs = sorted(float(k) for k in summary)
    q = float(payload["q"])
    p0 = float(payload["p_0"])
    grid = [c for c in costs if c > 0.0]
    if len(grid) < 2:
        raise ValueError("at least two positive cost levels are required")
    x = np.log(np.array(grid))
    fit = payload["fit"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for ax, (label, base, key_free, key_fixed) in zip(
        axes,
        (
            ("vs q", q, "free_vs_q", "fixed_two_thirds_vs_q"),
            ("vs p_0", p0, "free_vs_p0", "fixed_two_thirds_vs_p0"),
        ),
    ):
        excess = np.array([summary[f"{c:.6f}"]["entropic"]["mean"] - base for c in grid])
        err = np.array([summary[f"{c:.6f}"]["entropic"]["std"] for c in grid])
        y = np.log(np.clip(excess, 1e-12, None))
        ax.errorbar(x, y, yerr=err / np.clip(excess, 1e-12, None), fmt="o",
                    color="#2b6cb0", markersize=6.5, capsize=3.5, linewidth=1.3,
                    label="Heston deep hedge (seed mean)")
        line = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
        free = fit.get(key_free)
        if free:
            ax.plot(line, free["intercept"] + free["slope"] * line, color="#c53030",
                    linewidth=2.0, label=f"OLS slope {free['slope']:.3f}")
        fixed = fit.get(key_fixed)
        if fixed:
            ax.plot(line, fixed["intercept"] + (2.0 / 3.0) * line, color="#2f855a",
                    linewidth=1.8, linestyle="--", label="slope fixed at 2/3")
        ax.set_xlabel("log(cost rate)")
        ax.set_ylabel(f"log(p_eps - {'q' if label == 'vs q' else 'p_0'})")
        ax.set_title(f"Heston price excess {label}")
        ax.legend(frameon=False, fontsize=8.8)
        _style(ax)
    fig.suptitle(
        f"Paper Figure 11: Heston price asymptotics, q = {q:.4f}, p_0 = {p0:.4f}, "
        f"slope {fit.get('free_vs_q', {}).get('slope', float('nan')):.3f}",
        fontsize=11.5, color=TEXT_COLOR,
    )
    fig.tight_layout()
    return _save(fig, _output(results_path, "price_asymptotics_heston.png"))

def plot_eda_market(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # Simulator EDA: real terminal densities and the variance law
    market = payload["market"]
    samples = payload["market"].get("samples", {})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    for key, label, color in (
        ("gbm", "GBM sigma 0.2", COLORS["Black-Scholes"]),
        ("heston", "Heston Euler, xi 0.3", COLORS["Leland"]),
        ("paper_heston", "Paper Heston, exact CIR, xi 2", COLORS["Deep Hedging"]),
    ):
        values = np.asarray(samples.get(key, []), dtype=float)
        if values.size == 0:
            continue
        entry = market[key]["terminal"]
        ax.hist(values, bins=160, range=(80.0, 120.0), density=True, histtype="step",
                linewidth=1.7, color=color,
                label=(f"{label} (skew {entry['skew']:.2f}, "
                       f"ex kurt {entry['excess_kurtosis']:.2f})"))
    ax.set_yscale("log")
    ax.set_ylim(4e-4, 3e-1)
    ax.set_xlabel("Terminal spot")
    ax.set_ylabel("Density, log scale")
    ax.set_title("Terminal spot density, real samples")
    ax.legend(frameon=False, fontsize=7.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=1)
    _style(ax)

    ax = axes[1]
    # Cumulative view on a common axis, the tail mass is what separates the schemes
    for key, label, color in (
        ("heston", "Heston Euler, xi 0.3", COLORS["Leland"]),
        ("paper_heston", "Paper Heston, exact CIR, xi 2", COLORS["Deep Hedging"]),
    ):
        values = np.asarray(samples.get(key, []), dtype=float)
        if values.size == 0:
            continue
        ordered = np.sort(values)
        cdf = np.arange(1, ordered.size + 1) / ordered.size
        ax.plot(ordered, cdf, color=color, linewidth=1.8, label=label)
    ax.set_xlim(80.0, 100.0)
    ax.set_ylim(0.0, 0.20)
    ax.axvline(100.0, color=TEXT_COLOR, linestyle=":", linewidth=1.3, label="strike 100")
    ax.set_xlabel("Terminal spot, common axis")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Lower tail on a common axis: exact CIR carries more mass below 90")
    ax.legend(frameon=False, fontsize=8.4)
    _style(ax)
    fig.suptitle("Simulator EDA: the paper Heston set violates Feller by design",
                 fontsize=11.5, color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "eda_market.png"))


def plot_eda_moments(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # Tail quantiles and standard errors of the simulated payoffs
    market = payload["market"]
    keys = ("gbm", "heston", "paper_heston")
    labels = ("GBM", "Heston Euler", "Paper Heston")
    fields = (("q01", "1 percent"), ("q05", "5 percent"), ("median", "Median"),
              ("q95", "95 percent"), ("q99", "99 percent"))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    positions = np.arange(len(fields))
    width = 0.26
    colors = (COLORS["Black-Scholes"], COLORS["Leland"], COLORS["Deep Hedging"])
    for offset, key, label, color in zip((-width, 0.0, width), keys, labels, colors):
        entry = market[key]["payoff"]
        ax.bar(positions + offset, [entry[f] for f, _ in fields], width, color=color,
               alpha=0.9, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels([lbl for _, lbl in fields], fontsize=9.0)
    ax.set_ylabel("Call payoff")
    ax.set_title("Payoff quantiles by simulator")
    ax.legend(frameon=False, fontsize=8.6)
    _style(ax)

    ax = axes[1]
    names = []
    ratios = []
    for key, label in zip(keys, labels):
        entry = market[key]["payoff"]
        # Standard error relative to the mean, the Monte Carlo precision per payoff
        ratios.append(abs(entry["se_mean"] / entry["mean"]) if entry["mean"] else 0.0)
        names.append(label)
    ax.bar(names, ratios, color=colors, alpha=0.9)
    for i, value in enumerate(ratios):
        ax.annotate(f"{value:.5f}", (i, value), ha="center", va="bottom", fontsize=9.0,
                    color=TEXT_COLOR)
    ax.set_ylabel("Standard error over mean")
    ax.set_title("Monte Carlo precision of the mean payoff")
    _style(ax)
    fig.suptitle("Simulator EDA: payoff tails and Monte Carlo precision", fontsize=11.5,
                 color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "eda_moments.png"))


def plot_eda_results(payload: Mapping[str, object], results_path: str | Path) -> Path:
    # EDA on the results: seed noise floor against the gaps the documents claim
    studies = payload["stored_studies"]
    metrics = studies["multiseed"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    # Relative seed standard deviation per strategy, the noise floor of each metric
    names = list(metrics.keys())
    fields = (("cvar_95", "CVaR 95"), ("cvar_99", "CVaR 99"), ("entropic", "Entropic"),
              ("mean_pnl", "Mean PnL"), ("mean_cost", "Mean cost"))
    positions = np.arange(len(fields))
    width = 0.26
    colors = (COLORS["Deep Hedging"], COLORS["Black-Scholes"], COLORS["Leland"])
    for offset, name, color in zip((-width, 0.0, width), names, colors):
        block = metrics[name]
        values = []
        for key, _ in fields:
            mean = abs(float(block[key]["mean"]))
            values.append(100.0 * float(block[key]["std"]) / mean if mean > 0.0 else 0.0)
        ax.bar(positions + offset, values, width, color=color, alpha=0.9, label=name)
    ax.set_xticks(positions)
    ax.set_xticklabels([lbl for _, lbl in fields], fontsize=8.8)
    ax.set_yscale("log")
    ax.set_ylabel("Seed standard deviation, percent of mean")
    ax.set_title("Noise floor by strategy, 5 seeds, log scale")
    ax.legend(frameon=False, fontsize=8.4)
    _style(ax)

    ax = axes[1]
    # The gap to Leland against the noise floor of the same metric
    deep = metrics["Deep Hedging"]
    base = metrics["Leland"]
    paired = studies["multiseed"].get("paired_vs_leland", {})
    labels = []
    ratios = []
    for key, label in fields:
        gap = abs(float(base[key]["mean"]) - float(deep[key]["mean"]))
        se = float(deep[key].get("se", 0.0))
        labels.append(label)
        # Paired per-seed t when available, since that is the quoted statistic
        if key in paired and paired[key].get("t"):
            ratios.append(abs(float(paired[key]["t"])))
        else:
            ratios.append(gap / se if se > 0.0 else 0.0)
    bars = ax.bar(np.arange(len(labels)), ratios, 0.6, color=COLORS["Deep Hedging"], alpha=0.9)
    for bar, ratio in zip(bars, ratios):
        ax.annotate(f"{ratio:.1f}", (bar.get_x() + bar.get_width() / 2.0, ratio),
                    ha="center", va="bottom", fontsize=8.6, color=TEXT_COLOR)
    ax.axhline(2.0, color=COLORS["Black-Scholes"], linestyle="--", linewidth=1.5,
               label="t = 2, conventional threshold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.0)
    ax.set_ylim(0.0, max(ratios) * 1.18)
    ax.set_ylabel("Gap to Leland in standard errors")
    ax.set_title("Every claimed gap against its own seed noise")
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
    _style(ax)
    fig.suptitle("EDA on the stored results: is every reported gap above the seed noise floor",
                 fontsize=11.5, color=TEXT_COLOR)
    fig.tight_layout()
    return _save(fig, _output(results_path, "eda_results.png"))
