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

    left.errorbar(
        mean_pnl,
        cvar95,
        xerr=mean_pnl_sd,
        yerr=cvar95_sd,
        color=COLORS["Deep Hedging"],
        marker="o",
        markersize=6,
        linewidth=1.9,
        capsize=3,
        label="Deep Hedging",
    )
    for lam, x, y in zip(lambdas, mean_pnl, cvar95):
        left.annotate(f"{lam:.2f}", xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=TEXT_COLOR)
    left.set_xlabel("Mean PnL")
    left.set_ylabel("CVaR 95, loss basis")
    left.set_title("Risk return frontier, labels are lambda")
    left.legend(frameon=False, fontsize=9.0, loc="upper left")

    middle.errorbar(
        cost,
        cvar95,
        xerr=cost_sd,
        yerr=cvar95_sd,
        color=COLORS["Deep Hedging"],
        marker="o",
        markersize=6,
        linewidth=1.9,
        capsize=3,
        label="Deep Hedging",
    )
    for lam, x, y in zip(lambdas, cost, cvar95):
        middle.annotate(f"{lam:.2f}", xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=TEXT_COLOR)
    middle.set_xlabel("Mean transaction cost")
    middle.set_ylabel("CVaR 95, loss basis")
    middle.set_title("Cost against tail risk")
    middle.legend(frameon=False, fontsize=9.0, loc="upper left")

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

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.4))
    left, middle, right = axes[0], axes[1], axes[2]

    # Heatmap makes the variance dependence readable where curves overlap
    vol_grid = np.linspace(0.08, 0.36, 57)
    surface = np.zeros((vol_grid.size, spots.numel()))
    for i, vol in enumerate(vol_grid):
        variance = torch.full_like(spots, float(vol) ** 2)
        with torch.no_grad():
            surface[i] = model.policy_at_step(spots, step, 0.0, variance=variance).numpy()
    mesh = left.pcolormesh(
        spots.numpy(),
        vol_grid,
        surface,
        cmap="viridis",
        shading="auto",
    )
    bar = fig.colorbar(mesh, ax=left, pad=0.02)
    bar.set_label("Hedge position")
    bar.ax.tick_params(labelsize=8.5, colors=MUTED_COLOR)
    for vol in vol_levels:
        left.axhline(float(vol), color="white", linestyle=":", linewidth=1.2, alpha=0.85)
    left.set_xlabel("Spot price at the decision date")
    left.set_ylabel("Instantaneous volatility")
    left.set_title(f"Heston position surface at step {step}, previous position 0")

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
    for color, prev in zip(CURVE_COLORS, prev_deltas):
        variance = torch.full_like(spots, float(vol_levels[len(vol_levels) // 2]) ** 2)
        with torch.no_grad():
            curve = model.policy_at_step(spots, step, float(prev), variance=variance)
        right.plot(
            x,
            (curve - float(prev)).numpy(),
            color=color,
            linewidth=2.0,
            label=f"Previous position {prev:.2f}",
        )
    right.axhline(0.0, color=TEXT_COLOR, linewidth=0.9)
    right.set_xlabel("Spot price at the decision date")
    right.set_ylabel("Trade size")
    right.set_title(f"No-trade band at tolerance {no_trade_tol:.2f}")
    right.legend(frameon=False, fontsize=8.6, loc="upper left")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
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
    return _save(fig, path)


def plot_heston_cost_sweep(
    records: Sequence[Mapping[str, float]],
    results_path: str | Path,
) -> Path:
    # Heston CVaR across cost levels, same layout as the GBM study
    if not records:
        raise ValueError("records must not be empty")
    path = _output(results_path, "heston_cost_sensitivity.png")
    costs = [100.0 * float(r["cost"]) for r in records]
    strategies = (
        ("Deep Hedging", "deep"),
        ("Black-Scholes", "bs"),
        ("Leland", "leland"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    for ax, quantile in zip(axes, (95, 99)):
        for label, prefix in strategies:
            values = [float(r[f"{prefix}_cvar_{quantile}"]) for r in records]
            ax.plot(costs, values, marker="o", markersize=5.5, linewidth=1.9, color=COLORS[label], label=label)
        ax.set_title(f"Heston CVaR at {quantile} percent confidence")
        ax.set_xlabel("Proportional transaction cost (percent)")
        ax.set_ylabel("CVaR of terminal PnL, loss basis")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9.0, loc="upper left")
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
