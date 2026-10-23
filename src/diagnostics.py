from typing import Mapping, Sequence

import torch

from .benchmarks import bs_delta, leland_delta


def scenario_analysis(
    model,
    config,
    step: int | None = None,
    magnitudes: Sequence[float] = (0.88, 0.92, 0.96, 0.98, 1.0, 1.02, 1.04, 1.08, 1.12),
    reference: float | None = None,
) -> dict[str, object]:
    # Benchmark deltas and the learned response along a spot move
    step = model.n_steps // 2 if step is None else int(step)
    if not 0 <= step < model.n_steps:
        raise ValueError("step must lie in [0, n_steps)")
    if not magnitudes:
        raise ValueError("magnitudes must not be empty")
    reference = config.K if reference is None else float(reference)
    tau = (config.n_steps - step) * config.dt
    spots = torch.tensor([reference * float(m) for m in magnitudes], dtype=torch.float32)
    prev = bs_delta(
        torch.tensor([reference]), config.K, config.r, config.benchmark_sigma, tau
    )
    prev_delta = float(prev[0])
    model.eval()
    with torch.no_grad():
        policy = model.policy_at_step(spots, step, prev_delta)
    return {
        "step": step,
        "tau": tau,
        "prev_delta": prev_delta,
        "spots": spots.tolist(),
        "magnitudes": [float(m) for m in magnitudes],
        "policy": policy.tolist(),
        "bs": bs_delta(spots, config.K, config.r, config.benchmark_sigma, tau).tolist(),
        "leland": leland_delta(
            spots,
            config.K,
            config.r,
            config.benchmark_sigma,
            tau,
            config.cost_rate,
            config.dt,
        ).tolist(),
    }


def delta_path_by_level(
    model,
    config,
    magnitudes: Sequence[float],
    steps: Sequence[int] | None = None,
    reference: float | None = None,
) -> dict[float, list[float]]:
    # Learned delta term structure for fixed spot levels
    steps = range(0, model.n_steps, max(1, model.n_steps // 10)) if steps is None else steps
    reference = config.K if reference is None else float(reference)
    curves: dict[float, list[float]] = {float(m): [] for m in magnitudes}
    model.eval()
    with torch.no_grad():
        for step in steps:
            tau = (config.n_steps - step) * config.dt
            spots = torch.tensor(
                [reference * float(m) for m in magnitudes], dtype=torch.float32
            )
            prev = float(
                bs_delta(torch.tensor([reference]), config.K, config.r, config.benchmark_sigma, tau)[0]
            )
            values = model.policy_at_step(spots, step, prev)
            for magnitude, value in zip(magnitudes, values.tolist()):
                curves[float(magnitude)].append(value)
    return curves


def cost_drag(
    costs: Mapping[str, torch.Tensor],
    pnls: Mapping[str, torch.Tensor],
    level: float,
) -> list[tuple[str, float, float, float]]:
    # Realized cost, gross edge and net PnL per strategy
    rows: list[tuple[str, float, float, float]] = []
    for name, pnl in pnls.items():
        cost = costs[name]
        if cost.shape[0] != pnl.shape[0]:
            raise ValueError("costs and pnls must share the path dimension")
        rows.append(
            (
                name,
                float(cost.mean()),
                float((pnl + cost).mean()),
                float(pnl.mean()),
            )
        )
    return rows


def tail_attribution(
    pnl: torch.Tensor,
    costs: torch.Tensor,
    alpha: float = 0.95,
) -> dict[str, float]:
    # Share of the loss tail explained by transaction costs
    losses = -pnl
    n = losses.numel()
    k = max(1, min(int(round((1.0 - alpha) * n)), n))
    index = torch.topk(losses, k, largest=True).indices
    tail_cost = float(costs[index].mean())
    tail_loss = float(losses[index].mean())
    return {
        "tail_loss": tail_loss,
        "tail_cost": tail_cost,
        "tail_cost_share": tail_cost / tail_loss if tail_loss > 0.0 else float("nan"),
    }
