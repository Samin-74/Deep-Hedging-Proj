import torch


def terminal_payoff(
    spots: torch.Tensor,
    K: float,
    second_strike: float | None = None,
) -> torch.Tensor:
    # Short European call liability on instrument 0, or a call spread when a second strike is given
    terminal = spots[:, -1] if spots.dim() == 2 else spots[:, -1, 0]
    payoff = torch.relu(terminal - K)
    if second_strike is not None:
        payoff = payoff - torch.relu(terminal - float(second_strike))
    return payoff


def hedge_pnl(
    spots: torch.Tensor,
    deltas: torch.Tensor,
    K: float,
    cost_rate: float,
    return_path: bool = True,
    second_strike: float | None = None,
) -> dict[str, torch.Tensor]:
    # Discrete self-financing hedge with proportional costs in one or more instruments
    if spots.dim() not in (2, 3):
        raise ValueError("spots must be 2D (paths, steps) or 3D (paths, steps, instruments)")
    if deltas.dim() != spots.dim():
        raise ValueError("spots and deltas must share rank")
    if spots.shape[1] != deltas.shape[1] + 1:
        raise ValueError("spots must have one more column than deltas")
    if spots.shape[0] != deltas.shape[0]:
        raise ValueError("spots and deltas must share the batch dimension")
    if spots.dim() == 3 and spots.shape[2] != deltas.shape[2]:
        raise ValueError("spots and deltas must trade the same number of instruments")
    if cost_rate < 0.0:
        raise ValueError("cost_rate must be non-negative")

    prev = torch.cat([torch.zeros_like(deltas[:, :1]), deltas[:, :-1]], dim=1)
    trades = deltas - prev
    price_changes = spots[:, 1:] - spots[:, :-1]
    # Instrument gains and costs are summed before accumulating over time
    leg_gains = deltas * price_changes
    leg_costs = cost_rate * spots[:, :-1] * trades.abs()
    n_paths = deltas.shape[0]
    gains = leg_gains.reshape(n_paths, -1).sum(dim=1)
    costs = leg_costs.reshape(n_paths, -1).sum(dim=1)
    payoff = terminal_payoff(spots, K, second_strike)
    pnl = -payoff + gains - costs

    result = {
        "pnl": pnl,
        "payoff": payoff,
        "gains": gains,
        "costs": costs,
        "turnover": trades.abs().sum(dim=1),
        "deltas": deltas,
    }
    if return_path:
        step_pnl = leg_gains - leg_costs
        if step_pnl.dim() == 3:
            step_pnl = step_pnl.sum(dim=-1)
        cumulative = torch.cat(
            [torch.zeros_like(pnl).unsqueeze(1), step_pnl.cumsum(dim=1)], dim=1
        )
        # Pre-settlement path excludes the terminal liability
        result["pnl_path_pre"] = cumulative
        terminal = cumulative[:, -1:] - payoff.unsqueeze(1)
        result["pnl_path"] = torch.cat([cumulative[:, :-1], terminal], dim=1)
    return result
