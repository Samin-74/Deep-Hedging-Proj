import torch


def terminal_payoff(spots: torch.Tensor, K: float) -> torch.Tensor:
    # Short European call liability
    return torch.relu(spots[:, -1] - K)


def hedge_pnl(
    spots: torch.Tensor,
    deltas: torch.Tensor,
    K: float,
    cost_rate: float,
    return_path: bool = True,
) -> dict[str, torch.Tensor]:
    # Discrete self-financing hedge with proportional transaction costs
    if spots.dim() != 2 or deltas.dim() != 2:
        raise ValueError("spots and deltas must be 2D tensors")
    if spots.shape[1] != deltas.shape[1] + 1:
        raise ValueError("spots must have one more column than deltas")
    if spots.shape[0] != deltas.shape[0]:
        raise ValueError("spots and deltas must share the batch dimension")
    if cost_rate < 0.0:
        raise ValueError("cost_rate must be non-negative")

    prev = torch.cat([torch.zeros_like(deltas[:, :1]), deltas[:, :-1]], dim=1)
    trades = deltas - prev
    price_changes = spots[:, 1:] - spots[:, :-1]
    gains = (deltas * price_changes).sum(dim=1)
    costs = cost_rate * (spots[:, :-1] * trades.abs()).sum(dim=1)
    payoff = terminal_payoff(spots, K)
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
        step_pnl = deltas * price_changes - cost_rate * spots[:, :-1] * trades.abs()
        cumulative = torch.cat(
            [torch.zeros_like(pnl).unsqueeze(1), step_pnl.cumsum(dim=1)], dim=1
        )
        # Pre-settlement path excludes the terminal liability
        result["pnl_path_pre"] = cumulative
        terminal = cumulative[:, -1:] - payoff.unsqueeze(1)
        result["pnl_path"] = torch.cat([cumulative[:, :-1], terminal], dim=1)
    return result
