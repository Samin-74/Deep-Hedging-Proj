import math

import torch
from torch import nn


class EntropicRiskLoss(nn.Module):
    # Entropic risk measure of terminal PnL
    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        if lam <= 0.0:
            raise ValueError("lambda must be positive")
        self.lam = float(lam)

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        scaled = -self.lam * pnl.reshape(-1)
        log_mean_exp = torch.logsumexp(scaled, dim=0) - math.log(scaled.numel())
        return log_mean_exp / self.lam


class CVaRLoss(nn.Module):
    # Rockafellar-Uryasev CVaR with trainable VaR threshold
    def __init__(self, alpha: float = 0.95, init_w: float = 0.0) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.w = nn.Parameter(torch.tensor(float(init_w)))

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        losses = -pnl.reshape(-1)
        tail = torch.relu(losses - self.w)
        return self.w + tail.mean() / (1.0 - self.alpha)


def empirical_cvar(pnl: torch.Tensor, alpha: float) -> torch.Tensor:
    # Mean of the worst (1 - alpha) fraction of losses
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    losses = -pnl.reshape(-1)
    n = losses.numel()
    k = min(max(1, int(math.ceil((1.0 - alpha) * n))), n)
    return torch.topk(losses, k, largest=True).values.mean()


def var_threshold(pnl: torch.Tensor, alpha: float) -> float:
    # Rockafellar-Uryasev minimizer is the alpha quantile of losses
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    losses = -pnl.reshape(-1)
    return float(torch.quantile(losses, float(alpha), interpolation="lower"))


def max_drawdown(paths: torch.Tensor) -> torch.Tensor:
    # Per-path maximum drawdown of a cumulative PnL path
    running_max = torch.cummax(paths, dim=1).values
    return (running_max - paths).max(dim=1).values


def summarize_pnl(pnl: torch.Tensor, pnl_path: torch.Tensor | None = None) -> dict[str, float]:
    summary = {
        "mean": float(pnl.mean()),
        "std": float(pnl.std(unbiased=True)),
        "cvar_95": float(empirical_cvar(pnl, 0.95)),
        "cvar_99": float(empirical_cvar(pnl, 0.99)),
        "min": float(pnl.min()),
        "max": float(pnl.max()),
        "q05": float(torch.quantile(pnl, 0.05)),
        "q95": float(torch.quantile(pnl, 0.95)),
    }
    if pnl_path is not None:
        dd = max_drawdown(pnl_path)
        summary["max_drawdown_mean"] = float(dd.mean())
        summary["max_drawdown_p95"] = float(torch.quantile(dd, 0.95))
        summary["max_drawdown_worst"] = float(dd.max())
    return summary
