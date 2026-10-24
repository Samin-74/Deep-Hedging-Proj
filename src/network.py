import math

import torch
from torch import nn
from torch.nn import functional as F


class FlooredBatchNorm1d(nn.BatchNorm1d):
    # BatchNorm with a per-feature variance floor so constant inputs stay well defined
    def __init__(self, num_features: int, min_var: float = 1e-2, **kwargs) -> None:
        if kwargs.get("momentum", 0.1) is None:
            kwargs["momentum"] = 0.1
        super().__init__(num_features, **kwargs)
        self.min_var = float(min_var)

    def _running_norm(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluation path uses running statistics with a variance floor
        var = torch.clamp(self.running_var, min=self.min_var)
        out = (x - self.running_mean) / torch.sqrt(var + self.eps)
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            return super().forward(x)
        if x.shape[0] < 2:
            # A single sample cannot define batch statistics
            return self._running_norm(x)
        if not self.training:
            return self._running_norm(x)
        batch_var = x.var(dim=0, unbiased=False)
        degenerate = batch_var < self.min_var
        if not bool(degenerate.any()):
            return super().forward(x)
        # Degenerate features fall back to running statistics, the rest stay batch normalized
        with torch.no_grad():
            self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * x.mean(dim=0))
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * batch_var)
            self.num_batches_tracked.add_(1)
        mean = torch.where(degenerate, self.running_mean, x.mean(dim=0))
        var = torch.where(degenerate, self.running_var.clamp(min=self.min_var), batch_var)
        out = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            out = out * self.weight + self.bias
        return out


def make_norm(norm_type: str, dim: int) -> nn.Module:
    # LayerNorm has no running statistics, so train and eval agree exactly
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "batchnorm":
        return FlooredBatchNorm1d(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class MLP(nn.Module):
    # Linear + normalization + LeakyReLU stack
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        out_dim: int = 1,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    make_norm(norm_type, hidden_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            dim = hidden_dim
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dim, out_dim)
        nn.init.normal_(self.head.weight, std=0.05)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class SharedTimeMLP(nn.Module):
    # Shared linear weights with per-step normalization
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_steps: int,
        out_dim: int = 1,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim = in_dim
        for _ in range(n_layers):
            self.linears.append(nn.Linear(dim, hidden_dim))
            self.norms.append(
                nn.ModuleList([make_norm(norm_type, hidden_dim) for _ in range(n_steps)])
            )
            dim = hidden_dim
        self.head = nn.Linear(dim, out_dim)
        nn.init.normal_(self.head.weight, std=0.05)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        h = x
        for linear, norm_group in zip(self.linears, self.norms):
            h = F.leaky_relu(norm_group[step](linear(h)), 0.2)
        return self.head(h)


class DeepHedgingNetwork(nn.Module):
    # Semi-recurrent hedging policy over the full time grid
    def __init__(
        self,
        n_steps: int,
        K: float,
        T: float,
        base_variance: float,
        dt: float,
        max_position: float = 2.0,
        hidden_dim: int = 64,
        n_layers: int = 3,
        time_dependent: bool = False,
        feature_dim: int = 6,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if base_variance <= 0.0:
            raise ValueError("base_variance must be positive")
        if norm_type not in {"layernorm", "batchnorm"}:
            raise ValueError("norm_type must be 'layernorm' or 'batchnorm'")
        self.n_steps = int(n_steps)
        self.K = float(K)
        self.T = float(T)
        self.base_variance = float(base_variance)
        self.dt = float(dt)
        self.max_position = float(max_position)
        self.time_dependent = bool(time_dependent)
        self.feature_dim = int(feature_dim)
        self.norm_type = str(norm_type)

        if self.time_dependent:
            self.nets = nn.ModuleList(
                [
                    MLP(feature_dim, hidden_dim, n_layers, norm_type=self.norm_type)
                    for _ in range(self.n_steps)
                ]
            )
            self.shared = None
        else:
            self.nets = None
            self.shared = SharedTimeMLP(
                feature_dim, hidden_dim, n_layers, self.n_steps, norm_type=self.norm_type
            )

    def _delta_from_features(self, step: int, features: torch.Tensor) -> torch.Tensor:
        if self.time_dependent:
            raw = self.nets[step](features).squeeze(-1)
        else:
            raw = self.shared(features, step).squeeze(-1)
        return self.max_position * torch.tanh(raw)

    def _features(
        self,
        spot: torch.Tensor,
        log_return: torch.Tensor,
        variance: torch.Tensor,
        prev_delta: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        vol_T = math.sqrt(self.base_variance * self.T)
        vol_dt = math.sqrt(self.base_variance * self.dt)
        moneyness = (spot / self.K - 1.0) / vol_T
        log_ret_norm = log_return / vol_dt
        vol_scale = torch.sqrt(torch.clamp(variance, min=0.0)) * math.sqrt(self.T) / vol_T
        var_norm = variance / self.base_variance - 1.0
        time_frac = torch.full_like(spot, (self.n_steps - step) / self.n_steps)
        prev_norm = prev_delta / self.max_position
        return torch.stack(
            [moneyness, log_ret_norm, vol_scale, var_norm, time_frac, prev_norm], dim=-1
        )

    def forward(
        self,
        spots: torch.Tensor,
        variances: torch.Tensor | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        batch, n_cols = spots.shape
        horizon = self.n_steps if n_steps is None else int(n_steps)
        if not 1 <= horizon <= self.n_steps:
            raise ValueError("horizon must lie in [1, n_steps]")
        if n_cols < horizon + 1:
            raise ValueError(f"Expected {horizon + 1} spot columns, got {n_cols}")
        if variances is not None and variances.shape[1] < horizon:
            raise ValueError("variances must provide one column per decision step")
        prev_delta = spots.new_zeros(batch)
        log_return = spots.new_zeros(batch)
        deltas: list[torch.Tensor] = []
        for k in range(horizon):
            if k > 0:
                log_return = torch.log(spots[:, k] / spots[:, k - 1])
            if variances is None:
                variance = spots.new_full((batch,), self.base_variance)
            else:
                variance = variances[:, k]
            features = self._features(spots[:, k], log_return, variance, prev_delta, k)
            prev_delta = self._delta_from_features(k, features)
            deltas.append(prev_delta)
        return torch.stack(deltas, dim=1)

    @torch.no_grad()
    def policy_at_step(
        self,
        spots: torch.Tensor,
        step: int,
        prev_delta: torch.Tensor | float,
        variance: torch.Tensor | float | None = None,
        log_return: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        if not torch.is_tensor(spots):
            spots = torch.as_tensor(spots, dtype=torch.float32)
        if variance is None:
            variance = torch.full_like(spots, self.base_variance)
        elif not torch.is_tensor(variance):
            variance = torch.full_like(spots, float(variance))
        if not torch.is_tensor(prev_delta):
            prev_delta = torch.full_like(spots, float(prev_delta))
        if not torch.is_tensor(log_return):
            log_return = torch.full_like(spots, float(log_return))
        features = self._features(spots, log_return, variance, prev_delta, step)
        return self._delta_from_features(step, features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
