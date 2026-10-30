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


def _activation(name: str) -> nn.Module:
    return nn.ReLU() if name == "relu" else nn.LeakyReLU(0.2)


class MLP(nn.Module):
    # Linear + normalization + activation stack
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        out_dim: int = 1,
        norm_type: str = "layernorm",
        activation: str = "leaky_relu",
        linear_head: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    make_norm(norm_type, hidden_dim),
                    _activation(activation),
                ]
            )
            dim = hidden_dim
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dim, out_dim)
        self.linear_head = bool(linear_head)
        if not self.linear_head:
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
        activation: str = "leaky_relu",
        linear_head: bool = False,
    ) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = str(activation)
        dim = in_dim
        for _ in range(n_layers):
            self.linears.append(nn.Linear(dim, hidden_dim))
            self.norms.append(
                nn.ModuleList([make_norm(norm_type, hidden_dim) for _ in range(n_steps)])
            )
            dim = hidden_dim
        self.head = nn.Linear(dim, out_dim)
        self.linear_head = bool(linear_head)
        if not self.linear_head:
            nn.init.normal_(self.head.weight, std=0.05)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        h = x
        for linear, norm_group in zip(self.linears, self.norms):
            h = _activation(self.activation)(norm_group[step](linear(h)))
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
        output_dim: int = 1,
        use_prev_delta: bool = True,
        feature_mode: str = "engineered",
        activation: str = "leaky_relu",
        linear_head: bool = False,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if base_variance <= 0.0:
            raise ValueError("base_variance must be positive")
        if norm_type not in {"layernorm", "batchnorm"}:
            raise ValueError("norm_type must be 'layernorm' or 'batchnorm'")
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        if feature_mode not in {"engineered", "paper"}:
            raise ValueError("feature_mode must be 'engineered' or 'paper'")
        if activation not in {"leaky_relu", "relu"}:
            raise ValueError("activation must be 'leaky_relu' or 'relu'")
        self.n_steps = int(n_steps)
        self.K = float(K)
        self.T = float(T)
        self.base_variance = float(base_variance)
        self.dt = float(dt)
        self.max_position = float(max_position)
        self.time_dependent = bool(time_dependent)
        self.output_dim = int(output_dim)
        self.use_prev_delta = bool(use_prev_delta)
        self.feature_mode = str(feature_mode)
        self.activation = str(activation)
        self.linear_head = bool(linear_head)
        self.norm_type = str(norm_type)
        self.feature_dim = self._feature_dim(feature_dim)

        if self.time_dependent:
            self.nets = nn.ModuleList(
                [
                    MLP(
                        self.feature_dim,
                        hidden_dim,
                        n_layers,
                        out_dim=self.output_dim,
                        norm_type=self.norm_type,
                        activation=self.activation,
                        linear_head=self.linear_head,
                    )
                    for _ in range(self.n_steps)
                ]
            )
            self.shared = None
        else:
            self.nets = None
            self.shared = SharedTimeMLP(
                self.feature_dim,
                hidden_dim,
                n_layers,
                self.n_steps,
                out_dim=self.output_dim,
                norm_type=self.norm_type,
                activation=self.activation,
                linear_head=self.linear_head,
            )

    def _feature_dim(self, requested: int) -> int:
        # Paper mode uses (log S, V) plus the previous position per traded instrument
        if self.feature_mode == "paper":
            return 2 + (self.output_dim if self.use_prev_delta else 0)
        return int(requested) - (0 if self.use_prev_delta else 1)

    def _delta_from_features(self, step: int, features: torch.Tensor) -> torch.Tensor:
        if self.time_dependent:
            raw = self.nets[step](features)
        else:
            raw = self.shared(features, step)
        if self.linear_head:
            return raw
        return self.max_position * torch.tanh(raw)

    def _features(
        self,
        spot: torch.Tensor,
        log_return: torch.Tensor,
        variance: torch.Tensor,
        prev_delta: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.feature_mode == "paper":
            # Paper Section 5.1 inputs, scaled to keep the first layer well conditioned
            blocks = [torch.log(spot / self.K)[:, None], variance[:, None] / self.base_variance]
            if self.use_prev_delta:
                if prev_delta.dim() == 1:
                    prev_delta = prev_delta[:, None]
                blocks.append(prev_delta / self.max_position)
            return torch.cat(blocks, dim=-1)
        vol_T = math.sqrt(self.base_variance * self.T)
        vol_dt = math.sqrt(self.base_variance * self.dt)
        moneyness = (spot / self.K - 1.0) / vol_T
        log_ret_norm = log_return / vol_dt
        vol_scale = torch.sqrt(torch.clamp(variance, min=0.0)) * math.sqrt(self.T) / vol_T
        var_norm = variance / self.base_variance - 1.0
        time_frac = torch.full_like(spot, (self.n_steps - step) / self.n_steps)
        blocks = [moneyness, log_ret_norm, vol_scale, var_norm, time_frac]
        if self.use_prev_delta:
            blocks.append((prev_delta / self.max_position).reshape(-1))
        return torch.stack(blocks, dim=-1)

    def forward(
        self,
        spots: torch.Tensor,
        variances: torch.Tensor | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        if spots.dim() not in (2, 3):
            raise ValueError("spots must be 2D (paths, steps) or 3D (paths, steps, instruments)")
        batch = spots.shape[0]
        n_cols = spots.shape[1]
        horizon = self.n_steps if n_steps is None else int(n_steps)
        if not 1 <= horizon <= self.n_steps:
            raise ValueError("horizon must lie in [1, n_steps]")
        if n_cols < horizon + 1:
            raise ValueError(f"Expected {horizon + 1} spot columns, got {n_cols}")
        if variances is not None and variances.shape[1] < horizon:
            raise ValueError("variances must provide one column per decision step")
        # The policy is driven by the first instrument, extra columns are other legs
        primary = spots[:, :, 0] if spots.dim() == 3 else spots
        prev_delta = primary.new_zeros((batch, self.output_dim))
        log_return = primary.new_zeros(batch)
        deltas: list[torch.Tensor] = []
        for k in range(horizon):
            if k > 0:
                log_return = torch.log(primary[:, k] / primary[:, k - 1])
            if variances is None:
                variance = primary.new_full((batch,), self.base_variance)
            else:
                variance = variances[:, k]
            features = self._features(primary[:, k], log_return, variance, prev_delta, k)
            prev_delta = self._delta_from_features(k, features)
            deltas.append(prev_delta)
        stacked = torch.stack(deltas, dim=1)
        return stacked if self.output_dim > 1 else stacked.squeeze(-1)

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
            prev_delta = torch.full((spots.shape[0], self.output_dim), float(prev_delta))
        elif prev_delta.dim() == 1 and self.output_dim > 1:
            prev_delta = prev_delta[:, None].expand(-1, self.output_dim)
        if not torch.is_tensor(log_return):
            log_return = torch.full_like(spots, float(log_return))
        features = self._features(spots, log_return, variance, prev_delta, step)
        delta = self._delta_from_features(step, features)
        return delta.squeeze(-1) if self.output_dim == 1 else delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
