import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Market and contract
    S0: float = 100.0
    K: float = 100.0
    T: float = 30.0 / 365.0
    n_steps: int = 30
    r: float = 0.0
    sigma: float = 0.2

    # Frictions
    cost_rate: float = 0.01

    # Training
    n_paths: int = 100_000
    batch_size: int = 256
    epochs: int = 20
    lr: float = 0.001
    lambda_entropic: float = 1.0
    loss_type: str = "entropic"
    cvar_alpha: float = 0.95
    min_tail_samples: int = 20
    grad_clip: float = 5.0
    w_lr: float = 0.01

    # Network
    hidden_dim: int = 64
    n_layers: int = 3
    time_dependent: bool = False
    norm_type: str = "layernorm"
    max_position: float = 2.0

    # Simulator
    simulator: str = "gbm"
    heston_v0: float = 0.04
    heston_kappa: float = 2.0
    heston_theta: float = 0.04
    heston_xi: float = 0.3
    heston_rho: float = -0.7
    antithetic: bool = True

    # Evaluation
    n_eval_paths: int = 100_000
    seed: int = 42
    deterministic: bool = False
    device: str = "cpu"
    results_dir: str = "results"

    # Cost sensitivity study
    run_cost_sensitivity: bool = True
    cost_levels: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01)
    sensitivity_paths: int = 20_000
    sensitivity_epochs: int = 8

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    def effective_batch_size(self, n_paths: int | None = None) -> int:
        # Raise the batch so the CVaR tail holds enough samples
        available = self.n_paths if n_paths is None else int(n_paths)
        if self.loss_type != "cvar":
            return max(1, min(self.batch_size, available))
        tail = max(1.0 - self.cvar_alpha, 1e-6)
        required = int(math.ceil(self.min_tail_samples / tail))
        return max(1, min(max(self.batch_size, required), available))

    def tail_sample_size(self, n_paths: int | None = None) -> float:
        # Expected number of CVaR tail observations per batch
        batch = self.effective_batch_size(n_paths)
        return batch * (1.0 - self.cvar_alpha)

    @property
    def base_variance(self) -> float:
        return self.heston_v0 if self.simulator == "heston" else self.sigma**2

    @property
    def benchmark_sigma(self) -> float:
        return math.sqrt(self.base_variance)

    @property
    def results_path(self) -> Path:
        return Path(__file__).resolve().parent / self.results_dir

    def validate(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be positive")
        if self.T <= 0.0:
            raise ValueError("T must be positive")
        if self.S0 <= 0.0 or self.K <= 0.0:
            raise ValueError("S0 and K must be positive")
        if self.n_paths < 1 or self.batch_size < 1:
            raise ValueError("n_paths and batch_size must be positive")
        if self.n_eval_paths < 1:
            raise ValueError("n_eval_paths must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.hidden_dim < 1 or self.n_layers < 1:
            raise ValueError("hidden_dim and n_layers must be positive")
        if self.lr <= 0.0 or self.w_lr < 0.0:
            raise ValueError("lr must be positive and w_lr non-negative")
        if self.grad_clip < 0.0:
            raise ValueError("grad_clip must be non-negative")
        if self.min_tail_samples < 1:
            raise ValueError("min_tail_samples must be positive")
        if self.cost_rate < 0.0:
            raise ValueError("cost_rate must be non-negative")
        if self.lambda_entropic <= 0.0:
            raise ValueError("lambda_entropic must be positive")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie in (0, 1)")
        if self.loss_type not in {"entropic", "cvar"}:
            raise ValueError("loss_type must be 'entropic' or 'cvar'")
        if self.simulator not in {"gbm", "heston"}:
            raise ValueError("simulator must be 'gbm' or 'heston'")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.heston_v0 <= 0.0 or self.heston_theta <= 0.0:
            raise ValueError("Heston variances must be positive")
        if self.heston_xi < 0.0:
            raise ValueError("heston_xi must be non-negative")
        if not -1.0 <= self.heston_rho <= 1.0:
            raise ValueError("heston_rho must lie in [-1, 1]")
        if self.max_position <= 0.0:
            raise ValueError("max_position must be positive")
        if self.norm_type not in {"layernorm", "batchnorm"}:
            raise ValueError("norm_type must be 'layernorm' or 'batchnorm'")
        if not self.results_dir:
            raise ValueError("results_dir must not be empty")
        if self.run_cost_sensitivity:
            if self.sensitivity_paths < 1 or self.sensitivity_epochs < 1:
                raise ValueError("sensitivity_paths and sensitivity_epochs must be positive")
            if not self.cost_levels:
                raise ValueError("cost_levels must not be empty")
            if any(level < 0.0 for level in self.cost_levels):
                raise ValueError("cost_levels must be non-negative")
