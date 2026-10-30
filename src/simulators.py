import math

import numpy as np
import torch


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _normals(
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    antithetic: bool,
) -> torch.Tensor:
    if not antithetic:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    half = (shape[0] + 1) // 2
    base = torch.randn((half,) + shape[1:], generator=generator, device=device, dtype=dtype)
    return torch.cat([base, -base], dim=0)[: shape[0]]


class GBMSimulator:
    # Geometric Brownian motion path generator
    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        n_steps: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        antithetic: bool = True,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.S0 = float(S0)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.dt = self.T / self.n_steps
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.antithetic = bool(antithetic)

    def simulate(self, n_paths: int, seed: int | None = None) -> torch.Tensor:
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        generator = _generator(self.seed if seed is None else seed, self.device)
        z = _normals(
            (n_paths, self.n_steps), generator, self.device, self.dtype, self.antithetic
        )
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * z
        increments = drift + diffusion
        zeros = torch.zeros(n_paths, 1, device=self.device, dtype=self.dtype)
        log_paths = torch.cat([zeros, increments.cumsum(dim=1)], dim=1)
        return self.S0 * torch.exp(log_paths)


class HestonSimulator:
    # Heston stochastic volatility with full truncation Euler scheme
    def __init__(
        self,
        S0: float,
        r: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        T: float,
        n_steps: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        antithetic: bool = True,
    ) -> None:
        if v0 <= 0.0 or theta <= 0.0:
            raise ValueError("v0 and theta must be positive")
        if xi < 0.0:
            raise ValueError("xi must be non-negative")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.S0 = float(S0)
        self.r = float(r)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.dt = self.T / self.n_steps
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.antithetic = bool(antithetic)

    def simulate(self, n_paths: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        generator = _generator(self.seed if seed is None else seed, self.device)
        z_s = _normals(
            (n_paths, self.n_steps), generator, self.device, self.dtype, self.antithetic
        )
        z_indep = _normals(
            (n_paths, self.n_steps), generator, self.device, self.dtype, self.antithetic
        )
        z_v = self.rho * z_s + math.sqrt(max(1.0 - self.rho**2, 0.0)) * z_indep

        sqrt_dt = math.sqrt(self.dt)
        spots = torch.empty(n_paths, self.n_steps + 1, device=self.device, dtype=self.dtype)
        variances = torch.empty(n_paths, self.n_steps + 1, device=self.device, dtype=self.dtype)
        spots[:, 0] = self.S0
        variances[:, 0] = self.v0

        s = torch.full((n_paths,), self.S0, device=self.device, dtype=self.dtype)
        v = torch.full((n_paths,), self.v0, device=self.device, dtype=self.dtype)
        for k in range(self.n_steps):
            v_pos = torch.clamp(v, min=0.0)
            v_next = (
                v
                + self.kappa * (self.theta - v_pos) * self.dt
                + self.xi * torch.sqrt(v_pos) * sqrt_dt * z_v[:, k]
            )
            v = torch.clamp(v_next, min=0.0)
            s = s * torch.exp(
                (self.r - 0.5 * v_pos) * self.dt + torch.sqrt(v_pos) * sqrt_dt * z_s[:, k]
            )
            spots[:, k + 1] = s
            variances[:, k + 1] = v
        return spots, variances


class PaperHestonSimulator:
    # Paper Section 5.2 scheme: exact CIR step plus the simplified Broadie-Kaya stock step
    def __init__(
        self,
        S0: float,
        T: float,
        n_steps: int,
        kappa: float = 1.0,
        theta: float = 0.04,
        xi: float = 2.0,
        rho: float = -0.7,
        v0: float = 0.04,
        seed: int = 42,
        antithetic: bool = True,
    ) -> None:
        if v0 <= 0.0 or theta <= 0.0:
            raise ValueError("v0 and theta must be positive")
        if xi < 0.0:
            raise ValueError("xi must be non-negative")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.S0 = float(S0)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.dt = self.T / self.n_steps
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.v0 = float(v0)
        self.seed = int(seed)
        self.antithetic = bool(antithetic)

    @property
    def feller(self) -> float:
        return 2.0 * self.kappa * self.theta - self.xi**2

    def _draw(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        # Antithetic pairing keeps the sample mean of the normals close to zero
        if not self.antithetic:
            return rng.standard_normal(shape)
        half = (shape[0] + 1) // 2
        base = rng.standard_normal((half,) + shape[1:])
        return np.concatenate([base, -base], axis=0)[: shape[0]]

    def simulate(
        self,
        n_paths: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        z_s = self._draw(rng, (n_paths, self.n_steps))
        z_indep = self._draw(rng, (n_paths, self.n_steps))
        z_v = self.rho * z_s + math.sqrt(max(1.0 - self.rho**2, 0.0)) * z_indep

        # Exact CIR transition: V_{k+1} given V_k is c times a noncentral chi square
        scale = self.xi**2 * (1.0 - math.exp(-self.kappa * self.dt)) / (4.0 * self.kappa)
        df = 4.0 * self.kappa * self.theta / self.xi**2
        decay = math.exp(-self.kappa * self.dt)
        drift = 1.0 - decay

        variances = np.empty((n_paths, self.n_steps + 1), dtype=np.float64)
        spots = np.empty((n_paths, self.n_steps + 1), dtype=np.float64)
        variances[:, 0] = self.v0
        spots[:, 0] = self.S0

        log_spot = np.full(n_paths, math.log(self.S0), dtype=np.float64)
        v = np.full(n_paths, self.v0, dtype=np.float64)
        rho_over_xi = self.rho / self.xi
        sqrt_one_minus_rho2 = math.sqrt(max(1.0 - self.rho**2, 0.0))
        for k in range(self.n_steps):
            v_next = scale * rng.noncentral_chisquare(
                df, np.maximum(v * decay / scale, 0.0), size=n_paths
            )
            # Simplified Broadie-Kaya: the variance integral on the step is V_k dt
            log_spot = (
                log_spot
                - 0.5 * v * self.dt
                + rho_over_xi * (v_next - v - self.kappa * (self.theta - v) * self.dt)
                + sqrt_one_minus_rho2 * np.sqrt(np.maximum(v, 0.0) * self.dt) * z_v[:, k]
            )
            v = v_next
            variances[:, k + 1] = v
            spots[:, k + 1] = np.exp(log_spot)
        return spots, variances

    def variance_swap(self, variances: np.ndarray) -> np.ndarray:
        # Running integrated variance plus L(t, V_t), equation 30 and 31
        if variances.ndim != 2 or variances.shape[1] != self.n_steps + 1:
            raise ValueError("variances must have n_steps + 1 columns")
        running = np.zeros_like(variances)
        running[:, 1:] = np.cumsum(variances[:, :-1] * self.dt, axis=1)
        steps = np.arange(self.n_steps + 1)
        tau = self.T - steps * self.dt
        decay = 1.0 - np.exp(-self.kappa * tau)
        # L(t, v) = (v - theta) / kappa (1 - e^{-kappa (T - t)}) + theta (T - t)
        forward = (variances - self.theta) / self.kappa * decay[None, :] + self.theta * tau[None, :]
        return running + forward


def build_simulator(config) -> GBMSimulator | HestonSimulator | PaperHestonSimulator:
    if config.paper_heston or config.simulator == "paper_heston":
        return PaperHestonSimulator(
            S0=config.S0,
            T=config.T,
            n_steps=config.n_steps,
            kappa=config.paper_heston_kappa,
            theta=config.paper_heston_theta,
            xi=config.paper_heston_xi,
            rho=config.paper_heston_rho,
            v0=config.paper_heston_v0,
            seed=config.seed,
            antithetic=config.antithetic,
        )
    if config.simulator == "gbm":
        return GBMSimulator(
            S0=config.S0,
            r=config.r,
            sigma=config.sigma,
            T=config.T,
            n_steps=config.n_steps,
            seed=config.seed,
            device=config.device,
            antithetic=config.antithetic,
        )
    if config.simulator == "heston":
        return HestonSimulator(
            S0=config.S0,
            r=config.r,
            v0=config.heston_v0,
            kappa=config.heston_kappa,
            theta=config.heston_theta,
            xi=config.heston_xi,
            rho=config.heston_rho,
            T=config.T,
            n_steps=config.n_steps,
            seed=config.seed,
            device=config.device,
            antithetic=config.antithetic,
        )
    raise ValueError(f"Unknown simulator: {config.simulator}")


def simulate_paths(config, n_paths: int, seed: int | None = None):
    simulator = build_simulator(config)
    if isinstance(simulator, HestonSimulator):
        return simulator.simulate(n_paths, seed=seed)
    if isinstance(simulator, PaperHestonSimulator):
        spots, variances = simulator.simulate(n_paths, seed=seed)
        # Torch keeps the training loop unchanged, the swap leg is built from variances
        spot_t = torch.as_tensor(spots, dtype=torch.float32)
        var_t = torch.as_tensor(variances, dtype=torch.float32)
        swap = torch.as_tensor(simulator.variance_swap(variances), dtype=torch.float32)
        stacks = torch.stack([spot_t, config.swap_scale * swap], dim=-1)
        return stacks, var_t
    return simulator.simulate(n_paths, seed=seed), None
