import math

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


def build_simulator(config) -> GBMSimulator | HestonSimulator:
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
    return simulator.simulate(n_paths, seed=seed), None
