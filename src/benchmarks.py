import math

import torch

from .portfolio import hedge_pnl


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_delta(
    S: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    tau: torch.Tensor | float,
) -> torch.Tensor:
    # Black-Scholes delta of a European call
    S = torch.as_tensor(S, dtype=torch.float32)
    tau = torch.as_tensor(tau, dtype=S.dtype, device=S.device)
    tau = torch.clamp(tau, min=1e-8)
    sigma = max(float(sigma), 1e-8)
    sqrt_tau = torch.sqrt(tau)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    return _norm_cdf(d1)


def bs_call_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    tau: float,
) -> float:
    # Zero rate safe Black-Scholes call price
    if tau <= 0.0:
        return max(float(S) - float(K), 0.0)
    sigma = max(float(sigma), 1e-8)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(float(S) / float(K)) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    ncdf = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(S) * ncdf(d1) - float(K) * math.exp(-r * tau) * ncdf(d2)


def leland_volatility(sigma: float, cost_rate: float, dt: float) -> float:
    # Leland volatility adjusted for proportional costs
    if cost_rate <= 0.0:
        return float(sigma)
    if sigma <= 0.0 or dt <= 0.0:
        raise ValueError("sigma and dt must be positive")
    adjustment = math.sqrt(2.0 / math.pi) * cost_rate / (sigma * math.sqrt(dt))
    return float(sigma * math.sqrt(1.0 + adjustment))


def leland_delta(
    S: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    tau: torch.Tensor | float,
    cost_rate: float,
    dt: float,
) -> torch.Tensor:
    # Black-Scholes delta evaluated at the Leland volatility
    return bs_delta(S, K, r, leland_volatility(sigma, cost_rate, dt), tau)


class BenchmarkRunner:
    # Evaluates analytic hedging policies on a shared path set
    def __init__(
        self,
        spots: torch.Tensor,
        K: float,
        r: float,
        sigma: float,
        cost_rate: float,
        dt: float,
        n_steps: int,
    ) -> None:
        if spots.dim() != 2:
            raise ValueError("spots must be a 2D tensor")
        if spots.shape[1] != n_steps + 1:
            raise ValueError("spots must have n_steps + 1 columns")
        self.spots = spots
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.cost_rate = float(cost_rate)
        self.dt = float(dt)
        self.n_steps = int(n_steps)

    def _taus(self) -> torch.Tensor:
        steps = torch.arange(self.n_steps, device=self.spots.device, dtype=self.spots.dtype)
        return (self.n_steps - steps) * self.dt

    def _run(self, delta_fn) -> dict[str, torch.Tensor]:
        taus = self._taus()
        deltas = delta_fn(self.spots[:, :-1], taus)
        result = hedge_pnl(self.spots, deltas, self.K, self.cost_rate)
        result["taus"] = taus
        return result

    def run_bs(self) -> dict[str, torch.Tensor]:
        return self._run(lambda S, tau: bs_delta(S, self.K, self.r, self.sigma, tau))

    def run_leland(self) -> dict[str, torch.Tensor]:
        return self._run(
            lambda S, tau: leland_delta(
                S, self.K, self.r, self.sigma, tau, self.cost_rate, self.dt
            )
        )

    def run_policy(self, policy) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            deltas = policy(self.spots)
        return hedge_pnl(self.spots, deltas, self.K, self.cost_rate)
