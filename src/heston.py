import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    # Paper Section 5.2.1 parameter set
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 2.0
    rho: float = -0.7
    v0: float = 0.04

    def validate(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.xi < 0.0:
            raise ValueError("xi must be non-negative")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        if self.v0 <= 0.0:
            raise ValueError("v0 must be positive")

    @property
    def feller(self) -> float:
        # 2 kappa theta minus xi squared, negative means the CIR process can hit zero
        return 2.0 * self.kappa * self.theta - self.xi**2


# Truncation is set by the integrand decay, 200 was too short and biased the price by 1e-3
DEFAULT_NODES = 512
DEFAULT_UPPER = 800.0


def gauss_legendre(n_nodes: int, upper: float) -> tuple[np.ndarray, np.ndarray]:
    # Nodes and weights on [0, upper]
    nodes, weights = np.polynomial.legendre.leggauss(int(n_nodes))
    return 0.5 * upper * (nodes + 1.0), 0.5 * upper * weights


def _char_exponent(
    u: np.ndarray,
    tau: float,
    params: HestonParams,
) -> tuple[np.ndarray, np.ndarray]:
    # Riccati solution in the Albrecher little trap form, numerically stable for large u
    iu = 1j * u
    sigma2 = params.xi**2
    root = np.sqrt((params.rho * params.xi * iu - params.kappa) ** 2 + sigma2 * (iu + u**2))
    # Choose the branch with positive real part so the exponential decays
    root = np.where(root.real < 0.0, -root, root)
    num = params.kappa - params.rho * params.xi * iu - root
    den = params.kappa - params.rho * params.xi * iu + root
    g = num / den
    exp_dt = np.exp(-root * tau)
    log_term = (params.kappa - params.rho * params.xi * iu - root) * tau - 2.0 * np.log(
        (1.0 - g * exp_dt) / (1.0 - g)
    )
    c = (params.kappa * params.theta / sigma2) * log_term
    d = (num / sigma2) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return c, d


def characteristic_function(
    u: np.ndarray,
    tau: float,
    variance: np.ndarray | float,
    params: HestonParams,
) -> np.ndarray:
    # Zero rate log characteristic function of log S_T
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    c, d = _char_exponent(np.asarray(u, dtype=np.complex128), float(tau), params)
    var = np.asarray(variance, dtype=np.float64)[..., None]
    return np.exp(c[None, :] + d[None, :] * var)


def heston_call_price(
    tau: np.ndarray | float,
    spot: np.ndarray | float,
    variance: np.ndarray | float,
    strike: float,
    params: HestonParams,
    n_nodes: int = DEFAULT_NODES,
    upper: float = DEFAULT_UPPER,
) -> np.ndarray:
    # Lewis single integral with the u - i/2 shift, zero rate
    params.validate()
    tau_arr, spot_arr, var_arr = np.broadcast_arrays(
        np.asarray(tau, dtype=np.float64),
        np.asarray(spot, dtype=np.float64),
        np.clip(np.asarray(variance, dtype=np.float64), 1e-12, None),
    )
    shape = tau_arr.shape
    tau_flat = tau_arr.reshape(-1)
    spot_flat = spot_arr.reshape(-1)
    var_flat = var_arr.reshape(-1)

    nodes, weights = gauss_legendre(n_nodes, upper)
    kernel = weights / (nodes**2 + 0.25)

    price = np.empty_like(spot_flat)
    # Group by unique tau so the Riccati solve is shared across a maturity slice
    for value in np.unique(tau_flat):
        mask = tau_flat == value
        if float(value) <= 0.0:
            price[mask] = np.clip(spot_flat[mask] - float(strike), 0.0, None)
            continue
        phi = characteristic_function(nodes - 0.5j, float(value), var_flat[mask], params)
        # Lewis phase is exp(+i u log(S / K))
        log_forward = np.log(spot_flat[mask] / float(strike))
        phase = np.exp(1j * nodes[None, :] * log_forward[:, None])
        integral = np.sum((phase * phi).real * kernel[None, :], axis=1)
        forward = np.sqrt(spot_flat[mask] * float(strike))
        price[mask] = spot_flat[mask] - forward * integral / math.pi

    return np.clip(price, 0.0, None).reshape(shape)


def black_scholes_call_price(
    tau: float,
    spot: np.ndarray | float,
    variance: float,
    strike: float,
) -> np.ndarray:
    # Zero rate Black-Scholes reference price
    spot_arr = np.asarray(spot, dtype=np.float64)
    if tau <= 0.0:
        return np.clip(spot_arr - strike, 0.0, None)
    sigma = math.sqrt(max(float(variance), 1e-12))
    sqrt_tau = math.sqrt(tau)
    d1 = (np.log(spot_arr / strike) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    ncdf = lambda z: 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return spot_arr * ncdf(d1) - strike * ncdf(d2)


def _difference(
    values: np.ndarray,
    grid: np.ndarray,
    axis: int,
    lower_index: int = 0,
) -> np.ndarray:
    # Central differences, one sided at the lower boundary where the domain is truncated
    if values.shape[axis] != grid.size:
        raise ValueError("grid length must match the differentiated axis")
    if grid.size < 2:
        raise ValueError("need at least two grid points to differentiate")
    out = np.gradient(values, grid, axis=axis)
    first = [slice(None)] * values.ndim
    first[axis] = lower_index
    second = [slice(None)] * values.ndim
    second[axis] = lower_index + 1
    step = float(abs(grid[lower_index + 1] - grid[lower_index]))
    out[tuple(first)] = (values[tuple(second)] - values[tuple(first)]) / step
    return out


class ModelHedge:
    # Analytic Heston hedge in the stock and the variance swap
    def __init__(
        self,
        params: HestonParams,
        strike: float,
        maturity: float,
        n_steps: int,
        s_grid: np.ndarray,
        v_grid: np.ndarray,
        n_nodes: int = DEFAULT_NODES,
        upper: float = DEFAULT_UPPER,
    ) -> None:
        self.params = params
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.n_steps = int(n_steps)
        self.dt = self.maturity / self.n_steps
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        self.v_grid = np.asarray(v_grid, dtype=np.float64)
        self.n_nodes = int(n_nodes)
        self.upper = float(upper)
        if self.s_grid.size < 3 or self.v_grid.size < 3:
            raise ValueError("grids must hold at least three points")
        self._build()

    def _build(self) -> None:
        # Price on the grid at every decision step, then differentiate once
        n_steps = self.n_steps
        self.delta_stock = np.zeros((n_steps, self.s_grid.size, self.v_grid.size))
        self.delta_swap = np.zeros((n_steps, self.s_grid.size, self.v_grid.size))
        s_mesh, v_mesh = np.meshgrid(self.s_grid, self.v_grid, indexing="ij")
        for k in range(n_steps):
            tau = self.maturity - k * self.dt
            price = heston_call_price(
                tau, s_mesh, v_mesh, self.strike, self.params, self.n_nodes, self.upper
            )
            price = price.reshape(s_mesh.shape)
            ds = _difference(price, self.s_grid, axis=0, lower_index=1)
            dv = _difference(price, self.v_grid, axis=1, lower_index=1)
            # Sensitivity of the variance swap leg L(t, v) = (v - theta)/kappa (1 - e^{-kappa (T-t)}) + theta (T-t)
            decay = 1.0 - math.exp(-self.params.kappa * tau)
            d_swap_dv = decay / self.params.kappa
            self.delta_stock[k] = ds
            self.delta_swap[k] = dv / max(d_swap_dv, 1e-12)

    def _bilinear(self, table: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Bilinear interpolation with clamping at the grid boundary
        s = np.clip(np.asarray(s, dtype=np.float64), self.s_grid[0], self.s_grid[-1])
        v = np.clip(np.asarray(v, dtype=np.float64), self.v_grid[0], self.v_grid[-1])
        i = np.clip(np.searchsorted(self.s_grid, s, side="right") - 1, 0, self.s_grid.size - 2)
        j = np.clip(np.searchsorted(self.v_grid, v, side="right") - 1, 0, self.v_grid.size - 2)
        s0, s1 = self.s_grid[i], self.s_grid[i + 1]
        v0, v1 = self.v_grid[j], self.v_grid[j + 1]
        ws = (s - s0) / (s1 - s0)
        wv = (v - v0) / (v1 - v0)
        stride = self.v_grid.size
        flat = i * stride + j
        stacked = table.reshape(-1)
        c00 = stacked[flat]
        c01 = stacked[flat + 1]
        c10 = stacked[flat + stride]
        c11 = stacked[flat + stride + 1]
        return c00 * (1 - ws) * (1 - wv) + c10 * ws * (1 - wv) + c01 * (1 - ws) * wv + c11 * ws * wv

    def deltas(self, spots: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Hedge ratios along a path set, shape (paths, n_steps)
        spots = np.asarray(spots, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        if spots.shape[1] != self.n_steps + 1:
            raise ValueError("spots must hold n_steps + 1 columns")
        n_paths = spots.shape[0]
        d1 = np.zeros((n_paths, self.n_steps))
        d2 = np.zeros((n_paths, self.n_steps))
        for k in range(self.n_steps):
            d1[:, k] = self._bilinear(self.delta_stock[k], spots[:, k], variances[:, k])
            d2[:, k] = self._bilinear(self.delta_swap[k], spots[:, k], variances[:, k])
        return d1, d2


def integrated_variance_path(variances: np.ndarray, dt: float) -> np.ndarray:
    # Trapezoid approximation of the running integrated variance
    return np.cumsum(variances[:, :-1] * dt, axis=1)
