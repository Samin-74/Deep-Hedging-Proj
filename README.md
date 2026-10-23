# Deep Hedging

PyTorch replication of Buehler, Gonon, Teichmann and Wood, "Deep Hedging", Quantitative Finance 2019.

A neural network learns a semi-recurrent hedging policy for a short European call under proportional
transaction costs, optimising a convex risk measure of terminal profit and loss. All market data is
generated synthetically by Monte Carlo, so the project is fully self-contained.

## Layout

```
deep_hedging_project/
├── config.py              # Configuration dataclass, defaults and validation
├── main.py                # Training, evaluation, reporting and plotting entry point
├── requirements.txt
└── src/
    ├── simulators.py      # GBM and Heston path generators
    ├── network.py         # Semi-recurrent policy, shared or time dependent weights
    ├── portfolio.py       # Terminal wealth, proportional costs, turnover
    ├── loss.py            # Entropic risk and CVaR with trainable threshold
    ├── benchmarks.py      # Black-Scholes and Leland deltas, shared path evaluation
    ├── diagnostics.py     # Position response, cost drag and tail attribution
    └── visualize.py       # High resolution figures written to results/
```

## Method

State at decision step $k$ is the spot $S_k$, the realized log return, the variance $v_k$ (Heston)
and the previous position $\delta_{k-1}$. The policy is

$$\delta_k = F^{\theta_k}(I_k, \delta_{k-1}), \qquad \delta_k = \bar{\delta}\tanh(\cdot)$$

Terminal wealth for the short call position is

$$X_T = -Z + \sum_{k=0}^{n-1}\delta_k (S_{k+1} - S_k) - c\sum_{k=0}^{n-1} S_k |\delta_k - \delta_{k-1}|,
\qquad Z = \max(S_T - K, 0)$$

Two convex risk measures are supported. Entropic risk is computed as

$$L(\theta) = \frac{1}{\lambda}\log \mathbb{E}\left[e^{-\lambda X_T}\right]$$

and CVaR uses the Rockafellar and Uryasev representation with a trainable threshold

$$L(\theta) = \inf_{w}\left[w + \frac{1}{1-\alpha}\mathbb{E}\left[(-X_T - w)^{+}\right]\right]$$

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Useful overrides:

```bash
python main.py --simulator heston --loss cvar --cvar-alpha 0.99
python main.py --cost 0.005 --epochs 40 --time-dependent
python main.py --no-sensitivity --results-dir results_fast
python main.py --seed 7 --deterministic
python main.py --multi-seed
```

## Multi-seed study

The headline results table in `PRELIMINARY_ANALYSIS.md` is based on a 5-seed study, not a single run.
`python main.py --multi-seed` trains and evaluates the main configuration once per seed (42, 43, 44,
45, 46) at the full path budget, then reports mean and standard deviation across seeds for mean PnL,
standard deviation, 95 and 99 percent CVaR, entropic risk and mean transaction cost, for Deep Hedging,
Black-Scholes and Leland. The study writes `results/multiseed_summary.json` and does not touch the
canonical single-seed figures in `results/`. Use `--seeds` to override the seed list and
`--multi-seed-output` to redirect the JSON.

## Outputs

The run prints a comparison table with mean PnL, standard deviation, 95 and 99 percent CVaR and
maximum drawdown, then writes figures to `results/`:

| File | Content |
| --- | --- |
| `pnl_distributions.png` | Terminal PnL histograms for Deep Hedging, Black-Scholes and Leland |
| `policy_curves.png` | On-path position, off-path probe and the no-transaction band |
| `cost_sensitivity.png` | CVaR at 95 and 99 percent across cost levels |
| `training_loss.png` | Training risk by epoch |
| `learning_curves.png` | PnL distribution function and risk profile across confidence levels |
| `position_dynamics.png` | Delta trajectories and position against moneyness over time |
| `pnl_paths.png` | Cumulative PnL paths and per strategy bands |

## Notes

- Shared weight mode keeps one linear layer per hidden layer but per-step batch normalization
  statistics, so time dependence enters through normalization while the body stays shared.
- Batch normalization uses a variance floor. Constant features occur at the first step where every
  path starts at the same spot, and without the floor the running variance would be zero and the
  policy would saturate. Degenerate inputs fall back to the floored running statistics in both
  training and evaluation, so the first position stays well defined and gradient magnitudes are not
  amplified by the normalization epsilon.
- When the CVaR tail is very short, the batch is raised automatically so the tail holds at least
  `min_tail_samples` paths. Without this a 99 percent CVaR on 256 paths sees about 2.5 samples. The
  warm start and the per-epoch threshold projection share `var_threshold`, which returns the alpha
  quantile of losses and is the exact Rockafellar-Uryasev minimizer for a fixed sample.
- Runs are reproducible from `seed` alone. Pass `--deterministic` to also force deterministic
  kernels; this is off by default because it restricts some optimized CPU and CUDA paths.
- With zero transaction costs the learned policy converges close to the Black-Scholes delta, which is
  the expected limiting behaviour of the objective.
