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
├── PRELIMINARY_ANALYSIS.md  # Technical results log, including the paper reproduction
├── price_asymptotics.py   # Paper Figure 10, Black-Scholes price asymptotics
├── price_asymptotics_heston.py  # Paper Figure 11, Heston price asymptotics
├── paper_heston.py        # Paper Figures 2, 3, 6, 7, Heston with two instruments
├── recurrent_ablation.py  # Paper Figures 4, 5, recurrent against simple network
├── paper_call_spread.py   # Paper Figures 8, 9, call spread and risk aversion
├── paper_architecture.py  # Paper Section 5.1 architecture fidelity check
├── eda.py                 # Exploratory data analysis over simulators and results
├── aggregate_paper.py     # Console aggregation of the sharded paper runs
├── regen_paper_figures.py # Redraw every paper figure from stored JSON
├── heston_multiseed.py    # Multi-seed Heston cost sweep
├── gbm_cost_multiseed.py  # Multi-seed GBM cost sensitivity
├── shared_vs_timedep.py   # Multi-seed shared against time-dependent weights
├── lambda_multiseed.py    # Multi-seed entropic lambda sweep
├── plot_lambda_frontier.py  # Redraw the lambda frontier from JSON
├── results/               # Main study figures and summaries
├── results/paper/         # Paper reproduction figures and summaries
└── src/
    ├── simulators.py      # GBM, Heston and paper Heston path generators
    ├── heston.py          # Heston characteristic function pricer and model hedge
    ├── network.py         # Semi-recurrent policy, shared or time dependent weights
    ├── portfolio.py       # Terminal wealth, proportional costs, turnover, d instruments
    ├── loss.py            # Entropic risk and CVaR with trainable threshold
    ├── benchmarks.py      # Black-Scholes and Leland deltas, shared path evaluation
    ├── studies.py         # Aggregation, bootstrap and monotone interpolation helpers
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
python main.py --sensitivity-paths 100000 --sensitivity-epochs 20
python heston_multiseed.py 100000 20
python gbm_cost_multiseed.py --paths 100000 --epochs 20
python shared_vs_timedep.py --paths 100000 --epochs 20
python lambda_multiseed.py 0.25,0.5,1.0,2.0,4.0 100000 20
python plot_lambda_frontier.py
```

`plot_lambda_frontier.py` redraws `results/lambda_frontier.png` from the saved
`results/lambda_multiseed.json` without retraining. `gbm_cost_multiseed.py` runs the GBM cost
sensitivity over 5 seeds and writes `results/gbm_cost_multiseed.json` and
`results/cost_sensitivity_multiseed.png`. `shared_vs_timedep.py` runs the weight-sharing comparison over
5 seeds and writes `results/shared_vs_timedep.json`. Both accept `--shards N --shard i` and
`--aggregate 1` in the same way as the paper runners. `merge_shards.py` combines the per shard files of
the four derived studies (`lambda`, `heston`, `gbm_cost`, `shared`) into their final JSON.
`regen_heston_figures.py` retrains one seed 42 Heston model and redraws the three Heston figures from
it, which is the cheapest way to refresh them after a code change.

Every figure that carries a numeric claim writes a JSON sidecar beside itself, so the claim can be
checked without retraining: `policy_curves.json`, `position_dynamics.json`, `learning_curves.json`,
`heston_policy_curves.json` and `heston_diagnostics.json`. The canonical single-seed console output is
persisted in `results/canonical_run.json`, and the superseded BatchNorm study is kept as
`results/batchnorm_multiseed.json` so the normalization comparison can be checked against its original
source.

## Paper reproduction

The scripts below reproduce the paper's own Section 5 numerical experiments. They write only to
`results/paper/` and never touch the existing `results/` figures.

```bash
python price_asymptotics.py --paths 100000 --eval-paths 100000 --epochs 20   # Figure 10
python price_asymptotics_heston.py --paths 100000 --epochs 20                # Figure 11
python paper_heston.py --paths 100000 --eval-paths 200000 --epochs 20        # Figures 2, 3, 6, 7
python recurrent_ablation.py --paths 100000 --epochs 20                      # Figures 4, 5
python paper_call_spread.py --paths 100000 --epochs 20                       # Figures 8, 9
python paper_architecture.py --paths 100000 --epochs 20                      # Section 5.1
python eda.py --paths 200000                                                 # Exploratory analysis
python aggregate_paper.py                                                    # Console summary
python regen_paper_figures.py                                                # Redraw from JSON only
```

These runs are long. Each takes roughly 9 minutes per training at 100,000 paths and 20 epochs on CPU,
and the full grid is 30 trainings for Figure 10, 30 for Figure 11, 10 for the Heston experiment, 20 for
the ablation, 10 for the call spread and 10 for the architecture check. Every script accepts
`--shards N --shard i` to split the work across processes, `--raw-out` to redirect the per shard
records, and `--aggregate 1` to rebuild the final JSON and figures from the shard files without
retraining. `regen_paper_figures.py` redraws every figure from the stored JSON alone.

| File | Paper experiment |
| --- | --- |
| `price_asymptotics_bs.png` | Figure 10, Black-Scholes price asymptotics. Fitted slope 0.6898, 95 percent CI [0.6362, 0.7353], contains 2/3 |
| `cost_sensitivity_5seed.png` | The same cost grid as CVaR for all three strategies |
| `heston_fig2.png` | Figure 2, model hedge against deep hedge at alpha 0.5 |
| `heston_fig3.png` | Figure 3, stock holding on a spot and variance grid, with a difference panel and variance slices. Surfaces are stored in `heston_fig3.json` so the figure can be redrawn without retraining |
| `heston_fig6.png` | Figure 6, alpha 0.5 against alpha 0.99, each centred at its own risk measure |
| `heston_fig7.png` | Figure 7, both centred at the model price |
| `price_asymptotics_heston.png` | Figure 11, Heston price asymptotics. Fitted slope 0.6017 |
| `recurrent_ablation.png` | Figures 4 and 5, recurrent against simple network |
| `call_spread.png` | Figures 8 and 9, call spread and risk aversion. The alpha 0.99 policy collapses to a near constant position and is worse on every risk measure, so this experiment does not reproduce |
| `paper_architecture.png` | Section 5.1 fidelity comparison |
| `eda_market.png`, `eda_moments.png`, `eda_results.png` | Exploratory analysis of the simulators, the payoff tails and the seed noise against the reported gaps |

## Differences from the paper

| Aspect | Paper | This project |
| --- | --- | --- |
| Architecture | Separate network per step, two hidden layers of $d + 15$ units, ReLU, BatchNorm | Shared weights by default, 3 layers of 64 units, LeakyReLU, LayerNorm. A paper-faithful variant is available via `feature_mode = "paper"` and is compared in `results/paper/paper_architecture.png` |
| Learning rate | 0.005 | 0.001 by default, 0.005 in the paper-faithful variant |
| Network inputs | $\log S$ and the previous position | Six engineered features for the default; $\log S$, variance and the previous position for the paper-faithful variant |
| Normalisation | BatchNorm | LayerNorm by default, because per-step BatchNorm caused a train and evaluation mismatch. BatchNorm is retained behind `--norm batchnorm` |
| Heston scheme | Exact variance sampling plus a simplified Broadie-Kaya stock step | The original study uses Euler-Maruyama with full truncation; `PaperHestonSimulator` implements the paper's scheme exactly for the reproduction |
| Heston parameters | $\kappa = 1$, $\theta = 0.04$, $\xi = 2$, $\rho = -0.7$, $v_0 = 0.04$ | The original study uses $\kappa = 2$, $\xi = 0.3$; the paper's set is used for the reproduction |
| Instruments | Stock plus variance swap in Section 5.2 | Stock only in the original study; stock plus variance swap in the reproduction |
| Test set size | $10^6$ paths | $10^5$ paths per seed over 5 seeds for the reproduction, $10^5$ for the main study |
| Leland benchmark | Not used | Leland (1985) adjusted volatility. The adjustment is one way; a round trip is charged twice, which the proportional cost term already reflects |

## Reproduced paper results

Mean over 5 seeds (42 to 46), 100,000 training paths and 20 epochs, on the paper's own parameter sets.

| Quantity | Paper | This project | Difference |
| --- | --- | --- | --- |
| Heston model price $q$ | 1.69 | 1.6918 | +0.0018, the paper's rounding |
| Heston $p_\theta$ at $\alpha = 0.50$ | 1.94 | 1.9630 ± 0.0053 | +0.0230, 1.2 percent high |
| Heston $p_\theta$ at $\alpha = 0.99$ | 3.49 | 2.9750 ± 0.0740 | -0.5150, 14.8 percent low |
| Figure 10 fitted log-log slope | 2/3 | 0.6898, 95 percent CI [0.6362, 0.7353] | Consistent with 2/3 |
| Figure 11 fitted log-log slope | 2/3 | 0.6017 | Close, 10 percent below |
| Black-Scholes price $q$ | not printed | 2.2872 | Matches `bs_call_price` exactly |
| Learned price at zero cost $p_0$ | not printed | 2.3602 ± 0.0005 | Exceeds $q$, as Proposition 3.6(ii) requires |

## Multi-seed study

The headline results table in `PRELIMINARY_ANALYSIS.md` is based on a 5-seed study, not a single run.
`python main.py --multi-seed` trains and evaluates the main configuration once per seed (42, 43, 44,
45, 46) at the full path budget, then reports mean and standard deviation across seeds for mean PnL,
standard deviation, 95 and 99 percent CVaR, entropic risk and mean transaction cost, for Deep Hedging,
Black-Scholes and Leland. The study writes `results/multiseed_layernorm.json` and does not touch the
canonical single-seed figures in `results/`. Use `--seeds` to override the seed list and
`--multi-seed-output` to redirect the JSON.

Every saved study records its run metadata under a `config` key, including the path budget, epoch
count and normalization. Check that block before quoting numbers from a JSON file, since running a
reduced-budget smoke test against a results path will overwrite it.

## Normalization

The policy uses LayerNorm by default. Per-step BatchNorm was the original choice but it normalizes
with batch statistics during training and with lagged running statistics at evaluation, and the two
diverge as the policy sharpens. That mismatch inflated the seed-to-seed standard deviation of the risk
measures by a factor of 5 to 59 depending on the metric, and produced a 0.385 train and evaluation gap
on one seed. LayerNorm has no running statistics, so both modes are identical by construction. Pass
`--norm batchnorm` to reproduce the earlier behaviour.

## Outputs

The run prints a comparison table with mean PnL, standard deviation, 95 and 99 percent CVaR and
drawdown, then writes figures to `results/`. Drawdown is reported twice: a terminal-jump version that
includes the maturity liability, and an intra-horizon version computed on the pre-settlement path,
which is the interpretable one because the terminal version is dominated by the common payoff.

| File | Content |
| --- | --- |
| `pnl_distributions.png` | Terminal PnL histograms for Deep Hedging, Black-Scholes and Leland |
| `policy_curves.png` | On-path position, off-path probe and the no-transaction band |
| `cost_sensitivity.png` | CVaR at 95 and 99 percent across cost levels, canonical seed 42 |
| `cost_sensitivity_multiseed.png` | The same cost study over 5 seeds with one standard deviation error bars |
| `training_loss.png` | Training risk by epoch |
| `learning_curves.png` | PnL distribution function and risk profile across confidence levels |
| `position_dynamics.png` | Delta trajectories and position against moneyness over time |
| `pnl_paths.png` | Cumulative PnL paths and per strategy bands |
| `heston_cost_sensitivity.png` | Heston CVaR at 95 and 99 percent across cost levels |
| `heston_policy_curves.png` | Heston position surface, variance difference, and the no-trade band |
| `heston_diagnostics.png` | Heston variance paths and the policy response to volatility |
| `lambda_frontier.png` | Risk return frontier, computed points with a fitted curve |

## Notes

- Shared weight mode keeps one linear layer per hidden layer but per-step batch normalization
  statistics, so time dependence enters through normalization while the body stays shared.
- The BatchNorm path uses a variance floor. Constant features occur at the first step where every
  path starts at the same spot, and without the floor the running variance would be zero and the
  policy would saturate. Degenerate inputs fall back to the floored running statistics in both
  training and evaluation, so the first position stays well defined and gradient magnitudes are not
  amplified by the normalization epsilon. This path is not the default.
- When the CVaR tail is very short, the batch is raised automatically so the tail holds at least
  `min_tail_samples` paths. Without this a 99 percent CVaR on 256 paths sees about 2.5 samples. The
  warm start and the per-epoch threshold projection share `var_threshold`, which returns the alpha
  quantile of losses and is the exact Rockafellar-Uryasev minimizer for a fixed sample.
- Runs are reproducible from `seed` alone. Pass `--deterministic` to also force deterministic
  kernels; this is off by default because it restricts some optimized CPU and CUDA paths.
- With zero transaction costs the learned policy converges close to the Black-Scholes delta, which is
  the expected limiting behaviour of the objective.
