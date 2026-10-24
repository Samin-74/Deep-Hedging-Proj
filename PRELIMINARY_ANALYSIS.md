# Preliminary Analysis

Technical results log for the Deep Hedging replication. Working notes only, not the course report.

## Project status

The pipeline is implemented end to end and runs reproducibly from a single seed. A semi-recurrent
policy maps a six dimensional normalized state to a bounded hedge position, trained with Adam on
synthetic Monte Carlo paths under proportional transaction costs. Two simulators (GBM, Heston with
full truncation), two convex risk measures (entropic, CVaR with a Rockafellar-Uryasev threshold) and
two analytic benchmarks (Black-Scholes delta, Leland delta) are wired through one shared PnL engine,
so all three strategies are scored on identical paths and an identical cost structure.

The canonical run is configured as `S0 = K = 100`, `T = 30/365`, `n_steps = 30`, `n_paths = 100000`,
`n_eval_paths = 100000`, `r = 0`, `sigma = 0.2`, `cost_rate = 0.01`, `batch_size = 256`, `epochs = 20`,
`lr = 0.001`, `lambda_entropic = 1.0`, shared weights across time, 3 hidden layers of 64 units,
20,353 policy parameters, seed 42. Training takes about 198 s and the full run about 1030 s on CPU.

The headline results are reported from a 5-seed study (seeds 42, 43, 44, 45, 46). Each seed is a
complete train and evaluate cycle at the same budget, and the main table reports mean $\pm$ sample
standard deviation across the 5 seeds. The study took 989 s and writes
`results/multiseed_layernorm.json`.

Normalization was changed from per-step BatchNorm to LayerNorm after the first 5-seed study exposed a
train and evaluation mismatch. LayerNorm has no running statistics, so the policy evaluated in
training mode and in evaluation mode is identical by construction. This removed the mismatch, cut the
seed-to-seed standard deviation of the risk measures by roughly a factor of 7, and made training about
2.6 times faster. The BatchNorm path is retained behind `--norm batchnorm` for comparison.

Status: methodology and qualitative claims reproduce. Empirical coverage is partial, see the verdict.

## Formal setup

The state at decision step $k$ is the spot $S_k$, the realized log return, the variance $v_k$ (Heston
only) and the previous position $\delta_{k-1}$. The policy is semi-recurrent,

$$\delta_k = F^{\theta_k}(I_k, \delta_{k-1}), \qquad \delta_k = \bar{\delta}\tanh(\cdot)$$

Terminal wealth for the short European call, with proportional transaction costs, is

$$X_T = -Z + \sum_{k=0}^{n-1}\delta_k (S_{k+1} - S_k) - c\sum_{k=0}^{n-1} S_k |\delta_k - \delta_{k-1}|,
\qquad Z = \max(S_T - K, 0)$$

with $\delta_{-1} = 0$, so the initial trade from a flat book is charged. The entropic risk measure is

$$L(\theta) = \frac{1}{\lambda}\log \mathbb{E}\left[e^{-\lambda X_T}\right]$$

and CVaR uses the Rockafellar and Uryasev representation with a trainable threshold,

$$L(\theta) = \inf_{w}\left[w + \frac{1}{1-\alpha}\mathbb{E}\left[(-X_T - w)^{+}\right]\right]$$

The Leland benchmark replaces the volatility by the cost adjusted value

$$\sigma_L = \sigma\sqrt{1 + \sqrt{\frac{2}{\pi}}\,\frac{c}{\sigma\sqrt{\Delta t}}}$$

and evaluates the Black-Scholes delta at $\sigma_L$ instead of $\sigma$.

## Has the Buehler et al. (2019) paper been fully replicated?

**Verdict: Partially.**

### What matches the paper

- The optimization problem is the paper's: minimize a convex risk measure of terminal hedging PnL for
  a short European call, with the liability, the self-financing gains and the proportional cost term
  entering exactly as in the reference formulation.
- Both risk measures are implemented in the paper's form. Entropic risk uses a numerically stable
  log-sum-exp so that `lambda X_T` does not overflow, and CVaR uses the Rockafellar-Uryasev
  representation `w + E[(-X_T - w)^+] / (1 - alpha)` with an explicit threshold.
- The semi-recurrent structure is preserved: the policy observes the previous position
  `delta_{k-1}`, which is what produces the no-transaction region rather than a memoryless delta map.
- Both simulators required by the design are present. GBM is exact in log space. Heston uses
  Euler-Maruyama with full truncation of the variance in both the drift and the diffusion, which is
  the standard scheme for keeping the variance non-negative.
- The proportional cost convention matches: `C_T = c * sum_k S_k |delta_k - delta_{k-1}|` with
  `delta_{-1} = 0`, so the initial trade from a flat book is charged.
- The Leland adjustment `sigma_L = sigma * sqrt(1 + sqrt(2/pi) * c / (sigma * sqrt(dt)))` is
  implemented and produces the expected ordering against Black-Scholes on the same paths.
- The two headline qualitative predictions reproduce. First, the learned policy sits inside the
  Black-Scholes delta away from the boundaries, which is the discrete-time no-transaction band.
  Second, the advantage over analytic deltas grows with the cost level.
- The zero-cost limit reproduces at the full budget. With `cost_rate = 0` the learned policy matches
  the Black-Scholes delta to a mean absolute deviation of 0.0071, 0.0105 and 0.0148 at decision steps
  5, 15 and 25, with correlations of 0.9985, 0.9994 and 0.9996, and matches or slightly beats both
  benchmarks on CVaR 95 and CVaR 99.
- The advantage over both benchmarks is statistically unambiguous after the normalization fix. Over 5
  seeds the t ratios against Leland are 40.7 for mean PnL, 42.8 for cost, 62.1 for entropic risk, 19.4
  for CVaR 95 and 12.5 for CVaR 99.

### What is missing or approximated

- No real market data. The paper calibrates to historical index returns and evaluates out of sample
  on a held-out period. Here every path comes from a simulator with hand-set parameters, so the
  absolute PnL levels have no empirical meaning and only relative comparisons are informative.
- No calibration step. The Heston parameters (`v0 = theta = 0.04`, `kappa = 2`, `xi = 0.3`,
  `rho = -0.7`) are plausible but not fitted to anything, so the Heston results test robustness of
  the method, not a market scenario.
- The theoretical content of the paper is not reproduced. There is no analysis of the approximation
  error of the policy class, no convergence argument, and no sensitivity of the risk measure to the
  network capacity beyond the empirical comparison of shared and time-dependent weights.
- The architecture is a reasonable analogue, not the paper's exact network. The paper reports a
  feedforward body with a recurrent pass of the previous position; here the shared variant shares all
  linear weights across time but keeps per-step normalization statistics, and the time-dependent
  variant trains a separate MLP per step.
- Normalization is LayerNorm rather than the paper's BatchNorm. This was a deliberate change, made
  because per-step BatchNorm produced a train and evaluation mismatch that inflated the seed variance
  by a factor of 7. The BatchNorm path is retained behind `--norm batchnorm` for comparison.
- Only a European call is hedged. The paper's framework covers path-dependent and exotic payoffs, and
  the nested-simulation construction needed for those is absent.
- One friction model only. Proportional costs are implemented; no quadratic impact, no bid-ask
  spread decomposition, no borrowing or funding costs.

## Figure-by-figure analysis

All figures below are the current contents of `results/`, produced by the canonical single-seed run at
seed 42 described above, so every number quoted in this section is a seed 42 number and not a
multi-seed average. Where a figure number differs from the multi-seed table, the multi-seed table is
the authoritative estimate and the figure shows one realization of it.

### `pnl_distributions.png`

Three overlaid terminal PnL histograms on 100,000 evaluation paths, with each strategy's mean marked
by a dashed vertical line and the legend reporting mean and 95 percent CVaR (loss basis).

The learned policy shifts the whole distribution to the right rather than reshaping its tails: mean
PnL improves from -4.51 (Black-Scholes) and -4.28 (Leland) to -3.32, and CVaR 95 falls from 6.36 to
4.90. The Black-Scholes and Leland histograms are concentrated in a narrow band near -4.5 to -4.3,
which is the deterministic cost drag, while the deep hedging histogram is visibly wider and reaches
into the region above -1.0. This is the expected signature of a policy that trades less but carries
more residual risk: it buys a better mean at the price of higher dispersion, since standard deviation
rises from 0.757 to 0.776. Both the mean shift and the dispersion increase are stable features: over 5
seeds the learned policy's mean PnL is -3.252 $\pm$ 0.056 and its standard deviation is
0.823 $\pm$ 0.044.

This matches the paper's claim. The objective is a risk measure of terminal PnL under frictions, and
the optimum sacrifices some variance reduction to reduce cost. It also shows why reporting only the
mean would be misleading, since the improvement is a genuine shift of the loss distribution and not a
tail-only effect.

### `policy_curves.png`

Three panels at decision step 15 of 30. Left: the on-path hedge position binned by spot, with the 10
to 90 percent band, against the Black-Scholes and Leland deltas. Middle: an off-path probe sweeping
spot for three fixed previous positions. Right: the implied no-trade band, the trade size against
spot for the same three previous positions, with a shaded tolerance band of 0.01.

The left panel is the central result of the project. For the realized distribution of spot at step 15
(roughly 92 to 108) the learned delta is monotone in spot but lies below the Black-Scholes delta on
the upside and above it on the downside, so the curve is flatter: at spot 96 the policy holds about
0.43 against a Black-Scholes delta near 0.16, and at spot 106 it holds about 0.78 against 0.93. The
middle panel makes the mechanism explicit. For a previous position of plus 0.40 the position barely
moves for spot between 92 and 100, and the trade size in the right panel is essentially zero over
that interval. That interval is the no-transaction region, and it is asymmetric: it is wide when the
existing position is already close to the target and collapses when the position is far from the
money.

This matches the paper directly. The existence of a non-degenerate no-trade band is the defining
qualitative prediction of the discrete-time frictional problem, and a memoryless delta map cannot
produce it.

Caveat: the middle panel is an off-path probe. Far from the money and at extreme previous positions
the network is evaluated outside the training distribution, so the level of the probe curves is less
trustworthy than their shape. The left panel, computed only on realized paths, is the reliable view.

### `cost_sensitivity.png`

95 and 99 percent CVaR of the three strategies at cost levels 0.00, 0.10, 0.50 and 1.00 percent. The
policy is retrained from scratch at every cost level at the full main-run budget, 100,000 paths and 20
epochs, and all strategies are scored on the same 100,000 path set.

Deep hedging is best at every cost level, and the margin grows with cost. At 1.00 percent its 95
percent CVaR is 5.023 against 6.363 for Black-Scholes and 5.539 for Leland. At 0.50 percent it is
4.157 against 4.707 and 4.390. At 0.00 percent it is 3.110 against 3.127 for both analytic deltas, so
it matches them to within 0.02 and is marginally ahead.

The shape matches the paper at both ends. The benefit of learning increases with the friction level,
which is the central qualitative prediction, and at zero cost the learned policy recovers the
Black-Scholes delta rather than underperforming it. The zero-cost delta deviation is 0.007 to 0.015
with correlation 0.9996, so the theoretical limit is reproduced. An earlier version of this figure,
run at 20,000 paths and 8 epochs, showed the learned policy about 0.5 worse at zero cost; that was
under-training and is now resolved.

### `training_loss.png`

Epoch-level training risk over the 20 epochs of the canonical run, with a 3 epoch moving average.

Risk falls from 3.9649 to 3.6185. About 63 percent of the total reduction is achieved by epoch 2 and
about 82 percent by epoch 4, after which progress is slow and slightly noisy: epochs 7, 18 and 19 are
marginally above their predecessors. The curve is monotone in trend at the scale reported, with a
residual spread of about 0.01 over the second half.

This is consistent with the paper's reported behaviour, where the policy improves rapidly and then
flattens. The curve is noticeably smoother than the BatchNorm version, which had a residual spread of
about 0.04 over the same span, because LayerNorm removes the batch-statistic noise from the objective.
The run is converged at 20 epochs in the sense that further training yields little. The absence of a
held-out training curve means overfitting cannot be assessed from this figure alone. Overfitting is
unlikely here because the training set has 100,000 paths against 20,353 parameters, and the earlier
diagnostic showed evaluation risk on fresh paths matching evaluation risk on training paths to within
0.006, but this is an argument rather than a formal test.

### `learning_curves.png`

Left: the empirical distribution function of terminal PnL. Right: expected shortfall at 50, 75, 90,
95 and 99 percent confidence.

The distribution function shows the deep hedging curve lying to the right of both benchmarks over
almost the entire support, which is first order stochastic dominance on the realized sample. The
dominance is strict in the body and narrows in the extreme lower tail below about -7.0, where the
three curves converge, consistent with all strategies facing the same unbounded liability.

The right panel quantifies the same fact across confidence levels: 3.86 against 5.11 (Black-Scholes)
and 4.66 (Leland) at 50 percent, 4.90 against 6.36 and 5.54 at 95 percent, and 5.41 against 7.02 and
6.04 at 99 percent. The ordering and the sizes of the gaps are stable, so the improvement is not an
artifact of one particular confidence level.

This matches the paper's claim that deep hedging improves the full risk profile rather than shifting
risk between quantiles. The convergence of the curves in the far tail is also the expected result for
a short call: no delta policy can remove the unbounded loss when the underlying gaps far above the
strike.

### `position_dynamics.png`

Left: 120 sampled position trajectories through the 30 decision steps, with the mean trajectory of
each strategy. Right: every position against spot relative to the initial level, colored by decision
step, over 1,500 sampled paths.

The mean trajectory starts at about 0.40, rises to roughly 0.50 within four steps and then stays flat
near 0.50 to the end. The Black-Scholes and Leland mean deltas are essentially constant at 0.51 over
the whole horizon, which is correct at the money with `r = 0` since the delta of an at-the-money call
is close to 0.5 until the last step. The learned mean sits slightly below them, and the small gap is
the average amount of hedging the policy declines to do.

The right panel shows the position is a smooth monotone function of spot with almost no residual
dependence on the decision step: the color gradient runs along the curve rather than across it. In
other words the learned policy is close to time homogeneous in this GBM setting, which is what the
theory predicts when volatility is constant and the only time dependence comes from the shrinking
time to maturity.

This matches the paper. The individual trajectories show wide dispersion, with paths swinging between
about 0.05 and 1.0, so the flat mean is an average over very heterogeneous behavior rather than a
policy that refuses to trade.

### `pnl_paths.png`

Left: 60 sampled cumulative PnL paths of the learned policy through the decision steps, with the mean
path. Right: the mean path and the 5 to 95 percent band for all three strategies.

Cumulative PnL accumulates slowly and negatively through the horizon and then drops sharply at the
final step, where the terminal liability is booked. The mean path of a fully hedged benchmark declines
steadily, for reference the Black-Scholes mean reaches about -2.15 by step 29 before the payoff is
applied, and the learned policy's mean path stays above it for most of the horizon while its band is
wider, which is the same dispersion trade-off visible in the histogram.

The sharp terminal drop is a property of the accounting convention, not of the hedging: the premium is
not received at inception and the payoff is recognized entirely at maturity, so the plot shows the
financing cost of the hedge before the liability is settled. This matches the paper's PnL definition
and is the reason the meaningful comparison is the terminal distribution rather than the shape of the
intra-horizon path.

## Key numerical results

### Multi-seed study, main results table

The headline numbers are averaged over 5 independent seeds (42, 43, 44, 45, 46). Each seed is a
complete train and evaluate cycle of the main configuration: GBM, entropic objective with
`lambda = 1.0`, `cost_rate = 0.01`, 100,000 training paths, 100,000 evaluation paths, 20 epochs. The
reported uncertainty is the sample standard deviation across the 5 seeds, so each cell reads as
mean $\pm$ standard deviation over 5 independent seeds.

| Strategy | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost |
| --- | --- | --- | --- | --- | --- | --- |
| Deep Hedging | -3.252 ± 0.056 | 0.823 ± 0.044 | 4.980 ± 0.063 | 5.542 ± 0.086 | 3.600 ± 0.030 | 0.967 ± 0.054 |
| Black-Scholes | -4.511 ± 0.001 | 0.756 ± 0.001 | 6.352 ± 0.009 | 7.013 ± 0.016 | 4.859 ± 0.004 | 2.225 ± 0.002 |
| Leland | -4.280 ± 0.002 | 0.511 ± 0.001 | 5.529 ± 0.007 | 6.026 ± 0.012 | 4.427 ± 0.002 | 1.994 ± 0.001 |

The two analytic benchmarks are deterministic given the path set, so their standard deviations are
near zero by construction and reflect only the change of evaluation paths across seeds. All of the
seed-to-seed variation in the table is therefore attributable to the learned policy.

Per-seed values for the learned policy, showing where the dispersion comes from:

| Seed | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | -3.315 | 0.776 | 4.895 | 5.410 | 3.613 | 1.029 |
| 43 | -3.266 | 0.787 | 4.944 | 5.501 | 3.586 | 0.980 |
| 44 | -3.207 | 0.848 | 5.001 | 5.616 | 3.580 | 0.928 |
| 45 | -3.291 | 0.818 | 5.061 | 5.601 | 3.646 | 1.001 |
| 46 | -3.181 | 0.885 | 5.000 | 5.585 | 3.574 | 0.898 |

### What the multi-seed study changes

The multi-seed study increases the statistical reliability of the reported gaps. Using the standard
error of the mean across seeds, $\mathrm{SE} = s/\sqrt{5}$, the gaps against Leland are:

| Metric | Gap, Leland minus Deep Hedging | Standard error | t ratio | Reliable |
| --- | --- | --- | --- | --- |
| Mean PnL | +1.027 | 0.025 | 40.7 | yes |
| Mean cost | +1.026 | 0.024 | 42.8 | yes |
| Entropic | +0.827 | 0.013 | 62.1 | yes |
| CVaR 95 | +0.548 | 0.028 | 19.4 | yes |
| CVaR 99 | +0.484 | 0.039 | 12.5 | yes |

Every gap against Leland is now far outside the seed noise, including both CVaR measures. Against
Black-Scholes the t ratios are 49.9 for mean PnL, 52.4 for cost, 94.4 for entropic risk, 48.5 for
CVaR 95 and 38.1 for CVaR 99. The learned policy therefore dominates both analytic benchmarks on
every reported risk measure and on cost, and the dominance is statistically unambiguous.

### Effect of the normalization fix

The first 5-seed study used per-step BatchNorm and produced a large train and evaluation mismatch on
seed 44: the final minibatch training risk was 3.578 while the evaluation risk was 4.134, a gap of
0.55, against 0.013 for seed 42. The cause was that BatchNorm normalizes with batch statistics during
training but with lagged running statistics at evaluation, and the two diverge as the policy sharpens.
Replacing it with LayerNorm, which has no running statistics, makes the two modes identical.

| Metric | BatchNorm mean ± std | LayerNorm mean ± std | Std reduction |
| --- | --- | --- | --- |
| Mean PnL | -3.235 ± 0.043 | -3.252 ± 0.056 | 0.8x |
| Std | 0.973 ± 0.151 | 0.823 ± 0.044 | 3.4x |
| CVaR 95 | 5.272 ± 0.461 | 4.980 ± 0.063 | 7.3x |
| CVaR 99 | 5.932 ± 0.606 | 5.542 ± 0.086 | 7.0x |
| Entropic | 3.749 ± 0.237 | 3.600 ± 0.030 | 7.9x |
| Mean cost | 0.950 ± 0.047 | 0.967 ± 0.054 | 0.9x |

The risk measures became 7 to 8 times more reproducible and their means improved, since the BatchNorm
version was being scored with statistics that did not match its training. Seed 44's entropic risk fell
from 4.134 to 3.580, in line with the other seeds. The residual spread in mean PnL and mean cost is
not a defect: it reflects genuine differences in how much each seed trades, and the two move together
as expected. Training also became about 2.6 times faster, from 500 s to 198 s per seed.

### Single-seed canonical run

The canonical run at seed 42 is retained unchanged and is the source of every figure in `results/`.
It reproduces the seed 42 row of the multi-seed table exactly, which confirms the study and the
figure pipeline are consistent.

| Strategy | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost | Gross PnL | Turnover | Tail cost share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deep Hedging | -3.315 | 0.776 | 4.895 | 5.410 | 3.613 | 1.029 | -2.286 | 15.48 | 0.227 |
| Black-Scholes | -4.512 | 0.757 | 6.363 | 7.016 | 4.861 | 2.224 | -2.288 | 15.01 | 0.526 |
| Leland | -4.281 | 0.512 | 5.539 | 6.039 | 4.429 | 1.993 | -2.288 | 15.11 | 0.495 |

Maximum drawdown, on the path convention of this implementation: worst 26.662 (deep hedging), 26.739
(Black-Scholes), 26.739 (Leland); mean 5.240, 6.121, 5.896. The worst values are dominated by the
terminal payoff jump and are not an informative intra-horizon drawdown, see the limitations.

Policy response at step 15 with previous position 0.508, benchmark reference in brackets:

| Spot move | Spot | Deep Hedging | Black-Scholes | Leland |
| --- | --- | --- | --- | --- |
| -12% | 88.00 | 0.205 | 0.001 | 0.009 |
| -8% | 92.00 | 0.284 | 0.021 | 0.064 |
| -4% | 96.00 | 0.434 | 0.162 | 0.232 |
| -2% | 98.00 | 0.492 | 0.316 | 0.364 |
| 0% | 100.00 | 0.515 | 0.508 | 0.511 |
| +2% | 102.00 | 0.542 | 0.695 | 0.654 |
| +4% | 104.00 | 0.592 | 0.838 | 0.775 |
| +8% | 108.00 | 0.733 | 0.973 | 0.927 |
| +12% | 112.00 | 0.818 | 0.998 | 0.984 |

### Full-budget cost sensitivity

The cost study was re-run at the full main-run budget, 100,000 paths and 20 epochs per cost level,
instead of the earlier reduced 20,000 paths and 8 epochs. The evaluation protocol is unchanged, so the
numbers are directly comparable to the previous table. The study took 831 s.

| Cost | Deep mean | Deep CVaR95 | Deep CVaR99 | BS CVaR95 | BS CVaR99 | Leland CVaR95 | Leland CVaR99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00% | -2.288 | 3.110 | 3.471 | 3.127 | 3.518 | 3.127 | 3.518 |
| 0.10% | -2.448 | 3.342 | 3.673 | 3.430 | 3.849 | 3.390 | 3.791 |
| 0.50% | -2.843 | 4.157 | 4.576 | 4.707 | 5.230 | 4.390 | 4.837 |
| 1.00% | -3.206 | 5.023 | 5.574 | 6.363 | 7.016 | 5.539 | 6.039 |

The zero-cost case is the important one, because it is the limit in which the theory says the optimal
policy is the Black-Scholes delta. At the reduced budget the learned policy was worse than both
benchmarks at zero cost, with CVaR 95 of 3.611 against 3.111. At the full budget it is now slightly
better than both, with CVaR 95 of 3.110 against 3.127 and CVaR 99 of 3.471 against 3.518, and its mean
PnL of -2.288 matches the benchmarks to three decimals. The earlier deficit was under-training, not a
property of the method.

The delta convergence confirms the same conclusion directly. At zero cost, on realized paths, the mean
absolute deviation of the learned position from the Black-Scholes delta is 0.0071 at step 5, 0.0105 at
step 15 and 0.0148 at step 25, with correlations of 0.9985, 0.9994 and 0.9996. The learned policy has
recovered the analytic delta to within about one percent of a position unit.

The advantage over the benchmarks grows monotonically with the cost level, from a CVaR 95 gap of 0.017
at zero cost to 1.340 at one percent, which is the paper's central qualitative prediction.

### Heston transaction cost sweep

The same cost study was run under the Heston simulator, using the existing simulator and PnL engine
with no changes to either. The benchmark volatility is the variance-adjusted value
$\sigma = \sqrt{v_0} = 0.2$, so the analytic deltas are the same Black-Scholes and Leland formulas
evaluated at the long-run variance. Budget: 20,000 paths and 10 epochs per cost level, which is
reduced relative to the GBM study because the Heston path generation and the stochastic variance state
make each epoch more expensive. The study took 402 s.

| Cost | Deep mean | Deep CVaR95 | Deep CVaR99 | BS CVaR95 | BS CVaR99 | Leland CVaR95 | Leland CVaR99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00% | -2.274 | 3.468 | 3.843 | 3.294 | 3.817 | 3.294 | 3.817 |
| 0.10% | -2.431 | 3.478 | 3.878 | 3.603 | 4.163 | 3.562 | 4.103 |
| 0.50% | -2.875 | 4.257 | 4.746 | 4.899 | 5.588 | 4.585 | 5.186 |
| 1.00% | -3.396 | 4.989 | 5.577 | 6.575 | 7.412 | 5.758 | 6.425 |

The pattern matches the GBM study. The learned policy is slightly worse than the benchmarks at zero
cost, by 0.174 in CVaR 95, which is consistent with the reduced training budget rather than with the
simulator. It is better at every positive cost level, and the margin widens with cost: 0.125 at 0.10
percent, 0.328 at 0.50 percent and 0.769 at 1.00 percent against Leland, and 1.586 against
Black-Scholes at 1.00 percent. The zero-cost deficit is larger here than in the GBM study, which is
what the smaller budget predicts.

The Heston results also show that the policy uses the variance state. The learned policy is trained
with the variance path as an input, and it beats the variance-adjusted analytic benchmarks at positive
cost, which a policy ignoring the variance state could not do as consistently. This is a qualitative
observation, not a controlled ablation.

## Remaining limitations and open issues

1. No market data and no calibration. All conclusions are relative, within a synthetic world whose
   parameters were chosen rather than estimated. The absolute PnL values carry no economic meaning.
2. Seed dispersion is now measured but remains large relative to some gaps. The learned policy's
   entropic risk varies by a standard deviation of 0.237 and CVaR 95 by 0.461 across 5 seeds, against
   gaps to Leland of 0.678 and 0.256. Five seeds is enough to establish the entropic, mean PnL and
   cost advantages and not enough to resolve the CVaR advantage, which has a t ratio of 1.2 at 95
   percent and 0.3 at 99 percent. A larger seed count would be needed for the tail measures.
3. The maximum drawdown metric is dominated by the terminal payoff jump because the liability is
   booked in one step at maturity. The worst-case column is effectively a rescaled maximum payoff and
   should not be read as a drawdown. The mean column is more informative but still conflates the cost
   drag with the terminal settlement.
4. The Heston cost sweep uses 20,000 paths and 10 epochs, below the GBM study. Its zero-cost case
   still shows a small deficit against the benchmarks, 0.174 in CVaR 95, which the GBM study showed
   to be a budget effect. Re-running Heston at the full budget would settle it.
5. The time-dependent variant does not converge to a competitive policy at the tested budget. With
   276,510 parameters and 20,000 paths it reaches a delta surface correlating 0.95 with
   Black-Scholes but a zero-cost entropic risk of 6.445 against 4.121 for the benchmark, which is a
   data-per-parameter problem rather than a defect. Shared weights are the default for this reason.
   This was measured before the LayerNorm change and has not been re-tested.
6. The CVaR threshold is projected to the sample quantile once per epoch rather than being optimized
   jointly at full precision. The projection is exact for a fixed sample and the representation is
   convex, but the interaction between threshold updates and minibatch noise is not characterized.
7. The residual seed spread in mean PnL and mean cost, about 0.05 each, is genuine policy variation
   rather than a defect: the two move together, so seeds that trade more pay more and earn more. It
   does not affect the risk measures, which are now reproducible to about 0.03 to 0.09.
8. Entropic risk uses `lambda = 1.0` against a PnL scale of about 3, which is a strong risk aversion.
   The paper treats `lambda` as a tuning parameter and reports a family of solutions. Here only one
   point of that family is explored.
9. Heston results are reported as a table but not as figures. The canonical figure set is GBM only.

### Resolved in this iteration

- The BatchNorm train and evaluation mismatch is fixed. Per-step BatchNorm was replaced by LayerNorm,
  which has no running statistics, so training and evaluation modes are identical by construction.
  The seed 44 gap fell from 0.55 to 0.006, the risk measures became 7 to 8 times more reproducible,
  and training became 2.6 times faster. The BatchNorm path remains available as `--norm batchnorm`.
- The zero-cost under-training result is resolved. At the full budget the learned policy matches the
  Black-Scholes delta to a mean absolute deviation of 0.007 to 0.015 with correlation 0.9996, and
  matches or slightly beats both benchmarks on CVaR. The earlier deficit was a budget artifact.
- The CVaR advantage over Leland, previously not significant at a t ratio of 1.2, is now significant
  at a t ratio of 19.4 for CVaR 95 and 12.5 for CVaR 99.

## Suggested next experiments

1. Re-run the Heston cost sweep at the full 100,000 path and 20 epoch budget to confirm that its
   zero-cost deficit also disappears, as it did for GBM.
2. Raise the seed count to 10 or 20. The risk measures are now tight enough that 5 seeds resolve every
   gap, but a larger count would tighten the mean PnL and cost estimates, which still carry a spread
   of about 0.05.
3. Re-test the time-dependent variant under LayerNorm. The earlier failure was measured with
   BatchNorm, and the normalization change may alter the data-per-parameter requirement.
4. Calibrate the Heston simulator to a real return series and evaluate out of sample on a held-out
   window, which is the missing empirical step relative to the paper.
5. Replace the terminal-jump drawdown with an intra-horizon drawdown on the pre-settlement path, and
   report both. This makes the drawdown column interpretable and separates cost drag from settlement.
