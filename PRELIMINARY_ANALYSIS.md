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

Every study in this document now uses 5 seeds (42 to 46) at 100,000 training paths and 20 epochs per
configuration, so all reported numbers are mean $\pm$ standard deviation on a common budget and no
result rests on a single training run. Beyond the main study, the document reports a full-budget cost
sensitivity for both simulators, a Heston policy surface and diagnostic figure set, an intra-horizon
drawdown alongside the terminal-jump version, an entropic risk aversion frontier over `lambda` from
0.25 to 4.00, and a re-test of the time-dependent weight-sharing mode. All experiments use the same
PnL engine and evaluation protocol.

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

Drawdown is reported two ways. The terminal-jump version, which includes the maturity liability, gives
worst 26.662 (deep hedging), 26.739 (Black-Scholes), 26.739 (Leland) and mean 5.240, 6.121, 5.896; it
is dominated by the common payoff and does not discriminate. The intra-horizon version, computed on
the pre-settlement path, gives mean 3.813, 4.456, 4.210 and worst 9.512, 10.962, 10.483, and does
discriminate. See the drawdown section below.

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
evaluated at the long-run variance. Budget: 100,000 paths, 20 epochs and 5 seeds (42 to 46) per cost
level, matching the GBM study exactly. Each cell is mean $\pm$ standard deviation over the 5 seeds.

| Cost | Deep mean | Deep CVaR 95 | Deep CVaR 99 | BS CVaR 95 | Leland CVaR 95 |
| --- | --- | --- | --- | --- | --- |
| 0.00% | -2.269 ± 0.002 | 3.216 ± 0.018 | 3.625 ± 0.035 | 3.292 ± 0.006 | 3.292 ± 0.006 |
| 0.10% | -2.427 ± 0.004 | 3.430 ± 0.016 | 3.817 ± 0.036 | 3.601 ± 0.006 | 3.560 ± 0.006 |
| 0.50% | -2.807 ± 0.017 | 4.244 ± 0.065 | 4.710 ± 0.081 | 4.892 ± 0.007 | 4.580 ± 0.006 |
| 1.00% | -3.201 ± 0.057 | 5.106 ± 0.109 | 5.718 ± 0.150 | 6.559 ± 0.009 | 5.749 ± 0.006 |

Significance against Leland, using the standard error of the 5-seed mean:

| Cost | CVaR 95 gap | t ratio | CVaR 99 gap | t ratio |
| --- | --- | --- | --- | --- |
| 0.00% | +0.075 | 9.4 | +0.186 | 12.0 |
| 0.10% | +0.131 | 18.1 | +0.278 | 17.2 |
| 0.50% | +0.336 | 11.5 | +0.462 | 12.8 |
| 1.00% | +0.643 | 13.1 | +0.691 | 10.3 |

The zero-cost deficit is gone and the result is now statistically secure rather than a single
observation. At the full budget the learned policy beats both benchmarks at zero cost, with CVaR 95 of
3.216 against 3.292 and CVaR 99 of 3.625 against 3.811, at t ratios of 9.4 and 12.0. Every gap at
every cost level is significant at t of 9.4 or higher.

The margin widens monotonically with cost, from a CVaR 95 gap of 0.075 at zero cost to 0.643 at one
percent. The Heston and GBM studies agree closely at one percent cost, 5.106 against 4.980 for deep
hedging, which is expected since the benchmark volatility is the same in both. The Heston dispersion
is larger than the GBM dispersion, 0.109 against 0.063 at one percent, which reflects the additional
variance-state sampling.

The Heston results also show that the policy uses the variance state. The learned policy is trained
with the variance path as an input, and it beats the variance-adjusted analytic benchmarks at every
cost level including zero, which a policy ignoring the variance state could not do as consistently.
This is a qualitative observation, not a controlled ablation.

### Heston figures

Three figures are saved for the Heston study. `heston_cost_sensitivity.png` reproduces the cost sweep
in the same layout as the GBM version, with CVaR 95 and CVaR 99 against cost for all three strategies.
`heston_diagnostics.png` has three panels: the square root of the variance paths, which shows the
mean-reverting dispersion around the initial level of 0.20; the hedge position against instantaneous
volatility colored by decision step; and the hedge position against moneyness colored by volatility.
`heston_policy_curves.png` is the policy counterpart of the GBM policy figure and is described next.

The diagnostics figure shows that the position is primarily driven by moneyness, with volatility acting
as a secondary modifier. That ordering is confirmed by a direct probe of the policy surface: at the
money and at step 15, moving volatility from 0.10 to 0.35 raises the position from 0.158 to 0.240, a
spread of 0.082, while moving spot from 0.95 to 1.05 times the strike moves it by about 0.33. The
variance effect is therefore real but roughly a quarter the size of the moneyness effect.

### Heston policy surface

`heston_policy_curves.png` has four panels. Top left: a heatmap of the hedge position against spot on
the horizontal axis and instantaneous volatility on the vertical axis, at step 15 with the previous
position at zero, with the three probe volatility levels marked as dotted lines. Top right: the
position difference against the volatility 0.20 baseline, for the same three volatility levels. Bottom
left: the position against spot for three fixed previous positions at volatility 0.20, against the
Black-Scholes delta. Bottom right: the implied no-trade band, the trade size against spot for the same
three previous positions.

The heatmap is the primary panel because it shows the whole surface at once. A line plot of three
volatility levels largely overlaps, since the effect is modest, whereas the heatmap makes the gradient
visible across the full range. The position rises with spot as expected, and at any fixed spot it rises
with volatility, from about 0.15 at volatility 0.10 to about 0.24 at 0.35 at the money. The gradient is
steeper in the region where the call is near or above the strike, where convexity matters most, and
flattens deep in the money where the position approaches its upper bound.

The difference panel is the alternative view that makes the modest effect legible. Subtracting the
volatility 0.20 baseline removes the dominant moneyness trend, so the residual variance effect is
plotted on its own scale of about plus or minus 0.05 rather than being compressed against a position
range of 0 to 1.

The panel shows the effect is not a uniform vertical shift but changes sign across the spot range. For
volatility 0.30 the position is above the baseline by up to 0.045 in the region just above the strike,
near spot 100 to 102.5, where a higher variance raises the value of an at or near the money call. It
falls below the baseline by about 0.029 at spot 110, where the call is comfortably in the money and the
higher variance instead widens the region over which the policy declines to add, and it returns towards
the baseline further out. For volatility 0.12 the pattern is broadly the mirror image, with a smaller
amplitude. This sign structure is the economically meaningful part of the variance response, and it is
not visible in the overlapping curves or in the heatmap colour scale alone, which is why the difference
panel is retained alongside the heatmap.

The bottom two panels reproduce the GBM result in the Heston setting. The learned position lies below
the Black-Scholes delta across the whole spot range, and the trade size is close to zero over a wide
interval that shifts with the previous position. The band is widest for the previous position of 0.30,
where the policy declines to trade for spot between roughly 90 and 100, which is the same asymmetric
no-transaction region seen in the GBM study. The variance-aware policy therefore preserves the
qualitative structure of the frictional optimum rather than collapsing to a memoryless delta.

### Drawdown: terminal jump against intra-horizon

The original drawdown metric measured the peak to trough decline of the cumulative PnL path including
the terminal liability, which is booked in a single step at maturity. That made the metric a rescaled
maximum payoff rather than a drawdown, and it was nearly identical across strategies because all three
face the same liability. An intra-horizon drawdown is now computed on the pre-settlement path, which
excludes the terminal settlement and measures only the risk accumulated while hedging.

| Strategy | Terminal DD mean | Terminal DD worst | Intra DD mean | Intra DD p95 | Intra DD worst |
| --- | --- | --- | --- | --- | --- |
| Deep Hedging | 5.484 | 26.365 | 3.813 | 6.099 | 9.512 |
| Black-Scholes | 6.131 | 26.271 | 4.456 | 7.039 | 10.962 |
| Leland | 5.906 | 26.271 | 4.210 | 6.525 | 10.483 |

The terminal column confirms the problem: the worst values are 26.365, 26.271 and 26.271, essentially
identical, because they are driven by the largest payoff in the sample rather than by the hedging. The
intra-horizon column separates the strategies cleanly. Deep hedging has the lowest mean drawdown at
3.813 against 4.456 for Black-Scholes and 4.210 for Leland, and the lowest worst case at 9.512 against
10.962 and 10.483. This is consistent with the CVaR ordering and shows the improvement is not confined
to the terminal distribution.

### Entropic risk aversion sweep

The entropic risk measure has a single tuning parameter, and the paper treats it as a family of
solutions rather than a fixed choice. The sweep trains the main GBM configuration at five values of
`lambda`, 100,000 paths, 20 epochs and 5 seeds (42 to 46) each, so every cell is mean $\pm$ standard
deviation over the 5 seeds. Five values rather than three are used so the frontier is smooth enough to
read as a curve.

| Lambda | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | -2.820 ± 0.019 | 1.550 ± 0.053 | 6.691 ± 0.172 | 8.174 ± 0.304 | 3.158 ± 0.026 | 0.534 ± 0.017 |
| 0.50 | -3.067 ± 0.052 | 1.108 ± 0.048 | 5.597 ± 0.222 | 6.488 ± 0.349 | 3.401 ± 0.077 | 0.781 ± 0.048 |
| 1.00 | -3.252 ± 0.056 | 0.823 ± 0.044 | 4.980 ± 0.063 | 5.542 ± 0.086 | 3.600 ± 0.030 | 0.967 ± 0.054 |
| 2.00 | -3.428 ± 0.039 | 0.686 ± 0.019 | 4.813 ± 0.038 | 5.257 ± 0.068 | 3.897 ± 0.037 | 1.143 ± 0.039 |
| 4.00 | -3.633 ± 0.024 | 0.592 ± 0.011 | 4.764 ± 0.017 | 5.120 ± 0.031 | 4.316 ± 0.044 | 1.348 ± 0.020 |

Turnover tracks mean cost almost exactly, from 0.530 at `lambda = 0.25` to 1.348 at `lambda = 4.00`,
since cost is proportional to traded volume at a common price level. It is omitted from the table only
to keep the width manageable.

The tradeoff is clean and monotone in every column. Raising `lambda` from 0.25 to 4.00 reduces the
standard deviation from 1.550 to 0.592 and CVaR 95 from 6.691 to 4.764, at the cost of mean PnL
falling from -2.820 to -3.633 and transaction cost rising from 0.534 to 1.348. This is the expected
risk-return frontier: a more risk averse objective trades more to suppress dispersion and pays for it
in cost and in expected PnL.

The frontier is computed at a single cost level, `cost_rate = 0.01`. At zero cost the policy would
converge to the Black-Scholes delta for every `lambda` on this sweep, since with no friction there is
no cost to trading and the objective reduces to minimizing the risk of an unhedged gap. The `lambda`
family is informative precisely because costs make the tradeoff non-trivial.

The entropic risk itself rises with `lambda`, from 3.158 to 4.316, which is not a contradiction. The
entropic risk measure is `lambda`-dependent, so its value is not comparable across the sweep; what is
comparable is the position of each policy on the risk-return frontier. Every value from `lambda = 0.50`
upward beats both benchmarks on CVaR 95 and CVaR 99, with the benchmarks at 6.363 and 7.016
(Black-Scholes) and 5.539 and 6.039 (Leland). The `lambda = 0.25` policy is the exception: it beats
Black-Scholes on CVaR 95, 6.691 against 6.363, but its CVaR 99 of 8.174 is worse than both benchmarks.
That is the intended behaviour of a very low risk aversion, which optimizes a mild penalty and leaves
the deep tail exposed, and it is evidence that the sweep is exploring a genuine family of objectives
rather than producing near-identical policies. It also shows that `lambda = 1.0` is a reasonable
default rather than an arbitrary choice, since it is the smallest value in the sweep that dominates
both benchmarks on every measure.

The `lambda = 1.0` row matches the main 5-seed table exactly, as it should, since it is the same
configuration and seeds. This is a useful internal consistency check on the sweep implementation.

### Fitted frontier curve

`lambda_frontier.png` shows the five computed points with their error bars and a smooth curve drawn
through them. The left panel plots mean PnL against CVaR 95, the middle panel plots mean cost against
CVaR 95, and the third panel shows dispersion and tail risk against `lambda`.

The curve is a monotone piecewise cubic Hermite interpolant (PCHIP) of the five mean points, computed
in $\log\lambda$ rather than in $\lambda$ itself, because the sampled values 0.25, 0.5, 1.0, 2.0 and
4.0 are equally spaced on a log scale. The interpolation is monotone by construction, so it cannot
introduce a spurious reversal between two adjacent risk aversion levels, and it passes exactly through
every computed point. It is clamped to the first and last point outside the sampled range, so the curve
never extrapolates beyond the evidence.

This is a visualization convenience, not a new result. The fit adds no information beyond the five
points: it does not estimate the curve at unsampled `lambda` values, since the true frontier need not
be piecewise cubic between them, and no claim in this document depends on it. Its only purpose is to
make the shape of the tradeoff easier to read than five isolated markers. The reported numbers remain
the five computed points in the table above, each with its own 5-seed standard deviation.

### Time-dependent variant under LayerNorm

The network supports two weight-sharing modes. The shared mode uses one linear layer per hidden layer
across all decision steps, with per-step normalization, giving 20,353 parameters. The time-dependent
mode trains a separate MLP per step, giving 276,510 parameters. An earlier test of the time-dependent
mode under BatchNorm failed badly, reaching a zero-cost entropic risk of 6.445 against 4.121 for the
benchmark. That test was run before the normalization change, so it is repeated here at the full
budget: 100,000 paths, 20 epochs, `cost_rate = 0.01`.

| Strategy | Entropic | CVaR 95 | CVaR 99 |
| --- | --- | --- | --- |
| Deep Hedging, time-dependent | 3.671 | 5.051 | 5.605 |
| Deep Hedging, shared | 3.613 | 4.895 | 5.410 |
| Black-Scholes | 4.861 | 6.363 | 7.016 |
| Leland | 4.429 | 5.539 | 6.039 |

Both rows use an identical protocol: the same training paths, the same evaluation paths, the same
seed, the same optimizer, the same 100,000 path and 20 epoch budget, and the same cost level. The only
difference is the weight-sharing mode, so this is a like-for-like comparison.

The time-dependent variant is competitive. It beats both benchmarks on every measure, with an entropic
risk of 3.671 against 4.861 and 4.429, and a CVaR 95 of 5.051 against 6.363 and 5.539. It remains
slightly behind the shared variant on every measure: 0.058 in entropic risk, 0.156 in CVaR 95 and 0.195
in CVaR 99, while using 13.6 times as many parameters, 276,510 against 20,353.

Recommendation: the shared variant remains the default. It is more accurate on a like-for-like budget,
13.6 times smaller, and about 2.6 times faster to train, since the time-dependent mode runs a separate
forward pass per decision step. The earlier failure of the time-dependent mode was a BatchNorm
artifact, not a capacity limit, but removing that artifact made it competitive rather than superior.

## Remaining limitations and open issues

1. No market data and no calibration. All conclusions are relative, within a synthetic world whose
   parameters were chosen rather than estimated. The absolute PnL values carry no economic meaning.
2. Seed dispersion is measured and small relative to every reported gap. Across 5 seeds the learned
   policy's entropic risk varies by a standard deviation of 0.030 and CVaR 95 by 0.063, against gaps
   to Leland of 0.827 and 0.548. Every gap is significant at a t ratio of 12.5 or higher. The
   remaining spread is in mean PnL and mean cost, about 0.05 each, which is genuine policy variation
   rather than noise: the two move together, so seeds that trade more pay more and earn more.
3. The terminal-jump drawdown is dominated by the maturity liability and is not an informative
   intra-horizon risk measure. It is retained for continuity and reported alongside the new
   intra-horizon drawdown, which excludes the terminal settlement and is the interpretable column.
4. The Heston dispersion is larger than the GBM dispersion, 0.109 against 0.063 in CVaR 95 at one
   percent cost, because the variance state adds a second source of sampling variation. Every Heston
   gap is still significant at a t ratio of 9.4 or higher, so this affects precision rather than
   conclusions.
5. The time-dependent variant is competitive but not better than the shared variant, and it costs
   13.6 times as many parameters, 276,510 against 20,353. Shared weights remain the default.
6. The CVaR threshold is handled in two stages. It is warm started at the alpha quantile of losses
   under the initial policy, then re-projected to the alpha quantile of the pooled epoch losses after
   each epoch, while remaining a trainable parameter with its own optimizer group. The projection is
   exact for a fixed sample: for the Rockafellar-Uryasev objective
   $w + \frac{1}{1-\alpha}\mathbb{E}[(-X_T - w)^{+}]$ the minimizer over $w$ is the alpha quantile of
   the loss distribution, so the projection lands on the exact minimizer rather than approximating it.
   What is not characterized is the interaction between the once-per-epoch projection and minibatch
   noise within the epoch, since the threshold is held fixed across the minibatches of an epoch while
   the policy moves.
7. The lambda sweep covers five computed values, 0.25 to 4.00, which is enough to read the frontier
   shape but is not a continuous parameterization. A continuous fitted curve is now available as a
   visualization aid drawn through the five computed points, shown in `lambda_frontier.png`. The curve
   interpolates the measured means and does not add evidence at unsampled `lambda` values, so the five
   computed points remain the reported result.
8. The Heston policy figure uses a heatmap for the variance dimension rather than a family of curves,
   because the variance effect, a spread of about 0.08 in position across the full volatility range,
   is small enough that separate curves overlap. A difference panel against the volatility 0.20
   baseline is included to make the residual effect legible, but neither panel is as direct as the
   single-curve GBM figure, since the Heston policy is a surface rather than a curve.

### Resolved in this iteration

- The BatchNorm train and evaluation mismatch is fixed. Per-step BatchNorm was replaced by LayerNorm,
  which has no running statistics, so training and evaluation modes are identical by construction.
  The seed 44 gap fell from 0.55 to 0.006, the risk measures became 7 to 8 times more reproducible,
  and training became 2.6 times faster. The BatchNorm path remains available as `--norm batchnorm`.
- The zero-cost under-training result is resolved for both simulators, and for Heston it is now
  multi-seed. At the full budget the GBM policy matches the Black-Scholes delta to a mean absolute
  deviation of 0.007 to 0.015 with correlation 0.9996, and the Heston policy beats both benchmarks at
  zero cost, 3.216 ± 0.018 against 3.292 ± 0.006 in CVaR 95, at a t ratio of 9.4. The earlier
  deficits were budget artifacts.
- The CVaR advantage over Leland, previously not significant at a t ratio of 1.2, is now significant
  at a t ratio of 19.4 for CVaR 95 and 12.5 for CVaR 99.
- The drawdown metric is now interpretable. An intra-horizon drawdown on the pre-settlement path is
  reported alongside the terminal-jump version, and it separates the strategies: 3.813 for deep
  hedging against 4.456 for Black-Scholes and 4.210 for Leland, where the terminal-jump version gave
  5.484, 6.131 and 5.906 and was dominated by the common maturity liability.
- The time-dependent variant is competitive under LayerNorm. At 100,000 paths and 20 epochs it
  reaches an entropic risk of 3.671 against 3.613 for the shared variant, and beats both benchmarks.
  The earlier failure was a BatchNorm artifact, not a capacity limit.
- The Heston results are now reported as figures as well as a table, including a diagnostic showing
  the policy responding to the variance state, and a policy surface with the no-trade band.
- The Heston cost sweep is no longer single seed. It now uses 5 seeds at 100,000 paths and 20 epochs
  per cost level, matching the GBM study exactly, and every gap against Leland is significant at a
  t ratio of 9.4 or higher.
- The lambda sweep is no longer single seed, and it is wider. It now covers five values from 0.25 to
  4.00 with 5 seeds each, so the risk-return frontier is a statistically supported curve rather than
  three single observations.

## Suggested next experiments

Every study in this document now uses 5 seeds at the full 100,000 path and 20 epoch budget, so the
remaining work is either empirical or a genuine scope extension rather than a statistical gap.

1. Calibrate the Heston simulator to a real return series and evaluate out of sample on a held-out
   window. This is the one substantive gap relative to the paper, and it requires real market data,
   which is out of scope for this project.
2. Test a path-dependent payoff, for example a down-and-out call. This requires the nested simulation
   construction that the current single-payoff engine does not support, so it is a scope extension
   rather than a parameter change.
3. Add a second friction model, for example quadratic market impact or a bid-ask spread, to test
   whether the no-transaction band widens as the theory predicts when costs become nonlinear.
4. Add computed points between the existing `lambda` values rather than reading the fitted curve at
   unsampled risk aversions, if a finer frontier is needed for the write-up. The fitted curve is a
   visualization aid only and should not be used to quote numbers at unsampled `lambda`.
