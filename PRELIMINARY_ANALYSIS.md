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
20,353 policy parameters, seed 42. Training took 510 s and the full run 664 s on CPU.

The headline results are now reported from a 5-seed study (seeds 42, 43, 44, 45, 46) rather than from
the single canonical run. Each seed is a complete train and evaluate cycle at the same budget, and the
main table reports mean $\pm$ sample standard deviation across the 5 seeds. The canonical seed 42 run
is retained unchanged and remains the source of every figure in `results/`. The study took 3057 s and
writes `results/multiseed_summary.json`.

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
- The zero-cost limit reproduces. In an auxiliary zero-cost run at `lambda = 50` the learned policy
  beat the Black-Scholes delta on entropic risk (4.005 against 4.114) and the mean absolute deviation
  from the Black-Scholes delta was 0.045, 0.030 and 0.063 at decision steps 5, 15 and 25.

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
  linear weights across time but keeps per-step batch normalization statistics, and the
  time-dependent variant trains a separate MLP per step.
- Seed dispersion is larger than the smallest reported gaps. The headline table is a 5-seed mean
  $\pm$ standard deviation, which is enough to establish the mean PnL, cost and entropic advantages
  over Leland but not enough to resolve the CVaR advantage; see the significance table below.
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
PnL improves from -4.51 (Black-Scholes) and -4.28 (Leland) to -3.18, and CVaR 95 falls from 6.36 to
4.98. The Black-Scholes and Leland histograms are concentrated in a narrow band near -4.5 to -4.3,
which is the deterministic cost drag, while the deep hedging histogram is visibly wider and reaches
into the region above -1.0. This is the expected signature of a policy that trades less but carries
more residual risk: it buys a better mean at the price of higher dispersion, since standard deviation
rises from 0.757 to 0.930. Both the mean shift and the dispersion increase are stable features: over 5
seeds the learned policy's mean PnL is -3.235 $\pm$ 0.043 and its standard deviation is
0.973 $\pm$ 0.151.

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
policy is retrained from scratch at every cost level on 20,000 paths for 8 epochs, and all strategies
are scored on the same 20,000 path set.

Deep hedging is best at the highest cost and worst at zero cost. At 1.00 percent its 95 percent CVaR
is 5.105 against 6.358 for Black-Scholes and 5.532 for Leland. At 0.50 percent it is 4.611, close to
Leland's 4.380 and clearly better than Black-Scholes. At 0.00 percent it is 3.611 against 3.111 for
both analytic deltas, so the ordering reverses and the learned policy is about 0.5 worse.

The qualitative shape matches the paper, since the benefit of learning is increasing in the friction
level. The low-cost end does not. With zero costs the theoretical optimum is the Black-Scholes delta
and the learned policy should match it, not underperform it. The most likely explanation is the
reduced training budget of this study, 8 epochs against 20 in the main run, so the network has not
converged and its approximation error is exposed once there is no cost pressure to exploit. This is
an open item, not a contradiction of the paper.

### `training_loss.png`

Epoch-level training risk over the 20 epochs of the canonical run, with a 3 epoch moving average.

Risk falls from 4.6830 to 3.5799. About 82 percent of the total reduction is achieved by epoch 2 and
about 87 percent by epoch 4, after which progress is slow and slightly noisy: epochs 9, 14 and 17 are
marginally above their predecessors. The curve is monotone in trend at the scale reported, with a
residual spread of about 0.04 over the second half.

This is consistent with the paper's reported behaviour, where the policy improves rapidly and then
flattens. It also shows the run is converged at 20 epochs in the sense that further training yields
little, which matters for the cost study where only 8 epochs were used. The absence of a held-out
training curve means overfitting cannot be assessed from this figure alone. Overfitting is unlikely
here because the training set has 100,000 paths against 20,353 parameters, but this is an argument,
not evidence.

### `learning_curves.png`

Left: the empirical distribution function of terminal PnL. Right: expected shortfall at 50, 75, 90,
95 and 99 percent confidence.

The distribution function shows the deep hedging curve lying to the right of both benchmarks over
almost the entire support, which is first order stochastic dominance on the realized sample. The
dominance is strict in the body and narrows in the extreme lower tail below about -7.0, where the
three curves converge, consistent with all strategies facing the same unbounded liability.

The right panel quantifies the same fact across confidence levels: 3.94 against 5.11 (Black-Scholes)
and 4.66 (Leland) at 50 percent, 4.97 against 6.36 and 5.54 at 95 percent, and 5.53 against 7.02 and
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

The mean trajectory starts at about 0.39, rises to roughly 0.50 within four steps and then stays flat
near 0.49 to the end. The Black-Scholes and Leland mean deltas are essentially constant at 0.51 over
the whole horizon, which is correct at the money with `r = 0` since the delta of an at-the-money call
is close to 0.5 until the last step. The learned mean sits deliberately below them, and the gap of
about 0.02 is the average amount of hedging the policy declines to do.

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
| Deep Hedging | -3.235 ± 0.043 | 0.973 ± 0.151 | 5.272 ± 0.461 | 5.932 ± 0.606 | 3.749 ± 0.237 | 0.950 ± 0.047 |
| Black-Scholes | -4.511 ± 0.001 | 0.756 ± 0.001 | 6.352 ± 0.009 | 7.013 ± 0.016 | 4.859 ± 0.004 | 2.225 ± 0.002 |
| Leland | -4.280 ± 0.002 | 0.511 ± 0.001 | 5.529 ± 0.007 | 6.026 ± 0.012 | 4.427 ± 0.002 | 1.994 ± 0.001 |

The two analytic benchmarks are deterministic given the path set, so their standard deviations are
near zero by construction and reflect only the change of evaluation paths across seeds. All of the
seed-to-seed variation in the table is therefore attributable to the learned policy.

Per-seed values for the learned policy, showing where the dispersion comes from:

| Seed | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | -3.170 | 0.930 | 4.968 | 5.530 | 3.593 | 0.883 |
| 43 | -3.260 | 0.840 | 5.075 | 5.821 | 3.639 | 0.974 |
| 44 | -3.272 | 1.203 | 6.013 | 6.939 | 4.134 | 0.995 |
| 45 | -3.210 | 0.853 | 4.886 | 5.403 | 3.564 | 0.921 |
| 46 | -3.263 | 1.041 | 5.420 | 5.966 | 3.815 | 0.978 |

### What the multi-seed study changes

The multi-seed study increases the statistical reliability of the reported gaps, and it changes one
conclusion. Using the standard error of the mean across seeds, $\mathrm{SE} = s/\sqrt{5}$, the gaps
against Leland are:

| Metric | Gap, Leland minus Deep Hedging | Standard error | t ratio | Reliable |
| --- | --- | --- | --- | --- |
| Mean PnL | +1.045 | 0.019 | 53.8 | yes |
| Mean cost | +1.043 | 0.021 | 49.9 | yes |
| Entropic | +0.678 | 0.106 | 6.4 | yes |
| CVaR 95 | +0.256 | 0.206 | 1.2 | no |
| CVaR 99 | +0.094 | 0.271 | 0.3 | no |

The cost reduction and the improvement in mean PnL and entropic risk are far outside the seed noise.
The CVaR advantage over Leland is not. The single-seed canonical run showed a CVaR 95 gap of 0.571
against Leland, which the 5-seed study revises down to 0.256 with a standard error of 0.206. The
correct reading is that deep hedging matches Leland on CVaR and beats it on mean PnL, entropic risk
and cost, not that it dominates on every measure. Against Black-Scholes the CVaR 95 gap of 1.080 has
a t ratio of 5.2 and remains reliable.

Seed 44 is the source of most of the dispersion: its entropic risk is 4.134 against a range of 3.564
to 3.815 for the other four. Excluding it, the learned policy's entropic risk is 3.653 $\pm$ 0.113
and CVaR 95 is 5.087 $\pm$ 0.235. The cause was isolated and is a BatchNorm train and evaluation
discrepancy, not a training failure: for seed 44 the final minibatch training risk is 3.578, the
evaluation risk on the training paths is 4.131, and the evaluation risk on fresh paths is 4.134. The
policy is therefore not overfitting, since train and test agree; the batch statistics used during
training simply differ from the running statistics used at evaluation. The same diagnostic on seed 42
gives 3.580, 3.589 and 3.593, a gap of 0.013. This is recorded as an open issue below.

### Single-seed canonical run

The canonical run at seed 42 is retained unchanged and is the source of every figure in `results/`.
It reproduces the seed 42 row of the multi-seed table exactly, which confirms the study and the
figure pipeline are consistent.

| Strategy | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost | Gross PnL | Turnover | Tail cost share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deep Hedging | -3.170 | 0.930 | 4.968 | 5.530 | 3.593 | 0.883 | -2.287 | 14.83 | 0.220 |
| Black-Scholes | -4.512 | 0.757 | 6.363 | 7.016 | 4.861 | 2.224 | -2.288 | 15.01 | 0.526 |
| Leland | -4.281 | 0.512 | 5.539 | 6.039 | 4.429 | 1.993 | -2.288 | 15.11 | 0.495 |

Maximum drawdown, on the path convention of this implementation: worst 26.824 (deep hedging), 26.739
(Black-Scholes), 26.739 (Leland); mean 5.043, 6.121, 5.896. The worst values are dominated by the
terminal payoff jump and are not an informative intra-horizon drawdown, see the limitations.

Cost sensitivity, CVaR 95 and CVaR 99, policy retrained per cost level on 20,000 paths for 8 epochs:

| Cost | Deep mean | Deep CVaR95 | Deep CVaR99 | BS CVaR95 | BS CVaR99 | Leland CVaR95 | Leland CVaR99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00% | -2.287 | 3.611 | 4.077 | 3.111 | 3.508 | 3.111 | 3.508 |
| 0.10% | -2.489 | 3.963 | 4.419 | 3.416 | 3.846 | 3.375 | 3.788 |
| 0.50% | -2.908 | 4.611 | 5.150 | 4.696 | 5.245 | 4.380 | 4.850 |
| 1.00% | -3.402 | 5.105 | 5.534 | 6.358 | 7.051 | 5.532 | 6.059 |

Policy response at step 15 with previous position 0.508, benchmark reference in brackets:

| Spot move | Spot | Deep Hedging | Black-Scholes | Leland |
| --- | --- | --- | --- | --- |
| -12% | 88.00 | 0.258 | 0.001 | 0.009 |
| -8% | 92.00 | 0.344 | 0.021 | 0.064 |
| -4% | 96.00 | 0.432 | 0.162 | 0.232 |
| -2% | 98.00 | 0.495 | 0.316 | 0.364 |
| 0% | 100.00 | 0.497 | 0.508 | 0.511 |
| +2% | 102.00 | 0.519 | 0.695 | 0.654 |
| +4% | 104.00 | 0.578 | 0.838 | 0.775 |
| +8% | 108.00 | 0.654 | 0.973 | 0.927 |
| +12% | 112.00 | 0.710 | 0.998 | 0.984 |

Auxiliary runs, not saved as figures. Heston with CVaR 95 and 8 epochs, 12,000 paths: deep hedging
CVaR 95 5.503 against 6.598 (Black-Scholes) and 5.779 (Leland). Zero-cost GBM at `lambda = 50`,
30,000 paths and 30 epochs: deep hedging entropic 4.005 against 4.114 (Black-Scholes), with mean
absolute deviation from the Black-Scholes delta of 0.045, 0.030 and 0.063 at steps 5, 15 and 25.

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
4. The cost sensitivity study uses 8 epochs and 20,000 paths, well below the main run. The zero-cost
   result, where the learned policy is worse than Black-Scholes, is most plausibly under-training
   rather than a property of the method, but it has not been demonstrated.
5. The time-dependent variant does not converge to a competitive policy at the tested budget. With
   276,510 parameters and 20,000 paths it reaches a delta surface correlating 0.95 with
   Black-Scholes but a zero-cost entropic risk of 6.445 against 4.121 for the benchmark, which is a
   data-per-parameter problem rather than a defect. Shared weights are the default for this reason.
6. The CVaR threshold is projected to the sample quantile once per epoch rather than being optimized
   jointly at full precision. The projection is exact for a fixed sample and the representation is
   convex, but the interaction between threshold updates and minibatch noise is not characterized.
7. BatchNorm train and evaluation statistics diverge for some seeds and this materially affects the
   reported risk. Seed 44 shows a gap of 0.55 between the minibatch training risk of 3.578 and the
   evaluation risk of 4.134, while seed 42 shows 0.013. Because the training paths and fresh paths
   give the same evaluation risk, the effect is not overfitting; it is a mismatch between the batch
   statistics seen during training and the running statistics used at evaluation. This is the single
   largest uncontrolled source of seed variance and the first thing to fix. Multi-seed results are
   reported as they stand rather than filtered, since dropping the seed would understate the spread.
8. Entropic risk uses `lambda = 1.0` against a PnL scale of about 3, which is a strong risk aversion.
   The paper treats `lambda` as a tuning parameter and reports a family of solutions. Here only one
   point of that family is explored.
9. Heston results are not reported in the saved figures. They exist in the console output of
   auxiliary runs but the canonical figure set is GBM only.

## Suggested next experiments

1. Fix the BatchNorm train and evaluation mismatch, then repeat the 5-seed study. Replacing the
   per-step BatchNorm with LayerNorm, or freezing the running statistics over the last epoch, should
   remove the 0.55 gap seen at seed 44 and shrink the seed spread. This is the highest value change
   because it currently dominates the uncertainty on every learned policy metric.
2. Re-run the cost sensitivity with the full 20 epoch budget and 100,000 paths, then check whether the
   learned policy converges to the Black-Scholes delta at zero cost as the theory requires. This
   directly tests the one result that currently contradicts the paper.
3. Add a transactional-cost sweep with the Heston simulator and report the variance-aware policy
   against a variance-adjusted benchmark. The Heston state is already implemented, so only the
   benchmark and the reporting change.
4. Calibrate the Heston simulator to a real return series and evaluate out of sample on a held-out
   window, which is the missing empirical step relative to the paper.
5. Replace the terminal-jump drawdown with an intra-horizon drawdown on the pre-settlement path, and
   report both. This makes the drawdown column interpretable and separates cost drag from settlement.

If the BatchNorm mismatch is resolved, the natural follow-up is to raise the seed count to 10 or 20
and re-test the CVaR gap to Leland, which needs roughly 30 seeds at the current dispersion to reach
significance if the point estimate of 0.256 is correct.
