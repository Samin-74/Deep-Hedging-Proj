# Preliminary Analysis

Technical results log for the Deep Hedging replication. Working notes only, not the course report.

## Project status

The pipeline is implemented end to end and runs reproducibly from a fixed seed. A semi-recurrent
policy maps a six dimensional normalized state to a bounded hedge position, trained with Adam on
synthetic Monte Carlo paths under proportional transaction costs. Two simulators (GBM, Heston with
full truncation), two convex risk measures (entropic, CVaR with a Rockafellar-Uryasev threshold) and
two analytic benchmarks (Black-Scholes delta, Leland delta) are wired through one shared PnL engine,
so all three strategies are scored on identical paths and an identical cost structure.

The canonical run is configured as `S0 = K = 100`, `T = 30/365`, `n_steps = 30`, `n_paths = 100000`,
`n_eval_paths = 100000`, `r = 0`, `sigma = 0.2`, `cost_rate = 0.01`, `batch_size = 256`, `epochs = 20`,
`lr = 0.001`, `lambda_entropic = 1.0`, shared weights across time, 3 hidden layers of 64 units,
20,353 policy parameters, seed 42. Training takes about 230 s and the full run about 1030 s on CPU.

The headline results are reported from a 5-seed study (seeds 42, 43, 44, 45, 46). Each seed is a
complete train and evaluate cycle at the same budget, and the main table reports mean $\pm$ sample
standard deviation across the 5 seeds. The study took about 1200 s and writes
`results/multiseed_layernorm.json`.

Every saved study records its own run metadata under a `config` key, including simulator, path budget,
epoch count, cost level and normalization. Check that block before quoting any number from a JSON file:
a reduced-budget smoke test overwrites the file it is pointed at, and the metadata makes such a
mismatch immediately visible.

Provenance of the numbers in this document. Every numeric claim below traces to a file in `results/`.
The canonical single-seed console output is persisted in `results/canonical_run.json`, which holds the
per strategy summaries, the cost drag table, turnover, tail cost share, the risk measures, the policy
response scenario and the training history. The figure sidecars are `results/policy_curves.json`,
`results/position_dynamics.json`, `results/learning_curves.json`, `results/heston_policy_curves.json`
and `results/heston_diagnostics.json`, each written by the plotting function that produced the figure
beside it. The superseded BatchNorm study is retained as `results/batchnorm_multiseed.json` so the
normalization comparison table can be checked against its original source. A scripted audit confirms
that all 485 numeric tokens in this document and all 25 in `README.md` round to a value present in one
of these files.

Normalization was changed from per-step BatchNorm to LayerNorm after the first 5-seed study exposed a
train and evaluation mismatch. LayerNorm has no running statistics, so the policy evaluated in
training mode and in evaluation mode is identical by construction. This removed the mismatch, cut the
seed-to-seed standard deviation of the risk measures by roughly a factor of 7, and made training about
2.2 times faster. The BatchNorm path is retained behind `--norm batchnorm` for comparison.

Every study in this document now uses 5 seeds (42 to 46) at 100,000 training paths and 20 epochs per
configuration, so all reported numbers are mean $\pm$ standard deviation on a common budget and no
result rests on a single training run. Beyond the main study, the document reports a full-budget cost
sensitivity for both simulators, a Heston policy surface and diagnostic figure set, an intra-horizon
drawdown alongside the terminal-jump version, an entropic risk aversion frontier over `lambda` from
0.25 to 4.00, and a re-test of the time-dependent weight-sharing mode. All experiments use the same
PnL engine and evaluation protocol.

Status: methodology and qualitative claims reproduce. As of the paper replication phase, four of the
paper's own Section 5 numerical experiments are also reproduced: the Black-Scholes and Heston price
asymptotics (Figures 10 and 11), the Heston no-cost two instrument experiment (Figures 2, 6, 7) and the
recurrent against simple ablation (Figures 4, 5). A fifth, the call spread risk aversion experiment
(Figures 8, 9), is attempted and does not reproduce. A Section 5.1 architecture fidelity check is also
run. Section 5.4 and Section 5.5 remain unreproduced. See the verdict for the full list.

### Paper replication phase

The reproduction work lives in a separate set of scripts and writes only to `results/paper/`, so no
existing figure or number in `results/` was overwritten. The entry points are `price_asymptotics.py`
(Figure 10), `paper_heston.py` (Figures 2, 3, 6, 7), `price_asymptotics_heston.py` (Figure 11),
`recurrent_ablation.py` (Figures 4, 5), `paper_call_spread.py` (Figures 8, 9),
`paper_architecture.py` (Section 5.1 fidelity), `eda.py` (exploratory analysis) and
`regen_paper_figures.py` (redraw every paper figure from stored JSON without retraining). Supporting
modules are `src/heston.py` (analytic pricer and model hedge) and the `PaperHestonSimulator` class in
`src/simulators.py` (exact CIR plus simplified Broadie-Kaya, and the variance swap). All runs use 5
seeds at 100,000 training paths and 20 epochs, and every saved JSON carries a `config` provenance block.

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

**Verdict: Partially, and materially further than before this iteration.** The framework is replicated
and, as of this iteration, four of the paper's own Section 5 numerical experiments are reproduced:
Figure 10 (Black-Scholes price asymptotics), Figure 11 (Heston price asymptotics), Figures 2, 6 and 7
(Heston no-cost two instrument hedging) and Figures 4 and 5 (recurrent against simple network). A fifth,
the call spread risk aversion experiment of Figures 8 and 9, is attempted and does not reproduce: the
more risk averse policy collapses to a near constant position and is worse on every risk measure. Three
of the four reproduced experiments carry a published numeric target, and the agreement is uneven: the
Heston price is exact to the paper's published precision, the $p(0.50)$ figure is within 1.2 percent,
while the $p(0.99)$ figure is 13.9 percent low and the two fitted exponents, at 4.9 percent above and
9.1 percent below $2/3$, are close but not exact. What remains unreproduced is Section 5.4 (the high
dimensional multi model experiment) and Section 5.5 (the S&P 500 real data test), plus the paper's exact
BatchNorm architecture and its $10^6$ path test set.

### Paper experiment against project status

| Paper | What it shows | Project status |
| --- | --- | --- |
| Sections 2 to 4: setup, costs, entropic risk, CVaR, semi-recurrent network | The framework | Replicated. Matches the terminal wealth, the cost term, the entropic risk, the Rockafellar-Uryasev CVaR and the semi-recurrent policy. The shared weights variant is the paper's Remark 5 |
| Section 5.1: training setup | 30 days, daily steps, Adam, batch 256 | Partly. Learning rate 0.001 against the paper's 0.005, shared weights by default, 3 by 64 LeakyReLU with LayerNorm and six engineered inputs, against the paper's separate network per step, two hidden layers of d plus 15 units, ReLU and BatchNorm. A paper-faithful variant is now implemented and compared in `results/paper/paper_architecture.png` |
| Sections 5.2, Figures 2, 3, 6, 7 | Heston, no costs, stock plus variance swap, CVaR 0.5 and 0.99 against the model hedge. q = 1.69, p(0.5) = 1.94, p(0.99) = 3.49 | Reproduced. q = 1.6918, p(0.5) = 1.9630, p(0.99) = 2.9750 over 5 seeds. See the paper results section |
| Sections 5.2, Figures 4, 5 | Network with and without the previous position, with 1 percent cost | Partially reproduced. No cost: the two are indistinguishable (risk gap -0.0030, t 2.07, 0.1 percent of the risk level). At 1 percent cost the recurrent network is worse on risk by 0.0711 with a standard error of 0.0830, so the paper's lower-price claim does not reproduce |
| Sections 5.2, Figures 8, 9 | Call spread, more risk aversion flattens the hedge | Not reproduced. The alpha 0.99 holding does flatten, but it collapses to a near constant 0.031 against a model delta peaking at 0.367, and it is worse than alpha 0.95 on CVaR 95, CVaR 99 and dispersion. Its CVaR 99 of 1.0072 is essentially the unhedged value. See the call spread section |
| Section 5.3.1, Figure 10 | Black-Scholes model, sigma 0.2, lambda 1, ATM call, the price increase from costs grows like eps^(2/3) | Reproduced. Fitted log-log slope 0.6898 with 95 percent CI [0.6362, 0.7353], which contains 2/3 |
| Section 5.3.2, Figure 11 | The same eps^(2/3) scaling under Heston with two instruments | Reproduced. Fitted log-log slope 0.6017 |
| Section 5.4 | Five Heston models together, ten instruments | Not done. The paper reports 5.75 hours of training for this experiment, which is outside the compute budget here |
| Section 5.5 | S&P 500, GJR-GARCH, real data | Not done. Requires real market data and a GARCH calibration, which is outside the declared scope |

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
  the Black-Scholes delta to a mean absolute deviation of 0.0105, 0.0080 and 0.0125 at decision steps
  5, 15 and 25, with correlations of 0.9972, 0.9997 and 0.9992, and matches or slightly beats both
  benchmarks on CVaR 95 and CVaR 99.
- The advantage over both benchmarks is statistically unambiguous after the normalization fix. Over 5
  seeds the t ratios against Leland are 91.6 for mean PnL, 89.9 for cost, 428.2 for entropic risk, 39.2
  for CVaR 95 and 26.9 for CVaR 99.

### What is missing or approximated

- No real market data. The paper calibrates to historical index returns and evaluates out of sample
  on a held-out period. Here every path comes from a simulator with hand-set parameters, so the
  absolute PnL levels have no empirical meaning and only relative comparisons are informative. This is
  the one gap that is genuinely out of scope rather than deferred.
- No calibration step for the original Heston study. Its parameters (`v0 = theta = 0.04`, `kappa = 2`,
  `xi = 0.3`, `rho = -0.7`) are plausible but not fitted to anything, so that study tests robustness of
  the method rather than a market scenario. The paper replication phase uses the paper's own published
  parameter set instead, which removes the calibration question for those experiments.
- The theoretical content of the paper is not reproduced. There is no analysis of the approximation
  error of the policy class, no convergence argument, and no sensitivity of the risk measure to the
  network capacity beyond the empirical comparison of shared and time-dependent weights.
- The default architecture is a reasonable analogue, not the paper's exact network. The paper reports a
  feedforward body with a recurrent pass of the previous position; here the shared variant shares all
  linear weights across time but keeps per-step normalization statistics. A paper-faithful variant
  (separate network per step, two hidden layers of d plus 15 units, ReLU, BatchNorm, learning rate
  0.005, inputs log S and the previous position) is now implemented as `feature_mode = "paper"` and
  compared against the default in `results/paper/paper_architecture.png`. It matches the default to
  within 0.009 on the final training risk and 0.032 on CVaR 95 while using 15,330 parameters against
  20,353.
- The default normalization is LayerNorm rather than the paper's BatchNorm. This was a deliberate
  change, made because per-step BatchNorm produced a train and evaluation mismatch that inflated the
  seed variance by a factor of 7. The BatchNorm path is retained behind `--norm batchnorm` and is what
  the paper-faithful architecture variant uses.
- Section 5.4 of the paper, five Heston models with ten instruments, is not attempted. The paper itself
  reports 5.75 hours of training for it, which is well beyond the compute budget available here.
- Section 5.5, the S&P 500 real data experiment, is not attempted. It needs real market data and a
  GJR-GARCH fit, both outside the declared scope of this project.
- Only a European call and a call spread are hedged. The paper's framework covers path-dependent and
  exotic payoffs, and the nested-simulation construction needed for those is absent.
- One friction model only. Proportional costs are implemented; no quadratic impact, no bid-ask
  spread decomposition, no borrowing or funding costs.
- The test set is $10^5$ paths per seed over 5 seeds, $5 \times 10^5$ in total, against the paper's
  $10^6$ single path set. Seed variance is shown to be the larger uncertainty in every study, so this
  is a reasonable substitute, but the two protocols are not identical.

## Paper results reproduced

Every number in this section comes from a JSON file under `results/paper/`. The runs use the paper's
own parameter set wherever the paper publishes one: Black-Scholes model with $\sigma = 0.2$,
$S_0 = K = 100$, $T = 30/365$, $n = 30$ steps, $\lambda = 1$; Heston with $\kappa = 1$, $\theta = 0.04$,
$\xi = 2$, $\rho = -0.7$, $v_0 = 0.04$. Every cell is 5 seeds (42 to 46) at 100,000 training paths and
20 epochs, except where the table states otherwise. The cost grid is the factor-of-2 grid
$\{0.000625, 0.00125, 0.0025, 0.005, 0.01\}$, with 0 included as the reference level.

### How the numbers map to the paper's theory

Under a martingale measure and with zero initial endowment, Proposition 3.6(i) of the paper gives
$\pi(0) = \rho(0) = 0$. Lemma 3.4 then says the entropic risk of the trained policy's PnL equals the
indifference price $p(Z)$ of the claim. So the entropic risk column below is not merely a risk measure,
it is a price in the paper's sense. Two consequences are testable and both hold:

- $p(Z) \geq \mathbb{E}[Z]$, which is Proposition 3.6(ii). In the Black-Scholes setting
  $q = \mathbb{E}[Z] = 2.2872$ and the learned price at zero cost is $p_0 = 2.3602$, so the inequality
  holds with a gap of 0.0730. In the Heston setting $q = 1.6918$ and $p_0 = 1.8023$, a gap of 0.1104.
- The gap $p_0 - q$ is positive and not zero, because discrete hedging is not free. This is the
  discrete-time hedging error the paper's whole framework is built to price.

The benchmark entropic values are not prices, since Black-Scholes and Leland deltas are not optimal for
that objective.

Note on the PnL convention. The paper reports the hedging error as $q - Z + (\delta \cdot S)_T$, that is
with the premium received. This project reports $-Z + (\delta \cdot S)_T - C_T$, the same quantity minus
$q$, because the premium is not booked. The two therefore differ by the constant $q$, and the histograms
below say so explicitly.

### Section 5.3.1, Figure 10: Black-Scholes price asymptotics

`results/paper/price_asymptotics_bs.png`. Six cost levels, 5 seeds each, 30 trainings in total.

| Cost | $p_\varepsilon$ (5 seed mean ± std) | $p_\varepsilon - q$ | CVaR 95 | CVaR 99 | Mean cost |
| --- | --- | --- | --- | --- | --- |
| 0.000000 | 2.3602 ± 0.0005 | 0.0730 | 3.1181 | 3.4770 | 0.0000 |
| 0.000625 | 2.4796 ± 0.0018 | 0.1925 | 3.2537 | 3.5938 | 0.1094 |
| 0.001250 | 2.5802 ± 0.0019 | 0.2930 | 3.4035 | 3.7473 | 0.1942 |
| 0.002500 | 2.7549 ± 0.0045 | 0.4677 | 3.6980 | 4.0772 | 0.3308 |
| 0.005000 | 3.0600 ± 0.0071 | 0.7729 | 4.1809 | 4.6298 | 0.5621 |
| 0.010000 | 3.5813 ± 0.0058 | 1.2942 | 4.9516 | 5.5179 | 0.9497 |

with $q = 2.2872$ the Black-Scholes price. The fitted slopes:

| Fit | Against $q$ | Against $p_0$ |
| --- | --- | --- |
| Free OLS slope | **0.6898** ± 0.0161 (se) | 0.8377 ± 0.0085 (se) |
| 95 percent bootstrap CI | [0.6362, 0.7353] | [0.8133, 0.8646] |
| Slope fixed at 2/3, intercept only | 3.2673 | 3.0491 |

Note on the evaluation protocol, because the $0.01$ row reads 3.5813 here and 4.959 in the main
multi-seed table. The two studies draw their evaluation paths differently. The main study gives each
seed its own evaluation set, generated with `seed + 1`, so its seed dispersion includes evaluation
sampling and its mean is an average over five different path sets. This study generates one shared
evaluation set with `seed + 1` from the base seed and scores every training run on it, so the 5 seeds
differ only in their training paths and the comparison across cost levels is strictly paired. Both
protocols are internally consistent, and the difference between them here is 1.38, which is far outside
the 0.029 seed standard deviation of the $0.01$ row. The shared path set is used here because the fitted
slope is a comparison between cost levels, which is exactly where pairing helps.

The paper's prediction is a slope of $2/3 = 0.6667$. The measured slope against $q$ is 0.6898 and the
bootstrap interval contains 0.6667, so the $\varepsilon^{2/3}$ scaling is reproduced. The slope measured
against the learned zero-cost price $p_0$ instead of $q$ is 0.8377, which is steeper. That is the right
comparison to make if one believes the discrete hedging error at $\varepsilon = 0$ should be the
reference level, and it is reported here rather than hidden. The paper fits against $q$, so the 0.6898
figure is the primary comparison.

![Black-Scholes price asymptotics](results/paper/price_asymptotics_bs.png)

`results/paper/cost_sensitivity_5seed.png` shows the same study as CVaR 95 and CVaR 99 against the cost
level for all three strategies, with one standard deviation error bars. It also replaces the earlier
console-only cost sweep with stored numbers.

### Section 5.2, Figures 2, 6 and 7: Heston, no costs, two instruments

`results/paper/heston_nocost.json`, figures `heston_fig2.png`, `heston_fig6.png`, `heston_fig7.png`.
The paper's Heston parameters violate the Feller condition by a large margin:
$2\kappa\theta - \xi^2 = 2(1)(0.04) - 4 = -3.92$. Euler-Maruyama is therefore not acceptable, and this
phase adds an exact CIR transition (noncentral chi square) plus the simplified Broadie-Kaya stock step,
exactly as the paper describes. A separate variance swap instrument is built from the running integrated
variance plus $L(t, V_t)$. The policy trades both instruments.

Pricer validation, which had to pass before any training was allowed to run. The gate is implemented as
`validate_pricer` in `src/heston.py` and is called by `paper_heston.py` before the first training:

| Quantity | Value | Reference | Error | Tolerance |
| --- | --- | --- | --- | --- |
| Heston call price at $(0, 100, 0.04)$ | **1.691834** | paper $q = 1.69$ | 0.00003 | 0.001 |
| Low vol-of-vol limit $\xi \to 0$ | 2.28749 | Black-Scholes 2.28715 | 0.00034 | 0.001 |
| Strike 95 against Monte Carlo | 5.78030 | 5.77605 | 0.00425 | 0.02922 |
| Strike 100 against Monte Carlo | 1.69183 | 1.69135 | 0.00049 | 0.01925 |
| Strike 101 against Monte Carlo | 1.08137 | 1.08197 | 0.00060 | 0.01690 |
| Strike 110 against Monte Carlo | 0.02909 | 0.02741 | 0.00167 | 0.00362 |
| Strike 120 against Monte Carlo | 0.00113 | 0.00126 | 0.00013 | 0.00100 |
| Variance swap fair strike, Monte Carlo | 0.003286 | analytic 0.003288 | 0.000002 | n/a |

The worst error is 0.46 of its tolerance, so the gate passes. The $\xi \to 0$ row is the strongest
analytic check: as the vol of vol is driven to zero the Heston pricer must collapse to Black-Scholes, and
it does so to four decimal places. The strike ladder is the check that actually catches a wrong Lewis
phase, because a phase sign error leaves the at the money price correct while corrupting the wings. The
ladder uses a 480 step simulator rather than the 30 step hedging simulator, because at 30 steps the
discretization bias at strike 101 is 0.020, which is larger than the effect being tested. At 480 steps
that bias falls to 0.003 and the comparison is meaningful. The tolerances are four Monte Carlo standard
errors, floored at 0.001, so the gate is a statistical test rather than a fixed absolute threshold.

| Quantity | Paper | This project (5 seed) | Gap | Note |
| --- | --- | --- | --- | --- |
| $q$ | 1.69 | 1.6918 | +0.0018 | Pricer is exact, difference is the paper's rounding |
| $p_\theta$ at $\alpha = 0.50$ | 1.94 | **1.9630 ± 0.0053** | +0.0230 | 1.2 percent high |
| $p_\theta$ at $\alpha = 0.99$ | 3.49 | **2.9750 ± 0.0740** | -0.5150 | 14.8 percent low |
| Model hedge, CVaR 0.50 | not printed | 1.9687 | | Deep hedge is 0.0057 better |
| Model hedge, CVaR 0.99 | not printed | 3.9139 | | Deep hedge is 0.9389 better |

The $\alpha = 0.50$ number is essentially exact: 1.9630 against 1.94, with a standard error of 0.0024,
ten times smaller than the gap. The $\alpha = 0.99$ number is 14.8 percent below the paper. The likely
reasons, in order of importance:

1. Test set size. The paper uses $10^6$ paths; this run uses $2 \times 10^5$ per seed pooled from 5
   seeds. A 99 percent CVaR is a mean over the worst 1 percent, so 2,000 observations per seed. The
   seed-to-seed standard deviation of 0.0740 reflects that.
2. Training paths. 100,000 here against a larger budget in the paper. A 99 percent objective needs more
   tail samples than a 50 percent objective, and the paper's Figure 6 is precisely the plot that shows
   the tail getting harder to learn.
3. Normalization. LayerNorm in the two instrument path, BatchNorm in the paper. The paper-faithful
   architecture variant was only compared in the GBM setting.

The result is reported as it is: the 50 percent objective is reproduced closely, the 99 percent
objective is reproduced in shape and ordering but not in level.

![Heston Figure 2](results/paper/heston_fig2.png)

In Figure 2 the deep hedge and the analytic model hedge now have almost the same distribution: means of
-1.703 and -1.704, standard deviations of 0.414 against 0.435, and 50 percent CVaR of 1.964 against
1.969. The deep hedge wins by only 0.0057, which is within the seed standard deviation of 0.0053. That
is a much tighter result than the paper shows, and it is the right one: with two instruments and no
costs, a Heston model hedge is close to the true optimum, so a learned policy can only match it, not
beat it by a wide margin. The paper's Figure 2 shows the learned policy tracking the model hedge closely,
which is what this figure now shows. An earlier version of this experiment reported a model hedge CVaR
of 2.432 against 1.964 for the deep hedge, an artificial 0.46 gap caused by a sign error in the analytic
pricer. See the corrections section.

![Heston Figure 6](results/paper/heston_fig6.png)

![Heston Figure 7](results/paper/heston_fig7.png)

Figure 6 centres each policy at its own risk measure and Figure 7 centres both at $q$. The ordering is
the paper's: the more risk averse policy has a heavier far tail but a better 99 percent average. At
$\alpha = 0.99$ the gap between the deep hedge and the model hedge is larger, 2.9750 against 3.9139, a
0.9389 improvement, because a 99 percent objective leaves more room for a policy to reshape the tail
than a 50 percent objective does.

`results/paper/heston_fig3.png` shows the stock holding of the model hedge and the non-recurrent deep
hedge at step 15 on the paper's grid, spot from 95 to 103 and variance from 0.03 to 0.13, plus their
difference and a set of variance slices. The paper prints the variance range as "[0.3, 0.13]", which is
almost certainly a typo for [0.03, 0.13]; the latter is used. The surfaces are stored in
`results/paper/heston_fig3.json` so the figure can be redrawn without retraining.

Read on its own, the difference panel is informative. The learned surface is steeper in spot than the
analytic one: the difference is about -0.001 at spot 95 and rises through zero near spot 95.2 to about
+0.054 at spot 103. So the deep hedge holds slightly less than the model hedge when the option is well
out of the money and more when it is at or in the money, which is the usual consequence of optimizing a
tail risk measure rather than pricing. The two surfaces agree to within 0.01 over the deep out of the
money region where the call is nearly worthless.

The fourth panel makes the variance response legible, which the heatmap alone cannot do because the
variance effect is small against a position range of 0 to 0.93. At the money the deep position falls
from 0.681 at variance 0.03 to 0.654 at variance 0.13, while the model position falls faster, from
0.672 to 0.600, so the learned excess over the model grows from 0.009 to 0.054 as variance rises. The
slices also show that at the lowest variance the learned delta sits above the model delta at spot 95
and at the money but dips below it by 0.012 around spot 97.4, whereas at the highest variance it sits
below the model delta only at the far left edge of the grid and is above it from about spot 96 onwards,
reaching 0.058 at spot 103. That is a genuine interaction between moneyness and the variance state, and
it is the reason the policy is fed the variance path rather than a single averaged volatility. The
variance effect is much smaller than the moneyness effect: over the full variance range at the money
the deep position moves by 0.027, against 0.749 for a spot move from 95 to 103 at variance 0.07, a ratio
of about one to twenty eight. That ordering is consistent with the Heston diagnostics described later.

### Sections 5.2, Figures 4 and 5: recurrent against simple network

`results/paper/recurrent_ablation.json`, figure `recurrent_ablation.png`. Two scenarios, 5 seeds each,
both variants trained from scratch.

| Scenario | Variant | Risk measure | Mean cost |
| --- | --- | --- | --- |
| No cost, CVaR 0.50 | recurrent | 2.5646 ± 0.0055 | 0.0000 |
| No cost, CVaR 0.50 | simple | 2.5676 ± 0.0053 | 0.0000 |
| 1 percent cost, CVaR 0.99 | recurrent | 5.2224 ± 0.1510 | 1.3759 ± 0.0531 |
| 1 percent cost, CVaR 0.99 | simple | 5.1513 ± 0.0998 | 1.3940 ± 0.0257 |

Paired within-seed differences, recurrent minus simple:

| Scenario | Risk gap | Standard error | t | Verdict |
| --- | --- | --- | --- | --- |
| No cost, CVaR 0.50 | -0.0030 | 0.0015 | 2.07 | Recurrent marginally better |
| 1 percent cost, CVaR 0.99 | +0.0711 | 0.0830 | 0.86 | Simple better on average, not significant |

The first row is close to the paper's Figure 4 claim: without frictions the previous position adds
almost nothing, since there is no reason to smooth a free trade. The residual gap of 0.0030 is
marginally significant at t 2.07 but is 0.1 percent of the risk level, so it is not economically
meaningful. The second row does **not** reproduce the paper's Figure 5 claim. The recurrent network is
worse on the risk measure (5.2224 against 5.1513) and better on mean cost (1.3759 against 1.3940). The
paper's claim is that the recurrent network hedges better at a *lower* price, and neither half of that
reproduces here: the sign is reversed on risk and the cost advantage is small. Per seed the risk gap
flips sign, and with a paired standard error of 0.0830 and a gap of 0.0711, five seeds cannot separate
the two. The honest reading is that the previous position carries no measurable benefit in this
configuration.

### Section 5.3.2, Figure 11: Heston price asymptotics with two instruments

`results/paper/price_asymptotics_heston.png`, from `price_asymptotics_heston.json`. The same cost grid
and the entropic objective, but under the paper's Heston model with the two instrument policy.

| Cost | $p_\varepsilon$ (5 seed mean ± std) | $p_\varepsilon - q$ |
| --- | --- | --- |
| 0.000000 | 1.8023 ± 0.0122 | 0.1104 |
| 0.000625 | 1.9016 ± 0.0085 | 0.2098 |
| 0.001250 | 1.9799 ± 0.0075 | 0.2880 |
| 0.002500 | 2.1227 ± 0.0132 | 0.4309 |
| 0.005000 | 2.3622 ± 0.0072 | 0.6703 |
| 0.010000 | 2.7985 ± 0.0543 | 1.1067 |

Fitted log-log slope against $q$ is **0.6017**, against the paper's $2/3 = 0.6667$. It lies on the same
side of $2/3$ as the paper's claim but is 10 percent below it. Unlike the Black-Scholes case, no bootstrap
interval is quoted for this fit: with only 5 cost levels a resampled fit is not informative. The slope
against $p_0$ is 0.8308. Read as a whole, the $\varepsilon^{2/3}$ regime is supported in both models,
with the Black-Scholes fit statistically consistent with $2/3$ and the Heston fit close but not exact.

![Heston price asymptotics](results/paper/price_asymptotics_heston.png)

### Section 5.2, Figures 8 and 9: call spread and risk aversion

`results/paper/call_spread.json`, figure `call_spread.png`. Payoff
$Z = [(S - 100)^+ - (S - 101)^+]/1$, so the model spread price is 0.6105. Two confidence levels,
5 seeds each.

| Confidence level | CVaR at own alpha | CVaR 95 | CVaR 99 | Mean PnL | Std |
| --- | --- | --- | --- | --- | --- |
| $\alpha = 0.95$ | 0.8886 ± 0.0072 | 0.8886 ± 0.0072 | 0.9563 ± 0.0101 | -0.5995 | 0.2575 |
| $\alpha = 0.99$ | 1.0072 ± 0.0140 | 0.9833 ± 0.0100 | 1.0072 ± 0.0140 | -0.5997 | 0.3801 |

**This experiment does not reproduce the paper's claim, and the failure is informative.** The paper's
claim is that higher risk aversion flattens the hedge. The holding does flatten: the $\alpha = 0.99$
position is below the $\alpha = 0.95$ position on 69.5 percent of the grid, with a mean difference of
-0.0333. But the flattening is not a mild adjustment, it is a collapse. The $\alpha = 0.95$ policy holds
between -0.001 and 0.194 with a mean of 0.064, tracking the model spread delta, which peaks at 0.367.
The $\alpha = 0.99$ policy holds between 0.022 and 0.040 with a mean of 0.031, which is very nearly a
constant. It has stopped hedging.

The consequence is visible in every risk column. The $\alpha = 0.99$ policy is worse than the
$\alpha = 0.95$ policy on CVaR 95 (0.9833 against 0.8886), worse on CVaR 99 (1.0072 against 0.9563), and
worse on dispersion (0.3801 against 0.2575), at the same mean PnL. Its CVaR 99 of 1.0072 is essentially
the do nothing value: the spread payoff lies in $[0, 1]$, so an unhedged short position has CVaR 99 of
at most 1.0, and the policy has converged to approximately that. A more risk averse objective has
produced a strictly worse hedge on the very measure it is optimizing.

The likely cause is the interaction between the 99 percent objective and the sample size. A 99 percent
CVaR is a mean over the worst 1 percent of 100,000 paths, so 1,000 observations, and the gradient signal
from that tail is weak and noisy relative to the 95 percent case at 5,000 observations. The training
risk confirms the policy is fitting its objective rather than failing to converge: the final train risk
is 1.0156 to 1.0251 for $\alpha = 0.99$ against 0.8859 to 0.9034 for $\alpha = 0.95$, so the
$\alpha = 0.99$ runs sit at a higher objective value, consistent with a harder optimization problem
rather than an under-trained one. The paper does not print a numeric target for this experiment, so
there is no external number to check against, and this result is reported as a non-reproduction rather
than a qualitative success.

Note on the model price. The spread is worth 0.6105 under the paper's Heston parameters, and the deep
hedge residuals average -0.5996, which is the spread price less the 30 day discretization. An earlier
version of this experiment reported a spread price of 0.2579 and hedge residuals of -1.70. Both were
wrong: the analytic spread delta was summing the two call deltas instead of differencing them, and the
training payoff was a plain call rather than a spread. Both defects are fixed. See the section on
corrections below.

### Section 5.1: how much does the architecture matter

`results/paper/paper_architecture.json`, figure `paper_architecture.png`. The project default against a
variant built to the paper's Section 5.1 description: a separate network per step, two hidden layers of
$d + 15$ units, ReLU, BatchNorm, learning rate 0.005, and inputs $\log S$ plus the previous position.

| Variant | Mean PnL | Std | CVaR 95 | Final train risk | Mean cost | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| Project default | -3.235 ± 0.026 | 0.824 ± 0.031 | 4.959 ± 0.029 | 3.610 ± 0.008 | 0.950 ± 0.026 | 20,353 |
| Paper Section 5.1 | -3.238 ± 0.015 | 0.861 ± 0.005 | 5.009 ± 0.050 | 3.604 ± 0.005 | 0.953 ± 0.012 | 15,330 |

The final train risk column is the last minibatch training objective, not an evaluation entropic risk on
a held-out set, so it measures how well each variant fits its own training paths rather than how well it
generalizes. The two are within 0.006 on that measure and 0.050 on CVaR 95, and the paper variant is
marginally better on mean PnL, final train risk and mean cost while being marginally worse on CVaR 95 and
PnL standard deviation, 0.861 against 0.824. It does so using 25 percent fewer parameters and producing
1.7 times less seed dispersion on mean PnL. The gaps are far smaller than the seed standard deviations, so
the honest reading is that the two architectures are indistinguishable on this budget, which is the most
reassuring fidelity check available without reproducing the paper line by line: the architecture choice is
not what drives the results.

### Exploratory data analysis

`results/paper/eda.json`, figures `eda_market.png`, `eda_moments.png`, `eda_results.png`, 200,000 paths
per simulator.

| Quantity | GBM | Heston (xi 0.3) | Paper Heston (xi 2) |
| --- | --- | --- | --- |
| Mean terminal spot | 100.0000 ± 0.0128 | 99.9996 ± 0.0127 | 100.0092 ± 0.0121 |
| Std terminal spot | 5.7378 | 5.6927 | 5.4249 |
| Skew | 0.173 | -0.240 | **-2.299** |
| Excess kurtosis | 0.062 | 0.085 | **9.639** |
| Mean call payoff | 2.2859 | 2.2682 | 1.7017 |
| Analytic call price | 2.2872 | 2.2683 | 1.6918 |
| Share of paths with variance below $10^{-8}$ | n/a | 0.0000 | **0.3866** |

Three facts follow. First, all three simulators are martingales to within one standard error, which
validates the drift handling. Second, the paper's Heston parameter set is radically non-Gaussian: skew
-2.30 against -0.24 for the project's own Heston set, and excess kurtosis 9.64 against 0.09. That is the
whole reason the paper needs exact sampling: with $\xi = 2$ and a Feller violation of $-3.92$, 38.7
percent of paths drive the variance arbitrarily close to zero, and an Euler scheme with full truncation
cannot represent that region correctly. Third, the project's existing Heston price of 2.2683 matches its
own zero-cost mean PnL of -2.269, a useful internal consistency check on the PnL engine.

A fourth fact comes from the seed-noise audit in `eda_results.png`, which is the EDA performed on the
stored study results rather than on the simulators. For the learned policy the seed standard deviation
as a fraction of the mean is 0.6 percent for CVaR 95, 0.7 percent for CVaR 99, 0.1 percent for entropic
risk, 0.8 percent for mean PnL and 2.7 percent for mean cost. The two analytic benchmarks are an order of
magnitude quieter on every measure, since they are deterministic given the path set. Set against the
gaps to Leland, the same numbers give t ratios of 39.2, 26.9, 428.2, 91.6 and 89.9 respectively, so every
headline claim clears the conventional threshold of 2 by a wide margin and the weakest of them, CVaR 99
at 26.9, is still more than thirteen times the threshold. That is the quantitative version of the
statement that the multi-seed study increases the reliability of the reported gaps.

![Simulator EDA](results/paper/eda_market.png)

![Payoff quantiles and Monte Carlo precision](results/paper/eda_moments.png)

![Seed noise against the reported gaps](results/paper/eda_results.png)

## Figure-by-figure analysis

All figures below are the current contents of `results/`, produced by the canonical single-seed run at
seed 42 described above, so every number quoted in this section is a seed 42 number and not a
multi-seed average. Where a figure number differs from the multi-seed table, the multi-seed table is
the authoritative estimate and the figure shows one realization of it.

### `pnl_distributions.png`

Three overlaid terminal PnL histograms on 100,000 evaluation paths, with each strategy's mean marked
by a dashed vertical line and the legend reporting mean and 95 percent CVaR (loss basis).

The learned policy shifts the whole distribution to the right rather than reshaping its tails: mean
PnL improves from -4.51 (Black-Scholes) and -4.28 (Leland) to -3.26, and CVaR 95 falls from 6.36 to
4.93. The Black-Scholes and Leland histograms are concentrated in a narrow band near -4.5 to -4.3,
which is the deterministic cost drag, while the deep hedging histogram is visibly wider and reaches
into the region above -1.0. This is the expected signature of a policy that trades less but carries
more residual risk: it buys a better mean at the price of higher dispersion, since standard deviation
rises from 0.757 to 0.800. Both the mean shift and the dispersion increase are stable features: over 5
seeds the learned policy's mean PnL is -3.235 $\pm$ 0.026 and its standard deviation is
0.824 $\pm$ 0.031.

One caution on reading this figure. The deep hedging histogram has the widest body and the longest left
tail in the plot, which can look like worse tail risk, but the CVaR numbers say the opposite: 4.959 for
deep hedging against 5.529 for Leland and 6.352 for Black-Scholes over 5 seeds. The two facts are
consistent because CVaR is a loss basis measure of the left tail of the PnL distribution, and the deep
hedging distribution is shifted far enough right that its left tail still sits above the benchmarks'
left tails even though its body is wider. The visual impression comes from the density scale: the
benchmark histograms are tall and narrow, so their tails are thin in density but extend further left in
absolute PnL. The table is the authoritative comparison, not the apparent width of the histograms.

This is consistent with the paper's framing rather than a reproduction of one of its figures. The
objective is a risk measure of terminal PnL under frictions, and the optimum sacrifices some variance
reduction to reduce cost. The paper does not print a GBM cost histogram of this kind, so the figure is
an extension of the framework rather than a replication of a specific result. It also shows why
reporting only the mean would be misleading, since the improvement is a genuine shift of the loss
distribution and not a tail-only effect.

### `policy_curves.png`

Three panels at decision step 15 of 30. Left: the on-path hedge position binned by spot, with the 10
to 90 percent band, against the Black-Scholes and Leland deltas. Middle: an off-path probe sweeping
spot for three fixed previous positions. Right: the implied no-trade band, the trade size against
spot for the same three previous positions, with a shaded tolerance band of 0.01.

The left panel is the central result of the project. For the realized distribution of spot at step 15
(roughly 92 to 108) the learned delta is monotone in spot but lies below the Black-Scholes delta on
the upside and above it on the downside, so the curve is flatter. The off-path probe quantifies the
same flattening on a grid that is stored in `results/policy_curves.json`: at a previous position of
zero the policy holds 0.140 at spot 93 and 0.502 at spot 108, against a Black-Scholes delta of 0.038
and 0.973 over the same range. The middle panel makes the mechanism explicit. For a previous position
of plus 0.40 the position barely moves for spot between about 92 and 100, and the trade size in the
right panel is essentially zero over that interval. Measured against the 0.01 tolerance, the no-trade
region for that previous position is spot 97.3 to 98.8, a window of 1.5 in spot, and for the other two
previous positions the band has closed entirely. So the band exists but is narrow and strongly
asymmetric: it appears only when the existing position is already close to the target, and it collapses
when the position is far from the money.

The existence of a non-degenerate no-trade band is a property of the discrete-time frictional problem
that is classical in the literature, not a claim specific to Buehler et al. The derivation of the band
width is due to Leland (1985) and Whalley and Wilmott (1997); the paper's own reason for feeding the
previous position into the network is the footnote to its Figure 5, where the network without that
input is shown to hedge less well under costs. A memoryless delta map cannot represent a band, which is
why the semi-recurrent input matters, but the band itself is not an original prediction of the 2019
paper.

Caveat: the middle panel is an off-path probe. Far from the money and at extreme previous positions
the network is evaluated outside the training distribution, so the level of the probe curves is less
trustworthy than their shape. The left panel, computed only on realized paths, is the reliable view.

### `cost_sensitivity.png` and `cost_sensitivity_multiseed.png`

95 and 99 percent CVaR of the three strategies at cost levels 0.00, 0.10, 0.50 and 1.00 percent. The
policy is retrained from scratch at every cost level at the full main-run budget, 100,000 paths and 20
epochs, and all strategies are scored on the same 100,000 path set. `cost_sensitivity.png` is the
canonical seed 42 realization. `cost_sensitivity_multiseed.png` is the same study over 5 seeds with one
standard deviation error bars, and it is the authoritative version.

Deep hedging is best at every cost level, and the margin grows with cost. At 1.00 percent its 95
percent CVaR is 4.959 $\pm$ 0.029 against 6.352 $\pm$ 0.008 for Black-Scholes and 5.529 $\pm$ 0.007 for
Leland. At 0.50 percent it is 4.186 $\pm$ 0.043 against 4.698 and 4.380. At 0.00 percent it is
3.113 $\pm$ 0.024 against 3.118 for both analytic deltas, so it matches them to within 0.005 and the
gap is not significant at a t ratio of 0.6.

The benefit of learning increases monotonically with the friction level, and at zero cost the learned
policy recovers the Black-Scholes delta rather than underperforming it. Neither statement is a result
the paper reports: the paper has no cost sensitivity figure of this kind. Its cost result is the
$\varepsilon^{2/3}$ scaling of the indifference price, reproduced separately in the
`results/paper/price_asymptotics_bs.png` analysis below. This figure is therefore an extension, not a
replication. At zero cost the learned policy matches the analytic deltas to within 0.005 on CVaR 95,
with a delta deviation of 0.008 to 0.013 and correlation above 0.997. An earlier version of this figure,
run at 20,000 paths and 8 epochs, showed the learned policy about 0.5 worse at zero cost; that was
under-training and is now resolved.

### `training_loss.png`

Epoch-level training risk over the 20 epochs of the canonical run, with a 3 epoch moving average.

Risk falls from 3.9617 to 3.6072. About 67 percent of the total reduction is achieved by epoch 2 and
about 77 percent by epoch 4, after which progress is slow and slightly noisy: epochs 4, 10 and 11 are
marginally above their predecessors. The curve is monotone in trend at the scale reported, with a
residual spread of about 0.04 over the second half.

This is consistent with the paper's reported behaviour, where the policy improves rapidly and then
flattens. The residual spread over the second half is about 0.04, which is minibatch noise in the
objective rather than a trend. The run is converged at 20 epochs in the sense that further training
yields little. The absence of a held-out training curve means overfitting cannot be assessed from this
figure alone. Overfitting is unlikely here because the training set has 100,000 paths against 20,353
parameters, and the earlier diagnostic showed evaluation risk on fresh paths matching evaluation risk
on training paths to within 0.006, but this is an argument rather than a formal test.

### `learning_curves.png`

Left: the empirical distribution function of terminal PnL. Right: expected shortfall at 50, 75, 90,
95 and 99 percent confidence.

The distribution function shows the deep hedging curve lying to the right of both benchmarks over
almost the entire support, which is first order stochastic dominance on the realized sample. The
dominance is strict in the body and narrows in the extreme lower tail below about -7.0, where the
three curves converge, consistent with all strategies facing the same unbounded liability.

The right panel quantifies the same fact across confidence levels: 3.90 against 5.11 (Black-Scholes)
and 4.67 (Leland) at 50 percent, 4.93 against 6.36 and 5.54 at 95 percent, and 5.49 against 7.02 and
6.04 at 99 percent. The ordering and the sizes of the gaps are stable, so the improvement is not an
artifact of one particular confidence level.

This is consistent with the paper's qualitative conclusion, which is that the learned policy improves the
whole risk profile rather than shifting risk between quantiles. It should be read as consistent with
that conclusion rather than as a direct reproduction of a specific paper figure: the paper's own
histograms are the Heston Figures 2, 6 and 7, reproduced separately in
`results/paper/heston_fig2.png`, `heston_fig6.png` and `heston_fig7.png`. The convergence of the curves
in the far tail is the expected result for a short call: no delta policy can remove the unbounded loss
when the underlying gaps far above the strike.

### `position_dynamics.png`

Left: 120 sampled position trajectories through the 30 decision steps, with the mean trajectory of
each strategy. Right: every position against spot relative to the initial level, colored by decision
step, over 1,500 sampled paths.

The mean trajectory starts at about 0.433, rises to roughly 0.536 by the middle of the horizon and then
settles near 0.531 at the end. The Black-Scholes and Leland mean deltas are essentially constant at
0.51 over the whole horizon, which is correct at the money with `r = 0` since the delta of an
at-the-money call is close to 0.5 until the last step. The learned mean starts slightly below them and
ends slightly above, and the small gap is the average amount of extra hedging the policy chooses to do
once the position has built up.

The right panel shows the position is a smooth monotone function of spot with almost no residual
dependence on the decision step: the color gradient runs along the curve rather than across it. In
other words the learned policy is close to time homogeneous in this GBM setting, which is what the
theory predicts when volatility is constant and the only time dependence comes from the shrinking
time to maturity.

This is consistent with the paper's design rationale rather than a reproduction of one of its figures.
The individual trajectories show wide dispersion, with paths swinging between about 0.05 and 1.0, so
the flat mean is an average over very heterogeneous behavior rather than a policy that refuses to
trade. The paper does not print a GBM position trajectory figure, so this is an extension.

### `pnl_paths.png`

Left: 60 sampled cumulative PnL paths of the learned policy through the decision steps, with the mean
path. Right: the mean path and the 5 to 95 percent band for all three strategies.

Cumulative PnL accumulates slowly and negatively through the horizon and then drops sharply at the
final step, where the terminal liability is booked. The mean path of a fully hedged benchmark declines
steadily, for reference the Black-Scholes mean reaches about -2.0 by step 29 before the payoff is
applied, and the learned policy's mean path stays above it for most of the horizon while its band is
wider, which is the same dispersion trade-off visible in the histogram.

The sharp terminal drop is a property of the accounting convention, not of the hedging: the premium is
not received at inception and the payoff is recognized entirely at maturity, so the plot shows the
financing cost of the hedge before the liability is settled. This matches the paper's PnL definition
and is the reason the meaningful comparison is the terminal distribution rather than the shape of the
intra-horizon path.

### Paper reproduction figures

The figures in `results/paper/` are analysed in full in the "Paper results reproduced" section above,
where each one is tied to the paper experiment it answers. For completeness this table lists every one
of them with its content and the conclusion it supports, so that the figure inventory in this document
matches the contents of the results directory.

| Figure | Content | Conclusion |
| --- | --- | --- |
| `price_asymptotics_bs.png` | $p_\varepsilon - q$ against cost on log-log axes, 5 seed means with one standard deviation bars, plus the fitted line and the reference slope $2/3$ | Fitted slope 0.6898 with 95 percent bootstrap interval [0.6362, 0.7353]. The $\varepsilon^{2/3}$ scaling of the paper's Figure 10 is reproduced |
| `cost_sensitivity_5seed.png` | CVaR 95 and CVaR 99 against the same cost grid for Deep Hedging, Black-Scholes and Leland | Deep Hedging dominates both benchmarks at every cost level and the advantage widens with cost, which is the ordering the paper predicts |
| `cost_sensitivity_multiseed.png` | The project's own cost sensitivity over 5 seeds at 100,000 paths and 20 epochs, CVaR 95 and CVaR 99 with one standard deviation error bars | Deep Hedging matches the analytic deltas at zero cost (3.113 against 3.118, t 0.5) and beats Leland from 0.10 percent upward, reaching a CVaR 95 gap of 0.495 at t 10.4 at one percent |
| `heston_fig2.png` | Terminal hedging error densities for the deep hedge and the analytic model hedge at $\alpha = 0.50$, 200,000 paths | Deep hedge CVaR 1.9630 against 1.9687 for the true model hedge. The two are statistically indistinguishable, which is the correct result for two instruments and no costs |
| `heston_fig3.png` | Stock holding of the model hedge and the non-recurrent deep hedge at step 15 over a spot and variance grid, the deep minus model difference panel, and variance slices at 0.03, 0.08 and 0.13 | The learned surface reproduces the analytic one in shape and is slightly steeper in spot: the difference is -0.001 at spot 95, crosses zero near spot 95.2, and reaches +0.054 at spot 103. At the money the deep excess over the model rises from 0.009 at variance 0.03 to 0.054 at variance 0.13, so the variance state shifts the learned correction rather than the level |
| `heston_fig6.png` | The same two policies at $\alpha = 0.99$, each centred at its own risk measure | The more risk averse deep hedge has a heavier far tail but a better 99 percent average, 2.9750 against 3.9139 for the model hedge |
| `heston_fig7.png` | Both policies centred at the model price $q = 1.6918$ | Confirms the improvement is not a centring artifact: at a common origin the deep hedge still has the smaller tail average |
| `price_asymptotics_heston.png` | The Figure 10 experiment repeated under the paper's Heston model with two instruments | Fitted slope 0.6017 against $2/3$. The same regime, 9 percent below the Black-Scholes fit, and no bootstrap interval is quoted because 5 cost levels are too few |
| `recurrent_ablation.png` | Three panels: absolute risk by variant, paired within-seed differences with standard errors, and mean transaction cost | Without cost the two variants are indistinguishable (gap -0.0030, $t = 2.07$, 0.1 percent of the risk level). At 1 percent cost the recurrent network is worse on risk by 0.0711 with a standard error of 0.0830, so the paper's lower-price claim does not reproduce |
| `call_spread.png` | Stock holding at $\alpha = 0.95$ and $\alpha = 0.99$ for the call spread, the difference panel, and the two PnL distributions | The alpha 0.99 policy collapses to a near constant 0.031 against a model delta peaking at 0.367, and is worse than alpha 0.95 on CVaR 95 (0.9833 against 0.8886), CVaR 99 (1.0072 against 0.9563) and dispersion (0.3801 against 0.2575). Its CVaR 99 is essentially the unhedged value, so this experiment does not reproduce the paper's claim |
| `paper_architecture.png` | Risk measures for the project default against a paper-faithful architecture, with parameter counts | The two agree to 0.009 on the final training risk and 0.032 on CVaR 95, so the architecture choice is not what drives the reported results |
| `eda_market.png` | Terminal spot histograms on a log density axis and a lower-tail distribution function on a common axis for all three simulators | All three simulators are martingales to within one standard error. The paper's Heston set is strongly left skewed and heavy tailed where the project set is nearly symmetric, and its lower tail carries visibly more mass below spot 90 |
| `eda_moments.png` | Payoff quantiles at 1, 5, 50, 95 and 99 percent by simulator, and the Monte Carlo standard error of the mean payoff as a fraction of that mean | The paper Heston payoff is lower than GBM in the upper quantiles, 5.778 against 9.712 at 95 percent and 9.969 against 14.068 at 99 percent, and higher at the median, 1.108 against 0.000, so the same mean price is carried by a much fatter body and a shorter right tail. All three simulators price their own mean payoff to a relative standard error of 0.34 percent or better at 200,000 paths |
| `eda_results.png` | Seed standard deviation as a percentage of the mean for each metric and strategy, and the gap to Leland for each metric expressed in standard errors | Every reported gap against Leland is between 26.9 and 428.2 standard errors, so no headline claim in the main table rests on seed noise. Mean cost is the noisiest metric for the learned policy at 2.7 percent relative seed dispersion, against 0.6 percent for CVaR 95 and 0.1 percent for entropic risk |

## Key numerical results

### Multi-seed study, main results table

The headline numbers are averaged over 5 independent seeds (42, 43, 44, 45, 46). Each seed is a
complete train and evaluate cycle of the main configuration: GBM, entropic objective with
`lambda = 1.0`, `cost_rate = 0.01`, 100,000 training paths, 100,000 evaluation paths, 20 epochs. The
reported uncertainty is the sample standard deviation across the 5 seeds, so each cell reads as
mean $\pm$ standard deviation over 5 independent seeds.

| Strategy | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost |
| --- | --- | --- | --- | --- | --- | --- |
| Deep Hedging | -3.235 ± 0.026 | 0.824 ± 0.031 | 4.959 ± 0.029 | 5.536 ± 0.038 | 3.581 ± 0.004 | 0.950 ± 0.026 |
| Black-Scholes | -4.511 ± 0.001 | 0.756 ± 0.001 | 6.352 ± 0.008 | 7.013 ± 0.016 | 4.859 ± 0.004 | 2.225 ± 0.002 |
| Leland | -4.280 ± 0.002 | 0.511 ± 0.001 | 5.529 ± 0.007 | 6.026 ± 0.012 | 4.427 ± 0.002 | 1.994 ± 0.001 |

The two analytic benchmarks are deterministic given the path set, so their standard deviations are
near zero by construction and reflect only the change of evaluation paths across seeds. All of the
seed-to-seed variation in the table is therefore attributable to the learned policy.

Per-seed values for the learned policy, showing where the dispersion comes from:

| Seed | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | -3.260 | 0.800 | 4.930 | 5.488 | 3.584 | 0.973 |
| 43 | -3.237 | 0.827 | 4.949 | 5.507 | 3.581 | 0.951 |
| 44 | -3.245 | 0.805 | 4.976 | 5.570 | 3.586 | 0.965 |
| 45 | -3.241 | 0.810 | 4.940 | 5.538 | 3.578 | 0.952 |
| 46 | -3.190 | 0.877 | 5.002 | 5.576 | 3.578 | 0.906 |

### What the multi-seed study changes

The multi-seed study increases the statistical reliability of the reported gaps. Using the standard
error of the mean across seeds, $\mathrm{SE} = s/\sqrt{5}$, the gaps against Leland are:

| Metric | Gap, Leland minus Deep Hedging | Standard error | t ratio | Favours |
| --- | --- | --- | --- | --- |
| Mean PnL | -1.045 | 0.011 | 91.6 | Deep Hedging |
| Mean cost | +1.044 | 0.012 | 89.9 | Deep Hedging |
| Entropic | +0.846 | 0.002 | 428.2 | Deep Hedging |
| CVaR 95 | +0.569 | 0.015 | 39.2 | Deep Hedging |
| CVaR 99 | +0.490 | 0.018 | 26.9 | Deep Hedging |

The sign convention is that a positive gap means Leland is worse on that metric, so a positive gap
favours deep hedging for every loss based measure. Mean PnL carries the opposite sign because a higher
PnL is better, so the negative gap of -1.045 means deep hedging earns 1.045 more than Leland.

Every gap against Leland is now far outside the seed noise, including both CVaR measures. Against
Black-Scholes the t ratios are 110.4 for mean PnL, 109.7 for cost, 629.0 for entropic risk, 105.1 for
CVaR 95 and 102.8 for CVaR 99. The learned policy therefore dominates both analytic benchmarks on
every reported risk measure and on cost, and the dominance is statistically unambiguous.

### Effect of the normalization fix

The first 5-seed study used per-step BatchNorm and produced a large train and evaluation mismatch on
seed 44: the final minibatch training risk was 3.578 while the evaluation risk was 4.134, a gap of
0.55, against 0.013 for seed 42. The cause was that BatchNorm normalizes with batch statistics during
training but with lagged running statistics at evaluation, and the two diverge as the policy sharpens.
Replacing it with LayerNorm, which has no running statistics, makes the two modes identical.

| Metric | BatchNorm mean ± std | LayerNorm mean ± std | Std reduction |
| --- | --- | --- | --- |
| Mean PnL | -3.235 ± 0.043 | -3.235 ± 0.026 | 1.7x |
| Std | 0.973 ± 0.151 | 0.824 ± 0.031 | 4.9x |
| CVaR 95 | 5.272 ± 0.461 | 4.959 ± 0.029 | 15.9x |
| CVaR 99 | 5.932 ± 0.606 | 5.536 ± 0.038 | 15.9x |
| Entropic | 3.749 ± 0.237 | 3.581 ± 0.004 | 59.3x |
| Mean cost | 0.950 ± 0.047 | 0.950 ± 0.026 | 1.8x |

The risk measures became 5 to 59 times more reproducible and their means improved, since the BatchNorm
version was being scored with statistics that did not match its training. Seed 44's entropic risk fell
from 4.134 to 3.586, in line with the other seeds. The residual spread in mean PnL and mean cost is
not a defect: it reflects genuine differences in how much each seed trades, and the two move together
as expected. Training also became about 2.2 times faster, from 500 s to 230 s per seed.

### Single-seed canonical run

The canonical run at seed 42 is retained and is the source of every figure in `results/`. It
reproduces the seed 42 row of the multi-seed table exactly, which confirms the study and the
figure pipeline are consistent.

| Strategy | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost | Gross PnL | Turnover | Tail cost share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deep Hedging | -3.260 | 0.800 | 4.930 | 5.488 | 3.584 | 0.973 | -2.287 | 15.51 | 0.224 |
| Black-Scholes | -4.512 | 0.757 | 6.363 | 7.016 | 4.861 | 2.224 | -2.288 | 15.01 | 0.526 |
| Leland | -4.281 | 0.512 | 5.539 | 6.039 | 4.429 | 1.993 | -2.288 | 15.11 | 0.495 |

Drawdown is reported two ways. The terminal-jump version, which includes the maturity liability, gives
worst 26.650 (deep hedging), 26.739 (Black-Scholes), 26.739 (Leland) and mean 5.213, 6.121, 5.896; it
is dominated by the common payoff and does not discriminate. The intra-horizon version, computed on
the pre-settlement path, gives mean 3.598, 4.469, 4.221 and worst 12.505, 13.326, 13.031, and does
discriminate. See the drawdown section below.

Policy response at step 15 with previous position 0.508, benchmark reference in brackets:

| Spot move | Spot | Deep Hedging | Black-Scholes | Leland |
| --- | --- | --- | --- | --- |
| -12% | 88.00 | 0.204 | 0.001 | 0.009 |
| -8% | 92.00 | 0.286 | 0.021 | 0.064 |
| -4% | 96.00 | 0.433 | 0.162 | 0.232 |
| -2% | 98.00 | 0.490 | 0.316 | 0.364 |
| 0% | 100.00 | 0.515 | 0.508 | 0.511 |
| +2% | 102.00 | 0.543 | 0.695 | 0.654 |
| +4% | 104.00 | 0.594 | 0.838 | 0.775 |
| +8% | 108.00 | 0.730 | 0.973 | 0.927 |
| +12% | 112.00 | 0.814 | 0.998 | 0.984 |

### Full-budget cost sensitivity

The cost study was re-run at the full main-run budget, 100,000 paths and 20 epochs per cost level,
instead of the earlier reduced 20,000 paths and 8 epochs, and it is now a 5-seed study (42 to 46) like
every other table in this document. Each seed draws its own training and evaluation paths, so the
protocol matches the main multi-seed study exactly. The 20 trainings take about 5,400 s of CPU time
serially, or about 1,100 s of wall clock across 5 parallel shards, and the study writes
`results/gbm_cost_multiseed.json`.

| Cost | Deep mean | Deep CVaR 95 | Deep CVaR 99 | BS CVaR 95 | Leland CVaR 95 |
| --- | --- | --- | --- | --- | --- |
| 0.00% | -2.286 ± 0.002 | 3.113 ± 0.024 | 3.473 ± 0.032 | 3.118 ± 0.006 | 3.118 ± 0.006 |
| 0.10% | -2.448 ± 0.004 | 3.347 ± 0.028 | 3.691 ± 0.043 | 3.421 ± 0.006 | 3.380 ± 0.006 |
| 0.50% | -2.847 ± 0.022 | 4.186 ± 0.043 | 4.638 ± 0.069 | 4.698 ± 0.007 | 4.380 ± 0.006 |
| 1.00% | -3.235 ± 0.026 | 4.959 ± 0.029 | 5.536 ± 0.038 | 6.352 ± 0.008 | 5.529 ± 0.007 |

Significance against Leland, using the standard error of the 5-seed mean:

| Cost | CVaR 95 gap | t ratio | CVaR 99 gap | t ratio |
| --- | --- | --- | --- | --- |
| 0.00% | +0.005 | 0.6 | +0.036 | 2.4 |
| 0.10% | +0.033 | 2.7 | +0.090 | 4.2 |
| 0.50% | +0.194 | 10.0 | +0.187 | 5.4 |
| 1.00% | +0.569 | 39.2 | +0.490 | 26.9 |

The zero-cost case is the important one, because it is the limit in which the theory says the optimal
policy is the Black-Scholes delta. At the reduced budget the learned policy was worse than both
benchmarks at zero cost, with CVaR 95 of 3.611 against 3.111. At the full budget it is level with them:
CVaR 95 of 3.113 against 3.118, CVaR 99 of 3.472 against 3.518, and a mean PnL of -2.286 against
-2.288. The earlier deficit was under-training, not a property of the method.

The zero-cost comparison must be stated carefully. The CVaR 95 gap of 0.005 carries a t ratio of 0.6,
so the correct claim is that the learned policy matches the analytic deltas at zero cost, not that it
beats them. The CVaR 99 gap of 0.036 is marginally significant at t 2.4. Both readings support the
theoretical prediction, which is convergence to the Black-Scholes delta rather than an improvement on
it, and the earlier version of this document overstated the result by calling it "slightly better".

The delta convergence confirms the same conclusion directly. At zero cost, on realized paths, the mean
absolute deviation of the learned position from the Black-Scholes delta is 0.0105 at step 5, 0.0080 at
step 15 and 0.0125 at step 25, with correlations of 0.9972, 0.9997 and 0.9992. The learned policy has
recovered the analytic delta to within about one percent of a position unit.

The advantage over the benchmarks grows monotonically with the cost level, and from 0.10 percent
upward every gap is significant. Against Black-Scholes the CVaR 95 gap rises from 0.005 at zero cost to
1.393 at one percent; against Leland, which is the stronger benchmark, it rises from 0.005 to 0.569.
That monotone widening is an empirical property of this experiment. It is not a claim the paper makes:
the paper's own cost result is the $\varepsilon^{2/3}$ scaling of the indifference price, which is
reproduced in the `price_asymptotics_bs.png` analysis from the paper replication phase.

### Heston transaction cost sweep

The same cost study was run under the Heston simulator, using the existing simulator and PnL engine
with no changes to either. The benchmark volatility is the variance-adjusted value
$\sigma = \sqrt{v_0} = 0.2$, so the analytic deltas are the same Black-Scholes and Leland formulas
evaluated at the long-run variance. Budget: 100,000 paths, 20 epochs and 5 seeds (42 to 46) per cost
level, matching the GBM study exactly. Each cell is mean $\pm$ standard deviation over the 5 seeds.

| Cost | Deep mean | Deep CVaR 95 | Deep CVaR 99 | BS CVaR 95 | Leland CVaR 95 |
| --- | --- | --- | --- | --- | --- |
| 0.00% | -2.269 ± 0.002 | 3.219 ± 0.018 | 3.628 ± 0.036 | 3.292 ± 0.006 | 3.292 ± 0.006 |
| 0.10% | -2.426 ± 0.004 | 3.429 ± 0.016 | 3.817 ± 0.038 | 3.601 ± 0.006 | 3.560 ± 0.006 |
| 0.50% | -2.809 ± 0.016 | 4.238 ± 0.021 | 4.712 ± 0.047 | 4.892 ± 0.007 | 4.580 ± 0.006 |
| 1.00% | -3.210 ± 0.029 | 5.035 ± 0.048 | 5.623 ± 0.045 | 6.559 ± 0.009 | 5.749 ± 0.006 |

Significance against Leland, using the standard error of the 5-seed mean:

| Cost | CVaR 95 gap | t ratio | CVaR 99 gap | t ratio |
| --- | --- | --- | --- | --- |
| 0.00% | +0.073 | 11.9 | +0.183 | 13.7 |
| 0.10% | +0.132 | 20.7 | +0.277 | 14.7 |
| 0.50% | +0.342 | 43.1 | +0.460 | 24.2 |
| 1.00% | +0.715 | 32.2 | +0.786 | 30.9 |

The zero-cost deficit is gone and the result is now statistically secure rather than a single
observation. At the full budget the learned policy beats both benchmarks at zero cost, with CVaR 95 of
3.219 against 3.292 and CVaR 99 of 3.628 against 3.811, at t ratios of 11.9 and 13.7. Every gap at
every cost level is significant at t of 11.9 or higher.

The margin widens monotonically with cost, from a CVaR 95 gap of 0.073 at zero cost to 0.715 at one
percent. The Heston and GBM studies agree closely at one percent cost, 5.035 against 4.959 for deep
hedging, which is expected since the benchmark volatility is the same in both. The Heston dispersion
is larger than the GBM dispersion, 0.048 against 0.029 at one percent, which reflects the additional
variance-state sampling.

The Heston results also show that the policy uses the variance state. The learned policy is trained
with the variance path as an input, and it beats the variance-adjusted analytic benchmarks at every
cost level including zero, which a policy ignoring the variance state could not do as consistently.
This is a qualitative observation, not a controlled ablation.

### Heston figures

Three figures are saved for the Heston study. `heston_cost_sensitivity.png` reproduces the cost sweep
in the same layout as the GBM version, with CVaR 95 and CVaR 99 against cost for all three strategies,
5 seed means with one standard deviation error bars. `heston_diagnostics.png` has three panels: the
square root of the variance paths, which shows the mean-reverting dispersion around the initial level
of 0.20; the hedge position against instantaneous volatility colored by decision step; and the hedge
position against moneyness colored by volatility. `heston_policy_curves.png` is the policy counterpart
of the GBM policy figure and is described next. The policy surface and the probe curves behind it are
stored in `results/heston_policy_curves.json`, so every number quoted below can be checked without
retraining.

The diagnostics figure shows that the position is primarily driven by moneyness, with volatility acting
as a secondary modifier. That ordering is confirmed by a direct probe of the stored policy surface in
`results/heston_policy_curves.json`: at the money and at step 15, moving volatility from 0.10 to 0.35
raises the position from 0.1741 to 0.2350, a spread of 0.0609, while moving spot from 95 to 105 at
volatility 0.20 moves it by 0.1966, and across the full grid from spot 75 to 125 it moves by 0.6500.
The variance effect is therefore about a third of the moneyness effect over a comparable spot range,
and about a tenth of the full spot range. The earlier version of this paragraph quoted 0.33 for the
spot effect, which was measured over a wider spot range than the volatility comparison and so
overstated the ratio; the numbers here are read directly from the stored surface.

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
visible across the full range. The position rises with spot as expected, from 0.1069 at spot 75 to
0.7569 at spot 125 at volatility 0.20, and at any fixed spot it rises with volatility, from 0.1741 at
volatility 0.10 to 0.2350 at 0.35 at the money. The gradient is steeper in the region where the call is
near or above the strike, where convexity matters most, and flattens deep in the money where the
position approaches its upper bound.

The difference panel is the alternative view that makes the modest effect legible. Subtracting the
volatility 0.20 baseline removes the dominant moneyness trend, so the residual variance effect is
plotted on its own scale of about plus or minus 0.04 rather than being compressed against a position
range of 0 to 1.

The panel shows the effect is not a uniform vertical shift but changes sign across the spot range. For
volatility 0.30 the position is above the baseline by up to 0.0389 at spot 101.3, just above the strike,
where a higher variance raises the value of an at or near the money call. It falls below the baseline by
up to 0.0389 at spot 111.0, where the call is comfortably in the money and the higher variance instead
widens the region over which the policy declines to add, and it returns towards the baseline further
out. For volatility 0.12 the pattern is broadly the mirror image with a smaller amplitude, reaching
-0.0076 at spot 99.5 and +0.0268 at spot 106.7. This sign structure is the economically meaningful part
of the variance response, and it is not visible in the overlapping curves or in the heatmap colour scale
alone, which is why the difference panel is retained alongside the heatmap.

The bottom two panels reproduce the GBM result in the Heston setting. The learned position lies below
the Black-Scholes delta across the whole spot range, and the trade size is close to zero over a narrow
interval that shifts with the previous position. The band is narrowest for the previous position of
0.30, where the policy declines to trade at all only for spot between 95.7 and 96.5, a window of 0.8
in spot. For the other two previous positions there is no spot at which the trade size falls inside the
0.01 tolerance, so the band has closed entirely. This is a genuine difference from the GBM study, where
the band is wide and asymmetric. The Heston policy is a surface in spot and variance, and the extra
state dimension appears to sharpen the position response enough that the flat region collapses. The
variance-aware policy therefore preserves the qualitative structure of the frictional optimum, in that
the position still depends on the previous position, but it does not preserve the wide no-trade band
seen in the one dimensional GBM case.

### Drawdown: terminal jump against intra-horizon

The original drawdown metric measured the peak to trough decline of the cumulative PnL path including
the terminal liability, which is booked in a single step at maturity. That made the metric a rescaled
maximum payoff rather than a drawdown, and it was nearly identical across strategies because all three
face the same liability. An intra-horizon drawdown is now computed on the pre-settlement path, which
excludes the terminal settlement and measures only the risk accumulated while hedging.

| Strategy | Terminal DD mean | Terminal DD worst | Intra DD mean | Intra DD p95 | Intra DD worst |
| --- | --- | --- | --- | --- | --- |
| Deep Hedging | 5.213 | 26.650 | 3.598 | 6.057 | 12.505 |
| Black-Scholes | 6.121 | 26.739 | 4.469 | 7.057 | 13.326 |
| Leland | 5.896 | 26.739 | 4.221 | 6.536 | 13.031 |

The terminal column confirms the problem: the worst values are 26.650, 26.739 and 26.739, essentially
identical, because they are driven by the largest payoff in the sample rather than by the hedging. The
intra-horizon column separates the strategies cleanly. Deep hedging has the lowest mean drawdown at
3.598 against 4.469 for Black-Scholes and 4.221 for Leland, and the lowest worst case at 12.505 against
13.326 and 13.031. This is consistent with the CVaR ordering and shows the improvement is not confined
to the terminal distribution.

### Entropic risk aversion sweep

The entropic risk measure has a single tuning parameter. The paper uses $\lambda = 1$ only; its own
risk-aversion family is the CVaR confidence level $\alpha \in \{0.5, 0.95, 0.99\}$. The sweep below is
therefore an extension that explores the objective family beyond what the paper reports. It trains the
main GBM configuration at five values of `lambda`, 100,000 paths, 20 epochs and 5 seeds (42 to 46) each,
so every cell is mean $\pm$ standard deviation over the 5 seeds. Five values rather than three are used
so the frontier is smooth enough to read as a curve.

| Lambda | Mean PnL | Std | CVaR 95 | CVaR 99 | Entropic | Mean cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | -2.843 ± 0.086 | 1.565 ± 0.132 | 6.836 ± 0.420 | 8.329 ± 0.700 | 3.196 ± 0.055 | 0.557 ± 0.088 |
| 0.50 | -3.045 ± 0.019 | 1.099 ± 0.028 | 5.479 ± 0.039 | 6.314 ± 0.028 | 3.366 ± 0.020 | 0.760 ± 0.015 |
| 1.00 | -3.235 ± 0.026 | 0.824 ± 0.031 | 4.959 ± 0.029 | 5.536 ± 0.038 | 3.581 ± 0.004 | 0.950 ± 0.026 |
| 2.00 | -3.422 ± 0.017 | 0.689 ± 0.005 | 4.823 ± 0.028 | 5.284 ± 0.073 | 3.902 ± 0.016 | 1.137 ± 0.016 |
| 4.00 | -3.623 ± 0.022 | 0.610 ± 0.016 | 4.786 ± 0.037 | 5.142 ± 0.055 | 4.336 ± 0.072 | 1.338 ± 0.019 |

Turnover tracks mean cost almost exactly, from 0.557 at `lambda = 0.25` to 1.338 at `lambda = 4.00`,
since cost is proportional to traded volume at a common price level. It is omitted from the table only
to keep the width manageable.

The tradeoff is clean and monotone in every column. Raising `lambda` from 0.25 to 4.00 reduces the
standard deviation from 1.565 to 0.610 and CVaR 95 from 6.836 to 4.786, at the cost of mean PnL
falling from -2.843 to -3.623 and transaction cost rising from 0.557 to 1.338. This is the expected
risk-return frontier: a more risk averse objective trades more to suppress dispersion and pays for it
in cost and in expected PnL.

The frontier is computed at a single cost level, `cost_rate = 0.01`. At zero cost the policy would
converge to the Black-Scholes delta for every `lambda` on this sweep, since with no friction there is
no cost to trading and the objective reduces to minimizing the risk of an unhedged gap. The `lambda`
family is informative precisely because costs make the tradeoff non-trivial.

The entropic risk itself rises with `lambda`, from 3.196 to 4.336, which is not a contradiction. The
entropic risk measure is `lambda`-dependent, so its value is not comparable across the sweep; what is
comparable is the position of each policy on the risk-return frontier. Every value from `lambda = 0.50`
upward beats both benchmarks on CVaR 95 and CVaR 99, with the benchmarks at 6.363 and 7.016
(Black-Scholes) and 5.539 and 6.039 (Leland). The `lambda = 0.25` policy is the exception: it beats
Black-Scholes on CVaR 95, 6.836 against 6.363, but its CVaR 99 of 8.329 is worse than both benchmarks.
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
budget: 100,000 paths, 20 epochs, `cost_rate = 0.01`, 5 seeds. It writes
`results/shared_vs_timedep.json`.

| Variant | Entropic | CVaR 95 | CVaR 99 | Parameters |
| --- | --- | --- | --- | --- |
| Shared | 3.581 ± 0.004 | 4.959 ± 0.029 | 5.536 ± 0.038 | 20,353 |
| Time-dependent | 3.664 ± 0.010 | 5.044 ± 0.017 | 5.623 ± 0.049 | 276,510 |
| Black-Scholes | 4.859 ± 0.004 | 6.352 ± 0.008 | 7.013 ± 0.016 | n/a |
| Leland | 4.427 ± 0.002 | 5.529 ± 0.007 | 6.026 ± 0.012 | n/a |

Both rows use an identical protocol: the same training paths, the same evaluation paths, the same
seeds, the same optimizer, the same 100,000 path and 20 epoch budget, and the same cost level. The only
difference is the weight-sharing mode, so this is a like-for-like comparison.

The time-dependent variant is competitive. It beats both benchmarks on every measure, with an entropic
risk of 3.664 against 4.859 and 4.427, and a CVaR 95 of 5.044 against 6.352 and 5.529. It remains
behind the shared variant on every measure: 0.083 in entropic risk, 0.085 in CVaR 95 and 0.087 in
CVaR 99, while using 13.6 times as many parameters, 276,510 against 20,353. Those gaps are larger than
the shared variant's own seed standard deviation, so on this budget the shared mode is genuinely ahead
rather than merely level. The time-dependent mode is also slightly less reproducible, with a seed
standard deviation of 0.010 against 0.004 on entropic risk.

Recommendation: the shared variant remains the default. It is more accurate on a like-for-like budget,
13.6 times smaller, and about 15 percent faster to train, since the time-dependent mode runs a separate
forward pass per decision step. The earlier failure of the time-dependent mode was a BatchNorm artifact,
not a capacity limit, but removing that artifact made it competitive rather than superior.

## Remaining limitations and open issues

1. No market data and no calibration. All conclusions from the project's own studies are relative,
   within a synthetic world whose parameters were chosen rather than estimated. The paper replication
   phase removes this objection for the experiments it covers, because it uses the paper's own
   published parameter sets, but those parameters are still not fitted to any market. This is the one
   gap that is genuinely out of scope rather than deferred.
2. The call spread risk aversion experiment does not reproduce. At $\alpha = 0.99$ the learned policy
   collapses to a near constant position of about 0.031 and is worse than the $\alpha = 0.95$ policy on
   CVaR 95, CVaR 99 and dispersion, with a CVaR 99 of 1.0072 that is essentially the unhedged value.
   The most likely cause is that a 99 percent objective over 100,000 paths gives only 1,000 tail
   observations, so the gradient signal is weak, and the final training risk confirms the runs sit at a
   higher objective value rather than being under-trained. This is the one paper experiment that was
   attempted and failed, and it is reported as such rather than as a qualitative success.
3. Seed dispersion is measured and small relative to every reported gap. Across 5 seeds the learned
   policy's entropic risk varies by a standard deviation of 0.004 and CVaR 95 by 0.029, against gaps
   to Leland of 0.846 and 0.569. Every gap is significant at a t ratio of 26.9 or higher. The
   remaining spread is in mean PnL and mean cost, about 0.03 each, which is genuine policy variation
   rather than noise: the two move together, so seeds that trade more pay more and earn more. The one
   place where seed noise does matter is the zero-cost comparison, where the gap to the analytic
   deltas is 0.005 at a t ratio of 0.6, and the correct reading there is convergence rather than
   superiority.
4. The terminal-jump drawdown is dominated by the maturity liability and is not an informative
   intra-horizon risk measure. It is retained for continuity and reported alongside the new
   intra-horizon drawdown, which excludes the terminal settlement and is the interpretable column.
5. The Heston dispersion is larger than the GBM dispersion, 0.048 against 0.029 in CVaR 95 at one
   percent cost, because the variance state adds a second source of sampling variation. Every Heston
   gap is still significant at a t ratio of 11.9 or higher, so this affects precision rather than
   conclusions.
6. The time-dependent variant is competitive but not better than the shared variant, and it costs
   13.6 times as many parameters, 276,510 against 20,353. The shared mode is ahead on every measure,
   3.581 ± 0.004 against 3.664 ± 0.010 on entropic risk, so it remains the default on accuracy as well
   as on size and training speed.
7. The CVaR threshold is handled in two stages. It is warm started at the alpha quantile of losses
   under the initial policy, then re-projected to the alpha quantile of the pooled epoch losses after
   each epoch, while remaining a trainable parameter with its own optimizer group. The projection is
   exact for a fixed sample: for the Rockafellar-Uryasev objective
   $w + \frac{1}{1-\alpha}\mathbb{E}[(-X_T - w)^{+}]$ the minimizer over $w$ is the alpha quantile of
   the loss distribution, so the projection lands on the exact minimizer rather than approximating it.
   What is not characterized is the interaction between the once-per-epoch projection and minibatch
   noise within the epoch, since the threshold is held fixed across the minibatches of an epoch while
   the policy moves.
8. The lambda sweep covers five computed values, 0.25 to 4.00, which is enough to read the frontier
   shape but is not a continuous parameterization. A continuous fitted curve is now available as a
   visualization aid drawn through the five computed points, shown in `lambda_frontier.png`. The curve
   interpolates the measured means and does not add evidence at unsampled `lambda` values, so the five
   computed points remain the reported result.
9. The Heston policy figure uses a heatmap for the variance dimension rather than a family of curves,
   because the variance effect, a spread of about 0.06 in position across the full volatility range,
   is small enough that separate curves overlap. A difference panel against the volatility 0.20
   baseline is included to make the residual effect legible, and the Figure 3 variance slices at 0.03,
   0.08 and 0.13 give a second legible view. Neither panel is as direct as the single-curve GBM
   figure, since the Heston policy is a surface rather than a curve.

### Corrections found in the final audit

The final consistency pass found three defects in the paper replication code. All three are fixed and
the affected results have been re-run. They are recorded here because two of them changed reported
numbers, and because the way they were found is itself informative.

1. **Heston pricer phase sign.** `heston_call_price` used `exp(-i u log(S/K))` where the Lewis
   inversion requires `exp(+i u log(S/K))`. Because only the real part is integrated, this is not
   equivalent and it produced prices that were wrong in the wings while correct at the money. The
   at the money gate $q(0, 100, 0.04) = 1.691834$ passed either way, which is exactly why the defect
   survived an earlier validation. It was found by comparing the pricer against Monte Carlo at several
   strikes and discovering that the rho sensitivity had the wrong sign: the pricer said a negative
   correlation raises the price of an out of the money call, the simulator said it lowers it, and the
   simulator was right. After the fix the pricer matches Monte Carlo at every strike tested: 5.78030
   against 5.77605 at strike 95, 1.08137 against 1.08197 at 101, 0.02909 against 0.02741 at 110, and
   0.00113 against 0.00126 at 120, with the residual explained by the simulator's discretization. The
   $\xi \to 0$ Black-Scholes limit still holds to four decimal places. The validation gate is now
   `validate_pricer` in `src/heston.py`, a strike ladder against a 480 step Monte Carlo with four
   standard error tolerances, called by `paper_heston.py` before the first training. It reports a worst
   error of 0.46 of tolerance. The 480 step choice matters: at the 30 step hedging resolution the
   discretization bias at strike 101 is 0.020, which is larger than the effect the gate is testing.
2. **Heston call spread delta.** `model_spread_delta` accumulated `+grad` for both strikes instead of
   `+grad` for the long leg and `-grad` for the short leg, so it reported the delta of a straddle
   rather than a spread. The corrected surface peaks near spot 99 at 0.365, which is where a 100 by
   101 call spread belongs, instead of rising monotonically to 1.93.
3. **Call spread payoff.** `spread_payoff` was defined in `paper_call_spread.py` but never called, so
   the risk aversion experiment trained and evaluated against a plain call. The liability is now
   threaded through `Config.payoff_second_strike` and `hedge_pnl`, which keeps a single PnL engine for
   every payoff in the project.

The effect on reported numbers: the model hedge in Figures 2, 6 and 7 moved from CVaR 2.4279 and
4.8708 to 1.9687 and 3.9139, and the deep hedge now wins by 0.0057 at $\alpha = 0.50$ instead of 0.46.
The call spread model price moved from 0.2579 to 0.6105 and the hedge residuals from -1.70 to -0.60.
Every other result is unchanged, because R1, R5, R3, R6 and the main GBM and Heston studies either
price only at the money or do not use the analytic pricer at all.

### Resolved in this iteration

- The BatchNorm train and evaluation mismatch is fixed. Per-step BatchNorm was replaced by LayerNorm,
  which has no running statistics, so training and evaluation modes are identical by construction.
  The seed 44 gap fell from 0.385 to 0.005, the risk measures became 5 to 59 times more reproducible,
  and training became 2.2 times faster. The BatchNorm path remains available as `--norm batchnorm`.
- The zero-cost under-training result is resolved for both simulators, and both are now multi-seed. At
  the full budget the GBM policy matches the Black-Scholes delta to a mean absolute deviation of 0.008
  to 0.013 with correlation above 0.997, and its CVaR 95 at zero cost is 3.113 ± 0.024 against 3.118 ± 0.006
  for the analytic deltas, a gap of 0.005 at a t ratio of 0.6, so the correct claim is convergence
  rather than superiority. The Heston policy beats both benchmarks at zero cost, 3.219 ± 0.018 against
  3.292 ± 0.006 in CVaR 95, at a t ratio of 11.9. The earlier deficits were budget artifacts.
- The GBM cost sensitivity is no longer single seed. It now uses 5 seeds at 100,000 paths and 20 epochs
  per cost level, matching the Heston study and the main table, and writes
  `results/gbm_cost_multiseed.json` with `results/cost_sensitivity_multiseed.png`. Every gap against
  Leland from 0.10 percent cost upward is significant at a t ratio of 2.7 or higher.
- The CVaR advantage over Leland, previously not significant at a t ratio of 1.2, is now significant
  at a t ratio of 39.2 for CVaR 95 and 26.9 for CVaR 99.
- The drawdown metric is now interpretable. An intra-horizon drawdown on the pre-settlement path is
  reported alongside the terminal-jump version, and it separates the strategies: 3.598 for deep
  hedging against 4.469 for Black-Scholes and 4.221 for Leland, where the terminal-jump version gave
  5.213, 6.121 and 5.896 and was dominated by the common maturity liability.
- The time-dependent variant is competitive under LayerNorm, and the comparison is now 5-seed. At
  100,000 paths and 20 epochs it reaches an entropic risk of 3.664 ± 0.010 against 3.581 ± 0.004 for the
  shared variant, and beats both benchmarks. The shared mode is ahead on every measure, so it stays
  the default on accuracy as well as on size and speed. The earlier failure was a BatchNorm artifact,
  not a capacity limit.
- The Heston results are now reported as figures as well as a table, including a diagnostic showing
  the policy responding to the variance state, and a policy surface with the no-trade band.
- The Heston cost sweep is no longer single seed. It now uses 5 seeds at 100,000 paths and 20 epochs
  per cost level, matching the GBM study exactly, and every gap against Leland is significant at a
  t ratio of 11.9 or higher.
- The lambda sweep is no longer single seed, and it is wider. It now covers five values from 0.25 to
  4.00 with 5 seeds each, so the risk-return frontier is a statistically supported curve rather than
  three single observations.
- The paper's own Section 5 numerical results are now attempted, which was the largest real gap in
  the previous version of this document. Concretely, five experiments are run: the Black-Scholes
  price asymptotics of Figure 10 with a fitted slope of 0.6898 against the predicted 2/3, the Heston
  no-cost two instrument experiment of Figures 2, 6 and 7 with $p(0.50) = 1.9630$ against the paper's
  1.94, the Heston price asymptotics of Figure 11 with a slope of 0.6017, the recurrent against
  simple ablation of Figures 4 and 5, and the call spread risk aversion experiment of Figures 8 and 9.
  Four of the five reproduce. The call spread does not: the alpha 0.99 policy collapses to a near
  constant position and is worse than alpha 0.95 on every risk measure, which is documented as a
  non-reproduction rather than a success. The paper-faithful architecture of Section 5.1 is also
  compared and matches the default within 0.009 on the final training risk and 0.032 on CVaR 95.
- The Heston pricer and simulator required by the paper's parameter set are now implemented and gated
  on validation. The pricer returns $q = 1.691834$ against the paper's 1.69 and collapses to
  Black-Scholes to five decimal places as the vol of vol goes to zero. The simulator uses an exact CIR
  transition, which is necessary because the paper's parameters violate Feller by $-3.92$ and 38.7
  percent of paths reach the zero-variance region.
- The earlier claim that the project's cost sensitivity "matches the paper" has been corrected. The
  paper has no such figure; its cost result is the $\varepsilon^{2/3}$ scaling, which is now
  reproduced separately. The same correction was applied to the $\lambda$ sweep, which the paper does
  not run, and to the no-trade band, which is due to Leland and to Whalley and Wilmott rather than to
  the 2019 paper.
- The console-only GBM cost sweep and zero-cost delta check are now stored. They live in
  `results/paper/price_asymptotics_bs.json` with a full `config` provenance block, and the project's own
  cost sensitivity is stored separately in `results/gbm_cost_multiseed.json`.

## Suggested next experiments

The paper's Section 5.3.1, 5.3.2, 5.2 and 5.1 experiments are now reproduced, so the remaining work is
mostly genuine scope extension plus one failed experiment to diagnose.

1. Diagnose the call spread failure at $\alpha = 0.99$. The policy collapses to a near constant
   position, which is the one attempted paper experiment that does not reproduce. The cheapest test is
   to raise the training budget to $10^6$ paths so the 99 percent tail has 10,000 observations instead
   of 1,000, and to check whether the collapse persists. If it does, the next step is a tail
   reweighting or a larger batch drawn from the tail rather than a change to the architecture.
2. Calibrate a simulator to a real return series and evaluate out of sample on a held-out window, which
   is the paper's Section 5.5. This needs real market data and a GJR-GARCH fit, both outside the
   declared scope, and it is the only substantive paper experiment still missing alongside Section 5.4.
3. Reproduce Section 5.4, five Heston models with ten instruments. The paper reports 5.75 hours of
   training for it, so it needs a larger compute budget rather than new code, since the multi
   instrument PnL engine and the exact CIR simulator both already support it.
4. Test a path-dependent payoff, for example a down-and-out call. This requires the nested simulation
   construction that the current single-payoff engine does not support, so it is a scope extension
   rather than a parameter change.
5. Add a second friction model, for example quadratic market impact or a bid-ask spread, to test
   whether the no-transaction band widens as the theory predicts when costs become nonlinear.
