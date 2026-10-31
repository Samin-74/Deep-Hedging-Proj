import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import Config
from src.benchmarks import BenchmarkRunner
from src.diagnostics import cost_drag, scenario_analysis, tail_attribution
from src.loss import (
    CVaRLoss,
    EntropicRiskLoss,
    empirical_cvar,
    summarize_drawdown,
    summarize_pnl,
    var_threshold,
)
from src.multiseed import SEEDS, markdown_table, run_study
from src.network import DeepHedgingNetwork
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths
from src.visualize import (
    plot_cost_sensitivity,
    plot_learning_curves,
    plot_policy_curves,
    plot_pnl_distributions,
    plot_pnl_paths,
    plot_position_dynamics,
    plot_training_loss,
)

STRATEGIES = ("Deep Hedging", "Black-Scholes", "Leland")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    # Single entry point for reproducible runs
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Thread count changes floating point reduction order, so it must be pinned
    torch.set_num_threads(1)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Hedging replication")
    parser.add_argument("--simulator", choices=("gbm", "heston"), default=None)
    parser.add_argument("--loss", choices=("entropic", "cvar"), dest="loss_type", default=None)
    parser.add_argument("--cost", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--paths", type=int, dest="n_paths", default=None)
    parser.add_argument("--eval-paths", type=int, dest="n_eval_paths", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, dest="n_layers", default=None)
    parser.add_argument("--max-position", type=float, default=None)
    parser.add_argument("--lambda-entropic", type=float, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--time-dependent", action="store_true", default=None)
    parser.add_argument("--norm", choices=("layernorm", "batchnorm"), dest="norm_type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--no-sensitivity", action="store_false", dest="run_cost_sensitivity", default=None)
    parser.add_argument("--sensitivity-paths", type=int, default=None)
    parser.add_argument("--sensitivity-epochs", type=int, default=None)
    parser.add_argument("--multi-seed", action="store_true", default=False)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--multi-seed-output", default=None)
    return parser.parse_args()


CLI_ONLY = {"multi_seed", "seeds", "multi_seed_output"}


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key not in CLI_ONLY
    }
    if overrides:
        config = replace(config, **overrides)
    config.validate()
    return config


def build_loss(config: Config):

    if config.loss_type == "entropic":
        return EntropicRiskLoss(config.lambda_entropic)
    return CVaRLoss(config.cvar_alpha, init_w=0.0)


def build_model(config: Config) -> DeepHedgingNetwork:
    # Two instrument mode is only used by the paper Heston study
    output_dim = 2 if config.paper_heston or config.simulator == "paper_heston" else 1
    paper_mode = output_dim > 1
    feature_mode = "paper" if paper_mode else config.feature_mode
    return DeepHedgingNetwork(
        n_steps=config.n_steps,
        K=config.K,
        T=config.T,
        base_variance=config.base_variance,
        dt=config.dt,
        max_position=config.max_position,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        time_dependent=config.time_dependent,
        norm_type=config.norm_type,
        output_dim=output_dim,
        use_prev_delta=config.use_prev_delta,
        feature_mode=feature_mode,
        activation=config.activation,
        linear_head=paper_mode,
    ).to(config.device)


def calibrate_threshold(loss, model, spots, variances, config: Config) -> float:
    # Warm start the VaR threshold at the empirical quantile of losses
    if not isinstance(loss, CVaRLoss):
        return float("nan")
    n_sample = min(spots.shape[0], max(config.batch_size * 16, 4096))
    model.eval()
    with torch.no_grad():
        deltas = model(spots[:n_sample], None if variances is None else variances[:n_sample])
        pnl = hedge_pnl(spots[:n_sample], deltas, config.K, config.cost_rate,
                        second_strike=config.payoff_second_strike)["pnl"]
    init = var_threshold(pnl, config.cvar_alpha)
    loss.w.data.fill_(init)
    return init


def train_policy(
    config: Config,
    spots: torch.Tensor,
    variances: torch.Tensor | None,
    seed: int,
) -> tuple[DeepHedgingNetwork, list[float], float]:
    seed_everything(seed, deterministic=config.deterministic)
    model = build_model(config)
    loss_fn = build_loss(config)
    init_w = calibrate_threshold(loss_fn, model, spots, variances, config)

    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config.lr}]
    if isinstance(loss_fn, CVaRLoss):
        groups.append({"params": [loss_fn.w], "lr": config.w_lr})
    optimizer = torch.optim.Adam(groups)

    n_paths = spots.shape[0]
    batch_size = config.effective_batch_size(n_paths)
    steps_per_epoch = max(n_paths // batch_size, 1)
    history: list[float] = []
    model.train()
    for epoch in range(config.epochs):
        order = torch.randperm(n_paths, device=spots.device)
        total = 0.0
        seen = 0
        epoch_pnl: list[torch.Tensor] = []
        for start in range(0, n_paths - batch_size + 1, batch_size):
            index = order[start : start + batch_size]
            batch_spots = spots[index]
            batch_var = None if variances is None else variances[index]
            deltas = model(batch_spots, batch_var)
            pnl = hedge_pnl(batch_spots, deltas, config.K, config.cost_rate,
                            second_strike=config.payoff_second_strike)["pnl"]
            loss = loss_fn(pnl)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            weight = index.numel()
            total += float(loss.detach()) * weight
            seen += weight
            if isinstance(loss_fn, CVaRLoss):
                epoch_pnl.append(pnl.detach())
        if isinstance(loss_fn, CVaRLoss) and epoch_pnl:
            # Exact VaR projection keeps the threshold on the current policy tail
            pooled = torch.cat(epoch_pnl)
            with torch.no_grad():
                loss_fn.w.data.fill_(var_threshold(pooled, loss_fn.alpha))
        history.append(total / max(seen, 1))
        print(
            f"epoch {epoch + 1:3d}/{config.epochs}  "
            f"risk {history[-1]:9.4f}  batches {steps_per_epoch:4d}",
            flush=True,
        )
    return model, history, init_w


def run_policy(
    model: DeepHedgingNetwork,
    spots: torch.Tensor,
    variances: torch.Tensor | None,
    config: Config,
    chunk: int = 25_000,
    return_path: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model.eval()
    pnls: list[torch.Tensor] = []
    paths: list[torch.Tensor] = []
    pre_paths: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, spots.shape[0], chunk):
            sl = slice(start, start + chunk)
            var = None if variances is None else variances[sl]
            deltas = model(spots[sl], var)
            result = hedge_pnl(spots[sl], deltas, config.K, config.cost_rate, return_path,
                               second_strike=config.payoff_second_strike)
            pnls.append(result["pnl"])
            if return_path:
                paths.append(result["pnl_path"])
                pre_paths.append(result["pnl_path_pre"])
    if not return_path:
        return torch.cat(pnls), None
    return torch.cat(pnls), torch.cat(paths), torch.cat(pre_paths)


def benchmark_results(
    spots: torch.Tensor, config: Config, return_path: bool = True
) -> dict[str, dict[str, torch.Tensor]]:
    runner = BenchmarkRunner(
        spots=spots,
        K=config.K,
        r=config.r,
        sigma=config.benchmark_sigma,
        cost_rate=config.cost_rate,
        dt=config.dt,
        n_steps=config.n_steps,
    )
    return {"Black-Scholes": runner.run_bs(), "Leland": runner.run_leland()}


def format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    rule = "+".join("-" * (width + 2) for width in widths)
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    body = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows
    ]
    return "\n".join([rule, header, rule, *body, rule])


def comparison_rows(summaries: dict[str, dict[str, float]]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for name in STRATEGIES:
        summary = summaries[name]
        rows.append(
            (
                name,
                f"{summary['mean']:8.3f}",
                f"{summary['std']:8.3f}",
                f"{summary['cvar_95']:8.3f}",
                f"{summary['cvar_99']:8.3f}",
                f"{summary['dd_intra_mean']:8.3f}",
                f"{summary['dd_intra_worst']:8.3f}",
            )
        )
    return rows


def drawdown_rows(summaries: dict[str, dict[str, float]]) -> list[tuple[str, ...]]:
    # Terminal-jump and intra-horizon drawdown side by side
    rows: list[tuple[str, ...]] = []
    for name in STRATEGIES:
        s = summaries[name]
        rows.append(
            (
                name,
                f"{s['dd_terminal_mean']:8.3f}",
                f"{s['dd_terminal_worst']:8.3f}",
                f"{s['dd_intra_mean']:8.3f}",
                f"{s['dd_intra_p95']:8.3f}",
                f"{s['dd_intra_worst']:8.3f}",
            )
        )
    return rows


def print_header(config: Config, n_paths: int | None = None) -> None:
    print("=" * 78)
    print("Deep Hedging: Buehler, Gonon, Teichmann, Wood (Quantitative Finance 2019)")
    print("=" * 78)
    train_paths = config.n_paths if n_paths is None else int(n_paths)
    print(
        f"simulator {config.simulator.upper():<7} loss {config.loss_type:<9} "
        f"paths {train_paths:,} eval {config.n_eval_paths:,}"
    )
    print(
        f"S0 {config.S0:.2f}  K {config.K:.2f}  T {config.T:.6f}  "
        f"n_steps {config.n_steps}  dt {config.dt:.6f}"
    )
    print(
        f"r {config.r:.4f}  sigma {config.benchmark_sigma:.4f}  "
        f"cost {config.cost_rate:.4f}  lambda {config.lambda_entropic:.3f}  "
        f"alpha {config.cvar_alpha:.2f}"
    )
    effective_batch = config.effective_batch_size(train_paths)
    tail_note = (
        f" (raised for the {100.0 * (1.0 - config.cvar_alpha):.0f} percent tail)"
        if effective_batch != config.batch_size
        else ""
    )
    print(
        f"epochs {config.epochs}  batch {effective_batch}{tail_note}  lr {config.lr}  "
        f"hidden {config.hidden_dim}  layers {config.n_layers}  "
        f"shared_weights {not config.time_dependent}"
    )
    print(f"device {config.device}  deterministic {config.deterministic}  seed {config.seed}")
    print("=" * 78)


def print_comparison(summaries: dict[str, dict[str, float]]) -> None:
    headers = (
        "Strategy",
        "Mean PnL",
        "Std",
        "CVaR 95",
        "CVaR 99",
        "Intra DD mean",
        "Intra DD worst",
    )
    print()
    print("Terminal PnL statistics on the evaluation set")
    print(format_table(headers, comparison_rows(summaries)))
    print()
    print("Drawdown: terminal jump includes the maturity liability, intra excludes it")
    print(
        format_table(
            (
                "Strategy",
                "Term DD mean",
                "Term DD worst",
                "Intra DD mean",
                "Intra DD p95",
                "Intra DD worst",
            ),
            drawdown_rows(summaries),
        )
    )


def print_diagnostics(summaries: dict[str, dict[str, float]]) -> None:
    print()
    print("Tail and dispersion detail")
    for name in STRATEGIES:
        summary = summaries[name]
        print(
            f"  {name:<14} min {summary['min']:8.3f}  q05 {summary['q05']:8.3f}  "
            f"median {summary['median']:8.3f}  q95 {summary['q95']:8.3f}  "
            f"max {summary['max']:8.3f}"
        )


def extra_summary(
    pnl: torch.Tensor,
    pnl_path: torch.Tensor | None,
    pnl_path_pre: torch.Tensor | None = None,
) -> dict[str, float]:
    summary = summarize_pnl(pnl, pnl_path)
    summary["median"] = float(pnl.median())
    if pnl_path is not None and pnl_path_pre is not None:
        summary.update(summarize_drawdown(pnl_path, pnl_path_pre))
    return summary


def run_cost_study(config: Config) -> list[dict[str, float]]:
    # Retrain the policy at each cost level and score all strategies
    records: list[dict[str, float]] = []
    sens_spots, sens_var = simulate_paths(config, config.sensitivity_paths, seed=config.seed + 1)
    for cost in config.cost_levels:
        level = replace(
            config,
            cost_rate=float(cost),
            n_paths=config.sensitivity_paths,
            epochs=config.sensitivity_epochs,
        )
        batch = level.effective_batch_size(level.n_paths)
        print(
            f"  cost level {100.0 * cost:5.2f} percent, batch {batch}, "
            f"tail {level.tail_sample_size(level.n_paths):5.1f}, "
            f"epochs {level.epochs}"
        )
        train_spots, train_var = simulate_paths(level, level.n_paths, seed=level.seed + 7)
        model, _, _ = train_policy(level, train_spots, train_var, seed=level.seed + 11)
        deep_pnl, _ = run_policy(
            model, sens_spots, sens_var, level, chunk=10_000, return_path=False
        )
        benches = benchmark_results(sens_spots, level, return_path=False)
        record = {
            "cost": float(cost),
            "deep_cvar_95": float(empirical_cvar(deep_pnl, 0.95)),
            "deep_cvar_99": float(empirical_cvar(deep_pnl, 0.99)),
            "deep_mean": float(deep_pnl.mean()),
        }
        for key, payload in benches.items():
            prefix = "bs" if key == "Black-Scholes" else "leland"
            record[f"{prefix}_cvar_95"] = float(empirical_cvar(payload["pnl"], 0.95))
            record[f"{prefix}_cvar_99"] = float(empirical_cvar(payload["pnl"], 0.99))
            record[f"{prefix}_mean"] = float(payload["pnl"].mean())
        records.append(record)
    return records


def print_cost_study(records: list[dict[str, float]]) -> None:
    headers = (
        "Cost",
        "Deep mean",
        "Deep CVaR95",
        "Deep CVaR99",
        "BS CVaR95",
        "BS CVaR99",
        "Leland CVaR95",
        "Leland CVaR99",
    )
    rows = [
        (
            f"{100.0 * record['cost']:5.2f}%",
            f"{record['deep_mean']:9.3f}",
            f"{record['deep_cvar_95']:11.3f}",
            f"{record['deep_cvar_99']:11.3f}",
            f"{record['bs_cvar_95']:9.3f}",
            f"{record['bs_cvar_99']:9.3f}",
            f"{record['leland_cvar_95']:13.3f}",
            f"{record['leland_cvar_99']:13.3f}",
        )
        for record in records
    ]
    print()
    print("Cost sensitivity study")
    print(format_table(headers, rows))


def run_multi_seed(config: Config, seeds: list[int], output: str | None) -> int:
    # Multi-seed study, leaves the canonical results directory untouched
    print("=" * 78)
    print("Deep Hedging multi-seed study")
    print("=" * 78)
    print(
        f"simulator {config.simulator.upper()}  loss {config.loss_type}  "
        f"cost {config.cost_rate:.4f}  paths {config.n_paths:,}  "
        f"eval {config.n_eval_paths:,}  epochs {config.epochs}"
    )
    print(f"seeds {seeds}")
    print("=" * 78)
    target = output or str(config.results_path / "multiseed_layernorm.json")
    study = run_study(
        config=config,
        seeds=seeds,
        simulate=simulate_paths,
        train=train_policy,
        evaluate=run_policy,
        format_table=format_table,
        output_path=target,
    )
    print()
    print("Markdown table for the write-up")
    print(markdown_table(study["summary"], plus_minus="+/-"))
    return 0


def main() -> int:
    args = parse_args()
    config = build_config(args)
    seed_everything(config.seed, deterministic=config.deterministic)
    if args.multi_seed:
        return run_multi_seed(config, list(args.seeds or SEEDS), args.multi_seed_output)
    print_header(config)

    started = time.perf_counter()
    train_spots, train_var = simulate_paths(config, config.n_paths, seed=config.seed)
    eval_spots, eval_var = simulate_paths(config, config.n_eval_paths, seed=config.seed + 1)
    print(
        f"simulated {config.n_paths:,} training paths and "
        f"{config.n_eval_paths:,} evaluation paths in {time.perf_counter() - started:.2f}s"
    )
    print(
        f"terminal spot: mean {float(eval_spots[:, -1].mean()):.3f}  "
        f"std {float(eval_spots[:, -1].std()):.3f}  "
        f"mean payoff {float(torch.relu(eval_spots[:, -1] - config.K).mean()):.3f}"
    )

    model = build_model(config)
    print(f"policy parameters {model.count_parameters():,}")
    train_started = time.perf_counter()
    model, history, init_w = train_policy(config, train_spots, train_var, seed=config.seed)
    train_seconds = time.perf_counter() - train_started
    print(f"training finished in {train_seconds:.2f}s")
    if np.isfinite(init_w):
        print(f"CVaR threshold: initial {init_w:.4f}")

    deep_pnl, deep_path, deep_pre = run_policy(model, eval_spots, eval_var, config)
    benches = benchmark_results(eval_spots, config)

    summaries = {"Deep Hedging": extra_summary(deep_pnl, deep_path, deep_pre)}
    for name, payload in benches.items():
        summaries[name] = extra_summary(
            payload["pnl"], payload["pnl_path"], payload["pnl_path_pre"]
        )

    print_comparison(summaries)
    print_diagnostics(summaries)

    with torch.no_grad():
        deep_deltas = model(eval_spots, eval_var)
    deep_detail = hedge_pnl(eval_spots, deep_deltas, config.K, config.cost_rate,
                            second_strike=config.payoff_second_strike)
    costs = {"Deep Hedging": deep_detail["costs"]}
    pnls = {"Deep Hedging": deep_pnl}
    delta_paths = {"Deep Hedging": deep_deltas}
    pnl_paths = {"Deep Hedging": deep_path}
    for name, payload in benches.items():
        costs[name] = payload["costs"]
        pnls[name] = payload["pnl"]
        delta_paths[name] = payload["deltas"]
        pnl_paths[name] = payload["pnl_path"]

    print()
    print("Hedge diagnostics")
    print(
        format_table(
            ("Strategy", "Mean cost", "Gross PnL", "Net PnL", "Turnover", "Tail cost share"),
            [
                (
                    name,
                    f"{cost:.4f}",
                    f"{gross:.4f}",
                    f"{net:.4f}",
                    f"{float(delta_paths[name].abs().sum(dim=1).mean()):.4f}",
                    f"{tail_attribution(pnls[name], costs[name], config.cvar_alpha)['tail_cost_share']:.3f}",
                )
                for name, cost, gross, net in cost_drag(costs, pnls, config.cvar_alpha)
            ],
        )
    )
    risk_name = (
        "Entropic"
        if config.loss_type == "entropic"
        else f"CVaR {config.cvar_alpha:.2f}"
    )
    risk_rows = []
    for name in STRATEGIES:
        measure = EntropicRiskLoss(config.lambda_entropic)(pnls[name])
        risk_rows.append(
            (
                name,
                f"{float(measure):9.4f}",
                f"{float(empirical_cvar(pnls[name], 0.95)):9.4f}",
                f"{float(empirical_cvar(pnls[name], 0.99)):9.4f}",
            )
        )
    print()
    print(f"Risk measures (training objective: {risk_name})")
    print(
        format_table(
            ("Strategy", "Entropic", "CVaR 95", "CVaR 99"),
            risk_rows,
        )
    )
    deltas_edge = {
        "Black-Scholes": benches["Black-Scholes"]["deltas"],
        "Leland": benches["Leland"]["deltas"],
    }
    deltas_edge_pnls = {
        "Black-Scholes": benches["Black-Scholes"]["pnl"],
        "Leland": benches["Leland"]["pnl"],
    }
    scenario = scenario_analysis(model, config)
    print()
    print(
        f"Policy response around the strike at step {scenario['step']} "
        f"with previous delta {scenario['prev_delta']:.3f}"
    )
    print(
        format_table(
            ("Spot move", "Spot", "Deep Hedging", "Black-Scholes", "Leland"),
            [
                (
                    f"{100.0 * (float(m) - 1.0):+.2f}%",
                    f"{float(spot):.2f}",
                    f"{float(pol):.4f}",
                    f"{float(bs):.4f}",
                    f"{float(le):.4f}",
                )
                for m, spot, pol, bs, le in zip(
                    scenario["magnitudes"],
                    scenario["spots"],
                    scenario["policy"],
                    scenario["bs"],
                    scenario["leland"],
                )
            ],
        )
    )

    results = Path(config.results_path)
    results.mkdir(parents=True, exist_ok=True)
    artifacts = [
        plot_pnl_distributions(
            {
                "Deep Hedging": deep_pnl,
                "Black-Scholes": benches["Black-Scholes"]["pnl"],
                "Leland": benches["Leland"]["pnl"],
            },
            results,
        ),
        plot_policy_curves(model, config, results, eval_spots=eval_spots, eval_variances=eval_var),
        plot_training_loss(
            history,
            results,
            loss_name=(
                "Entropic risk"
                if config.loss_type == "entropic"
                else f"CVaR at {config.cvar_alpha:.2f}"
            ),
        ),
        plot_learning_curves(deep_pnl, deltas_edge_pnls, results),
        plot_position_dynamics(deep_deltas, deltas_edge, eval_spots, results),
        plot_pnl_paths(deep_path, pnl_paths, results),
    ]

    if config.run_cost_sensitivity:
        print()
        print("Running cost sensitivity study")
        study_started = time.perf_counter()
        records = run_cost_study(config)
        print_cost_study(records)
        artifacts.append(plot_cost_sensitivity(records, results))
        print(f"cost study finished in {time.perf_counter() - study_started:.2f}s")

    print()
    print("Saved figures")
    for artifact in artifacts:
        print(f"  {artifact}")
    # Persist the console numbers so every quoted value is checkable from disk
    canonical = {
        "config": {
            "simulator": config.simulator,
            "loss_type": config.loss_type,
            "cost_rate": config.cost_rate,
            "n_paths": config.n_paths,
            "n_eval_paths": config.n_eval_paths,
            "epochs": config.epochs,
            "seed": config.seed,
            "norm_type": config.norm_type,
            "time_dependent": config.time_dependent,
        },
        "summaries": {name: {k: float(v) for k, v in block.items()} for name, block in summaries.items()},
        "cost_drag": [
            {"strategy": name, "mean_cost": float(cost), "gross_pnl": float(gross), "net_pnl": float(net)}
            for name, cost, gross, net in cost_drag(costs, pnls, config.cvar_alpha)
        ],
        "turnover": {name: float(paths.abs().sum(dim=1).mean()) for name, paths in delta_paths.items()},        "tail_cost_share": {
            name: float(tail_attribution(pnls[name], costs[name], config.cvar_alpha)["tail_cost_share"])
            for name in STRATEGIES
        },
        "risk_measures": {
            name: {
                "entropic": float(EntropicRiskLoss(config.lambda_entropic)(pnls[name])),
                "cvar_95": float(empirical_cvar(pnls[name], 0.95)),
                "cvar_99": float(empirical_cvar(pnls[name], 0.99)),
            }
            for name in STRATEGIES
        },
        "scenario": {
            "step": int(scenario["step"]),
            "prev_delta": float(scenario["prev_delta"]),
            "spots": [float(v) for v in scenario["spots"]],
            "policy": [float(v) for v in scenario["policy"]],
            "bs": [float(v) for v in scenario["bs"]],
            "leland": [float(v) for v in scenario["leland"]],
        },
        "training_history": [float(v) for v in history],
        "n_params": int(model.count_parameters()),
    }
    (results / "canonical_run.json").write_text(json.dumps(canonical, indent=2), encoding="utf-8")
    print(f"  {results / 'canonical_run.json'}")
    print(f"total runtime {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
