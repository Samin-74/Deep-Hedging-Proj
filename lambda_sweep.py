import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from main import (
    benchmark_results,
    build_model,
    format_table,
    run_policy,
    seed_everything,
    train_policy,
)
from src.loss import EntropicRiskLoss, empirical_cvar
from src.portfolio import hedge_pnl
from src.simulators import simulate_paths


def main() -> int:
    # Entropic risk aversion sweep under the main GBM configuration
    lambdas = [float(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.5", "1.0", "2.0"])]
    paths = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    output = sys.argv[4] if len(sys.argv) > 4 else "results/lambda_sweep.json"

    base = replace(Config(), n_paths=paths, epochs=epochs)
    base.validate()
    seed_everything(base.seed, deterministic=base.deterministic)

    print("=" * 78)
    print("Entropic risk aversion sweep")
    print("=" * 78)
    print(f"lambdas {lambdas}  paths {paths:,}  epochs {epochs}  cost {base.cost_rate:.4f}")
    print("=" * 78)

    train_spots, train_var = simulate_paths(base, base.n_paths, seed=base.seed)
    eval_spots, eval_var = simulate_paths(base, base.n_eval_paths, seed=base.seed + 1)
    benches = benchmark_results(eval_spots, base, return_path=False)

    records: list[dict[str, float]] = []
    for lam in lambdas:
        cfg = replace(base, lambda_entropic=lam)
        cfg.validate()
        model, history, _ = train_policy(cfg, train_spots, train_var, seed=cfg.seed)
        pnl, _ = run_policy(model, eval_spots, eval_var, cfg, return_path=False)
        with torch.no_grad():
            deltas = model(eval_spots, eval_var)
        detail = hedge_pnl(eval_spots, deltas, cfg.K, cfg.cost_rate, return_path=False)
        record = {
            "lambda": lam,
            "mean_pnl": float(pnl.mean()),
            "std_pnl": float(pnl.std(unbiased=True)),
            "cvar_95": float(empirical_cvar(pnl, 0.95)),
            "cvar_99": float(empirical_cvar(pnl, 0.99)),
            "entropic": float(EntropicRiskLoss(lam)(pnl)),
            "mean_cost": float(detail["costs"].mean()),
            "turnover": float(detail["turnover"].mean()),
            "final_train_risk": float(history[-1]),
        }
        records.append(record)
        print(
            f"  lambda {lam:4.2f}: mean {record['mean_pnl']:7.3f}  std {record['std_pnl']:6.3f}  "
            f"cvar95 {record['cvar_95']:6.3f}  cvar99 {record['cvar_99']:6.3f}  "
            f"entropic {record['entropic']:6.3f}  cost {record['mean_cost']:5.3f}",
            flush=True,
        )

    rows = [
        (
            f"{r['lambda']:5.2f}",
            f"{r['mean_pnl']:8.3f}",
            f"{r['std_pnl']:8.3f}",
            f"{r['cvar_95']:8.3f}",
            f"{r['cvar_99']:8.3f}",
            f"{r['entropic']:8.3f}",
            f"{r['mean_cost']:8.3f}",
            f"{r['turnover']:8.3f}",
        )
        for r in records
    ]
    print()
    print("Lambda sweep, Deep Hedging")
    print(
        format_table(
            ("Lambda", "Mean PnL", "Std", "CVaR 95", "CVaR 99", "Entropic", "Mean cost", "Turnover"),
            rows,
        )
    )

    bench_rows = []
    for name, payload in benches.items():
        p = payload["pnl"]
        bench_rows.append(
            (
                name,
                f"{float(p.mean()):8.3f}",
                f"{float(p.std(unbiased=True)):8.3f}",
                f"{float(empirical_cvar(p, 0.95)):8.3f}",
                f"{float(empirical_cvar(p, 0.99)):8.3f}",
                f"{float(payload['costs'].mean()):8.3f}",
            )
        )
    print()
    print("Benchmarks on the same evaluation paths")
    print(
        format_table(
            ("Strategy", "Mean PnL", "Std", "CVaR 95", "CVaR 99", "Mean cost"),
            bench_rows,
        )
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"lambda sweep written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
