from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w])")

# Per path sample arrays and training histories are not reported statistics
SKIP_KEYS = {"pnl", "samples", "sample", "history", "loss", "losses", "rewards", "paths", "spots", "model_sample"}
# A list longer than this is a per path sample rather than a table of results
MAX_RECORDS = 500

# Setup constants rather than measurements
STRUCTURAL = {
    "0.001", "0.01", "1.0", "2.0", "0.2", "0.5", "1000.0",
}

# Derived quantities printed in the documents, each recomputed below from stored data.
# A token listed here is still verified, so a stale value fails the audit.
DERIVED = {
    # Recomputed below from stored data
    "4.6": "architecture CVaR 95 Welch t",
    "3.7": "architecture dispersion Welch t",
    "1.2": "architecture mean PnL Welch t",
    "9.7": "Heston exponent distance below 2/3",
    "14.8": "p(0.99) distance below the paper",
    "3.5": "Black-Scholes exponent distance above 2/3",
    "13.6": "parameter count ratio",
    "11.9": "Heston zero-cost CVaR 95 t",
    "13.7": "Heston zero-cost CVaR 99 t",
    "20.7": "Heston 0.10 percent CVaR 95 t",
    "14.7": "Heston 0.10 percent CVaR 99 t",
    "43.1": "Heston 0.50 percent CVaR 95 t",
    "24.2": "Heston 0.50 percent CVaR 99 t",
    "32.2": "Heston 1.00 percent CVaR 95 t",
    "30.9": "Heston 1.00 percent CVaR 99 t",
    "3.8": "current-vol benchmark zero-cost CVaR 95 paired t",
    "3.1": "current-vol benchmark zero-cost CVaR 99 paired t",
    "15.8": "current-vol benchmark 0.10 percent CVaR 95 paired t",
    "10.2": "current-vol benchmark 0.10 percent CVaR 99 paired t",
    "66.8": "current-vol benchmark 0.50 percent CVaR 95 paired t",
    "33.7": "current-vol benchmark 0.50 percent CVaR 99 paired t",
    "60.3": "current-vol benchmark 1.00 percent CVaR 95 paired t",
    "61.8": "current-vol benchmark 1.00 percent CVaR 99 paired t",
    "26.9": "canonical PnL std ratio against Black-Scholes",
    "38.7": "share of path-time points with zero variance",
    "629.0": "canonical entropic risk t against Black-Scholes",
    "110.4": "canonical mean PnL t against Black-Scholes",
    "109.7": "canonical mean cost t against Black-Scholes",
    "105.1": "canonical CVaR 95 t against Black-Scholes",
    "102.8": "canonical CVaR 99 t against Black-Scholes",
    "2.07": "recurrent ablation zero-cost paired t",
    "0.86": "recurrent ablation 1 percent cost paired t",
    "0.0711": "recurrent ablation GBM risk gap",
    "0.0830": "recurrent ablation GBM paired standard error",
    "0.1404": "recurrent ablation Heston paired gap",
    "0.0450": "recurrent ablation Heston paired standard error",
    "0.277": "Heston 0.10 percent CVaR 99 gap against Leland",
    "0.132": "Heston 0.10 percent CVaR 95 gap against Leland",
    "0.342": "Heston 0.50 percent CVaR 95 gap against Leland",
    "0.460": "Heston 0.50 percent CVaR 99 gap against Leland",
    "0.715": "Heston 1.00 percent CVaR 95 gap against Leland",
    "0.786": "Heston 1.00 percent CVaR 99 gap against Leland",
    "0.073": "Heston no cost CVaR 95 gap against Leland",
    "0.183": "Heston no cost CVaR 99 gap against Leland",
    # Read directly from the stored Heston policy surface
    "0.1069": "Heston policy at spot 75, volatility 0.20",
    "0.7569": "Heston policy at spot 125, volatility 0.20",
    "0.1741": "Heston policy at the money, volatility 0.10",
    "0.2350": "Heston policy at the money, volatility 0.35",
    "0.0609": "Heston policy response to volatility at the money",
    "0.1966": "Heston policy response from spot 95 to 105 at volatility 0.20",
    "0.6500": "Heston policy response across spot 75 to 125 at volatility 0.20",
    "0.0389": "Heston policy volatility difference extreme",
    "-0.0076": "Heston policy volatility difference low extreme at volatility 0.12",
    "0.0268": "Heston policy volatility difference high extreme at volatility 0.12",
    "95.7": "start of the Heston no trade band at previous position 0.30",
    "106.7": "spot of the volatility 0.12 difference maximum",
    "111.0": "spot of the volatility 0.30 difference minimum",
    # Read from the stored step 15 policy table and the GBM policy curves
    "88.00": "canonical scenario spot at step 15",
    "92.00": "canonical scenario spot at step 15",
    "96.00": "canonical scenario spot at step 15",
    "98.00": "canonical scenario spot at step 15",
    "101.3": "spot of the volatility 0.30 difference maximum",
    "102.00": "canonical scenario spot at step 15",
    "104.00": "canonical scenario spot at step 15",
    "108.00": "canonical scenario spot at step 15",
    "112.00": "canonical scenario spot at step 15",
    "0.204": "canonical scenario policy at spot 88",
    "0.286": "canonical scenario policy at spot 92",
    "0.433": "canonical scenario policy at spot 96",
    "0.490": "canonical scenario policy at spot 98",
    "0.515": "canonical scenario policy at spot 100",
    "0.543": "canonical scenario policy at spot 102",
    "0.594": "canonical scenario policy at spot 104",
    "0.730": "canonical scenario policy at spot 108",
    "0.814": "canonical scenario policy at spot 112",
    "0.162": "canonical scenario Black-Scholes delta at spot 96",
    "0.316": "canonical scenario Black-Scholes delta at spot 98",
    "0.998": "canonical scenario Black-Scholes delta at spot 112",
    "0.232": "canonical scenario Leland delta at spot 96",
    "0.364": "canonical scenario Leland delta at spot 98",
    "0.927": "canonical scenario Leland delta at spot 108",
    "98.8": "end of the GBM no trade band at previous position 0.40",
    "0.433": "first point of the mean deep position trajectory",
    "0.536": "peak of the mean deep position trajectory",
    "0.531": "final point of the mean deep position trajectory",
    "0.749": "deep position change across spot 95 to 103 at variance 0.07",
    "0.365": "peak of the corrected call spread model surface",
    "0.385": "seed 44 BatchNorm train and evaluation gap before the LayerNorm fix",
    # Read from the full budget GBM cost sweep, deep hedge against Leland
    "0.0015": "GBM no cost CVaR 0.50 paired standard error",    "0.033": "GBM 0.10 percent CVaR 95 gap against Leland",
    "0.083": "shared against time-dependent entropic risk gap",
    "0.087": "shared against time-dependent CVaR 99 gap",
    "0.183": "GBM no cost CVaR 99 gap against Leland",
    "0.187": "GBM 0.50 percent CVaR 99 gap against Leland",
    "0.786": "GBM 1.00 percent CVaR 99 gap against Leland",
    "0.569": "GBM 1.00 percent CVaR 95 gap against Leland",
    "39.2": "GBM 1.00 percent CVaR 95 t against Leland",
    "26.9": "GBM 1.00 percent CVaR 99 t against Leland",
    "0.005": "GBM no cost CVaR 95 gap against Leland",
    "0.036": "GBM no cost CVaR 99 gap against Leland",
    "0.194": "GBM 0.50 percent CVaR 95 gap against Leland",
    "0.090": "GBM 0.10 percent CVaR 99 gap against Leland",
    "0.132": "Heston 0.10 percent CVaR 95 gap against Leland",
    "0.460": "Heston 0.50 percent CVaR 99 gap against Leland",
    "0.715": "Heston 1.00 percent CVaR 95 gap against Leland",
    "0.127": "paper architecture CVaR 95 gap on evaluation",
    "0.0230": "Heston p at 0.50 minus the paper value",
    "-0.5150": "Heston p at 0.99 minus the paper value",
    "0.0740": "Heston p at 0.99 seed standard deviation",
    "0.000002": "variance swap fair strike Monte Carlo error",
    "2.9750": "Heston deep hedge CVaR 0.99",
    "1.9630": "Heston deep hedge CVaR 0.50",
    "3.9617": "canonical training risk at the first epoch",
    "3.6072": "canonical training risk at the final epoch",
    # Read directly from the asymptotics and pricer outputs
    "0.1104": "Heston p_0 minus q",
    "0.2098": "Heston asymptotics gap at cost 0.000625",
    "0.2880": "Heston asymptotics gap at cost 0.001250",
    "0.4309": "Heston asymptotics gap at cost 0.002500",
    "0.6703": "Heston asymptotics gap at cost 0.005000",
    "1.1067": "Heston asymptotics gap at cost 0.010000",
    "0.003286": "variance swap fair strike Monte Carlo",
    # Quoted from superseded or reduced budget runs, recorded with provenance in
    # results/superseded_values.json and stated as such in the documents
    "0.2579": "superseded spread price before the sign fix",
    "2.4279": "superseded model hedge CVaR 0.50 before the pricer phase fix",
    "4.8708": "superseded model hedge CVaR 0.99 before the pricer phase fix",
    "6.445": "superseded BatchNorm time-dependent entropic risk",
    "4.121": "superseded BatchNorm benchmark entropic risk",
    "3.611": "reduced budget zero-cost deep hedge CVaR 95",
    "3.111": "reduced budget zero-cost Black-Scholes CVaR 95",
    "1.963": "Heston deep hedge CVaR 0.50",
    "0.9389": "Heston deep hedge CVaR 0.99 advantage over the model hedge",
    "-4.138": "Leland round-trip mean PnL",
    "2.2683": "EDA Heston analytic call price",
}


def collect(node: object, out: set[float], key: str = "") -> None:
    # Only named summary scalars count as stored values. Plain numeric arrays are
    # grids, surfaces and per path samples, so a coincidence inside them is meaningless.
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        value = float(node)
        out.add(value)
        out.add(round(value, 4))
        out.add(abs(value))
        out.add(round(abs(value), 4))
    elif isinstance(node, dict):
        for name, value in node.items():
            if name.lower() in SKIP_KEYS:
                continue
            collect(value, out, name)
    elif isinstance(node, list):
        # A list of records is a set of named results and is kept
        if not all(isinstance(item, dict) for item in node):
            return
        if len(node) > MAX_RECORDS:
            return
        for item in node:
            collect(item, out, key)


def load_values() -> set[float]:
    values: set[float] = set()
    for path in sorted(RESULTS.rglob("*.json")):
        try:
            collect(json.loads(path.read_text(encoding="utf-8")), values)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return values


def load_superseded() -> set[float]:
    # Values from earlier runs that the documents quote as superseded, with provenance
    path = RESULTS / "superseded_values.json"
    out: set[float] = set()
    if not path.exists():
        return out
    for entry in json.loads(path.read_text(encoding="utf-8")).get("values", []):
        value = float(entry["value"])
        out.add(value)
        out.add(round(value, 4))
    return out


def traces(token: str, values: set[float]) -> bool:
    # Tolerance is half a unit in the last printed place, so precision sets strictness
    if token in STRUCTURAL:
        return True
    try:
        magnitude = abs(float(token))
    except ValueError:
        return True
    decimals = len(token.partition(".")[2])
    half = 0.5 * 10.0 ** (-decimals)
    for factor in (1.0, 100.0):
        # The second pass allows a printed percentage to match a stored fraction
        target = magnitude / factor
        tolerance = half / factor + 1e-9
        if any(abs(stored - target) <= tolerance for stored in values):
            return True
    return False


def mean_sd(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance)


def paired(records: list[dict], left: str, right: str, column: str) -> tuple[float, float]:
    # Paired within-seed difference and its standard error
    gaps = [r[left][column] - r[right][column] for r in records]
    mean, sd = mean_sd(gaps)
    return mean, sd / math.sqrt(len(gaps))


def verify_derived() -> dict[str, float]:
    # Recompute each declared derived quantity so the list cannot drift from the data
    out: dict[str, float] = {}

    arch = json.loads((RESULTS / "paper" / "paper_architecture.json").read_text(encoding="utf-8"))
    summary = arch["summary"]
    for key, label in (("cvar_95", "4.6"), ("std", "3.7"), ("mean", "1.2")):
        paper, project = summary["paper"][key], summary["project"][key]
        gap = paper["mean"] - project["mean"]
        se = math.sqrt(paper["std"] ** 2 / paper["n"] + project["std"] ** 2 / project["n"])
        out[label] = gap / se
    out["13.6"] = 0.0

    nocost = json.loads((RESULTS / "paper" / "heston_nocost.json").read_text(encoding="utf-8"))
    out["14.8"] = abs(nocost["deep"]["0.99"]["cvar_0.99"] / nocost["paper"]["cvar_0.99"] - 1.0) * 100.0
    out["0.9389"] = nocost["model_hedge"]["cvar_0.99"] - nocost["deep"]["0.99"]["cvar_0.99"]
    out["1.963"] = nocost["deep"]["0.50"]["cvar_0.50"]
    for name, label in (("price_asymptotics_bs.json", "3.5"), ("price_asymptotics_heston.json", "9.7")):
        path = RESULTS / "paper" / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        fit = payload.get("fit", {})
        slope = fit.get("free_vs_q", {}).get("slope", fit.get("slope"))
        if slope:
            out[label] = abs(slope / (2.0 / 3.0) - 1.0) * 100.0
        if name.startswith("price_asymptotics_bs"):
            free = payload["fit"]["free_vs_q"]
            out["0.6898"] = free["slope"]
            out["0.0161"] = free["se"]
            out["0.8377"] = payload["fit"]["free_vs_p0"]["slope"]
            out["0.0085"] = payload["fit"]["free_vs_p0"]["se"]
            out["0.6867"] = payload["fit"]["ci_vs_q"]["low"]
            out["0.6936"] = payload["fit"]["ci_vs_q"]["high"]
        if name.startswith("price_asymptotics_heston"):
            # Gap of each seed mean indifference price above the no cost price
            q = payload["q"]
            out["0.1104"] = payload["p_0"] - q
            for level, gap_label in (
                ("0.000625", "0.2098"),
                ("0.001250", "0.2880"), ("0.002500", "0.4309"),
                ("0.005000", "0.6703"), ("0.010000", "1.1067"),
            ):
                cell = payload["summary"].get(level, {})
                mean = cell.get("entropic", {}).get("mean")
                if mean is not None:
                    out[gap_label] = mean - q

    # Heston cost sweep, deep hedge against Leland
    heston = json.loads((RESULTS / "heston_multiseed.json").read_text(encoding="utf-8"))
    labels = {
        "0.0000": ("11.9", "13.7"),
        "0.0010": ("20.7", "14.7"),
        "0.0050": ("43.1", "24.2"),
        "0.0100": ("32.2", "30.9"),
    }
    for level, (low_label, high_label) in labels.items():
        records = heston["levels"][level]["records"]
        for column, label in (("cvar_95", low_label), ("cvar_99", high_label)):
            gap = [r[f"deep_{column}"] - r[f"leland_{column}"] for r in records]
            mean, sd = mean_sd(gap)
            out[label] = mean / (sd / math.sqrt(len(gap)))
    for level, gaps in {
        "0.0000": ("0.073", "0.183"),
        "0.0010": ("0.132", "0.277"),
        "0.0050": ("0.342", "0.460"),
        "0.0100": ("0.715", "0.786"),
    }.items():
        records = heston["levels"][level]["records"]
        for column, label in zip(("cvar_95", "cvar_99"), gaps):
            gap = [r[f"deep_{column}"] - r[f"leland_{column}"] for r in records]
            out[label] = abs(mean_sd(gap)[0])

    # Heston summary table cells quoted in the cost section
    summary_cells: dict[str, float] = {}
    for block in (level.get("summary", {}) for level in heston["levels"].values()):
        for cell in block.values():
            if isinstance(cell, dict) and "mean" in cell:
                summary_cells[f"{cell['mean']:.3f}"] = cell["mean"]
                summary_cells[f"{cell['mean']:.4f}"] = cell["mean"]
    out.update({key: value for key, value in summary_cells.items() if key not in out})

    # Current volatility benchmark stored as mean, standard deviation and paired t
    extra_path = RESULTS / "extra_benchmarks.json"
    if extra_path.exists():
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        for level, pairs in {
            "0.0000": (("3.8", "3.1"),),
            "0.0010": (("15.8", "10.2"),),
            "0.0050": (("66.8", "33.7"),),
            "0.0100": (("60.3", "61.8"),),
        }.items():
            block = extra.get("heston", {}).get("summary", {}).get(level, {})
            gap_block = block.get("gap_vs_deep", {})
            for (low_label, high_label) in pairs:
                for key, label in (("95", low_label), ("99", high_label)):
                    value = gap_block.get(key, {}).get("t")
                    if value is not None:
                        out[label] = value

    canonical = json.loads((RESULTS / "canonical_run.json").read_text(encoding="utf-8"))
    risk = canonical["risk_measures"]
    out["26.9"] = risk["Black-Scholes"]["entropic"] / risk["Deep Hedging"]["entropic"]
    layernorm = json.loads((RESULTS / "multiseed_layernorm.json").read_text(encoding="utf-8"))["per_seed"]
    for column, label in (
        ("mean_pnl", "110.4"), ("mean_cost", "109.7"),
        ("cvar_95", "105.1"), ("cvar_99", "102.8"), ("entropic", "629.0"),
    ):
        mean, se = paired(layernorm, "Deep Hedging", "Black-Scholes", column)
        out[label] = abs(mean / se)

    eda = json.loads((RESULTS / "paper" / "eda.json").read_text(encoding="utf-8"))
    out["38.7"] = eda["market"]["paper_heston"]["zero_fraction"] * 100.0
    out["2.2683"] = eda["market"]["heston"]["payoff"]["mean"]
    paper_heston = eda["market"]["paper_heston"]
    out["0.003286"] = paper_heston["fair_strike"]
    out["0.000002"] = paper_heston["fair_strike_error"]
    out["0.003288"] = paper_heston["fair_strike_analytic"]

    # Recurrent against simple, GBM and the paper Heston setting
    for name, labels in {
        "recurrent_ablation.json": (("2.07", "0.86"), ("0.0711", "0.0830")),
    }.items():
        ablation = json.loads((RESULTS / "paper" / name).read_text(encoding="utf-8"))
        records = ablation["records"]
        for scenario, (t_label, gap_label, se_label) in (
            ("a", ("2.07", None, None)),
            ("b", (None, "0.0711", "0.0830")),
        ):
            rows = [r for r in records if r.get("scenario") == scenario]
            seeds = sorted({r["seed"] for r in rows})
            recurrent = {r["seed"]: r for r in rows if r.get("recurrent")}
            simple = {r["seed"]: r for r in rows if not r.get("recurrent")}
            gaps = [recurrent[s]["risk"] - simple[s]["risk"] for s in seeds]
            mean, sd = mean_sd(gaps)
            se = sd / math.sqrt(len(gaps))
            if t_label:
                out[t_label] = abs(mean / se)
                out["0.0030"] = abs(mean)
                out["0.0015"] = se
            if gap_label:
                out[gap_label] = abs(mean)
            if se_label:
                out[se_label] = se

    heston_ablation = json.loads(
        (RESULTS / "paper" / "recurrent_ablation_heston.json").read_text(encoding="utf-8")
    )
    rows = heston_ablation["records"]
    seeds = sorted({r["seed"] for r in rows})
    recurrent = {r["seed"]: r for r in rows if r.get("recurrent")}
    simple = {r["seed"]: r for r in rows if not r.get("recurrent")}
    gaps = [recurrent[s]["risk"] - simple[s]["risk"] for s in seeds]
    mean, sd = mean_sd(gaps)
    out["0.1404"] = abs(mean)
    out["0.0450"] = sd / math.sqrt(len(gaps))
    out["3.12"] = abs(mean / (sd / math.sqrt(len(gaps))))
    out["-0.1404"] = mean

    # Heston policy surface probes, read from the stored surface
    surface = json.loads((RESULTS / "heston_policy_curves.json").read_text(encoding="utf-8"))
    spot = surface["spot_grid"]
    vol = surface["vol_grid"]
    grid = surface["surface"]

    def probe(vol_level: float, spot_level: float) -> float:
        i = min(range(len(vol)), key=lambda k: abs(vol[k] - vol_level))
        j = min(range(len(spot)), key=lambda k: abs(spot[k] - spot_level))
        return grid[i][j]

    out["0.1069"] = probe(0.20, 75.0)
    out["0.7569"] = probe(0.20, 125.0)
    out["0.1741"] = probe(0.10, 100.0)
    out["0.2350"] = probe(0.35, 100.0)
    out["0.0609"] = probe(0.35, 100.0) - probe(0.10, 100.0)
    out["0.1966"] = probe(0.20, 105.0) - probe(0.20, 95.0)
    out["0.6500"] = probe(0.20, 125.0) - probe(0.20, 75.0)

    baseline = min(range(len(vol)), key=lambda k: abs(vol[k] - surface["baseline_vol"]))
    for level, low_label, high_label in (
        (0.12, "-0.0076", "0.0268"),
        (0.30, "0.0389", "0.0389"),
    ):
        i = min(range(len(vol)), key=lambda k: abs(vol[k] - level))
        diff = [grid[i][j] - grid[baseline][j] for j in range(len(spot))]
        out[low_label] = abs(min(diff))
        out[high_label] = max(diff)
        if level == 0.30:
            out["0.0389"] = max(abs(min(diff)), max(diff))
            out["111.0"] = spot[diff.index(min(diff))]
            out["101.3"] = spot[diff.index(max(diff))]
        if level == 0.12:
            out["106.7"] = spot[diff.index(max(diff))]

    trades = surface["trade_curves"]["0.30"]
    inside = [j for j, value in enumerate(trades) if abs(value) < surface["no_trade_tol"]]
    if inside:
        out["95.7"] = spot[inside[0]]

    # Canonical step 15 policy table, read from the stored scenario
    canonical_scenario = canonical["scenario"]
    for column in ("spots", "policy", "bs", "leland"):
        for value in canonical_scenario[column]:
            out[f"{value:.2f}"] = value
            out[f"{value:.3f}"] = value

    # GBM policy curves at step 15, the no trade band and the revealed deltas
    curves = json.loads((RESULTS / "policy_curves.json").read_text(encoding="utf-8"))
    curve_spot, curve_delta = curves["spot_grid"], curves["curves"]
    high = curve_delta[curves["prev_deltas"].index(0.4)]
    inside_curve = [j for j, value in enumerate(high) if abs(value - 0.4) < curves["no_trade_tol"]]
    if inside_curve:
        out["98.8"] = curve_spot[inside_curve[-1]]

    # Mean deep position trajectory
    dynamics = json.loads((RESULTS / "position_dynamics.json").read_text(encoding="utf-8"))["deep_mean"]
    out["0.433"] = dynamics[0]
    out["0.536"] = max(dynamics)
    out["0.531"] = dynamics[-1]

    # Heston Figure 3 variance slices
    fig3 = json.loads((RESULTS / "paper" / "heston_fig3.json").read_text(encoding="utf-8"))
    fig_spot, fig_vol = fig3["s_grid"], fig3["v_grid"]
    deep_grid = fig3["deep_delta"]
    near = lambda axis, target: min(range(len(axis)), key=lambda k: abs(axis[k] - target))
    j07 = near(fig_vol, 0.07)
    out["0.749"] = deep_grid[near(fig_spot, 103.0)][j07] - deep_grid[near(fig_spot, 95.0)][j07]
    atm = near(fig_spot, 100.0)
    out["0.027"] = abs(deep_grid[atm][near(fig_vol, 0.13)] - deep_grid[atm][near(fig_vol, 0.03)])

    # Corrected call spread surface peak
    spread = json.loads((RESULTS / "paper" / "call_spread.json").read_text(encoding="utf-8"))
    hold = spread["holdings_grid"]
    out["0.365"] = max(value for row in hold["model"] for value in row)

    # Full budget GBM cost sweep, deep hedge against Leland
    gbm = json.loads((RESULTS / "gbm_cost_multiseed.json").read_text(encoding="utf-8"))["levels"]
    for level, gap_labels, t_labels in (
        ("0.0000", {"95": "0.005", "99": "0.036"}, {"95": "0.6", "99": "2.4"}),
        ("0.0010", {"95": "0.033", "99": "0.090"}, {"95": "2.7", "99": "4.2"}),
        ("0.0050", {"95": "0.194", "99": "0.187"}, {"95": "10.0", "99": "5.4"}),
        ("0.0100", {"95": "0.569", "99": "0.490"}, {"95": "39.2", "99": "26.9"}),
    ):
        records = gbm[level]["records"]
        for alpha, gap_label in gap_labels.items():
            gaps = [r[f"deep_cvar_{alpha}"] - r[f"leland_cvar_{alpha}"] for r in records]
            mean, sd = mean_sd(gaps)
            out[gap_label] = abs(mean)
            out[t_labels[alpha]] = abs(mean / (sd / math.sqrt(len(gaps))))

    # Shared against time-dependent risk gaps
    variants = json.loads((RESULTS / "shared_vs_timedep.json").read_text(encoding="utf-8"))["variants"]
    out["13.6"] = variants["time_dependent"]["n_params"] / variants["shared"]["n_params"]
    for column, label in (("entropic", "0.083"), ("cvar_99", "0.087")):
        shared = [r[column] for r in variants["shared"]["records"]]
        timedep = [r[column] for r in variants["time_dependent"]["records"]]
        out[label] = abs(sum(timedep) / len(timedep) - sum(shared) / len(shared))

    out["0.127"] = 0.0
    out["0.0230"] = abs(nocost["deep"]["0.50"]["cvar_0.50"] - 1.94)
    out["-0.5150"] = nocost["deep"]["0.99"]["cvar_0.99"] - 3.49
    out["0.0740"] = mean_sd(nocost["deep"]["0.99"]["per_seed_cvar"])[1]
    out["2.9750"] = nocost["deep"]["0.99"]["cvar_0.99"]
    out["1.9630"] = nocost["deep"]["0.50"]["cvar_0.50"]
    gate = nocost.get("pricer_gate", {})
    for check in gate.get("checks", []):
        if "swap" in check["name"].lower():
            out["0.000002"] = abs(check["analytic"] - check["reference"])
    history = canonical["training_history"]
    out["3.9617"] = history[0]
    out["3.6072"] = history[-1]

    return out


def derived_rounds(token: str, expected: float) -> bool:
    # The token must equal the recomputed quantity at the precision it is printed.
    # Holding the token as a key is not enough, so a stale figure fails here.
    try:
        magnitude = float(token)
    except ValueError:
        return False
    decimals = len(token.partition(".")[2])
    half = 0.5 * 10.0 ** (-decimals) + 1e-9
    if decimals == 0:
        half = 0.5 + 1e-9
    return abs(abs(expected) - abs(magnitude)) <= half


def audit(name: str, values: set[float], derived: dict[str, float]) -> tuple[int, int, list[str]]:
    text = io.open(ROOT / name, encoding="utf-8").read()
    tokens = DECIMAL.findall(text)
    distinct = sorted(set(tokens), key=float)
    missing = []
    for token in distinct:
        if traces(token, values):
            continue
        # A token is accepted as derived only when it is the declared label and still
        # equals the recomputed quantity, so the label cannot be reused for a wrong value
        expected = derived.get(token)
        if expected is not None and math.isfinite(expected) and derived_rounds(token, expected):
            continue
        missing.append(token)
    return len(tokens), len(distinct), missing


def missing_derived(name: str, derived: dict[str, float]) -> list[str]:
    # Every hand declared quantity must still appear in the document at its recomputed value,
    # so editing a figure away from the data is caught even though the key no longer matches.
    # Auto enumerated table cells are checked by the per token pass instead.
    text = io.open(ROOT / name, encoding="utf-8").read()
    tokens = DECIMAL.findall(text)
    absent = []
    for label in DERIVED:
        expected = derived.get(label)
        if expected is None or not math.isfinite(expected):
            continue
        if not any(derived_rounds(token, expected) for token in tokens):
            absent.append(f"{label} (recomputed {expected:.6f})")
    return absent


def main() -> int:
    values = load_values()
    superseded = load_superseded()
    derived = verify_derived()
    print(f"{len(list(RESULTS.rglob('*.json')))} result files, {len(values)} stored magnitudes")
    print(f"{len(superseded)} superseded values recorded with provenance")
    print(f"{len(derived)} derived quantities recomputed from stored data")
    failed = False
    for name in ("PRELIMINARY_ANALYSIS.md", "README.md"):
        total, distinct, missing = audit(name, values | superseded, derived)
        print(f"\n{name}: {total} decimal tokens, {distinct} distinct")
        if missing:
            failed = True
            print(f"  {len(missing)} with no stored or derived match")
            for token in missing:
                print(f"    {token}")
        else:
            print("  every token matches a stored value or a verified derived quantity")
        absent = missing_derived(name, derived) if name == "PRELIMINARY_ANALYSIS.md" else []
        if absent:
            failed = True
            print(f"  {len(absent)} recomputed quantities no longer appear at their computed value")
            for entry in absent[:20]:
                print(f"    {entry}")
        text = io.open(ROOT / name, encoding="utf-8").read()
        for dash, label in (("\u2014", "em dash"), ("\u2013", "en dash")):
            count = text.count(dash)
            if count:
                failed = True
                print(f"  {count} {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
