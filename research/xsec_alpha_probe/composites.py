"""Equal-weight composites of the run.py signals ("all signals together"), 2026-09-25.

    python -m research.xsec_alpha_probe.composites

Pre-set before running: every component enters in its hypothesised + direction (no sign flips
chosen from results), mean of per-month winsorized z-scores, no fitting. Bar (user, 2026-09-24):
|NW t| > 1.5 in 2023-26 at 1m or 3m. 4 composites x 2 horizons = 8 tests (~1 false pass expected).
Caveat: component ICs were already seen in run.py, so this is not an out-of-sample test.
"""
from __future__ import annotations

import pandas as pd

from research.xsec_alpha_probe import data, evaluate as ev, run, signals as sg

POWER_BETAS = ["beta_energy", "beta_congestion", "beta_scarcity"]
POWER_SHOCKS = ["bxs_energy", "bxs_congestion", "bxs_scarcity"]


def composite(sigs: dict[str, pd.DataFrame], names: list[str], univ: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for n in names:
        raw = sigs[n]
        cols = raw.columns.intersection(univ.columns)
        parts.append(ev.prep(raw.reindex(index=univ.index, columns=cols).where(univ[cols])))
    stacked = pd.concat(parts, keys=range(len(parts)))
    return stacked.groupby(level=1).mean()  # mean of available z's per name-month


def main() -> pd.DataFrame:
    px_d = data.prices()
    univ = data.universe()
    px_m = data.monthly_prices(px_d)
    ret_m = px_m.pct_change(fill_method=None)
    fwd1, fwd3 = ret_m.shift(-1), px_m.shift(-3) / px_m - 1
    bm = data.monthly_prices(data.benchmarks())
    grid_fwd = bm["GRID"].pct_change().shift(-1)
    sigs = {}
    sigs.update(sg.fundamental_signals())
    sigs.update(sg.price_signals(px_m))
    sigs.update(sg.power_signals(px_m, bm["SPY"]))
    cases = {
        "power_all (6)": POWER_BETAS + POWER_SHOCKS,
        "power_betas (3)": POWER_BETAS,
        "power_shocks (3)": POWER_SHOCKS,
        f"everything ({len(sigs)})": list(sigs),
    }
    risks = run.risk_panels(px_d)
    rows = []
    for name, members in cases.items():
        comp = composite(sigs, members, univ)
        r = run.evaluate_signal(name, comp, univ, fwd1, fwd3, risks, grid_fwd)
        r["PASS (|t|>1.5 2023-26, 1m or 3m)"] = max(abs(r["t1m_2023-26"]), abs(r["t3m_2023-26"])) > 1.5
        rows.append(r)
    res = pd.DataFrame(rows).set_index("signal")
    res.to_csv(data.CACHE / "results_composites.csv")
    return res


if __name__ == "__main__":
    res = main()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = [c for c in res.columns if c.startswith(("t1m", "t3m", "IC", "SR_eq", "SR_grid", "names", "PASS"))]
    print(res[cols].round(2).T.to_string())
