"""
STEO forward spark-spread as a JOINT regressor alongside the EIA Form 860M
generator-pipeline signal (Thread 6, docs/interconnection-queue-signal-results.md),
on the same pooled 5-name generator-owner basket (AES/BEP/CWEN/NEE/NRG).

Context: Thread 6 tested the EIA pipeline signal ALONE and found a real but modest
in-sample correlation to the basket's forward 3-month return (r=-0.34, t=-1.74/-1.76,
n=26 quarters, 2020-01->2026-04) that failed to survive being turned into any
tradeable rule (5 separate constructions, all underperform a static long). This
probe asks a narrower, genuinely untried question: does STEO's forward power-price
signal add information ALONGSIDE the EIA pipeline signal (not instead of it, and not
paired with GSI as in the other STEO threads) on this specific basket? NRG/NEE lean
more merchant than the electrification-supplier book STEO already failed on, so the
mechanism is more plausible here -- but EIA-alone already failed on every tradeable
construction tried, so the prior is low.

Data: `research/datacenter_reit_cep/eia_pooled5_check.csv` (Thread 6's exact cached
panel -- reproduced its r=-0.335/-0.338, t=-1.739/-1.762 before using it, see this
script's own printed sanity check) + `research/steo_spark_spread/data/steo_trend_
{PJM,ERCOT}.parquet` (already point-in-time safe by construction).

Design: STEO only covers 2021-01-19 onward, so the joint model necessarily drops to
a shorter window than Thread 6's full n=26 -- flagged explicitly, not hidden. Per the
Thread 7 precedent (same STEO-short-history problem), test at the fit-period stage
first: does STEO's own coefficient clear a bar in a joint OLS, on the SAME restricted
window as an EIA-alone comparison (apples-to-apples)? Only worth a fit/holdout split
or a combined-book number if that clears -- otherwise report FAIL and stop, exactly
as Thread 7 did.

Pre-registered gate: STEO coefficient |t| >= 1.5 in the joint model, AND the joint
model's R^2 improves over EIA-alone on the identical (STEO-covered) window.

Throwaway probe. Findings -> docs/interconnection-queue-signal-results.md,
docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, ".")

HERE = pathlib.Path(__file__).parent
EIA_PATH = pathlib.Path("research/datacenter_reit_cep/eia_pooled5_check.csv")


def load_steo_composite() -> pd.Series:
    pjm = pd.read_parquet(HERE / "data" / "steo_trend_PJM.parquet")["steo_spark_trend_z"]
    ercot = pd.read_parquet(HERE / "data" / "steo_trend_ERCOT.parquet")["steo_spark_trend_z"]
    df = pd.concat([pjm.rename("pjm"), ercot.rename("ercot")], axis=1)
    return df.mean(axis=1, skipna=True).dropna().rename("steo_composite").sort_index()


def main():
    eia = pd.read_csv(EIA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # Sanity check: reproduce Thread 6's reported stats before trusting this cache.
    r_level, _ = stats.pearsonr(eia["planned_mw"], eia["fwd_ret"])
    n0 = len(eia)
    t_level = r_level * (n0 - 2) ** 0.5 / (1 - r_level ** 2) ** 0.5
    print(f"Sanity check vs. Thread 6 (expect r=-0.335, t=-1.739, n=26): "
          f"r={r_level:.3f}, t={t_level:.3f}, n={n0}")
    assert abs(r_level - (-0.335)) < 0.01, "EIA panel doesn't match Thread 6 -- stop."

    steo = load_steo_composite()
    print(f"STEO composite: {len(steo)} days, {steo.index.min().date()} -> {steo.index.max().date()}")

    # Align each EIA quarter-date to STEO's point-in-time value as of that date
    # (merge_asof, backward -- most recent STEO value known on/before the EIA date).
    eia_sorted = eia.sort_values("date")
    steo_df = steo.reset_index()
    steo_df.columns = ["date", "steo_composite"]
    merged = pd.merge_asof(eia_sorted, steo_df, on="date", direction="backward")
    merged = merged.dropna(subset=["steo_composite", "fwd_ret", "planned_mw"])
    n = len(merged)
    print(f"\nSTEO-covered window: n={n} quarters, "
          f"{merged['date'].min().date()} -> {merged['date'].max().date()} "
          f"(Thread 6 full sample was n={n0})")

    if n < 8:
        print("Too few STEO-covered quarters for a meaningful regression. STOP.")
        return

    y = merged["fwd_ret"].to_numpy()

    # EIA-alone, restricted to the SAME window (apples-to-apples with the joint model).
    X_eia = sm.add_constant(merged[["planned_mw"]])
    m_eia = sm.OLS(y, X_eia).fit()
    print(f"\nEIA-alone, STEO-covered window only (n={n}):")
    print(f"  planned_mw coef t-stat: {m_eia.tvalues['planned_mw']:.3f}   R^2: {m_eia.rsquared:.3f}")

    # Joint: EIA + STEO.
    X_joint = sm.add_constant(merged[["planned_mw", "steo_composite"]])
    m_joint = sm.OLS(y, X_joint).fit()
    print(f"\nJoint (EIA + STEO), n={n}:")
    print(f"  planned_mw coef t-stat:     {m_joint.tvalues['planned_mw']:.3f}")
    print(f"  steo_composite coef t-stat: {m_joint.tvalues['steo_composite']:.3f}")
    print(f"  R^2: {m_joint.rsquared:.3f}  (vs. EIA-alone R^2 {m_eia.rsquared:.3f})")

    gate_a = abs(m_joint.tvalues["steo_composite"]) >= 1.5
    gate_b = m_joint.rsquared > m_eia.rsquared
    print(f"\nGate (a) STEO |t| >= 1.5 in joint model: {gate_a} "
          f"(actual {abs(m_joint.tvalues['steo_composite']):.3f})")
    print(f"Gate (b) joint R^2 > EIA-alone R^2 (same window): {gate_b} "
          f"({m_joint.rsquared:.3f} vs {m_eia.rsquared:.3f})")
    print(f"VERDICT: {'PASS -- worth a fit/holdout follow-up' if (gate_a and gate_b) else 'FAIL -- stop here'}")

    merged.to_csv(HERE / "data" / "eia_steo_joint_panel.csv", index=False)


if __name__ == "__main__":
    main()
