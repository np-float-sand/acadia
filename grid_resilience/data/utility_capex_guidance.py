"""Utility 5-yr capex-guidance revision panel -- "Deliverable D" data layer.

A hand/web-assembled table of forward-multi-year capital-program guidance for
15 large US electric utilities, one row per distinct dated point where the
company stated or revised its plan. See
docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md s3.2 for the
full column contract and sourcing discipline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED_CSV = Path(__file__).parent / "seed" / "utility_capex_guidance.csv"

DC_BASIS = ("stated", "derived", "qualitative", "none")


def load_capex_guidance(seed: Path = SEED_CSV) -> pd.DataFrame:
    """Load the seed panel. Adds `revision_quality` ('stated' when the CSV
    itself carried a revision figure, 'derived' when computed here from
    consecutive plan levels, 'n/a' for a utility's first vintage) and
    `prior_capex_plan_usd_m` (the immediately-prior plan level, the base a
    revision is measured against -- used by aggregate_revision_series's
    percent form)."""
    df = pd.read_csv(seed)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["utility", "report_date"]).reset_index(drop=True)

    stated = df["revision_vs_prior_usd_m"].notna()
    prior_plan = df.groupby("utility")["capex_plan_usd_m"].shift(1)
    derived_ok = prior_plan.notna() & df["capex_plan_usd_m"].notna()
    derived_fill = df["capex_plan_usd_m"] - prior_plan

    df["prior_capex_plan_usd_m"] = prior_plan
    df["revision_vs_prior_usd_m"] = df["revision_vs_prior_usd_m"].where(
        stated, derived_fill.where(derived_ok))
    df["revision_quality"] = np.select(
        [stated, ~stated & derived_ok], ["stated", "derived"], default="n/a")
    return df


def feasibility_summary(df: pd.DataFrame,
                        window: tuple[str, str] = ("2023-01-01", "2026-08-31")) -> dict:
    """Per-utility usable-plan / in-window-revision counts (spec s3.3). A
    utility is "usable" if it has >=1 row with a non-null `capex_plan_usd_m`
    AND >=1 row with a non-null `revision_vs_prior_usd_m` whose `report_date`
    falls inside `window`."""
    ws, we = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    per_utility: dict[str, dict] = {}
    for u, g in df.groupby("utility"):
        has_plan = bool(g["capex_plan_usd_m"].notna().any())
        in_window = g[(g["report_date"] >= ws) & (g["report_date"] <= we)]
        n_revisions = int(in_window["revision_vs_prior_usd_m"].notna().sum())
        per_utility[u] = {"has_plan": has_plan, "n_revisions": n_revisions,
                          "usable": bool(has_plan and n_revisions >= 1)}
    n_usable = sum(1 for v in per_utility.values() if v["usable"])
    return {"per_utility": per_utility, "n_usable": n_usable, "n_total": len(per_utility)}


def panel_asof(df: pd.DataFrame, asof) -> pd.DataFrame:
    """Most-recent row per utility with `report_date <= asof` (point-in-time
    slice). A utility with no row on or before `asof` is simply absent."""
    asof_ts = pd.Timestamp(asof)
    known = df[df["report_date"] <= asof_ts]
    if known.empty:
        return known.iloc[0:0]
    idx = known.groupby("utility")["report_date"].idxmax()
    return known.loc[idx].sort_values("utility").reset_index(drop=True)


def impute_dc_attributed(df: pd.DataFrame, *, max_iter: int = 5,
                         tol: float = 0.005, imputed_weight: float = 0.5) -> pd.DataFrame:
    """Point-in-time, EM-style fill of the data-center-attributed dollar figure
    for rows where the utility only qualitatively mentioned data centers (or
    not at all). Spec s4.2.

    For a row with `dc_basis` in {"qualitative", "none"}, the fill is
    `revision_vs_prior_usd_m * ratio(t)`, where `ratio(t)` is the mean of
    `dc_attributed_usd_m / revision_vs_prior_usd_m` over every "stated"/
    "derived" row with `report_date <= t` -- only already-disclosed data as of
    that date, never a later quarter's. After the first pass, `ratio(t)` is
    recomputed including the freshly-imputed rows (down-weighted
    `imputed_weight` vs. a genuinely observed row) and the fill redone, for up
    to `max_iter` rounds or until the total imputed $ changes by less than
    `tol` (relative) between rounds. A row with no stated/derived observation
    anywhere in the panel before its own date keeps `dc_attributed_usd_m_filled`
    NaN -- there is nothing yet to calibrate the ratio from.

    Sorting is by `["report_date", "utility"]` with a stable sort so that
    same-day rows from different utilities get a deterministic tie order
    (alphabetical by utility) rather than one that depends on quicksort's
    behaviour on the input's pre-sort row order -- `ratio_asof` is a cumsum
    inclusive of each row's own sorted position, so an undefined same-day
    tiebreak would make results depend on incidental input row order.
    """
    out = df.sort_values(["report_date", "utility"], kind="stable").reset_index(drop=True).copy()
    observed = out["dc_basis"].isin(["stated", "derived"]) & out["dc_attributed_usd_m"].notna()
    has_rev = out["revision_vs_prior_usd_m"].notna() & (out["revision_vs_prior_usd_m"] != 0)
    needs_fill = out["dc_basis"].isin(["qualitative", "none"]) & has_rev

    out["dc_attributed_usd_m_filled"] = np.where(observed, out["dc_attributed_usd_m"], np.nan)
    out["dc_imputed"] = False

    pool_mask = observed & has_rev
    ratio_val = (out["dc_attributed_usd_m"] / out["revision_vs_prior_usd_m"]).where(pool_mask)
    ratio_wt = pd.Series(np.where(pool_mask, 1.0, np.nan), index=out.index)

    prev_total = None
    for _ in range(max_iter):
        weighted_sum = (ratio_val.fillna(0.0) * ratio_wt.fillna(0.0)).cumsum()
        weight_sum = ratio_wt.fillna(0.0).cumsum()
        ratio_asof = weighted_sum / weight_sum.replace(0.0, np.nan)

        fill_val = out["revision_vs_prior_usd_m"] * ratio_asof
        new_filled = out["dc_attributed_usd_m_filled"].where(observed, fill_val.where(needs_fill))
        out["dc_attributed_usd_m_filled"] = new_filled
        out["dc_imputed"] = needs_fill & new_filled.notna()

        total_now = float(np.nansum(new_filled[out["dc_imputed"]]))

        ratio_val = (out["dc_attributed_usd_m_filled"] / out["revision_vs_prior_usd_m"]).where(
            observed | out["dc_imputed"])
        ratio_wt = pd.Series(
            np.where(observed, 1.0, np.where(out["dc_imputed"], imputed_weight, np.nan)),
            index=out.index)

        if prev_total is not None and abs(prev_total) > 1e-9 and \
                abs(total_now - prev_total) / abs(prev_total) < tol:
            break
        prev_total = total_now

    return out
