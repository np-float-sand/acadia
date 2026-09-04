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
