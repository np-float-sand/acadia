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
