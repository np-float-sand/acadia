"""
Assemble a monthly grid-demand nowcast from the cached daily GSI series.

``grid_resilience.main`` writes one CSV per ISO (``output/gsi_<iso>.csv``) with a
daily ``gsi`` column in [0, 1] plus its four z-scored sub-signals. Only ERCOT and
PJM have the full 2018-2025 history; MISO/CAISO/SPP start in 2023-2025 and are
excluded from the default nowcast so the panel has a consistent span.

The nowcast is deliberately simple: the cross-sectional average of the chosen
ISOs' daily ``gsi``, resampled to month-end. ``build_monthly_nowcast`` can return
either the level or the month-over-month change (the "sensitivity" regressor).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FULL_HISTORY_ISOS: tuple[str, ...] = ("ercot", "pjm")
SUBSIGNALS: tuple[str, ...] = (
    "lmp_zscore",
    "congestion_frac_z",
    "reserve_tightness_z",
    "event_flag",
    "gsi",
)


def load_daily_gsi(
    output_dir: str | Path = "output",
    isos: tuple[str, ...] = FULL_HISTORY_ISOS,
    column: str = "gsi",
) -> pd.DataFrame:
    """
    Load ``output/gsi_<iso>.csv`` for each iso and return a wide DataFrame
    (index = date, columns = uppercased iso) of the requested column.

    Raises FileNotFoundError if a requested iso CSV is missing.
    """
    if column not in SUBSIGNALS:
        raise ValueError(f"column must be one of {SUBSIGNALS}, got {column!r}")

    out_dir = Path(output_dir)
    series: dict[str, pd.Series] = {}
    for iso in isos:
        path = out_dir / f"gsi_{iso}.csv"
        if not path.exists():
            raise FileNotFoundError(f"cached GSI not found: {path}")
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        series[iso.upper()] = df[column]

    wide = pd.DataFrame(series).sort_index()
    return wide


def build_monthly_nowcast(
    daily_gsi: pd.DataFrame,
    how: str = "change",
    standardize: bool = True,
) -> pd.Series:
    """
    Collapse a wide daily GSI frame to a single monthly nowcast series.

    Steps: average across ISO columns (skipping NaN) -> month-end mean ->
    optionally first-difference (``how="change"``) -> optionally z-score the
    whole series (mean/std over the sample).

    Parameters
    ----------
    how         : "level" keeps the monthly level; "change" returns the
                  month-over-month difference (the sensitivity regressor).
    standardize : subtract mean / divide by std of the resulting series.

    Returns
    -------
    Series indexed by month-end Timestamp, name "nowcast".
    """
    if how not in ("level", "change"):
        raise ValueError(f"how must be 'level' or 'change', got {how!r}")

    cross_iso = daily_gsi.mean(axis=1, skipna=True)
    monthly = cross_iso.resample("ME").mean().dropna()

    if how == "change":
        monthly = monthly.diff().dropna()

    if standardize and monthly.std(ddof=0) > 0:
        monthly = (monthly - monthly.mean()) / monthly.std(ddof=0)

    return monthly.rename("nowcast")


def seasonality_strength(monthly_level: pd.Series) -> dict[str, float]:
    """
    Crude diagnostic: how much of the monthly nowcast *level* variance is a
    fixed calendar-month effect (i.e. weather seasonality) vs. everything else.

    Returns {"month_r2": R^2 of a month-of-year dummy regression,
             "resid_ac1": lag-1 autocorrelation of the residual}.
    A high ``month_r2`` means the series is mostly a seasonal weather pattern
    and carries little independent business-cycle information.
    """
    s = monthly_level.dropna()
    if len(s) < 24:
        return {"month_r2": float("nan"), "resid_ac1": float("nan")}

    month = s.index.month
    grand = s.mean()
    fitted = s.groupby(month).transform("mean")
    ss_tot = ((s - grand) ** 2).sum()
    ss_res = ((s - fitted) ** 2).sum()
    month_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    resid = s - fitted
    if resid.std(ddof=0) < 1e-12:
        resid_ac1 = float("nan")  # perfectly seasonal -> no residual to correlate
    else:
        resid_ac1 = resid.autocorr(lag=1)
    return {"month_r2": float(month_r2), "resid_ac1": float(resid_ac1)}
