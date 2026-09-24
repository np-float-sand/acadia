"""Monthly power-market factor series from the grid_resilience ISO caches.

Three series, each built per ISO and then averaged in logs so no single market
dominates:

* ``energy``    - mean day-ahead LMP at PJM trading hubs and the ERCOT hub average.
* ``congestion``- mean |congestion component| of day-ahead LMP at PJM zones
                  (ERCOT's feed has none).

MISO and CAISO are deliberately excluded: their caches are empty before 2023, and
a basket whose membership changes mid-sample has level breaks that look like shocks.
* ``scarcity``  - 99th-percentile hourly ERCOT hub-average price in the month.

A series is published at month-end m and uses only hours inside month m, so it is
known at m. Power prices are strongly seasonal, so the *shock* used downstream is
the seasonal difference of the monthly log change (see `shocks`).
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd

from research.xsec_alpha_probe import data

FACTORS = ("energy", "congestion", "scarcity")


def _month(path: str) -> pd.Timestamp:
    ym = re.search(r"(\d{4})-(\d{2})\.parquet$", path).group(0)[:7]
    return pd.Timestamp(ym) + pd.offsets.MonthEnd(0)


def _files(prefix: str) -> list[str]:
    return sorted(glob.glob(str(data.iso_cache() / f"{prefix}_*.parquet")))


def _per_month(prefix: str, reader) -> pd.Series:
    out = {}
    for f in _files(prefix):
        try:
            v = reader(f)
        except Exception:  # a corrupt month is a gap, not a crash
            v = np.nan
        out[_month(f)] = v
    return pd.Series(out, dtype=float).sort_index()


def _pjm(f, col, types):
    d = pd.read_parquet(f, columns=["type", col])
    d = d[d["type"].isin(types)]
    return d[col].abs().mean() if col == "congestion" else d[col].mean()


def _ercot(f, stat):
    d = pd.read_parquet(f, columns=["location", "lmp"])
    s = d.loc[d["location"] == "HB_HUBAVG", "lmp"]
    return s.mean() if stat == "mean" else s.quantile(0.99)


def build(refresh: bool = False) -> pd.DataFrame:
    path = data.CACHE / "power_factors.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    energy = pd.concat({
        "pjm": _per_month("pjm_lmp_DAY_AHEAD_HOURLY", lambda f: _pjm(f, "lmp", {"HUB"})),
        "ercot": _per_month("ercot_lmp_settlement point", lambda f: _ercot(f, "mean")),
    }, axis=1)
    cong = pd.concat({
        "pjm": _per_month("pjm_lmp_ZONE", lambda f: _pjm(f, "congestion", {"ZONE"})),
    }, axis=1)
    scar = _per_month("ercot_lmp_settlement point", lambda f: _ercot(f, "p99"))
    out = pd.DataFrame({
        "energy": np.log(energy.where(energy > 0)).mean(axis=1, skipna=False),
        "congestion": np.log(cong.where(cong > 0)).mean(axis=1, skipna=False),
        "scarcity": np.log(scar.where(scar > 0)),
    })
    out.to_parquet(path)
    return out


def shocks(levels: pd.DataFrame) -> pd.DataFrame:
    """Seasonally-differenced monthly log change: (x_m - x_{m-1}) - (x_{m-12} - x_{m-13})."""
    d = levels.diff()
    return d - d.shift(12)


def yoy(levels: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year log change - the deseasonalised 'state' of each market."""
    return levels - levels.shift(12)
