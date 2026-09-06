"""Graduated 'sell-when-euphoric' de-risk multiplier (spec 4.1).

A trim, not a crash shield: ~2.7pp MaxDD improvement on the concentrated book,
~0 on the broadened book but +Sharpe. Better-behaved than a binary trend gate.
Decided at each month-end, applied to the following month (no lookahead).
"""
from __future__ import annotations

import pandas as pd

from electrification_strategy import config
from electrification_strategy._util import month_hold


def extension_multiplier(basket_index: pd.Series, spy_ret: pd.Series, scale: float = 1.0) -> pd.Series:
    lvl = basket_index.astype(float).sort_index()
    spy_lvl = (1.0 + spy_ret.reindex(lvl.index).fillna(0.0)).cumprod()

    ext = lvl / lvl.rolling(config.VAL_MA_DAYS).mean() - 1.0
    rs12 = lvl.pct_change(config.VAL_RS_DAYS) - spy_lvl.pct_change(config.VAL_RS_DAYS)

    lo_e, hi_e = config.VAL_EXT_LO * scale, config.VAL_EXT_HI * scale
    lo_r, hi_r = config.VAL_RS_LO * scale, config.VAL_RS_HI * scale

    m = pd.Series(config.VAL_MULT_MID, index=lvl.index)
    m[(ext <= lo_e) & (rs12 <= lo_r)] = 1.0
    m[(ext > hi_e) | (rs12 > hi_r)] = config.VAL_MULT_LOW
    m[ext.isna() | rs12.isna()] = 1.0            # warm-up -> fully invested

    return month_hold(m, lvl.index, fill=1.0)
