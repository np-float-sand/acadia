"""
Split-sample hub selection (pre-registration Decision 2).

Hub selection is done ONLY on the train window (2023-01-01 -> 2024-12-31) using
the GARCH-style variance regression (same spec, same controls, as the
gate-bearing test-window regression) against all 19 hubs. The hub with the
largest |t| AND the correct pre-registered sign is selected. If no hub has the
correct sign, "no hub selected" is recorded -- never fall back to the
least-wrong hub.
"""
from __future__ import annotations

import pandas as pd

from research.electricity_price_volatility import data as dm
from research.electricity_price_volatility.regressions import variance_regression
from research.electricity_price_volatility.universe import ALL_HUBS


def scan_hubs_train_window(
    equity_var: pd.Series,
    hub_frames: dict[str, pd.DataFrame],
    control_var: pd.Series,
    control_ret: pd.Series,
    predicted_sign: int,
    equity_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Run the variance regression against every (iso, hub) in `hub_frames` over
    the equity_index window supplied (caller restricts to the train window).

    Returns a DataFrame indexed by hub key ("ISO:HUB") with columns
    coef, tstat, pvalue, n, sign_ok.
    """
    rows = []
    for hub_key, hub_frame in hub_frames.items():
        aligned = dm.align_lagged_hub_to_equity(hub_frame, equity_index, lag_days=1)
        result = variance_regression(
            equity_var.reindex(equity_index),
            aligned["hub_var"],
            control_var.reindex(equity_index),
            control_ret.reindex(equity_index),
        )
        sign_ok = (
            pd.notna(result["coef"])
            and (result["coef"] < 0 if predicted_sign < 0 else result["coef"] > 0)
        )
        rows.append(dict(hub=hub_key, sign_ok=sign_ok, **result))
    return pd.DataFrame(rows).set_index("hub")


def select_hub(scan_df: pd.DataFrame) -> str | None:
    """
    Pick the hub with the largest |t| among sign_ok rows. None if no hub
    has the correct sign (recorded explicitly, per Decision 2 -- never
    fall back to the least-wrong hub).
    """
    ok = scan_df[scan_df["sign_ok"]].dropna(subset=["tstat"])
    if ok.empty:
        return None
    return ok["tstat"].abs().idxmax()
