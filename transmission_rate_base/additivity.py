from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_STYLE_PATH = Path(__file__).parent / "data" / "style_inputs.csv"
_CONTROL_COLS = ["util_beta", "xlu", "dividend_yield", "low_vol", "size",
                 "momentum", "capex_intensity", "value"]


def load_style_inputs(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or _STYLE_PATH)


def _ls_from_char(char: pd.Series, fwd_monthly: pd.DataFrame, high_minus_low: bool) -> pd.Series:
    s = char.dropna().sort_values()
    if len(s) < 2:
        return pd.Series(0.0, index=fwd_monthly.index)
    k = max(1, len(s) // 5)
    top, bot = s.index[-k:], s.index[:k]
    long_, short_ = (top, bot) if high_minus_low else (bot, top)
    return fwd_monthly[list(long_)].mean(axis=1) - fwd_monthly[list(short_)].mean(axis=1)


def build_controls(daily_ret: pd.DataFrame, ew_util_daily: pd.Series, xlu_daily: pd.Series,
                   style_inputs: pd.DataFrame, rebal_dates) -> pd.DataFrame:
    """Month-end control-factor panel. util_beta / xlu are index returns; the
    other six are annually-rebalanced top-minus-bottom-quintile L/S returns on
    the named characteristic."""
    monthly = (1 + daily_ret).resample("ME").prod() - 1
    ctrl = pd.DataFrame(index=monthly.index)
    ctrl["util_beta"] = (1 + ew_util_daily).resample("ME").prod() - 1
    ctrl["xlu"] = (1 + xlu_daily).resample("ME").prod() - 1

    char_specs = [
        ("dividend_yield", lambda si: si.set_index("ticker")["dividend_yield"], True),
        ("low_vol", None, False),
        ("size", lambda si: np.log(si.set_index("ticker")["market_cap"]), False),
        ("momentum", None, True),
        ("capex_intensity",
         lambda si: si.set_index("ticker")["capex"] / si.set_index("ticker")["net_ppe"], True),
        ("value",
         lambda si: si.set_index("ticker")["book_value"] / si.set_index("ticker")["market_cap"], True),
    ]
    for name, fn, hml in char_specs:
        ctrl[name] = 0.0
        for d in rebal_dates:
            fy = d.year - 1
            si = style_inputs[style_inputs["fy"] == fy]
            span = monthly.index[(monthly.index > d) &
                                 (monthly.index <= d + pd.DateOffset(years=1))]
            if len(span) == 0:
                continue
            fwd = monthly.loc[span]
            hist = daily_ret.loc[:d]
            if name == "low_vol":
                if len(hist) < 60:
                    continue
                char = hist.iloc[-252:].std()
            elif name == "momentum":
                if len(hist) < 252:
                    continue
                px = (1 + hist).cumprod()
                char = px.iloc[-1] / px.iloc[-252:-21].iloc[0] - 1
            else:
                if si.empty:
                    continue
                char = fn(si)
            ctrl.loc[span, name] = _ls_from_char(char.reindex(fwd.columns), fwd, hml).values
    return ctrl[_CONTROL_COLS]


def _ols_hac(y: np.ndarray, X: np.ndarray, maxlags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with Newey-West (Bartlett-kernel) HAC standard errors.

    Returns (beta, se, r2). X must already include an intercept column.
    """
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    xu = X * resid[:, None]
    S = xu.T @ xu
    n = len(y)
    for lag in range(1, min(maxlags, n - 1) + 1):
        w = 1.0 - lag / (maxlags + 1.0)
        g = xu[lag:].T @ xu[:-lag]
        S += w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, r2


def run_additivity(factor_monthly_ret: pd.Series, controls: pd.DataFrame,
                   maxlags: int = 12) -> dict:
    """Regress factor ~ 1 + controls with Newey-West (12-lag) SEs. Returns the
    annualised intercept ('residual alpha'), its t-stat, R², coefficients, N."""
    df = pd.concat([factor_monthly_ret.rename("y"), controls], axis=1).dropna()
    y = df["y"].to_numpy(dtype=float)
    Xdf = df[_CONTROL_COLS]
    X = np.column_stack([np.ones(len(df)), Xdf.to_numpy(dtype=float)])
    beta, se, r2 = _ols_hac(y, X, maxlags)
    names = ["const"] + _CONTROL_COLS
    alpha, alpha_se = beta[0], se[0]
    return {
        "alpha_ann": float(alpha * 12),
        "alpha_t": float(alpha / alpha_se) if alpha_se > 0 else np.nan,
        "r2": float(r2),
        "coef": {n: float(b) for n, b in zip(names[1:], beta[1:])},
        "n": int(len(df)),
    }
