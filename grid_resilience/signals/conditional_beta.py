"""
Conditional price sensitivity estimation.

For each utility in the universe we estimate how its excess return
(return minus sector average) responds to rising grid stress.

Two complementary approaches:

  1. Event study  — compute cumulative abnormal returns around each
                    identified stress event window.  The average CAR
                    is the ticker's "event sensitivity."

  2. Rolling OLS  — regress daily excess return on the lagged GSI
                    over a rolling window, conditional on stress days.
                    The slope coefficient is the "stress beta."

A ticker with a large negative stress beta underperforms on high-stress
days → it is stress-sensitive → it goes in the short book.
A ticker with a near-zero or positive stress beta is resilient → long book.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

from grid_resilience.config import EVENT_WINDOW_PRE, EVENT_WINDOW_POST, MIN_STRESS_OBS


# ── Event study ───────────────────────────────────────────────────────────────

def event_study(
    returns: pd.DataFrame,
    events_df: pd.DataFrame,
    sector_return: pd.Series,
    pre:  int = EVENT_WINDOW_PRE,
    post: int = EVENT_WINDOW_POST,
) -> pd.DataFrame:
    """
    Compute cumulative abnormal returns (CAR) around each stress event.

    For each event window [t0, t1]:
      - Estimation window: [t0 - 60, t0 - 6] trading days
      - Event window: [t0 - pre, t0 + post] trading days
      - Expected return estimated from OLS on market/sector return
      - Abnormal return = actual - expected
      - CAR = sum of abnormal returns over the event window

    Parameters
    ----------
    returns        : daily log-return DataFrame (columns = tickers)
    events_df      : output of stress_events.build_full_event_calendar()
    sector_return  : equal-weighted sector return series
    pre / post     : days before/after event start to include in the window

    Returns
    -------
    DataFrame with columns: event_name, ticker, car, t_stat
    (one row per ticker × event)
    """
    rows = []

    for _, ev in events_df.iterrows():
        t0 = ev["window_start"]
        t1 = ev["window_end"]

        # Event window (calendar to business-day index)
        ev_start = returns.index[returns.index.searchsorted(t0 - pd.Timedelta(days=pre * 1.5))]
        ev_end   = returns.index[
            min(returns.index.searchsorted(t1 + pd.Timedelta(days=post * 1.5)),
                len(returns.index) - 1)
        ]
        ev_window_idx = returns.loc[ev_start:ev_end].index
        if len(ev_window_idx) < 3:
            continue

        # Estimation window (60 trading days before pre-window)
        est_end_pos   = returns.index.searchsorted(t0) - pre - 1
        est_start_pos = max(0, est_end_pos - 60)
        if est_end_pos <= est_start_pos:
            continue
        est_idx = returns.index[est_start_pos:est_end_pos]

        for ticker in returns.columns:
            r = returns[ticker]
            est_r = r.reindex(est_idx).dropna()
            est_mkt = sector_return.reindex(est_idx).dropna()
            common_est = est_r.index.intersection(est_mkt.index)
            if len(common_est) < 20:
                continue

            # OLS expected return
            if _HAS_SM:
                X = sm.add_constant(est_mkt.loc[common_est].values)
                Y = est_r.loc[common_est].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.OLS(Y, X).fit()
                alpha, beta = model.params[0], model.params[1]
            else:
                beta, alpha = np.polyfit(
                    est_mkt.loc[common_est].values,
                    est_r.loc[common_est].values, 1
                )

            # Abnormal returns over event window
            ev_r   = r.reindex(ev_window_idx).dropna()
            ev_mkt = sector_return.reindex(ev_window_idx).dropna()
            common_ev = ev_r.index.intersection(ev_mkt.index)
            if common_ev.empty:
                continue

            abnormal = ev_r.loc[common_ev] - (alpha + beta * ev_mkt.loc[common_ev])
            car = abnormal.sum()

            # t-stat using estimation-period residual std
            resid = est_r.loc[common_est] - (alpha + beta * est_mkt.loc[common_est])
            resid_std = resid.std()
            t_stat = car / (resid_std * np.sqrt(len(common_ev))) if resid_std > 0 else np.nan

            rows.append({
                "event_name": ev["name"],
                "event_type": ev["type"],
                "ticker": ticker,
                "car": car,
                "t_stat": t_stat,
                "n_days": len(common_ev),
            })

    return pd.DataFrame(rows)


# ── Rolling stress beta ───────────────────────────────────────────────────────

def compute_stress_betas(
    returns: pd.DataFrame,
    gsi_by_iso: dict[str, pd.DataFrame],
    ticker_iso_map: dict[str, str],
    sector_return: pd.Series,
    lookback_days: int = 252,
    stress_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Estimate each ticker's "stress beta" — its conditional sensitivity
    to the Grid Stress Index on high-stress days.

    Method:
      excess_return_t = alpha + stress_beta * GSI_t + epsilon_t
      (regression run only on days where GSI >= stress_threshold)

    Parameters
    ----------
    returns          : daily log-return DataFrame
    gsi_by_iso       : {iso: gsi_df} from grid_stress_index.build_multi_iso_gsi()
    ticker_iso_map   : {ticker: iso}
    sector_return    : equal-weighted sector return
    lookback_days    : rolling window length
    stress_threshold : GSI cutoff for "high-stress" day inclusion

    Returns
    -------
    DataFrame indexed by ticker with columns:
        stress_beta, alpha, r_squared, n_obs, mean_gsi_on_stress_days
    """
    results = []

    for ticker in returns.columns:
        iso = ticker_iso_map.get(ticker)
        if iso is None or iso not in gsi_by_iso:
            continue

        gsi_series = gsi_by_iso[iso]["gsi"]
        excess_ret = (returns[ticker] - sector_return).dropna()

        # Align on common dates
        common = excess_ret.index.intersection(gsi_series.index)
        if len(common) < MIN_STRESS_OBS:
            continue

        er  = excess_ret.loc[common]
        gsi = gsi_series.loc[common]

        # Filter to high-stress observations
        stress_mask = gsi >= stress_threshold
        er_stress   = er[stress_mask]
        gsi_stress  = gsi[stress_mask]

        if len(er_stress) < MIN_STRESS_OBS:
            continue

        # OLS: excess_return ~ GSI
        if _HAS_SM:
            X = sm.add_constant(gsi_stress.values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.OLS(er_stress.values, X).fit()
            stress_beta = model.params[1]
            alpha       = model.params[0]
            r_sq        = model.rsquared
        else:
            stress_beta, alpha = np.polyfit(gsi_stress.values, er_stress.values, 1)
            ss_res = np.sum((er_stress.values - (alpha + stress_beta * gsi_stress.values)) ** 2)
            ss_tot = np.sum((er_stress.values - er_stress.mean()) ** 2)
            r_sq   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results.append({
            "ticker":                  ticker,
            "iso":                     iso,
            "stress_beta":             stress_beta,
            "alpha":                   alpha,
            "r_squared":               r_sq,
            "n_obs":                   len(er_stress),
            "mean_gsi_on_stress_days": gsi_stress.mean(),
        })

    return pd.DataFrame(results).set_index("ticker") if results else pd.DataFrame()


def rolling_stress_betas(
    returns: pd.DataFrame,
    gsi_by_iso: dict[str, pd.DataFrame],
    ticker_iso_map: dict[str, str],
    sector_return: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    lookback_days: int = 252,
    stress_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute stress betas at each rebalance date using a trailing window.
    Returns a DataFrame with MultiIndex (date, ticker) and a 'stress_beta' column,
    suitable for time-varying factor construction.
    """
    frames = []
    for date in rebalance_dates:
        start = date - pd.Timedelta(days=int(lookback_days * 1.5))
        window_returns = returns.loc[start:date]
        window_sector  = sector_return.loc[start:date]
        window_gsi     = {iso: df.loc[start:date] for iso, df in gsi_by_iso.items()}

        betas = compute_stress_betas(
            returns          = window_returns,
            gsi_by_iso       = window_gsi,
            ticker_iso_map   = ticker_iso_map,
            sector_return    = window_sector,
            stress_threshold = stress_threshold,
        )
        if not betas.empty:
            betas["date"] = date
            frames.append(betas.reset_index())

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result.set_index(["date", "ticker"])
