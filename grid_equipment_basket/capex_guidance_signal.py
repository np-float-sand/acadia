"""Utility capex-guidance revision signal ("Deliverable D").

Spec: docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md

Aggregate US electric utilities revise their forward 5-yr capital-program
guidance at each earnings call / 10-K / analyst day; since ~2023 those
revisions are increasingly data-center-driven and increasingly stated as
such. This module turns the hand/web-assembled panel in
`grid_resilience.data.utility_capex_guidance` into (1) a daily z-scored
composite, (2) a one-directional de-risk multiplier and a two-sided scaler
for the grid-equipment basket's exposure, and (3) the pre-registered gate
report testing both as a timing signal and as an exposure scaler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.capex_signal import fetch_bigfour_capex
from grid_equipment_basket.ftr_signal import broadcast_daily
from grid_equipment_basket.grid_regime import _trailing_zscore


def guidance_composite(events: pd.Series, start: str, end: str, *,
                       zscore_window: int = config.DC_GUIDANCE_ZSCORE_WINDOW,
                       zscore_minp: int = config.DC_GUIDANCE_ZSCORE_MINP,
                       winsor: float = config.DC_GUIDANCE_ZSCORE_WINSOR) -> pd.Series:
    """`events` indexed by `report_date` (already point-in-time -- see
    `utility_capex_guidance.aggregate_revision_series`). Forward-filled onto a
    daily calendar index over `[start, end]`, then trailing-z-scored
    (`grid_regime._trailing_zscore`'s rolling window only uses
    past-and-current observations, so this composite carries the same
    no-future-leak contract as `ftr_signal.ftr_composite` /
    `grid_regime.regime_composite`)."""
    if events.empty:
        # DatetimeIndex, not the default RangeIndex: this empty series flows
        # straight into `timing_report`'s `.resample("ME")` and into
        # `.loc[start:end]` string slices, both of which raise on a RangeIndex.
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name="guidance_composite")
    daily = broadcast_daily(events, start, end)
    return _trailing_zscore(daily, zscore_window, zscore_minp, winsor).rename("guidance_composite")


def guidance_derisk_multiplier(composite: pd.Series, *,
                               floor_z: float = config.DC_GUIDANCE_DERISK_FLOOR_Z,
                               lo_mult: float = config.DC_GUIDANCE_DERISK_LO_MULT) -> pd.Series:
    """`{lo_mult, 1.0}` -- `lo_mult` on any day the composite z-score is below
    `floor_z` (revision-flow decelerating/reversing), back to 1.0 once it
    clears. NaN days (not-yet-warm) default to 1.0. Never exceeds 1.0. Same
    one-directional shape as `capex_signal.capex_derisk_multiplier` (spec s6.1)."""
    lo = composite < floor_z
    return pd.Series(np.where(lo.fillna(False), lo_mult, 1.0),
                     index=composite.index).rename("guidance_derisk_mult")


def guidance_scaler(composite: pd.Series, *,
                    k: float = config.DC_GUIDANCE_SCALER_K,
                    lo: float = config.DC_GUIDANCE_SCALER_LO,
                    hi: float = config.DC_GUIDANCE_SCALER_HI) -> pd.Series:
    """`clip(1 + k*z, lo, hi)` -- two-sided: leans in when the composite is
    positive (revision-flow accelerating), leans out when negative. NaN days
    default to 1.0. Same shape as `grid_regime`'s continuous-mode multiplier
    (spec s6.2)."""
    z = composite.fillna(0.0)
    raw = (1.0 + k * z).clip(lower=lo, upper=hi)
    return raw.where(composite.notna(), 1.0).rename("guidance_scaler")


def _hac_ols(y: pd.Series, X: pd.DataFrame, *, lag: int) -> dict:
    """OLS of `y` on `X` (a constant is added automatically) with Newey-West
    HAC standard errors at `lag`. Used both for the rank-IC t-stat (on ranked
    series) and the multi-control regression (on raw series) -- spec s5.1/s5.2.
    Returns `{"coef": {col: value}, "t": {col: value}, "n": n_obs}`; a NaN dict
    when there aren't enough observations to fit, rather than raising."""
    import statsmodels.api as sm

    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    cols = list(X.columns)
    if len(frame) < len(cols) + 3:
        return {"coef": {c: np.nan for c in cols}, "t": {c: np.nan for c in cols}, "n": len(frame)}
    yy = frame["__y__"]
    XX = sm.add_constant(frame[cols])
    fit = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": max(int(lag), 1)})
    return {"coef": {c: float(fit.params[c]) for c in cols},
           "t": {c: float(fit.tvalues[c]) for c in cols}, "n": int(len(frame))}


def _rank_ic(signal: pd.Series, fwd_return: pd.Series, *, lag: int) -> dict:
    """Spearman rank-IC (point estimate via `scipy.stats.spearmanr`) plus a
    serial-correlation-robust t-stat: `_hac_ols` of `rank(fwd_return)` on
    `rank(signal)` with Newey-West lag=`lag` (the overlapping-forward-window
    horizons h=3/6 have serially correlated residuals month to month, which a
    plain Spearman significance test would understate). Spec s5.1."""
    from scipy.stats import spearmanr

    pair = pd.concat([signal.rename("s"), fwd_return.rename("r")], axis=1).dropna()
    if len(pair) < 5:
        return {"ic": np.nan, "t": np.nan, "n": len(pair)}
    ic, _ = spearmanr(pair["s"], pair["r"])
    ranks = pair.rank()
    hac = _hac_ols(ranks["r"], ranks[["s"]].rename(columns={"s": "signal"}), lag=lag)
    return {"ic": float(ic), "t": hac["t"]["signal"], "n": len(pair)}


def _monthly_nav(daily_ret: pd.Series) -> pd.Series:
    """Month-end NAV level (base 1.0) from a daily simple-return series."""
    nav = (1.0 + daily_ret.fillna(0.0)).cumprod()
    return nav.resample("ME").last()


def _forward_return(monthly_nav: pd.Series, h: int) -> pd.Series:
    """At each month-end, the realized return over the NEXT `h` months
    (`nav[t+h] / nav[t] - 1`); NaN for the trailing `h` month-ends where the
    future NAV isn't known yet."""
    return (monthly_nav.shift(-h) / monthly_nav - 1.0).rename(f"fwd_{h}m")


_HORIZONS: tuple[int, ...] = (1, 3, 6)


def feasibility_gate(panel_df: pd.DataFrame,
                     primary_window: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW
                     ) -> tuple[pd.DataFrame | None, dict, str]:
    """Spec s3.3: <8 usable utilities in `primary_window` -> fall back to
    `config.DC_GUIDANCE_FALLBACK`; the fallback itself unusable (0 usable) ->
    `(None, feas, "not_testable")` so the caller stops before backtesting on
    data too thin to trust. Returns `(panel_to_use, feasibility_dict,
    universe_used)` with `universe_used` one of `"full"/"fallback"/"not_testable"`."""
    from grid_resilience.data import utility_capex_guidance as udg

    feas = udg.feasibility_summary(panel_df, window=primary_window)
    if feas["n_usable"] >= config.DC_GUIDANCE_MIN_UTILITIES:
        return panel_df, feas, "full"

    fallback = panel_df[panel_df["utility"].isin(config.DC_GUIDANCE_FALLBACK)]
    feas_fb = udg.feasibility_summary(fallback, window=primary_window)
    if feas_fb["n_usable"] == 0:
        return None, feas_fb, "not_testable"
    return fallback, feas_fb, "fallback"


def timing_report(composites: dict[str, pd.Series], basket_ret: pd.Series,
                  controls: pd.DataFrame, *,
                  primary: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW,
                  holdout: tuple[str, str] = config.DC_GUIDANCE_HOLDOUT_WINDOW,
                  horizons: tuple[int, ...] = _HORIZONS) -> dict:
    """Spec s5: for each named composite series and each forward horizon, the
    rank-IC (primary + holdout windows, s5.1) and the with/without-hyperscaler
    control regression (primary window, s5.2), plus the combined pass/fail
    (s5.4: BOTH the primary-window rank-IC and the without-hyperscaler control
    must clear `|t| >= config.DC_GUIDANCE_RANK_IC_MIN_T`).

    `controls` is a monthly-indexed DataFrame with columns `d10y`, `smh`, and
    optionally `bigfour`; the without-hyperscaler regression uses `d10y` +
    `smh` only, the with-hyperscaler regression adds `bigfour` when present."""
    basket_nav = _monthly_nav(basket_ret)
    out: dict = {}
    for name, comp in composites.items():
        comp_monthly = comp.resample("ME").last()
        out[name] = {}
        for h in horizons:
            fwd = _forward_return(basket_nav, h)
            rank_ic_primary = _rank_ic(comp_monthly.loc[primary[0]:primary[1]],
                                       fwd.loc[primary[0]:primary[1]], lag=h)
            rank_ic_holdout = _rank_ic(comp_monthly.loc[holdout[0]:holdout[1]],
                                       fwd.loc[holdout[0]:holdout[1]], lag=h)

            X = pd.DataFrame({"signal": comp_monthly, "d10y": controls.get("d10y"),
                              "smh": controls.get("smh")}).loc[primary[0]:primary[1]]
            fwd_primary = fwd.loc[primary[0]:primary[1]]
            ctrl_wo = _hac_ols(fwd_primary, X, lag=h)
            if "bigfour" in controls.columns:
                X_w = X.assign(bigfour=controls["bigfour"].loc[primary[0]:primary[1]])
                ctrl_w = _hac_ols(fwd_primary, X_w, lag=h)
            else:
                ctrl_w = {"coef": {}, "t": {}, "n": 0}

            sig_t = ctrl_wo["t"].get("signal", np.nan)
            passed = (abs(rank_ic_primary["t"]) >= config.DC_GUIDANCE_RANK_IC_MIN_T
                     and abs(sig_t) >= config.DC_GUIDANCE_RANK_IC_MIN_T)
            out[name][h] = {
                "rank_ic_primary": rank_ic_primary, "rank_ic_holdout": rank_ic_holdout,
                "control_without_hyperscaler": ctrl_wo, "control_with_hyperscaler": ctrl_w,
                "passed": bool(passed),
            }
    return out


def derisk_scaler_report(ret: pd.Series, lvl: pd.Series, composite: pd.Series, *,
                         primary: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW,
                         prior: tuple[str, str] = config.DC_GUIDANCE_PRIOR_WINDOW) -> dict:
    """Spec s6: both the one-directional de-risk (s6.1) and the two-sided
    scaler (s6.2), each on its parameter plateau, scored against the same
    gate `grid_regime.gate_check`/`final_verdict` already uses -- "beats
    layer-1 (price trend gate + vol target) on Sharpe AND Calmar, both
    windows, non-marginal, survives the parameter-neighbour plateau probe".

    Every grid row also carries `active_days_primary` / `active_days_prior`:
    the number of days in that window where the multiplier differs from 1.0.
    A row with 0 active days in a window was NOT EXERCISED there -- its block
    is mechanically identical to the `vol_target_only` baseline (which is why
    that baseline is returned alongside `buy_and_hold` and `layer1_only`), and
    its gate result says nothing about the signal. The one-directional de-risk
    leg in particular is silent whenever the composite never drops below any
    `floor_z` in the grid, so these counts must be read before any gate
    number in this block is interpreted as evidence."""
    from grid_equipment_basket import overlay
    from grid_equipment_basket.backtest import calendar_year_returns, compute_metrics
    from grid_equipment_basket.grid_regime import final_verdict, gate_check

    rf, af = config.RISK_FREE_RATE, config.ANN_FACTOR
    windows = {"primary": primary, "prior": prior}

    def _block(series: pd.Series) -> dict:
        return {wk: {"metrics": compute_metrics(series.loc[a:b].dropna(), rf, af),
                    "calendar": calendar_year_returns(series.loc[a:b].dropna())}
               for wk, (a, b) in windows.items()}

    bh = _block(ret)
    # trend-gate off (ma_days=10**9 -> the MA is never warm -> gate == 1.0),
    # vol-target on: the same `vt_only` construction grid_regime/ftr_signal use.
    # A multiplier that is a constant 1.0 over a window reproduces this exactly,
    # so it is the reference that says "this leg did nothing here".
    vt_only = _block(overlay.apply_overlay(ret, lvl, rf, ma_days=10 ** 9))
    l1 = _block(overlay.apply_overlay(ret, lvl, rf))

    def _score(mult: pd.Series) -> dict:
        return _block(overlay.apply_overlay_l2(ret, mult, rf))

    def _active(mult: pd.Series) -> dict:
        return {f"active_days_{wk}": int((mult.loc[a:b] != 1.0).sum())
                for wk, (a, b) in windows.items()}

    def _row(params: dict, mult: pd.Series) -> dict:
        return {**params, "block": _score(mult), **_active(mult)}

    derisk_grid = [_row({"floor_z": fz, "lo_mult": lm},
                        guidance_derisk_multiplier(composite, floor_z=fz, lo_mult=lm))
                   for fz in config.DC_GUIDANCE_DERISK_GRID_FLOOR
                   for lm in config.DC_GUIDANCE_DERISK_GRID_LO]
    scaler_grid = [_row({"k": k, "hi": hi},
                        guidance_scaler(composite, k=k, hi=hi, lo=config.DC_GUIDANCE_SCALER_LO))
                   for k in config.DC_GUIDANCE_SCALER_GRID_K
                   for hi in config.DC_GUIDANCE_SCALER_GRID_HI]

    def _central(grid: list[dict], key: dict) -> dict:
        for row in grid:
            if all(row[k] == v for k, v in key.items()):
                return row
        return grid[0]

    derisk_central = _central(derisk_grid, {"floor_z": config.DC_GUIDANCE_DERISK_FLOOR_Z,
                                            "lo_mult": config.DC_GUIDANCE_DERISK_LO_MULT})
    scaler_central = _central(scaler_grid, {"k": config.DC_GUIDANCE_SCALER_K,
                                            "hi": config.DC_GUIDANCE_SCALER_HI})

    def _neighbour_passes(central: dict, grid: list[dict]) -> list[bool]:
        out = []
        for row in grid:
            if row is central:
                continue
            g = gate_check(row["block"]["primary"], row["block"]["prior"],
                           l1["primary"], l1["prior"], bh["primary"], bh["prior"])
            out.append(bool(g["G1"] and g["G2"]))
        return out or [True]

    def _gate_for(central: dict, grid: list[dict]) -> dict:
        gate = gate_check(central["block"]["primary"], central["block"]["prior"],
                          l1["primary"], l1["prior"], bh["primary"], bh["prior"])
        verdict = final_verdict(gate, _neighbour_passes(central, grid))
        return {"gate": gate, "verdict": verdict}

    return {
        "baselines": {"buy_and_hold": bh, "vol_target_only": vt_only, "layer1_only": l1},
        "derisk": {"grid": derisk_grid, "central": derisk_central,
                  **_gate_for(derisk_central, derisk_grid)},
        "scaler": {"grid": scaler_grid, "central": scaler_central,
                  **_gate_for(scaler_central, scaler_grid)},
    }


def lead_lag_table(signal_monthly: pd.Series, fwd_return_monthly: pd.Series,
                   ks: range = range(-6, 7)) -> pd.Series:
    """Plain Pearson correlation between `signal_monthly` and
    `fwd_return_monthly` shifted by each offset `k` in `ks` (negative k =
    signal leads; positive k = basket leads) -- spec s5.3's continuity table,
    same shape as the VA-transmission-probe's and Table-B9's lead-lag tables,
    printed alongside the formal rank-IC/HAC test so a reader can see at a
    glance whether this result looks different from those two negatives."""
    out = {}
    for k in ks:
        pair = pd.concat([signal_monthly.rename("s"),
                          fwd_return_monthly.shift(k).rename("r")], axis=1).dropna()
        out[k] = float(pair["s"].corr(pair["r"])) if len(pair) >= 3 else float("nan")
    return pd.Series(out).rename("lead_lag_corr")


def _default_price_fn(tickers, start, end):
    from grid_equipment_basket.data.prices import fetch_prices
    return fetch_prices(tickers, start, end)


def _empty_daily() -> pd.Series:
    """An empty float Series carrying an empty *DatetimeIndex*. Used for every
    "this control isn't available" fallback: a default RangeIndex would make
    the assembled `controls` frame fall back to an object index, and the
    downstream `.loc["2023-01-01":"2026-08-31"]` slice would then raise
    `TypeError: '<' not supported between instances of 'Timestamp' and 'str'`."""
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


_SERIES_SPECS: tuple[tuple[str, str, str | None], ...] = (
    ("all_usd",       "revision_vs_prior_usd_m",   None),
    ("all_pct",       "revision_vs_prior_usd_m",   "prior_capex_plan_usd_m"),
    ("dc_stated_usd", "dc_attributed_usd_m",        None),
    ("dc_stated_pct", "dc_attributed_usd_m",        "prior_capex_plan_usd_m"),
    ("dc_filled_usd", "dc_attributed_usd_m_filled",  None),
    ("dc_filled_pct", "dc_attributed_usd_m_filled",  "prior_capex_plan_usd_m"),
)


def guidance_signal_report(price_fn=None, panel_df=None, bigfour_fn=None,
                           primary: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW,
                           prior: tuple[str, str] = config.DC_GUIDANCE_PRIOR_WINDOW,
                           holdout: tuple[str, str] = config.DC_GUIDANCE_HOLDOUT_WINDOW) -> dict:
    """Orchestrates the full spec: feasibility kill (s3.3) -> six series
    (s4.1/s4.2) -> daily composites (s4.3) -> timing bar (s5, basket AND
    long/short-spread) -> de-risk/scaler gate (s6, on the headline `all_usd`
    series). `price_fn(tickers, start, end) -> DataFrame`, `panel_df`, and
    `bigfour_fn() -> DataFrame` are injectable for tests; defaults are the
    live yfinance fetcher, `utility_capex_guidance.load_capex_guidance()`, and
    `capex_signal.fetch_bigfour_capex`. A non-empty `bigfour_fn()` result must
    carry both `decel2` and `known_date` (the point-in-time stamp the control
    is aligned on -- see below); an empty frame is handled as "control not
    available" rather than an error, so a cold `bigfour_capex.parquet` cache or
    a failed SEC/XBRL fetch degrades the with-hyperscaler regression to n=0
    instead of killing the whole report."""
    from grid_equipment_basket.basket import simulate_basket
    from grid_resilience.data import utility_capex_guidance as udg

    price_fn = price_fn or _default_price_fn
    bigfour_fn = bigfour_fn or fetch_bigfour_capex
    panel_df = panel_df if panel_df is not None else udg.load_capex_guidance()

    used_panel, feas, universe_used = feasibility_gate(panel_df, primary)
    if universe_used == "not_testable":
        return {"universe_used": universe_used, "feasibility": feas, "verdict": "not_yet_testable"}

    panel_filled = udg.impute_dc_attributed(used_panel)

    series = {}
    for name, value_col, denom_col in _SERIES_SPECS:
        src = panel_filled if value_col.endswith("_filled") else used_panel
        series[name] = udg.aggregate_revision_series(
            src, value_col=value_col, denom_col=denom_col,
            ttm_quarters=config.DC_GUIDANCE_TTM_QUARTERS)

    span_start, span_end = prior[0], primary[1]
    composites = {name: guidance_composite(ev, span_start, span_end) for name, ev in series.items()}

    tickers = sorted(set(config.UNIVERSE + config.DER_SHORT_SLEEVE + ["SMH", "^TNX"]))
    prices = price_fn(tickers, span_start, span_end)
    basket_cols = [t for t in config.UNIVERSE if t in prices.columns]
    der_cols = [t for t in config.DER_SHORT_SLEEVE if t in prices.columns]

    basket = simulate_basket(prices[basket_cols], span_start, span_end,
                             config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, None)
    ret, lvl = basket.returns, (1.0 + basket.returns).cumprod()

    der_ret = (prices[der_cols].pct_change().dropna(how="all").mean(axis=1)
              if der_cols else _empty_daily())
    spread_ret = (ret - der_ret.reindex(ret.index)).dropna()

    # Every "control not available" fallback below goes through `_empty_daily`
    # (see its docstring): without a DatetimeIndex the whole report dies on any
    # machine with a cold big-four cache or a partially failed price fetch.
    tnx = prices["^TNX"].dropna() if "^TNX" in prices.columns else _empty_daily()
    d10y_monthly = tnx.resample("ME").last().diff() if not tnx.empty else _empty_daily()
    smh_monthly = ((1.0 + prices["SMH"].pct_change().dropna()).resample("ME").prod() - 1.0
                  if "SMH" in prices.columns else _empty_daily())
    bigfour = bigfour_fn()
    if not bigfour.empty:
        # Point-in-time: `fetch_bigfour_capex` stamps each quarter with
        # `known_date` (period end + 50d, when the prints are actually public).
        # Indexing on the quarter END instead would regress forward returns on a
        # control that was not yet observable -- the same 50-day look-ahead
        # `capex_signal.capex_derisk_multiplier` already avoids by using
        # `known_date`.
        bigfour_s = pd.Series(bigfour["decel2"].to_numpy(),
                              index=pd.DatetimeIndex(pd.to_datetime(bigfour["known_date"]))
                              ).sort_index()
        bigfour_monthly = bigfour_s.resample("ME").ffill()
    else:
        bigfour_monthly = _empty_daily()

    controls = pd.DataFrame({"d10y": d10y_monthly, "smh": smh_monthly, "bigfour": bigfour_monthly})

    timing_basket = timing_report(composites, ret, controls, primary=primary, holdout=holdout)
    timing_spread = (timing_report(composites, spread_ret, controls, primary=primary, holdout=holdout)
                     if not spread_ret.empty else {})

    basket_nav = _monthly_nav(ret)
    fwd_1m = _forward_return(basket_nav, 1)
    continuity = {name: lead_lag_table(comp.resample("ME").last(), fwd_1m)
                 for name, comp in composites.items()}

    derisk_scaler = derisk_scaler_report(ret, lvl, composites["all_usd"], primary=primary, prior=prior)

    return {
        "universe_used": universe_used, "feasibility": feas,
        "windows": {"primary": primary, "prior": prior, "holdout": holdout},
        "timing_basket": timing_basket, "timing_spread": timing_spread,
        "continuity_lead_lag": continuity,
        "derisk_scaler_gate": derisk_scaler,
    }


def _f(x) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def guidance_table(rep: dict) -> str:
    L = ["CAPEX-GUIDANCE REVISION SIGNAL (Deliverable D)", ""]
    if rep.get("universe_used") == "not_testable":
        L.append("  FEASIBILITY: NOT YET TESTABLE")
        feas = rep["feasibility"]
        L.append(f"  n_usable={feas['n_usable']} / n_total={feas['n_total']}")
        return "\n".join(L)

    L.append(f"  universe_used: {rep['universe_used']}  "
             f"(n_usable={rep['feasibility']['n_usable']}/{rep['feasibility']['n_total']})")
    w = rep["windows"]
    L.append(f"  windows: primary {w['primary']}  prior {w['prior']}  holdout {w['holdout']}")
    L.append("")
    L.append("  TIMING BAR (vs basket) -- rank-IC t | control(w/o hyperscaler) t | PASS?")
    for name, by_h in rep["timing_basket"].items():
        for h, cell in by_h.items():
            ic = cell["rank_ic_primary"]
            ctrl = cell["control_without_hyperscaler"]
            L.append(f"    {name:<16} h={h}m  ic={_f(ic['ic'])} t={_f(ic['t'])}  "
                     f"ctrl_t={_f(ctrl['t'].get('signal', float('nan')))}  "
                     f"{'PASS' if cell['passed'] else 'fail'}")
    L.append("")
    if "all_usd" in rep.get("continuity_lead_lag", {}):
        ll = rep["continuity_lead_lag"]["all_usd"]
        L.append("  CONTINUITY -- lead-lag corr, all_usd vs 1m-fwd basket return "
                 "(negative k = signal leads):")
        L.append("    " + "  ".join(f"k={k}:{_f(v)}" for k, v in ll.items()))
    L.append("")
    for leg in ("derisk", "scaler"):
        g = rep["derisk_scaler_gate"][leg]
        # `active_days(primary)` is printed on the same line as the verdict on
        # purpose: 0 means the multiplier was a flat 1.0 all window, so the
        # gate result is the vol-target-only baseline's, NOT evidence about
        # this signal. Read it before the G1/G2/G3 flags.
        act = g["central"].get("active_days_primary")
        L.append(f"  {leg.upper()} GATE: verdict={g['verdict']}  "
                 f"G1={g['gate']['G1']} G2={g['gate']['G2']} G3={g['gate']['G3']} "
                 f"marginal={g['gate']['marginal']}  "
                 f"active_days(primary)={'n/a' if act is None else act}"
                 + ("  <- NOT EXERCISED (multiplier flat 1.0)" if act == 0 else ""))
    return "\n".join(L)
