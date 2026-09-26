"""
Merchant stress-beta orthogonalization probe (untried-tasks-log item 3, fix #5).

Diagnosis being tested (docs/grid-stress-signal-evidence.md §5, docs/merchant-beta-
orthogonalization-results.md): the rolling single-ISO stress beta for VST/NRG/CEG/TLN
whipsaws because (a) it regresses each name against ONLY its single assigned ISO's GSI
(ticker_iso_map: VST/NRG -> ERCOT, CEG/TLN -> PJM) even though all four are disclosed,
multi-ISO businesses, and (b) it treats 100% of the name's equity as merchant/spot-
exposed regardless of how much of next year's generation is actually hedged/contracted.
Fix #5 replaces the regressor and the final beta scale with numbers hand-extracted from
each company's own 10-Ks (fundamentals.py), updated only when a new filing lands
(annual step function keyed to the actual filing date -- no look-ahead).

Two variants, tested separately so a null result on one doesn't get laundered by the
other (same discipline as fixes #1/#2/#4, each of which isolated one mechanism):

  Variant A ("iso_blend"): replace the ticker's single-ISO GSI regressor with
      blended_gsi = w_ERCOT * GSI_ERCOT + w_PJM * GSI_PJM  (weights renormalized to sum
      to 1 over the two ISOs this pipeline's GSI actually covers; the disclosed "OTHER"
      region fraction -- CAISO/NYISO/ISO-NE/WECC -- is unmodeled in both baseline and
      this variant, so the comparison is like-for-like). No change to the beta's scale.

  Variant B ("iso_blend_shrink"): Variant A's regression, then the resulting slope is
      multiplied by the ticker-year's disclosed spot-exposure fraction (1 - hedged %),
      so a name entering a year >90% hedged (VST in 2024) contributes a proportionally
      smaller beta to the factor than a name with a lower hedge ratio.

Both variants fall back to the ORIGINAL single-ISO, unshrunk `compute_stress_betas` for
any (ticker, date) pair that predates that ticker's first available 10-K filing date --
using fundamentals data that did not yet exist at that point in time would be look-ahead
bias. This matters most for TLN, whose first post-restructuring 10-K (FY2024) wasn't
filed until 2025-02-28 -- roughly the first 20 months of its re-listed trading history
(2023-06-02 onward) have NO usable annual fundamentals disclosure at all.

Method: monkeypatch `conditional_beta.compute_stress_betas` exactly as fix #2's probe
did, run the real, unmodified `grid_resilience.main.run()` end to end, and compare the
same metrics fixes #1/#2/#4 reported: full/stress/calm Sharpe, full ann return, full max
DD, IC@21d/63d t-stat, plus the four merchant names' rolling-beta stability vs. baseline.

Pre-registered gate (identical to fix #2's, for direct comparability):
  (a) IC@21d t-stat improves vs. the single-ISO baseline.
  (b) full-sample Sharpe does not get worse, AND the merchant names' rolling beta shows
      measurably fewer/smaller sign flips across the four documented narrative-flip
      windows (late-2023, 2024-05, 2025-01 DeepSeek, 2025-05).
Anything less -> FAIL, log per the fix #1/#2/#4 pattern.

Throwaway. Re-run to regenerate. Findings -> docs/merchant-fundamental-beta-results.md,
docs/merchant-beta-orthogonalization-results.md, docs/untried-tasks-log.md item 3,
docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import functools
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import grid_resilience.signals.conditional_beta as cb
from grid_resilience.config import MIN_STRESS_OBS

sys.path.insert(0, "research/merchant_fundamental_beta")
from fundamentals import FUNDAMENTALS, PASS_THROUGH_FALLBACK  # noqa: E402

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

MERCHANT_TICKERS = ["VST", "NRG", "CEG", "TLN"]

_ORIGINAL_compute_stress_betas = cb.compute_stress_betas


# ── Point-in-time fundamentals lookup (no look-ahead) ──────────────────────────────

def _record_asof(ticker: str, asof_date: pd.Timestamp) -> dict | None:
    """Most recent fundamentals record whose FILING DATE is <= asof_date, or None if
    the ticker had no filed 10-K with usable fundamentals data yet at that date."""
    records = FUNDAMENTALS.get(ticker, {})
    candidates = sorted(
        records.values(), key=lambda rec: pd.Timestamp(rec["filed"])
    )
    valid = [rec for rec in candidates if pd.Timestamp(rec["filed"]) <= asof_date]
    return valid[-1] if valid else None


def _blended_weights(rec: dict) -> tuple[float, float]:
    """Renormalize {ERCOT, PJM_PROXY, OTHER} to (w_ercot, w_pjm) summing to 1 over the
    two ISOs this pipeline's GSI covers -- OTHER is unmodeled, exactly like the
    baseline's 100%-on-one-ISO treatment."""
    w = rec["iso_weights"]
    e, p = w.get("ERCOT", 0.0), w.get("PJM_PROXY", 0.0)
    total = e + p
    if total <= 0:
        return 0.5, 0.5
    return e / total, p / total


def _pjm_series_for(ticker: str, gsi_by_iso: dict) -> pd.Series:
    """Prefer the ticker's own PJM zone-level GSI (already built by main.py's zone_gsi
    step for PJM-primary tickers) over the PJM system-hub series."""
    key = f"PJM:{ticker}"
    if key in gsi_by_iso and not gsi_by_iso[key].empty:
        return gsi_by_iso[key]["gsi"]
    if "PJM" in gsi_by_iso and not gsi_by_iso["PJM"].empty:
        return gsi_by_iso["PJM"]["gsi"]
    return pd.Series(dtype=float)


def _ercot_series_for(gsi_by_iso: dict) -> pd.Series:
    if "ERCOT" in gsi_by_iso and not gsi_by_iso["ERCOT"].empty:
        return gsi_by_iso["ERCOT"]["gsi"]
    return pd.Series(dtype=float)


def _fit_ols(er_stress: pd.Series, gsi_stress: pd.Series) -> tuple[float, float, float]:
    if _HAS_SM:
        X = sm.add_constant(gsi_stress.values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(er_stress.values, X).fit()
        return model.params[1], model.params[0], model.rsquared
    beta, alpha = np.polyfit(gsi_stress.values, er_stress.values, 1)
    ss_res = np.sum((er_stress.values - (alpha + beta * gsi_stress.values)) ** 2)
    ss_tot = np.sum((er_stress.values - er_stress.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, alpha, r_sq


def _compute_one_fundamental_beta(
    ticker: str, returns: pd.DataFrame, gsi_by_iso: dict, sector_return: pd.Series,
    stress_threshold: float, asof_date: pd.Timestamp, shrink: bool,
) -> dict | None:
    rec = _record_asof(ticker, asof_date)
    if rec is None:
        return None  # no fundamentals yet at this date -- caller falls back to baseline

    w_ercot, w_pjm = _blended_weights(rec)
    ercot_s = _ercot_series_for(gsi_by_iso)
    pjm_s = _pjm_series_for(ticker, gsi_by_iso)
    if ercot_s.empty and pjm_s.empty:
        return None
    idx = ercot_s.index.union(pjm_s.index)
    blended = (w_ercot * ercot_s.reindex(idx).fillna(0.0)
               + w_pjm * pjm_s.reindex(idx).fillna(0.0))
    blended = blended.rename("blended_gsi")

    if ticker not in returns.columns:
        return None
    excess_ret = (returns[ticker] - sector_return).dropna()
    common = excess_ret.index.intersection(blended.index)
    if len(common) < MIN_STRESS_OBS:
        return None
    er, gsi = excess_ret.loc[common], blended.loc[common]
    stress_mask = gsi >= stress_threshold
    er_stress, gsi_stress = er[stress_mask], gsi[stress_mask]
    if len(er_stress) < MIN_STRESS_OBS:
        return None
    gsi_stress = gsi_stress.clip(upper=gsi_stress.quantile(0.95))

    slope, alpha, r_sq = _fit_ols(er_stress, gsi_stress)

    spot_pct = rec.get("spot_exposure_pct")
    if spot_pct is None:
        spot_pct = PASS_THROUGH_FALLBACK.get(ticker, 1.0)
    final_beta = slope * spot_pct if shrink else slope

    return {
        "ticker": ticker, "iso": "MULTI", "stress_beta": final_beta, "alpha": alpha,
        "r_squared": r_sq, "n_obs": len(er_stress),
        "mean_gsi_on_stress_days": gsi_stress.mean(),
        "raw_slope": slope, "w_ercot": w_ercot, "w_pjm": w_pjm, "spot_pct": spot_pct,
    }


def make_fundamental_compute_stress_betas(shrink: bool):
    def compute_stress_betas_fundamental(
        returns: pd.DataFrame, gsi_by_iso: dict, ticker_iso_map: dict,
        sector_return: pd.Series, lookback_days: int = 252, stress_threshold: float = 0.5,
    ) -> pd.DataFrame:
        baseline = _ORIGINAL_compute_stress_betas(
            returns=returns, gsi_by_iso=gsi_by_iso, ticker_iso_map=ticker_iso_map,
            sector_return=sector_return, lookback_days=lookback_days,
            stress_threshold=stress_threshold,
        )
        if returns.empty:
            return baseline
        asof_date = returns.index.max()
        fundamental_rows = []
        for ticker in MERCHANT_TICKERS:
            result = _compute_one_fundamental_beta(
                ticker, returns, gsi_by_iso, sector_return, stress_threshold,
                asof_date, shrink,
            )
            if result is not None:
                fundamental_rows.append(result)
        if not fundamental_rows:
            return baseline  # no ticker had usable fundamentals yet at this date
        fundamental_df = pd.DataFrame(fundamental_rows).set_index("ticker")
        if baseline.empty:
            return fundamental_df
        # Keep baseline rows for every ticker NOT replaced (all 21 regulated names,
        # plus any merchant ticker predating its first 10-K filing date), then swap
        # in the fundamentals-derived rows for the rest.
        keep = [t for t in baseline.index if t not in fundamental_df.index]
        return pd.concat([baseline.loc[keep], fundamental_df])

    return compute_stress_betas_fundamental


import collections
_CAPTURED: dict[str, list[pd.DataFrame]] = collections.defaultdict(list)


def _capture_wrapper(original_fn, tag):
    @functools.wraps(original_fn)
    def wrapped(*args, **kwargs):
        out = original_fn(*args, **kwargs)
        _CAPTURED[tag].append(out)
        return out
    return wrapped


def run_variant(tag: str, shrink: bool, save_dir: str, start: str, end: str):
    import grid_resilience.main as main_mod
    import grid_resilience.signals.conditional_beta as cb_mod

    cb_mod.compute_stress_betas = make_fundamental_compute_stress_betas(shrink=shrink)
    # main.py imports `rolling_stress_betas` by name; its body resolves
    # `compute_stress_betas` from conditional_beta's module globals at call time (same
    # note as fix #2's probe), so patching cb_mod above is sufficient for the betas
    # themselves. Wrap main's bound name too, purely to capture the dated output.
    original_rolling = main_mod.rolling_stress_betas
    main_mod.rolling_stress_betas = _capture_wrapper(original_rolling, tag)

    print(f"\n{'='*70}\nRunning variant {tag} (shrink={shrink}) [{start} -> {end}]\n{'='*70}")
    result = main_mod.run(plot=False, save_dir=save_dir, start=start, end=end)
    print(f"\nMETRICS [{tag}]:", result.get("metrics"))
    print(f"\nIC METRICS [{tag}]:", result.get("ic_metrics"))

    main_mod.rolling_stress_betas = original_rolling  # restore for the next variant

    if _CAPTURED[tag]:
        rolling_betas = _CAPTURED[tag][-1]
        out_path = f"research/merchant_fundamental_beta/rolling_betas_{tag}.parquet"
        rolling_betas.to_parquet(out_path)
        print(f"Saved rolling betas: {rolling_betas.shape} -> {out_path}")

    return result


def run_baseline(save_dir: str, start: str, end: str):
    """Fresh, unmodified baseline run in THIS environment/session -- comparable to
    fix #2/#4's baseline number, but not assumed identical to it (different session,
    different data-fetch timestamp for anything gridstatus/yfinance revises)."""
    import grid_resilience.main as main_mod
    import grid_resilience.signals.conditional_beta as cb_mod

    cb_mod.compute_stress_betas = _ORIGINAL_compute_stress_betas
    original_rolling = main_mod.rolling_stress_betas
    main_mod.rolling_stress_betas = _capture_wrapper(original_rolling, "baseline")
    _CAPTURED["baseline"] = []

    print(f"\n{'='*70}\nRunning BASELINE (unmodified, this session) [{start} -> {end}]\n{'='*70}")
    result = main_mod.run(plot=False, save_dir=save_dir, start=start, end=end)
    print("\nMETRICS [baseline]:", result.get("metrics"))
    print("\nIC METRICS [baseline]:", result.get("ic_metrics"))

    main_mod.rolling_stress_betas = original_rolling
    if _CAPTURED["baseline"]:
        rolling_betas = _CAPTURED["baseline"][-1]
        out_path = "research/merchant_fundamental_beta/rolling_betas_baseline_session.parquet"
        rolling_betas.to_parquet(out_path)
        print(f"Saved rolling betas: {rolling_betas.shape} -> {out_path}")
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    args = p.parse_args()
    start, end = args.start, args.end
    print(f"Backtest window: {start} -> {end}")

    result_base = run_baseline(save_dir="/tmp/gr_fund_baseline", start=start, end=end)
    result_a = run_variant("A_isoblend", shrink=False, save_dir="/tmp/gr_fund_a",
                            start=start, end=end)
    result_b = run_variant("B_isoblend_shrink", shrink=True, save_dir="/tmp/gr_fund_b",
                            start=start, end=end)

    import json
    summary = {
        "baseline": {"metrics": result_base.get("metrics"), "ic_metrics": {
            h: {k: v for k, v in m.items() if k != "stress_dates"}
            for h, m in (result_base.get("ic_metrics") or {}).items() if isinstance(h, int)
        }},
        "A_isoblend": {"metrics": result_a.get("metrics"), "ic_metrics": {
            h: {k: v for k, v in m.items() if k != "stress_dates"}
            for h, m in (result_a.get("ic_metrics") or {}).items() if isinstance(h, int)
        }},
        "B_isoblend_shrink": {"metrics": result_b.get("metrics"), "ic_metrics": {
            h: {k: v for k, v in m.items() if k != "stress_dates"}
            for h, m in (result_b.get("ic_metrics") or {}).items() if isinstance(h, int)
        }},
    }

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    with open("research/merchant_fundamental_beta/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_default)
    print("\nSaved results_summary.json")


if __name__ == "__main__":
    main()
