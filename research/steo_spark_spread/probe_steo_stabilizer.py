"""
STEO forward-spark-spread stress-beta stabilizer probe (RESEARCH-LOG item 9 / untried-tasks
item 6, redirect target of fix #2 in docs/merchant-beta-orthogonalization-results.md).

Fix #2 orthogonalized VST/NRG's stress beta against a *trailing realized* LMP trend and
FAILED (IC got worse, beta barely moved) -- diagnosed reason: the merchant re-rating is a
forward-looking narrative repricing, and a trailing/realized statistic can't reach it.

This probe tests the theoretically-motivated fix: replace the trailing-LMP regressor with
`research/steo_spark_spread`'s point-in-time forward implied spark spread (STEO's own
ELWHU_PJ/ELWHU_TX forward power forecast minus HEAT_RATE * STEO's own NGHHUUS forward gas
forecast, as vintage-consistent, non-lookahead point-in-time data -- see build_spark_spread.py),
run through the IDENTICAL two-variable-OLS harness fix #2 used
(`compute_stress_betas_orthogonal` from `research/merchant_beta_orthogonalization/probe.py`,
imported unchanged) so the comparison to fixes #1/#2/#4 is apples-to-apples: same production
pipeline, same monkeypatch point, same metrics.

Pre-registered comparison (stated before looking at the result, per the task brief -- reuses
fix #1/#2/#4's own table format exactly):
  - Full Sharpe, full ann. return, full max DD
  - Stress-period Sharpe, calm-period Sharpe
  - IC@21d (t-stat), IC@63d (t-stat)
"better" = IC@21d and/or IC@63d t-stat improves vs baseline (0.0628/1.352, 0.0953/1.889)
AND full-sample Sharpe does not get worse -- the same bar fix #2 was held to.

Throwaway probe. Findings -> docs/steo-spark-spread-results.md, docs/RESEARCH-LOG.md,
docs/untried-tasks-log.md item 6.
"""
from __future__ import annotations

import functools
import pathlib
import sys

import pandas as pd

sys.path.insert(0, ".")

import grid_resilience.signals.conditional_beta as cb
from research.merchant_beta_orthogonalization.probe import (
    compute_stress_betas_orthogonal,
    _capture_wrapper,
)

HERE = pathlib.Path(__file__).parent


def load_steo_trend_by_iso() -> dict[str, pd.Series]:
    out = {}
    for iso in ("PJM", "ERCOT"):
        path = HERE / "data" / f"steo_trend_{iso}.parquet"
        s = pd.read_parquet(path)["steo_spark_trend_z"]
        out[iso] = s
    return out


def main():
    trend_by_iso = load_steo_trend_by_iso()
    for iso, s in trend_by_iso.items():
        print(f"  {iso}: {len(s)} days, {s.index.min().date() if len(s) else None} -> "
              f"{s.index.max().date() if len(s) else None}")

    cb.compute_stress_betas = functools.partial(
        compute_stress_betas_orthogonal, power_trend_by_iso=trend_by_iso
    )
    import grid_resilience.main as main_mod
    main_mod.rolling_stress_betas = _capture_wrapper(main_mod.rolling_stress_betas)

    print("\nRunning full pipeline with STEO forward-spark-spread orthogonalized stress beta...")
    result = main_mod.run(plot=False, save_dir="/tmp/gr_steo_spark_test")
    metrics = result.get("metrics", {})
    ic_metrics = result.get("ic_metrics", {})
    print("\nMETRICS:", metrics)
    print("\nIC_METRICS:", {h: ic_metrics.get(h) for h in (21, 63)})

    print("\n=== Comparison table (same format as fixes #1/#2/#4) ===")
    print(f"{'metric':30s} {'baseline (GSI only)':>22s} {'STEO spark-spread fix':>22s}")
    rows = [
        ("Full Sharpe", "full_sharpe"),
        ("Full ann. return", "full_ann_return"),
        ("Full max DD", "full_max_dd"),
        ("Stress-period Sharpe", "stress_sharpe"),
        ("Calm-period Sharpe", "calm_sharpe"),
    ]
    baseline_metrics = {
        "full_sharpe": -0.007, "full_ann_return": 0.0390, "full_max_dd": -0.314,
        "stress_sharpe": -0.145, "calm_sharpe": 0.379,
    }
    for label, key in rows:
        print(f"{label:30s} {baseline_metrics.get(key, float('nan')):>22} {metrics.get(key, float('nan')):>22}")
    baseline_ic = {21: (0.0628, 1.352), 63: (0.0953, 1.889)}
    for h in (21, 63):
        b_ic, b_t = baseline_ic[h]
        entry = ic_metrics.get(h, {})
        print(f"IC@{h}d (mean/t)               {b_ic}/{b_t:>14} "
              f"{entry.get('mean_ic')}/{entry.get('t_stat')}")

    from research.merchant_beta_orthogonalization.probe import _CAPTURED_ROLLING_BETAS
    if _CAPTURED_ROLLING_BETAS:
        rolling_betas = _CAPTURED_ROLLING_BETAS[-1]
        rolling_betas.to_parquet(HERE / "data" / "rolling_betas_steo_spark.parquet")
        print(f"Saved rolling betas: {rolling_betas.shape}")


if __name__ == "__main__":
    main()
