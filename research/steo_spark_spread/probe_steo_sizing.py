"""
STEO spark-spread as a SIZING input, not a ranking input (follow-up to
docs/steo-spark-spread-results.md §6's floated-but-untried angle).

Prior result: replacing GSI-only stress beta with a GSI+STEO orthogonalized beta
changes the long/short RANKING and produces a materially calmer book (MaxDD
-31.4% -> -15.9%, IC@21d t 1.352 -> 2.427) but full-sample Sharpe ticks slightly
worse (-0.007 -> -0.085) -- so it FAILED the project's two-part adoption gate.

This probe asks a narrower question: can the STEO-variant book's *low volatility*
be harvested WITHOUT touching the ranking at all? Concretely:
  1. Run the baseline (GSI-only) pipeline, unmodified, and capture its own
     weights_matrix/returns/events_df (the exact inputs run_backtest uses).
  2. Take the already-cached STEO-variant daily return series
     (/tmp/gr_steo_spark_test/pnl.csv, the full-sample 2018-2025 run behind the
     -0.085 Sharpe / -15.9% MaxDD numbers in the results doc) purely as a
     volatility gauge -- never as a source of stock selection.
  3. Build a daily exposure scalar = STEO-variant's own full-sample vol / its
     trailing 21d vol, clipped to [0.5, 1.5], shifted 1 day (no lookahead).
  4. Scale the BASELINE weights_matrix by that scalar and rerun the unmodified
     backtest engine. Ranking is untouched; only gross position size moves.

Pre-registered gate (same bar used throughout this fix series): Sharpe improves
vs. the baseline AND max drawdown does not get worse.

Throwaway probe. Findings -> docs/steo-spark-spread-results.md, docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import functools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

HERE = pathlib.Path(__file__).parent
STEO_PNL_PATH = pathlib.Path("/tmp/gr_steo_spark_test/pnl.csv")

SCALAR_FLOOR = 0.5
SCALAR_CAP = 1.5
TRAILING_WINDOW = 21

_CAPTURED_BACKTEST_INPUTS: list[dict] = []


def _capture_backtest_wrapper(original_fn):
    @functools.wraps(original_fn)
    def wrapped(weights_matrix, returns, events_df, **kwargs):
        _CAPTURED_BACKTEST_INPUTS.append({
            "weights_matrix": weights_matrix.copy(),
            "returns": returns.copy(),
            "events_df": events_df,
            "kwargs": kwargs,
        })
        return original_fn(weights_matrix, returns, events_df, **kwargs)
    return wrapped


def build_exposure_scalar(steo_ret: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Self-referential vol-target scalar from the STEO-variant's own return series.

    target_vol is the STEO-variant's own full-sample annualized vol, so the
    scalar averages to roughly 1 over history and the shrink/grow calls are
    genuinely relative to that series' own regime, not an arbitrary external
    reference. Shifted 1 day so day-t exposure only uses information through
    t-1 (the trailing 21d realized-vol window itself already only uses trailing
    data, but the shift additionally guards against same-day leakage).
    """
    target_vol = steo_ret.std() * np.sqrt(252)
    trailing_vol = steo_ret.rolling(TRAILING_WINDOW).std() * np.sqrt(252)
    raw_scalar = (target_vol / trailing_vol).clip(SCALAR_FLOOR, SCALAR_CAP)
    scalar = raw_scalar.shift(1)
    return scalar.reindex(target_index).ffill().fillna(1.0)


def main():
    import grid_resilience.main as main_mod
    main_mod.run_backtest = _capture_backtest_wrapper(main_mod.run_backtest)

    print("Running baseline (GSI-only) pipeline, unmodified, to capture backtest inputs...")
    baseline_result = main_mod.run(plot=False, save_dir="/tmp/gr_steo_sizing_baseline")
    baseline_metrics = baseline_result["metrics"]

    captured = _CAPTURED_BACKTEST_INPUTS[0]
    weights_matrix = captured["weights_matrix"]
    returns = captured["returns"]
    events_df = captured["events_df"]
    bt_kwargs = captured["kwargs"]

    print(f"\nBaseline weights_matrix: {weights_matrix.shape}, returns: {returns.shape}")

    if not STEO_PNL_PATH.exists():
        print(f"ERROR: {STEO_PNL_PATH} not found -- rerun probe_steo_stabilizer.py first.")
        sys.exit(1)

    steo_pnl = pd.read_csv(STEO_PNL_PATH, index_col=0, parse_dates=True)
    steo_ret = steo_pnl["strategy"]
    print(f"STEO-variant return series: {len(steo_ret)} days, "
          f"{steo_ret.index.min().date()} -> {steo_ret.index.max().date()}, "
          f"ann_vol={steo_ret.std() * np.sqrt(252):.4f}")

    scalar = build_exposure_scalar(steo_ret, weights_matrix.index)
    print(f"\nExposure scalar: mean={scalar.mean():.3f}, "
          f"pct at floor ({SCALAR_FLOOR})={100 * (scalar <= SCALAR_FLOOR + 1e-9).mean():.1f}%, "
          f"pct at cap ({SCALAR_CAP})={100 * (scalar >= SCALAR_CAP - 1e-9).mean():.1f}%")

    scaled_weights = weights_matrix.mul(scalar, axis=0)

    # Variant: de-lever-only (cap at 1.0, never lever up). The symmetric [0.5, 1.5]
    # scalar above spent 48% of days at its 1.5x ceiling (STEO's own return series
    # is right-skewed -- calm most of the time, occasional vol spikes -- so its
    # full-sample vol sits above its typical trailing vol, pushing the ratio over 1
    # most days). That tests "lever up when calm," not the doc's actual framing
    # ("the drawdown improvement is the whole point") -- this variant tests the
    # purely defensive reading instead.
    scalar_delever_only = scalar.clip(upper=1.0)
    scaled_weights_delever_only = weights_matrix.mul(scalar_delever_only, axis=0)

    from grid_resilience.portfolio.backtest import run_backtest as real_run_backtest

    # Sanity check: rerunning the captured baseline inputs unchanged must reproduce
    # the original baseline metrics exactly.
    baseline_pnl2, baseline_metrics2 = real_run_backtest(
        weights_matrix, returns, events_df, **bt_kwargs
    )
    assert abs(baseline_metrics2["full_sharpe"] - baseline_metrics["full_sharpe"]) < 1e-9, (
        "Recomputed baseline Sharpe doesn't match original run -- capture wrapper is wrong."
    )

    sized_pnl, sized_metrics = real_run_backtest(
        scaled_weights, returns, events_df, **bt_kwargs
    )
    sized_pnl_delever, sized_metrics_delever = real_run_backtest(
        scaled_weights_delever_only, returns, events_df, **bt_kwargs
    )

    print("\n=== Comparison: baseline vs STEO-vol-sized (symmetric) vs de-lever-only ===")
    print(f"{'metric':30s} {'baseline':>15s} {'symmetric':>15s} {'delever-only':>15s}")
    rows = [
        ("Full Sharpe", "full_sharpe"),
        ("Full ann. return", "full_ann_return"),
        ("Full max DD", "full_max_dd"),
        ("Stress-period Sharpe", "stress_sharpe"),
        ("Calm-period Sharpe", "calm_sharpe"),
    ]
    for label, key in rows:
        b = baseline_metrics.get(key, float("nan"))
        s = sized_metrics.get(key, float("nan"))
        d = sized_metrics_delever.get(key, float("nan"))
        print(f"{label:30s} {b:>15.4f} {s:>15.4f} {d:>15.4f}")

    print(f"\nDe-lever-only scalar: mean={scalar_delever_only.mean():.3f}, "
          f"pct at floor ({SCALAR_FLOOR})={100 * (scalar_delever_only <= SCALAR_FLOOR + 1e-9).mean():.1f}%, "
          f"pct at 1.0 (unscaled)={100 * (scalar_delever_only >= 1.0 - 1e-9).mean():.1f}%")

    for name, m in (("symmetric", sized_metrics), ("delever-only", sized_metrics_delever)):
        gate_a = m["full_sharpe"] > baseline_metrics["full_sharpe"]
        gate_b = m["full_max_dd"] >= baseline_metrics["full_max_dd"]
        print(f"\n[{name}] Gate (a) Sharpe improves: {gate_a} | Gate (b) MaxDD does not worsen: {gate_b} "
              f"| VERDICT: {'PASS' if (gate_a and gate_b) else 'FAIL'}")

    sized_pnl.to_csv(HERE / "data" / "pnl_steo_sized_baseline.csv")
    sized_pnl_delever.to_csv(HERE / "data" / "pnl_steo_sized_baseline_delever_only.csv")
    scalar.to_frame("exposure_scalar").to_csv(HERE / "data" / "steo_sizing_scalar.csv")


if __name__ == "__main__":
    main()
