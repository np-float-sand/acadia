import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import grid_regime as gr

# rung 6 is deliberately fed an empty ERCOT frame in these structural tests; the
# "fall back to the PJM composite" warning is the path under test, not a defect.
pytestmark = pytest.mark.filterwarnings(
    "ignore:grid_regime rung 6.*:RuntimeWarning")


# ── stubs ───────────────────────────────────────────────────────────────────

_TICKERS = ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]
_BENCH = ["XLI", "SPY", "XLU", "GRID", "PAVE"]


def _price_fn_with_crashes(crash_windows):
    """Basket drifts up; each (start, end) in crash_windows is a steep decline
    shared by every name. Benchmarks drift up quietly."""
    idx = pd.bdate_range("2019-06-03", "2026-02-27")
    daily = np.full(len(idx), 0.0009)
    for cs, ce in crash_windows:
        m = (idx >= pd.Timestamp(cs)) & (idx <= pd.Timestamp(ce))
        daily[m] = -0.012
    base = pd.Series(daily, index=idx)

    def price_fn(names, start, end):
        rng = np.random.default_rng(7)
        out = {}
        for t in names:
            if t in _TICKERS:
                noise = rng.normal(0, 0.004, len(idx))
                out[t] = 100 * (1 + base + pd.Series(noise, index=idx)).cumprod()
            else:
                out[t] = 100 * (1 + pd.Series(rng.normal(0.0003, 0.008, len(idx)), index=idx)).cumprod()
        return pd.DataFrame({k: v for k, v in out.items() if k in names}).loc[start:end]

    return price_fn


def _empty_ercot_fn(start, end):
    return pd.DataFrame(columns=["time", "location", "lmp"])


def _composite_fn_easing_before(crash_windows, lead_days=45):
    """Composite goes to -2.0 for `lead_days` before each crash and through it
    (regime says 'grid bottleneck easing -> step back'); ~0 otherwise."""
    idx = pd.bdate_range("2018-01-01", "2025-12-31")
    v = pd.Series(0.0, index=idx)
    for cs, ce in crash_windows:
        lo = pd.Timestamp(cs) - pd.Timedelta(days=lead_days)
        v.loc[lo:pd.Timestamp(ce)] = -2.0

    def composite_fn(start, end, *, zones=None, w_cong=None, w_reserve=None, zone_weight=None):
        return v.loc[start:end]

    return composite_fn


# ── structure ───────────────────────────────────────────────────────────────

def test_regime_report_has_baselines_and_runs_active_rungs():
    crashes = [("2021-05-01", "2021-06-15"), ("2024-08-01", "2024-09-20")]
    rep = gr.regime_report(price_fn=_price_fn_with_crashes(crashes),
                           composite_fn=_composite_fn_easing_before([]),   # flat -> unhelpful
                           ercot_fn=_empty_ercot_fn)
    for k in ("buy_and_hold", "vol_target_only", "layer1_only"):
        assert set(rep["baselines"][k]) == {"primary", "prior"}
        assert {"cagr", "sharpe", "max_dd"} <= set(rep["baselines"][k]["primary"]["metrics"])
    names = [r["name"] for r in rep["rungs"]]
    assert names[0].startswith("1:")
    deferred = [r for r in rep["rungs"] if r.get("status") == "deferred"]
    assert deferred and "RT/DA" in deferred[0]["name"]
    for r in rep["rungs"]:
        if r.get("status") == "deferred":
            continue
        assert {"G1", "G2", "G3"} <= set(r["gate"])
        assert r["verdict"] in {"PASS", "FAIL", "marginal", "knife-edge"}
        assert {"avg_mult", "flips"} <= set(r["mechanics"])


def test_regime_report_stops_at_first_pass_and_skips_later_rungs():
    crashes = [("2021-05-01", "2021-06-15"), ("2024-08-01", "2024-09-20")]
    rep = gr.regime_report(price_fn=_price_fn_with_crashes(crashes),
                           composite_fn=_composite_fn_easing_before(crashes),
                           ercot_fn=_empty_ercot_fn)
    run = [r for r in rep["rungs"] if r.get("status") != "deferred"]
    assert rep["stopped_at"] == run[0]["name"]            # rung 1 passes
    assert run[0]["verdict"] == "PASS"
    assert len(run) == 1                                  # later rungs not evaluated
    # and it genuinely beat layer-1-only on both windows
    g = run[0]["gate"]
    assert g["G1"] and g["G2"] and g["G3"]


def test_regime_report_noise_composite_does_not_pass():
    crashes = [("2021-05-01", "2021-06-15"), ("2024-08-01", "2024-09-20")]
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", "2025-12-31")
    noise = pd.Series(rng.normal(0, 1.0, len(idx)), index=idx)

    def composite_fn(start, end, *, zones=None, w_cong=None, w_reserve=None, zone_weight=None):
        return noise.loc[start:end]

    rep = gr.regime_report(price_fn=_price_fn_with_crashes(crashes),
                           composite_fn=composite_fn, ercot_fn=_empty_ercot_fn)
    assert rep["stopped_at"] is None
    assert all(r["verdict"] != "PASS" for r in rep["rungs"] if r.get("status") != "deferred")


def test_regime_table_renders_without_error():
    rep = gr.regime_report(price_fn=_price_fn_with_crashes([("2024-08-01", "2024-09-20")]),
                           composite_fn=_composite_fn_easing_before([]),
                           ercot_fn=_empty_ercot_fn)
    txt = gr.regime_table(rep)
    assert "layer1_only" in txt or "layer-1" in txt
    assert "PASS" in txt or "FAIL" in txt
