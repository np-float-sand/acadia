import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import grid_regime as gr, config

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _hourly(zone_cong: dict, start, end, step_h=6):
    """zone_cong: {zone: series-like of daily |congestion $| OR a scalar}."""
    days = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(0)
    rows = []
    for i, d in enumerate(days):
        for z, c in zone_cong.items():
            cv = c[i] if hasattr(c, "__len__") else c
            for h in range(0, 24, step_h):
                rows.append({"time": d + pd.Timedelta(hours=h), "location": z,
                             "lmp": 30.0, "congestion": -cv + rng.normal(0, 0.03),
                             "energy": 30.0, "loss": 0.0})
    return pd.DataFrame(rows)


# ── _rest_of_pjm_congestion ─────────────────────────────────────────────────

def test_rest_of_pjm_congestion_excludes_dc_and_aggregate_zones():
    df = _hourly({"DOM": 9.0, "AEP": 9.0, "PECO": 2.0, "BGE": 4.0,
                  "PJM-RTO": 50.0, "OVEC": 99.0}, "2021-01-01", "2021-01-05")
    rest = gr._rest_of_pjm_congestion(df, dc_zones=["DOM", "AEP"])
    # only PECO (2) + BGE (4) -> ~3, nowhere near DOM/AEP (9) or the aggregates
    assert 2.0 < rest.mean() < 4.5
    assert rest.index.name == "date"


# ── relative_regime_composite: common-mode rejection ────────────────────────

def test_relative_composite_rejects_a_system_wide_congestion_move():
    n = 400
    ramp = 2.0 + 0.02 * np.arange(n)          # everything rises together
    df = _hourly({"DOM": ramp, "AEP": ramp, "COMED": ramp, "PPL": ramp,
                  "PECO": ramp, "BGE": ramp, "JCPL": ramp, "APS": ramp},
                 "2021-01-01", periods_end := "2022-02-04")
    comp = gr.relative_regime_composite("2021-01-01", "2022-02-04",
                                        dc_zones=["DOM", "AEP", "COMED", "PPL"],
                                        lmp_fn=lambda *a, **k: df,
                                        zscore_window=90, zscore_minp=40)
    # z-scoring always rescales to ~unit spread, so "rejection" means no
    # persistent bias and no drift -- not small magnitude.
    c = comp.dropna()
    assert abs(c.iloc[:80].mean()) < 0.5
    assert abs(c.iloc[-80:].mean()) < 0.5
    assert abs(c.iloc[-80:].mean() - c.iloc[:80].mean()) < 0.5


def test_relative_composite_fires_when_only_dc_zones_congest():
    n = 400
    flat = np.full(n, 2.0)
    dc = 2.0 + 0.03 * np.arange(n)            # only the DC zones climb
    df = _hourly({"DOM": dc, "AEP": dc, "COMED": dc, "PPL": dc,
                  "PECO": flat, "BGE": flat, "JCPL": flat, "APS": flat},
                 "2021-01-01", "2022-02-04")
    comp = gr.relative_regime_composite("2021-01-01", "2022-02-04",
                                        dc_zones=["DOM", "AEP", "COMED", "PPL"],
                                        lmp_fn=lambda *a, **k: df,
                                        zscore_window=90, zscore_minp=40)
    assert comp.dropna().iloc[-60:].mean() > 1.0


# ── relative_signal_report ─────────────────────────────────────────────────

def _price_fn():
    idx = pd.bdate_range("2019-06-03", "2026-02-27")
    rng = np.random.default_rng(4)
    base = pd.Series(0.0009, index=idx)
    base.loc["2021-05-01":"2021-06-15"] = -0.012
    base.loc["2024-08-01":"2024-09-20"] = -0.012
    out = {}
    for t in ["ETN", "GEV", "HUBB", "VRT", "NVT", "PWR", "MYRG", "PRIM", "FLNC"]:
        out[t] = 100 * (1 + base + pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)).cumprod()
    for b in config.BENCHMARKS:
        out[b] = 100 * (1 + pd.Series(rng.normal(3e-4, 8e-3, len(idx)), index=idx)).cumprod()
    return lambda names, s, e: pd.DataFrame({k: v for k, v in out.items() if k in names}).loc[s:e]


def test_relative_signal_report_has_baselines_absolute_and_relative():
    n = 3000
    dc = 2.0 + 1.5 * np.sin(np.arange(n) / 120)
    rest = 2.0 + 1.5 * np.sin(np.arange(n) / 120 + 0.3)
    def lmp_fn(*a, **k):
        return _hourly({"DOM": dc, "AEP": dc, "COMED": dc, "PPL": dc,
                        "PECO": rest, "BGE": rest, "JCPL": rest, "APS": rest,
                        "ATSI": rest, "DPL": rest},
                       "2018-01-01", "2025-12-31", step_h=24)
    rep = gr.relative_signal_report(price_fn=_price_fn(), lmp_fn=lmp_fn,
                                    zscore_window=200, zscore_minp=120)
    assert {"buy_and_hold", "vol_target_only", "layer1_only"} <= set(rep["baselines"])
    for key in ("absolute", "relative"):
        assert {"primary", "prior"} <= set(rep[key]["block"])
        assert {"G1", "G2", "G3"} <= set(rep[key]["gate"])
    assert rep["verdict"] in {"PASS", "FAIL", "marginal", "knife-edge"}
    assert isinstance(gr.relative_signal_table(rep), str)
