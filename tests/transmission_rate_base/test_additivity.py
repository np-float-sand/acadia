import numpy as np
import pandas as pd
import pytest

from transmission_rate_base import additivity as ad


def test_run_additivity_recovers_planted_alpha_and_loading():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2011-01", "2020-12", freq="M").to_timestamp("M")
    mkt = pd.Series(rng.normal(0, 0.03, len(idx)), index=idx, name="util_beta")
    other = pd.DataFrame({c: rng.normal(0, 0.02, len(idx)) for c in
                          ["xlu", "dividend_yield", "low_vol", "size", "momentum",
                           "capex_intensity", "value"]}, index=idx)
    controls = pd.concat([mkt, other], axis=1)
    factor = 0.004 + 0.5 * mkt + rng.normal(0, 0.005, len(idx))
    out = ad.run_additivity(factor, controls)
    assert out["alpha_ann"] == pytest.approx(0.048, abs=0.02)
    assert out["coef"]["util_beta"] == pytest.approx(0.5, abs=0.15)
    assert out["alpha_t"] > 2.0
    assert 0.0 <= out["r2"] <= 1.0


def test_load_style_inputs_schema(tmp_path):
    p = tmp_path / "si.csv"
    p.write_text("ticker,fy,dividend_yield,market_cap,book_value,capex,net_ppe\n"
                 "AEP,2020,0.035,4.5e10,2.0e10,7e9,7e10\n")
    si = ad.load_style_inputs(p)
    assert list(si.columns) == ["ticker", "fy", "dividend_yield", "market_cap",
                                "book_value", "capex", "net_ppe"]


def test_build_controls_produces_all_eight_columns():
    idx = pd.bdate_range("2013-01-01", "2019-12-31")
    tickers = [f"T{i:02d}" for i in range(12)]
    rng = np.random.default_rng(3)
    dr = pd.DataFrame(rng.normal(0, 0.008, (len(idx), len(tickers))), index=idx, columns=tickers)
    ew = dr.mean(axis=1)
    xlu = ew + rng.normal(0, 0.002, len(idx))
    si = pd.DataFrame([dict(ticker=t, fy=fy, dividend_yield=0.03 + i * 0.001,
                            market_cap=1e10 * (i + 1), book_value=5e9 * (i + 1),
                            capex=1e9 * (i + 1), net_ppe=8e9 * (i + 1))
                       for fy in range(2013, 2019) for i, t in enumerate(tickers)])
    from transmission_rate_base.portfolio import rebalance_dates
    rd = rebalance_dates(idx, "2015-01-01", "2019-12-31")
    ctrl = ad.build_controls(dr, ew, xlu, si, rd)
    assert list(ctrl.columns) == ["util_beta", "xlu", "dividend_yield", "low_vol",
                                  "size", "momentum", "capex_intensity", "value"]
    assert len(ctrl) > 12
