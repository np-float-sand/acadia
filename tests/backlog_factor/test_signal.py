import numpy as np
import pandas as pd
import pytest

from backlog_factor import signal as sig


def _rpo_one(mvs, avs=None):
    qe = pd.date_range("2021-03-31", periods=len(mvs), freq="QE")
    av = avs or [q + pd.Timedelta(days=40) for q in qe]
    return pd.DataFrame({"quarter_end": qe, "availability_date": pd.to_datetime(av),
                         "metric_value": [float(x) for x in mvs]})


def test_surprise_is_growth_minus_trailing_mean():
    # 6 quarters. g between consecutive = log(1.10) for q2..q5, then log(1.25) at q6.
    mvs = [100, 110, 121, 133.1, 146.41, 183.0125]
    out = sig.name_surprise_series(_rpo_one(mvs), min_quarters=6)
    last = out.iloc[-1]
    g6 = np.log(183.0125 / 146.41)
    exp6 = np.log(1.10)                         # trailing 4 g's are all log(1.10)
    assert last["g"] == pytest.approx(g6)
    assert last["expected_g"] == pytest.approx(exp6)
    assert last["surprise"] == pytest.approx(g6 - exp6)


def test_needs_min_quarters():
    out = sig.name_surprise_series(_rpo_one([100, 110, 121, 133.1, 146.41]), min_quarters=6)
    assert out["surprise"].dropna().empty


def test_gappy_span_is_nan():
    qe = pd.to_datetime(["2021-03-31", "2021-06-30", "2022-06-30", "2022-12-31", "2023-06-30", "2023-12-31"])
    df = pd.DataFrame({"quarter_end": qe, "availability_date": qe + pd.Timedelta(days=40),
                       "metric_value": [100.0, 110, 121, 133.1, 146.41, 161.0]})
    out = sig.name_surprise_series(df, min_quarters=6, span_min=300, span_max=430)
    assert out["surprise"].dropna().empty          # iloc[-1]..iloc[-5] spans ~30 months


def test_level_discontinuity_nans_surprise_across_break_and_next_4():
    mvs = [100, 110, 121, 133.1, 146.41, 900.0, 990, 1089, 1197.9, 1317.7, 1449.5]   # ~6x jump at index 5
    out = sig.name_surprise_series(_rpo_one(mvs), min_quarters=6)
    assert out["surprise"].isna().sum() >= 5       # the jump row + next 4 -> NaN
    assert out["surprise"].notna().any()           # a later row is valid again


def test_staleness_decay_linear_to_zero():
    assert sig.staleness_decay(0.0, 60) == pytest.approx(1.0)
    assert sig.staleness_decay(30.0, 60) == pytest.approx(0.5)
    assert sig.staleness_decay(60.0, 60) == pytest.approx(0.0)
    assert sig.staleness_decay(90.0, 60) == pytest.approx(0.0)
    s = sig.staleness_decay(pd.Series([0.0, 15.0, 60.0]), 60)
    assert list(s) == pytest.approx([1.0, 0.75, 0.0])


def test_neutralize_cross_section_residual_is_orthogonal_to_group_and_size():
    raw = pd.Series({"A": 2.0, "B": 1.0, "C": -1.0, "D": -2.0, "E": 0.5, "F": -0.5})
    grp = pd.Series({"A": "m", "B": "m", "C": "m", "D": "e", "E": "e", "F": "e"})
    lmc = pd.Series({"A": 9.0, "B": 10.0, "C": 11.0, "D": 9.5, "E": 10.5, "F": 11.5})
    out = sig.neutralize_cross_section(raw, grp, lmc, winsor_sigma=3.0)
    assert abs(float(np.corrcoef(out.values, lmc.reindex(out.index).values)[0, 1])) < 1e-6
    for gname in ["m", "e"]:
        assert out[grp.reindex(out.index) == gname].mean() == pytest.approx(0.0, abs=1e-9)


def test_signal_on_date_uses_latest_visible_surprise_within_staleness():
    sp = pd.DataFrame({
        "ticker": ["A", "A", "B", "C"],
        "availability_date": pd.to_datetime(["2023-02-01", "2023-05-01", "2023-04-20", "2022-01-01"]),
        "surprise": [0.1, 0.4, -0.3, 0.9],
    })
    grp = {"A": "m", "B": "m", "C": "m"}
    lmc = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
    out = sig.signal_on_date(sp, pd.Timestamp("2023-06-01"), industry_group_map=grp,
                             log_mktcap=lmc, horizon_days=63, staleness_max_days=200)
    assert "C" not in out.index           # 2022-01 surprise is > 200 days stale
    assert set(out.index) == {"A", "B"}
