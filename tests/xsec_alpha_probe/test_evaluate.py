import numpy as np
import pandas as pd
import pytest

from research.xsec_alpha_probe import evaluate as ev


def test_winsorize_clips_each_row_to_its_own_quantiles():
    df = pd.DataFrame([[1, 2, 3, 4, 100.0], [-50, 1, 2, 3, 4.0]])
    out = ev.winsorize(df, 0.2)
    assert out.iloc[0].max() < 100 and out.iloc[0].max() >= 4
    assert out.iloc[1].min() > -50
    # middle values untouched
    assert out.iloc[0, 2] == 3


def test_zscore_rows_has_zero_mean_unit_std():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 30)))
    z = ev.zscore(df)
    assert np.allclose(z.mean(axis=1), 0)
    assert np.allclose(z.std(axis=1), 1)


def test_ic_series_perfect_and_inverse():
    sig = pd.DataFrame([[1, 2, 3, 4, 5.0]] * 3)
    fwd = sig * 2
    assert np.allclose(ev.ic_series(sig, fwd, method="spearman", min_names=3), 1.0)
    assert np.allclose(ev.ic_series(sig, -fwd, method="spearman", min_names=3), -1.0)


def test_ic_series_skips_months_with_too_few_names():
    sig = pd.DataFrame([[1, 2, np.nan, np.nan, np.nan]])
    fwd = pd.DataFrame([[1, 2, 3, 4, 5.0]])
    assert ev.ic_series(sig, fwd, min_names=5).isna().all()


def test_newey_west_t_matches_plain_t_for_iid_with_zero_lags():
    x = pd.Series(np.random.default_rng(1).normal(0.1, 1, 400))
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert ev.nw_tstat(x, lags=0) == pytest.approx(plain, rel=0.01)


def test_newey_west_shrinks_t_for_overlapping_series():
    rng = np.random.default_rng(2)
    e = rng.normal(0.05, 1, 600)
    overlapped = pd.Series(np.convolve(e, np.ones(3), "valid"))
    assert abs(ev.nw_tstat(overlapped, lags=2)) < abs(ev.nw_tstat(overlapped, lags=0))


def test_quantile_weights_equal_sum_to_one_on_top_bucket():
    sig = pd.Series(np.arange(10.0), index=list("abcdefghij"))
    w = ev.bucket_weights(sig, q=0.2, top=True, scale=None)
    assert set(w[w > 0].index) == {"i", "j"}
    assert w.sum() == pytest.approx(1.0)


def test_inverse_risk_weighting_tilts_to_lower_risk_name():
    sig = pd.Series([0.0, 1, 2, 3, 4], index=list("abcde"))
    risk = pd.Series([1, 1, 1, 0.1, 0.4], index=list("abcde"))
    w = ev.bucket_weights(sig, q=0.4, top=True, scale=risk)
    assert w["d"] > w["e"] > 0
    assert w.sum() == pytest.approx(1.0)


def test_downside_semivol_ignores_up_moves():
    r = pd.DataFrame({"up": [0.05, 0.05, -0.01, 0.05], "dn": [-0.05, 0.01, -0.05, 0.01]})
    sv = ev.semivol(r)
    assert sv["dn"] > sv["up"]


def test_signal_verdict_bar():
    assert ev.verdict(2.0, 1.6, 1.7) == "PASS"
    assert ev.verdict(-2.0, -1.6, -1.7) == "PASS (inverted)"
    assert ev.verdict(1.8, 1.9, 0.4) == "CLOSE"
    assert ev.verdict(1.8, 2.0, -0.5) == "FAIL"
    assert ev.verdict(0.5, 1.0, 0.2) == "FAIL"
