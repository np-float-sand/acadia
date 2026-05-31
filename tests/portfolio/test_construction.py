import pandas as pd
import pytest
from grid_resilience.portfolio.construction import build_weights


def _scores(tickers=("VST", "NRG", "CNP"), values=(1.0, 0.0, -1.0)):
    return pd.Series(dict(zip(tickers, values)))


def test_xlu_hedge_true_no_individual_shorts():
    """When xlu_hedge=True, no ticker from factor_scores should have a negative weight."""
    scores = _scores()
    weights = build_weights(scores, xlu_hedge=True)
    for ticker in ["VST", "NRG", "CNP"]:
        assert weights.get(ticker, 0.0) >= 0.0


def test_xlu_hedge_true_xlu_weight_is_minus_half():
    """XLU hedge position must be exactly -0.5."""
    weights = build_weights(_scores(), xlu_hedge=True)
    assert weights["XLU"] == pytest.approx(-0.5)


def test_xlu_hedge_true_long_weights_sum_to_half():
    """Long book must still sum to +0.5."""
    weights = build_weights(_scores(), n_long=1, xlu_hedge=True)
    assert weights[weights > 0].sum() == pytest.approx(0.5)


def test_xlu_hedge_false_no_xlu_in_weights():
    """When xlu_hedge=False, XLU must not appear in the output."""
    weights = build_weights(_scores(), xlu_hedge=False)
    assert "XLU" not in weights.index


def test_xlu_hedge_false_short_weights_sum_to_minus_half():
    """Original L/S: short side sums to -0.5."""
    weights = build_weights(_scores(), n_short=2, xlu_hedge=False)
    assert weights[weights < 0].sum() == pytest.approx(-0.5)


def test_xlu_is_excluded_from_ranking():
    """XLU in factor_scores input must not be ranked into long positions."""
    scores = _scores(("VST", "NRG", "CNP", "XLU"), (1.0, 0.5, -0.5, 99.0))
    weights = build_weights(scores, n_long=1, xlu_hedge=True)
    assert weights.get("XLU", 0.0) == pytest.approx(-0.5)
    # VST should still be long (highest score among non-XLU names)
    assert weights.get("VST", 0.0) > 0


def test_all_nan_scores_returns_empty():
    """All-NaN factor scores must return empty Series — no naked short."""
    import numpy as np
    scores = pd.Series({"VST": np.nan, "NRG": np.nan, "CNP": np.nan})
    weights = build_weights(scores, xlu_hedge=True)
    assert weights.empty
