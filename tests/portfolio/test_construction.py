import pandas as pd
import pytest
from grid_resilience.portfolio.construction import build_weights, build_grouped_weights, build_rolling_weights


def _scores(tickers=("VST", "NRG", "CNP"), values=(1.0, 0.0, -1.0)):
    return pd.Series(dict(zip(tickers, values)))


# Real active-universe tickers (grid_resilience.data.universe.UNIVERSE filtered to the
# 5 supported ISOs), grouped by business_model in utility_node_map.TICKER_NODE_MAP:
#   merchant (2):  VST, NRG
#   mixed (5):     CNP, DTE, ETR, EXC, PEG
#   regulated (11): AEP, PPL, FE, D, AEE, EVRG, WEC, CMS, PCG, EIX, XEL
def _grouped_scores():
    return pd.Series({
        # merchant — VST scores higher than NRG
        "VST": 2.0, "NRG": 1.0,
        # mixed — descending EXC > PEG > ETR > DTE > CNP
        "EXC": 5.0, "PEG": 4.0, "ETR": 3.0, "DTE": 2.0, "CNP": 1.0,
        # regulated — descending AEP > PPL > FE > D > AEE > EVRG > WEC > CMS > PCG > EIX > XEL
        "AEP": 11.0, "PPL": 10.0, "FE": 9.0, "D": 8.0, "AEE": 7.0, "EVRG": 6.0,
        "WEC": 5.0, "CMS": 4.0, "PCG": 3.0, "EIX": 2.0, "XEL": 1.0,
    })


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


# ── build_grouped_weights() — peer-group (basket-vs-basket) construction ──────

def test_grouped_weights_total_book_sums_to_half():
    """Total long book sums to +0.5 and short book to -0.5, same as build_weights()."""
    weights = build_grouped_weights(_grouped_scores())
    assert weights[weights > 0].sum() == pytest.approx(0.5)
    assert weights[weights < 0].sum() == pytest.approx(-0.5)


def test_grouped_weights_each_sleeve_is_dollar_neutral():
    """Each business-model sleeve's long and short legs must cancel independently."""
    weights = build_grouped_weights(_grouped_scores())
    merchant  = ["VST", "NRG"]
    mixed     = ["CNP", "DTE", "ETR", "EXC", "PEG"]
    regulated = ["AEP", "PPL", "FE", "D", "AEE", "EVRG", "WEC", "CMS", "PCG", "EIX", "XEL"]
    for group in (merchant, mixed, regulated):
        assert weights.reindex(group).fillna(0.0).sum() == pytest.approx(0.0, abs=1e-9)


def test_grouped_weights_selects_correct_tickers_per_group():
    """Highest-scored name(s) per sleeve go long, lowest-scored go short."""
    weights = build_grouped_weights(_grouped_scores())
    # merchant: 2 names shrink to 1 long / 1 short
    assert weights["VST"] > 0
    assert weights["NRG"] < 0
    # mixed: n_long=1, n_short=2
    assert weights["EXC"] > 0
    assert weights["DTE"] < 0
    assert weights["CNP"] < 0
    assert weights["PEG"] == 0.0
    assert weights["ETR"] == 0.0
    # regulated: n_long=2, n_short=3
    assert weights["AEP"] > 0
    assert weights["PPL"] > 0
    assert weights["PCG"] < 0
    assert weights["EIX"] < 0
    assert weights["XEL"] < 0
    assert weights["FE"] == 0.0


def test_grouped_weights_capital_scaling_matches_group_size():
    """Each sleeve's capital share is proportional to its share of the 18-name universe."""
    weights = build_grouped_weights(_grouped_scores())
    merchant_long  = weights.reindex(["VST", "NRG"]).clip(lower=0).sum()
    mixed_long     = weights.reindex(["CNP", "DTE", "ETR", "EXC", "PEG"]).clip(lower=0).sum()
    regulated_long = weights.reindex(
        ["AEP", "PPL", "FE", "D", "AEE", "EVRG", "WEC", "CMS", "PCG", "EIX", "XEL"]
    ).clip(lower=0).sum()
    assert merchant_long  == pytest.approx(0.5 * 2 / 18)
    assert mixed_long     == pytest.approx(0.5 * 5 / 18)
    assert regulated_long == pytest.approx(0.5 * 11 / 18)


def test_rolling_weights_grouped_produces_same_output_shape():
    """build_rolling_weights(grouped=True) must yield the same [date, ticker, weight]
    shape as the ungrouped path, so weights_to_matrix() consumes it unchanged."""
    scores = _grouped_scores()
    rolling_factors = pd.DataFrame({
        "date": pd.Timestamp("2024-01-31"),
        "ticker": scores.index,
        "factor_score": scores.values,
    })
    weights_df = build_rolling_weights(rolling_factors, grouped=True)
    assert list(weights_df.columns) == ["date", "ticker", "weight"]
    assert not weights_df.empty
    # matches what build_grouped_weights() produces directly for the same date
    expected = build_grouped_weights(scores)
    expected = expected[expected != 0.0]
    assert set(weights_df["ticker"]) == set(expected.index)
