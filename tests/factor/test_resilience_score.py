import pandas as pd
import pytest
from grid_resilience.factor.resilience_score import build_factor


def _betas(tickers, values):
    return pd.DataFrame({"stress_beta": values}, index=tickers)


def test_generator_positive_beta_ranks_above_td_negative_beta():
    """NRG (generator) with +beta should outrank CNP (T&D) with -beta."""
    betas = _betas(["NRG", "CNP"], [1.0, -1.0])
    scores = build_factor(betas)
    assert scores["NRG"] > scores["CNP"]


def test_generator_positive_beta_is_long_candidate():
    """Generator with positive beta (profits from LMP spikes) must score positive."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    assert scores["NRG"] > 0
    assert scores["VST"] > 0


def test_td_negative_beta_is_short_candidate():
    """T&D hurt by grid stress (negative beta) must score negative."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    assert scores["CNP"] < 0


def test_generator_ranks_highest_among_three():
    """With one high-beta generator and two T&D names, generator tops the ranking."""
    betas = _betas(["NRG", "EXC", "CNP"], [1.5, -0.2, -1.0])
    scores = build_factor(betas)
    assert scores.idxmax() == "NRG"


def test_non_generator_ranking_unchanged():
    """Two non-generators with identical beta magnitudes: more negative = lower score."""
    betas = _betas(["EXC", "CNP"], [-0.5, -1.5])
    scores = build_factor(betas)
    assert scores["EXC"] > scores["CNP"]
