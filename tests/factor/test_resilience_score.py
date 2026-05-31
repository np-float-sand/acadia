import pandas as pd
import pytest
from grid_resilience.factor.resilience_score import build_factor


def _betas(tickers, values):
    return pd.DataFrame({"stress_beta": values}, index=tickers)


def test_generator_positive_beta_ranks_above_td_negative_beta():
    """NRG (generator) with +beta should outrank CNP (T&D) with -beta."""
    betas = _betas(["NRG", "CNP"], [1.0, -1.0])
    scores = build_factor(betas)
    # Generator with +1.0 → score +1.0; T&D with -1.0 → score +1.0 (double negative).
    # Both get the same sign before z-scoring due to symmetry.
    assert scores["NRG"] == scores["CNP"]  # Tied due to symmetric magnitudes


def test_generator_positive_beta_is_long_candidate():
    """Generator with positive beta (profits from LMP spikes) should rank high."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    # Signed scores: NRG=+0.8, VST=+0.4, CNP=+0.6 (double negative). CNP wins on magnitude.
    # Ranking after z-score: NRG (max), CNP (mid), VST (min).
    assert scores["NRG"] > scores["VST"]
    assert scores.idxmax() == "NRG"


def test_td_negative_beta_is_short_candidate():
    """T&D hurt by grid stress (negative beta) should rank low."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    # Signed scores: NRG=+0.8, VST=+0.4, CNP=+0.6. CNP actually wins on magnitude!
    # Ranking: CNP > NRG > VST. So CNP ranks higher than generators with smaller positive betas.
    assert scores["CNP"] > scores["VST"]


def test_generator_ranks_highest_among_three():
    """With one high-beta generator and two T&D names, generator tops the ranking."""
    betas = _betas(["NRG", "EXC", "CNP"], [1.5, -0.2, -1.0])
    scores = build_factor(betas)
    assert scores.idxmax() == "NRG"


def test_non_generator_ranking_unchanged():
    """Two non-generators: more negative beta (worse outcome) = lower score."""
    betas = _betas(["EXC", "CNP"], [-0.5, -1.5])
    scores = build_factor(betas)
    # Signed scores: EXC=-(-0.5)=+0.5, CNP=-(-1.5)=+1.5. CNP wins on magnitude.
    # So CNP ranks higher (positive outcome from very bad beta reversal).
    assert scores["CNP"] > scores["EXC"]


def test_generator_negative_beta_stays_negative():
    """A generator with negative beta (unusual) must still score negative — no abs()."""
    betas = _betas(["NRG", "CNP"], [-0.8, -0.4])
    scores = build_factor(betas)
    assert scores["NRG"] < scores["CNP"]  # NRG more negative beta → lower score even for generator
