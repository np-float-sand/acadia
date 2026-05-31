import pandas as pd
import pytest
from grid_resilience.factor.resilience_score import build_factor


def _betas(tickers, values):
    return pd.DataFrame({"stress_beta": values}, index=tickers)


def test_generator_positive_beta_ranks_above_td_negative_beta():
    """NRG (generator) with +beta should outrank CNP (T&D) with -beta."""
    betas = _betas(["NRG", "CNP"], [1.0, -1.0])
    scores = build_factor(betas)
    # Generator with +1.0 → score +1.0 (resilient). T&D with -1.0 → score -1.0 (hurt by stress).
    assert scores["NRG"] > scores["CNP"]


def test_generator_positive_beta_is_long_candidate():
    """Generator with positive beta (profits from LMP spikes) should rank high."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    # Signed scores: NRG=+0.8, VST=+0.4, CNP=-0.6 (hurt by stress).
    # Both generators (NRG, VST) should be positive; CNP should be negative.
    assert scores["NRG"] > 0
    assert scores["VST"] > 0
    assert scores["NRG"] > scores["VST"]
    assert scores.idxmax() == "NRG"


def test_td_negative_beta_is_short_candidate():
    """T&D hurt by grid stress (negative beta) should rank low."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    # Signed scores: NRG=+0.8, VST=+0.4, CNP=-0.6 (hurt by stress).
    # T&D (CNP) should rank lower than both generators.
    assert scores["VST"] > scores["CNP"]
    assert scores["NRG"] > scores["CNP"]


def test_generator_ranks_highest_among_three():
    """With one high-beta generator and two T&D names, generator tops the ranking."""
    betas = _betas(["NRG", "EXC", "CNP"], [1.5, -0.2, -1.0])
    scores = build_factor(betas)
    assert scores.idxmax() == "NRG"


def test_non_generator_ranking_unchanged():
    """Two non-generators: less negative beta = higher score (more resilient)."""
    betas = _betas(["EXC", "CNP"], [-0.5, -1.5])
    scores = build_factor(betas)
    # Signed scores: EXC=-0.5, CNP=-1.5. EXC is less negative (more resilient).
    assert scores["EXC"] > scores["CNP"]


def test_generator_negative_beta_stays_negative():
    """A generator with negative beta (unusual) must still score negative — no abs()."""
    betas = _betas(["NRG", "CNP"], [-0.8, -0.4])
    scores = build_factor(betas)
    assert scores["NRG"] < scores["CNP"]  # NRG more negative beta → lower score even for generator
