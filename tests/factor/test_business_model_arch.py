import pandas as pd
import pytest
from grid_resilience.factor.resilience_score import build_factor


def _betas(tickers, values):
    return pd.DataFrame({"stress_beta": values}, index=tickers)


def _pt(tickers, values):
    return pd.Series(values, index=tickers)


# ── Hard switch ────────────────────────────────────────────────────────────────

def test_hard_switch_merchant_uses_beta():
    """Merchant (pass_through=1.0) score is driven by stress_beta, not ICR."""
    betas = _betas(["VST", "PPL"], [2.0, 2.0])        # equal betas
    icr   = pd.Series({"VST": 1.0, "PPL": 10.0})     # PPL has much better ICR
    pt    = _pt(["VST", "PPL"], [1.0, 0.05])
    scores = build_factor(betas, icr=icr, arch="hard_switch", pass_through=pt)
    # VST routed to beta path (equal betas → neutral beta score).
    # PPL routed to ICR path (high ICR → positive score).
    # PPL should outscore VST, proving VST is not elevated by ICR (it stays at neutral beta).
    assert scores["PPL"] > scores["VST"]


def test_hard_switch_regulated_uses_icr():
    """Regulated ticker (pass_through<0.5) with high ICR should outscore low-ICR peer."""
    betas = _betas(["PPL", "FE"], [-0.5, -0.5])       # equal (bad) stress betas
    icr   = pd.Series({"PPL": 5.0, "FE": 1.0})       # PPL much better covered
    pt    = _pt(["PPL", "FE"], [0.05, 0.05])
    scores = build_factor(betas, icr=icr, arch="hard_switch", pass_through=pt)
    assert scores["PPL"] > scores["FE"]


def test_hard_switch_regulated_no_icr_falls_back_to_beta():
    """Regulated ticker with no ICR data falls back to stress_beta signal."""
    betas = _betas(["AEP", "FE"], [0.5, -0.5])
    pt    = _pt(["AEP", "FE"], [0.05, 0.05])
    scores = build_factor(betas, icr=None, arch="hard_switch", pass_through=pt)
    # Fallback to beta: AEP (+0.5) should outscore FE (-0.5)
    assert scores["AEP"] > scores["FE"]


def test_hard_switch_mixed_ticker_threshold():
    """pass_through >= 0.5 routes to merchant (beta) path; < 0.5 routes to regulated (ICR)."""
    betas = _betas(["A", "B"], [0.0, 0.0])            # neutral betas
    icr   = pd.Series({"A": 1.0, "B": 10.0})
    # A is merchant (0.6), B is regulated (0.4)
    pt    = _pt(["A", "B"], [0.6, 0.4])
    scores = build_factor(betas, icr=icr, arch="hard_switch", pass_through=pt)
    # B (regulated) gets ICR signal → high ICR → higher score
    assert scores["B"] > scores["A"]


def test_hard_switch_no_pass_through_is_backward_compatible():
    """Without pass_through, hard_switch behaves identically to original function."""
    betas = _betas(["NRG", "CNP"], [1.0, -1.0])
    original = build_factor(betas)
    with_arch = build_factor(betas, arch="hard_switch", pass_through=None)
    pd.testing.assert_series_equal(original, with_arch)


# ── Revenue mix ────────────────────────────────────────────────────────────────

def test_revenue_mix_pure_merchant_equals_beta_only():
    """Ticker with pass_through=1.0 gets pure beta score regardless of ICR."""
    betas  = _betas(["VST", "VST2"], [1.5, -1.5])
    icr    = pd.Series({"VST": 0.1, "VST2": 99.0})   # ICR would flip ranking
    pt     = _pt(["VST", "VST2"], [1.0, 1.0])
    scores = build_factor(betas, icr=icr, arch="revenue_mix", pass_through=pt)
    assert scores["VST"] > scores["VST2"]


def test_revenue_mix_pure_regulated_equals_icr_only():
    """Ticker with pass_through=0.0 gets pure ICR score regardless of beta."""
    betas  = _betas(["PPL", "FE"], [-0.5, -0.5])      # equal betas
    icr    = pd.Series({"PPL": 8.0, "FE": 1.0})
    pt     = _pt(["PPL", "FE"], [0.0, 0.0])
    scores = build_factor(betas, icr=icr, arch="revenue_mix", pass_through=pt)
    assert scores["PPL"] > scores["FE"]


def test_revenue_mix_midpoint_blends_both():
    """Ticker with pass_through=0.5 is intermediate between pure beta and pure ICR."""
    betas  = _betas(["A", "B", "C"], [1.0, 1.0, 1.0])
    icr    = pd.Series({"A": 1.0, "B": 5.0, "C": 10.0})
    pt     = _pt(["A", "B", "C"], [0.5, 0.5, 0.5])
    scores = build_factor(betas, icr=icr, arch="revenue_mix", pass_through=pt)
    # Equal betas + higher ICR → higher score
    assert scores["C"] > scores["B"] > scores["A"]


# ── Dual-track ─────────────────────────────────────────────────────────────────

def test_dual_track_merchant_ranked_within_merchant_group():
    """
    VST with moderate beta outranks NRG with low beta when both are in merchant group,
    even if the regulated tickers have extreme betas that would distort global z-scoring.
    """
    # Regulated tickers with huge betas that would dominate a global z-score
    betas = _betas(["VST", "NRG", "PPL", "FE"], [0.6, 0.3, 5.0, -5.0])
    icr   = pd.Series({"VST": 2.0, "NRG": 2.0, "PPL": 5.0, "FE": 1.0})
    pt    = _pt(["VST", "NRG", "PPL", "FE"], [1.0, 0.85, 0.05, 0.05])
    scores = build_factor(betas, icr=icr, arch="dual_track", pass_through=pt)
    # VST (higher beta among merchants) should still outrank NRG
    assert scores["VST"] > scores["NRG"]


def test_dual_track_regulated_ranked_within_regulated_group():
    """PPL (high ICR) outranks FE (low ICR) within the regulated group."""
    betas = _betas(["VST", "PPL", "FE"], [1.0, -0.5, -0.5])
    icr   = pd.Series({"VST": 2.0, "PPL": 8.0, "FE": 1.0})
    pt    = _pt(["VST", "PPL", "FE"], [1.0, 0.05, 0.05])
    scores = build_factor(betas, icr=icr, arch="dual_track", pass_through=pt)
    assert scores["PPL"] > scores["FE"]
