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
