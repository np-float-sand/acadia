import pandas as pd

from grid_equipment_basket import grid_regime as gr


def _blk(cagr, sharpe, max_dd, calendar):
    return {"metrics": {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd},
            "calendar": pd.Series(calendar, dtype=float)}


# ── _calmar ─────────────────────────────────────────────────────────────────

def test_calmar_is_cagr_over_abs_maxdd():
    assert gr._calmar({"cagr": 0.30, "max_dd": -0.15}) == 2.0
    assert gr._calmar({"cagr": 0.10, "max_dd": 0.0}) != gr._calmar({"cagr": 0.10, "max_dd": -0.1})


# ── gate_check: G1 / G2 need BOTH Sharpe and Calmar ─────────────────────────

def test_g1_false_when_calmar_worse_even_if_sharpe_better():
    rung_p = _blk(0.20, 1.60, -0.20, [0.2, 0.2])      # calmar 1.0
    l1_p = _blk(0.20, 1.40, -0.10, [0.2, 0.2])        # calmar 2.0  -> rung worse
    same_prior = _blk(0.10, 0.50, -0.30, [0.1, 0.1])
    bh = _blk(0.25, 1.2, -0.4, [0.25, 0.25])
    g = gr.gate_check(rung_p, same_prior, l1_p, same_prior, bh, bh)
    assert g["G1"] is False


def test_g1_true_when_both_sharpe_and_calmar_beat_l1():
    rung_p = _blk(0.25, 1.70, -0.12, [0.25, 0.25])    # calmar ~2.08
    l1_p = _blk(0.20, 1.40, -0.15, [0.20, 0.20])      # calmar ~1.33
    prior = _blk(0.10, 0.50, -0.30, [0.1, 0.1])
    bh = _blk(0.25, 1.2, -0.4, [0.25, 0.25])
    g = gr.gate_check(rung_p, prior, l1_p, prior, bh, bh)
    assert g["G1"] is True


def test_g2_evaluated_on_the_prior_window_independently():
    rung_p = _blk(0.30, 1.90, -0.10, [0.3, 0.3])         # calmar 3.0
    l1_p = _blk(0.22, 1.55, -0.14, [0.22, 0.22])         # calmar ~1.57 -> rung beats primary
    rung_prior = _blk(0.05, 0.30, -0.40, [0.05, 0.05])   # worse than L1 prior
    l1_prior = _blk(0.06, 0.35, -0.38, [0.06, 0.06])
    bh = _blk(0.2, 1.0, -0.45, [0.2, 0.2])
    g = gr.gate_check(rung_p, rung_prior, l1_p, l1_prior, bh, bh)
    assert g["G1"] is True
    assert g["G2"] is False


# ── G3: not a bigger drag than layer 1 ─────────────────────────────────────

def test_g3_false_when_rung_mean_annual_return_below_l1_on_a_window():
    good = _blk(0.25, 1.7, -0.12, [0.25, 0.25])
    rung_prior = _blk(0.02, 0.6, -0.20, [0.00, 0.04])    # mean 0.02
    l1_prior = _blk(0.08, 0.6, -0.20, [0.06, 0.10])      # mean 0.08 -> rung is a bigger drag
    bh = _blk(0.15, 1.0, -0.4, [0.15, 0.15])
    g = gr.gate_check(good, rung_prior, good, l1_prior, bh, bh)
    assert g["G3"] is False


def test_g3_true_when_rung_holds_up_on_both_windows():
    rung = _blk(0.25, 1.7, -0.12, [0.24, 0.26])
    l1 = _blk(0.20, 1.5, -0.15, [0.18, 0.22])
    bh = _blk(0.30, 1.2, -0.4, [0.30, 0.30])
    g = gr.gate_check(rung, rung, l1, l1, bh, bh)
    assert g["G3"] is True


# ── marginal flag ─────────────────────────────────────────────────────────

def test_marginal_true_when_a_g1_gap_is_under_0_05():
    rung_p = _blk(0.201, 1.43, -0.149, [0.2, 0.2])
    l1_p = _blk(0.20, 1.40, -0.15, [0.2, 0.2])           # sharpe gap 0.03 < 0.05
    prior = _blk(0.10, 0.50, -0.30, [0.1, 0.1])
    bh = _blk(0.25, 1.2, -0.4, [0.25, 0.25])
    g = gr.gate_check(rung_p, prior, l1_p, prior, bh, bh)
    assert g["G1"] is True
    assert g["marginal"] is True


# ── final_verdict folds in the plateau (G4) ───────────────────────────────

def _passing_gate():
    return {"G1": True, "G2": True, "G3": True, "marginal": False}


def test_final_verdict_pass_when_gates_and_all_neighbours_pass():
    assert gr.final_verdict(_passing_gate(), neighbours_pass=[True, True, True]) == "PASS"


def test_final_verdict_knife_edge_when_a_neighbour_fails():
    assert gr.final_verdict(_passing_gate(), neighbours_pass=[True, False, True]) == "knife-edge"


def test_final_verdict_marginal_when_gates_pass_but_gap_tiny():
    g = _passing_gate() | {"marginal": True}
    assert gr.final_verdict(g, neighbours_pass=[True, True]) == "marginal"


def test_final_verdict_fail_when_a_core_gate_is_false():
    g = _passing_gate() | {"G2": False}
    assert gr.final_verdict(g, neighbours_pass=[True, True]) == "FAIL"
