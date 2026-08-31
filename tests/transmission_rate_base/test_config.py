import math

from transmission_rate_base import config as c


def test_gate_thresholds_match_spec():
    assert c.GATE["min_mean_ic"] == 0.03
    assert c.GATE["min_ic_t"] == 2.0
    assert c.GATE["min_ls_sharpe"] == 0.40
    assert c.GATE["max_quintile_inversions"] == 1
    assert c.GATE["min_positive_years_frac"] == 0.60
    assert c.GATE["min_alpha_ann"] == 0.03
    assert c.GATE["min_alpha_t"] == 2.0
    assert c.GATE["max_additivity_r2"] == 0.60
    assert c.GATE["pre_thesis_min_t"] == -2.0


def test_windows_and_signal_constants():
    assert c.PRIMARY_START == "2011-01-01"
    assert c.PRIMARY_END == "2025-12-31"
    assert c.PRE_THESIS_START == "2003-01-01"
    assert c.PRE_THESIS_END == "2010-12-31"
    assert c.REBALANCE_MONTH == 5
    assert c.SIGNAL_CAGR_YEARS == 3
    assert c.MIN_HISTORY_YEARS == 4
    assert c.STRUCTURAL_BREAK_LOG == math.log(2.0)


def test_ferc_table_keys_and_pudl_base():
    assert c.PUDL_BASE == "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/"
    assert set(c.FERC_TABLES) == {"plant_in_service", "dep_by_function", "plant_summary", "utility_xwalk"}
    assert c.FERC_TABLES["plant_in_service"] == "core_ferc1__yearly_plant_in_service_sched204"


def test_universe_seed_is_nonempty_unique_tuple():
    assert isinstance(c.UNIVERSE_SEED, tuple)
    assert len(c.UNIVERSE_SEED) >= 30
    assert len(set(c.UNIVERSE_SEED)) == len(c.UNIVERSE_SEED)
