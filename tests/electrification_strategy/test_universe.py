import pandas as pd
import pytest

from electrification_strategy import config, universe


# -- loaders --------------------------------------------------------------
def test_load_seed_schema():
    df = universe.load_seed()
    assert df.index.name == "ticker"
    assert set(df.columns) == {
        "sub_industry", "profitable_2026",
        "customer_institutional", "earnings_valued", "low_policy_dependence",
    }
    assert df["sub_industry"].isin(config.SUB_INDUSTRIES).all()
    for col in ("profitable_2026", "customer_institutional", "earnings_valued", "low_policy_dependence"):
        assert df[col].dtype == bool
    assert bool(df.loc["FLNC", "profitable_2026"]) is False
    assert bool(df.loc["ETN", "profitable_2026"]) is True
    assert bool(df.loc["GNRC", "customer_institutional"]) is False   # residential-heavy backup power
    assert bool(df.loc["GEV", "low_policy_dependence"]) is False      # wind segment leans on PTC/ITC


def test_load_membership_schema():
    m = universe.load_membership()
    assert m.index.name == "ticker"
    assert list(m.columns) == ["VOLT", "ELFY", "ZAP", "GRID", "PAVE"]
    assert m.to_numpy().dtype == bool
    assert bool(m.loc["ETN", "GRID"]) is True
    assert bool(m.loc["VRT", "VOLT"]) is True
    assert bool(m.loc["VRT", "GRID"]) is False


def test_thematic_etfs_excludes_pave():
    assert universe.THEMATIC_ETFS == ["VOLT", "ELFY", "ZAP", "GRID"]
    assert universe.CONSTRUCTIONS == ("marquee", "frozen", "thematic", "screen")


# -- constructions ------------------------------------------------------
def _members(construction, prices, asof="2026-06-30"):
    return universe.members(construction, pd.Timestamp(asof), prices)


def test_marquee_is_the_nine(synthetic_prices):
    assert _members("marquee", synthetic_prices) == sorted(config.MARQUEE_UNIVERSE)


def test_thematic_is_the_ge2_consensus(synthetic_prices):
    m = set(_members("thematic", synthetic_prices))
    assert m == {"ETN", "HUBB", "GEV", "PWR", "NVT", "AME", "JCI"}
    assert "VRT" not in m and "POWL" not in m and "AEIS" not in m   # VOLT-only -> 1
    assert "EMR" not in m                                           # ELFY-only (PAVE doesn't count)


def test_frozen_is_in_any_etf_and_profitable(synthetic_prices):
    m = set(_members("frozen", synthetic_prices))
    assert {"ETN", "HUBB", "VRT", "POWL", "AEIS", "TT", "EME", "MOD"} <= m
    assert "FLNC" not in m and "STEM" not in m          # profitable_2026 == False
    assert "RRX" not in m and "WCC" not in m and "MYRG" not in m   # in zero visible ETF holdings


def test_screen_is_the_supplier_rule_no_etf_gate(synthetic_prices):
    m = set(_members("screen", synthetic_prices))
    # picked on stated properties, incl. names no thematic ETF holds
    assert {"ETN", "HUBB", "EMR", "AME", "RRX", "POWL", "ATKR", "AEIS", "AYI", "WCC",
            "VRT", "PWR", "MYRG", "PRIM", "EME", "FIX", "MTZ", "TT", "CARR", "JCI",
            "SPXC", "MOD", "GEV"} <= m
    assert "GNRC" not in m                     # customer_institutional == False
    assert "FLNC" not in m and "STEM" not in m  # 0 of 3
    # the differentiator: screen holds names the ETF-consensus construction misses
    assert "RRX" in m and "RRX" not in _members("thematic", synthetic_prices)
    assert "WCC" in m and "WCC" not in _members("frozen", synthetic_prices)


def test_screen_two_of_three_keeps_gev_prim_mtz(synthetic_prices):
    # these fail low_policy_dependence but pass profitable + earnings_valued -> 2 of 3
    m = set(_members("screen", synthetic_prices))
    assert {"GEV", "PRIM", "MTZ"} <= m


def test_listing_gate_excludes_young_names(synthetic_prices):
    early = universe.members("screen", pd.Timestamp("2019-06-30"), synthetic_prices)
    assert "GEV" not in early
    late = universe.members("screen", pd.Timestamp("2026-06-30"), synthetic_prices)
    assert "GEV" in late


def test_target_fn_equal_weight_capped(synthetic_prices):
    fn = universe.target_fn("thematic", synthetic_prices)
    w = fn(list(synthetic_prices.columns), pd.Timestamp("2026-06-30"))
    assert w.sum() == pytest.approx(1.0)
    assert (w <= config.MAX_SINGLE_NAME_WEIGHT + 1e-9).all()
    assert w.nunique() == 1  # 7 names, cap not binding -> equal


def test_target_fn_empty_when_no_members(synthetic_prices):
    fn = universe.target_fn("thematic", synthetic_prices)
    w = fn(["NOTATICKER"], pd.Timestamp("2026-06-30"))
    assert w.empty
