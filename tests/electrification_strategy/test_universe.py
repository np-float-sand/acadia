import numpy as np
import pandas as pd
import pytest

from electrification_strategy import config, universe


# -- loaders --------------------------------------------------------------
def test_load_seed_schema():
    df = universe.load_seed()
    assert df.index.name == "ticker"
    assert set(df.columns) == {
        "sub_industry",
        "customer_institutional", "earnings_valued", "low_policy_dependence",
    }
    assert df["sub_industry"].isin(config.SUB_INDUSTRIES).all()
    for col in ("customer_institutional", "earnings_valued", "low_policy_dependence"):
        assert df[col].dtype == bool
    assert "profitable_2026" not in df.columns      # removed 2026-09-22, provably redundant
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
    assert universe.CONSTRUCTIONS == ("marquee", "frozen", "thematic", "screen", "screen_reit")


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


def test_frozen_is_in_any_etf(synthetic_prices):
    m = set(_members("frozen", synthetic_prices))
    assert {"ETN", "HUBB", "VRT", "POWL", "AEIS", "TT", "EME", "MOD"} <= m
    assert "FLNC" not in m and "STEM" not in m          # in zero visible ETF holdings
    assert "RRX" not in m and "WCC" not in m and "MYRG" not in m   # in zero visible ETF holdings


def test_screen_is_the_supplier_rule_no_etf_gate(synthetic_prices):
    m = set(_members("screen", synthetic_prices))
    # picked on stated properties, incl. names no thematic ETF holds
    assert {"ETN", "HUBB", "EMR", "AME", "RRX", "POWL", "ATKR", "AEIS", "AYI", "WCC",
            "VRT", "PWR", "MYRG", "PRIM", "EME", "FIX", "MTZ", "TT", "CARR", "JCI",
            "SPXC", "MOD", "GEV"} <= m
    assert "GNRC" not in m                     # customer_institutional == False
    assert "FLNC" not in m and "STEM" not in m  # fail earnings_valued AND low_policy_dependence
    # the differentiator: screen holds names the ETF-consensus construction misses
    assert "RRX" in m and "RRX" not in _members("thematic", synthetic_prices)
    assert "WCC" in m and "WCC" not in _members("frozen", synthetic_prices)


def test_screen_quality_or_keeps_gev_prim_mtz(synthetic_prices):
    # these fail low_policy_dependence but pass earnings_valued -> clear the OR bar
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


def test_screen_excludes_reit_rows_even_though_they_are_in_the_seed(synthetic_prices):
    asof = synthetic_prices.index[-1]
    screen = set(universe.members("screen", asof, synthetic_prices))
    assert "IRM" not in screen and "AMT" not in screen


def test_screen_reit_is_a_superset_of_screen(synthetic_prices):
    asof = synthetic_prices.index[-1]
    screen = set(universe.members("screen", asof, synthetic_prices))
    screen_reit = set(universe.members("screen_reit", asof, synthetic_prices))
    assert screen.issubset(screen_reit)
    assert "IRM" in screen_reit  # passes all 3 judgment columns


def test_screen_auto_is_selectable_but_not_in_the_default_grid():
    """A plain CLI run must not depend on the build artifact `screen_auto` needs."""
    assert "screen_auto" in universe.ALL_CONSTRUCTIONS
    assert "screen_auto" not in universe.CONSTRUCTIONS


def test_screen_auto_respects_the_listing_gate(monkeypatch):
    """Membership is precomputed, but a name still cannot trade before it lists."""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    members = pd.DataFrame([[True, True]], index=idx, columns=["OLD", "NEW"])
    monkeypatch.setattr(universe.auto_screen, "load_membership", lambda: members)
    prices = pd.DataFrame(
        {"OLD": 1.0, "NEW": np.nan},
        index=pd.bdate_range("2022-01-03", "2024-01-31"))
    prices.loc["2024-01-02":, "NEW"] = 1.0
    assert universe.members("screen_auto", pd.Timestamp("2024-01-31"), prices) == ["OLD"]


def test_candidate_tickers_include_screen_auto_members_when_requested(monkeypatch):
    """`screen_auto` picks from the SEC universe, not the seed list: its names must be
    priced, or run_comparison silently drops them (24 of 46 were dropped, 2026-09-25)."""
    held = pd.DataFrame({"OFFSEED": [True], "ALSO": [False]}, index=[pd.Timestamp("2024-01-31")])
    monkeypatch.setattr(universe.auto_screen, "load_membership", lambda: held)
    base = set(universe.candidate_tickers(universe.CONSTRUCTIONS))
    assert "OFFSEED" not in base                     # default grid never needs the artifact
    with_auto = set(universe.candidate_tickers(("screen", "screen_auto")))
    assert "OFFSEED" in with_auto and "ALSO" not in with_auto
    assert base <= with_auto
