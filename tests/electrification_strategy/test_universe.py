import pandas as pd
import pytest

from electrification_strategy import config, universe


# -- loaders --------------------------------------------------------------
def test_load_seed_schema():
    df = universe.load_seed()
    assert df.index.name == "ticker"
    assert set(df.columns) == {"sub_industry", "profitable_2026"}
    assert df["sub_industry"].isin(config.SUB_INDUSTRIES).all()
    assert df["profitable_2026"].dtype == bool
    assert bool(df.loc["FLNC", "profitable_2026"]) is False
    assert bool(df.loc["STEM", "profitable_2026"]) is False
    assert bool(df.loc["ETN", "profitable_2026"]) is True


def test_load_membership_schema():
    m = universe.load_membership()
    assert m.index.name == "ticker"
    assert list(m.columns) == ["VOLT", "ELFY", "ZAP", "GRID", "PAVE"]
    assert m.to_numpy().dtype == bool
    assert bool(m.loc["ETN", "GRID"]) is True
    assert bool(m.loc["FLNC", "GRID"]) is False


def test_thematic_etfs_excludes_pave():
    assert universe.THEMATIC_ETFS == ["VOLT", "ELFY", "ZAP", "GRID"]


# -- constructions ------------------------------------------------------
def _members(construction, prices, asof="2026-06-30"):
    return universe.members(construction, pd.Timestamp(asof), prices)


def test_marquee_is_the_nine(synthetic_prices):
    assert _members("marquee", synthetic_prices) == sorted(config.MARQUEE_UNIVERSE)


def test_frozen_drops_lossmakers_and_non_etf_names(synthetic_prices):
    m = _members("frozen", synthetic_prices)
    assert "FLNC" not in m and "STEM" not in m          # profitable_2026 == False
    assert "THR" not in m and "ENS" not in m            # in zero ETFs
    assert {"ETN", "HUBB", "NVT", "EMR", "AME", "POWL", "GEV", "VRT", "PWR"} <= set(m)


def test_thematic_is_the_consensus_subset(synthetic_prices):
    m = set(_members("thematic", synthetic_prices))
    assert {"ETN", "HUBB", "GEV", "VRT", "PWR", "NVT", "POWL", "AEIS", "GNRC", "AME"} <= m
    assert "EMR" not in m       # ELFY only -> 1
    assert "RRX" not in m       # ELFY only -> 1
    assert "MYRG" not in m      # VOLT only -> 1
    assert "PAVE" not in m and "ATKR" not in m   # ATKR: PAVE only, PAVE excluded from the >=2 rule


def test_listing_gate_excludes_young_names(synthetic_prices):
    early = universe.members("frozen", pd.Timestamp("2019-06-30"), synthetic_prices)
    assert "GEV" not in early
    late = universe.members("frozen", pd.Timestamp("2026-06-30"), synthetic_prices)
    assert "GEV" in late


def test_target_fn_equal_weight_capped(synthetic_prices):
    fn = universe.target_fn("thematic", synthetic_prices)
    w = fn(list(synthetic_prices.columns), pd.Timestamp("2026-06-30"))
    assert w.sum() == pytest.approx(1.0)
    assert (w <= config.MAX_SINGLE_NAME_WEIGHT + 1e-9).all()
    assert w.nunique() == 1  # 10 names, cap not binding -> equal


def test_target_fn_empty_when_no_members(synthetic_prices):
    fn = universe.target_fn("thematic", synthetic_prices)
    w = fn(["NOTATICKER"], pd.Timestamp("2026-06-30"))
    assert w.empty
