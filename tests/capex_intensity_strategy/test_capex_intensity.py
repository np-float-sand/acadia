import numpy as np
import pandas as pd
import pytest

from capex_intensity_strategy import backtest, config, edgar, fundamentals, universe
from tests.capex_intensity_strategy.conftest import FUEL, HEAVY, LIGHT


# -- config ---------------------------------------------------------------
def test_config_is_internally_consistent():
    assert 0 < config.QUANTILE <= 0.5
    assert 0 <= config.SHORT_WEIGHT <= 1
    assert config.TTM_QUARTERS == 4
    assert config.QUARTER_MIN_DAYS < 92 < config.QUARTER_MAX_DAYS
    assert config.RF == 0.04 and config.ANN == 252


def test_sic_table_excludes_the_off_thesis_codes():
    codes = set(universe.load_sic_codes()["sic"])
    # deliberately out: semiconductors, crude oil & gas, metal mining
    assert 3674 not in codes and 1311 not in codes and 1040 not in codes
    # deliberately in: power line construction recovers electrical contractors
    assert 1623 in codes and 4911 in codes and 3613 in codes


# -- point-in-time fundamentals -------------------------------------------
def test_ttm_panel_respects_filing_dates(synthetic_facts):
    """A quarter must be invisible until the day it is filed."""
    end = pd.Timestamp("2020-03-31")
    filed = end + pd.Timedelta(days=45)
    before = fundamentals.ttm_panel(synthetic_facts, pd.DatetimeIndex([filed - pd.Timedelta(days=1)]))
    after = fundamentals.ttm_panel(synthetic_facts, pd.DatetimeIndex([filed]))
    rev_before = before.set_index("ticker").loc["ETN", "revenue"]
    rev_after = after.set_index("ticker").loc["ETN", "revenue"]
    # the newly filed quarter replaces the oldest one in the trailing window
    assert rev_after == pytest.approx(rev_before)
    # and the window has actually rolled forward
    assert before["date"].iloc[0] < after["date"].iloc[0]


def test_ttm_panel_needs_four_filed_quarters(synthetic_facts):
    early = fundamentals.ttm_panel(synthetic_facts, pd.DatetimeIndex([pd.Timestamp("2017-06-30")]))
    assert early.empty


def test_capex_intensity_separates_light_from_heavy(synthetic_facts, dates):
    panel = fundamentals.ttm_panel(synthetic_facts, dates)
    score = fundamentals.capex_intensity(panel)
    light = score[LIGHT].mean().mean()
    heavy = score[HEAVY].mean().mean()
    assert light == pytest.approx(0.03, abs=1e-6)
    assert heavy == pytest.approx(0.25, abs=1e-6)
    assert light < heavy


def test_capex_intensity_nan_on_nonpositive_revenue():
    panel = pd.DataFrame({"date": [pd.Timestamp("2021-01-31")] * 2,
                          "ticker": ["A", "B"], "revenue": [0.0, 100.0],
                          "capex": [5.0, 10.0]})
    score = fundamentals.capex_intensity(panel)
    assert np.isnan(score.loc[pd.Timestamp("2021-01-31"), "A"])
    assert score.loc[pd.Timestamp("2021-01-31"), "B"] == pytest.approx(0.1)


# -- universe -------------------------------------------------------------
def test_liquidity_mask_excludes_persistently_illiquid(synthetic_dollar_volume):
    mask = universe.liquidity_mask(synthetic_dollar_volume)
    assert not mask["MYRG"].any()
    assert mask["ETN"].iloc[-1]


def test_lexicon_mask_is_effective_dated(synthetic_dollar_volume):
    idx = universe.liquidity_mask(synthetic_dollar_volume).index
    mask = universe.lexicon_mask(idx, ["CCJ", "ETN"], {"CCJ": [2021, 2022]})
    assert not mask.loc[mask.index.year == 2019, "CCJ"].any()   # before discovery
    assert mask.loc[mask.index.year == 2023, "CCJ"].all()       # stays in afterwards
    assert mask["ETN"].all()                                    # structural member


def test_membership_combines_liquidity_and_lexicon(synthetic_dollar_volume):
    mask = universe.membership(synthetic_dollar_volume, LIGHT + HEAVY + FUEL,
                               {"CCJ": [2022], "LEU": [2022]})
    assert not mask["MYRG"].any()                               # illiquid
    assert not mask.loc[mask.index.year == 2020, "CCJ"].any()   # not yet discovered
    assert mask.loc[mask.index.year == 2023, "CCJ"].all()


def test_discover_lexicon_members_uses_only_prior_filings():
    seen = []

    def fake_fetch(phrase, start, end):
        seen.append((start, end))
        return {"0000000001"}

    out = universe.discover_lexicon_members(fake_fetch, {"0000000001": "CCJ"}, range(2023, 2024))
    assert out == {"CCJ": [2023]}
    # the search window must close strictly before the universe-year starts
    for start, end in seen:
        assert pd.Timestamp(end) < pd.Timestamp("2023-01-01")


# -- portfolio ------------------------------------------------------------
def test_target_weights_go_long_light_and_short_heavy(synthetic_facts, dates,
                                                      synthetic_dollar_volume):
    score = fundamentals.capex_intensity(fundamentals.ttm_panel(synthetic_facts, dates))
    mask = universe.membership(synthetic_dollar_volume, list(score.columns), None)
    mask = mask.reindex(score.index).ffill().fillna(False)
    w = backtest.target_weights(score, mask)
    last = w.iloc[-1]
    assert (last[[t for t in LIGHT if t != "MYRG"]] >= 0).all()
    assert (last[HEAVY] <= 0).all()
    assert last.get("MYRG", 0.0) == 0.0          # illiquid never held


def test_long_leg_is_fully_invested_and_short_is_scaled(synthetic_facts, dates,
                                                        synthetic_dollar_volume):
    score = fundamentals.capex_intensity(fundamentals.ttm_panel(synthetic_facts, dates))
    mask = universe.membership(synthetic_dollar_volume, list(score.columns), None)
    mask = mask.reindex(score.index).ffill().fillna(False)
    w = backtest.target_weights(score, mask, short_weight=0.5)
    row = w.iloc[-1]
    assert row[row > 0].sum() == pytest.approx(1.0)
    assert row[row < 0].sum() == pytest.approx(-0.5)


def test_short_weight_zero_is_long_only(synthetic_facts, dates, synthetic_dollar_volume):
    score = fundamentals.capex_intensity(fundamentals.ttm_panel(synthetic_facts, dates))
    mask = universe.membership(synthetic_dollar_volume, list(score.columns), None)
    mask = mask.reindex(score.index).ffill().fillna(False)
    w = backtest.target_weights(score, mask, short_weight=0.0)
    assert (w.iloc[-1] >= 0).all()


def test_run_applies_lag_costs_and_vol_target(synthetic_prices):
    returns = synthetic_prices.pct_change()
    cols = list(synthetic_prices.columns)
    monthly = returns.resample("ME").last().index
    w = pd.DataFrame(0.0, index=monthly, columns=cols)
    w[cols[0]] = 1.0
    free = backtest.run(w, returns, cost=0.0, vol_target=0)
    charged = backtest.run(w, returns, cost=0.01, vol_target=0)
    assert charged.sum() < free.sum()                    # costs bite
    targeted = backtest.run(w, returns, cost=0.0, vol_target=0.10)
    realised = targeted.dropna().std() * np.sqrt(config.ANN)
    assert 0.03 < realised < 0.25                        # vol target roughly binds


def test_run_has_no_lookahead(synthetic_prices):
    """Weights must apply from the day AFTER they are set."""
    returns = synthetic_prices.pct_change()
    cols = list(synthetic_prices.columns)
    w = pd.DataFrame(0.0, index=returns.index, columns=cols)
    w.iloc[:, 0] = 1.0
    net = backtest.run(w, returns, cost=0.0, vol_target=0)
    expected = returns[cols[0]]
    # net on day t equals the asset return on day t (weight known at t-1)
    aligned = pd.concat([net.rename("net"), expected.rename("exp")], axis=1).dropna()
    assert aligned["net"].iloc[1:].corr(aligned["exp"].iloc[1:]) == pytest.approx(1.0, abs=1e-9)


def test_diagnostics_reports_breadth(synthetic_facts, dates, synthetic_dollar_volume,
                                     synthetic_prices):
    score = fundamentals.capex_intensity(fundamentals.ttm_panel(synthetic_facts, dates))
    mask = universe.membership(synthetic_dollar_volume, list(score.columns), None)
    mask = mask.reindex(score.index).ffill().fillna(False)
    w = backtest.target_weights(score, mask)
    d = backtest.diagnostics(w, synthetic_prices.pct_change())
    assert d["avg_long_names"] >= 1 and d["avg_short_names"] >= 1
    assert d["annual_turnover"] >= 0


def test_summarise_produces_expected_columns(synthetic_prices):
    net = synthetic_prices.pct_change().mean(axis=1)
    out = backtest.summarise(net, {"all": ("2019-01-01", "2024-12-31")})
    assert list(out.columns) == ["window", "cagr", "ann_vol", "sharpe",
                                 "max_dd", "calmar", "n_obs"]
    assert len(out) == 1


def test_vol_target_none_uses_config_zero_disables(synthetic_prices):
    """None means 'use the config default'; 0 means 'off'. They must differ."""
    returns = synthetic_prices.pct_change()
    cols = list(synthetic_prices.columns)
    w = pd.DataFrame(0.0, index=returns.index, columns=cols)
    w.iloc[:, 0] = 1.0
    off = backtest.run(w, returns, cost=0.0, vol_target=0)
    defaulted = backtest.run(w, returns, cost=0.0, vol_target=None)
    assert off.notna().all()                       # no warm-up when disabled
    assert defaulted.isna().sum() >= config.VOL_LOOKBACK_DAYS - 1
    scaled = defaulted.dropna().std() * np.sqrt(config.ANN)
    assert abs(scaled - config.VOL_TARGET) < 0.15  # roughly on target


# -- tag coverage and staleness (2026-09-23 fix) --------------------------
def _facts(rows):
    return pd.DataFrame(rows, columns=["ticker", "item", "tag", "start", "end", "filed", "value"])


def _quarters(ticker, item, tag, periods, value, rank=0):
    out = []
    for period in periods:
        start, end = period.start_time, period.end_time.normalize()
        out.append((ticker, item, tag, start, end, end + pd.Timedelta(days=45), value))
    return out


def test_parse_company_facts_falls_through_to_a_later_tag():
    """`Revenues` present but with only annual spans must not shadow the ASC-606 tag.

    This is the ETN/EMR/PWR failure: the old code broke out of the tag loop on the
    first tag present in the JSON, so a filer whose `Revenues` facts are all annual
    never got a quarterly revenue series at all.
    """
    gaap = {
        "Revenues": {"units": {"USD": [
            {"start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-15",
             "form": "10-K", "val": 4000.0}]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"start": "2023-01-01", "end": "2023-03-31", "filed": "2023-05-01",
             "form": "10-Q", "val": 1000.0}]}},
    }
    parsed = edgar.parse_company_facts(gaap)
    tags = set(parsed[parsed["item"] == "revenue"]["tag"])
    # both tags survive the parse -- the quarterly ASC-606 fact is no longer shadowed
    assert "RevenueFromContractWithCustomerExcludingAssessedTax" in tags
    assert "Revenues" in tags


def test_capex_tag_list_covers_productive_assets():
    """HUBB/GEV/CARR/JCI/EME/WCC tag capex only as PaymentsToAcquireProductiveAssets."""
    assert "PaymentsToAcquireProductiveAssets" in edgar.FACT_TAGS["capex"]


def test_ttm_panel_prefers_the_tag_with_the_most_visible_quarters():
    """A thin legacy tag must not win over a fully populated one for the same item."""
    periods = list(pd.period_range("2023Q1", "2023Q4", freq="Q"))
    rows = (_quarters("A", "revenue", "Revenues", periods[:1], 10.0, rank=0)
            + _quarters("A", "revenue", "RevenueFromContractWithCustomerExcludingAssessedTax",
                        periods, 100.0, rank=1)
            + _quarters("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment",
                        periods, 5.0, rank=0))
    panel = fundamentals.ttm_panel(_facts(rows), pd.DatetimeIndex([pd.Timestamp("2024-03-31")]))
    assert panel["revenue"].iloc[0] == pytest.approx(400.0)


def test_ttm_panel_drops_stale_series():
    """A filer whose facts stop years earlier must not be scored off them (AEP)."""
    periods = list(pd.period_range("2020Q1", "2020Q4", freq="Q"))
    rows = (_quarters("A", "revenue", "Revenues", periods, 100.0)
            + _quarters("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment",
                        periods, 5.0))
    facts = _facts(rows)
    fresh = fundamentals.ttm_panel(facts, pd.DatetimeIndex([pd.Timestamp("2021-06-30")]))
    stale = fundamentals.ttm_panel(facts, pd.DatetimeIndex([pd.Timestamp("2026-08-31")]))
    assert not fresh.empty
    assert stale.empty


def test_ttm_panel_requires_four_contiguous_quarters():
    """Four Q1s spread over four years are not a trailing twelve months."""
    gappy = [p for p in pd.period_range("2020Q1", "2023Q4", freq="Q") if p.quarter == 1]
    rows = (_quarters("A", "revenue", "Revenues", gappy, 100.0)
            + _quarters("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment",
                        gappy, 5.0))
    panel = fundamentals.ttm_panel(_facts(rows), pd.DatetimeIndex([pd.Timestamp("2024-03-31")]))
    assert panel.empty


def test_ttm_panel_requires_revenue_and_capex_to_cover_the_same_window():
    """A 2026 revenue TTM over a 2021 capex TTM is the AEP mis-score: reject it."""
    rev_p = list(pd.period_range("2023Q1", "2023Q4", freq="Q"))
    cap_p = list(pd.period_range("2022Q1", "2022Q4", freq="Q"))
    rows = (_quarters("A", "revenue", "Revenues", rev_p, 100.0)
            + _quarters("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment",
                        cap_p, 5.0))
    panel = fundamentals.ttm_panel(_facts(rows), pd.DatetimeIndex([pd.Timestamp("2024-03-31")]))
    score = fundamentals.capex_intensity(panel)
    assert score.empty or bool(score.isna().all().all())


# -- YTD differencing (2026-09-23 fix) ------------------------------------
def _ytd(ticker, item, tag, fy_start, ends, values, filed):
    return [(ticker, item, tag, fy_start, e, f, v)
            for e, v, f in zip(ends, values, filed)]


def test_derive_quarters_differences_a_ytd_ladder():
    """Cash-flow facts are 3/6/9/12-month cumulative; quarters come from differencing.

    This is the capex failure: a 10-Q reports capex year-to-date, so only Q1 is ever
    a discrete quarter. The old 60..200-day span filter kept Q1 *and* the 6-month YTD
    and called both quarters, then silently ran out of data for everyone else.
    """
    rows = _ytd("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment",
                pd.Timestamp("2023-01-01"),
                [pd.Timestamp(d) for d in ("2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31")],
                [10.0, 25.0, 45.0, 70.0],
                [pd.Timestamp(d) for d in ("2023-05-01", "2023-08-01", "2023-11-01", "2024-02-15")])
    q = fundamentals.derive_quarters(fundamentals.load_facts(_facts(rows)))
    q = q.sort_values("end")
    assert len(q) == 4
    assert list(q["value"]) == pytest.approx([10.0, 15.0, 20.0, 25.0])
    # the derived Q4 is only visible once the 10-K that completes it is filed
    assert q["filed"].iloc[-1] == pd.Timestamp("2024-02-15")


def test_derive_quarters_keeps_discrete_facts_and_drops_half_years():
    """A genuine 3-month fact is used as-is; a stray 6-month span is never a quarter."""
    rows = [("A", "revenue", "Revenues", pd.Timestamp("2023-04-01"), pd.Timestamp("2023-06-30"),
             pd.Timestamp("2023-08-01"), 100.0),
            ("A", "revenue", "Revenues", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-30"),
             pd.Timestamp("2023-08-01"), 190.0)]
    q = fundamentals.derive_quarters(fundamentals.load_facts(_facts(rows)))
    assert len(q) == 1
    assert q["value"].iloc[0] == pytest.approx(100.0)


def test_ttm_panel_builds_a_full_year_from_a_ytd_capex_ladder():
    """End to end: YTD capex + discrete quarterly revenue -> one scored name."""
    periods = list(pd.period_range("2023Q1", "2023Q4", freq="Q"))
    rev = _quarters("A", "revenue", "Revenues", periods, 100.0)
    cap = _ytd("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment",
               pd.Timestamp("2023-01-01"),
               [p.end_time.normalize() for p in periods], [5.0, 10.0, 15.0, 20.0],
               [p.end_time.normalize() + pd.Timedelta(days=45) for p in periods])
    panel = fundamentals.ttm_panel(_facts(rev + cap), pd.DatetimeIndex([pd.Timestamp("2024-03-31")]))
    score = fundamentals.capex_intensity(panel)
    assert score["A"].iloc[0] == pytest.approx(20.0 / 400.0)


# -- item-agnostic panel + mention counts (screen_auto plumbing) -----------
def test_ttm_panel_carries_any_item():
    """`net_income` must survive the panel, not just revenue and capex."""
    periods = list(pd.period_range("2023Q1", "2023Q4", freq="Q"))
    rows = (_quarters("A", "revenue", "Revenues", periods, 100.0)
            + _quarters("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment", periods, 5.0)
            + _quarters("A", "net_income", "NetIncomeLoss", periods, 7.0))
    panel = fundamentals.ttm_panel(_facts(rows), pd.DatetimeIndex([pd.Timestamp("2024-03-31")]))
    assert "net_income" in panel.columns
    assert panel["net_income"].iloc[0] == pytest.approx(28.0)


def test_capex_intensity_unaffected_by_extra_items():
    """Adding an item must not change the score."""
    periods = list(pd.period_range("2023Q1", "2023Q4", freq="Q"))
    base = (_quarters("A", "revenue", "Revenues", periods, 100.0)
            + _quarters("A", "capex", "PaymentsToAcquirePropertyPlantAndEquipment", periods, 5.0))
    extra = base + _quarters("A", "net_income", "NetIncomeLoss", periods, 7.0)
    dates = pd.DatetimeIndex([pd.Timestamp("2024-03-31")])
    a = fundamentals.capex_intensity(fundamentals.ttm_panel(_facts(base), dates))
    b = fundamentals.capex_intensity(fundamentals.ttm_panel(_facts(extra), dates))
    assert a["A"].iloc[0] == pytest.approx(b["A"].iloc[0])


def test_net_income_tag_is_fetched():
    assert "NetIncomeLoss" in edgar.FACT_TAGS["net_income"]


def test_parse_mention_count_reads_the_hit_total():
    """A CIK-filtered full-text search answers 'how many of this filer's 10-Ks say X'."""
    assert edgar.parse_mention_count({"hits": {"total": {"value": 4}}}) == 4
    assert edgar.parse_mention_count({}) == 0


def test_parse_mention_dates_reads_every_hit_filing_date():
    """One query per (filer, phrase) over the whole span beats one per year."""
    payload = {"hits": {"hits": [
        {"_source": {"file_date": "2024-02-15"}},
        {"_source": {"file_date": "2023-02-17"}},
        {"_source": {}},
    ]}}
    assert edgar.parse_mention_dates(payload) == ["2024-02-15", "2023-02-17"]
    assert edgar.parse_mention_dates({}) == []
