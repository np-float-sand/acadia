"""The mechanical (`screen_auto`) replacement for the hand-recorded seed list."""
import numpy as np
import pandas as pd
import pytest

from electrification_strategy import auto_screen, config


@pytest.fixture
def ciks():
    return {"ETN": "0001551182", "ROK": "0001024478", "DUK": "0001326160"}


def test_vocabulary_members_uses_only_prior_year_filings(ciks):
    """Universe-year Y is decided by filings published strictly before Y begins."""
    seen = []

    def fetch(phrase, cik, start, end):
        seen.append((start, end))
        return 1

    auto_screen.vocabulary_members(fetch, {"ETN": ciks["ETN"]}, ('"data center"',), range(2024, 2025))
    assert seen == [("2022-01-01", "2023-12-31")]


def test_vocabulary_members_requires_the_minimum_hit_count(ciks):
    def fetch(phrase, cik, start, end):
        return 2 if cik == ciks["ETN"] else 0

    out = auto_screen.vocabulary_members(fetch, ciks, ('"data center"',), range(2024, 2025),
                                         min_hits=1)
    assert out == {"ETN": [2024]}


def test_vocabulary_members_unions_terms(ciks):
    """A filer clears the bar on any one term, not all of them."""
    def fetch(phrase, cik, start, end):
        return 1 if (cik == ciks["ROK"] and phrase == '"substation"') else 0

    out = auto_screen.vocabulary_members(fetch, ciks, ('"data center"', '"substation"'),
                                         range(2024, 2025), min_hits=1)
    assert sorted(out) == ["ROK"]


def test_profitable_mask_is_true_only_on_positive_trailing_earnings():
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-31")] * 3,
        "ticker": ["A", "B", "C"],
        "net_income": [10.0, -4.0, np.nan],
    })
    out = auto_screen.profitable_mask(panel)
    assert bool(out.loc[pd.Timestamp("2024-01-31"), "A"])
    assert not bool(out.loc[pd.Timestamp("2024-01-31"), "B"])
    assert not bool(out.loc[pd.Timestamp("2024-01-31"), "C"])


def test_light_capex_mask_keeps_the_light_side_of_the_eligible_pool():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    score = pd.DataFrame([[0.01, 0.02, 0.30, 0.40]], index=idx, columns=list("ABCD"))
    mask = pd.DataFrame([[True, True, True, True]], index=idx, columns=list("ABCD"))
    out = auto_screen.light_capex_mask(score, mask, quantile=0.5)
    assert list(out.columns[out.iloc[0]]) == ["A", "B"]


def test_light_capex_mask_ignores_ineligible_names_when_ranking():
    """An illiquid name must neither be picked nor shift the light/heavy cut.

    D is the lightest name of the four but is ineligible: it must not appear, and
    the cut must be taken over the three eligible names (2 of 3), not four (2 of 4).
    """
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    score = pd.DataFrame([[0.01, 0.02, 0.30, 0.005]], index=idx, columns=list("ABCD"))
    mask = pd.DataFrame([[True, True, True, False]], index=idx, columns=list("ABCD"))
    out = auto_screen.light_capex_mask(score, mask, quantile=0.5)
    assert list(out.columns[out.iloc[0]]) == ["A", "B"]


def test_members_is_the_intersection_of_every_active_filter():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    cols = list("ABC")
    light = pd.DataFrame([[True, True, False]], index=idx, columns=cols)
    profit = pd.DataFrame([[True, False, True]], index=idx, columns=cols)
    vocab = {"A": [2024], "B": [2024], "C": [2024]}
    out = auto_screen.members(idx[0], light=light, profitable=profit, vocabulary=vocab,
                              policy_flagged={})
    assert out == ["A"]


def test_members_drops_policy_flagged_names():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    cols = list("AB")
    ones = pd.DataFrame([[True, True]], index=idx, columns=cols)
    out = auto_screen.members(idx[0], light=ones, profitable=ones,
                              vocabulary={"A": [2024], "B": [2024]},
                              policy_flagged={"B": [2024]})
    assert out == ["A"]


def test_config_vocabulary_is_non_empty_and_quoted():
    """Exact-phrase search needs the quotes; an unquoted term matches any word."""
    assert config.AUTO_CUSTOMER_TERMS
    for term in config.AUTO_CUSTOMER_TERMS + config.AUTO_POLICY_TERMS:
        assert term.startswith('"') and term.endswith('"'), term


def test_vocabulary_members_from_dates_assigns_the_lookback_years(ciks):
    """A 2023 filing puts the name in universe-years 2024 and 2025 (lookback 2)."""
    def fetch(phrase, cik, start, end):
        return ["2023-02-17"] if cik == ciks["ETN"] else []

    out = auto_screen.vocabulary_members_from_dates(
        fetch, ciks, ('"data center"',), range(2019, 2027))
    assert out == {"ETN": [2024, 2025]}


def test_vocabulary_members_from_dates_queries_once_per_filer_and_term(ciks):
    calls = []

    def fetch(phrase, cik, start, end):
        calls.append((phrase, cik))
        return []

    auto_screen.vocabulary_members_from_dates(
        fetch, ciks, ('"data center"', '"substation"'), range(2019, 2027))
    assert len(calls) == 2 * len(ciks)


def test_vocabulary_years_maps_filing_years_to_universe_years():
    hits = pd.DataFrame({"ticker": ["A", "A", "B"],
                         "phrase": ['"data center"', '"substation"', '"investment tax credit"'],
                         "file_date": ["2023-02-17", "2021-02-10", "2023-03-01"]})
    out = auto_screen.vocabulary_years(hits, ('"data center"', '"substation"'), range(2019, 2027))
    assert out == {"A": [2022, 2023, 2024, 2025]}


def test_build_membership_applies_every_filter_and_excludes_power_names():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    cols = ["SUPP", "UTIL", "LOSS", "SUBS"]
    score = pd.DataFrame([[0.01, 0.02, 0.03, 0.04]], index=idx, columns=cols)
    mask = pd.DataFrame([[True] * 4], index=idx, columns=cols)
    panel = pd.DataFrame({"date": list(idx) * 4, "ticker": cols,
                          "net_income": [5.0, 5.0, -1.0, 5.0]})
    hits = pd.DataFrame({
        "ticker": cols + ["SUBS"],
        "phrase": ['"data center"'] * 4 + ['"investment tax credit"'],
        "file_date": ["2023-02-01"] * 5,
    })
    out = auto_screen.build_membership(score, mask, panel, hits, power_tickers={"UTIL"},
                                       quantile=1.0)
    assert list(out.columns[out.loc[idx[0]]]) == ["SUPP"]


# -- Part A breadth levers (handoff 2026-09-23) ---------------------------------

def test_vocabulary_years_honours_an_explicit_lookback():
    """A1: a longer lookback lets one filing cover more universe-years.

    Under the default lookback of 2 a 2023 filing covers 2024-2025 only; at 4 it
    covers 2024-2027, so 2026 joins.
    """
    hits = pd.DataFrame({"ticker": ["A"], "phrase": ['"data center"'],
                         "file_date": ["2023-02-17"]})
    out = auto_screen.vocabulary_years(hits, ('"data center"',), range(2019, 2027),
                                       lookback=4)
    assert out == {"A": [2024, 2025, 2026]}


def test_vocabulary_years_counts_hits_inside_each_universe_years_window():
    """A3: `min_hits` counts matching filings in [Y-lookback, Y-1], not ever.

    B has hits in 2022 and 2023, so universe-year 2024 (window 2022-2023) sees two
    but 2025 (window 2023-2024) sees only one. A has a single hit and never clears.
    """
    hits = pd.DataFrame({
        "ticker": ["A", "B", "B"],
        "phrase": ['"investment tax credit"'] * 3,
        "file_date": ["2023-02-17", "2022-02-10", "2023-02-14"],
    })
    out = auto_screen.vocabulary_years(hits, ('"investment tax credit"',),
                                       range(2019, 2027), min_hits=2)
    assert out == {"B": [2024]}


def test_build_membership_policy_exclusion_can_require_two_hits():
    """A3 end to end: one passing mention no longer discards the name."""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    cols = ["SUPP", "SUBS"]
    score = pd.DataFrame([[0.01, 0.02]], index=idx, columns=cols)
    mask = pd.DataFrame([[True, True]], index=idx, columns=cols)
    panel = pd.DataFrame({"date": list(idx) * 2, "ticker": cols, "net_income": [5.0, 5.0]})
    hits = pd.DataFrame({
        "ticker": cols + ["SUBS"],
        "phrase": ['"data center"'] * 2 + ['"investment tax credit"'],
        "file_date": ["2023-02-01"] * 3,
    })
    out = auto_screen.build_membership(score, mask, panel, hits, quantile=1.0,
                                       policy_min_hits=2)
    assert sorted(out.columns[out.loc[idx[0]]]) == ["SUBS", "SUPP"]


def test_build_membership_honours_an_explicit_lookback():
    """A1 end to end: a 2021 filing decides 2024 only once the lookback reaches 4."""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    cols = ["OLD", "NEW"]
    score = pd.DataFrame([[0.01, 0.02]], index=idx, columns=cols)
    mask = pd.DataFrame([[True, True]], index=idx, columns=cols)
    panel = pd.DataFrame({"date": list(idx) * 2, "ticker": cols, "net_income": [5.0, 5.0]})
    hits = pd.DataFrame({"ticker": ["OLD", "NEW"], "phrase": ['"data center"'] * 2,
                         "file_date": ["2021-02-01", "2023-02-01"]})
    short = auto_screen.build_membership(score, mask, panel, hits, quantile=1.0)
    long = auto_screen.build_membership(score, mask, panel, hits, quantile=1.0, lookback=4)
    assert sorted(short.columns[short.loc[idx[0]]]) == ["NEW"]
    assert sorted(long.columns[long.loc[idx[0]]]) == ["NEW", "OLD"]


def test_config_policy_min_hits_defaults_to_the_shipped_behaviour():
    assert config.AUTO_POLICY_MIN_HITS == 1
