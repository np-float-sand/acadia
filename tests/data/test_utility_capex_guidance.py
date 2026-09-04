import numpy as np
import pandas as pd
import pytest

from grid_resilience.data import utility_capex_guidance as udg


_FIXTURE_CSV = """utility,report_date,plan_start_year,plan_end_year,capex_plan_usd_m,revision_vs_prior_usd_m,dc_attributed_usd_m,dc_basis,source_type,source_detail,confidence
FIXTURE,2023-01-01,2023,2027,10000,,,none,earnings_call,fixture row - first vintage no prior,H
FIXTURE,2023-07-01,2023,2027,11000,,,none,earnings_call,fixture row - revision derived not stated,H
FIXTURE,2024-01-01,2024,2028,13000,2000,1500,stated,earnings_call,fixture row - stated revision + stated DC $,H
FIXTURE,2024-07-01,2024,2028,13500,500,,qualitative,earnings_call,fixture row - stated revision qualitative DC mention,M
"""


@pytest.fixture
def fixture_csv(tmp_path):
    dst = tmp_path / "fixture.csv"
    dst.write_text(_FIXTURE_CSV)
    return dst


def test_load_capex_guidance_parses_dates_and_sorts(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    assert list(df["utility"].unique()) == ["FIXTURE"]
    assert df["report_date"].is_monotonic_increasing
    assert df["report_date"].dtype.kind == "M"


def test_load_capex_guidance_keeps_stated_revision_as_is(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2024-01-01"].iloc[0]
    assert row["revision_vs_prior_usd_m"] == pytest.approx(2000.0)
    assert row["revision_quality"] == "stated"


def test_load_capex_guidance_derives_blank_revision_from_consecutive_levels(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2023-07-01"].iloc[0]
    # 11000 - 10000 (the prior row's level) = 1000, derived not stated
    assert row["revision_vs_prior_usd_m"] == pytest.approx(1000.0)
    assert row["revision_quality"] == "derived"


def test_load_capex_guidance_first_vintage_has_no_revision(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2023-01-01"].iloc[0]
    assert pd.isna(row["revision_vs_prior_usd_m"])
    assert row["revision_quality"] == "n/a"
    assert pd.isna(row["prior_capex_plan_usd_m"])


def test_load_capex_guidance_prior_capex_plan_is_the_immediately_prior_level(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2024-01-01"].iloc[0]
    assert row["prior_capex_plan_usd_m"] == pytest.approx(11000.0)


def test_feasibility_summary_counts_usable_utilities():
    df = pd.DataFrame({
        "utility":                ["AAA", "AAA", "BBB", "CCC"],
        "report_date": pd.to_datetime(["2022-01-01", "2023-06-01", "2023-01-01", "2020-01-01"]),
        "capex_plan_usd_m":       [1000, 1200, 500, 300],
        "revision_vs_prior_usd_m": [np.nan, 200, np.nan, np.nan],
    })
    feas = udg.feasibility_summary(df, window=("2023-01-01", "2026-08-31"))
    # AAA: has a plan and one in-window revision (2023-06-01) -> usable
    assert feas["per_utility"]["AAA"]["usable"] is True
    # BBB: has a plan but no revision anywhere -> not usable
    assert feas["per_utility"]["BBB"]["usable"] is False
    # CCC: has a plan, and a "revision" but it's before the window (2020) and
    # blank anyway -> not usable
    assert feas["per_utility"]["CCC"]["usable"] is False
    assert feas["n_usable"] == 1
    assert feas["n_total"] == 3
