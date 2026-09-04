import numpy as np
import pandas as pd
import pytest

from grid_resilience.data import utility_capex_guidance as udg


@pytest.fixture
def fixture_csv(tmp_path):
    src = udg.SEED_CSV
    dst = tmp_path / "fixture.csv"
    dst.write_text(src.read_text())
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
