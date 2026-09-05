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


def test_panel_asof_returns_latest_row_per_utility_not_after_asof():
    df = pd.DataFrame({
        "utility":     ["AAA", "AAA", "BBB"],
        "report_date": pd.to_datetime(["2022-01-01", "2023-01-01", "2023-06-01"]),
        "capex_plan_usd_m": [1000, 1200, 500],
    })
    out = udg.panel_asof(df, "2023-03-01")
    # AAA's 2023-01-01 row is the latest not after asof; BBB has none yet
    assert set(out["utility"]) == {"AAA"}
    assert out.iloc[0]["report_date"] == pd.Timestamp("2023-01-01")


def test_panel_asof_empty_when_nothing_known_yet():
    df = pd.DataFrame({
        "utility": ["AAA"], "report_date": pd.to_datetime(["2023-01-01"]),
        "capex_plan_usd_m": [1000],
    })
    out = udg.panel_asof(df, "2020-01-01")
    assert out.empty


def test_impute_dc_attributed_keeps_stated_rows_unchanged():
    df = pd.DataFrame({
        "utility": ["AAA"], "report_date": pd.to_datetime(["2023-01-01"]),
        "revision_vs_prior_usd_m": [1000.0], "dc_attributed_usd_m": [500.0],
        "dc_basis": ["stated"],
    })
    out = udg.impute_dc_attributed(df)
    assert out.iloc[0]["dc_attributed_usd_m_filled"] == pytest.approx(500.0)
    assert out.iloc[0]["dc_imputed"] == False


def test_impute_dc_attributed_fills_qualitative_row_from_prior_observed_ratio():
    df = pd.DataFrame({
        "utility":     ["AAA", "BBB"],
        "report_date": pd.to_datetime(["2023-01-01", "2023-06-01"]),
        "revision_vs_prior_usd_m": [1000.0, 800.0],
        "dc_attributed_usd_m":     [500.0, np.nan],
        "dc_basis":                ["stated", "qualitative"],
    })
    out = udg.impute_dc_attributed(df)
    bbb = out[out["utility"] == "BBB"].iloc[0]
    # ratio from AAA alone = 500/1000 = 0.5; BBB's revision 800 * 0.5 = 400
    assert bbb["dc_attributed_usd_m_filled"] == pytest.approx(400.0)
    assert bbb["dc_imputed"] == True


def test_impute_dc_attributed_is_point_in_time_safe():
    base_rows = [
        {"utility": "AAA", "report_date": "2023-01-01", "revision_vs_prior_usd_m": 1000.0,
         "dc_attributed_usd_m": 500.0, "dc_basis": "stated"},
        {"utility": "BBB", "report_date": "2023-06-01", "revision_vs_prior_usd_m": 800.0,
         "dc_attributed_usd_m": np.nan, "dc_basis": "qualitative"},
    ]
    later_row = {"utility": "CCC", "report_date": "2024-01-01", "revision_vs_prior_usd_m": 2000.0,
                "dc_attributed_usd_m": 100.0, "dc_basis": "stated"}

    df_without = pd.DataFrame(base_rows)
    df_without["report_date"] = pd.to_datetime(df_without["report_date"])
    out_without = udg.impute_dc_attributed(df_without)

    df_with = pd.DataFrame(base_rows + [later_row])
    df_with["report_date"] = pd.to_datetime(df_with["report_date"])
    out_with = udg.impute_dc_attributed(df_with)

    v_without = out_without.loc[out_without["utility"] == "BBB", "dc_attributed_usd_m_filled"].iloc[0]
    v_with = out_with.loc[out_with["utility"] == "BBB", "dc_attributed_usd_m_filled"].iloc[0]
    # a LATER observation (CCC, 2024) must not change BBB's already-computed
    # (2023-06) imputed value -- that would be a future-information leak.
    assert v_without == pytest.approx(v_with)
    assert v_without == pytest.approx(400.0)


def test_impute_dc_attributed_before_any_observation_stays_nan():
    rows = [
        {"utility": "AAA", "report_date": "2021-01-01", "revision_vs_prior_usd_m": 500.0,
         "dc_attributed_usd_m": np.nan, "dc_basis": "none"},
        {"utility": "AAA", "report_date": "2023-01-01", "revision_vs_prior_usd_m": 1000.0,
         "dc_attributed_usd_m": 600.0, "dc_basis": "stated"},
    ]
    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    out = udg.impute_dc_attributed(df)
    first = out.iloc[0]
    assert pd.isna(first["dc_attributed_usd_m_filled"])
    assert first["dc_imputed"] == False


def test_impute_dc_attributed_converges_within_max_iter():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        stated = i % 3 == 0
        rows.append({
            "utility": f"U{i % 5}",
            "report_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=30 * i),
            "revision_vs_prior_usd_m": float(rng.uniform(100, 1000)),
            "dc_attributed_usd_m": float(rng.uniform(50, 400)) if stated else np.nan,
            "dc_basis": "stated" if stated else "qualitative",
        })
    df = pd.DataFrame(rows)
    out5 = udg.impute_dc_attributed(df, max_iter=5)
    out20 = udg.impute_dc_attributed(df, max_iter=20)
    assert out5["dc_attributed_usd_m_filled"].notna().sum() > 0
    pd.testing.assert_series_equal(
        out5["dc_attributed_usd_m_filled"].fillna(-1.0),
        out20["dc_attributed_usd_m_filled"].fillna(-1.0),
        check_exact=False, atol=1.0, rtol=0.02)


def test_impute_dc_attributed_same_day_ties_are_deterministic():
    aaa_stated = {"utility": "AAA", "report_date": "2023-06-01",
                  "revision_vs_prior_usd_m": 1000.0, "dc_attributed_usd_m": 500.0,
                  "dc_basis": "stated"}
    bbb_qualitative = {"utility": "BBB", "report_date": "2023-06-01",
                       "revision_vs_prior_usd_m": 800.0, "dc_attributed_usd_m": np.nan,
                       "dc_basis": "qualitative"}

    df_ab = pd.DataFrame([aaa_stated, bbb_qualitative])
    df_ab["report_date"] = pd.to_datetime(df_ab["report_date"])
    out_ab = udg.impute_dc_attributed(df_ab)

    df_ba = pd.DataFrame([bbb_qualitative, aaa_stated])
    df_ba["report_date"] = pd.to_datetime(df_ba["report_date"])
    out_ba = udg.impute_dc_attributed(df_ba)

    v_ab = out_ab.loc[out_ab["utility"] == "BBB", "dc_attributed_usd_m_filled"].iloc[0]
    v_ba = out_ba.loc[out_ba["utility"] == "BBB", "dc_attributed_usd_m_filled"].iloc[0]
    # same report_date for AAA (stated) and BBB (qualitative) -- the imputed
    # value for BBB must not depend on which order the two rows appeared in
    # the input DataFrame.
    assert v_ab == pytest.approx(v_ba)
    assert v_ab == pytest.approx(400.0)


def test_aggregate_revision_series_usd_is_trailing_sum_within_window():
    df = pd.DataFrame({
        "utility":     ["AAA", "BBB", "AAA"],
        "report_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-11-01"]),
        "revision_vs_prior_usd_m": [100.0, 50.0, 200.0],
    })
    s = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m", ttm_quarters=4)
    # at 2023-02-01, trailing 4Q (~365 days) window includes both prior events -> 150
    assert s.loc[pd.Timestamp("2023-02-01")] == pytest.approx(150.0)
    # at 2023-11-01, the Jan event (>365 days back is not yet true, ~304 days,
    # still inside the window) plus itself -> all three sum to 350
    assert s.loc[pd.Timestamp("2023-11-01")] == pytest.approx(350.0)


def test_aggregate_revision_series_drops_out_after_ttm_window():
    df = pd.DataFrame({
        "utility":     ["AAA", "AAA"],
        "report_date": pd.to_datetime(["2022-01-01", "2023-06-01"]),
        "revision_vs_prior_usd_m": [100.0, 50.0],
    })
    s = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m", ttm_quarters=4)
    # by 2023-06-01 the 2022-01-01 event is >365 days old -> only the new one counts
    assert s.loc[pd.Timestamp("2023-06-01")] == pytest.approx(50.0)


def test_aggregate_revision_series_pct_is_size_weighted_not_mean_of_percents():
    df = pd.DataFrame({
        "utility":     ["AAA", "BBB"],
        "report_date": pd.to_datetime(["2023-01-01", "2023-01-15"]),
        "revision_vs_prior_usd_m": [100.0, 10.0],
        "prior_capex_plan_usd_m":  [1000.0, 20.0],   # AAA 10%, BBB 50%
    })
    s = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m",
                                      denom_col="prior_capex_plan_usd_m", ttm_quarters=4)
    # size-weighted: (100+10)/(1000+20) = 0.1078..., NOT mean(10%, 50%) = 30%
    assert s.loc[pd.Timestamp("2023-01-15")] == pytest.approx(110.0 / 1020.0)


def test_aggregate_revision_series_pct_drops_rows_with_a_null_denominator():
    # AAA has a revision but NO prior plan level (a first-vintage row that
    # nonetheless carried a stated revision -- 6 such rows exist in the live
    # seed panel). It must be dropped from BOTH sums. Counting it in the
    # numerator against a 0 denominator contribution inflated the ratio by a
    # time-varying factor and manufactured a spurious downtrend.
    df = pd.DataFrame({
        "utility":     ["AAA", "BBB", "CCC"],
        "report_date": pd.to_datetime(["2023-01-01", "2023-01-10", "2023-01-20"]),
        "revision_vs_prior_usd_m": [900.0, 100.0, 10.0],
        "prior_capex_plan_usd_m":  [np.nan, 1000.0, 20.0],
    })
    s = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m",
                                      denom_col="prior_capex_plan_usd_m", ttm_quarters=4)
    # only BBB + CCC contribute: (100+10)/(1000+20), NOT (900+100+10)/(1000+20)
    assert s.loc[pd.Timestamp("2023-01-20")] == pytest.approx(110.0 / 1020.0)
    # AAA's date is not even an event date for the pct series
    assert pd.Timestamp("2023-01-01") not in s.index
    # ... while the $ series is unaffected -- it has no denominator to be null
    s_usd = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m",
                                          ttm_quarters=4)
    assert s_usd.loc[pd.Timestamp("2023-01-20")] == pytest.approx(1010.0)


def test_aggregate_revision_series_drops_rows_with_missing_value():
    df = pd.DataFrame({
        "utility": ["AAA", "BBB"], "report_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "dc_attributed_usd_m": [np.nan, 40.0],
    })
    s = udg.aggregate_revision_series(df, value_col="dc_attributed_usd_m", ttm_quarters=4)
    assert list(s.index) == [pd.Timestamp("2023-01-02")]
    assert s.iloc[0] == pytest.approx(40.0)
