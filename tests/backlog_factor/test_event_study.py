import numpy as np
import pandas as pd
import pytest

from backlog_factor import event_study as es


def test_event_pairs_day0_is_first_trading_day_after_filing():
    px = pd.DataFrame(
        {"A": [10, 11, 12, 13, 14], "ETF": [10, 10, 10, 10, 10]},
        index=pd.to_datetime(["2023-05-10", "2023-05-11", "2023-05-12", "2023-05-15", "2023-05-16"]),
    )
    sp = pd.DataFrame({"ticker": ["A"], "availability_date": [pd.Timestamp("2023-05-11")], "surprise": [1.0]})
    pairs = es.event_pairs(sp, px[["A"]], px[["ETF"]], {"A": "m"}, {"m": "ETF"}, car_windows=[2])
    # day 0 = 2023-05-12 (first bday strictly after 05-11); measure the CAR over the next 2 trading days:
    # close(12)->close(13) + close(13)->close(14), abnormal (ETF flat) so same as raw
    assert pairs["car_2"].iloc[0] == pytest.approx((13 / 12 - 1) + (14 / 13 - 1))


def _monotone_panel(seed=1):
    rng = np.random.default_rng(seed)
    groups = ["machinery", "electrical_equipment", "aerospace_defense",
              "engineering_construction", "building_products"]
    rows = []
    for m in pd.period_range("2020-01", "2023-12", freq="M"):
        for k in range(50):
            s = rng.normal()
            car = 0.02 * s + rng.normal(0, 0.01)
            rows.append({"ticker": f"T{k}", "availability_date": m.to_timestamp(), "month": m,
                         "industry_group": groups[k % 5], "surprise": s,
                         "car_21": car, "car_63": car * 1.1})
    return pd.DataFrame(rows)


def test_gate_passes_on_monotone_panel():
    g = es.evaluate_gate(_monotone_panel(), car_windows=[21, 63])
    assert g["passed"] is True
    assert g["monotone"] is True
    assert g["tstat"] > 2
    assert g["n_groups_right_sign"] >= 3
    assert g["drift_horizon_days"] in (21, 63)


def test_clustered_tstat_equals_monthly_spread_mean_over_se():
    # three months, 5 names each with surprise 1..5 -> qcut puts each in its own quintile.
    # per-month Q5-Q1 car spread = [0.10, 0.02, 0.06]
    rows = []
    for m, q5_car in [("2021-01", 0.10), ("2021-02", 0.02), ("2021-03", 0.06)]:
        for s, car in zip([1, 2, 3, 4, 5], [0.0, 0.0, 0.0, 0.0, q5_car]):
            rows.append({"month": pd.Period(m), "surprise": float(s), "car_21": car})
    got = es.clustered_tstat(pd.DataFrame(rows), "car_21")
    spr = pd.Series([0.10, 0.02, 0.06])
    assert got == pytest.approx(spr.mean() / (spr.std() / np.sqrt(3)))


def test_gate_fails_on_noise_panel():
    rng = np.random.default_rng(2)
    rows = [{"ticker": f"T{k}", "availability_date": m.to_timestamp(), "month": m,
             "industry_group": "machinery", "surprise": rng.normal(),
             "car_21": rng.normal(0, 0.05), "car_63": rng.normal(0, 0.05)}
            for m in pd.period_range("2020-01", "2023-12", freq="M") for k in range(30)]
    g = es.evaluate_gate(pd.DataFrame(rows), car_windows=[21, 63])
    assert g["passed"] is False


def test_monotonic_flag():
    assert es.monotonic([1, 2, 3, 4, 5]) is True
    assert es.monotonic([5, 4, 3, 2, 1]) is True
    assert es.monotonic([1, 3, 2, 4, 5]) is False
