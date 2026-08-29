import warnings

import pandas as pd
import pytest

from grid_equipment_basket import basket as bk


# ── apply_cap / equal_weight_targets ────────────────────────────────────────
def test_equal_weight_sums_to_one_and_is_equal():
    w = bk.equal_weight_targets(["C", "A", "B"])
    assert list(w.index) == ["A", "B", "C"]
    assert w.sum() == pytest.approx(1.0)
    assert w.nunique() == 1


def test_equal_weight_cap_not_binding_for_seven_names():
    w = bk.equal_weight_targets([f"T{i}" for i in range(7)], cap=0.25)
    assert w.max() == pytest.approx(1 / 7)


def test_apply_cap_binding_redistributes_pro_rata():
    w = bk.apply_cap(pd.Series({"A": 0.60, "B": 0.20, "C": 0.20}), cap=0.40)
    assert w["A"] == pytest.approx(0.40)
    assert w["B"] == pytest.approx(0.30)
    assert w["C"] == pytest.approx(0.30)
    assert w.sum() == pytest.approx(1.0)


def test_apply_cap_infeasible_returns_equal_and_warns():
    with pytest.warns(RuntimeWarning):
        w = bk.apply_cap(pd.Series({"A": 0.9, "B": 0.05, "C": 0.05}), cap=0.25)
    assert w.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_equal_weight_empty_input():
    assert bk.equal_weight_targets([]).empty


# ── rebalance_dates ─────────────────────────────────────────────────────────
def test_rebalance_dates_are_lagged_quarter_ends():
    dates = bk.rebalance_dates("2023-01-01", "2023-12-31", lag_days=42)
    expected = [
        pd.Timestamp("2022-12-31") + pd.Timedelta(days=42),
        pd.Timestamp("2023-03-31") + pd.Timedelta(days=42),
        pd.Timestamp("2023-06-30") + pd.Timedelta(days=42),
        pd.Timestamp("2023-09-30") + pd.Timedelta(days=42),
    ]
    assert dates == expected


# ── simulate_basket ────────────────────────────────────────────────────────
def test_simulate_two_asset_drift_hand_computed():
    idx = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame({"A": [10.0, 11.0, 11.0], "B": [10.0, 10.0, 12.0]}, index=idx)
    res = bk.simulate_basket(prices, "2023-01-02", "2023-01-04", lag_days=42)
    # shares: 0.5/10 each. day1 value 1.05 -> r=0.05 ; day2 value 1.15 -> r=1.15/1.05-1
    assert res.returns.iloc[0] == pytest.approx(0.05)
    assert res.returns.iloc[1] == pytest.approx(1.15 / 1.05 - 1.0)


def test_simulate_weight_rows_sum_to_one():
    idx = pd.bdate_range("2023-01-02", periods=200)
    prices = pd.DataFrame(
        {t: [100.0 + i for i in range(len(idx))] for t in ["A", "B", "C"]}, index=idx
    )
    res = bk.simulate_basket(prices, "2023-01-02", str(idx[-1].date()))
    assert (res.weights.sum(axis=1) - 1.0).abs().max() < 1e-9
    assert len(res.rebalances) >= 1


def test_simulate_excludes_ticker_until_it_has_a_price():
    idx = pd.bdate_range("2023-01-02", periods=200)
    a = [100.0 + i for i in range(len(idx))]
    b = [float("nan")] * 120 + [50.0 + i for i in range(len(idx) - 120)]
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    res = bk.simulate_basket(prices, "2023-01-02", str(idx[-1].date()))
    first_row = res.weights.iloc[0]
    last_row = res.weights.iloc[-1]
    assert first_row.get("B", 0.0) == 0.0
    assert last_row["B"] > 0.0


def test_simulate_rebalance_day_return_is_not_dropped():
    idx = pd.to_datetime(["2023-05-10", "2023-05-11", "2023-05-12", "2023-05-15", "2023-05-16"])
    prices = pd.DataFrame(
        {"A": [10.0, 10.0, 20.0, 20.0, 20.0], "B": [10.0, 10.0, 10.0, 10.0, 40.0]},
        index=idx,
    )
    res = bk.simulate_basket(prices, "2023-05-10", "2023-05-16", lag_days=42)
    # 2023-05-12 is the Q1 quarter-end (2023-03-31) + 42d rebalance date.
    assert res.rebalances == [pd.Timestamp("2023-05-12")]
    # On the rebalance day the old 50/50 book holds A (10->20) and B (flat):
    # value goes 1.0 -> 1.5, so the return must be +0.5, NOT 0.0.
    assert res.returns.loc["2023-05-12"] == pytest.approx(0.5)
    # After rebalancing into a fresh 50/50 at that day's prices, B quadruples
    # 10->40 on 05-16: half the book *4 -> value 1.5 -> 3.75, return +1.5.
    assert res.returns.loc["2023-05-16"] == pytest.approx(1.5)
