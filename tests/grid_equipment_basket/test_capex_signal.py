import numpy as np
import pandas as pd

from grid_equipment_basket import capex_signal as cs


def _capex_df(rows):
    # rows: list of (period 'YYYYQn', capex_$B, yoy, decel2, known 'YYYY-MM-DD')
    idx = pd.PeriodIndex([r[0] for r in rows], freq="Q")
    return pd.DataFrame({"capex": [r[1] * 1e9 for r in rows],
                         "yoy": [r[2] for r in rows],
                         "decel2": [r[3] for r in rows],
                         "known_date": [pd.Timestamp(r[4]) for r in rows]}, index=idx)


def test_derisk_multiplier_is_point_in_time():
    df = _capex_df([
        ("2024Q3", 59, 0.59, 0.06, "2024-11-20"),   # booming
        ("2024Q4", 72, 0.68, 0.09, "2025-02-20"),   # booming
        ("2025Q1", 72, 0.05, -0.20, "2025-05-20"),  # rolls over — but only known 2025-05-20
    ])
    idx = pd.bdate_range("2024-10-01", "2025-06-30")
    m = cs.capex_derisk_multiplier(idx, df, floor_yoy=0.15, require_decel=True, lo_mult=0.5)
    # before the 2025Q1 print is filed, exposure is full even though Q1 was weak
    assert (m.loc["2024-11-25":"2025-05-19"] == 1.0).all()
    # from the known_date onward, de-risked
    assert (m.loc["2025-05-21":] == 0.5).all()


def test_derisk_is_one_directional_never_above_one():
    df = _capex_df([
        ("2025Q1", 72, 0.62, 0.10, "2025-05-20"),   # strong growth, accelerating
        ("2025Q2", 118, 0.66, 0.04, "2025-08-20"),
    ])
    idx = pd.bdate_range("2025-05-01", "2025-12-31")
    m = cs.capex_derisk_multiplier(idx, df, floor_yoy=0.15)
    assert (m <= 1.0 + 1e-9).all()
    assert (m == 1.0).all()          # never triggers when growth is strong


def test_derisk_recovers_when_next_print_clears_the_floor():
    df = _capex_df([
        ("2025Q1", 72, 0.05, -0.20, "2025-05-20"),   # weak -> de-risk
        ("2025Q2", 118, 0.40, 0.35, "2025-08-20"),   # re-accelerates -> back to full
    ])
    idx = pd.bdate_range("2025-05-01", "2025-10-31")
    m = cs.capex_derisk_multiplier(idx, df, floor_yoy=0.15)
    assert (m.loc["2025-05-21":"2025-08-19"] == 0.5).all()
    assert (m.loc["2025-08-21":] == 1.0).all()
