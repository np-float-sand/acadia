import pandas as pd
import pytest

from grid_equipment_basket import basket as bk

_HEADER = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
           "disclosure_type,segment_scope,source_url,notes\n")


def _bdf(rows):
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    txt = _HEADER + "".join(rows)
    p = StringIO(txt)
    return load_backlog_csv(p)


def _yoy_rows(ticker, latest, year_ago):
    qs = ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]
    avs = ["2023-05-01", "2023-08-01", "2023-11-01", "2024-02-01", "2024-05-01"]
    vals = [year_ago, year_ago, year_ago, year_ago, latest]
    return [f"{ticker},{q},{a},{v},USD_million,xbrl_rpo,total,http://x,\n"
            for q, a, v in zip(qs, avs, vals)]


def test_tilt_overweights_top_half_underweights_bottom():
    rows = _yoy_rows("A", 200, 100) + _yoy_rows("B", 150, 100) + _yoy_rows("C", 130, 100) \
        + _yoy_rows("D", 90, 100) + _yoy_rows("E", 80, 100)
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABCDE"), pd.Timestamp("2024-06-01"), bdf,
                                cap=1.0, top=1.25, bottom=0.75)
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] > 0.2 and w["B"] > 0.2
    assert w["D"] < 0.2 and w["E"] < 0.2


def test_tilt_median_name_untilted_on_odd_count():
    rows = _yoy_rows("A", 200, 100) + _yoy_rows("B", 150, 100) + _yoy_rows("C", 130, 100) \
        + _yoy_rows("D", 90, 100) + _yoy_rows("E", 80, 100)
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABCDE"), pd.Timestamp("2024-06-01"), bdf,
                                cap=1.0, top=1.25, bottom=0.75)
    assert w["C"] == pytest.approx(0.2 / (0.2 * (2 * 1.25 + 2 * 0.75 + 1.0)))


def test_tilt_name_without_signal_is_untilted():
    rows = _yoy_rows("A", 200, 100) + _yoy_rows("B", 80, 100)  # C has no rows
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABC"), pd.Timestamp("2024-06-01"), bdf,
                                cap=1.0, top=1.25, bottom=0.75)
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] > w["C"] > w["B"]


def test_tilt_respects_cap():
    rows = _yoy_rows("A", 500, 100) + _yoy_rows("B", 120, 100) + _yoy_rows("C", 80, 100) \
        + _yoy_rows("D", 70, 100)
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABCD"), pd.Timestamp("2024-06-01"), bdf,
                                cap=0.25, top=1.25, bottom=0.75)
    assert w.max() <= 0.25 + 1e-9
    assert w.sum() == pytest.approx(1.0)
