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


# ── coverage_tilt_targets (handoff 2026-09-01 §8.3(a): coverage alone, not
# blended with margin the way value_chain_tilt_targets does) ────────────────

def _fund_df(ticker, quarterly_rev):
    return pd.DataFrame(
        [(pd.Timestamp(qe), pd.Timestamp(av), float(rv), float(rv) * 0.2) for qe, av, rv in quarterly_rev],
        columns=["quarter_end", "availability_date", "revenue", "gross_profit"],
    ).assign(ticker=ticker)


def _coverage_rows(ticker, backlog_year_ago, backlog_now, ttm_rev=1_000_000_000):
    """One name's backlog_df + fund_df rows such that its coverage a year ago was
    backlog_year_ago/ttm_rev and now is backlog_now/ttm_rev (USD_million backlog,
    constant quarterly revenue so TTM = 4x one quarter)."""
    bl = (f"{ticker},2023-03-31,2023-05-01,{backlog_year_ago},USD_million,xbrl_rpo,total,http://x,\n"
         f"{ticker},2024-03-31,2024-05-01,{backlog_now},USD_million,xbrl_rpo,total,http://x,\n")
    q = ttm_rev / 4
    fund = _fund_df(ticker, [
        ("2022-06-30", "2022-08-01", q), ("2022-09-30", "2022-11-01", q),
        ("2022-12-31", "2023-02-01", q), ("2023-03-31", "2023-05-01", q),
        ("2023-06-30", "2023-08-01", q), ("2023-09-30", "2023-11-01", q),
        ("2023-12-31", "2024-02-01", q), ("2024-03-31", "2024-05-01", q),
    ])
    return bl, fund


def _bldf(bl_text):
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    header = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
             "disclosure_type,segment_scope,source_url,notes\n")
    return load_backlog_csv(StringIO(header + bl_text))


def test_coverage_tilt_overweights_rising_coverage_underweights_falling():
    # A: coverage 1.0 -> 2.0 (rising). B: coverage 1.0 -> 0.5 (falling).
    bl_a, fund_a = _coverage_rows("A", 1000, 2000)
    bl_b, fund_b = _coverage_rows("B", 1000, 500)
    bdf = _bldf(bl_a + bl_b)
    fund = pd.concat([fund_a, fund_b], ignore_index=True)
    w = bk.coverage_tilt_targets(["A", "B"], pd.Timestamp("2024-06-01"), fund, bdf,
                                 cap=1.0, top=1.25, bottom=0.75)
    assert w["A"] > w["B"]
    assert w.sum() == pytest.approx(1.0)


def test_coverage_tilt_name_without_signal_is_untilted():
    bl_a, fund_a = _coverage_rows("A", 1000, 2000)
    bdf = _bldf(bl_a)  # C has no backlog rows at all
    fund = pd.concat([fund_a, _fund_df("C", [])], ignore_index=True)
    w = bk.coverage_tilt_targets(["A", "C"], pd.Timestamp("2024-06-01"), fund, bdf,
                                 cap=1.0, top=1.25, bottom=0.75)
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(w["C"])   # only one name has a signal -> no tilt


def test_coverage_tilt_respects_cap():
    bl_a, fund_a = _coverage_rows("A", 1000, 5000)   # sharply rising
    bl_b, fund_b = _coverage_rows("B", 1000, 900)
    bl_c, fund_c = _coverage_rows("C", 1000, 800)
    bdf = _bldf(bl_a + bl_b + bl_c)
    fund = pd.concat([fund_a, fund_b, fund_c], ignore_index=True)
    w = bk.coverage_tilt_targets(["A", "B", "C"], pd.Timestamp("2024-06-01"), fund, bdf,
                                 cap=0.5, top=1.25, bottom=0.75)
    assert w.max() <= 0.5 + 1e-9
    assert w.sum() == pytest.approx(1.0)
