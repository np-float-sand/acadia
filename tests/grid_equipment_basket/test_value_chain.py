import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import config, value_chain as vc


def test_buckets_are_the_frozen_split():
    assert sorted(config.BUCKET_MAKERS) == ["ETN", "GEV", "HUBB", "NVT", "VRT"]
    assert sorted(config.BUCKET_CONTRACTORS) == ["FLNC", "MYRG", "PRIM", "PWR"]
    assert set(config.BUCKET_MAKERS) & set(config.BUCKET_CONTRACTORS) == set()


def test_bucket_of_classifies_and_rejects_unknown():
    assert vc.bucket_of("ETN") == "maker"
    assert vc.bucket_of("PWR") == "contractor"
    with pytest.raises(KeyError):
        vc.bucket_of("AAPL")


def test_composite_rank_is_mean_of_ascending_ranks():
    a = pd.Series({"X": 0.05, "Y": -0.02, "Z": 0.10})   # ranks 2, 1, 3
    b = pd.Series({"X": 1.0, "Y": 3.0, "Z": 2.0})        # ranks 1, 3, 2
    out = vc.composite_rank(a, b, ["X", "Y", "Z"])
    assert out["X"] == pytest.approx(1.5)
    assert out["Y"] == pytest.approx(2.0)
    assert out["Z"] == pytest.approx(2.5)


def test_composite_rank_one_missing_component_uses_the_other():
    a = pd.Series({"X": 0.05, "Y": -0.02})               # Z missing from a
    b = pd.Series({"X": 1.0, "Y": 3.0, "Z": 2.0})
    out = vc.composite_rank(a, b, ["X", "Y", "Z"])
    assert not np.isnan(out["Z"])                        # ranked on b alone
    assert np.isnan(vc.composite_rank(pd.Series(dtype=float),
                                     pd.Series(dtype=float), ["Z"])["Z"])


_BL_HEADER = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
              "disclosure_type,segment_scope,source_url,notes\n")


def _backlog_df(rows):
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    return load_backlog_csv(StringIO(_BL_HEADER + "".join(rows)))


def _bl_row(t, qe, av, val):
    return f"{t},{qe},{av},{val},USD_million,xbrl_rpo,total,http://x,\n"


def _fund_df(ticker, quarterly_rev):
    # quarterly_rev: list of (quarter_end, availability_date, revenue)
    return pd.DataFrame(
        [(pd.Timestamp(qe), pd.Timestamp(av), float(rv), float(rv) * 0.2) for qe, av, rv in quarterly_rev],
        columns=["quarter_end", "availability_date", "revenue", "gross_profit"],
    ).assign(ticker=ticker)


def test_coverage_ratio_is_latest_backlog_over_ttm_revenue():
    bl = _backlog_df([
        _bl_row("PWR", "2024-03-31", "2024-05-01", 8000),  # USD_million
    ])
    # TTM revenue needs to be in same scale as backlog (millions) when backlog is in USD_million
    fund = _fund_df("PWR", [
        ("2023-06-30", "2023-08-01", 1000000000), ("2023-09-30", "2023-11-01", 1000000000),
        ("2023-12-31", "2024-02-01", 1000000000), ("2024-03-31", "2024-05-01", 1000000000),
    ])
    out = vc.coverage_ratio(bl, fund, pd.Timestamp("2024-06-01"))
    assert out["PWR"] == pytest.approx(2.0)          # 8000*1e6 / (4*1e9) = 2.0


def test_coverage_change_signal_is_yoy_difference_in_coverage():
    bl = _backlog_df([
        _bl_row("PWR", "2023-03-31", "2023-05-01", 4000),  # USD_million
        _bl_row("PWR", "2024-03-31", "2024-05-01", 8000),  # USD_million
    ])
    # Consistent scale: backlog in USD_million, revenue in full dollars
    fund = _fund_df("PWR", [
        ("2022-06-30", "2022-08-01", 1000000000), ("2022-09-30", "2022-11-01", 1000000000),
        ("2022-12-31", "2023-02-01", 1000000000), ("2023-03-31", "2023-05-01", 1000000000),
        ("2023-06-30", "2023-08-01", 1000000000), ("2023-09-30", "2023-11-01", 1000000000),
        ("2023-12-31", "2024-02-01", 1000000000), ("2024-03-31", "2024-05-01", 1000000000),
    ])
    out = vc.coverage_change_signal(bl, fund, pd.Timestamp("2024-06-01"))
    # coverage now 8000*1e6 / (4*1e9) = 2.0 ; a year earlier 4000*1e6 / (4*1e9) = 1.0 -> +1.0
    assert out["PWR"] == pytest.approx(1.0)


def test_coverage_ratio_normalizes_metric_unit():
    # Same economic coverage (2.0), different units: USD vs USD_million
    # Ticker A: 8_000_000_000 USD / 4_000_000_000 TTM = 2.0
    # Ticker B: 8000 USD_million / 4_000_000_000 TTM = (8000 * 1e6) / 4_000_000_000 = 2.0
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    bl_rows = [
        "A,2024-03-31,2024-05-01,8000000000,USD,xbrl_rpo,total,http://x,\n",
        "B,2024-03-31,2024-05-01,8000,USD_million,xbrl_rpo,total,http://x,\n",
    ]
    bl = load_backlog_csv(StringIO(_BL_HEADER + "".join(bl_rows)))

    fund_a = _fund_df("A", [
        ("2023-06-30", "2023-08-01", 1000000000),
        ("2023-09-30", "2023-11-01", 1000000000),
        ("2023-12-31", "2024-02-01", 1000000000),
        ("2024-03-31", "2024-05-01", 1000000000),
    ])
    fund_b = _fund_df("B", [
        ("2023-06-30", "2023-08-01", 1000000000),
        ("2023-09-30", "2023-11-01", 1000000000),
        ("2023-12-31", "2024-02-01", 1000000000),
        ("2024-03-31", "2024-05-01", 1000000000),
    ])
    fund = pd.concat([fund_a, fund_b], ignore_index=True)

    out = vc.coverage_ratio(bl, fund, pd.Timestamp("2024-06-01"))
    # Both should normalize to 2.0
    assert out["A"] == pytest.approx(2.0)
    assert out["B"] == pytest.approx(2.0)

    # Test unrecognized unit raises ValueError
    bl_bad = load_backlog_csv(StringIO(_BL_HEADER +
        "C,2024-03-31,2024-05-01,8000,USD_billion,xbrl_rpo,total,http://x,\n"))
    fund_c = _fund_df("C", [
        ("2023-06-30", "2023-08-01", 1000000000),
        ("2023-09-30", "2023-11-01", 1000000000),
        ("2023-12-31", "2024-02-01", 1000000000),
        ("2024-03-31", "2024-05-01", 1000000000),
    ])
    with pytest.raises(ValueError, match="unhandled metric_unit"):
        vc.coverage_ratio(bl_bad, fund_c, pd.Timestamp("2024-06-01"))


def test_coverage_ratio_skips_book_to_bill_rows():
    # Ticker has Q1 xbrl_rpo dollar row and later Q2 book_to_bill_only row
    # coverage_ratio should use Q1 dollar figure, not Q2 b2b ratio
    bl = _backlog_df([
        _bl_row("PWR", "2024-03-31", "2024-05-01", 8000),  # USD_million, xbrl_rpo
        f"PWR,2024-06-30,2024-08-01,1.05,ratio,book_to_bill_only,total,http://x,\n",
    ])
    # Consistent scale: backlog in USD_million, revenue in full dollars
    fund = _fund_df("PWR", [
        ("2023-06-30", "2023-08-01", 1000000000), ("2023-09-30", "2023-11-01", 1000000000),
        ("2023-12-31", "2024-02-01", 1000000000), ("2024-03-31", "2024-05-01", 1000000000),
    ])
    out = vc.coverage_ratio(bl, fund, pd.Timestamp("2024-09-01"))
    # Should use Q1 xbrl_rpo dollar row (8000*1e6), not the b2b row
    assert out["PWR"] == pytest.approx(2.0)  # 8000*1e6 / (4*1e9) = 2.0
