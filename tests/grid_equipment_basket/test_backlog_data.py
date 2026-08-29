import pandas as pd
import pytest

from grid_equipment_basket import backlog_data as bd

_HEADER = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
           "disclosure_type,segment_scope,source_url,notes\n")


def _write(tmp_path, body: str):
    p = tmp_path / "backlog_quarterly.csv"
    p.write_text(_HEADER + body)
    return str(p)


def test_load_backlog_csv_parses_dates(tmp_path):
    path = _write(tmp_path,
        "PWR,2024-03-31,2024-05-02,30000,USD_million,xbrl_rpo,total,http://x,\n")
    df = bd.load_backlog_csv(path)
    assert pd.api.types.is_datetime64_any_dtype(df["quarter_end"])
    assert pd.api.types.is_datetime64_any_dtype(df["availability_date"])


def test_load_backlog_csv_rejects_bad_disclosure_type(tmp_path):
    path = _write(tmp_path,
        "PWR,2024-03-31,2024-05-02,30000,USD_million,guesstimate,total,http://x,\n")
    with pytest.raises(ValueError):
        bd.load_backlog_csv(path)


def test_backlog_growth_signal_yoy(tmp_path):
    body = "".join(
        f"AAA,{qe},{av},{val},USD_million,xbrl_rpo,total,http://x,\n"
        for qe, av, val in [
            ("2023-03-31", "2023-05-01", 100),
            ("2023-06-30", "2023-08-01", 105),
            ("2023-09-30", "2023-11-01", 110),
            ("2023-12-31", "2024-02-01", 120),
            ("2024-03-31", "2024-05-01", 130),
        ]
    )
    df = bd.load_backlog_csv(_write(tmp_path, body))
    sig = bd.backlog_growth_signal(df, pd.Timestamp("2024-06-01"))
    assert sig["AAA"] == pytest.approx(0.30)


def test_backlog_growth_signal_gappy_span_returns_nan(tmp_path):
    # 5 rows, but iloc[-5] -> iloc[-1] spans ~30 months (>430 days) because the
    # series is gappy. Positional iloc[-5] would inflate the "YoY" ratio, so the
    # lookback-span guard must NaN this ticker instead.
    body = "".join(
        f"GAP,{qe},{av},{val},USD_million,xbrl_rpo,total,http://x,\n"
        for qe, av, val in [
            ("2022-12-31", "2023-02-15", 100),
            ("2023-06-30", "2023-08-15", 110),
            ("2024-06-30", "2024-08-15", 150),
            ("2024-12-31", "2025-02-15", 170),
            ("2025-06-30", "2025-08-15", 200),
        ]
    )
    df = bd.load_backlog_csv(_write(tmp_path, body))
    sig = bd.backlog_growth_signal(df, pd.Timestamp("2025-09-01"))
    assert "GAP" not in sig.dropna().index


def test_backlog_growth_signal_respects_availability_date(tmp_path):
    body = "".join(
        f"AAA,{qe},{av},{val},USD_million,xbrl_rpo,total,http://x,\n"
        for qe, av, val in [
            ("2023-03-31", "2023-05-01", 100),
            ("2024-03-31", "2024-05-01", 130),
        ]
    )
    df = bd.load_backlog_csv(_write(tmp_path, body))
    sig = bd.backlog_growth_signal(df, pd.Timestamp("2024-04-15"))  # before the 2024-05 filing
    assert "AAA" not in sig.dropna().index


def test_backlog_growth_signal_book_to_bill_level(tmp_path):
    body = ("BBB,2024-03-31,2024-05-01,1.20,ratio,book_to_bill_only,total,http://x,\n")
    df = bd.load_backlog_csv(_write(tmp_path, body))
    sig = bd.backlog_growth_signal(df, pd.Timestamp("2024-06-01"))
    assert sig["BBB"] == pytest.approx(0.20)


def test_backlog_ranks_ascending_higher_signal_higher_rank():
    sig = pd.Series({"A": -0.1, "B": 0.4, "C": 0.05, "D": float("nan")})
    ranks = bd.backlog_ranks(sig)
    assert ranks["B"] == 3
    assert ranks["A"] == 1
    assert pd.isna(ranks["D"])
