import pandas as pd
import pytest

from grid_equipment_basket import margin_data as md


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _facts(tag_rows):
    # tag_rows: list of (start, end, filed, val)
    return {"units": {"USD": [
        {"start": s, "end": e, "filed": f, "val": v} for (s, e, f, v) in tag_rows
    ]}}


def test_fetch_fundamentals_keeps_only_quarterly_facts_and_files_dates(monkeypatch):
    revenue = _facts([
        ("2023-01-01", "2023-03-31", "2023-05-01", 1000),   # quarterly - keep
        ("2023-01-01", "2023-12-31", "2024-02-01", 4300),   # annual    - drop
        ("2023-04-01", "2023-06-30", "2023-08-01", 1100),   # quarterly - keep
    ])
    gross = _facts([
        ("2023-01-01", "2023-03-31", "2023-05-01", 300),
        ("2023-04-01", "2023-06-30", "2023-08-01", 350),
    ])

    def fake_get(url, headers=None, timeout=None):
        if "Revenues" in url:
            return _Resp(revenue)
        if "GrossProfit" in url:
            return _Resp(gross)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    df = md.fetch_fundamentals("ETN", use_cache=False)

    assert list(df.columns) == ["quarter_end", "availability_date", "revenue", "gross_profit"]
    assert len(df) == 2
    assert df["quarter_end"].tolist() == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30")]
    assert df["availability_date"].iloc[0] == pd.Timestamp("2023-05-01")
    assert df["gross_profit"].tolist() == [300.0, 350.0]


def test_fetch_fundamentals_derives_gross_profit_from_cost_when_untagged(monkeypatch):
    revenue = _facts([("2023-01-01", "2023-03-31", "2023-05-01", 1000)])
    cogs = _facts([("2023-01-01", "2023-03-31", "2023-05-01", 820)])

    def fake_get(url, headers=None, timeout=None):
        if "Revenues" in url:
            return _Resp(revenue)
        if "GrossProfit" in url:
            return _Resp({"units": {}}, status=404)
        if "CostOfGoodsAndServicesSold" in url:
            return _Resp(cogs)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    df = md.fetch_fundamentals("PWR", use_cache=False)
    assert df["gross_profit"].iloc[0] == pytest.approx(180.0)   # 1000 - 820


def test_fetch_fundamentals_uses_later_cost_filing_date_when_deriving_gross_profit(monkeypatch):
    revenue = _facts([("2023-01-01", "2023-03-31", "2023-05-01", 1000)])
    cogs = _facts([("2023-01-01", "2023-03-31", "2023-06-15", 820)])  # filed later

    def fake_get(url, headers=None, timeout=None):
        if "Revenues" in url:
            return _Resp(revenue)
        if "GrossProfit" in url:
            return _Resp({"units": {}}, status=404)
        if "CostOfGoodsAndServicesSold" in url:
            return _Resp(cogs)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    df = md.fetch_fundamentals("PWR", use_cache=False)
    assert df["availability_date"].iloc[0] == pd.Timestamp("2023-06-15")  # later date
    assert df["gross_profit"].iloc[0] == pytest.approx(180.0)   # 1000 - 820


def test_combine_fundamentals_adds_ticker_column():
    a = pd.DataFrame({"quarter_end": [pd.Timestamp("2023-03-31")],
                      "availability_date": [pd.Timestamp("2023-05-01")],
                      "revenue": [1.0], "gross_profit": [0.3]})
    out = md.combine_fundamentals({"ETN": a, "PWR": a})
    assert set(out["ticker"]) == {"ETN", "PWR"}
    assert len(out) == 2


def _fund_rows(ticker, quarters):
    # quarters: list of (quarter_end, availability_date, revenue, gross_profit)
    return pd.DataFrame(
        [(pd.Timestamp(qe), pd.Timestamp(av), float(rv), float(gp)) for qe, av, rv, gp in quarters],
        columns=["quarter_end", "availability_date", "revenue", "gross_profit"],
    ).assign(ticker=ticker)


def _eight_clean_quarters(ticker, margins):
    # margins: 8 quarterly gross-margin fractions, oldest first. Revenue fixed at 1000.
    qe = ["2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
          "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
    av = ["2022-11-01", "2023-02-01", "2023-05-01", "2023-08-01",
          "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01"]
    return _fund_rows(ticker, [(q, a, 1000.0, 1000.0 * m) for q, a, m in zip(qe, av, margins)])


def test_ttm_gross_margin_signal_yoy_change_in_ttm_margin():
    # prior TTM (q1..q4) avg margin 0.20 ; recent TTM (q5..q8) avg margin 0.25 -> +0.05
    df = _eight_clean_quarters("AAA", [0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25])
    from grid_equipment_basket import margin_data as md
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2024-09-15"))
    assert sig["AAA"] == pytest.approx(0.05)


def test_ttm_gross_margin_signal_needs_eight_quarters():
    df = _eight_clean_quarters("AAA", [0.2] * 8).iloc[:7]
    from grid_equipment_basket import margin_data as md
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2024-09-15"))
    assert "AAA" not in sig.dropna().index


def test_ttm_gross_margin_signal_respects_availability_date():
    df = _eight_clean_quarters("AAA", [0.2] * 8)
    from grid_equipment_basket import margin_data as md
    # asof before the 8th quarter's 2024-08-01 filing -> only 7 visible -> NaN
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2024-07-15"))
    assert "AAA" not in sig.dropna().index


def test_ttm_gross_margin_signal_stale_latest_quarter_is_nan():
    df = _eight_clean_quarters("AAA", [0.2] * 8)
    from grid_equipment_basket import margin_data as md
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2025-06-01"))   # >200d past 2024-06-30
    assert "AAA" not in sig.dropna().index


def test_ttm_revenue_sums_last_four_visible_quarters():
    df = _eight_clean_quarters("AAA", [0.2] * 8)
    from grid_equipment_basket import margin_data as md
    rev = md.ttm_revenue(df, pd.Timestamp("2024-09-15"))
    assert rev["AAA"] == pytest.approx(4000.0)
