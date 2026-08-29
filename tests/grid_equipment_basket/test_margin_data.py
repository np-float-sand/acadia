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
