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


def _q(qe):
    # ~90-day span ending at quarter_end qe
    end = pd.Timestamp(qe)
    start = (end - pd.Timedelta(days=89)).strftime("%Y-%m-%d")
    return start, qe


def test_fetch_fundamentals_unions_revenue_tags_and_does_not_shadow_modern_tag(monkeypatch):
    # `Revenues` returns 3 stale pre-ASC-606 facts; the modern
    # `RevenueFromContractWithCustomerExcludingAssessedTax` returns 10 recent facts.
    # fetch_fundamentals must return the UNION (>=10 rows, recent range), deduped, sorted.
    old_qe = ["2016-12-31", "2017-03-31", "2017-06-30"]
    modern_qe = ["2017-06-30",  # overlaps the old tag -> dedupe, earliest filing wins
                 "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30",
                 "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"]

    def _rows(qes, filed_offset_days, base):
        out = []
        for i, qe in enumerate(qes):
            s, e = _q(qe)
            filed = (pd.Timestamp(qe) + pd.Timedelta(days=filed_offset_days)).strftime("%Y-%m-%d")
            out.append((s, e, filed, base + i))
        return out

    revenues = _facts(_rows(old_qe, 40, 900))
    # the overlapping 2017-06-30 fact is filed LATER in the modern tag -> the old
    # tag's earlier filing must win after dedupe.
    modern = _facts(_rows(modern_qe, 300, 1000))
    gross = _facts(_rows(old_qe + modern_qe[1:], 40, 200))

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/Revenues.json"):
            return _Resp(revenues)
        if "RevenueFromContractWithCustomerExcludingAssessedTax" in url:
            return _Resp(modern)
        if "GrossProfit" in url:
            return _Resp(gross)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    df = md.fetch_fundamentals("MYRG", use_cache=False)

    assert len(df) >= 10
    assert df["quarter_end"].is_monotonic_increasing
    assert df["quarter_end"].duplicated().sum() == 0
    # modern-only quarter present -> the modern tag was reached, not shadowed
    assert pd.Timestamp("2025-09-30") in set(df["quarter_end"])
    # old-only quarter present -> the union kept the stale tag too
    assert pd.Timestamp("2016-12-31") in set(df["quarter_end"])
    assert df["quarter_end"].max() == pd.Timestamp("2025-09-30")


def test_fetch_fundamentals_caches_empty_result_as_sentinel(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    monkeypatch.setattr(md, "CACHE_DIR", tmp_path)

    df1 = md.fetch_fundamentals("HUBB")   # use_cache defaults True
    assert df1.empty
    n_after_first = len(calls)
    assert n_after_first > 0
    assert (tmp_path / "fundamentals_HUBB.parquet").exists()   # sentinel written

    df2 = md.fetch_fundamentals("HUBB")
    assert df2.empty
    assert len(calls) == n_after_first   # sentinel hit -> no new SEC requests


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


def test_ttm_gross_margin_signal_recent_revenue_zero_is_nan_no_warning():
    import warnings
    # All 8 quarters visible, but the most recent 4 have revenue=0
    qe = ["2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
          "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
    av = ["2022-11-01", "2023-02-01", "2023-05-01", "2023-08-01",
          "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01"]
    quarters = [(qe[i], av[i], 1000.0, 200.0) if i < 4 else (qe[i], av[i], 0.0, 0.0)
                for i in range(8)]
    df = _fund_rows("AAA", quarters)
    from grid_equipment_basket import margin_data as md
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2024-09-15"))
    assert "AAA" not in sig.dropna().index


def test_ttm_gross_margin_signal_span_out_of_range_is_nan():
    # 8 clean quarters but quarter_end[-1]..quarter_end[-5] span > 430 days (gappy series)
    qe = ["2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
          "2023-12-31", "2024-06-30", "2024-09-30", "2024-12-31"]  # gappy: 2023-09-30 skipped
    av = ["2022-11-01", "2023-02-01", "2023-05-01", "2023-08-01",
          "2024-02-01", "2024-08-01", "2024-11-01", "2025-02-01"]
    df = _fund_rows("AAA", [(q, a, 1000.0, 200.0) for q, a in zip(qe, av)])
    from grid_equipment_basket import margin_data as md
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2025-03-15"))
    # quarter_end[-1] = 2024-12-31, quarter_end[-5] = 2023-12-31 = 365 days, out of range
    assert "AAA" not in sig.dropna().index
