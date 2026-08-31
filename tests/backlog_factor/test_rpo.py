import pandas as pd
import pytest

from backlog_factor.data import rpo


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_discover_rpo_filers_unions_frames_and_filters_by_sic(monkeypatch):
    frame_q4_2023 = {"data": [
        {"cik": 1050915, "entityName": "QUANTA SERVICES INC", "val": 1},
        {"cik": 320193, "entityName": "APPLE INC", "val": 1},
    ]}
    frame_q4_2024 = {"data": [
        {"cik": 1050915, "entityName": "QUANTA SERVICES INC", "val": 1},
        {"cik": 37996, "entityName": "FORD MOTOR CO", "val": 1},
    ]}
    subs = {
        "0001050915": {"sic": "1731", "sicDescription": "Electrical Work", "tickers": ["PWR"], "name": "Quanta Services"},
        "0000320193": {"sic": "3571", "sicDescription": "Electronic Computers", "tickers": ["AAPL"], "name": "Apple"},
        "0000037996": {"sic": "3711", "sicDescription": "Motor Vehicles", "tickers": ["F"], "name": "Ford"},
    }

    def fake_get(url, headers=None, timeout=None):
        if "CY2023Q4I" in url:
            return _Resp(frame_q4_2023)
        if "CY2024Q4I" in url:
            return _Resp(frame_q4_2024)
        for cik10, body in subs.items():
            if f"CIK{cik10}" in url:
                return _Resp(body)
        return _Resp({}, status=404)

    monkeypatch.setattr(rpo.requests, "get", fake_get)
    out = rpo.discover_rpo_filers(["CY2023Q4I", "CY2024Q4I"], use_cache=False)

    assert set(out["ticker"]) == {"PWR", "F"}          # AAPL (SIC 3571) filtered out; PWR + F kept
    assert out.loc[out["ticker"] == "PWR", "cik"].iloc[0] == "0001050915"   # zero-padded to 10
    assert "AAPL" not in set(out["ticker"])


def _facts(rows):   # rows: list of (end, filed, val)
    return {"units": {"USD": [{"end": e, "filed": f, "val": v} for e, f, v in rows]}}


def test_fetch_rpo_parses_and_files_dates(monkeypatch):
    payload = _facts([
        ("2023-03-31", "2023-05-02", 3.0e10),
        ("2023-03-31", "2023-05-09", 3.1e10),   # later-filed dup for the same quarter -> dropped
        ("2023-06-30", "2023-08-01", 3.2e10),
    ])

    def fake_get(url, headers=None, timeout=None):
        if "RevenueRemainingPerformanceObligation.json" in url and "Current" not in url and "Noncurrent" not in url:
            return _Resp(payload)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(rpo.requests, "get", fake_get)

    df = rpo.fetch_rpo("PWR", "0001050915", use_cache=False)
    assert list(df.columns) == ["quarter_end", "availability_date", "metric_value"]
    assert df["quarter_end"].tolist() == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30")]
    assert df["availability_date"].iloc[0] == pd.Timestamp("2023-05-02")   # earliest filing wins
    assert df["metric_value"].iloc[0] == pytest.approx(3.0e10)


def test_fetch_rpo_merges_current_and_noncurrent_when_base_missing(monkeypatch):
    cur = _facts([("2023-03-31", "2023-05-02", 2.0e10)])
    non = _facts([("2023-03-31", "2023-05-02", 1.0e10)])

    def fake_get(url, headers=None, timeout=None):
        if "ObligationCurrent.json" in url:
            return _Resp(cur)
        if "ObligationNoncurrent.json" in url:
            return _Resp(non)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(rpo.requests, "get", fake_get)
    df = rpo.fetch_rpo("X", "0000000001", use_cache=False)
    assert df["metric_value"].iloc[0] == pytest.approx(3.0e10)


def test_load_rpo_panel_adds_ticker_and_concats(monkeypatch):
    def fake_fetch(ticker, cik, use_cache=True):
        return pd.DataFrame({
            "quarter_end": [pd.Timestamp("2023-03-31")],
            "availability_date": [pd.Timestamp("2023-05-01")],
            "metric_value": [1.0e10],
        })

    monkeypatch.setattr(rpo, "fetch_rpo", fake_fetch)
    panel = rpo.load_rpo_panel(
        [{"ticker": "A", "cik": "1"}, {"ticker": "B", "cik": "2"}], use_cache=False
    )
    assert set(panel["ticker"]) == {"A", "B"}
    assert list(panel.columns) == ["quarter_end", "availability_date", "metric_value", "ticker"]
