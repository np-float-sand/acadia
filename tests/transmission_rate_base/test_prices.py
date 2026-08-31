import pandas as pd

from transmission_rate_base.data import prices as P


def test_monthly_and_daily_returns_from_stubbed_prices(monkeypatch):
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    fake = pd.DataFrame({"AEP": range(len(idx)), "D": range(len(idx))}, index=idx).astype(float) + 100
    monkeypatch.setattr(P, "_fetch_prices", lambda t, s, e, offline: fake[t])

    dr = P.daily_returns(["AEP"], "2020-01-01", "2020-03-31")
    assert list(dr.columns) == ["AEP"]
    assert dr.notna().all().all()

    mr = P.monthly_returns(["AEP", "D"], "2020-01-01", "2020-03-31")
    assert list(mr.index.month) == [2, 3]
    assert (mr["AEP"] > 0).all()


def test_offline_flag_is_forwarded(monkeypatch):
    seen = {}

    def _spy(t, s, e, offline):
        seen["offline"] = offline
        return pd.DataFrame({"AEP": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    monkeypatch.setattr(P, "_fetch_prices", _spy)
    P.daily_prices(["AEP"], "2020-01-01", "2020-01-02", offline=True)
    assert seen["offline"] is True


def test_missing_tickers_are_dropped(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    fake = pd.DataFrame({"AEP": [1.0, 2, 3, 4, 5]}, index=idx)
    monkeypatch.setattr(P, "_fetch_prices", lambda t, s, e, offline: fake[[c for c in t if c in fake.columns]])
    out = P.daily_prices(["AEP", "MISSING"], "2020-01-01", "2020-01-31")
    assert list(out.columns) == ["AEP"]
