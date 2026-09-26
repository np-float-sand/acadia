import numpy as np
import pandas as pd
import pytest

# Asset-light suppliers (low capex/revenue) and asset-heavy owners (high).
LIGHT = ["ETN", "POWL", "ATKR", "MYRG", "EME", "NVT"]
HEAVY = ["DUK", "PCG", "ETR", "AEP", "XEL", "SO"]
FUEL = ["CCJ", "LEU"]          # lexicon-discovered, join partway through
ALL = LIGHT + HEAVY + FUEL


@pytest.fixture
def dates():
    return pd.date_range("2019-01-31", "2024-12-31", freq="ME")


@pytest.fixture
def synthetic_prices():
    idx = pd.bdate_range("2018-01-01", "2024-12-31")
    rng = np.random.default_rng(7)
    data = {}
    for i, t in enumerate(ALL):
        steps = rng.normal(0.0004, 0.014, len(idx))
        data[t] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def synthetic_dollar_volume(synthetic_prices):
    dv = pd.DataFrame(50e6, index=synthetic_prices.index, columns=synthetic_prices.columns)
    dv["MYRG"] = 1e6          # persistently illiquid -> must never be eligible
    return dv


@pytest.fixture
def synthetic_facts():
    """Quarterly revenue/capex with a realistic filing lag.

    Light names spend ~3% of revenue on capex, heavy names ~25%. Each quarter is
    filed 45 days after period end, so point-in-time logic has something to bite.
    """
    rows = []
    for ticker in ALL:
        intensity = 0.03 if ticker in LIGHT + FUEL else 0.25
        for period in pd.period_range("2017Q1", "2024Q3", freq="Q"):
            start, end = period.start_time, period.end_time.normalize()
            revenue = 1_000_000.0
            rows.append((ticker, "revenue", start, end, end + pd.Timedelta(days=45), revenue))
            rows.append((ticker, "capex", start, end, end + pd.Timedelta(days=45),
                         revenue * intensity))
    return pd.DataFrame(rows, columns=["ticker", "item", "start", "end", "filed", "value"])
