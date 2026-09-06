import numpy as np
import pandas as pd
import pytest

_TICKERS = [
    "ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM",
    "EMR", "AME", "RRX", "POWL", "ATKR", "AEIS", "GNRC", "AYI", "EME", "FIX", "MTZ",
    "WCC", "TT", "CARR", "JCI", "SPXC", "MOD", "IESC", "APG", "ENS", "THR",
    "GLD", "IEF", "DLR", "EQIX", "SPY", "XLI", "PAVE", "GRID", "VOLT", "TAN",
]


@pytest.fixture
def synthetic_prices():
    idx = pd.bdate_range("2017-06-01", "2026-08-31")
    rng = np.random.default_rng(0)
    out = {}
    for i, t in enumerate(_TICKERS):
        mu = 0.0003 + 0.00002 * (i % 5)
        steps = rng.normal(mu, 0.015, len(idx))
        series = 100.0 * np.exp(np.cumsum(steps))
        s = pd.Series(series, index=idx, name=t)
        if t == "GEV":
            s.iloc[:1700] = np.nan
        out[t] = s
    return pd.DataFrame(out)


@pytest.fixture
def synthetic_dfii10():
    idx = pd.bdate_range("2003-01-01", "2026-09-03")
    x = np.linspace(0, 12 * np.pi, len(idx))
    return pd.Series(1.5 + 1.0 * np.sin(x), index=idx, name="DFII10")
