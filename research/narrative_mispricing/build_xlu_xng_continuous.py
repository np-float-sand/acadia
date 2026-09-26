"""Rebuild the XLU/XNG correlation spread as a continuous monthly feature
(item 3 fix #1 only saved the boolean exclude mask; need the raw spread for ML use)."""
import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/sandhyapersad/acadia")
from grid_resilience.data.universe import UNIVERSE

CORR_WINDOW = 63
tickers = list(UNIVERSE)
px = yf.download(tickers + ["XLU", "^XNG"], start="2016-06-01", end="2026-08-01",
                  auto_adjust=True, progress=False)["Close"]
ret = px.pct_change()

corr_xlu = pd.DataFrame({t: ret[t].rolling(CORR_WINDOW).corr(ret["XLU"]) for t in tickers})
corr_xng = pd.DataFrame({t: ret[t].rolling(CORR_WINDOW).corr(ret["^XNG"]) for t in tickers})
spread = corr_xng - corr_xlu  # positive = trading more like gas than utilities

monthly_dates = pd.date_range("2018-01-01", "2025-09-01", freq="MS")
spread_monthly = spread.reindex(spread.index.union(monthly_dates)).sort_index().ffill().reindex(monthly_dates)
spread_monthly.to_parquet("research/narrative_mispricing/xlu_xng_spread_continuous.parquet")
print(spread_monthly.describe().T[["mean","std","min","max"]])
