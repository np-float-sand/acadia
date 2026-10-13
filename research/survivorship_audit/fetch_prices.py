"""Fetch Tiingo daily prices for mapped dropouts; cut each series at its exit.

Exit = 8-K/25 event date (+5 trading days) for ACQUIRED/DELISTED/DEREGISTERED; for BANKRUPTCY /
STOPPED FILING, Tiingo's own end (OTC trading after delisting is kept, so the wipe-out is captured).
Tickers already in the production price cache are dropped (subsidiaries mapped to a live parent,
re-registrations such as NorthWestern)."""
import os, pathlib, sys, time, requests, pandas as pd
D = pathlib.Path(__file__).parent; P = D/"_cache"/"tiingo_px"; P.mkdir(parents=True, exist_ok=True)
ROOT = D.parents[1]
for line in (ROOT/".env").read_text().splitlines():
    if line.startswith("TIINGO_API_KEY="): os.environ.setdefault("TIINGO_API_KEY", line.split("=", 1)[1].strip())
H = {"Authorization": f"Token {os.environ['TIINGO_API_KEY']}"}
m = pd.read_csv(D/"ticker_map_final.csv", dtype={"cik": str}).dropna(subset=["ticker"])
c = pd.read_csv(D/"survivorship_candidates.csv", dtype={"cik": str}); c["cik"] = c.cik.str.zfill(10)
m["cik"] = m.cik.str.zfill(10); m = m.merge(c[["cik", "event_date"]], on="cik")
live = set(pd.read_parquet(ROOT/"capex_intensity_strategy/data/cache/prices.parquet").columns)
m = m[~m.ticker.isin(live)].drop_duplicates("ticker")
# Only names still alive after the backtest starts (2019-01) can touch the P&L.
m = m[pd.to_datetime(m[["event_date", "last_10k"]].max(axis=1)) >= "2019-03-01"]
px, dv, kept = {}, {}, []
for _, r in m.iterrows():
    f = P/f"{r.ticker}.parquet"
    if not f.exists():
        for k in range(4):
            resp = requests.get(f"https://api.tiingo.com/tiingo/daily/{r.ticker}/prices",
                                params={"startDate": "2014-01-01", "endDate": "2026-09-30"}, headers=H, timeout=60)
            break
        if resp.status_code == 429: print("rate limited at", r.ticker, "- rerun after the hourly reset", flush=True); break
        if resp.status_code != 200 or not resp.text.startswith("["): print("skip", r.ticker, resp.status_code, resp.text[:80]); continue
        df = pd.DataFrame(resp.json())
        if df.empty: print("empty", r.ticker); continue
        df["date"] = pd.to_datetime(df.date).dt.tz_localize(None); df.set_index("date")[["adjClose", "close", "volume"]].to_parquet(f)
        time.sleep(1.0)
    df = pd.read_parquet(f)
    # Exit cut. A live filer files a 10-K every year, so the company is gone within ~13 months of
    # its last one; past that, a series under the same ticker is a reused symbol (SE->Sea Ltd,
    # B->Barrick, DYN->Dyne). Inside that window, cut at the 8-K / Form 25 exit date if known,
    # and at the first trading gap > 10 calendar days (where a re-used ticker's history starts).
    l10 = pd.Timestamp(r.last_10k); cut = l10 + pd.DateOffset(months=13)
    ev = pd.Timestamp(r.event_date)
    if r.outcome in ("ACQUIRED", "DELISTED", "DEREGISTERED") and l10 - pd.DateOffset(months=3) <= ev <= cut:
        cut = ev + pd.Timedelta(days=7)
    df = df[df.index <= cut]
    gaps = df.index.to_series().diff() > pd.Timedelta(days=10)
    late = gaps[gaps & (df.index > l10 - pd.DateOffset(years=1))]
    if len(late): df = df[df.index < late.index[0]]
    if len(df) < 20: continue
    px[r.ticker] = df.adjClose; dv[r.ticker] = df.close * df.volume; kept.append(r)
pd.DataFrame(px).to_parquet(D/"dropout_prices.parquet"); pd.DataFrame(dv).to_parquet(D/"dropout_dollar_volume.parquet")
k = pd.DataFrame(kept); k.to_csv(D/"dropout_universe.csv", index=False)
print("priced", len(k), k.outcome.value_counts().to_dict())
