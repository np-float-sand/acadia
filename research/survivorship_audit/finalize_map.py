"""Combine 10-K-extracted symbols + hand map; accept a ticker only if Tiingo's endDate is in
[last_10k - 1y, last_10k + 2y] and the ticker is no longer active."""
import pathlib, pandas as pd
D = pathlib.Path(__file__).parent
m = pd.read_csv(D/"ticker_map.csv", dtype={"cik": str})
c = pd.read_csv(D/"survivorship_candidates.csv", dtype={"cik": str}); c["cik"] = c.cik.str.zfill(10)
m["cik"] = m.cik.str.zfill(10)
hand = pd.read_csv(D/"hand_tickers.csv", comment="#", sep="|")
tg = pd.read_parquet(D/"_cache"/"tiingo_supported.parquet")
tg = tg[(tg.assetType == "Stock") & (tg.priceCurrency == "USD")].copy()
tg["ticker"] = tg.ticker.str.upper(); tg["endDate"] = pd.to_datetime(tg.endDate, errors="coerce")
x = m.merge(c[["cik", "last_10k", "max_float_musd", "sic", "event_date"]], on="cik")
x = x.merge(hand.rename(columns={"ticker": "hand"}), on="name", how="left")
rows = []
for _, r in x.iterrows():
    lo, hi = pd.Timestamp(r.last_10k) - pd.DateOffset(years=2), pd.Timestamp(r.last_10k) + pd.DateOffset(years=2)
    cands = ([r.hand] if isinstance(r.hand, str) else []) + (r.symbols.split(",") if isinstance(r.symbols, str) else [])
    pick = None
    for s in dict.fromkeys(cands):
        h = tg[(tg.ticker == s) & tg.endDate.between(lo, hi) & (tg.endDate < "2026-06-01")]
        if len(h): pick = (s, h.endDate.max().date(), h.startDate.min(), "ended"); break
        # Stale 'active' rows: same listing must predate the last 10-K by a year; truncated at exit later.
        h = tg[(tg.ticker == s) & (tg.endDate >= "2026-06-01") & (pd.to_datetime(tg.startDate) <= lo + pd.DateOffset(years=1))]
        if len(h): pick = (s, None, h.startDate.min(), "truncate"); break
    rows.append(dict(cik=r.cik, name=r["name"], outcome=r.outcome, sic=r.sic, last_10k=r.last_10k,
                     max_float_musd=r.max_float_musd, tried=",".join(dict.fromkeys(cands)),
                     ticker=pick[0] if pick else None, tiingo_end=pick[1] if pick else None,
                     tiingo_start=pick[2] if pick else None, mode=pick[3] if pick else None))
out = pd.DataFrame(rows); out.to_csv(D/"ticker_map_final.csv", index=False)
ok = out[out.ticker.notna()]
print("mapped", len(ok), "of", len(out)); print(ok.outcome.value_counts().to_dict())
print("unmapped with float>=100M:"); print(out[out.ticker.isna() & (out.max_float_musd >= 100)][["name", "outcome", "tried"]].to_string(index=False))
