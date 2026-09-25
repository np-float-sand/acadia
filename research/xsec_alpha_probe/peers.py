"""Plan step 3a: text-similar peer momentum.

    python -m research.xsec_alpha_probe.peers

Links (Hoberg-Phillips style, rebuilt from free EDGAR text): at each month-end t,
take every firm's latest 10-K Item 1 filed on or before t (and within 18 months;
filings where Item 1 could not be located are dropped -- ~12% of filings, 109 firms),
TF-IDF them (English stop words dropped, words in >25% of documents dropped,
sublinear tf), and link each eligible-universe follower to its K=10 highest-cosine
leaders. Pool = eligible-universe names ever + current S&P 500.

Signal (fixed 2026-09-24, before results) = cosine-weighted mean of the leaders'
returns over month t, predicting the follower's return in month t+1:
  peer_day_out  PRIMARY  leaders OUTSIDE the month's eligible universe, daytime
                         (open->close, compounded) returns -- overnight reverses
  peer_day_all           leaders from the whole pool (standard TNIC momentum)
  peer_cc_out            outside leaders, close-to-close returns
Same bar as run.py (NW |t| >= 1.5 pooled and both halves, same sign). Then the
benchmark test: does the top half of the book by mean z(composite, peer_day_out)
beat TOP_HALF by composite alone (GRID-hedged Sharpe 0.81, Calmar 0.41)?
Caveat: the S&P 500 pool is today's list (survivor-selected leaders).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer

from research.xsec_alpha_probe import data, evaluate as ev, run, tenk, tilt

K = 10
STALE = pd.DateOffset(months=18)
MIN_CHARS = 2_000


def ohlc(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    op, cl = data.CACHE / "pool_open.parquet", data.CACHE / "pool_close.parquet"
    if op.exists() and cl.exists() and not refresh:
        return pd.read_parquet(op), pd.read_parquet(cl)
    tickers = sorted(tenk.firm_ciks())
    raw = yf.download(tickers, start="2017-06-01", end="2026-09-01", auto_adjust=True,
                      progress=False, threads=True)
    raw["Open"].to_parquet(op)
    raw["Close"].to_parquet(cl)
    return raw["Open"], raw["Close"]


def monthly_returns(open_: pd.DataFrame, close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = close / open_
    day = day.where((day > 0.5) & (day < 2.0))                          # drop bad prints
    day_m = np.exp(np.log(day).resample("ME").sum(min_count=10)) - 1
    cc_m = close.resample("ME").last().pct_change(fill_method=None)
    return day_m, cc_m


def links(dates: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.DataFrame]:
    """month-end -> cosine similarity matrix (ticker x ticker) from docs available then."""
    idx = tenk.index()
    # whole-10-K fallbacks match each other on legal boilerplate, not products: drop them
    idx = idx[(idx["chars"] >= MIN_CHARS) & ~idx["full_text"]]
    text = {p: p.read_text() for p in idx["path"]}
    out, prev_key = {}, None
    for dt in dates:
        avail = idx[(idx["filed"] <= dt) & (idx["filed"] > dt - STALE)]
        latest = avail.sort_values("filed").groupby("ticker").tail(1).sort_values("ticker")
        key = tuple(latest["path"])
        if key == prev_key:
            out[dt] = out[prev_dt]
            continue
        vec = TfidfVectorizer(stop_words="english", max_df=0.25, min_df=2, sublinear_tf=True,
                              token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        m = vec.fit_transform([text[p] for p in latest["path"]])
        sim = (m @ m.T).toarray()
        np.fill_diagonal(sim, 0.0)
        out[dt] = pd.DataFrame(sim, index=latest["ticker"], columns=latest["ticker"])
        prev_key, prev_dt = key, dt
    return out


def peer_signal(sims: dict, ret_m: pd.DataFrame, univ: pd.DataFrame, outside: bool) -> pd.DataFrame:
    rows = {}
    for dt, sim in sims.items():
        if dt not in univ.index or dt not in ret_m.index:
            continue
        inside = set(univ.columns[univ.loc[dt]])
        r = ret_m.loc[dt]
        leaders = [t for t in sim.columns if pd.notna(r.get(t)) and (not outside or t not in inside)]
        row = {}
        for f in inside & set(sim.index):
            s = sim.loc[f, [t for t in leaders if t != f]].nlargest(K)
            s = s[s > 0]
            if len(s):
                row[f] = float((s * r[s.index]).sum() / s.sum())
        rows[dt] = row
    return pd.DataFrame(rows).T.sort_index()


def main() -> None:
    pd.set_option("display.width", 250, "display.max_columns", 40)
    px_d = data.prices()
    univ = data.universe()
    px_m = data.monthly_prices(px_d)
    fwd1 = px_m.pct_change(fill_method=None).shift(-1)
    fwd3 = px_m.shift(-3) / px_m - 1
    grid_fwd = data.monthly_prices(data.benchmarks())["GRID"].pct_change().shift(-1)
    day_m, cc_m = monthly_returns(*ohlc())
    sims = links(univ.loc[run.START:run.END].index)
    print("pool firms/month:", min(len(s) for s in sims.values()), "-", max(len(s) for s in sims.values()))
    sigs = {"peer_day_out": peer_signal(sims, day_m, univ, True),
            "peer_day_all": peer_signal(sims, day_m, univ, False),
            "peer_cc_out": peer_signal(sims, cc_m, univ, True)}
    for k, v in sigs.items():
        v.to_parquet(data.CACHE / f"sig_{k}.parquet")
    risks = {"eq": None}
    res = pd.DataFrame([run.evaluate_signal(k, v, univ, fwd1, fwd3, risks, grid_fwd)
                        for k, v in sigs.items()]).set_index("signal")
    res.to_csv(data.CACHE / "results_peers.csv")
    run._print(res)

    # benchmark test inside the book
    book = tilt.book_membership(univ.index)
    comp = tilt.composite(univ, book)
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(columns=cols, fill_value=False)
    peer_z = ev.prep(sigs["peer_day_out"].reindex(index=univ.index, columns=cols).where(mask))
    combo = comp.add(peer_z, fill_value=0) / 2
    rows = []
    for name, score in {"TOP_HALF composite (benchmark)": comp,
                        "TOP_HALF peer_day_out alone": peer_z,
                        "TOP_HALF composite + peer": combo}.items():
        br = tilt.book_returns(score)
        hedged, beta = tilt.grid_hedge(br["TOP_HALF"], br["GRID"])
        for row in tilt.stats(br["TOP_HALF"], hedged):
            rows.append({"book": name, **row})
    bench = pd.DataFrame(rows).set_index(["book", "window"])
    bench.to_csv(data.CACHE / "results_peers_book.csv")
    print(bench[["SR", "ann", "maxDD", "Calmar", "hedged SR", "alpha ann", "hedged Calmar",
                 "alpha t (NW1)", "label"]].round(2).to_string())


if __name__ == "__main__":
    main()
