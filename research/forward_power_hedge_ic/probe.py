"""Forward power (STEO) x hedge-adjusted open share -> 21d excess return. Pre-registered in
docs/forward-power-hedge-ic-preregistration.md. Run: poetry run python probe.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

import hedges

HERE = pathlib.Path(__file__).parent
STEO = HERE.parent / "steo_spark_spread" / "data" / "steo_panel.parquet"
WIN = ("2023-01-01", "2026-08-31")
STALE_DAYS, HORIZON, N_BOOT = 130, 21, 2000
HUB_SERIES = {"PJ": "ELWHU_PJ", "TX": "ELWHU_TX"}


def steo_vintages() -> pd.DataFrame:
    p = pd.read_parquet(STEO)
    p["date"] = pd.to_datetime(p.vintage_ts.str[:8])
    p["edition"] = pd.to_datetime(p.forecast_month, format="%B %Y").dt.strftime("%Y%m").astype(int)
    return p[p.period >= p.edition]  # forecast months only


def revision(p: pd.DataFrame, series: str, d_new: pd.Timestamp, d_old: pd.Timestamp, year: int) -> float:
    """% change in mean forecast over year's months present (as forecasts) in both vintages."""
    a = p[(p.date == d_new) & (p.series == series)].set_index("period").value
    b = p[(p.date == d_old) & (p.series == series)].set_index("period").value
    months = [m for m in a.index.intersection(b.index) if m // 100 == year]
    return float(a[months].mean() / b[months].mean() - 1) if months else 0.0


def build_panel() -> pd.DataFrame:
    p = steo_vintages()
    hed = hedges.build()
    dates = sorted(p.date.unique())
    rows = []
    for d_old, d_new in zip(dates[:-1], dates[1:]):
        if not (pd.Timestamp(WIN[0]) <= d_new <= pd.Timestamp(WIN[1])):
            continue
        y0 = d_new.year
        w = {y0: (12 - d_new.month + 1) / 12, y0 + 1: 1.0}
        rev = {(h, y): revision(p, s, d_new, d_old, y) for h, s in HUB_SERIES.items() for y in w}
        gas = sum(wy * revision(p, "NGHHUUS", d_new, d_old, y) for y, wy in w.items())
        for tkr, g in hed.groupby("ticker"):
            filed = g.filed[g.filed <= d_new].max()
            if pd.isna(filed) or (d_new - filed).days > STALE_DAYS:
                continue
            f = g[g.filed == filed].set_index(["year", "hub"]).open_share
            if not all((y, "PJ") in f.index for y in w):
                continue  # hedge % for a needed year not disclosed -> drop pair
            s = s0 = 0.0
            gross = f.groupby("hub").sum() if tkr != "CEG" else None
            for y, wy in w.items():
                for hub in ("PJ", "TX"):
                    if (y, hub) not in f.index:
                        continue
                    s += wy * f[(y, hub)] * rev[(hub, y)]
                    share = _gross_share(tkr, filed, hub)
                    s0 += wy * share * rev[(hub, y)]
            rows.append(dict(ticker=tkr, vintage=d_new, filed=filed, signal=s, naive=s0, gas=gas))
    return pd.DataFrame(rows)


def _gross_share(tkr: str, filed: pd.Timestamp, hub: str) -> float:
    """Comparator h == 0: generation share of the hub, ignoring hedges."""
    if tkr == "CEG":
        regs = hedges._CEG_RAW[filed.strftime("%Y-%m-%d")][2024]
        tot = sum(gw for gw, _ in regs.values())
        return sum(gw for k, (gw, _) in regs.items() if hedges._CEG_HUB[k] == hub) / tot
    return hedges._region(tkr, filed).get(hub, 0.0)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    px = yf.download(["VST", "CEG", "TLN", "SPY"], start="2022-12-01", auto_adjust=True, progress=False)["Close"]
    out = []
    for r in df.itertuples():
        idx = px.index[px.index > r.vintage]
        if len(idx) <= HORIZON:
            out.append(np.nan); continue
        t0, t1 = idx[0], idx[HORIZON]
        ri = px.at[t1, r.ticker] / px.at[t0, r.ticker] - 1
        rm = px.at[t1, "SPY"] / px.at[t0, "SPY"] - 1
        out.append(ri - rm)
    return df.assign(xret=out).dropna(subset=["xret"])


def ic_t(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    rho = stats.spearmanr(x, y).statistic
    n = len(x)
    return rho, rho * np.sqrt(n - 2) / np.sqrt(1 - rho ** 2), n


def partial_ic(df: pd.DataFrame, col: str, ctrl: str) -> tuple[float, float, int]:
    r = df[[col, "xret", ctrl]].rank()
    res = lambda a: a - np.polyval(np.polyfit(r[ctrl], a, 1), r[ctrl])
    return ic_t(res(r[col]), res(r.xret))


def boot_p(df: pd.DataFrame, col: str, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("vintage")]
    obs = stats.spearmanr(df[col], df.xret).statistic
    draws = []
    for _ in range(N_BOOT):
        b = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(stats.spearmanr(b[col], b.xret).statistic)
    draws = np.array(draws)
    return float(2 * min((draws <= 0).mean(), (draws >= 0).mean())), obs


def main() -> None:
    df = add_returns(build_panel())
    df.to_csv(HERE / "panel.csv", index=False)
    print(f"pairs n={len(df)}, vintages={df.vintage.nunique()}, by name={df.ticker.value_counts().to_dict()}")
    print(df.round(4).to_string(index=False))
    print("\n== PRIMARY (hedge-adjusted signal)")
    rho, t, n = ic_t(df.signal, df.xret)
    print(f"rank-IC {rho:+.3f}  t {t:+.2f}  n {n}  -> {'PASS' if abs(t) > 1.5 else 'FAIL'} (|t|>1.5)")
    p, _ = boot_p(df, "signal")
    print(f"vintage-clustered bootstrap p (two-sided, sign) = {p:.3f}")
    rho0, t0, _ = ic_t(df.naive, df.xret)
    print(f"comparator h==0: rank-IC {rho0:+.3f} t {t0:+.2f}")
    print("partial vs gas revision: rank-IC {:+.3f} t {:+.2f}".format(*partial_ic(df, "signal", "gas")[:2]))
    for tk in df.ticker.unique():
        print(f"leave out {tk}: rank-IC {{:+.3f}} t {{:+.2f}} n {{}}".format(*ic_t(*[df[df.ticker != tk][c] for c in ("signal", "xret")])))
        sub = df[df.ticker == tk]
        if len(sub) > 3:
            print(f"   {tk} only: rank-IC {{:+.3f}} t {{:+.2f}} n {{}}".format(*ic_t(sub.signal, sub.xret)))


if __name__ == "__main__":
    main()
