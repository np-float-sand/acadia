"""Every selection variant through the SHIPPED backtest (not the monthly proxy).

    python -m research.xsec_alpha_probe.realbook

Reproduces the PUBLISHED book (output_html/screened-electrification-mechanical:
`screen_auto` with 4-yr vocabulary lookback + 0.60 capex quantile, seed-free prices,
Sharpe 1.05 / Calmar 1.12) from its build cache, then changes only the target
function: which members are held, and at what weight, using each score's latest
month-end value on or before the rebalance date. Path: daily `simulate_basket`
(42-day rebalance, 25% cap, listing gate) -> 20% vol target -> valuation
de-lever -> 15% GLD/IEF sleeve, 2019-01 -> 2026-08.

NB: the package's own `run_comparison` path prices only the 33-name seed +
marquee list, so it silently drops 24 of the 46 names `screen_auto` ever holds.

Also reported: `plain` (vol target only) and the long/short = the base basket
(no overlay) minus GRID at full-sample daily beta.
"""
from __future__ import annotations

import pickle

import pandas as pd

from electrification_strategy import backtest, config, hedge_overlay
from electrification_strategy.fred import fetch_series
from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket.basket import apply_cap, simulate_basket
from grid_equipment_basket.data.prices import fetch_prices
from research.xsec_alpha_probe import data, evaluate as ev, geo, peers, tilt

START, END = "2019-01-01", "2026-08-31"
PUBLISHED = data.data_root() / "output_html" / "img" / "a1a4" / "_build_cache.pkl"


def scores() -> dict[str, pd.DataFrame | None]:
    """Variant name -> monthly score panel (None = equal weight)."""
    univ = data.universe()
    book = tilt.book_membership(univ.index)
    comp = tilt.composite(univ, book)
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(columns=cols, fill_value=False)

    def z(sig):
        return ev.prep(sig.reindex(index=univ.index, columns=cols).where(mask))

    day_m, _ = peers.monthly_returns(*peers.ohlc())
    addr = geo.addresses()
    metro = geo.metro_locations(addr)
    text = pd.read_parquet(data.CACHE / "sig_peer_day_out.parquet")
    city = z(geo.geo_signal(addr, day_m, univ, True))
    city_xo = z(geo.geo_signal(addr, day_m, univ, True, geo.ENERGY_SIC2))
    met = z(geo.geo_signal(metro, day_m, univ, True))
    return {
        "EW (published book)": None,
        "tilt 0.5 by composite": ("tilt", 0.5, comp),
        "tilt 1.0 by composite": ("tilt", 1.0, comp),
        "TOP_HALF composite": ("top", None, comp),
        "TOP_HALF composite + text peers (3a)": ("top", None, comp.add(z(text), fill_value=0) / 2),
        "TOP_HALF composite + city (3b)": ("top", None, comp.add(city, fill_value=0) / 2),
        "TOP_HALF composite + city ex-oil": ("top", None, comp.add(city_xo, fill_value=0) / 2),
        "TOP_HALF composite + metro (3b retry)": ("top", None, comp.add(met, fill_value=0) / 2),
    }


def target_fn(spec, members: pd.DataFrame, px: pd.DataFrame):
    def _fn(available, asof):
        asof = pd.Timestamp(asof)
        upto = members.loc[:asof]
        if upto.empty:
            return pd.Series(dtype=float)
        row = upto.iloc[-1]
        counts = px.loc[:asof].notna().sum()
        listed = set(counts[counts >= config.LISTING_MIN_DAYS].index)
        elig = sorted(set(row.index[row.astype(bool)]) & set(available) & listed)
        if not elig:
            return pd.Series(dtype=float)
        if spec is None:
            raw = pd.Series(1.0, index=elig)
        else:
            kind, k, panel = spec
            up = panel.loc[:asof]
            zz = (up.iloc[-1] if not up.empty else pd.Series(dtype=float)).reindex(elig).fillna(0.0)
            if kind == "tilt":
                raw = (1 + k * zz).clip(lower=0)
            else:
                top = zz.sort_values(ascending=False).index[: max(1, len(zz) // 2)]
                raw = pd.Series(1.0, index=sorted(top))
        return apply_cap(raw / raw.sum(), config.MAX_SINGLE_NAME_WEIGHT)
    return _fn


def main() -> pd.DataFrame:
    built = pickle.loads(PUBLISHED.read_bytes())
    members = built["members"]
    px = data.prices().copy()
    extra = sorted(set(config.BENCHMARKS) | set(config.HEDGE_SLEEVE_TICKERS)
                   | set(config.HEDGE_SHORT_TICKERS) | {"GRID", "SPY"})
    spx = fetch_prices(extra, "2015-01-01", END)
    for t in spx.columns:
        if t not in px.columns:
            px[t] = spx[t].reindex(px.index)
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", END)
    spy_ret = px["SPY"].loc[START:END].pct_change()
    grid_ret = px["GRID"].pct_change()
    sleeve_r = hedge_overlay.sleeve_returns(px)
    short_r = hedge_overlay.short_returns(px)
    cand_prices = px[[c for c in members.columns if c in px.columns]]

    rows = []
    for name, spec in scores().items():
        base = simulate_basket(cand_prices, START, END, config.REBALANCE_LAG_DAYS,
                               config.MAX_SINGLE_NAME_WEIGHT, target_fn(spec, members, px)).returns
        plain = backtest.compose(base, spy_ret, use_valuation=False)
        core = backtest.compose(base, spy_ret, use_valuation=True)
        mask = hedge_overlay.real_yield_rising_mask(dfii10, plain.index)
        shipped = hedge_overlay.apply_hedge(core, sleeve_r, short_r, mask, True, False)
        g = grid_ret.reindex(base.index)
        beta = backtest._beta(base, g)
        ls = (base - beta * g).dropna()
        row = {"variant": name}
        for label, r in (("shipped", shipped), ("plain", plain), ("L/S vs GRID", ls)):
            m = compute_metrics(r.loc[START:END].dropna(), config.RF if label != "L/S vs GRID" else 0.0,
                                config.ANN)
            row[f"{label} SR"] = m["sharpe"]
            row[f"{label} Calmar"] = m["cagr"] / abs(m["max_dd"]) if m["max_dd"] < 0 else float("nan")
            row[f"{label} MaxDD"] = m["max_dd"]
            row[f"{label} CAGR"] = m["cagr"]
            for w, (a, b) in (("19-22", ("2019-01-01", "2022-12-31")), ("23-26", ("2023-01-01", END))):
                row[f"{label} SR {w}"] = compute_metrics(r.loc[a:b].dropna(),
                                                         config.RF if label != "L/S vs GRID" else 0.0,
                                                         config.ANN)["sharpe"]
        row["beta GRID"] = beta
        rows.append(row)
        print(f"  {name}: shipped SR {row['shipped SR']:.3f}", flush=True)
    out = pd.DataFrame(rows).set_index("variant")
    out.to_csv(data.CACHE / "results_realbook.csv")
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 40)
    res = main()
    for label in ("shipped", "plain", "L/S vs GRID"):
        cols = [c for c in res.columns if c.startswith(label)]
        print(f"\n== {label} ==")
        print(res[cols].round(3).to_string())
