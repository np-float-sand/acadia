"""
Runnable script -- satellite-sleeve diversification test (2026-09-18).

Question: does adding a fixed-weight, equal-weight, non-timed satellite sleeve
of crypto-miner and/or merchant-generator stocks to the shipped
`electrification_strategy` book raise combined Sharpe/Calmar or lower MaxDD,
purely through low correlation -- independent of whether the sleeve has any
predictive signal of its own (that question was already asked and FAILED, see
docs/electricity-price-volatility-miner-results.md).

Method mirrors electrification_strategy's own conventions exactly (see
sleeve.py's docstring): equal-weight-within-sleeve daily returns, blended into
the fully-composed base-book return series (`screen +val+sleeve` primary,
`screen_reit +val+sleeve` secondary) at a FIXED daily weight,
`(1-w)*book + w*sleeve`. Does NOT modify the shipped electrification_strategy
package.

Usage:
    python3 -m research.satellite_sleeve_diversification.run_probe
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grid_equipment_basket.backtest import compute_metrics

from electrification_strategy import backtest, config, universe as ustrat
from electrification_strategy.fred import fetch_series

from research.satellite_sleeve_diversification import data as D
from research.satellite_sleeve_diversification import sleeve as S
from research.satellite_sleeve_diversification import universe as U

END = "2026-09-17"
OUT_DIR = Path(__file__).resolve().parent / "_output"
OUT_DIR.mkdir(exist_ok=True)


def _calmar(cagr: float, max_dd: float) -> float:
    if cagr != cagr or max_dd != max_dd or max_dd == 0:
        return float("nan")
    return float(cagr / abs(max_dd))


def _corr(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=False).dropna()
    return float(d["a"].corr(d["b"])) if len(d) > 2 else float("nan")


def _metrics_row(r: pd.Series) -> dict:
    m = compute_metrics(r, config.RF, config.ANN)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
            "calmar": _calmar(m["cagr"], m["max_dd"]), "n_days": m["n_obs"]}


def build_price_panel() -> tuple[pd.DataFrame, pd.Series]:
    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(ustrat.load_seed().index))
    base_tickers = sorted(set(cand) | set(config.BENCHMARKS) | set(config.HEDGE_SLEEVE_TICKERS)
                          | set(config.HEDGE_SHORT_TICKERS) | {config.FEARED_PROXY})
    tickers = sorted(set(base_tickers) | set(U.COMBINED))
    prices = D.fetch_all(tickers, config.START_DEFAULT, END)
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", END)
    return prices, dfii10


def run() -> dict:
    prices, dfii10 = build_price_panel()
    print(f"price panel: {prices.shape[0]} days x {prices.shape[1]} tickers, "
          f"{prices.index.min().date()} -> {prices.index.max().date()}")

    base_constructions = sorted({con for con, _ in U.BASE_CELLS})
    result = backtest.run_comparison(config.START_DEFAULT, END, prices, dfii10,
                                     constructions=base_constructions)
    book_returns = {cell: result["cells"][cell]["returns"] for cell in U.BASE_CELLS}

    rows = []
    sleeve_windows = {}
    for sleeve_name, tickers in U.SLEEVES.items():
        common_start = D.sleeve_common_start(prices, tickers)
        if common_start is None:
            missing = [t for t in tickers if t not in prices.columns]
            raise RuntimeError(f"{sleeve_name}: missing price data for {missing}")

        full_start = max(pd.Timestamp(config.START_DEFAULT), common_start)
        recent_start = max(pd.Timestamp(U.RECENT_WINDOW_START), common_start)
        windows = {
            "full_available_sample": (full_start, pd.Timestamp(END)),
            "2023-01-01_to_present": (recent_start, pd.Timestamp(END)),
        }
        sleeve_windows[sleeve_name] = {
            "common_start": str(common_start.date()),
            "per_ticker_effective_start": {
                t: str(D.effective_start(prices, t).date()) for t in tickers
            },
            "windows": {k: (str(a.date()), str(b.date())) for k, (a, b) in windows.items()},
        }
        raw_sleeve_r = S.sleeve_returns(prices, tickers)

        for cell in U.BASE_CELLS:
            book_r_full = book_returns[cell]
            for window_name, (wstart, wend) in windows.items():
                book_r = book_r_full.loc[wstart:wend]
                sleeve_r = raw_sleeve_r.loc[wstart:wend]
                book_m = _metrics_row(book_r)
                corr = _corr(sleeve_r, book_r)
                sleeve_m = _metrics_row(sleeve_r)
                for w in U.WEIGHTS:
                    combined_r = S.blend(book_r, sleeve_r, w)
                    comb_m = _metrics_row(combined_r)
                    rows.append({
                        "sleeve": sleeve_name, "weight": w,
                        "base_cell": f"{cell[0]} {cell[1]}", "window": window_name,
                        "window_start": str(wstart.date()), "window_end": str(wend.date()),
                        "book_cagr": book_m["cagr"], "book_sharpe": book_m["sharpe"],
                        "book_max_dd": book_m["max_dd"], "book_calmar": book_m["calmar"],
                        "sleeve_cagr": sleeve_m["cagr"], "sleeve_sharpe": sleeve_m["sharpe"],
                        "sleeve_max_dd": sleeve_m["max_dd"],
                        "corr_sleeve_book": corr,
                        "combined_cagr": comb_m["cagr"], "combined_sharpe": comb_m["sharpe"],
                        "combined_max_dd": comb_m["max_dd"], "combined_calmar": comb_m["calmar"],
                        "d_sharpe": (comb_m["sharpe"] - book_m["sharpe"])
                                    if comb_m["sharpe"] == comb_m["sharpe"] and book_m["sharpe"] == book_m["sharpe"]
                                    else float("nan"),
                        "d_max_dd": (comb_m["max_dd"] - book_m["max_dd"])
                                    if comb_m["max_dd"] == comb_m["max_dd"] and book_m["max_dd"] == book_m["max_dd"]
                                    else float("nan"),
                        "d_calmar": (comb_m["calmar"] - book_m["calmar"])
                                    if comb_m["calmar"] == comb_m["calmar"] and book_m["calmar"] == book_m["calmar"]
                                    else float("nan"),
                        "n_days": comb_m["n_days"],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "combination_results.csv", index=False)
    (OUT_DIR / "sleeve_windows.json").write_text(json.dumps(sleeve_windows, indent=2))
    print(f"wrote {OUT_DIR / 'combination_results.csv'}, {OUT_DIR / 'sleeve_windows.json'}")
    return {"table": df, "sleeve_windows": sleeve_windows}


def print_summary(out: dict) -> None:
    df = out["table"]
    print("\n=== sleeve effective windows (post SPAC/reverse-merger truncation) ===")
    for name, info in out["sleeve_windows"].items():
        print(f"  {name}: common_start={info['common_start']}  windows={info['windows']}")

    print("\n=== best combined Sharpe per (sleeve, base_cell, window) ===")
    for (sleeve_name, cell, window), g in df.groupby(["sleeve", "base_cell", "window"]):
        best = g.loc[g["combined_sharpe"].idxmax()]
        book_sharpe = g["book_sharpe"].iloc[0]
        print(f"  {sleeve_name:<11} {cell:<22} {window:<24} "
              f"book Sharpe={book_sharpe:.3f}  best combined @ w={best['weight']:.2f} "
              f"Sharpe={best['combined_sharpe']:.3f} (d={best['d_sharpe']:+.3f})  "
              f"MaxDD {best['combined_max_dd']*100:.1f}% (d={best['d_max_dd']*100:+.1f}pp)  "
              f"corr={best['corr_sleeve_book']:.2f}")


if __name__ == "__main__":
    out = run()
    print_summary(out)
