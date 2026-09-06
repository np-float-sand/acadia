"""CLI: python -m electrification_strategy [options]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grid_equipment_basket.data.prices import fetch_prices

from electrification_strategy import backtest, config, universe
from electrification_strategy.fred import fetch_series


def _default_end() -> str:
    today = pd.Timestamp.today().normalize()
    return (today.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="electrification_strategy")
    ap.add_argument("--start", default=config.START_DEFAULT)
    ap.add_argument("--end", default=_default_end())
    ap.add_argument("--universe", choices=["marquee", "frozen", "thematic", "screen", "all"], default="all")
    ap.add_argument("--output", default="./output_electrification")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    cons = universe.CONSTRUCTIONS if args.universe == "all" else (args.universe,)

    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    tickers = sorted(set(cand) | set(config.BENCHMARKS) | set(config.HEDGE_SLEEVE_TICKERS)
                     | set(config.HEDGE_SHORT_TICKERS) | {config.FEARED_PROXY})

    prices = fetch_prices(tickers, args.start, args.end)
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", args.end)

    result = backtest.run_comparison(args.start, args.end, prices, dfii10, constructions=cons)
    print(backtest.comparison_table(result))

    win = backtest.pick_winner(result)
    print(f"\nWINNER (pre-registered rule): {win['winner']}   runner-up: {win['runner_up']}   "
          f"({win['n_passing']}/{len(result['cells'])} cells pass)")

    out = Path(args.output)
    backtest.write_outputs(result, win, out, make_plot=not args.no_plot)
    extra = "" if args.no_plot else ", performance.png"
    print(f"\nwrote {out}/metrics.csv, episode_drawdowns.csv, plateau.json, returns.csv{extra}")


if __name__ == "__main__":
    main()
