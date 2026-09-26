"""CLI: python -m capex_intensity_strategy [options]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from capex_intensity_strategy import backtest, config, pipeline, plots


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="capex_intensity_strategy")
    ap.add_argument("--start", default=config.START_DEFAULT)
    ap.add_argument("--end", default=None)
    ap.add_argument("--short-weight", type=float, default=None,
                    help=f"short leg size (default {config.SHORT_WEIGHT}); 0 = long only")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch EDGAR and prices instead of using the cache")
    ap.add_argument("--output", default="./output_capex_intensity")
    ap.add_argument("--plot", action="store_true",
                    help="write comparison charts against the cheap alternatives")
    args = ap.parse_args(argv)

    result = pipeline.build(start=args.start, end=args.end, refresh=args.refresh,
                            short_weight=args.short_weight)

    sw = config.SHORT_WEIGHT if args.short_weight is None else args.short_weight
    print(backtest.format_summary(result["summary"],
                                  f"Capex-intensity L/S (short weight {sw:.2f})"))
    diag = result["diagnostics"]
    print(f"\navg long {diag['avg_long_names']:.0f} names / short "
          f"{diag['avg_short_names']:.0f} names   turnover "
          f"{diag['annual_turnover']:.1f}x/yr")

    for name, summary in result["benchmarks"].items():
        print("\n" + backtest.format_summary(summary, f"benchmark: {name}"))

    # The final month-end can sit past the last priced day, leaving it unscored.
    # Report the most recent month that actually has positions.
    weights = result["weights"]
    held = weights.index[(weights != 0).any(axis=1)]
    if len(held):
        latest = weights.loc[held[-1]]
        longs = sorted(latest[latest > 0].index)
        shorts = sorted(latest[latest < 0].index)
        print(f"\ncurrent book ({held[-1]:%Y-%m-%d})")
        print(f"  LONG  ({len(longs)}): {', '.join(longs)}")
        print(f"  SHORT ({len(shorts)}): {', '.join(shorts)}")
    else:
        print("\ncurrent book: no scored names in the final period")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(out / "summary.csv", index=False)
    result["net"].rename("ret").to_frame().to_csv(out / "daily_returns.csv")
    result["weights"].to_csv(out / "weights.csv")
    pd.Series(diag).to_csv(out / "diagnostics.csv")
    print(f"\nwrote {out}/summary.csv, daily_returns.csv, weights.csv, diagnostics.csv")

    if args.plot:
        figs = plots.make_figures(result["net"], result["prices"], out)
        for name, path in figs.items():
            print(f"  chart {name}  ->  {path}")
        stats = pd.read_csv(out / "stats_long_history.csv", index_col=0)
        print("\n" + plots.format_table(stats, "Versus the cheaper alternatives"))


if __name__ == "__main__":
    main()
