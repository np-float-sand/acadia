from __future__ import annotations

"""CLI: python -m grid_equipment_basket [options]"""

import argparse
from pathlib import Path

import pandas as pd

from grid_equipment_basket import backtest, config


def _default_end() -> str:
    today = pd.Timestamp.today().normalize()
    last_month_end = today.replace(day=1) - pd.Timedelta(days=1)
    return last_month_end.strftime("%Y-%m-%d")


def _backlog_target_fn():
    try:
        from grid_equipment_basket.backlog_data import load_backlog_csv
        from grid_equipment_basket.basket import backlog_tilt_targets
    except ImportError as exc:  # Step 2 not built yet
        raise SystemExit(
            "--tilt backlog requires the Step 2 backlog module, which is only "
            "built if the Step 1 decision gate passed. See "
            "docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md."
        ) from exc
    bdf = load_backlog_csv()

    def _fn(available, asof):
        return backlog_tilt_targets(
            available, asof, bdf,
            cap=config.MAX_SINGLE_NAME_WEIGHT,
            top=config.BACKLOG_TILT_TOP, bottom=config.BACKLOG_TILT_BOTTOM,
        )

    return _fn


def main() -> None:
    ap = argparse.ArgumentParser(prog="grid_equipment_basket")
    ap.add_argument("--start", default=config.PRIMARY_START)
    ap.add_argument("--end", default=_default_end())
    ap.add_argument("--prior-regime", action="store_true",
                    help="run the 2020-2022 panel instead of the primary window")
    ap.add_argument("--tilt", choices=["none", "backlog"], default="none")
    ap.add_argument("--output", default="./output_grid_equipment")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    start, end = args.start, args.end
    if args.prior_regime:
        start, end = config.PRIOR_REGIME_START, config.PRIOR_REGIME_END

    target_fn = _backlog_target_fn() if args.tilt == "backlog" else None

    res = backtest.run(start, end, target_fn=target_fn)
    print(backtest.results_table(res))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"name": "BASKET", **res["basket"]}]
    for b, blk in res["benchmarks"].items():
        rows.append({"name": b, **blk["metrics"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    res["basket_returns"].rename("basket_return").to_csv(out / "basket_returns.csv")
    if not args.no_plot:
        backtest.plot(res, str(out / "performance.png"))
        print(f"\nwrote {out}/metrics.csv, basket_returns.csv, performance.png")


if __name__ == "__main__":
    main()
