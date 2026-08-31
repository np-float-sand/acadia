from __future__ import annotations

"""CLI: python -m backlog_factor --event-study [--start YYYY-MM-DD] [--end YYYY-MM-DD]"""

import argparse
import json
from pathlib import Path

import pandas as pd

from backlog_factor import config, event_study, signal
from backlog_factor.data import prices as px
from backlog_factor.data import rpo as rpo_mod


def _run_event_study(start: str, end: str, out: Path) -> dict:
    uni = config.UNIVERSE
    if not uni:
        raise SystemExit("config.UNIVERSE is empty - run the Task-1 curation step first.")
    group_map = {r["ticker"]: r["industry_group"] for r in uni}
    tickers = [r["ticker"] for r in uni]
    etfs = sorted(set(config.INDUSTRY_GROUP_ETF.values()))

    rpo_panel = rpo_mod.load_rpo_panel(uni, use_cache=True)
    sp = signal.surprise_panel(rpo_panel)
    prices = px.fetch_prices(tickers, start, end, use_cache=True)
    etf_prices = px.fetch_prices(etfs, start, end, use_cache=True)

    pairs = event_study.event_pairs(sp, prices, etf_prices, group_map,
                                    config.INDUSTRY_GROUP_ETF, config.CAR_WINDOWS)
    pairs = pairs.dropna(subset=[f"car_{config.CAR_WINDOWS[-1]}"])
    tbl = event_study.quintile_car_table(pairs, config.CAR_WINDOWS)
    gate = event_study.evaluate_gate(pairs, config.CAR_WINDOWS)

    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "event_pairs.csv", index=False)
    tbl.to_csv(out / "quintile_car_table.csv")
    (out / "gate.json").write_text(json.dumps(gate, indent=2, default=str))

    print(f"universe {len(uni)} names | surprise events {len(sp)} | scored pairs {len(pairs)} "
          f"| names contributing {pairs['ticker'].nunique()}")
    print()
    print(tbl.round(4).to_string())
    print()
    print(f"peak window {gate['peak_window']}d | Q5-Q1 abnormal CAR {gate['q5_q1_car'] * 100:.2f}% "
          f"| clustered t {gate['tstat']:.2f} | monotone {gate['monotone']} "
          f"| groups right sign {gate['n_groups_right_sign']}/5 "
          f"| sub1 {gate['sub1_sign']:+.0f}/{gate['sub1_tstat']:.2f} "
          f"sub2 {gate['sub2_sign']:+.0f}/{gate['sub2_tstat']:.2f}")
    print(f"GATE: {'PASS' if gate['passed'] else 'FAIL'}")
    return gate


def main() -> None:
    ap = argparse.ArgumentParser(prog="backlog_factor")
    ap.add_argument("--event-study", action="store_true")
    ap.add_argument("--start", default=config.STRUCTURED_START)
    ap.add_argument("--end", default=(pd.Timestamp.today().normalize().replace(day=1)
                                      - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    ap.add_argument("--output", default="./output_backlog_factor")
    args = ap.parse_args()
    if not args.event_study:
        raise SystemExit("nothing to do - pass --event-study")
    _run_event_study(args.start, args.end, Path(args.output))


if __name__ == "__main__":
    main()
