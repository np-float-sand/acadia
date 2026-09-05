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
    ap.add_argument("--construction",
                    choices=["equal-weight", "backlog-tilt", "value-chain-tilt", "pair"],
                    default="equal-weight")
    ap.add_argument("--tilt", choices=["none", "backlog"], default=None,
                    help="deprecated alias: --tilt backlog == --construction backlog-tilt")
    ap.add_argument("--drop-winners", action="store_true",
                    help="value-chain constructions only: drop VRT, GEV from the makers bucket")
    ap.add_argument("--overlay", action="store_true",
                    help="also print the layer-1 risk overlay (trend gate + vol target) report")
    ap.add_argument("--overlay-l2", dest="overlay_l2", action="store_true",
                    help="also run the layer-2 grid-congestion regime ladder (both windows) "
                         "and write regime_metrics.csv / regime_timeline.csv")
    ap.add_argument("--capex-guidance", dest="capex_guidance", action="store_true",
                    help="also run the utility capex-guidance revision signal report "
                         "(Deliverable D) and write capex_guidance_timing.csv / "
                         "capex_guidance_gates.json")
    ap.add_argument("--output", default="./output_grid_equipment")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    construction = args.construction
    if args.tilt == "backlog":
        construction = "backlog-tilt"

    start, end = args.start, args.end
    if args.prior_regime:
        start, end = config.PRIOR_REGIME_START, config.PRIOR_REGIME_END

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if construction in ("value-chain-tilt", "pair"):
        rep = backtest.value_chain_report(start, end, drop_winners=args.drop_winners)
        print(backtest.value_chain_table(rep))
        _write_value_chain_outputs(rep, out)
        return

    target_fn = _backlog_target_fn() if construction == "backlog-tilt" else None
    res = backtest.run(start, end, target_fn=target_fn)
    print(backtest.results_table(res))

    if args.overlay:
        from grid_equipment_basket import overlay
        orep = overlay.overlay_report(start, end)
        print("\n" + overlay.overlay_table(orep))
        orep_pr = overlay.overlay_report(start, end, prior_regime=True)
        print("\n" + overlay.overlay_table(orep_pr))

    if args.overlay_l2:
        from grid_equipment_basket import grid_regime
        rrep = grid_regime.regime_report()
        print("\n" + grid_regime.regime_table(rrep))
        _write_regime_outputs(rrep, out)

    if args.capex_guidance:
        from grid_equipment_basket import capex_guidance_signal
        grep = capex_guidance_signal.guidance_signal_report()
        print("\n" + capex_guidance_signal.guidance_table(grep))
        _write_guidance_outputs(grep, out)

    rows = [{"name": "BASKET", **res["basket"]}]
    for b, blk in res["benchmarks"].items():
        rows.append({"name": b, **blk["metrics"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    res["basket_returns"].rename("basket_return").to_csv(out / "basket_returns.csv")
    if not args.no_plot:
        backtest.plot(res, str(out / "performance.png"))
        print(f"\nwrote {out}/metrics.csv, basket_returns.csv, performance.png")


def _write_regime_outputs(rep: dict, out: Path) -> None:
    rows: list[dict] = []
    for key, blk in rep["baselines"].items():
        for wk, wblk in blk.items():
            rows.append({"name": key, "window": wk, "verdict": "baseline", **wblk["metrics"]})
    timeline: dict = {}
    for r in rep["rungs"]:
        if r.get("status") == "deferred":
            rows.append({"name": r["name"], "window": "-", "verdict": "deferred"})
            continue
        for wk, wblk in r["block"].items():
            rows.append({"name": r["name"], "window": wk, "verdict": r["verdict"],
                         **wblk["metrics"], **{f"gate_{k}": v for k, v in r["gate"].items()
                                               if k in ("G1", "G2", "G3", "marginal")}})
        if "multiplier" in r:
            timeline[r["name"]] = r["multiplier"]
    pd.DataFrame(rows).to_csv(out / "regime_metrics.csv", index=False)
    pd.DataFrame(timeline).to_csv(out / "regime_timeline.csv")
    print(f"\nwrote {out}/regime_metrics.csv, regime_timeline.csv")


def _write_guidance_outputs(rep: dict, out: Path) -> None:
    import json

    if rep.get("universe_used") == "not_testable":
        (out / "capex_guidance_gates.json").write_text(json.dumps(rep, default=str, indent=2))
        print(f"\nwrote {out}/capex_guidance_gates.json (not yet testable)")
        return

    rows = []
    for name, by_h in rep["timing_basket"].items():
        for h, cell in by_h.items():
            rows.append({
                "series": name, "horizon_months": h,
                "rank_ic": cell["rank_ic_primary"]["ic"], "rank_ic_t": cell["rank_ic_primary"]["t"],
                "control_t": cell["control_without_hyperscaler"]["t"].get("signal"),
                "passed": cell["passed"],
            })
    pd.DataFrame(rows).to_csv(out / "capex_guidance_timing.csv", index=False)
    (out / "capex_guidance_gates.json").write_text(json.dumps({
        "universe_used": rep["universe_used"], "feasibility": rep["feasibility"],
        "derisk": {"verdict": rep["derisk_scaler_gate"]["derisk"]["verdict"],
                  "gate": rep["derisk_scaler_gate"]["derisk"]["gate"]},
        "scaler": {"verdict": rep["derisk_scaler_gate"]["scaler"]["verdict"],
                  "gate": rep["derisk_scaler_gate"]["scaler"]["gate"]},
    }, default=str, indent=2))
    print(f"\nwrote {out}/capex_guidance_timing.csv, capex_guidance_gates.json")


def _write_value_chain_outputs(rep: dict, out: Path) -> None:
    rows = [{"name": "equal_weight", **rep["equal_weight"]},
            {"name": "value_chain_tilt", **rep["value_chain_tilt"]},
            {"name": "pair_standalone", **rep["pair_standalone"]}]
    for b, blk in rep["benchmarks"].items():
        rows.append({"name": b, **blk["metrics"]})
    pd.DataFrame(rows).to_csv(out / "value_chain_metrics.csv", index=False)
    import json
    (out / "value_chain_gates.json").write_text(json.dumps(
        {"gate1": rep["gate1"], "gate2": rep["gate2"],
         "episode": {k: str(v) for k, v in rep["episode"].items()}}, indent=2))
    print(f"\nwrote {out}/value_chain_metrics.csv, value_chain_gates.json")


if __name__ == "__main__":
    main()
