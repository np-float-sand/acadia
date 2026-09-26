"""
Runnable script — electricity-price-volatility / crypto-miner probe.

Reproduces every number in docs/electricity-price-volatility-miner-results.md.
Follows docs/electricity-price-volatility-miner-preregistration.md Decisions 1-7
exactly; see that doc and research/electricity_price_volatility/{universe,data,
regressions,hub_selection}.py for the locked spec.

Usage:
    python3 -m research.electricity_price_volatility.run_probe
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from grid_resilience.data.equity_prices import fetch_prices
from research.electricity_price_volatility import data as dm
from research.electricity_price_volatility import universe as U
from research.electricity_price_volatility.hub_loader import load_hub_daily
from research.electricity_price_volatility.hub_selection import scan_hubs_train_window, select_hub
from research.electricity_price_volatility.regressions import (
    variance_regression, return_regression, gbm_oos, tail_cut_test,
)

warnings.filterwarnings("ignore")

END = "2026-09-15"          # generous upper bound; loaders truncate to what's cached
OUT_DIR = Path(__file__).resolve().parent / "_output"
OUT_DIR.mkdir(exist_ok=True)


def load_all_hub_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for iso, hub in U.ALL_HUBS:
        price = load_hub_daily(iso, hub, U.TRAIN_START, END)
        if price.empty:
            print(f"  [warn] no data for {iso}:{hub}")
            continue
        frames[f"{iso}:{hub}"] = dm.build_hub_frame(price)
    return frames


def build_full_panel(
    equity_price: pd.Series, hub_frame: pd.DataFrame,
    control_ret: pd.Series, control_var: pd.Series,
) -> pd.DataFrame:
    eq = dm.build_asset_frame(equity_price.dropna(), "eq")
    idx = eq.index
    aligned_hub = dm.align_lagged_hub_to_equity(hub_frame, idx, lag_days=1)
    panel = pd.DataFrame({
        "eq_ret": eq["eq_ret"],
        "eq_var": eq["eq_var"],
        "hub_var_lag": aligned_hub["hub_var"],
        "hub_ret_lag": aligned_hub["hub_ret"],
        "control_ret": control_ret.reindex(idx),
        "control_var": control_var.reindex(idx),
    })
    return panel


def window_linear_specs(panel: pd.DataFrame, start: str, end: str) -> dict:
    w = panel.loc[start:end]
    lin_var = variance_regression(w["eq_var"], w["hub_var_lag"], w["control_var"], w["control_ret"])
    lin_ret = return_regression(w["eq_ret"], w["hub_ret_lag"], w["control_ret"], w["control_var"])
    tail = tail_cut_test(w["hub_ret_lag"], w["eq_var"], quantile=0.90)
    return dict(linear_var=lin_var, linear_ret=lin_ret, tail_cut=tail, n_days=len(w.dropna()))


def run_name(name: str, arm: str, price_panel: pd.DataFrame,
             hub_frames: dict[str, pd.DataFrame], predicted_sign: int) -> dict:
    print(f"\n=== {name} ({arm}) ===")
    equity_price = price_panel[name]
    if arm == "miner":
        control_ret = np.log(price_panel[U.BTC_TICKER] / price_panel[U.BTC_TICKER].shift(1))
        control_var = control_ret ** 2
    else:
        control_ret = np.log(price_panel[U.SPY_TICKER] / price_panel[U.SPY_TICKER].shift(1))
        control_var = control_ret ** 2

    eq_full = dm.build_asset_frame(equity_price.dropna(), "eq")
    train_index = eq_full.loc[U.TRAIN_START:U.TRAIN_END].index
    if len(train_index) < 50:
        print(f"  [skip] insufficient train-window history ({len(train_index)} days)")
        return dict(name=name, arm=arm, selected_hub=None, note="insufficient train history")

    scan_df = scan_hubs_train_window(
        eq_full["eq_var"], hub_frames, control_var, control_ret, predicted_sign, train_index,
    )
    selected = select_hub(scan_df)
    print(f"  hub scan (train {U.TRAIN_START}->{U.TRAIN_END}): "
          f"{scan_df['sign_ok'].sum()}/19 correct sign; selected = {selected}")

    result = dict(
        name=name, arm=arm, predicted_sign=predicted_sign,
        scan_table=scan_df.reset_index().to_dict(orient="records"),
        selected_hub=selected,
    )
    if selected is None:
        print("  -> NO HUB SELECTED (no hub had correct sign in train window)")
        return result

    hub_frame = hub_frames[selected]
    panel = build_full_panel(equity_price, hub_frame, control_ret, control_var)

    train_end_avail = panel.loc[:U.TRAIN_END].index.max()
    result["train"] = window_linear_specs(panel, U.TRAIN_START, U.TRAIN_END)
    result["test"] = window_linear_specs(panel, U.TEST_START, END)
    result["ytd"] = window_linear_specs(panel, U.YTD_START, END)

    # Non-linear: GBM trained on TRAIN window, evaluated OOS on TEST (and, from the
    # same fitted model with no retraining, on the YTD subset) -- Decision 5: the
    # train/test split IS the OOS check.
    feat_cols = ["hub_var_lag", "hub_ret_lag", "control_var", "control_ret"]
    train_df = panel.loc[U.TRAIN_START:U.TRAIN_END, feat_cols]
    train_y = panel.loc[U.TRAIN_START:U.TRAIN_END, "eq_var"]
    test_df = panel.loc[U.TEST_START:END, feat_cols]
    test_y = panel.loc[U.TEST_START:END, "eq_var"]
    ytd_df = panel.loc[U.YTD_START:END, feat_cols]
    ytd_y = panel.loc[U.YTD_START:END, "eq_var"]

    result["test"]["gbm"] = gbm_oos(train_df, train_y, test_df, test_y)
    result["ytd"]["gbm"] = gbm_oos(train_df, train_y, ytd_df, ytd_y)

    lv_test = result["test"]["linear_var"]
    gate_pass = (
        pd.notna(lv_test["tstat"])
        and abs(lv_test["tstat"]) >= U.GATE_ABS_T
        and (lv_test["coef"] < 0 if predicted_sign < 0 else lv_test["coef"] > 0)
    )
    result["gate_pass"] = bool(gate_pass)

    print(f"  selected hub: {selected}")
    print(f"  TEST  linear-var: coef={lv_test['coef']:.6g} t={lv_test['tstat']:.3f} n={lv_test['n']}")
    print(f"  gate ({'|t|>=2.0, correct sign'}): {'PASS' if gate_pass else 'FAIL'}")
    ytd_lv = result["ytd"]["linear_var"]
    print(f"  YTD   linear-var: coef={ytd_lv['coef']:.6g} t={ytd_lv['tstat']:.3f} n={ytd_lv['n']}")
    return result


def main():
    print("Loading equity + BTC + SPY prices ...")
    tickers = sorted(set(U.MINERS_ALL + U.GENERATORS + [U.BTC_TICKER, U.SPY_TICKER]))
    price_panel = fetch_prices(tickers, U.TRAIN_START, END)
    print(f"  price panel: {price_panel.shape}, columns={list(price_panel.columns)}")

    print("\nLoading 19-hub daily price frames ...")
    hub_frames = load_all_hub_frames()
    print(f"  loaded {len(hub_frames)}/19 hubs")

    all_results = []
    for name in U.MINERS_PRIMARY:
        all_results.append(run_name(name, "miner", price_panel, hub_frames, U.MINER_PREDICTED_SIGN))
    for name in U.MINERS_WIDENED:
        all_results.append(run_name(name, "miner_widened", price_panel, hub_frames, U.MINER_PREDICTED_SIGN))
    for name in U.GENERATORS:
        all_results.append(run_name(name, "generator", price_panel, hub_frames, U.GENERATOR_PREDICTED_SIGN))

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: None if pd.isna(o) else float(o))
    print(f"\nWrote {out_path}")
    return all_results


if __name__ == "__main__":
    main()
