"""Quality/balance-sheet weighting probe -- RESEARCH-LOG open item 13.
docs/quality-weighting-preregistration.md has the locked formula and gate.

Build:
  1. Sanity check: reproduce the documented shipped `screen +val+sleeve` baseline
     AS THIS WORKTREE'S CODE ACTUALLY COMPUTES IT (this worktree's `universe_seed.csv`
     resolves `screen` to 28 names, including THR, one more than the 27-name figure
     cited in the task/RESEARCH-LOG -- see docs/quality-weighting-results.md §1 for
     the reconciliation: worktree-pool minus THR == the 27-name list verbatim).
  2. Primary comparison: equal-weight vs quality-weighted, BOTH restricted to the
     same fixed 27-name pool (screen-minus-THR) for a clean apples-to-apples test,
     both composed through the identical +val+sleeve overlay stack.
  3. Gate (a): quality beats equal-weight on Sharpe in both config.SUBWINDOWS.
  4. Gate (b): regress monthly (quality - equal-weight) excess return on a monthly
     SPLV-minus-SPY low-vol factor; require |t(alpha)| >= 1.5.

Throwaway probe. Run: python3 research/quality_weighting/run_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))

from grid_equipment_basket.basket import apply_cap, simulate_basket  # noqa: E402
from grid_equipment_basket.backtest import compute_metrics  # noqa: E402
from grid_equipment_basket.data.prices import fetch_prices  # noqa: E402

from electrification_strategy import backtest as ebt  # noqa: E402
from electrification_strategy import config, hedge_overlay, universe  # noqa: E402

from cik_map_27 import SCREEN_27  # noqa: E402
import fetch_fundamentals as ff  # noqa: E402
import signals as qs  # noqa: E402

START = config.START_DEFAULT
END = "2026-09-21"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def screen27_members(prices: pd.DataFrame, asof) -> list[str]:
    listed = universe._listed_eligible(prices, pd.Timestamp(asof))
    return sorted(set(SCREEN_27) & listed)


def equal_weight_27_target_fn(prices: pd.DataFrame):
    def _fn(available, asof):
        elig = [t for t in screen27_members(prices, asof) if t in set(available)]
        if not elig:
            return pd.Series(dtype=float)
        raw = pd.Series(1.0 / len(elig), index=sorted(elig))
        return apply_cap(raw, config.MAX_SINGLE_NAME_WEIGHT)
    return _fn


def quality_target_fn(prices: pd.DataFrame, leverage_d, margins_d, fcf_d, diag: list):
    def _fn(available, asof):
        elig = sorted(t for t in screen27_members(prices, asof) if t in set(available))
        if not elig:
            return pd.Series(dtype=float)
        comp = qs.composite_z(elig, asof, leverage_d, margins_d, fcf_d)
        factor = qs.weight_factor(comp)
        diag.append({"asof": pd.Timestamp(asof), "n_elig": len(elig),
                     "n_complete": int(comp.notna().sum()),
                     "mean_factor": float(factor.mean()), "min_factor": float(factor.min()),
                     "max_factor": float(factor.max())})
        raw = pd.Series(1.0 / len(elig), index=elig) * factor.reindex(elig)
        raw = raw / raw.sum()
        return apply_cap(raw, config.MAX_SINGLE_NAME_WEIGHT)
    return _fn


def main():
    print("=== 1. Fetch prices ===")
    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    tickers = sorted(set(cand) | set(config.BENCHMARKS) | set(config.HEDGE_SLEEVE_TICKERS)
                     | set(config.HEDGE_SHORT_TICKERS) | {"SPLV"})
    prices = fetch_prices(tickers, START, END)
    cand_prices = prices[[c for c in cand if c in prices.columns]]
    spy_ret = prices["SPY"].pct_change()

    print("=== 2. Sanity check: as-shipped-in-this-worktree screen +val+sleeve ===")
    dummy_dfii10 = pd.Series(0.0, index=pd.date_range(START, END, freq="D"))
    shipped = ebt.run_comparison(START, END, prices, dummy_dfii10, constructions=("screen",))
    shipped_cell = shipped["cells"][("screen", "+val+sleeve")]
    sm_ = shipped_cell["metrics"]
    print(f"  screen +val+sleeve (28-name, as committed in this worktree): "
          f"Sharpe {sm_['sharpe']:.3f}  CAGR {sm_['cagr']*100:.2f}%  MaxDD {sm_['max_dd']*100:.2f}%  "
          f"beta {shipped_cell['beta']:.2f}")
    print("  (documented reference, main repo RESEARCH-LOG.md: Sharpe 0.69, MaxDD -24%, beta 0.59)")

    print("=== 3. Fetch fundamentals (leverage, margins, FCF) for the 27-name pool ===")
    leverage_d, fcf_d, margins_d = ff.fetch_all(SCREEN_27, use_cache=True)
    n_lev = sum(1 for d in leverage_d.values() if not d.empty)
    n_fcf = sum(1 for d in fcf_d.values() if not d.empty)
    n_marg = sum(1 for d in margins_d.values() if not d.empty)
    print(f"  leverage data: {n_lev}/27   fcf data: {n_fcf}/27   margin data: {n_marg}/27")

    print("=== 4. Build equal-weight (27-name) baseline and quality-weighted variant ===")
    base_r_eq = simulate_basket(cand_prices, START, END, config.REBALANCE_LAG_DAYS,
                                config.MAX_SINGLE_NAME_WEIGHT,
                                equal_weight_27_target_fn(cand_prices)).returns

    diag = []
    base_r_q = simulate_basket(cand_prices, START, END, config.REBALANCE_LAG_DAYS,
                               config.MAX_SINGLE_NAME_WEIGHT,
                               quality_target_fn(cand_prices, leverage_d, margins_d, fcf_d, diag)).returns

    core_eq = ebt.compose(base_r_eq, spy_ret, use_valuation=True)
    core_q = ebt.compose(base_r_q, spy_ret, use_valuation=True)

    sleeve_r = hedge_overlay.sleeve_returns(prices)
    dummy_short = pd.Series(0.0, index=core_eq.index)
    dummy_mask = pd.Series(False, index=core_eq.index)

    cell_eq = hedge_overlay.apply_hedge(core_eq, sleeve_r, dummy_short, dummy_mask, True, False)
    cell_q = hedge_overlay.apply_hedge(core_q, sleeve_r, dummy_short, dummy_mask, True, False)

    m_eq = compute_metrics(cell_eq, config.RF, config.ANN)
    m_q = compute_metrics(cell_q, config.RF, config.ANN)
    print(f"  27-name equal-weight   +val+sleeve: Sharpe {m_eq['sharpe']:.3f}  "
          f"CAGR {m_eq['cagr']*100:.2f}%  MaxDD {m_eq['max_dd']*100:.2f}%  Vol {m_eq['ann_vol']*100:.2f}%")
    print(f"  27-name quality-weight +val+sleeve: Sharpe {m_q['sharpe']:.3f}  "
          f"CAGR {m_q['cagr']*100:.2f}%  MaxDD {m_q['max_dd']*100:.2f}%  Vol {m_q['ann_vol']*100:.2f}%")

    print("=== 5. Sub-window Sharpes (config.SUBWINDOWS) ===")
    sub_results = {}
    for name, (a, b) in config.SUBWINDOWS.items():
        s_eq = compute_metrics(cell_eq.loc[a:b], config.RF, config.ANN)["sharpe"]
        s_q = compute_metrics(cell_q.loc[a:b], config.RF, config.ANN)["sharpe"]
        sub_results[name] = (s_eq, s_q)
        print(f"  {name}: equal-weight Sharpe {s_eq:.3f}   quality Sharpe {s_q:.3f}   "
              f"beats: {s_q > s_eq}")

    gate_a = all(s_q > s_eq for s_eq, s_q in sub_results.values())
    print(f"  GATE (a) beats equal-weight in BOTH sub-windows: {gate_a}")

    print("=== 6. Low-vol-control regression (gate b) ===")
    monthly_eq = (1.0 + cell_eq).resample("ME").prod() - 1.0
    monthly_q = (1.0 + cell_q).resample("ME").prod() - 1.0
    excess_m = (monthly_q - monthly_eq).dropna()

    splv_ret = prices["SPLV"].pct_change()
    monthly_splv = (1.0 + splv_ret).resample("ME").prod() - 1.0
    monthly_spy = (1.0 + spy_ret).resample("ME").prod() - 1.0
    lowvol_factor_m = (monthly_splv - monthly_spy).dropna()

    reg_df = pd.concat([excess_m.rename("excess"), lowvol_factor_m.rename("lowvol")],
                       axis=1, join="inner").dropna()
    X = sm.add_constant(reg_df["lowvol"].values)
    y = reg_df["excess"].values
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params
    t_alpha, t_beta = model.tvalues
    r2 = model.rsquared
    n_months = len(reg_df)
    print(f"  n_months={n_months}  alpha(monthly)={alpha:.5f} (t={t_alpha:.2f})  "
          f"beta={beta:.3f} (t={t_beta:.2f})  R2={r2:.3f}")
    print(f"  alpha annualized (x12, simple): {alpha*12*100:.2f}%/yr")
    gate_b = abs(t_alpha) >= 1.5
    print(f"  GATE (b) |t(alpha)| >= 1.5: {gate_b}   (t={t_alpha:.2f})")

    print("=== 7. Composite-weight mechanism diagnostics ===")
    diag_df = pd.DataFrame(diag)
    if not diag_df.empty:
        print(f"  {len(diag_df)} rebalances. mean n_elig={diag_df['n_elig'].mean():.1f}  "
              f"mean n_complete(3/3 data)={diag_df['n_complete'].mean():.1f}  "
              f"mean factor range=[{diag_df['min_factor'].mean():.3f}, {diag_df['max_factor'].mean():.3f}]")
        diag_df.to_csv(OUT_DIR / "quality_weight_diagnostics.csv", index=False)

    print("=== VERDICT ===")
    verdict = "PASS" if (gate_a and gate_b) else "FAIL / NOT ADOPTED"
    print(f"  Gate (a) both-subwindow Sharpe beat: {gate_a}")
    print(f"  Gate (b) residual (alpha) t>=1.5 after low-vol control: {gate_b} (t={t_alpha:.2f})")
    print(f"  VERDICT: {verdict}")

    # -- persist everything needed to write the results doc -----------------------
    out_rows = []
    full_eq = compute_metrics(cell_eq, config.RF, config.ANN)
    full_q = compute_metrics(cell_q, config.RF, config.ANN)
    out_rows.append({"window": "full", "cell": "equal_weight", **full_eq})
    out_rows.append({"window": "full", "cell": "quality_weight", **full_q})
    for name, (a, b) in config.SUBWINDOWS.items():
        out_rows.append({"window": name, "cell": "equal_weight",
                         **compute_metrics(cell_eq.loc[a:b], config.RF, config.ANN)})
        out_rows.append({"window": name, "cell": "quality_weight",
                         **compute_metrics(cell_q.loc[a:b], config.RF, config.ANN)})
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "metrics.csv", index=False)

    pd.DataFrame({"equal_weight": cell_eq, "quality_weight": cell_q}).to_csv(OUT_DIR / "returns.csv")

    reg_summary = {
        "n_months": n_months, "alpha_monthly": alpha, "alpha_annualized": alpha * 12,
        "t_alpha": t_alpha, "beta": beta, "t_beta": t_beta, "r2": r2,
        "gate_a": gate_a, "gate_b": gate_b, "verdict": verdict,
        "shipped_28name_sharpe": sm_["sharpe"], "shipped_28name_maxdd": sm_["max_dd"],
        "shipped_28name_cagr": sm_["cagr"], "shipped_28name_beta": shipped_cell["beta"],
        "n_lev": n_lev, "n_fcf": n_fcf, "n_marg": n_marg,
    }
    pd.Series(reg_summary).to_csv(OUT_DIR / "gate_summary.csv")
    print(f"\nwrote {OUT_DIR}/metrics.csv, returns.csv, gate_summary.csv, quality_weight_diagnostics.csv")


if __name__ == "__main__":
    main()
