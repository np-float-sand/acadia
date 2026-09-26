"""
ICR-via-EDGAR feasibility + retest probe (untried-tasks-log item 2).

Question 1 (feasibility): does pulling Interest Coverage Ratio (ICR = operating
income / |interest expense|) from SEC EDGAR XBRL companyconcept give materially
more history than yfinance's documented ~4-12 quarters per ticker for the
grid_resilience 25-name utility/merchant-generator universe?

Question 2 (the original ICR hypothesis, CLAUDE.md "Future Work"): does low ICR
flag better SHORT candidates during a rate-stress event? Tested on the one
genuine rate-stress event in the backtest window: the 2022-03 -> 2023-07 Fed
hiking cycle (leverage-sensitive names should underperform more as rates rise).

Pre-registered bars:
  - Feasibility PASS: median per-ticker quarters of coverage >= 20 (vs yfinance's
    ~4-12) AND coverage back to at least 2018 for >= 15/25 tickers.
  - Hypothesis PASS: Spearman corr(ICR at 2022-03, cumulative return 2022-03 ->
    2023-07) is positive (low ICR -> worse return) with |t| >= 1.5, OR the
    bottom-ICR tercile underperforms the top-ICR tercile by a margin that isn't
    reversed in a placebo window (a calm, non-hiking period of the same length).

Throwaway. Re-run to regenerate. Findings -> docs/RESEARCH-LOG.md,
docs/untried-tasks-log.md item 2.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "electricity_earnings_exposure"))
from edgar import quarterly_flow  # noqa: E402

CIK_BY_TICKER: dict[str, str] = {
    "NRG": "0001013871", "VST": "0001692819", "CNP": "0001130310", "AEP": "0000004904",
    "EXC": "0001109357", "PPL": "0000922224", "FE": "0001031296", "CEG": "0001868275",
    "TLN": "0001622536", "ETR": "0000065984", "WEC": "0000783325", "DTE": "0000936340",
    "CMS": "0000811156", "PCG": "0001004980", "EIX": "0000827052", "XEL": "0000072903",
    "ES": "0000072741", "ED": "0001047862", "D": "0000715957", "PEG": "0000788784",
    "AEE": "0001002910", "EVRG": "0001711269", "DUK": "0001326160", "SO": "0000092122",
    "NEE": "0000753308",
}

EBIT_TAGS = ["OperatingIncomeLoss"]
INT_TAGS = [
    "InterestExpense", "InterestExpenseDebt",
    "InterestExpenseDebtExcludingAmortization",
    "InterestAndDebtExpense", "InterestExpenseTrustPreferredSecurities",
]


def fetch_icr_edgar(cik_by_ticker: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (icr_wide[date x ticker], coverage_report[ticker -> stats])."""
    frames: dict[str, pd.Series] = {}
    cov_rows = []
    for t, cik in cik_by_ticker.items():
        ebit = quarterly_flow(cik, EBIT_TAGS)
        interest = quarterly_flow(cik, INT_TAGS)
        if ebit.empty or interest.empty:
            cov_rows.append({"ticker": t, "n_quarters": 0, "first": None, "last": None,
                              "ebit_tag_found": not ebit.empty, "interest_tag_found": not interest.empty})
            continue
        e = ebit.set_index("quarter_end")["val"].rename("ebit")
        i = interest.set_index("quarter_end")["val"].abs().rename("interest")
        merged = pd.concat([e, i], axis=1).dropna()
        merged = merged[merged["interest"] > 0]
        icr = (merged["ebit"] / merged["interest"]).clip(0, 20)
        if icr.empty:
            cov_rows.append({"ticker": t, "n_quarters": 0, "first": None, "last": None,
                              "ebit_tag_found": True, "interest_tag_found": True})
            continue
        frames[t] = icr
        cov_rows.append({
            "ticker": t, "n_quarters": len(icr),
            "first": icr.index.min().date(), "last": icr.index.max().date(),
            "ebit_tag_found": True, "interest_tag_found": True,
        })
    wide = pd.DataFrame(frames).sort_index()
    cov = pd.DataFrame(cov_rows).set_index("ticker")
    return wide, cov


def main() -> None:
    print("=== Fetching ICR from EDGAR XBRL for 25-name universe ===")
    icr_wide, cov = fetch_icr_edgar(CIK_BY_TICKER)
    cov = cov.sort_values("n_quarters", ascending=False)
    print(cov.to_string())

    n_q = cov["n_quarters"]
    print(f"\nmedian quarters of coverage: {n_q.median():.0f}  "
          f"(yfinance limitation was ~4-12)")
    n_back_to_2018 = (cov["first"].apply(lambda d: d is not None and d.year <= 2019)).sum()
    print(f"tickers with data back to 2019 or earlier: {n_back_to_2018} / {len(cov)}")
    feasibility_pass = (n_q.median() >= 20) and (n_back_to_2018 >= 15)
    print(f"FEASIBILITY GATE: {'PASS' if feasibility_pass else 'FAIL'}")

    icr_wide.to_parquet(os.path.join(os.path.dirname(__file__), "icr_edgar_panel.parquet"))
    print(f"\nSaved full panel: {icr_wide.shape[0]} quarters x {icr_wide.shape[1]} tickers "
          f"-> icr_edgar_panel.parquet")

    if not feasibility_pass:
        print("\nFeasibility gate failed -- stopping before the hypothesis test.")
        return

    # ---- Hypothesis test: low ICR -> worse return in the 2022-23 hiking cycle ----
    print("\n=== Hypothesis test: ICR vs forward return, hiking cycle vs placebo ===")
    tickers = [t for t in icr_wide.columns if icr_wide[t].notna().sum() >= 4]
    px = yf.download(tickers, start="2018-01-01", end="2026-08-01",
                      auto_adjust=True, progress=False)["Close"]

    def icr_asof(date: str, lag_days: int = 45) -> pd.Series:
        cutoff = pd.Timestamp(date) - pd.Timedelta(days=lag_days)
        avail = icr_wide[icr_wide.index <= cutoff]
        return avail.iloc[-1] if not avail.empty else pd.Series(dtype=float)

    def fwd_return(start: str, end: str) -> pd.Series:
        p0 = px.loc[:start].iloc[-1]
        p1 = px.loc[:end].iloc[-1]
        return (p1 / p0 - 1.0)

    def run_window(label: str, icr_date: str, ret_start: str, ret_end: str):
        icr_now = icr_asof(icr_date).reindex(tickers)
        ret = fwd_return(ret_start, ret_end).reindex(tickers)
        both = pd.concat([icr_now.rename("icr"), ret.rename("ret")], axis=1).dropna()
        if len(both) < 6:
            print(f"{label}: insufficient overlap (n={len(both)})")
            return
        rho = both["icr"].corr(both["ret"], method="spearman")
        n = len(both)
        t_stat = rho * np.sqrt((n - 2) / max(1e-9, (1 - rho**2)))
        low = both.nsmallest(max(1, n // 3), "icr")["ret"].mean()
        high = both.nlargest(max(1, n // 3), "icr")["ret"].mean()
        print(f"{label}: n={n}  spearman={rho:+.3f} (t={t_stat:+.2f})  "
              f"bottom-ICR-tercile ret={low:+.1%}  top-ICR-tercile ret={high:+.1%}  "
              f"spread(low-high)={low-high:+.1%}")

    run_window("HIKING  (ICR@2022-02, ret 2022-03->2023-07)",
               "2022-02-01", "2022-03-01", "2023-07-01")
    run_window("PLACEBO (ICR@2019-02, ret 2019-03->2020-07, pre-COVID calm-ish)",
               "2019-02-01", "2019-03-01", "2020-07-01")
    run_window("PLACEBO (ICR@2024-02, ret 2024-03->2025-07, cutting cycle)",
               "2024-02-01", "2024-03-01", "2025-07-01")


if __name__ == "__main__":
    main()
