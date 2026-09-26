"""
Comprehensive sensitivity scan: interconnection-queue backlog (LBNL national index)
and forward power (STEO spark-spread) against EVERY ticker named anywhere in this
session -- user request, loosened bar (|t|>=1.5, not this project's default >=2.0),
explicitly to see "how sensitive" the relationship is, not to find a tradeable signal.

Universe = union of every ticker discussed this session:
  - screen basket (27, via the real `universe.members("screen", ...)` call)
  - REIT names from Thread 1c (SBAC/WPC/EQIX/CCI/AMT/DLR/IRM)
  - merchant generators (VST/NRG/CEG/TLN)
  - yieldco/renewable-developer names across threads (AES/BEP/CWEN/NEE/AQN/ORA/ENLT/BEPC/FRVO)
  - solar-rotation sleeve (ENPH/SEDG/CHPT/RUN/BLNK/STEM, item 17)

Signals:
  - queue: LBNL national interconnection-queue backlog, 3-month change (Thread 3's
    own construction), research/datacenter_reit_cep/lbnl_national_queue_mw.parquet
  - power: STEO PJM/ERCOT composite forward spark-spread z-trend (this session's
    STEO threads), research/steo_spark_spread/data/steo_trend_{PJM,ERCOT}.parquet

Method, for each (ticker, signal, lag in {0,1,2,3} months):
  (A) MEAN regression: monthly_ret_t ~ signal_(t-lag) + spy_ret_t  [t-stat on signal]
  (B) "GARCH-style" VARIANCE regression, matching this project's established convention
      (research/electricity_price_volatility/regressions.py `variance_regression`):
      monthly_realized_var_t ~ signal_(t-lag) + spy_realized_var_t  [t-stat on signal]
      (realized_var = sum of squared daily log returns within the calendar month;
      NOT a fitted GARCH(1,1) -- this project's own prior "GARCH-style" work uses this
      same realized-variance-regression definition, not a literal ARCH/GARCH fit,
      kept for methodological consistency.)

IMPORTANT CAVEAT computed and printed, not hidden: at a two-sided |t|>=1.5 bar, ~13.4%
of tests will clear it under pure noise (no real relationship anywhere). With this many
(ticker x signal x lag x test-type) combinations, a naive count of "hits" is expected to
be substantial by chance alone -- the script reports the expected-by-chance count
alongside the actual count so the two can be compared, not just a bare hit list.

Full in-sample, no fit/holdout split (a sensitivity scan, not a tradeable-signal test
per this project's usual discipline -- explicitly framed as such per the user's request).

Throwaway probe. Findings -> docs/interconnection-queue-signal-results.md,
docs/steo-spark-spread-results.md, docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, ".")

from grid_equipment_basket.data.prices import fetch_prices
from electrification_strategy import config, universe

HERE = pathlib.Path(__file__).parent

REIT = ["SBAC", "WPC", "EQIX", "CCI", "AMT", "DLR", "IRM"]
MERCHANT = ["VST", "NRG", "CEG", "TLN"]
YIELDCO = ["AES", "BEP", "CWEN", "NEE", "AQN", "ORA", "ENLT", "BEPC", "FRVO"]
SOLAR = ["ENPH", "SEDG", "CHPT", "RUN", "BLNK", "STEM"]

LAGS = [0, 1, 2, 3]
T_BAR = 1.5


def load_steo_composite() -> pd.Series:
    pjm = pd.read_parquet(HERE / "data" / "steo_trend_PJM.parquet")["steo_spark_trend_z"]
    ercot = pd.read_parquet(HERE / "data" / "steo_trend_ERCOT.parquet")["steo_spark_trend_z"]
    df = pd.concat([pjm.rename("pjm"), ercot.rename("ercot")], axis=1)
    daily = df.mean(axis=1, skipna=True).dropna().sort_index()
    m = daily.resample("ME").last()
    return m.rename("power")


def load_queue_signal() -> pd.Series:
    df = pd.read_parquet("research/datacenter_reit_cep/lbnl_national_queue_mw.parquet")
    s = df["queue_mw"]
    s.index = pd.PeriodIndex(s.index, freq="M").to_timestamp("M")
    return s.pct_change(3).rename("queue")  # 3-month % change, Thread 3's construction


def monthly_ret(price: pd.Series) -> pd.Series:
    m = price.resample("ME").last()
    return m.pct_change()


def monthly_realized_var(price: pd.Series) -> pd.Series:
    logret = np.log(price / price.shift(1))
    return (logret ** 2).resample("ME").sum()


def run_test(y: pd.Series, x: pd.Series, control: pd.Series, lag: int):
    """OLS t-stat AND a HAC(Newey-West, 6-lag)-robust t-stat side by side. Slow-
    trending series (esp. the queue signal and realized-variance targets, both of
    which can be highly autocorrelated) violate OLS's iid-error assumption and
    produce inflated t-stats -- this project already hit exactly this failure mode
    once (docs/ancillary-scarcity-pricing-feasibility-results.md: naive overlapping-
    window IC t up to 4.61, corrected to <1.1 with monthly+HAC). Report both so a
    naive-vs-corrected gap is visible, not hidden."""
    xl = x.shift(lag)
    d = pd.DataFrame({"y": y, "x": xl, "c": control}).dropna()
    if len(d) < 12:
        return None, None, len(d)
    X = sm.add_constant(d[["x", "c"]])
    m = sm.OLS(d["y"], X).fit()
    m_hac = m.get_robustcov_results(cov_type="HAC", maxlags=6)
    # get_robustcov_results returns params/tvalues in the same column order as X
    idx = list(X.columns).index("x")
    return m.tvalues["x"], m_hac.tvalues[idx], len(d)


def main():
    screen_cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    universe_all = sorted(set(screen_cand) | set(REIT) | set(MERCHANT) | set(YIELDCO) | set(SOLAR))
    print(f"Total unique tickers in scan: {len(universe_all)}")

    prices = fetch_prices(sorted(set(universe_all) | {"SPY"}), "2015-01-01", "2026-09-01")

    cand_prices = prices[[c for c in screen_cand if c in prices.columns]]
    screen_members = set(universe.members("screen", cand_prices.index[-1], cand_prices))
    print(f"screen basket (27 expected): {len(screen_members)}")

    tickers = sorted(set(screen_members) | set(REIT) | set(MERCHANT) | set(YIELDCO) | set(SOLAR))
    no_data = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    tickers = [t for t in tickers if t not in no_data]
    print(f"Tickers with any price data: {len(tickers)}  |  no data: {no_data}")

    spy_ret = monthly_ret(prices["SPY"])
    spy_var = monthly_realized_var(prices["SPY"])
    queue = load_queue_signal()
    power = load_steo_composite()
    signals = {"queue": queue, "power": power}

    rows = []
    skipped_short_history = []
    for t in tickers:
        px = prices[t].dropna()
        depth_months = len(px.resample("ME").last())
        if depth_months < 16:  # need room for a lag-3 regression with n>=12
            skipped_short_history.append((t, depth_months))
            continue
        ret = monthly_ret(px)
        rvar = monthly_realized_var(px)
        for sig_name, sig in signals.items():
            for lag in LAGS:
                t_mean, t_mean_hac, n_mean = run_test(ret, sig, spy_ret, lag)
                if t_mean is not None:
                    rows.append({"ticker": t, "signal": sig_name, "lag": lag,
                                "test": "mean", "tstat": t_mean, "tstat_hac": t_mean_hac,
                                "n": n_mean})
                t_var, t_var_hac, n_var = run_test(rvar, sig, spy_var, lag)
                if t_var is not None:
                    rows.append({"ticker": t, "signal": sig_name, "lag": lag,
                                "test": "variance", "tstat": t_var, "tstat_hac": t_var_hac,
                                "n": n_var})

    res = pd.DataFrame(rows)
    n_tests = len(res)
    expected_by_chance = n_tests * 0.1336  # P(|t|>1.5) two-sided under the null

    hits_ols = res[res["tstat"].abs() >= T_BAR]
    hits_hac = res[res["tstat_hac"].abs() >= T_BAR].sort_values(
        "tstat_hac", key=lambda s: s.abs(), ascending=False)

    print(f"\nSkipped (insufficient history, <16 months): {skipped_short_history}")
    print(f"\nTotal regressions run: {n_tests}")
    print(f"Expected hits at |t|>=1.5 under PURE NOISE (13.36% x n_tests): {expected_by_chance:.1f}")
    print(f"\n--- Naive OLS t-stats ---")
    print(f"Hits at |t|>=1.5: {len(hits_ols)}  (rate {100*len(hits_ols)/n_tests:.1f}%, chance rate 13.4%)")
    print(f"\n--- HAC(6)-robust t-stats ---")
    print(f"Hits at |t|>=1.5: {len(hits_hac)}  (rate {100*len(hits_hac)/n_tests:.1f}%, chance rate 13.4%)")

    print(f"\n=== All {len(hits_hac)} HAC-robust hits, ranked by |t_hac| ===")
    print(hits_hac.to_string(index=False))

    strict = hits_hac[hits_hac["tstat_hac"].abs() >= 2.0]
    print(f"\n=== Of those, {len(strict)} also clear this project's default |t_hac|>=2.0 bar ===")
    print(strict.to_string(index=False))

    res.to_csv(HERE / "data" / "full_sensitivity_scan_all_results.csv", index=False)
    hits_hac.to_csv(HERE / "data" / "full_sensitivity_scan_hits.csv", index=False)


if __name__ == "__main__":
    main()
