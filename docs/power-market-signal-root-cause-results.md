# Why power-market data does not predict power equities — root-cause result

**Date:** 2026-09-21
**Status:** NEGATIVE (13 tests) + one **decisive mechanism finding**
**Mandate tested:** "use information/knowledge about power markets to profitably trade a
quantitative strategy in equities", with a required bar of profitability in **2023-2026** and
**2025-2026**.

---

## 1. Headline

Thirteen independent tests, all negative for a genuine power-market edge. But unlike the
prior ~29 attempts, this session identified **the mechanism that explains the failures**:

> **Energy futures markets efficiently price public power/gas fundamentals. Power-levered
> equities price off those futures. Therefore public, aggregate power-market data cannot
> predict power equities — the information is already impounded one layer upstream.**

This is a structural explanation, not another null result. It predicts that *every* signal
built from public aggregate ISO/EIA data will fail, which is exactly what the log records.

### The decisive test
Natural-gas **storage deviation from the 5-year norm** is the single most-watched fundamental
in the US gas market. Regressed on forward Henry Hub futures returns (HAC, publication-lagged):

| horizon | 2010-2018 | 2019-2022 | 2023-2026 |
|---|---|---|---|
| 21d  | t=+0.28 | t=+0.70 | t=-0.05 |
| 63d  | t=+0.10 | t=+0.58 | t=+0.97 |
| 126d | t=+0.51 | t=+0.80 | t=+2.42 |

One marginal hit in nine cells is chance. **If the most-watched gas fundamental cannot predict
gas prices, no gas/power fundamental can predict gas/power equities through the price channel.**

---

## 2. What was tested and what happened

New data plumbed this session (none of it previously used in the log):
`/electricity/rto/daily-fuel-type-data/` (daily generation by fuel by BA, 2019-2026, 2,820 obs),
`/electricity/rto/region-data/` (hourly demand), weekly L48 gas storage (2010-2026),
Henry Hub daily spot (2010-2026), plus ERCOT/PJM/CAISO/MISO regional daily generation.
Universe widened beyond the equipment book to **coal, gas E&P, oil E&P, uranium, IPPs,
regulated utilities, midstream/LNG, solar** — coal, uranium and gas/oil E&P had never been tested.

| # | Test | Outcome |
|---|---|---|
| 1 | Coal dispatch-competitiveness composite (gas burn + HH + storage + coal share) -> thermal coal | FAIL. Train t=+1.3/+1.6, holdout t=-0.01/-0.08 |
| 2 | Fuel-share momentum, cross-sectional rotation across 4 fuel-linked baskets | FAIL. Sharpe 1.92 train -> 0.08 holdout; worse than equal-weight (0.70) |
| 3 | Gas futures -> gas-equity lead-lag (underreaction) | FAIL. Train t=2.2-2.3, holdout t≈0. Market is efficient |
| 4 | ERCOT/PJM load anomaly -> IPP returns (exogenous weather shock) | FAIL. t≈0 in all windows |
| 5 | Thermal-vs-met coal spread (natural control: power demand vs steel demand) | **Mechanism absent** — beta to gas +0.015 (t=1.10). Thermal coal equities do not respond to gas even contemporaneously |
| 6 | Gas-vs-oil E&P spread, predicted by storage/HH across 4 windows | FAIL. No signal consistent in sign AND strength |
| 7 | Energy-factor residual reversal (SPY+NG+CL) | Not robust — positive only post-2022; turnover costs remove most of it |
| 8 | Gas-trend/vol regime gate on reversal | FAIL. Gate made every cell worse; hypothesis falsified |
| 9 | **Storage -> gas price (decisive diagnostic)** | **FAIL — see §1. Futures are efficient** |
| 10 | IPP-minus-regulated-utility spread (pure merchant power exposure) | Mostly equity beta: beta_SPY +0.573 vs beta_NG +0.025. 2023-26 +41.7%/yr but 2025-26 +1.5%/yr |
| 11 | Seasonality across power sub-sectors | FAIL. Early-vs-late monthly-pattern correlation NEGATIVE for 4 of 5 sub-sectors |
| 12 | Cross-sectional residual momentum in power complex | **Profitable but NOT power-specific** — see §3 |
| 13 | Structural power-demand growth as signal and as risk gate | FAIL. Signs flip across windows; as a gate it cut 2023-26 Sharpe 1.37 -> 0.23 |

### Falsified sub-hypotheses worth recording
- **Coal-gas switching (pre-registered before testing).** US thermal coal equities have
  essentially zero contemporaneous beta to gas prices versus met coal (+0.015, t=1.10). Likely
  causes: long-term contract pricing, and large seaborne/export books at BTU/CNR that behave
  met-coal-like. There is no dispatch-economics channel into the equity. **Do not retry.**
- **Spurious-regression trap.** The strongest single cell in the initial scan
  (`gas_burn_yoy` -> coal, t=-4.64) collapsed to t=+0.75 once z-scored. The signal lived
  entirely in the slow-moving *level* of two near-unit-root series. HAC with maxlags=horizon
  does **not** rescue this. Level-vs-level tests in this domain should be treated as invalid
  by default.
- **A YoY-of-30d-moving-average generation signal read +12.4% in Sept 2026**; true like-for-like
  YTD growth was **+2.8%**. These YoY-of-MA constructions are weather noise, not demand signal.

---

## 3. The one profitable strategy found — and why it is not the answer

Cross-sectional **residual momentum** (252d formation skipping the last month, top quintile,
monthly rebalance, SPY-beta-neutralised, 25% vol target, 25bp costs) over the 44-name
power/electrification universe:

| window | CAGR | vol | Sharpe | MaxDD |
|---|---|---|---|---|
| 2015-2018 | 1.7% | 21.9% | 0.08 | -30.3% |
| 2019-2022 | 19.1% | 28.0% | 0.68 | -55.0% |
| **2023-2026** | **35.9%** | 26.3% | **1.37** | -30.5% |
| **2025-2026** | **40.1%** | 27.0% | **1.48** | -23.9% |
| full 2015-2026 | 17.9% | 25.6% | 0.70 | -55.0% |
| SPY 2015-2026 | 13.7% | 17.5% | 0.78 | -33.7% |

It **meets the stated bar** and is robust across parameters (positive in 45 of 48
lookback x quintile x rebalance x cost cells; positive in *every* cell for both target windows).

**But it is not a power-market strategy, and it should not be sold as one:**
1. **The universe does no work.** Identical machinery on 50 non-energy large-caps:
   2023-26 Sharpe **1.46** vs the power complex's **1.45**; 2015-18 **0.96** vs **0.02**.
   The control universe wins outright on the full sample.
2. **The energy factors do no work.** Residualising on SPY alone beats SPY+NG+CL in three of
   four windows (2023-26: 1.58 vs 1.25; 2025-26: 2.00 vs 1.30). The gain over raw momentum comes
   from removing *market beta*, not from gas/oil factors.
3. **It does not beat buy-and-hold full-sample** on a risk-adjusted basis (0.70 vs SPY 0.78),
   and carries a -55% drawdown.

Conclusion: this is generic cross-sectional momentum. The 2023-26 / 2025-26 numbers are the
electrification theme's bull run plus a well-known anomaly, not power-market alpha.

---

## 4. Where an edge could still exist

The mechanism finding in §1 is constructive: it says to look **where there is no futures market
to price the information**. Aggregate ISO/EIA/hub data is priced. Not obviously priced:

1. **Nodal basis for a specific company's fleet.** Hub prices have liquid futures; individual
   pnodes largely do not. A generator's realised capture price vs. the hub is company-specific.
   *Caveat:* the log's Probe A already showed fleet x realised price tracks **revenue**
   (Spearman +0.45) but not **operating income** (hedges/retail segments sever it) — so this
   must be tested against revenue surprise or capture-rate disclosure, not earnings.
2. **Plant-level physical generation by operator** (EIA Form 923), as a within-quarter nowcast
   of utilisation, tested against *revenue* surprise rather than EPS.
3. **Options / variance rather than direction.** Returns are unpredictable here; volatility is
   not. Untested in this project because no options data is wired in.

**Not worth retrying:** any signal built from national/hub-level EIA or ISO series against
equity returns. That space is now well covered — 29 prior attempts plus these 13, with a
mechanism that explains why.

---

## 5. Reproduction

Throwaway probe scripts (scratchpad, not committed): `eia_fetch.py`, `panel.py`, `scan_train.py`,
`holdout.py`, `fuelmom.py`, `leadlag.py`, `spreads.py`, `multiwin.py`, `resrev.py`, `regrev.py`,
`decisive.py`, `mom.py`, `factortest.py`, `robust.py`, `final.py`, `lasttest.py`.
Data via EIA API v2 (key in `.env`) and yfinance. All tests use publication lags (5d on daily
EIA series, +7d on weekly storage) and HAC standard errors with maxlags = forecast horizon.
