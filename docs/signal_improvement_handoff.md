# Grid Resilience Strategy — Signal Improvement Handoff

**Prepared:** June 2026  
**Backtest window:** Apr 2018 – Dec 2025  
**Universe:** 18 U.S. utility stocks (NEE, DUK, SO, D, AEP, EXC, SRE, XEL, ES, ETR, FE, CNP, NRG, VST, WEC, DTE, CMS, PNW)  
**ISOs:** ERCOT, PJM, MISO, CAISO, SPP

---

## Baseline Performance (Current Signal)

| Metric | Strategy | XLU Benchmark |
|---|---|---|
| Cumulative return | +90% | +113% |
| Ann. return | 8.1% | 12.0% |
| Sharpe ratio | 0.31 | 0.24 |
| Max drawdown | −13.7% | −36.1% |
| Mean IC (stressed months) | 0.15 | — |

The strategy does not aim to beat XLU on raw return — it delivers a comparable return per unit of risk with a 2.6× drawdown reduction. Alpha comes from the short book picking individual losers, not from long-only stock selection.

---

## How the Current Signal Is Built

| Step | What | Source |
|---|---|---|
| 1 | Daily LMP prices across 5 ISOs | gridstatus library |
| 2 | Grid Stress Index (GSI) per ISO | Composite 4-signal score (see below) |
| 3 | Named stress events catalog | Polar vortices, heat domes, etc. |
| 4 | Conditional stress beta per stock | OLS on GSI × sector-excess returns, rolling window |
| 5 | Factor score → portfolio weights | 3 long / 5 short, equal-weight, monthly rebalance |

**GSI sub-signal weights (current):**

| Sub-signal | Weight | Notes |
|---|---|---|
| LMP z-score | 40% | Daily max LMP vs 90-day rolling baseline |
| Congestion fraction | 25% | Congestion component share of total LMP |
| Reserve tightness | 20% | Load / capacity proxy |
| Event flag | 15% | Binary with ±5-day decay around named events |

**Grid-searched parameters** (Jun 2026, 243 combinations):  
spike pct `0.97` · cong. threshold `0.30` · cong. min-days `10` · long book `3 names` · short book `5 names` · XLU hedge `off`

---

## Key Findings That Inform Future Signal Work

### Finding 1 — Short Book Drives Alpha
The edge lies in identifying specific losers during stress events — names with concentrated transmission or generation exposure — not in picking sector-relative winners. Every XLU-hedge configuration (87 of 243) produced negative Sharpe. Any new signal component should be evaluated primarily on its ability to discriminate the short book.

### Finding 2 — Duration Beats Threshold
A 30% congestion-fraction threshold + 10 consecutive-day minimum beats a higher threshold with a shorter duration filter. The duration floor carries the signal-to-noise burden. New congestion signals should apply the same discipline: lower the detection threshold, raise the duration requirement.

### Finding 3 — Concentrate the Long Book
3 long names outperforms 5 or 7 across all grid-search configurations. The factor has enough cross-sectional dispersion that wider long books dilute return. New signal components should be judged on whether they increase cross-sectional dispersion, not on raw Sharpe alone.

---

## Known Weaknesses to Address

| Weakness | Detail |
|---|---|
| Data depth | ~7 years covers three major stress regimes but not a full interest-rate cycle. Caution interpreting 2022–2023 rate-driven selloffs. |
| Universe size | 18 tickers. Liquidity and capacity are real constraints at institutional scale. |
| Beta lag | OLS betas react slowly to structural changes (asset sales, territory changes). |
| SPP / CAISO depth | LMP history shorter than ERCOT/PJM — stress betas noisier for those utilities. |
| Transaction costs | Backtest assumes zero slippage. Estimated 0.05–0.10 Sharpe impact at live costs. |

---

## Signal Improvement Roadmap

### Near-term

**1. Interest Coverage Ratio (ICR) as factor component**  
- **What:** ICR = EBIT / |Interest Expense|, cross-sectionally z-scored, 15% weight on the factor  
- **Why:** Leveraged utilities face a double hit during stress — operating cost shock + rising refinancing costs. ICR is mainly useful as a short-side discriminator: low-ICR names are structurally fragile when rates move during grid stress events  
- **Status:** Infrastructure complete (`fetch_icr()` in `data/equity_prices.py`, wired into `factor/resilience_score.py`). Off by default (`USE_ICR = False` in `config.py`, toggle with `--icr` CLI flag)  
- **Validation needed:** Run backtest with `--icr` and check IC on the short book specifically during stressed months. If stressed-month IC improves, turn on. If it degrades or is flat, drop the weight  
- **Limitation:** yfinance returns ~4–12 quarters of history per ticker. Pre-2022 backtest ICR values are the oldest available reading, not true historical — consider EDGAR for full depth  

**2. RT/DA LMP spread as 5th GSI sub-signal**  
- **What:** Real-time minus day-ahead LMP, computed hourly and aggregated daily. Measures the premium generators earn from real-time price volatility vs. day-ahead commitments  
- **Why:** RT > DA means grid is tighter than day-ahead forecast; generators profit, T&D utilities are hurt by congestion charges. Cleanest existing data separation between generator-type longs and T&D-type shorts  
- **Status:** ~90% of infrastructure exists. Need a new `ISO_RT_LOCATION_TYPE` config dict (ERCOT: `REAL_TIME_SCED`, MISO: `REAL_TIME_HOURLY`, CAISO: `REAL_TIME_15_MIN`, PJM: `REAL_TIME_HOURLY`) and a `fetch_lmp_rt()` variant in `grid_data.py`. Then add as Signal 5 in `grid_stress_index.py` with a new weight in `GSI_WEIGHTS`  
- **Spec:** See `docs/superpowers/specs/2026-05-28-congestion-spread-design.md` for the congestion/spread design that this extends  

### Medium-term

**3. Expand universe to 30–40 names**  
- Use EDGAR as backup to yfinance for financial data, removing coverage limits and extending historical depth for the ICR signal and renewable quality components  
- Prioritise adding names with CAISO and SPP ISO exposure to reduce beta noise from thin LMP history  

**4. Inter-zonal congestion spread as standalone GSI component**  
- LMP component decomposition (energy / congestion / loss) from gridstatus to sharpen the congestion sub-signal beyond the current fraction proxy  
- Planned as Phase 2 of the congestion signal. See `docs/superpowers/specs/2026-05-28-congestion-spread-design.md` for full design  

**5. EIA thermal capacity utilization**  
- Thermal utilization = thermal generation (MWh) / nameplate thermal capacity, from existing EIA data fetchers in `data/eia_data.py`  
- Additive to `build_factor()` alongside existing `renewable_quality` component. Interpret as: high thermal utilization → constrained → higher stress sensitivity for T&D, higher revenue for merchant generators  

---

## Files to Read Before Touching the Signal

| File | Purpose |
|---|---|
| `grid_resilience/config.py` | All weights, thresholds, ISO mappings — single source of truth |
| `grid_resilience/signals/grid_stress_index.py` | GSI construction — add new sub-signals here |
| `grid_resilience/factor/resilience_score.py` | Factor assembly — add new cross-sectional components here |
| `grid_resilience/data/eia_data.py` | EIA generation mix / capacity fetchers |
| `grid_resilience/data/equity_prices.py` | Equity + ICR data fetchers |
| `grid_resilience/portfolio/backtest.py` | IC computation — use `compute_ic()` to validate new signals |
