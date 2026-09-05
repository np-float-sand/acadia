# Henry Hub forward-gas signal probe — results (FAILED, data-limited)

**Date:** 2026-09-05
**Context:** `docs/handoff-forward-power-price-signal.md` (Phase 1 = "use long-dated Henry Hub gas
as the free proxy for long-dated power"), `docs/handoff_2026-09-01-grid-buildout-long-short.md`.
Attempt ~17 against the grid-equipment long/short's search for a signal-based edge.

---

## TL;DR

- **The instrument the thesis points at — the long end (18–36 mo) of the power / gas forward curve —
  is not free.** Confirmed three ways: yfinance drops history for expired dated NG contracts
  (only Oct-2026-forward strips have a back-history); EIA's NG futures API stops at contract 4
  (~4 months); EIA's RNGC1–4 daily series was **discontinued in 2024**. True depth needs CME
  DataMine / ICE (paid) — same status as the options hedge.
- **Tested the free near-curve proxies instead:** NG front-month 12/6/3-month change (structural
  level trend, de-seasonalised) and the RNGC4/RNGC1 4-month term-structure slope ("contango").
- **All fail the pre-registered bar.** No proxy has a forward relationship with the basket that
  clears |t| ≥ 2 on *both* sub-windows, and none survives an SMH + Δ10y-yield control.
- **Verdict: FAILED, logged.** Do not pursue further without a funded power-forward feed; even
  then the prior is now lower (the free proxies are flat, and every related instrument — PJM
  capacity auctions, MW forecasts, category-demand orders — has already failed).

---

## Results (monthly, 2019-01 → 2026-08; two sub-windows)

**Spearman IC — signal_t vs basket return_{t+1}:**

| signal | 2019–22 (r / t) | 2023–26 (r / t) | pooled t |
|---|---|---|---|
| NG 12-mo change | −0.10 / −0.67 | −0.11 / −0.74 | −1.45 |
| NG 6-mo change | +0.06 / +0.38 | −0.16 / −1.06 | −1.05 |
| NG 3-mo change | +0.06 / +0.41 | **−0.31 / −2.04** | −1.64 |
| 4-mo contango (RNGC4/RNGC1) | +0.24 / +1.61 | −0.26 / −1.01 (n=16*) | +1.22 |

\* RNGC1–4 discontinued 2024 → only 16 months in the 2023–26 window; that cell is not
interpretable.

The only |t| ≥ 2 anywhere is NG 3-mo change in 2023–26 — and it's **negative** (rising gas →
basket *underperforms* next month), the opposite sign to the thesis, and absent in 2019–22.

**Spearman IC — signal_t vs (basket − SMH)_{t+1}:** NG 12-mo change is +0.39 / t=+2.64 in
2019–22 but +0.05 / t=−0.35 in 2023–26 — one regime only, and it collapses under control:

**Control OLS — bret_{t+1} ~ signal_t + Δ10y_t + SMH_t** (signal coef t): NG 12-mo change
t = −0.83 (2019–22) / −0.50 (2023–26); contango t = +1.71 (2019–22) / −1.10 (2023–26, n=16).
Nothing ≥ 1.5 with the right sign on both windows — the raw 2019–22 spread IC was just
"gas up ≈ industrials up ≈ basket beats semis."

**Lead/lag** (NG 12-mo change vs basket, k = −3…+3): all |t| < 1.5; gas does not lead the basket.

**Scaler backtest** (exposure = clip(1 + β·z(signal), 0.3, 1.5), monthly, vs static): NG 12-mo
and 6-mo change **lose on Sharpe and Calmar on both windows**. The contango scaler "wins"
2023–26 (Sharpe 2.27 vs 1.43) but on 16 months, with a signal whose IC that window is
*negative* — a de-lever-by-luck artifact, not signal.

---

## Why it fails / what's untested

1. **Wrong part of the curve.** Free data only reaches ~4 months out. The thesis is about the
   *structural* view priced 2–5 years out; that is exactly the segment that costs money.
2. **~75% gas beta anyway.** Even a true long-dated power forward would be mostly "the gas curve
   moved"; the demand/scarcity signal lives in the **back-end spark spread / implied heat rate**,
   which needs *both* power and gas forwards — doubly blocked.
3. **Related instruments already failed:** PJM capacity-auction prices (lag 18–24 mo, inverted —
   `capex-cycle-pair-classifier`), PJM MW-revision forecasts (`pjm-mw-revision-signal-negative`),
   national equipment-category demand (`category-demand-rotation-negative`).
4. Front-month gas is a macro/seasonal series heavily driven by weather and storage, only loosely
   coupled to a multi-year industrial capex cycle.

## Recommendation

Logged as a data-blocked negative. Re-open only if a CME DataMine or ICE power-forward feed is
funded — and scope it then as a *basket-level confirmation gauge* (spark-spread level at the
2–4-year point), not cross-sectional alpha. Standing conclusion unchanged
(`docs/handoff_2026-09-01…` §7): ship the concentrated long/short as a discretionary,
risk-managed thematic position.

## Round 2 (2026-09-05) — smoothing + alternative sources

Re-ran with **UNL** (US 12-Month Natural Gas Fund = a 12-contract strip, deseasonalised ~1-year-forward
gas, free, 2010→), smoothing variants (3-mo MA, EMA-6 momentum, 36-mo rolling z), and water-equity
ETFs (PHO/FIW/CGW). No change to the verdict:

- UNL 12/6/3-mo change and the smoothed variants: forward-return IC |t| < 1.6 on both sub-windows;
  smoothing makes the cross-regime sign *less* stable, not more. The 2019–22 "+0.39 vs basket−SMH"
  reappears and again dies under the SMH + Δ10y control (coef t ≈ −1.4) and is absent in 2023–26.
- Water ETFs are water-*equity* baskets — rank-corr +0.25 (1-mo) to +0.49 (3-mo) with the grid
  basket (shared XYL/PNR), i.e. contaminated industrial/utility beta, no forward IC.
- FCG / UNG add nothing over NG=F front-month.

Smoothing and alternative free sources do not rescue it; the genuinely-forward segment of the curve
remains the paywalled part.
