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

## Round 3 (2026-09-05) — a free forward *power* series does exist: EIA STEO ELWHU_*

**EIA Short-Term Energy Outlook** publishes monthly wholesale electricity price series per hub,
each with a **~16-month forward forecast tail**, 2010→present, free via the API key in `.env`:

| series | hub | span |
|---|---|---|
| `ELWHU_PJ` | PJM RTO Western hub | 2010-01 → 2027-12 |
| `ELWHU_TX` | ERCOT North hub | 2010-12 → 2027-12 |
| `ELWHU_CA` | CAISO SP15 | 2010-01 → 2027-12 |

(+ `ELWHU_NE/NY/MW/NW/SW/SE`.) Endpoint:
`https://api.eia.gov/v2/steo/data/?frequency=monthly&data[0]=value&facets[seriesId][]=ELWHU_PJ&api_key=...`

It is an EIA-modeled, futures-informed **forecast**, not a market clearing price — but it is
power-specific, genuinely forward (~16 mo), hub-level, covers every DC-heavy market, and has
15 years of history. First free series in this thread that is forward + power + has history.

**Backtest caveat:** a single API pull returns only the *latest* vintage, whose forecast tail is
hindsight-revised. A point-in-time test needs the STEO monthly archive
(`eia.gov/outlooks/steo/archives/`, ~90 files for 2019–2026), extracting each vintage’s
12–16-month-ahead `ELWHU_*` forecast. ~half a day.

### Scoped next step (not yet run)

1. Scrape the STEO archive; build a point-in-time panel: for each vintage month v, the v+12 and
   v+16 forecast for `ELWHU_PJ`, `ELWHU_TX`, `ELWHU_CA`, plus the implied slope (v+16 forecast ÷
   spot).
2. Signal = 3–6-month change in the v+12/v+16 forward forecast (level and slope), z-scored.
3. Test vs the 9-name basket forward 1m/3m return, both sub-windows (2019–22, 2023–26):
   pre-registered bar rank-IC |t| ≥ 2 on both windows, **survives a control for the 12-month
   Henry Hub gas-strip change and SMH** (the residual power-specific / implied-heat-rate move is
   the only part that could be new — the gas-curve component is already known to be flat, Rounds
   1–2), and a scaler that beats price-gate+vol-target on Sharpe AND Calmar on both windows.
4. Kill: IC |t| < 1.5 pooled, or works one window only, or fully explained by the gas-strip +
   SMH control → log, done.

**Other free forward-tightness indicators** (not power price, but forward scarcity): ERCOT CDR
planning reserve margin (semi-annual, no gas contamination); ISO-NE Forward Capacity Auction
clearing prices (3-yr forward, history to 2008); NYISO ICAP. CME publishes its power forward
curve as free *delayed* quotes — no history but capturable going forward.

## Round 4 (2026-09-05) — nonlinear / combined (GBM + RF)

Put all session buildout-heat signals (25 features: NG/UNL 3-12mo changes, EMA, strip slope,
STEO ELWHU_PJ/TX 3-12mo changes, PJM & ERCOT spark-spread proxies, 4 category-demand momentum
z-scores + composite, SMH, d10y, basket own momentum) into a walk-forward TimeSeriesSplit(5)
GBM and RandomForest predicting basket forward 1-month return.

| window | GBM OOF rank-IC | RF OOF rank-IC | R2 | perm p(IC>=obs) |
|---|---|---|---|---|
| 2023-26 (n=43) | +0.18 | +0.29 | ~0 | 0.08 / 0.00 |
| 2018-26 (n=98) | +0.02 | +0.00 | negative | 0.37 / 0.42 |

The 2023-26 flicker beats the permutation null but **vanishes entirely on the extended 2018-26
sample**, R2 ~= 0 throughout, and feeding the GBM walk-forward predictions back as an exposure
signal **loses to static** (Sharpe 1.04 vs 1.10). Overfit to the 43-month window, not signal.

**Inverted composite overlay** ("fade the buildout-heat"): rank-IC(heat, fwd-ret) = +0.08 in
2023-26 (mildly positive -> fading does not help), -0.01 in 2018-26. Overlay Sharpe 1.32 vs
static 1.31 vs vol-target-only 1.22; active 80-93% of months. Adds +0.01. Nothing.

The "9-of-10 negative sub-threshold cells" pattern (crowding / already-priced) lives in the
*other session*'s 15-utility capex-guidance panel, not in these gas/power/category features (where
the composite is weakly positive 2023-26). The one remaining untested combination: union both
feature sets into one walk-forward GBM, both windows. Needs `capex_guidance_signal.py` output
exported to run in one place.

**Cross-session stop-rule (per the Deliverable-D synthesis):** three independent demand-data
classes -- forecast (PJM Table B-9), physical-flow (congestion, FTR), commitment (capex
guidance) -- have now each failed to time this basket under pre-registered discipline. The
monthly-resolution probes do not even reproduce the spurious +0.8 annual k=-1 artifact the VA /
Table-B-9 work threw. Defensible prior: the timing edge is not reachable with available data.

## Round 5 (2026-09-05) — IS/OOS split WITHIN the AI regime (2023-26)

Per "2018 was a different regime, split 2023-26 for in/out-of-sample":

| protocol | model | test rank-IC | perm p | overlay Sh vs static |
|---|---|---|---|---|
| train 2023-24 -> test 2025-26 (n=19) | Ridge 25f | +0.29 | 0.11 | 1.53 / 1.15 |
| | GBM 6f | +0.18 | 0.19 | 1.08 / 1.15 |
| | RF 6f | +0.18 | 0.19 | 1.11 / 1.15 |
| train 2025-26 -> test 2023-24 (n=24) | RF 6f | +0.40 | 0.18 | 1.94 / 1.80 |
| | Ridge 6f | +0.40 | 0.03 | R2 = -0.65 (broken) |
| expanding walk-fwd in 2023-26 (OOS n=25) | GBM 25f | +0.32 | -- | -- |
| | GBM 6f | +0.06 | -- | -- |

(LEAN 6 features = pjw_chg6, sparkPJ_chg6, unl_ema6, catscore, ng_chg12, d10y.)

Everything leans **mildly positive** (dir-acc 0.58-0.71, ICs positive, overlays >= static -- not
anti-predictive), but **nothing clears a permutation null** (best real p ~= 0.11); the FULL-model
walk-forward +0.32 collapses to +0.06 with a sensible 6-feature model (25 features exploiting
noise on ~20 training rows); the one p=0.03 cell has R2 = -0.65. At n=19-25 with target
autocorrelation, a faint positive tilt is indistinguishable from momentum structure. Splitting
the AI regime does not rescue it.

**Not included:** the other session's IESC/EME gross-margin vs DC-hub QCEW-wage-index feature
(+4.19 level / +2.66 spread, the only cell same-sign pre/post-2022). Different pipeline, different
universe (contractor sleeve, not the 9). Its own OOS test should cover: 2023-26 IS/OOS half-split;
Bonferroni for 1-of-24 (|t|>=2.9 -> level clears, spread does not); and whether it predicts
contractor-sleeve *returns*, not just co-moves with margins. The one remaining combined test is
unioning that feature set with these STEO/gas/category features in one walk-forward model.
