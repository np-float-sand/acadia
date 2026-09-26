# Merchant stress-beta whipsaw fixes — results (2026-09-08)

**Status: fix #2 FAILED, fix #1 marginal/NOT ADOPTED, fixes #4/#5 not yet tested.**
Untried-tasks-log item 3. Probes: `research/merchant_beta_orthogonalization/probe.py`
(fix #2), `probe_fix1_xlu_xng_filter.py` (fix #1).

---

## TL;DR

- Tested the "most elegant" of the five documented candidate fixes for the VST/NRG
  stress-beta whipsaw (`docs/grid-stress-signal-evidence.md` §5): add each ISO's own
  slow-moving power-price trend as a second regressor in the stress-beta OLS, so the GSI
  coefficient is orthogonalized to the market's structural power-price narrative instead of
  conflating the two.
- **The fix barely moves the actual beta estimates.** VST's orthogonalized beta correlates
  0.996 with the original single-variable beta (mean move 7.2%); NRG 0.986 (11.7%). The
  merchant names' rolling beta series — including at every documented narrative-flip date
  (late-2023, 2024-05, 2025-01 DeepSeek, 2025-05) — is essentially unchanged.
- **The one aggregate metric that matters got worse, not better.** IC@21d t-stat: 1.352 →
  1.027. IC@63d t-stat: 1.889 → 1.782. Full-period Sharpe ticked up from −0.007 to +0.046 —
  noise-level, and not the metric this fix was supposed to move.
- **Pre-registered gate: FAILED** (IC did not improve; the two required conditions both
  miss). Do not adopt.
- **Why it failed, and what's still worth trying:** a trailing 252-day *realized* LMP trend
  is a weak proxy for the actual mechanism — an AI-power-demand *valuation narrative*, which
  is forward-looking and event-driven (a specific quarter's guidance, a restart
  announcement), not something a smoothed trailing spot-price statistic captures. This
  points at item 6 (STEO forward-power series) as the more theoretically appropriate
  "structural" regressor, not a re-run of this same trailing-price idea with a different
  window.

---

## 1. Method

Reused the real, unmodified `grid_resilience.main.run()` pipeline end to end (equity
prices, GSI construction, factor scoring, portfolio backtest, IC test all identical to the
documented baseline) and monkeypatched only `conditional_beta.compute_stress_betas` — the
function `rolling_stress_betas` calls at each monthly rebalance — with a two-variable
variant:

```
excess_return_t = alpha + stress_beta * GSI_t + trend_beta * PowerTrend_t + epsilon_t
```

restricted to the same high-stress days (`GSI >= 0.5`) as production. `stress_beta` (the
GSI coefficient) is what feeds the factor; `PowerTrend` is discarded downstream except for
diagnostic inspection.

**PowerTrend construction** (ERCOT + PJM only, the two ISOs with merchant contamination):
63-day-smoothed daily mean LMP → trailing 252-trading-day % change → rolling-756-day
z-score (matches the GSI's own scaling convention), clipped to ±3. Falls back to the
original single-variable regression when trend coverage is thin (<50% of the stress-day
window) — this never actually triggered for VST/NRG (75/93 and 74/93 months had a live
`trend_beta`).

Ran both variants (baseline, orthogonalized) through the identical harness and compared:
(a) the full performance/IC report, (b) the merchant names' (VST, NRG, CEG, TLN) rolling
beta series directly, date by date.

## 2. Results

**Aggregate metrics:**

| metric | baseline (GSI only) | orthogonalized (+ power trend) |
|---|---|---|
| Full Sharpe | −0.007 | +0.046 |
| Full ann. return | 3.90% | 4.71% |
| Full max DD | −31.4% | −32.4% |
| Stress-period Sharpe | −0.145 | −0.109 |
| Calm-period Sharpe | 0.379 | 0.516 |
| **IC@21d (t-stat)** | **0.0628 (1.352)** | **0.0473 (1.027)** |
| **IC@63d (t-stat)** | **0.0953 (1.889)** | **0.0878 (1.782)** |

Sharpe moved in the right direction but by an amount indistinguishable from noise on a
93-month backtest with one parameter change. **IC — the metric this fix targeted — moved
the wrong way at both horizons.**

**Merchant-name beta stability** (93 monthly observations, VST/NRG; 45/30 for the
shorter-lived CEG/TLN):

| ticker | corr(baseline, orthogonalized) | mean |Δ| | relative change |
|---|---|---|---|
| VST | 0.996 | 0.0019 | 7.2% |
| NRG | 0.986 | 0.0027 | 11.7% |
| CEG | 0.962 | 0.0043 | 11.0% |
| TLN | 0.798 | 0.0057 | 45.3% |

VST and NRG — the two names the fix was specifically built for — show almost no change.
Spot-checking every documented flip date (2023-09, 2023-12, 2024-04, 2024-06, 2024-12,
2025-02, 2025-04, 2025-06) confirms the beta values match the baseline to within
estimation noise at each one; the sign-flip count is identical for VST (9 in both) and
essentially unchanged for NRG (8 → 10, i.e. no reduction). Only TLN moved meaningfully
(corr 0.80), on a 30-observation sample too small to read anything into.

## 3. Why the fix didn't engage

The trend regressor's own coefficient (`trend_beta`) is consistently small
(|value| mostly < 0.02 vs `stress_beta`'s own similar scale) and only weakly correlated
with GSI in this data — so orthogonalizing against it barely reallocates any explanatory
power. The diagnosed mechanism in `grid-stress-signal-evidence.md` §5 is a **narrative**
flip (the market re-rating VST/NRG's AI-power exposure from liability to asset), which is
forward-looking, discrete, and event-driven (a specific restart announcement, an earnings
beat, a hyperscaler contract). A trailing 252-day realized spot-price trend is a poor proxy
for that — it measures what power already cost, not what the market now expects it to be
worth. This is the same gap the June-2026 forward-power-price handoff diagnosed and the
same reason the Henry Hub trailing-gas proxy failed as a basket-timing signal
(`docs/hhub-forward-gas-signal-results.md`): the informative part of this story lives in
the **forward-looking** segment of the price curve, and a trailing/realized statistic
doesn't reach it no matter how it's smoothed.

## 4. Disposition

**Fix #2 (orthogonalize beta to trailing power-price returns): closed, FAILED.** Do not
merge into `conditional_beta.py`.

**Redirects, not dead ends:**
- The real test of "does a *forward-looking* power-price series stabilize the merchant
  beta" is item 6 (STEO `ELWHU_*`, ~16-month genuine forecast tail) — this probe's null
  result is evidence *for* trying that, not against it, since the mechanism tested here
  was explicitly the wrong (trailing) half of the curve.
- Fixes #1 (XLU/XNG correlation filter), #3 (regime flag), #4 (structural-break-adaptive
  window), and #5 (fundamentally-anchored beta) remain untried and are not informed by this
  result — they don't rely on a price-trend regressor at all.

Kept: `research/merchant_beta_orthogonalization/probe.py` (the monkeypatch-based harness
pattern is reusable for testing any beta-estimation variant against the real pipeline
without touching production code) and both rolling-beta parquet dumps for further
inspection if needed.

---

# Fix #1: XLU/XNG correlation exclusion filter — results (2026-09-08)

**Status: works as diagnosed, but the effect is too small to change the strategy's
verdict. NOT ADOPTED for now** — kept as a documented, validated candidate.

## Method

Rather than fix the beta estimate, exclude a ticker from that month's long/short
*eligibility* whenever it's demonstrably trading like gas, not like a regulated utility:
`corr(stock, ^XNG) > corr(stock, XLU)` (63-trading-day rolling correlation) for 2
consecutive rebalance months. Implemented by monkeypatching
`grid_resilience.main.build_rolling_weights` to drop the flagged (date, ticker) rows from
`rolling_factors` before ranking — the factor score itself, the GSI, and the IC test all
stay untouched.

**Important methodological note:** `compute_ic()` is called on the *unfiltered*
`rolling_factors` (`main.py:491`), upstream of `build_rolling_weights` — so unlike fix #2,
this fix **cannot move the IC numbers by construction**, and the pre-registered "IC
improves" bar from fix #2 doesn't apply here. The correct test for an eligibility filter is
the realized backtest performance, which is what's reported below.

## Results

**Validity sanity check: clean pass.** Of 1,171 ticker-months, exactly 63 were flagged —
**100% on the four known merchant names** (NRG 23, VST 19, CEG 12, TLN 9), zero on any of
the 21 regulated names. The proxy is doing exactly what it's supposed to: it isolates the
names already known to have the AI-power/merchant re-rating problem, with no false
positives on the regulated book.

| metric | baseline (no filter) | fix #1 (XLU/XNG filter) | fix #2 (price-trend orthogonalization) |
|---|---|---|---|
| Full Sharpe | −0.007 | **+0.050** | +0.046 |
| Full ann. vol | 14.46% | 14.31% | 15.49% |
| Full max DD | −31.4% | −31.4% (unchanged) | −32.4% |
| **Stress-period Sharpe** | **−0.145** | **−0.085** | −0.109 |
| Calm-period Sharpe | 0.379 | 0.449 | 0.516 |

Fix #1 beats fix #2 on the metric that matters most for this diagnosis — stress-period
Sharpe improves by about 40% of the baseline's shortfall (−0.145 → −0.085) — and does so
with a clean, interpretable, zero-false-positive mechanism rather than a regressor that
barely engages.

## Verdict

**Real, positive, mechanistically sound — but too small to change anything.** Full Sharpe
of +0.050 is still nowhere close to XLU's 0.261 benchmark, and max drawdown is unchanged
(the −31.4% drawdown episode doesn't overlap with the 63 flagged ticker-months). This is
consistent with the 2026-08-26 closeout audit's conclusion that the constraint is signal
strength / sample size, not construction: even a fix that correctly and precisely isolates
the known problem names moves the needle by only a few points of Sharpe, not enough to
cross the beats-XLU bar.

**Disposition: NOT ADOPTED**, but flagged as the best of the constructions tried so far —
if the GSI signal is ever revisited for a reason other than this whipsaw (e.g. paired with
a genuinely new signal), this filter is a cheap, validated, no-downside addition (it never
hurts max drawdown and improves stress-period behavior) worth carrying along rather than
re-deriving. Kept: `research/merchant_beta_orthogonalization/probe_fix1_xlu_xng_filter.py`,
`xlu_xng_exclude_mask.parquet`.

---

# Fix #4: structural-break-adaptive (exponentially-weighted) beta window — results (2026-09-08)

**Status: FAILED, decisively.** Probe: `probe_fix4_ewls_beta.py`.

## Method

Replaced the flat, equal-weighted 252-day OLS with a weighted least squares regression
using the same window and stress-day filter, but exponentially decaying weights
(halflife = 63 trading days) so a recent regime dominates the slope estimate faster than
an equal-weighted window would let it. No other pipeline stage touched.

## Result

Every metric got substantially worse, not better:

| metric | baseline | fix #4 (EWLS, 63d halflife) |
|---|---|---|
| Full Sharpe | −0.007 | **−0.448** |
| Stress-period Sharpe | −0.145 | **−0.597** |
| IC@21d (t-stat) | 0.0628 (1.352) | 0.0369 (0.809) |
| IC@63d (t-stat) | 0.0953 (1.889) | 0.0563 (1.202) |
| Full max DD | −31.4% | −30.0% (marginally better, incidental) |

## Why it failed

The stress-day filter (`GSI >= 0.5`) already restricts the regression to a thin subset of
the 252-day window; overlaying an aggressive 63-day-halflife recency weight on top of that
sparse sample shrinks the *effective* sample size further and lets a handful of recent,
possibly idiosyncratic high-stress days dominate the slope. It buys faster adaptation to a
real regime change at the cost of far more noise on every window that isn't mid-flip — and
the useful stress-day windows are already too sparse to absorb that trade. Recency-weighting
a well-populated window (e.g. all trading days, not just stress days) might behave
differently, but that changes the estimator's fundamental design (whether beta is
"conditional on stress days" at all), not a parameter tweak — out of scope for this probe.

## Disposition

**FAILED.** Do not adopt. Confirms the general shape of this whole research line: making
the estimator more locally reactive trades bias for variance in a sample that's already
too small to afford it.

---

# Item 3 synthesis — where this lands

| fix | status | verdict |
|---|---|---|
| #1 XLU/XNG correlation filter | tested | works exactly as diagnosed (100% precision on merchant names), stress Sharpe −0.145→−0.085, but too small to beat XLU — NOT ADOPTED |
| #2 orthogonalize vs. trailing power-price trend | tested | FAILED — barely moves the beta (corr 0.996 with baseline), IC gets worse |
| #3 power-price momentum / forward-curve regime flag | redirected | folded into item 6 (STEO forward series) — see below, now tested |
| #4 structural-break-adaptive (exponentially-weighted) window | tested | FAILED, decisively — full Sharpe −0.007→−0.448, IC worse at every horizon |
| #5 fundamentally-anchored beta (revenue % spot-exposed, fuel mix) | tested 2026-09-18, corrected 2026-09-20 | **FAILED both variants** (iso-blend, iso-blend+shrink) — corrected, main-checkout-verified numbers (original worktree run had a bad cache, caught via a benchmark-ETF sanity check): full Sharpe 0.120→−0.032→−0.584 (both variants now negative); IC@21d t *improves* 0.34→0.499→0.721 while IC@63d worsens 1.207→0.393→−0.570. `docs/merchant-fundamental-beta-results.md` §7. |
| item 6 STEO forward spark-spread (fix #3's redirect target) | tested 2026-09-19 | **PARTIAL** — first fix in the series with a genuine IC improvement (IC@21d t 1.352→2.427, clears \|t\|≥2.0) and a large MaxDD cut (−31.4%→−15.9%), but full/stress Sharpe both tick slightly worse, so the strict two-part gate isn't fully met. NOT ADOPTED as-is; flagged as the most promising lead in the series. `docs/steo-spark-spread-results.md`. |

**Bottom line, updated 2026-09-19 — all 5 fixes plus the item-6 redirect now tested.** One
(#1) works exactly as diagnosed but is too small to matter; three (#2, #4, #5) make things
worse on every metric. The one exception is the STEO forward-power spark-spread (fix #3's
redirect target): it is the first candidate in this whole line to move IC in the right
direction by a statistically real amount and to meaningfully cut drawdown, even though
realized Sharpe doesn't follow — worth a narrower follow-up (using the improved ranking
for sizing/vol-targeting rather than the return-based gate this series has used
throughout) rather than being written off with the rest. Otherwise this closes out
consistent with the 2026-08-26 closeout audit's conclusion that the GSI's problem is
signal strength / sample size — construction fixes aimed at the known merchant-whipsaw
mechanism top out at a few points of Sharpe, nowhere near the ~26-point gap to XLU.
