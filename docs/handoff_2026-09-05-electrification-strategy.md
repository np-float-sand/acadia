# Handoff — electrification-equipment strategy: state + open short-leg ideas

**Date:** 2026-09-05
**For:** a fresh conversation to review (a) the strategy as it stands — long-only and the
modified long/short — and (b) the externally-suggested hedge/short ideas below.
**Read first:** `docs/electrification-ls-strategy-note.md` (§§1–8, the full investigation),
`docs/electrification-strategy-proposal.html` (the 1-page IC proposal + charts),
memory `capex-cycle-pair-classifier`, `docs/handoff_2026-09-01-grid-buildout-long-short.md`.

---

## 1. Where the strategy stands

### 1a. Long-only core — the proposal

Rules-based, fully mechanical. **Universe:** US names in ≥2 of the electrification ETFs
{VOLT, ELFY, ZAP, GRID}, filtered to GICS Electrical Equipment + Construction & Engineering;
point-in-time membership + 42-day reporting lag. **Weighting:** equal-weight, 25% cap,
quarterly reconstitution. **Risk:** 20% annualised vol target (1.5× cap, slack at rf).
**Overlay:** 0–15% diversifier sleeve (GLD + short Treasuries) sized by a real-yield +
HY-credit-spread signal.

Backtest 2016–2026: **CAGR 17.3% / Sharpe 0.71 / MaxDD −25.9% / SPY β 0.78**; positive Sharpe
every sub-window (2016–18 0.29, 2019–22 0.54, 2023–26 1.48) though behind SPY in the 2016–18
low-vol bull. Since VOLT's Dec-2024 launch: strategy +43.5% / Sharpe 0.90 / DD −19.4% vs
VOLT +42.3% / 0.76 / −23.4% — **matches the ETF on return, better risk control**; the edge over
just buying VOLT is discipline + a track that reconstructs to 2016, not return.

Honest label: **enhanced thematic beta, not alpha.** Primary risk: a sustained falling-rate,
risk-on reversal (washed-out clean-energy rallies, theme de-rates).

### 1b. Long/short, modified — conditional rate-regime short

Long the ~33 industrial suppliers / short the consumer clean-energy sleeve
(ENPH SEDG RUN NOVA CHPT EVGO BLNK STEM), **short weight scaled 0 → 0.55× by the 6-month change
in the 10-year real yield** (rising → short on; falling → short off, book reverts to long-only).
Whole thing vol-targeted.

2013–2026: **CAGR 14.0% / Sharpe 0.62 / MaxDD −31% / SPY β 0.70** — i.e. same Sharpe & CAGR as
long-only, *wider* drawdown, β essentially unchanged. Short-leg contribution: **+3.4%/yr in
2023–26 (t=2.6)**, −1.0 to −1.1%/yr in each pre-AI window; **full-cycle +0.1%/yr, t=0.1**.
Net long exposure median 0.95 (min 0.69) — it never actually becomes market-neutral because the
legs are +0.45 correlated and the short is only ~0.45× notional.

**Verdict:** the conditional structure is right; with *only* the rate signal it does not earn its
place. Documented as a Phase-2 overlay pending more consumer-vs-industrial signals
(relative valuation, earnings-revision breadth, policy calendar, credit).

### 1c. The short-leg search — CLOSED as of this handoff

~25 short/hedge candidates tested (note §§6–8): clean energy (solar/EV/storage/hydrogen,
+0.25–0.6 corr, 2023–26 only), firm generation (gas turbine / nuclear / IPP, +0.60 corr, falls
harder in drawdowns), 7 other industries (only aero-supply/airlines works standalone, and
blending it with others destroys it), China broad (least correlated at +0.03 recent, positive
every window, **but** 0.85 SPY β, recurring fat left tail when China decouples and rallies, and
no exit signal works). **No holdable negatively-correlated leg exists** — only VIXY (−0.60, but
−30%/yr carry) and USD (−0.31, faded post-2022) are genuine negatives, both tactical.

---

## 2. Externally-suggested short/hedge ideas (Google AI, 2026-08, re: "anti-correlated to the
data-center trade") — for the next conversation to weigh

The framing: rather than one anti-correlated ticker, a "structural hedge" from the
data-center-moratorium / local-pushback storyline (real 2026 news — protests, state/county
moratoriums in Virginia, Georgia, Ohio). Four legs suggested:

| # | suggested trade | investable? | quick assessment / test result |
|---|---|---|---|
| A | **Short** speculative / smaller-scale data-center *developers* and "high-grid-dependency suppliers vulnerable to zoning bans & permit denials" | **Weak.** DC developers are mostly private (QTS, Vantage, Aligned, CloudHQ) or inside DLR/EQIX (which are *strong*, not vulnerable). "Vulnerable grid-dependent suppliers" ≈ **our long book** (VRT, GEV, ETN, PWR) — this is "short what you're long." | Not coherent as a hedge for this strategy. The only cleanish version: short DLR/EQIX vs long the equipment names — untested, but both are the same trade. |
| B | **Long** "consumer-advocacy / ratepayer baskets" that profit if regulators cap industrial power allocation | **Not investable.** There is no ratepayer equity. Nearest proxy = regulated distribution utilities protected by rate caps = XLU names. | XLU tested: +0.31–0.49 corr with the long book, low-beta bond proxy — not a hedge (note §6). |
| C | **Long** alternative real estate — standard retail / suburban-office REITs — in regions where DC moratoriums redirect local capital | **Testable, and tested.** | Retail+office REIT basket (SPG KIM REG FRT MAC BXP VNO CUZ HIW DEI KRC): **corr +0.58**, L/S Sharpe 13-18 **+0.09** / 19-22 +0.92 / 23-26 +1.37 / full +0.70; falls −5.2% in the long book's worst months. Office-only +0.72 full; XLRE +0.73 full; XLP (staples) lowest corr +0.43 but 13-18 −0.19. **Same pattern as all prior candidates** — correlated rate-sensitive equity, flat pre-2019, works 2019+; not a hedge. |
| D | **Policy / moratorium** angle directly — states enacting moratoriums, utilities facing local pushback | **A tail risk to the long book, not a trade.** No clean listed beneficiary of a moratorium. If moratoriums bite materially, the effect is a de-rate of the long book — which is already the disclosed primary risk. | Monitor as a risk factor (permit-approval counts, moratorium tracker), not a position. |

**Net:** none of the four is an investable, orthogonal hedge. (C) is the only new testable
idea and it behaves exactly like the 25 candidates already rejected — +0.5 correlated,
regime-dependent, not a hedge. (D) reinforces that the moratorium storyline belongs in the
**risk-monitoring framework** for the long-only book, not as a short leg.

---

## 3. Questions for the next conversation

1. **Ship the long-only?** It is defensible now as enhanced thematic beta. Decide whether that
   clears the bar for a paper-trading track, or whether it needs a differentiator first.
2. **Is the Phase-2 conditional short worth the signal-research spend?** It needs signals beyond
   rates; every signal tried in ~20 prior attempts failed OOS. Low prior.
3. **Point-in-time universe** — the current backtest uses today's ETF holdings + hand-picked
   shorts. Rebuild membership from historical holdings / GICS vintages before any live decision.
4. **Moratorium risk monitoring** — build a simple tracker (state/county moratorium status,
   permit-approval counts in Loudoun / Prince William / Central OH / N. Texas) as a de-risk input
   for the long-only book, per idea (D).
5. **Aero-supply / airlines** — bookmarked as a *separate* L/S book (the one cross-industry
   supplier/downstream pair that is positive every regime, full Sharpe ~0.80). Different mandate;
   worth its own evaluation.

## 4. What's built (in `scratchpad/xsec/`, throwaway) vs. in the repo

- **Repo:** `docs/electrification-ls-strategy-note.md`, `docs/electrification-strategy-proposal.html`
  (+ `p_cumret.png`, `p_rollsharpe.png`, `p_voltcloseup.png`), this handoff.
- **Scratchpad only (not committed):** `ls_strategy.py`, `mom_stress.py`, `pairs.py`,
  `factors.py`, the conditional-L/S and REIT test scripts. No production module yet — that is
  step 1 of "turning it into a real strategy" (proposal §6).
