# ICR-via-EDGAR retest — results (2026-09-08)

**Status: FEASIBILITY PASS, HYPOTHESIS FAIL.** Untried-tasks-log item 2. Probe:
`research/icr_edgar_retest/icr_probe.py`.

---

## TL;DR

- **The yfinance-depth blocker that shelved `USE_ICR` is resolved.** SEC EDGAR XBRL
  `companyconcept` gives a median of **63 quarters** of interest-coverage-ratio (ICR)
  history per name across the 25-ticker `grid_resilience` universe (vs yfinance's
  documented ~4–12), with 22/25 tickers covered back to 2019 or earlier and most back to
  2008. `CIK_BY_TICKER` for the full universe and the reusable `edgar.py` helper made this
  a same-session build, not a new research project.
- **The underlying hypothesis still fails, now with no data excuse left.** Full-sample
  monthly rank-IC (regulated-only, 102 months, 2019–2026): **mean IC −0.025, t = −0.95** —
  not significant, wrong sign for the thesis, 48% of months positive (coin-flip). The
  targeted rate-hiking-cycle test (ICR as of 2022-02 vs. return 2022-03→2023-07) is
  **t = −0.24**, also flat and wrong-signed.
- **Every window that showed a large correlation is a name-specific confound, not a
  leverage effect** — PCG/EIX (California wildfire-liability litigation, unrelated to rate
  stress) anchor the low-ICR end; VST/CEG/D/ETR (AI-power-demand re-rating) anchor the
  high-ICR end. This is the same single-name-narrative contamination documented everywhere
  else in this universe (`docs/grid-stress-signal-evidence.md` §5).
- **Verdict: do not turn `USE_ICR` on.** The original limitation (thin yfinance history)
  is now moot, but removing it didn't rescue the signal — same failure mode as the 8 other
  cross-sectional attempts on this universe. Keep `USE_ICR=False`.

---

## 1. Feasibility check — does EDGAR actually solve the depth problem?

**Method:** resolved CIK for all 25 `grid_resilience` universe tickers via SEC's
`company_tickers.json`; pulled `OperatingIncomeLoss` and `InterestExpense` (+ 4 fallback
interest-expense tags) via the existing `research/electricity_earnings_exposure/edgar.py`
`quarterly_flow()` helper (discrete-quarter extraction with FY-minus-9mo Q4 derivation
already built in). ICR = operating income / |interest expense|, clipped to [0, 20] to
match the existing `fetch_icr()` contract.

**Pre-registered bar:** median coverage ≥ 20 quarters AND ≥ 15/25 tickers back to 2019.

| | yfinance (documented) | EDGAR (this probe) |
|---|---|---|
| Median quarters/ticker | ~4–12 | **63** |
| Tickers back to 2019+ | — | **22 / 25** |
| Typical earliest date | recent | 2008-06-30 (18/25 names) |

**Gaps:** `CNP` and `NEE` returned zero quarters — both fail on the interest-expense side
(`interest_tag_found=False`); worth a wider tag search if either name matters later, not
chased here since the aggregate gate already clears. `TLN` (9 quarters, 2014–2016 only —
its pre-2021-emergence-from-bankruptcy CIK) and `CEG` (13 quarters, spun off 2022) have
genuinely short histories for structural reasons, not a data-access problem.

**Verdict: PASS.** Full panel saved to `research/icr_edgar_retest/icr_edgar_panel.parquet`
(72 quarters × 23 tickers).

---

## 2. Hypothesis test — does low ICR flag better shorts during rate stress?

**Method:** the CLAUDE.md open question, tested three ways, all with a 45-day reporting
lag (matches `_ICR_LAG_DAYS` in `resilience_score.py`):

1. **Targeted event test** — ICR as of the month before the 2022 Fed hiking cycle began vs.
   cumulative return through the cycle's end (2022-03 → 2023-07), full 25-name universe and
   the regulated-only 19-name subset (excluding NRG/VST/CEG/TLN — ICR's actual intended
   domain in the `hard-switch` architecture, since merchants get stress-beta instead).
2. **Placebo windows** — same test structure over a calm pre-COVID window (2019-03→2020-07)
   and a rate-cutting window (2024-03→2025-07), to check whether any hiking-window effect
   is genuinely regime-specific or just noise that shows up everywhere.
3. **Full-sample monthly rank-IC** — Spearman corr(ICR, next-month return), regulated-only,
   102 months, 2019-01→2026-06 — the actual pre-registered-style test that matters most.

**Results:**

| test | n | Spearman ρ | t-stat | read |
|---|---|---|---|---|
| Full universe, hiking (2022-03→2023-07) | 21 | +0.017 | +0.07 | flat |
| Full universe, placebo pre-COVID | 20 | +0.027 | +0.11 | flat |
| Full universe, placebo cutting-cycle 2024-25 | 21 | +0.816 | **+6.14** | huge — but see §3 |
| Regulated-only, hiking (2022-03→2023-07) | 18 | **−0.060** | **−0.24** | flat, wrong sign |
| Regulated-only, placebo pre-COVID | 18 | +0.127 | +0.51 | flat |
| Regulated-only, placebo cutting-cycle 2024-25 | 18 | +0.777 | **+4.94** | huge — but see §3 |
| **Regulated-only, full-sample monthly rank-IC** | **102 mo** | **mean IC −0.025** | **−0.95** | **the real answer: null, wrong sign, 48% positive** |

**The hiking-cycle test — the one this was built to answer — is flat in both universes**
(t = +0.07 full, t = −0.24 regulated-only). No support for "low ICR flags better shorts
during rate stress."

## 3. Why two windows show a huge correlation — and why it isn't the ICR effect

Both "huge" numbers come from the same window (2024-03→2025-07) and evaporate on
inspection:

- **Full universe:** the bottom-ICR names are PCG (−14%) and EIX (−16%); the top-ICR names
  are led by VST (**+244%**) and CEG (+83%) — the AI-power merchant-generation re-rating,
  the exact single-name narrative contamination already documented in
  `docs/grid-stress-signal-evidence.md` §5 as the dominant failure mode for *every*
  cross-sectional signal tried on this universe.
- **Regulated-only (VST/CEG/NRG/TLN excluded):** the correlation survives almost unchanged
  (t = +4.94) — but the low-ICR anchor is still **PCG/EIX**, both depressed by the
  2024–2026 California wildfire-liability crisis (a litigation/regulatory risk, unrelated
  to interest coverage or rate stress), while the whole sector rallied 25–70% together on
  the data-center-demand narrative (WEC +40%, SO +44%, XEL +46%, ETR +71%, D +37%) — a
  sector-wide beta move, not cross-sectional discrimination by leverage.

Two unrelated, already-known idiosyncratic effects (wildfire litigation, AI-demand
re-rating) manufacture a large correlation in one three-year window; the 102-month
full-sample IC — which averages over many regimes instead of cherry-picking one — is
indistinguishable from zero. This is the same shape of failure as the capex-guidance signal
and the category-demand rotation: an apparent regime-specific hit that doesn't survive a
full-sample or sub-period test.

---

## 4. Verdict and disposition

**Do not turn on `USE_ICR`.** The specific concern that kept it off (thin yfinance history)
is resolved — EDGAR gives 5–8× the depth — but that was never the real problem. With a full
102-month, mostly-clean history, ICR shows no cross-sectional forward-return relationship in
either the targeted rate-stress window or the full sample. This closes the CLAUDE.md open
question with a real answer instead of a data-availability shrug.

**What's kept:** `CIK_BY_TICKER` (25 names) and `research/icr_edgar_retest/icr_probe.py`
are reusable if a *different* fundamentals question needs full-history data for this
universe — the EDGAR data-access pattern is now proven for it, even though ICR itself
isn't the answer. `icr_edgar_panel.parquet` is the raw panel if anyone wants to re-slice it.

**Not done:** wiring `fetch_icr()` itself over to EDGAR in production code. Given the
hypothesis failed on genuinely good data, there's no reason to — `USE_ICR=False` stays the
right default and the yfinance-backed `fetch_icr()` can stay as-is (it was never the
limiting factor).
