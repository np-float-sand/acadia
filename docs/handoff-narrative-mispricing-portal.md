# Handoff → main strategy portal: narrative-mispricing thread (item 9/10)

**Date:** 2026-09-08. **One doc, self-contained.** Carry this back to the portal;
the detail lives in the linked files.

---

## Bottom line for the portal

The "AI-power narrative mispricing" short thesis, tested as a signal on its own
(EDGAR filing mentions + GDELT news volume), **FAILED** every pre-registered bar —
same as the earnings-exposure channel before it. **But** a side result is worth
keeping on the portal's radar: a multivariate model over **weak individual signals
in combination** — merchant stress-beta + interest-coverage ratio + EDGAR
disclosure counts + price momentum — produced the **first out-of-sample result in
this research line's ~28 tracked attempts to clear a shuffled-label null cleanly**.
It is **not actionable** (6 months / ~50 observations, one train/test split, flat on
the pre-2023 sample) and **not a narrative signal** (the attention data is not what
drives it). Treat it as a lead to re-check quarterly as 2026 data accrues, not a
strategy to fund.

**Pre-registered verdicts:** narrative-as-signal = **FAILED** ·
weak-signals-in-combination (Fold B) = **PROMISING / NOT ACTIONABLE** ·
merchant-confound explanation = **RULED OUT** ·
3-feature-core-is-the-whole-story = **NOT CONFIRMED** (see isolation check below).

---

## What was tested (the whole arc)

| step | what | verdict |
|---|---|---|
| Approach A | narrative-extremity short: cross-sectional rank on the EDGAR+GDELT composite | NOT ADOPTED — rank-IC \|t\|<1.4 every horizon, before and after fixing VST's measurement gap |
| Approach C (broadened) | narrative + every alignable built feature (ICR, stress-beta, GSI factor, momentum, 5 macro) → walk-forward GBM/RF, FULL (2018-25) vs RECENT (2023-25) windows | FAILED — RECENT cells looked strong (rank-IC t up to 3.7) but FULL window flat/negative everywhere; the Henry Hub overfit shape |
| Two-fold walk-forward | proper train→test folds: Fold A (2022-23→2024), Fold B (2024-25→2026) | Fold A weak (early regime). **Fold B: OOS AUC 0.758**, see below |
| Merchant-exclusion check | re-run Fold B dropping VST/CEG/TLN/NRG (the recurring confound on this universe) | Fold B **survives** — AUC 0.719 on 19 regulated names, still 100th-pct vs null |

---

## The promising result (Fold B), precisely

- **Setup:** GBM classifier, target = top-vs-bottom-tercile 3-month forward return.
  Train 2024-01→2025-12, test 2026-01→2026-08. 7 features.
- **OOS AUC = 0.758** (0.719 on regulated-only, VST/CEG/TLN/NRG excluded).
- **Stress tests it clears:** sits at the **100th percentile of a 50-draw
  shuffled-label null** (null mean 0.49, std 0.09); **stable across 10 seeds**
  (0.751–0.759); **not one-month-driven** (per-month AUC 0.55–1.00, all ≥0.5).
- **What drives it (drop-one-feature ablation):** `stress_beta_z` is load-bearing
  (dropping it: 0.758→0.694). `icr_z` and `edgar_z` contribute. `mom_z` and
  `xlu_xng_spread` look like noise here (dropping them *improves* AUC) — but the
  isolation check in next-step 2 shows that read is too strong: those features do
  real work *in combination*. `gdelt_z` is **stale** in the test window (2026
  extension never completed — rate-limited) so its role is uninterpretable.
- **So the finding is:** *merchant stress-beta (individually IC≈0.14, t≈1.9 at its
  best) + ICR + EDGAR disclosure counts, in combination, separate 2026 winners from
  losers among grid utilities* — confirming the "weak signals gain power combined"
  hypothesis, but on stress-beta, **not** on the narrative/attention data this
  thread set out to test.

## Why it is not actionable

- **n = 48–54 test observations across 6 months.** Six independent time periods
  cannot confirm a discovered pattern, however cleanly it clears a noise floor —
  short winning streaks are what small samples produce.
- **Flat on the full 2018–2025 sample.** This is an AI-regime-only effect on a thin
  window — exactly the profile the pre-registered both-windows bar exists to not
  reward.
- **One train/test split.** Not yet re-derived on an independent fold.
- **GDELT stale** for the entire test window; one of the seven features is frozen.

## Next steps to promote it

1. **Let the window grow** — re-run the exact Fold B quarterly as 2026 accumulates
   realized 3-month-forward outcomes. Still the single highest-value cheap check —
   not done.
2. **Isolate the 3-feature interaction** (stress-beta × ICR × EDGAR only) — **done
   2026-09-08, result is mixed, not confirmatory.** On the full 23-name universe the
   isolated 3-feature model is much weaker than the 7-feature one (AUC 0.567, only
   78th-pct vs. its own shuffled-label null — nowhere near the 100th-pct the full
   model clears). On the 19 regulated-only names it holds up close to the full model
   (AUC 0.645 vs 0.670, ~96-98th pct), but one test month (2026-06, AUC 1.00) carries
   a lot of that. **Revises the read**: the original drop-one-feature ablation made it
   look like momentum/XLU-XNG-spread/ancillary-pricing were pure noise and the
   3-feature core was the "real" signal — isolating them shows the fuller feature set
   is doing real work, at least on the full universe. Treat "stress-beta+ICR+EDGAR"
   as *necessary but not sufficient*, not as the whole story.
   Script: `research/narrative_mispricing/foldb_3feat_isolation.py`.
3. **Fix GDELT 2026** (more patient retry or a different API) before trusting or
   discounting its role. Not done.
4. **Do not wire anything live.** A single small stress-tested OOS result is a lead,
   not a strategy — the isolation result above is a reason for more caution, not
   less.

## Pointers

- Detail: `docs/ml-two-fold-walkforward-results.md` (Fold B + merchant-exclusion addendum)
- Approach A/C verdicts: `docs/narrative-mispricing-approach-a-results.md`,
  `docs/narrative-mispricing-approach-c-results.md`
- Scripts: `research/narrative_mispricing/{ml_two_fold_test.py, stress_test_foldb_auc.py,
  foldb_merchant_exclusion.py}`; panel `ml_panel_2022_2026.parquet`
- Reviewed alternative framing (continuous rank-IC, 25 names, single fold):
  `research/narrative_mispricing/spike_c_2026_holdout.py` — **run 2026-09-08** as a breadth
  cross-check (see the Dispatch addendum below): **FAILED the 2026 holdout** — the 2023-25
  "strong" cells decay to \|t\|≈1.4 on +2026 data, and the train≤2025-12 / test-2026 holdout IC
  is flat-to-**significantly negative** (RF 3-month t −3.8 / −2.1)
- RESEARCH-LOG: §3 ledger row "Narrative-mispricing Approach C…" + session blocks dated 2026-09-08
- Untried-tasks-log: item 9 (narrative) / item 10 (ML combination follow-ups)

---

## 2026-09-08 addendum — Fold B turned into a L/S ("Dispatch"): what's going on

Detail: `docs/handoff-2026-09-08-dispatch-ls.md`, RESEARCH-LOG §3 / §4 item 19 / §6.

- Built as a real dollar-neutral L/S (GBM rank, monthly, tercile), the Fold-B model is only **Sharpe 0.22 / +2%/yr** and loses to long-only XLI/PAVE/VOLT — not a standalone book.
- The 2026 losses are one-legged: the **long book**, and mostly **one name (TLN, −4.3%/mo, held long 9 of 12 months)** — the model chases momentum into a merchant-power reversal.
- Two features are structurally broken on this panel — `icr_z` for TLN **frozen at a 2016 value**, `gdelt_z` **frozen at ~2025-09** — so part of the "weak signals in combination" story runs on dead inputs.
- A **point-in-time trailing-net-margin screen on the long leg** (keeps loss-makers like TLN out — TLN is −$219M TTM NI by Q4-25) lifts the walk-forward L/S to **Sharpe 0.74 / +9.5%/yr, MaxDD −13%**; the profitability screen, not the narrative data, is the lever.
- Packaged as **"Dispatch"** with a build plan + pre-registered gates — then built & tested to a verdict: **FAILED the regime split** (see the 2026-09-09 update at the end of this section).

**Does combining Dispatch + mispricing buy breadth? No.** Dispatch *is* the Fold-B model as a
book (same panel / 23 names / 7 features / GBM). To settle it, `spike_c_2026_holdout.py` was
run — the thread's widest independent framing (25 names, 9 features, 2018-start, continuous
rank-IC, GBM & RF, proper train≤2025-12 / test-2026 holdout). Result: the 2023-25 "strong"
cells (best lean fwd-3m GBM t=2.25) decay to \|t\|≈1.4 on +2026 data, and **every 2026-holdout
cell is flat-to-negative** — RF 3-month IC −0.167 (t −3.77) and −0.186 (t −2.12), i.e.
*inverted* out-of-sample. `narrative_z` is bottom-2 of 9 in feature importance; `own_mom6m`
dominates (0.28–0.34). So the wider net catches the same fish: a momentum-loaded model that
worked 2023-25 and inverts in 2026. **Breadth must come from pruning the dead features
(`gdelt_z`, stale `icr_z`) to widen the universe + push the panel back to 2018, plus the
profitability screen — not from merging the two narrative framings.**

**Update 2026-09-09 — Dispatch closed, FAILED the regime split.** The pruning + widening was
done (dropped `gdelt_z`/`xlu_xng_spread`/`anc_composite`, rebuilt `icr_z` point-in-time from
EDGAR, widened `stress_beta_z` via the union of both rolling-beta panels → **breadth 13 → 19
names/mo**). Headline Sharpe rose to 0.77–1.08 **but ~100% of it is the 2024 merchant-vs-
regulated melt-up**: full GBM classifier ex-2024 (2023+2025+2026) Sharpe **−0.05**, regressor
+0.07; the merchant−regulated return spread was +68 / +127 / +35 / −27 % across 2023–26. A
regulated-only / sector-neutral cut removes 2024 and leaves only a trivial, non-fundable
signal (GBM clf ex-2024 Sharpe +0.19 raw, +0.36 with the margin screen, 2023 still negative).
**Dispatch is not a standalone cross-sectional strategy** — it's a momentum-timed sector bet
whose one regime has passed. What survives: the **point-in-time trailing-net-margin screen** as
a quality overlay, and the pruned/widened point-in-time panel as reusable infra. Fold B's
2026 tercile-classification AUC (item 18) is unaffected — that is not this returns expression.
See `docs/handoff-2026-09-08-dispatch-ls.md` §§v2–v3.
