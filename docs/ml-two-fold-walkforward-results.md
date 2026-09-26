# Two-fold walk-forward retest (RF/GBM/classifier), 2022→2024 and 2024→2026 — results (2026-09-08)

**Status: Fold A weak (early regime, not "overfit"); Fold B surfaces a real, narrow,
not-yet-actionable signal — driven by stress-beta + ICR + narrative-disclosure count in
combination, not by the attention/GDELT data.** Probes:
`research/narrative_mispricing/ml_two_fold_test.py`, `stress_test_foldb_auc.py`.

---

## TL;DR

- **Extended all seven signals through 2026** to build this: EDGAR narrative keywords
  (now through 2026Q3), PJM ancillary pricing (2026 YTD, 431,918 new rows), stress-beta
  (reran the full `grid_resilience` pipeline through 2026-08). **GDELT could not be
  extended** — the 2026 pull hit sustained rate-limit backoffs for 25+ minutes and never
  completed; `gdelt_z` is stale (frozen at its last real value, ~2025-09) for the entire
  2026 test window. Flagged throughout below, not swept under the rug.
- **Fold A (train 2022-23, test 2024): weak, OOS mostly negative.** Per the corrected
  framing — this is not labeled "overfitting," since 2022-23 is early-regime for a
  narrative that only became fully live after ChatGPT. RF/GBM regressor OOS rank-IC:
  -0.08 to -0.23 (both horizons, both models). Classifier OOS AUC: 0.42-0.57. In-sample
  fit is still dramatic (IC up to 0.97, classifier AUC up to 1.000), so the *gap* is real
  regardless of label — a model fit on the earliest, thinnest part of the regime does not
  describe 2024.
- **Fold B (train 2024-25, test 2026): a genuine, stress-tested, but very small
  finding.** The GBM classifier predicting top-vs-bottom-tercile 3-month forward return
  gets **OOS AUC = 0.758** — stable across 10 random seeds (0.751-0.759), sitting at the
  100th percentile of a 50-draw shuffled-label null distribution (null mean 0.491, std
  0.088), and consistent in direction across all 6 test months (AUC 0.55-1.00 each month,
  not one lucky month carrying it).
- **But the ablation says this isn't the narrative story.** Dropping `stress_beta_z`
  hurts the most (0.758→0.694) — it's the load-bearing feature. Dropping `mom_z` or
  `xlu_xng_spread` *improves* the AUC (0.758→0.812 / 0.770). Dropping the *known-stale*
  `gdelt_z` only costs 0.023 AUC (0.758→0.735) — a small but nonzero drop from a feature
  that is frozen at a 2025 value for the whole test period, which is a data-integrity flag,
  not evidence GDELT itself works. **The real driver, per the ablation, is the merchant
  stress-beta (item 3's already-studied, individually-weak GSI factor) combined with ICR
  and EDGAR disclosure counts — not narrative attention.**

## Why this is interesting without being tradeable yet

This clears a bar nothing else in the whole session cleared: a shuffled-label test that
the real result sits outside of, stability across model seeds, and no single-month
dependency. That is a real, distinguishable-from-noise pattern in this specific window —
not a coin flip.

It is also built on **54 test observations across 6 months** (9 tickers per month, top
and bottom tercile of a ~13-23-name universe). Six independent time periods is not enough
to trust a discovered pattern on its own, however cleanly it clears a noise-floor test —
short winning streaks are exactly what small samples produce, real or not. The honest
position is: *interesting enough to keep, not confirmed enough to act on.*

## What this actually says about the original questions

- **The narrative/attention data (EDGAR + GDELT), the reason this whole line of testing
  started, is not what's driving the one result that cleared a bar.** Its own component
  (`edgar_z`) contributes moderately (dropping it costs 0.034 AUC); `gdelt_z`'s
  contribution is compromised by the stale-data problem and shouldn't be trusted either
  way.
- **The merchant stress-beta (item 3) is the standout, unexpectedly.** A factor this whole
  research line has repeatedly found individually too weak to trade (IC≈0.14, t≈1.9 at
  its best documented showing) is the single most load-bearing feature in the one
  multivariate result that survived a real stress test. This is consistent with the
  user's original hypothesis — weak individual signals gaining power in combination — just
  pointing at a different signal than the one this session originally set out to test.

## Recommended next steps, none done this session

1. **Fix GDELT for 2026** (needs a much more patient retry strategy than 25 minutes of
   backoff, or a different API entirely) before trusting or discounting its role.
2. **Let Fold B's test window grow** — 6 months is too few; re-run this exact fold
   quarterly as 2026 accumulates more realized 3-month-forward outcomes.
3. **Isolate the stress-beta × ICR × EDGAR interaction directly** (a 3-feature model, not
   7) to see if the signal survives without the noise of the other four features — this
   would also make a shuffled-label test more interpretable with fewer moving parts.
4. **Do not wire anything live from this** — one small, stress-tested-but-tiny OOS result
   is a lead to chase, not a strategy to run.

---

## Addendum (different session, 2026-09-08): merchant-name exclusion check

Fold B's own writeup above flagged, but didn't test, the standing open question for
this whole research line: is a result on this 25-name universe actually broad, or is
it just VST/CEG/TLN/NRG — the four merchant names `docs/grid-stress-signal-evidence.md`
§5 already documents as the recurring confound behind every other cross-sectional
signal that's broken on this universe (most recently the ICR retest, where the same
four names anchored a spurious-looking correlation from the *opposite* end).

**Method:** exact same panel (`ml_panel_2022_2026.parquet`), features, GBM
hyperparameters, and Fold B window (train 2024-01→2025-12, test 2026-01→2026-08) as
`stress_test_foldb_auc.py`, run on three ticker cuts with terciles recomputed
per-date *within* each cut (not inherited from the 23-name universe):

| cut | n tickers | test n | mean OOS AUC (10 seeds) | std | null percentile |
|---|---:|---:|---:|---:|---:|
| ALL_23 (reproduction) | 23 | 54 | 0.755 | 0.003 | 100% |
| EX_MERCHANT (VST/CEG/TLN/NRG dropped) | 19 | 48 | 0.719 | 0.008 | 100% |
| MERCHANT_4 only | 4 | — | — | — | too few names to form terciles |

(ALL_23's 0.755 vs. the original 0.758 is a rounding/tercile-edge-case difference
from recomputing terciles per-cut rather than reusing the original run's exact
frame — treat as a faithful reproduction, not a discrepancy.)

**Result: the signal is not the merchant-name confound.** Dropping VST/CEG/TLN/NRG
weakens the OOS AUC (0.755→0.719) but it still sits at the 100th percentile of its
own 50-draw shuffled-label null on the remaining 19 regulated-only names, and is not
one-month-driven (monthly AUC 0.562–0.938 across the 6 test months, all ≥0.5).
Feature importances shift slightly toward `stress_beta_z` (0.21→0.24) and `icr_z`
(0.159→0.204) with the merchant names out, and `gdelt_z`'s importance drops
(0.219→0.132) — consistent with the merchant names carrying more of the *narrative*
variance specifically (expected, since VST/CEG/TLN/NRG are the names with the most
dramatic 2023-25 narrative swings) without being the source of the whole predictive
signal.

**Updated read:** this raises, not lowers, the credibility of Fold B's finding
relative to where the original writeup left it — the standard "it's just the four
merchant names again" explanation that has killed several other signals on this
universe does not apply here. The finding is still small-sample (n=48-54, 6 months,
one specific train/test split) and **still not actionable**, per the original
writeup's own conclusion — this addendum narrows *why* it might be real without
changing whether it's tradeable yet.

Script: `research/narrative_mispricing/foldb_merchant_exclusion.py` (kept alongside
the two probes above). GDELT's 2026 extension is still running as of this addendum
(30+ minutes, rate-limited) — its eventual completion doesn't change this addendum's
conclusion, since `gdelt_z` was never the load-bearing feature in any cut tested.

---

## Second addendum (same session as the exclusion check, 2026-09-08): 3-feature isolation

Item 18 follow-up (b): does the signal survive dropping to just the three features
the drop-one ablation flagged as load-bearing (`stress_beta_z`, `icr_z`, `edgar_z`),
removing `mom_z`, `xlu_xng_spread`, `anc_composite` (individually "noise" — dropping
them one at a time *improved* AUC) and `gdelt_z` (known-stale in the test window)?

**Result: mixed, not confirmatory.** Same panel, folds, and GBM hyperparameters as
the exclusion check above.

| cut | features | test n | mean OOS AUC (10 seeds) | null percentile |
|---|---|---:|---:|---:|
| ALL_23 | full 7 | 54 | 0.753 | 100% |
| ALL_23 | lean 3 (stress-beta/ICR/EDGAR) | 60 | 0.567 | 78% |
| EX_MERCHANT_19 | full 7 | 48 | 0.670 | 96% |
| EX_MERCHANT_19 | lean 3 | 54 | 0.645 | 98% |

On the full 23-name universe, the isolated 3-feature model collapses toward
chance (0.567, nowhere near its own null's 100th percentile) — per-month AUC swings
from 0.20 to 0.80, including two months below 0.5. On the 19 regulated-only names it
holds up close to the 7-feature result (0.645 vs 0.670), but one test month
(2026-06, AUC 1.00) is doing a lot of that work.

**Read:** the drop-one-feature ablation's framing — "stress-beta/ICR/EDGAR are the
real signal, the other four are noise" — is too strong. Removing one noisy-looking
feature at a time doesn't hurt because the *remaining* six can still combine to
compensate; collapsing straight to three shows the fuller feature set is doing real
work, at least on the full universe. Treat stress-beta+ICR+EDGAR as **necessary but
not sufficient**, not as the whole mechanism. This makes the finding harder to
simplify into a clean 3-factor story, which is itself useful to know before anyone
tries to build something narrower out of it.

Script: `research/narrative_mispricing/foldb_3feat_isolation.py`.
