# ML combined retest (RF/GBM/classifier/quantile), 2022-2025 only — results (2026-09-08)

**Status: SEVERE OVERFITTING SIGNATURE, all seven signals tested — no credible combined
signal found.** Probe: `research/narrative_mispricing/ml_model_test.py` (features
assembled by `ml_combined_test.py`; PJM ancillary pricing pulled fresh for 2022-2025 via
`pull_ancillary_2022_2025.py`).

---

## TL;DR

- Combined every weak/failed signal from this session — ICR (item 2), EDGAR narrative
  mentions + GDELT self-baseline attention (item 9), merchant stress-beta + the XLU/XNG
  correlation spread (item 3), 6-month momentum — into one panel, **restricted to
  2022-01 through 2025-09 per instruction** (electrification narrative wasn't live before
  ChatGPT). PJM ancillary-scarcity (item 7) was still being pulled for 2022-2025 when this
  first pass ran — see the "still open" note below.
- **Pre-registered gate: FAIL.** In-sample fit is dramatic (RF rank-IC up to +0.66, GBM up
  to +0.97, GBM classifier AUC = 1.000 — literal memorization on ~107 training rows with 7
  features). Out-of-sample (trained on 2022-2023, tested on 2024-2025) it collapses: RF
  rank-IC +0.08 to +0.13, GBM +0.05 to +0.19, classifier AUC 0.61-0.66. This is the
  textbook shape of an overfit model, not a validated combined effect.
- **The built-in sanity check confirms it.** Refitting on a *shuffled* target produces
  feature "importances" with the same general shape as the real target (`xlu_xng_spread`
  and `mom_z` dominate in both the real and shuffled runs) — meaning a meaningful chunk of
  the apparent importance is a structural artifact of which features have more variance /
  get chosen more often for splits, not evidence those features predict anything.
- **This is a second, direct confirmation of a failure mode this project already hit
  once** — the Henry Hub "kitchen-sink GBM on 25 pooled signals" got an OOS-looking result
  that evaporated on the full window (`docs/hhub-forward-gas-signal-results.md`). The
  Approach C handoff flagged this exact risk before this test ran; the risk materialized.

## Why this isn't a surprise, and doesn't disprove the user's hypothesis

The hypothesis (weak signals might combine into something real) is legitimate and the
right kind of thing to check — but this specific test was never going to have the
statistical power to confirm it. Restricting to 2022-2025 (per instruction, for good
reason — electrification wasn't a live narrative before then) leaves ~45 months, ~24 for
training. Seven correlated, mostly-flat-standalone features fit on ~107 training rows will
essentially always find *some* apparent pattern; the shuffled-target check is exactly the
tool needed to tell a real pattern from an artifact, and it says artifact here.

## Re-run with all seven signals (PJM ancillary 2022-2025 pull completed)

The 2022-2025 PJM ancillary pull (45 months, 1.96M rows, monthly-chunked with retries to
work around the post-market-redesign rate limits) finished and was folded in. Full result:

| target | model | in-sample IC/AUC | OOS IC/AUC |
|---|---|---|---|
| fwd1 | RF regressor | +0.579 | +0.035 |
| fwd1 | GBM regressor | +0.908 | **-0.070** |
| fwd1 | RF classifier | 0.792 | 0.622 |
| fwd1 | GBM classifier | 1.000 | 0.527 (~chance) |
| fwd3 | RF regressor | +0.689 | +0.178 |
| fwd3 | GBM regressor | +0.939 | +0.216 |
| fwd3 | RF classifier | 0.899 | 0.620 |
| fwd3 | GBM classifier | 1.000 | 0.580 |

**Ancillary pricing did not rescue the hypothesis — if anything it made the 1-month case
worse.** It earns real feature importance once included (0.15-0.27 across models, no
longer a placeholder), but the 1-month GBM regressor's OOS rank-IC flips negative (-0.070,
vs +0.045 with ancillary excluded) and the 1-month GBM classifier's OOS AUC drops to
essentially chance (0.527, vs 0.653 excluded) — consistent with an added feature giving the
model more surface to overfit on a still-tiny training set, not new information. The 3-month
numbers move a little (RF 0.125→0.178, GBM 0.191→0.216) but stay inside the same
not-to-be-trusted range given the in-sample/OOS gap and the shuffled-target check.

**Verdict unchanged, now confirmed on all seven signals: no credible combined effect.**

## Disposition

**Do not trust the OOS numbers reported above as found signal — this is the pre-registered
expected shape of a false positive, not a surprising negative result.** If this angle is
revisited: use far more regularization (shallower trees, fewer features, explicit
L1/L2 shrinkage), and treat any OOS number as provisional until it survives a bootstrap or
multiple-split stability check — a single ~20-month test window is not enough to trust a
marginal AUC of 0.62-0.66 either way. The deeper constraint is structural, not a modeling
choice: restricting to 2022-2025 (correctly, for the stated reason) caps the achievable
sample size at a point where 7-feature tree ensembles will reliably overfit regardless of
whether any of the underlying signals carry real information.
