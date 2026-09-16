# Handoff — stress-beta decile/non-linear follow-up (not started)

**Date:** 2026-09-15
**Status:** SCOPED, NOT RUN — deliberately deferred to a future session, not part of the
Screened Electrification SOW execution plan
(`docs/superpowers/plans/2026-09-15-screened-electrification-sow-execution.md`).

## Where this comes from

Task 5 of that plan ran a pre-registered, run-once stress-beta gate on the Screened
Electrification 27-name supplier universe: national GSI (blended across ISOs, reusing
`grid_resilience`'s GSI/conditional-beta machinery unmodified), monthly **tercile** spread
(top third minus bottom third of `stress_beta`), pooled t-stat on the 2021-01-01-onward window.
**Result: FAIL** — t=0.144, n=67 months, nowhere near the pre-registered \|t\|≥2.0 bar. Full
methodology and numbers: `docs/electrification-stress-beta-gate-results.md`.

The user asked, correctly, whether this was too quick to write off given the test's linearity,
and separately pointed at `ssrn-5736562.pdf` (Mamkhezri, Sun & Yang, 2026, "The Hidden Cost of
the Cloud") — a causal estimate that data-center entry raises PJM's transmission-induced price
component ~70% ($2.49/MWh) — as evidence that "these prices matter" economically, i.e. the
underlying grid-stress phenomenon is real and large even though the linear tercile test on
supplier *stock returns* didn't detect a significant relationship.

## The proposed refinement

Instead of a tercile spread (33rd/33rd percentile split, which averages over the whole
cross-section and can dilute a threshold or convex effect), use a more extreme cut that isolates
the tails, where a non-linear/threshold effect would show up most clearly if it exists at all:

- **9th decile minus 2nd decile**, or
- **10th decile minus 1st decile** (most extreme — top/bottom ~10% only)

Rationale: `stress_beta` itself is estimated via OLS (linear) in `conditional_beta.py`, and the
gate's own spread-and-t-stat step is already a monotonic rank test rather than a raw linear
regression on returns — but if the true GSI→return relationship is a threshold effect (only
genuinely extreme stress registers) rather than smoothly linear-in-GSI, a tercile spread that
pools the "somewhat stressed" middle names together with the true extremes could wash out a real
tail effect that a tighter decile cut would preserve.

## Why this is a new probe, not a reopening of Task 5's gate

Task 5's gate is closed, reviewed, and logged (`docs/RESEARCH-LOG.md` §3, "Timing signals" —
2026-09-15 row). Retrying with a different cut *now*, having already seen the tercile result
fail, is the exact p-hacking pattern this project's discipline exists to prevent (see
`docs/RESEARCH-LOG.md` §2's "~28-29 signal attempts" history — several of those failures were
exactly this shape: a specification that looked better only after the first one failed). If this
follow-up is picked up, it must be:

1. **Pre-registered before looking at results** — pick one specific cut (9-vs-2 or 10-vs-1, not
   both, and not "whichever looks better") and a gate threshold, in writing, before running it.
2. **Reported honestly regardless of outcome** — including if it also fails, or if it "looks
   promising" but doesn't survive a regime split (the pattern that killed the narrative-mispricing
   Fold-B curiosity and the Henry Hub GBM result elsewhere in this project's history).
3. **Treated as exploratory/watch-list even on a PASS** unless it survives the same regime-split
   rigor Task 5 applied (pre-2021 context window, no sign-flip requirement waived).

## The real statistical-power concern to flag up front

The universe is **27-31 tickers**. A 10th-vs-1st-decile cut on ~30 names puts roughly **3
tickers per bucket** each month — an extremely thin cross-section for a monthly spread. A 9th-vs-
2nd cut is similarly thin. Before running this, whoever picks it up should think about whether:
- A standard t-test is even appropriate at n≈3 per bucket (very sensitive to single-name noise —
  one idiosyncratic mover swings the whole bucket's return), or whether a rank-based test
  (Mann-Whitney on the monthly cross-section, or pooling several months per observation) is more
  defensible.
- Whether decile cuts are even meaningful with this few names, versus e.g. top-3-vs-bottom-3 by
  raw count (equivalent to ~10th/1st decile here) stated explicitly in ticker-count terms rather
  than percentile terms, so the thinness is visible rather than obscured by "decile" language.

## Reusable infrastructure

- `electrification_stress_beta/national_gsi.py` — the national GSI blend (Task 4, tested,
  unchanged) is directly reusable, no changes needed.
- `electrification_stress_beta/gate_test.py` (Task 5) — the overall structure (fetch prices,
  build national GSI, `rolling_stress_betas`, forward-return join) is reusable; only
  `_tercile_spread`'s `nlargest`/`nsmallest` fractions (currently `n // 3`) would change to a
  decile-based split, and the small-n concern above should be addressed in the same pass rather
  than bolted on after.

## References

- `docs/electrification-stress-beta-gate-results.md` — Task 5's closed gate, methodology and
  numbers.
- `ssrn-5736562.pdf` (repo root) — Mamkhezri, Sun & Yang (2026), "The Hidden Cost of the Cloud:
  Data Centers and Transmission-Induced Electricity Prices." Causal DiD estimate: data-center
  entry raises PJM's transmission-induced price component ~70% ($2.49/MWh); effect larger with
  distant generation, larger with co-located renewables vs. fossil, concentrated in peak
  hours/spring. No REIT or equity-return evidence in the paper itself — it's grounding for the
  economic reality of grid stress, not direct evidence for this specific equity signal.
- `docs/RESEARCH-LOG.md` §2 — the project's accumulated base rate (~28-29 prior signal-search
  attempts, no tradable edge survived an honest regime split) — the prior this follow-up should
  be weighed against.
