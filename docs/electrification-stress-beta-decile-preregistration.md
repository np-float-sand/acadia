# Pre-registration — stress-beta decile follow-up

**Date:** 2026-09-15
**Status:** PRE-REGISTERED, ABOUT TO RUN — written before touching any spread/t-stat output.
**Origin:** `docs/handoff-2026-09-15-stress-beta-decile-nonlinear.md`, picked up per user
instruction to pre-register and run (not just scope).

## Decision 1 — which cut

**10th decile minus 1st decile** (the most extreme tails), not 9th-vs-2nd.

Rationale: the question motivating this follow-up is whether the GSI→return relationship is a
*threshold effect* — "only genuinely extreme stress registers" — as opposed to the smooth
linear-in-rank relationship the tercile spread already tested and failed (t=0.144). The 10-vs-1
cut is the most literal test of that specific hypothesis. 9-vs-2 is a weaker version of the same
test (drops the single most extreme name each side) with no independent rationale for
preference here, so per the handoff's own rule ("pick one... not both, not whichever looks
better") it is not run.

## Decision 2 — bucket construction

**Fixed top-3 / bottom-3 by raw count**, not `n // 10`.

Rationale: this universe has ~27-31 tickers/month. `n // 10` would give 2-3 depending on monthly
coverage and floor to a shifting bucket size; stating the cut in ticker-count terms (as the
handoff itself flags) makes the thinness visible rather than obscured by "decile" language. Fixed
top-3/bottom-3 is stable across months.

## Decision 3 — minimum monthly cross-section

**Require ≥15 valid (stress_beta, fwd_ret) names in a month to include it.** Below that, a
3-name bucket exceeds 20% of the cross-section and stops reading as a tail. This is looser than
a literal 10% floor (which would need n≥30 and drop most months) but mirrors the tradeoff Task
5's tercile test already made (min n=6 for 2-name terciles, i.e. also ~33% buckets at the floor).

## Decision 4 — test statistic

**Same pooled monthly-spread t-test as Task 5** (mean of monthly top-3-minus-bottom-3 spread
returns, divided by its own standard error, over the time series of months) — not a
cross-sectional Mann-Whitney within a single month. Kept identical to Task 5 for direct
comparability; the added noise from thinner monthly buckets flows through the variance term of
the same test rather than requiring a different test. This is a deliberate choice, not an
oversight — flagged here so it can't be swapped post-hoc if the result disappoints.

## Decision 5 — window, threshold, gate

Unchanged from Task 5: primary window 2021-01-01-onward, pre-2021 reported for context only
(no sign-match requirement on it), gate threshold |t| ≥ 2.0. Same universe-loading path as
`gate_test.py` (`universe.load_seed()`, currently 33 seed tickers; THR is expected to drop out
again as all-NaN per Task 5's coverage note).

## Decision 6 — reporting commitment

Reported regardless of outcome. Per the handoff: even a nominal PASS on Gate 1 is treated as
exploratory/watch-list only, not adopted, unless it also survives the same regime-split rigor
Task 5 applied (sign doesn't need to match pre-2021, but a PASS driven by a handful of monthly
observations with visibly unstable sign will be called out as such, not presented as clean).

## Implementation

New script `electrification_stress_beta/decile_gate_test.py`, reusing `national_gsi.py` and the
`rolling_stress_betas` pipeline unchanged from `gate_test.py`; only the spread function changes
(`_tercile_spread` → `_decile_spread`, top/bottom-3-by-count, min-15 filter per Decisions 2-3).
`gate_test.py` itself is left untouched (Task 5 stays closed/reproducible as-is).
