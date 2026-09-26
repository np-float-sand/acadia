# Stress-beta decile follow-up — results

**Date:** 2026-09-15
**Pre-registration:** `docs/electrification-stress-beta-decile-preregistration.md` (written
before this script was run; decisions on cut, bucket size, min cross-section, test statistic,
window and threshold are locked there — none were changed after seeing this result).
**Script:** `electrification_stress_beta/decile_gate_test.py`
**Prior result being followed up:** Task 5's tercile-spread gate, FAILED, t=0.144 n=67
(`docs/electrification-stress-beta-gate-results.md`).

## Method (as pre-registered)

10th-minus-1st decile spread, operationalized as **top-3 / bottom-3 tickers by `stress_beta`,
by raw count**, on months with ≥15 valid names (avg realized monthly cross-section: 29.6 of
33 seed tickers, 1 of 108 months dropped below the floor). Same pooled monthly-spread t-test,
same 2021-01-01-onward primary window / pre-2021 context window, same |t| ≥ 2.0 gate as Task 5.

## Result

| window | t-stat | n (months) |
|---|--:|--:|
| 2021-01 → present (primary) | 0.318 | 67 |
| pre-2021 (context) | -0.564 | 41 |

**Gate 1: FAIL** (sign: positive, same sign as the tercile test's primary window)

## Verdict

**FAIL — no non-linear/threshold effect surfaces at the tail either.** The decile cut's primary
t-stat (0.318) is slightly larger in magnitude than the tercile test's (0.144) but is still an
order of magnitude short of the |t| ≥ 2.0 gate, and the pre-2021 context t-stat *shrank toward
zero* (-1.042 → -0.564) rather than growing, which is the opposite of what a genuine
threshold-effect hypothesis would predict — a real "only extreme stress matters" relationship
should sharpen, not flatten, when the buckets are narrowed to the true extremes. The most
plausible reading is that narrowing from ~9-10 names/bucket (tercile) to 3 names/bucket (decile)
mostly added idiosyncratic single-name noise without adding real explanatory tail signal, which
is consistent with the small-n concern flagged in the pre-registration and the source handoff.

This closes the follow-up per the handoff's own rule 3: report honestly even if it also fails.
No promotion to exploratory/watch-list — a positive-sign, sub-1.0 |t| result driven by 3-name
monthly buckets does not clear even the lower "exploratory" bar the handoff set (regime-split
rigor with a real magnitude, not just a matching sign).

## Bottom line for this line of research

Grid stress (national GSI, blended across ISOs) does not show a statistically significant
relationship with Screened Electrification supplier returns under either a tercile or a decile
cut. The earlier tercile FAIL was not an artifact of over-averaging a threshold effect; the
relationship stays flat (or weakens) as the cut gets more extreme. Consistent with the ~29-30
prior signal-search attempts logged in `docs/RESEARCH-LOG.md` §2.
