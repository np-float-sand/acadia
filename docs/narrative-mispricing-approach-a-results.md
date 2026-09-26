# Narrative-extremity short (Approach A) — spike results (2026-09-08)

**Status: NOT SIGNIFICANT at any horizon, in the original pass or after refining both
data sources.** Untried-tasks-log item 9, approach A. Probes:
`research/narrative_mispricing/spike_a_composite.py`, `spike_a_gdelt_retry.py`/`retry2.py`,
`spike_a_ic_test.py`, `spike_a_refined_edgar.py`, `spike_a_refined_test.py`.

---

## TL;DR

- **Composite:** per ticker per month, average of (a) z-scored trailing-4-quarter EDGAR
  "data center" mention count in that company's own 8-K/10-Q/10-K filings, and (b)
  z-scored trailing-63-day GDELT news-volume intensity for that company's name. Tested as
  a cross-sectional rank across the 25-name `grid_resilience` universe (22 with usable
  GDELT coverage after two rounds of rate-limit retries — CEG, PCG, D, SO missing GDELT;
  everyone has EDGAR).
- **Cross-sectional rank-IC vs forward return: negative (right sign for a short thesis)
  but not significant at any horizon.** 1mo t=-0.39, 3mo t=-1.15, 6mo t=-0.55, n≈85-87
  months. The best case (3mo) is the same shape as most of this research line's near-misses
  — a plausible-looking number that doesn't clear |t|≥2.
- **VST/NRG-specific check surfaced a real measurement problem, not just noise.** VST's
  composite score is *consistently negative* (under-narrated relative to peers) all the way
  through 2023-2025, the exact period everyone agrees its narrative was dominant — because
  VST is a generation company, not a data-center operator, so it rarely says "data center"
  in its own filings (total EDGAR mentions: 1, vs. Talen 19, Southern 25, Exelon 16), and
  GDELT's raw volume for a mid-cap merchant generator is swamped in cross-section by
  larger, more universally-covered utilities' baseline news flow regardless of any
  narrative spike. NRG's composite does show a real escalation (flips from ~-0.4 to +0.2
  right around when NRG's 2024-25 re-rating accelerated) — so the mechanism isn't
  completely blind, just noisy and keyword-narrow.

## Disposition

**Directionally interesting, not yet validatable, exactly as the source that raised this
idea warned ("harder to validate").**

---

## Refinement pass (2026-09-08, same session) — measurement fixed, significance didn't move

Applied both fixes flagged above:
1. **Broader EDGAR keywords.** Added "hyperscaler" (near-useless — 0-3 mentions almost
   everywhere, dropped) and "large load" (much more informative — PJM/ISO planning
   terminology that merchant generators and utilities actually use in filings; VST alone
   goes from 1 mention ("data center") to 13 combined). Combined EDGAR score = "data
   center" + "large load" counts.
2. **GDELT self-baseline normalization.** Replaced the raw cross-sectional z-score with
   each ticker's own trailing-756-day rolling z-score, so a smaller company's real spike
   isn't diluted by comparison to a mega-cap's routine coverage baseline.

**Effect on measurement quality: real and clean.** VST's composite score now genuinely
varies with known events instead of sitting permanently negative — 1.83 in Sep 2023 (the
initial narrative build), 1.31 in Feb 2025 (continued momentum), dipping to -0.59 in April
2024. This is a believable, interpretable narrative-intensity series for the first time.

**Effect on the rank-IC: still not significant, and the sign structure got more
interesting, not simpler.**

| horizon | original IC (t) | refined IC (t) |
|---|---|---|
| 1mo | -0.008 (-0.39) | **+0.033 (+1.40)** |
| 3mo | -0.023 (-1.15) | -0.026 (-1.09) |
| 6mo | -0.012 (-0.55) | -0.014 (-0.57) |

The 1-month sign flipped positive (narrative intensity as short-term momentum — "buy the
rumor" — not overhype). The 3- and 6-month signs stayed negative (consistent with an
eventual fade) but didn't strengthen. **Nothing clears |t|≥2 at any horizon, before or
after the refinement.** The honest read: better data did not rescue the hypothesis, but it
did rule out "the null result was just bad measurement" as an explanation — the underlying
relationship (if any) is genuinely weak, not merely mismeasured.

## Final disposition

**Not adopted, not fully killed either.** Two clean sub-findings worth carrying forward:
(a) narrative intensity may behave more like a short-horizon momentum signal than an
overhype-short signal — worth a passing thought if Approach B or C surface something
similar; (b) the EDGAR "large load" + GDELT self-baseline combination is now a validated,
reusable measurement approach for any future narrative work on this universe, independent
of whether Approach A itself ships. Per the user's sequencing, proceeding to **Approach B**
next — handoff written to `docs/handoff-narrative-mispricing-approach-b.md` for
continuation in a fresh session.
