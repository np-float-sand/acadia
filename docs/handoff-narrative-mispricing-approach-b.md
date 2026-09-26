# Handoff: narrative mispricing — Approach B (event study)

**Context:** `docs/untried-tasks-log.md` item 9. Approach A (EDGAR+GDELT composite,
cross-sectional rank-IC) is done: weak, right-signed, not significant (3mo t=-1.15).
See `docs/narrative-mispricing-approach-a-results.md`. Data already pulled and cached —
reuse it, don't re-fetch: `research/narrative_mispricing/edgar_quarterly_mentions.parquet`
(25 tickers, quarterly), `gdelt_daily_volume.parquet` (22 tickers, daily, 2018-2025),
`composite_monthly.parquet`.

**Approach B — the idea:** instead of a continuous intensity score, treat each *new*
EDGAR narrative-mention event (first quarter a ticker's count steps up from ~0, or a
volume spike vs its own trailing baseline) as a discrete trigger. Use GDELT daily volume
as the decay/intensity curve in the days/weeks after. Test for a pop-then-reverse pattern
(event-study CAR, same methodology as `grid_resilience/signals/conditional_beta.py::event_study()`)
— a reversal would support shorting into the pop.

**Biggest risk, address first:** open item #15 (regulatory-calendar DER short) FAILED on
n≈1 real events and telegraphed news already being priced by the time it fired. Approach B
is shaped the same way. Before building the CAR machinery: define "new event" precisely
(first-mention-after-N-quarters-silence? a z-score spike threshold?) and count how many
qualify across the 25-name universe, 2018-2025. If n is small (single digits), treat that
as a pre-registered kill signal, same bar as #15, rather than building the full event study
on too few events to mean anything.

**Suggested order:**
1. Define & count candidate events from the already-cached EDGAR/GDELT data (cheap, no
   new fetches). Gate on n before doing anything else.
2. If n clears the bar, run the event-study CAR (reuse `event_study()`'s pattern: estimation
   window before, event window around, abnormal return vs. expected).
3. Pre-register the PASS bar (CAR sign, magnitude, sub-period stability) before looking at
   results — every signal in this research line that skipped this step got a messier verdict.

**Not done:** the event-count step (step 1) — this is where the next session should start.
