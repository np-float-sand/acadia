# Handoff — EIA Form 860M generator-pipeline signal (data source found, not built out)

**Date:** 2026-09-17
**Status:** SUPERSEDED (2026-09-20) — do not treat this as an open task. Every "recommended
next step" below (widen past AES-only, go beyond one quarterly snapshot, test an actual
tradeable rule) was carried out in a fuller session and reached a clean **FAILED** verdict:
`docs/interconnection-queue-signal-results.md` Thread 6 (EIA-alone: real but modest
correlation, r=-0.34/t=-1.74-1.76, n=26 quarters; 5 different tradeable-rule constructions
all underperform a static long) and Thread 8 (STEO added as a joint regressor: no rescue,
STEO t=0.06, R² unchanged, and the EIA correlation itself collapses on the STEO-covered
window). Kept below for its original data-source scoping notes only — the "recommended
next steps" section is stale.

## Where this comes from

Continuation of the CEP → grid-equipment-momentum → interconnection-queue thread
(`docs/datacenter-reit-cep-signal-results.md`, `docs/interconnection-queue-signal-results.md`).
The original hypothesis (data-center REITs and renewable-developer yieldcos are both hurt by
grid/interconnection tightness) survived a grid-equipment-momentum cross-validation on two
baskets, but the two "queue backlog" data sources actually tested (PJM raw queue, LBNL's
national "Queued Up" aggregate) both either failed outright or couldn't attribute projects to
specific developer tickers (CWEN/BEP/AES/AQN). This handoff picks up after finding a
structurally better source but not yet building on it.

## What's new: EIA Form 860M via `gridstatus.EIA`

`gridstatus.EIA().get_generators(date)` (package already installed, uses the existing
`EIA_API_KEY` in `.env`) returns a dict with `operating`, `planned`, `retired`,
`canceled_or_postponed` DataFrames. Confirmed by direct testing this session:

- **National** — one call covers every balancing authority, not region-by-region like PJM/CAISO.
- **Real entity/owner names** in `Entity Name` — clean matches for AES, Brookfield, Clearway,
  Algonquin subsidiaries (e.g., "AES Clean Energy," "Brookfield White Pine Hydro LLC,"
  "Clearway Energy Operating LLC," "Algonquin Power Co"). This is the fix for the attribution
  problem that killed every earlier attempt (PJM's `Name`/`Commercial Name` fields are generic
  project labels with near-zero true-positive match rate; LBNL's `developer` field exists but is
  too thin per company, 2-45 rows).
- **Genuine historical snapshots** — each monthly `Period` carries its own `Updated At` publish
  date (e.g., 2020-06 period published 2020-08-24), pullable back to at least 2020-01, and
  (as far as tested) not silently revised later. This is the fix for the point-in-time problem
  that PJM/LBNL both had (their MW figures reflect the current/latest amended project size, not
  what was known at each historical date).
- `planned` = rolling ~12-month-ahead pipeline (the "queue" analogue). `canceled_or_postponed`
  = the "withdrawn" analogue, but **no date field** — same caveat PJM/LBNL had for their
  cancellation timing, not fixed here.

## What was tested (one pass, AES only)

Pulled quarterly snapshots (`get_generators`, 26 calls, ~2020-01→2026-04) and filtered
`Entity Name` containing "AES" (chosen because it had by far the most matched rows of the 4
target tickers: 26 planned/1,932MW, 26 canceled/1,550MW vs. near-zero for the other 3 in the
current snapshot). Correlated planned-MW level, net pipeline (planned − canceled), and QoQ %
change against AES's own forward-3-month stock return:

| signal | r | t (n=19) |
|---|--:|--:|
| planned MW (level) | -0.384 | -1.71 |
| net pipeline (planned − canceled) | -0.382 | -1.70 |
| planned MW (QoQ % change) | -0.349 | -1.54 |

Clears |t|>1.5 (the threshold this thread adopted after
`docs/datacenter-reit-cep-signal-results.md`'s amendment), not |t|≥2.0. Sign: more near-term
planned capacity → weaker forward return — plausibly a capex/dilution-overhang story, not
necessarily the original "hurt by interconnection delay" mechanism. **n=19 (quarterly, single
name) is thin even by this session's own standard**, and no out-of-sample check has been run.
Given the day's repeated pattern — CEP and the LBNL queue index both looked marginal-to-good
in-sample and then failed cleanly on a real OOS test — this result should be treated as
unverified, not as a finding.

## Recommended next steps, in order

1. **Pool the full target basket, not just AES.** Brookfield/Clearway/Algonquin showed few-to-
   zero matches in the single snapshot checked — but that was only the *current* month's
   snapshot. Pull their `Entity Name` matches across the same historical quarters; they may show
   up more in earlier/later periods, or under subsidiary names not yet searched (e.g., Brookfield
   entities vary a lot — "Brookfield Renewable," "Brookfield Power [Region]," specific project
   LLCs). Widening the name-match list before concluding they're absent is the first thing to do.
2. **Go monthly, not quarterly**, and extend before 2020 if the API supports it (not checked) —
   n=19 quarters is the main reason nothing here is trustworthy yet; monthly cadence alone would
   ~3x the sample for the same calendar span.
3. **Pre-register before re-running** — window, entity-matching rule (which subsidiary-name
   patterns count), signal transform (level vs. change vs. net), target (own-name forward return
   vs. pooled basket), and the |t| gate — in writing, before looking at output. This thread has
   repeatedly found marginal-looking results dissolve under a genuine OOS check; don't skip the
   discipline that would catch that early.
4. **Build an actual OOS test analogous to Thread 3/5** in `docs/interconnection-queue-signal-
   results.md` — fit on an early window, forecast a later one, compare to realized returns —
   before treating any in-sample correlation as informative.
5. **Only if 1-4 hold up**, revisit whether this becomes a 4th candidate gate on the existing
   DLR/EQIX sleeve (`hedge_overlay.py`), following the same combination-test pattern already
   built in `research/datacenter_reit_cep/combination_test.py` (real-yield / CEP / grid-momentum
   already tested there).

## What NOT to do

Don't re-run the AES-only, quarterly, n=19 test again expecting a different answer — the
sample is the binding constraint, not the method. Don't skip straight to a portfolio-level
Sharpe/Calmar test (as was done for CEP and the LBNL queue index) before the standalone signal
itself has cleared a real gate — that pattern produced a lot of session-time spent on things
that were already going to fail once checked out of sample.

## Feasibility notes for whoever picks this up

- `gridstatus.EIA()` requires `EIA_API_KEY` (already in `.env`, same key used for the CEP
  fetch) — no new credentials needed.
- Each `get_generators(date)` call takes ~5-8 seconds; a monthly pull over 2020-2026 is ~75
  calls, roughly 8-10 minutes serial. Fine to run, just don't expect it instant.
- `canceled_or_postponed` has no cancellation date — if precise timing matters, this needs the
  same successive-snapshot-diffing approach as tracking `planned` membership changes month to
  month, not a single-file date-field reconstruction.

## References

- `docs/interconnection-queue-signal-results.md` §Thread 6 — the test this handoff continues.
- `docs/datacenter-reit-cep-signal-results.md` — the originating CEP thread, |t|>1.5 threshold
  amendment (adopted user-directed, not this project's default |t|≥2.0).
- `docs/RESEARCH-LOG.md` — ledger entry for the whole CEP/queue/EIA thread.
- `research/datacenter_reit_cep/` — all scripts and cached series from this session
  (`probe.py`, `combination_test.py`, `lbnl_national_queue_mw.parquet`,
  `pjm_caiso_queue_mw.parquet`, `eia_aes_pipeline.parquet`).
