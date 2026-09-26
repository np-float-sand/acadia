# RT/DA LMP spread — feasibility check results (2026-09-08)

**Status: ERCOT feasible now (and easier than documented); PJM confirmed blocked
pre-2024.** Untried-tasks-log item 4. This is the check flagged as "the next step" in
three separate handoffs since 2026-08-15 and never actually run until now.

---

## TL;DR

- **ERCOT: fully feasible, and cheaper than every prior doc assumed.** The production
  pipeline's existing `ercot.get_lmp()` call — already cached, already feeding the live
  GSI's `lmp_z` component — is gridstatus's **real-time** (SCED, 5-minute) price, not
  day-ahead as every prior handoff assumed. The missing leg is day-ahead
  (`ercot.get_dam_spp(year)`), which pulls a **full year in ~8 seconds**, 2011→present,
  on identical location names (`HB_NORTH`, `LZ_NORTH`, …) to what's already cached. No new
  RT fetcher needs to be built — `CLAUDE.md`'s "needs a new `fetch_lmp_rt()` variant" had
  it backwards for ERCOT.
- **PJM: confirmed blocked, but precisely — not "always," only pre-~2024.** Fetching
  `market="REAL_TIME_HOURLY"` for any date before PJM's ~2-year archive cutoff forces a
  full nodal-network download (all ~10,000+ pnodes) before client-side location filtering,
  regardless of how narrow the date range requested. A single day already exceeds 300,000
  rows and 429s out within 6 paginated requests using gridstatus's default retry budget —
  reproduced live, twice (1-day and 1-week ranges, both failed the same way). The **same
  call for a recent week (2025-06) succeeds in under 5 seconds** — PJM's API supports
  server-side pnode filtering only inside the ~2-year archive window.
- **Net effect on the original thesis:** the merchant names this signal is meant to help
  (VST, NRG — the two names causing the whole stress-beta whipsaw, §5 of
  `grid-stress-signal-evidence.md`) are **both ERCOT**, the ISO where this is now fully
  practical. PJM's blocker mostly affects CEG/TLN, which have much shorter listed
  histories anyway (13 and 9 quarters of usable data per the ICR retest, item 2). The
  historically-important part of this idea doesn't need PJM's pre-2024 history at all.

---

## 1. What was checked

Per the recommendation logged in three separate places since 2026-08-15
(`handoff_2026-08-15-rt-da-spread-signal.md`, `handoff_2026-08-15-outage-reserve-margin-signal.md`,
`handoff_2026-09-01-grid-buildout-long-short.md`) — live-verify gridstatus actually returns
clean historical RT and DA LMP for ERCOT and PJM before any signal design work. Never
executed until this session.

**ERCOT:**
```
ercot.get_lmp(date, location_type="settlement point")   # RT, SCED 5-min — ALREADY the production fetch
ercot.get_dam_spp(year)                                 # DA — the leg that's actually missing
```
Pulled `get_dam_spp(2018)` live: 122,640 rows, 7.9 seconds, columns
`[Time, Interval Start, Interval End, Location, Location Type, Market, SPP]`, locations
`{HB_BUSAVG, HB_HOUSTON, HB_HUBAVG, HB_NORTH, HB_SOUTH, HB_WEST, LZ_AEN, LZ_CPS,
LZ_HOUSTON, LZ_LCRA, LZ_NORTH, LZ_RAYBN, LZ_SOUTH, LZ_WEST}` — an exact match to the
already-cached RT settlement-point universe.

**PJM:**
```
pjm.get_lmp(date, end, market="REAL_TIME_HOURLY", locations="hubs")
```
- `date="2019-01-01", end="2019-01-08"` (1 week, pre-archive-cutoff): failed. Paginated
  through 300,000+ rows in 50,000-row chunks, hit a 429 on page 6, exhausted gridstatus's
  default retry budget, raised `ValueError: No objects to concatenate`.
- `date="2019-01-01", end="2019-01-02"` (1 day, same range): **also failed the same way**
  — the full-network download size doesn't shrink with a narrower date range, so even one
  day already exceeds gridstatus's default retry tolerance for pre-archive dates.
- `date="2025-06-01", end="2025-06-08"` (1 week, recent): **succeeded in 4.7s**, 2,016 rows,
  server-side pnode filtering confirmed via the request log (`'pnode_id': '51217;...'` in
  the params, vs. no such filter in the pre-archive requests).

## 2. Why ERCOT is easier than assumed

Every prior document (the June-2026 forward-power handoff, the August RT/DA handoff,
`CLAUDE.md`'s "Future Work" note) treated "real-time LMP" as the missing fetcher needing a
new `ISO_RT_LOCATION_TYPE` config entry and `fetch_lmp_rt()` function. That's backwards for
ERCOT: `gridstatus.Ercot.get_lmp()`'s own docstring says it returns data "normally produced
by SCED every five minutes" — i.e. **real-time**, not day-ahead. `config.py`'s
`ISO_LOCATION_TYPE["ERCOT"] = "settlement point"` and the production cache
(`ercot_lmp_settlement point_*.parquet`) are the RT leg already. ERCOT's day-ahead product
is a structurally different gridstatus method (`get_dam_spp`, whole-year batch downloads),
not a `location_type` variant of `get_lmp`.

## 3. Disposition and recommendation

**Feasibility gate: PASS for ERCOT, FAIL (pre-2024) for PJM.** This closes the specific
"is this buildable" question three handoffs left open.

**Recommended next step, if picked up:** build the ERCOT RT/DA spread first and only —
`spread = RT_LMP - DA_SPP` at the same hub, joined on the already-matching location names.
This directly targets VST and NRG, the two names actually driving the documented whipsaw,
without needing PJM's blocked pre-2024 history at all. Treat PJM as out of scope unless a
bulk DataMiner-style historical export (like the one already used for `pjm_ftr_bids_mnt`,
which bypassed the same pagination problem for FTR data) is found for `rt_hrl_lmps` —
not checked this session; worth 15 minutes before writing PJM off completely.

**Not done this session:** the actual signal design/build (what the spread predicts, how
it feeds the factor, a pre-registered gate) — this was scoped as a feasibility spike only,
per the original recommendation in every handoff that raised it. `CLAUDE.md`'s RT/DA note
should be corrected to reflect the ERCOT-is-actually-easy / PJM-is-precisely-blocked
finding.
