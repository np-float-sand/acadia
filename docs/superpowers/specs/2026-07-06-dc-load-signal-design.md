# Design Spec — Data Center Load Signal (Regulated-Path Phase 2)

**Date:** 2026-07-06
**Branch:** res2
**Status:** Approved for implementation

---

## Problem

The hard-switch business-model architecture ([2026-06-29 spec](2026-06-29-business-model-aware-factor-design.md)) routes rate-regulated PJM T&D utilities to an ICR (interest coverage ratio) signal instead of stress beta. This fixed the Sharpe problem (-0.009 → 0.331) but ICR measures the wrong thing for the current macro theme: it captures balance-sheet fragility under rate stress, not exposure to data-center load growth.

**Confirmed distinction (2026-07-06 session):** D (Dominion) and PEG (PSEG) sit in the two largest data-center corridors in PJM (Northern Virginia, NJ). Their ICR reflects debt from nuclear builds and general leverage — it says nothing about whether their *territory* is capturing the data-center demand wave. A utility with weak ICR but massive DC load growth would incorrectly score as a short under the current signal.

---

## Goal

Replace ICR in the regulated-path signal slot with a two-component data-center load signal, built entirely from data already available via `gridstatus` (no new vendor/API required):

- **Level (60%):** how much DC-driven generation interconnection capacity is queued relative to the utility's zone size
- **Momentum (40%):** how fast that queued capacity is growing (new filings in the trailing 3 months)

ICR remains available as a fallback (`--regulated-signal icr`); DC load becomes the new default.

**Success measure:** IC@21d and IC@63d for PJM regulated tickers (D, PEG, EXC, AEP, PPL, FE) improve vs. the hard-switch-with-ICR baseline (IC@21d 0.1434, t-stat 1.865).

---

## Research Findings (This Session)

### PJM has no data-center-specific queue
Data centers connect at the distribution level through the local utility — they don't file a PJM interconnection queue entry themselves. The queue only contains **generation** interconnection requests. The proxy signal is co-located generation build in DC-heavy zones (e.g. Microsoft/Constellation at Three Mile Island shows up in the DOM/PSEG/ComEd queue as a generation project, not a load project). This is the only PJM-public signal available for v1; a true load-side signal would require PJM's annual "Large Load Additions" workshop PDFs (deferred to v2 — manual scraping, annual frequency only).

### `gridstatus.PJM.get_interconnection_queue()` is broken in the installed version
`gridstatus==0.34.0` raises `AssertionError` on `get_interconnection_queue()` (a `"Revised In Service Date"` column PJM removed upstream isn't handled by this gridstatus version). Use `get_raw_interconnection_queue()` instead — returns a `BytesIO` of the raw PJM Excel export, read with `pd.read_excel()`. Verified live: 9,263 rows, 43 columns, covering `Submitted Date` back to 1998.

Relevant columns:
- `Transmission Owner` — maps to `TICKER_NODE_MAP[...]["load_zones"]` zone names, with a few name mismatches to fix (`ComEd`→`COMED`, `Dominion`→`DOM`, `Dayton`→`DAYTON`, `PSEG; PSEG`→`PSEG`)
- `MW Capacity`, `Project Type` (filter to `"Generation Interconnection"`), `Status`
- `Submitted Date`, `Withdrawal Date`, `Actual In Service Date` — present on all 9,263 rows including historical ones, not just currently-active projects

### Key design insight: point-in-time reconstruction from a single snapshot
The queue endpoint returns the **current** full history, not a point-in-time snapshot — but because every row carries its actual `Submitted Date` / `Withdrawal Date` / `Actual In Service Date`, a single pull is enough to reconstruct "what was queued as of date D" for any historical D:

```
in_queue(project, D) = submitted_date <= D
                        AND (withdrawal_date is null OR withdrawal_date > D)
                        AND (actual_in_service_date is null OR actual_in_service_date > D)
```

**`Status` is not used as a filter.** `Status` (e.g. `"Withdrawn"`, `"Active"`) is a current-snapshot field with no associated date — a project's status today doesn't tell us its status as of historical D. Filtering rows by today's `Status` would silently drop projects that were validly in-queue as of D but were later withdrawn, double-counting the withdrawal logic already handled correctly by `withdrawal_date` above and corrupting the point-in-time reconstruction. `DC_QUEUE_EXCLUDED_STATUSES` from the original draft of this spec is dropped for that reason — `in_queue()` is the only membership filter.

This avoids the monthly-snapshot-caching approach originally considered, and gives clean coverage across the full 2018–2025 backtest window. `Projected In Service Date` is **not** used as an exit condition — projects routinely slip their projected date by years while still sitting in the queue.

**Caveat (accepted for v1):** this reconstruction assumes PJM's public list doesn't silently purge old entries. Extent of any such gap is unknown but assumed small given 1998-era entries are still present. For live/forward use after this ships, add monthly snapshot caching as a v2 hardening step so future queue corrections/backfills by PJM don't retroactively change history.

**Known limitation — MW Capacity is not itself point-in-time (flagged in review, not solved in v1):** the *presence* of a project in the queue as of date D is correctly reconstructed from dated fields, but the `MW Capacity` value attached to each row is PJM's **current, latest-amended** figure. Interconnection projects are routinely resized during their study process — a project queued at 50MW in 2019 that was upsized to 400MW by 2023 will contribute 400MW to `queued_mw(t, 2019)`, which the project didn't actually represent at the time. This is a real look-ahead bias, concentrated in exactly the early backtest years (2018–2021) where the strategy most needs to demonstrate edge. Fully solving it requires historical queue archives (PJM does not appear to publish them going back further than the current snapshot). Accepted as a v1 limitation — flag explicitly in the results writeup rather than presenting early-window IC as clean.

### Zone-level load requires a new fetch function
`grid_data.fetch_load()` only returns ISO-wide total load, not zone-level. Zone-level load exists via `gridstatus.PJM.get_load_metered_hourly(start, end)`, which returns a `Zone` column with values matching `TICKER_NODE_MAP` zone names directly (e.g. `DOM`, `AEP`, `PSEG`). This needs a new `fetch_zonal_load()` function in `grid_data.py`, PJM-only.

---

## Signal Construction

For ticker `t` with zones `Z_t = TICKER_NODE_MAP[t]["load_zones"]`, as of date `D`:

```
queued_mw(t, D) = Σ MW Capacity
                  over queue projects p where:
                    TO_ZONE[p.transmission_owner] ∈ Z_t
                    AND p.project_type == "Generation Interconnection"
                    AND p.mw_capacity >= DC_QUEUE_MW_MIN (100)
                    AND in_queue(p, D)   # point-in-time filter above

new_mw_3mo(t, D) = same filter as queued_mw, plus
                    p.submitted_date > D - DC_MOMENTUM_WINDOW_DAYS (90)

zone_size(t, D)  = Σ trailing-12-month average hourly load (MW)
                    over zones z ∈ Z_t, from fetch_zonal_load()

level(t, D)     = queued_mw(t, D)    / zone_size(t, D)
momentum(t, D)  = new_mw_3mo(t, D)   / zone_size(t, D)

dc_score(t, D) = DC_LEVEL_WEIGHT * level(t, D) + DC_MOMENTUM_WEIGHT * momentum(t, D)
              = 0.6 * level(t, D) + 0.4 * momentum(t, D)
```

`dc_score` is a raw (unnormalized) ratio. It flows into `build_factor()`'s existing regulated-path slot exactly where ICR currently plugs in — that function already cross-section z-scores and winsorizes whatever series it receives, so no changes to the blending math are needed.

Multi-zone tickers (EXC has 4 zones) sum MW and zone size across all their zones before dividing — a single blended ratio, not zone-by-zone averaging.

### v1 scope limitation: PJM-only, with an ICR fallback (not a flat mean)
The DC load signal only has data for PJM tickers — 6 of the 14 non-merchant tickers (D, PEG, EXC, AEP, PPL, FE). The original draft of this spec let `build_factor()`'s existing NaN-fill (cross-sectional mean of available values) handle the other 8 (WEC, CMS, AEE, PCG, EIX, XEL, EVRG, plus mixed tickers ETR/DTE/CNP). Review caught that this is worse than it looks: those tickers currently get a real (if noisy) differentiated ICR reading, and collapsing 8 of 14 regulated-path tickers to one identical constant concentrates all cross-sectional dispersion — and winsorization — onto just the 6 PJM names. Since the evaluation plan only checks IC on the 6 PJM tickers, this regression on the other 8 wouldn't be caught by the stated success measure.

**Fix:** `compute_dc_load_signal()` returns `NaN` for non-PJM tickers as before, but the caller in `main.py` fills those `NaN`s from that ticker's own ICR reading (`fetch_icr()`, still fetched alongside) rather than letting `build_factor()` fall through to a flat cross-sectional mean. Only fall through to the mean if *both* DC and ICR are unavailable for a ticker. This keeps real per-ticker differentiation for the 8 non-PJM names instead of flattening them, at the cost of mixing two different signal scales before z-scoring — acceptable since `build_factor()` z-scores and winsorizes the combined series anyway. `dc_load_data.py` gains a small `fill_with_icr(dc_series, icr_series)` helper; `main.py`'s `--regulated-signal dc-queue` path calls both fetchers.

### Reporting lag
ICR needs a 45-day lag (`_ICR_LAG_DAYS`) because it's a quarterly SEC filing. The interconnection queue is a live public dataset with real submission dates — no lag is needed. `build_rolling_factor()` gets a new `regulated_signal_lag_days` parameter (default `_ICR_LAG_DAYS`, `main.py` passes `0` for `dc_queue`).

---

## Config Additions

`config.py`:
```python
from typing import Literal

REGULATED_SIGNAL: Literal["icr", "dc_queue"] = "dc_queue"

DC_QUEUE_MW_MIN         = 100.0   # ignore sub-100MW queue entries as noise
DC_QUEUE_PROJECT_TYPES  = ["Generation Interconnection"]
DC_MOMENTUM_WINDOW_DAYS = 90
DC_LEVEL_WEIGHT    = 0.6
DC_MOMENTUM_WEIGHT = 0.4
```

**These are v1 defaults, not derived.** Unlike `STRESS_SPIKE_PCT` / `STRESS_CONG_THRESHOLD` / `STRESS_CONG_MIN_DAYS` (CLAUDE.md, validated via a 243-combo grid search), `DC_QUEUE_MW_MIN`, `DC_MOMENTUM_WINDOW_DAYS`, and the 60/40 level/momentum split are asserted based on reasoning, not swept. Do not treat the first backtest result as conclusive about whether the DC signal beats ICR — a parameter sweep (mirroring `grid_search.py`) should follow once the signal is wired up and produces sane output, before drawing a final verdict.

CLI flag in `main.py`:
```
--regulated-signal [icr|dc-queue]   Signal used for the regulated path of --arch hard-switch
                                     (default: dc-queue)
```

---

## Files Changed

| File | Change |
|---|---|
| `grid_resilience/data/grid_data.py` | New `fetch_zonal_load()` — PJM `get_load_metered_hourly()`, cached as monthly parquet chunks (same pattern as `fetch_load`), dataset key `"load_zonal"` |
| `grid_resilience/data/dc_load_data.py` (new) | `fetch_interconnection_queue()` (cached raw-Excel pull via `get_raw_interconnection_queue()`), `TO_TO_ZONE` mapping dict, `compute_dc_load_signal(tickers, node_map, as_of_dates, zonal_load)` → DataFrame indexed by date with ticker columns, matching `icr_history`'s shape; `fill_with_icr(dc_series, icr_series)` helper for the non-PJM fallback |
| `grid_resilience/config.py` | Add `REGULATED_SIGNAL`, `DC_QUEUE_MW_MIN`, `DC_QUEUE_PROJECT_TYPES`, `DC_QUEUE_EXCLUDED_STATUSES`, `DC_MOMENTUM_WINDOW_DAYS`, `DC_LEVEL_WEIGHT`, `DC_MOMENTUM_WEIGHT` |
| `grid_resilience/factor/resilience_score.py` | No signal-math changes. Add `regulated_signal_lag_days: int = _ICR_LAG_DAYS` param to `build_rolling_factor()`, threaded into `_icr_at_date()`'s cutoff calc. Docstrings updated to describe the `icr` param generically ("regulated-path signal — ICR or DC load score") |
| `grid_resilience/main.py` | Add `--regulated-signal` CLI flag; branch between `fetch_icr()` and `dc_load_data.compute_dc_load_signal()`; pass `regulated_signal_lag_days=0` when `dc_queue` |
| `tests/data/test_dc_load_data.py` (new) | Point-in-time filter correctness (submitted/withdrawn/in-service boundary cases), `TO_TO_ZONE` mapping, multi-zone summation, level/momentum math |
| `tests/factor/test_business_model_arch.py` | No changes needed — `icr=` kwarg name is unchanged, existing tests keep passing unmodified |

No existing tests should break — `REGULATED_SIGNAL` only affects behavior when `--arch` is passed, and `build_factor()`'s signature and math are untouched.

---

## Implementation Sequence

1. **`fetch_zonal_load()`** in `grid_data.py` — new PJM zonal load fetch + cache. Verify against a spot-check zone total vs. ISO-wide `fetch_load()` total. Also verify `get_load_metered_hourly()` actually has coverage back to early 2017 (needed for a trailing-12-month `zone_size` at the 2018-01-01 backtest start) — if PJM's metered-load history is shorter than that, the first ~12 months of the backtest will have unstable or `NaN` denominators and need an explicit handling decision (e.g. shrink the trailing window near the start, or accept `NaN` factor scores for those months).
2. **`dc_load_data.py`** — queue fetch/cache, `TO_TO_ZONE` mapping, point-in-time filter, level/momentum computation. Unit tests first (TDD) on synthetic queue rows covering submit/withdraw/in-service boundaries.
3. **Config + CLI** — add config constants and `--regulated-signal` flag.
4. **Wire into `main.py`** — branch on `regulated_signal`, build the DC-load history DataFrame in the same shape `build_rolling_factor()` expects for `icr_history`.
5. **`resilience_score.py`** — add `regulated_signal_lag_days` param, thread through `_icr_at_date`.
6. **Run backtest** — `--arch hard-switch --regulated-signal dc-queue`, compare IC@21d/63d against the ICR baseline (0.1434 / 0.2027).
7. **Record results** in a follow-up compact doc; decide whether `dc_queue` becomes the config default (it already does per this spec) or needs another iteration.

---

## Evaluation

```bash
python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal icr
python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-queue
```

**⚠️ ICR baseline has almost no historical coverage.** The cached ICR frame only has usable values from 2024-12-31 onward (7 quarters). For ~85% of the 2018-2025 backtest window, "Hard switch + ICR" has NO regulated-path signal at all and silently falls back to the cross-sectional-mean fill. Treat the ICR row below as measuring mostly 2025 behavior, not full-history behavior.

**⚠️ Results below are NOT comparable to the `0.331` figure this spec was written against.** During implementation (2026-08-13), a pre-existing PJM data gap (one silently-missing LMP month) was discovered and incidentally fixed, and a second, broader equity-price data drift was found. Both change backtest numbers independent of this feature's code — see `docs/compact_2026-08-13-dc-load-signal-results.md` for the full account. The table below reports both configurations measured on the *same* (today's, complete) data, which is the only valid comparison right now.

**⚠️ 2026-08-13 UPDATE:** The row below was originally measured with a Critical zone-name-mismatch bug (only 3 of 6 PJM tickers actually had real DC data) and an Important z-score-mixing bug in `fill_with_icr`. Both are now fixed — see `docs/compact_2026-08-13-dc-load-signal-results.md` for the full account, including the retraction of the original "PEG flips to a long candidate" claim (PEG was on the ICR fallback path when that claim was made, not the DC signal).

| Configuration | Sharpe | Ann Ret | Max DD | IC@21d | IC@63d |
|---|---|---|---|---|---|
| Hard switch + ICR (today's data) | -0.276 | 0.80% | -21.93% | 0.0615 (t=1.261) | 0.1028 (t=2.174) |
| Hard switch + DC load signal (today's data, post-fix) | -0.213 | 2.06% | -19.63% | 0.0290 (t=0.766) | 0.0680 (t=2.008) |

DC load signal improves Sharpe, Ann Ret, and Max DD over ICR on identical data, but still has weaker IC at both horizons — a mixed result, not a clean win, qualitatively unchanged from the pre-fix numbers even though every absolute value moved. Now genuinely reflects all 6 PJM tickers (AEP, D, EXC, FE, PPL, PEG) on the DC path ("DC load signal: 6 PJM tickers with real data, 17 filled from ICR"). See the compact doc for the corrected qualitative check — PEG does NOT flip to a long candidate on the real DC signal (that was an ICR-fallback artifact); D flips sign but stays mid-pack; FE is now the top-scoring name, unanticipated by the original motivation.

---

## Deferred to v2

- PJM "Large Load Additions" workshop PDFs (Layer 3) — real zone-level DC MW projections, requires PDF scraping, annual frequency
- EIA-861 realized load-by-customer-class — extends coverage beyond PJM, tracks realized (not just queued) growth
- Monthly interconnection-queue snapshot caching for forward/live use, so future PJM corrections don't retroactively rewrite backtest history
- `pass_through` verification against 10-K disclosures (already flagged as open in the business-model-arch spec, unaffected by this change)
