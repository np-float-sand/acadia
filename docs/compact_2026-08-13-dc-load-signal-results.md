# Session Compact — 2026-08-13: DC Load Signal Implementation + Data-Drift Discovery
**Branch:** res2
**All tests:** 79 passing
**Status:** DC load signal feature complete and merged into `main.py`; a pre-existing PJM data gap and a broader equity-price data drift were discovered mid-implementation — **historical performance numbers in CLAUDE.md and prior compact docs (Sharpe 0.331 / 0.272) should not be trusted until re-validated on today's data.**

---

## TL;DR

1. Implemented the DC load signal (level + momentum from PJM's interconnection queue) as designed in `docs/superpowers/specs/2026-07-06-dc-load-signal-design.md`, via 8 reviewed subagent-driven tasks. All code is committed, tested, reviewed clean.
2. While running the live-verification backtest for this feature, discovered that a documented "reference" Sharpe (0.331) from June/July sessions does not reproduce on the current codebase — not because of a bug in this feature, but because of two data issues unrelated to it. Both are flagged here as the priority follow-up, ahead of any further signal work.
3. On today's (complete, but possibly still-drifting) data, DC load signal beats ICR on Sharpe/Max DD but is weaker on IC. Not a clean win — see below.

---

## Part 1 — Data-Drift Discovery (higher priority than the feature itself)

### Finding A: PJM LMP cache was silently incomplete
`grid_resilience/data/cache/pjm_lmp_DAY_AHEAD_HOURLY_2024-06.parquet` had no cache file until today. PJM's fetcher deliberately never writes an empty placeholder on failure (`grid_data.py`, so a 429-rate-limited month keeps retrying on every future run) — but apparently no run between whenever that gap first appeared and today ever succeeded in backfilling it, so the strategy has been silently running on **incomplete PJM data for at least one month** (June 2024 — plausibly a real PJM heat-wave stress month) every time "PJM included" numbers were recorded, including the 0.331 Sharpe documented in `CLAUDE.md` and `docs/compact_2026-07-05.md`.

Today's live-verification API calls (Tasks 1, 2, and 8 of the implementation plan) incidentally succeeded in fetching that month. The cache is now **fully complete for the first time** — verified 0 missing months for both PJM LMP and PJM load across the full 2018–2025 backtest window.

There's a prior, never-fully-executed plan (`docs/superpowers/plans/2026-06-19-pjm-data-quality-fix.md`) that already diagnosed this class of problem (18 missing LMP months as of 2026-06-19) and implemented the retry/backoff fix and a GSI forward-fill safeguard (both already committed, already covered by `tests/data/test_pjm_retry.py` and `tests/signals/test_gsi_ffill.py`) — but its Task 4 (actually refetching all missing months) was apparently never fully completed, or a fresh gap reappeared afterward. Either way: **as of today, for the first time, the cache is complete.**

**Effect measured:** re-running `--arch hard-switch --regulated-signal icr` (identical code, identical CLI args to what previously produced Sharpe 0.331) now gives **Sharpe -0.276, IC@21d 0.0615** on the complete cache.

### Finding B: a second, broader drift — likely equity price data
Excluding PJM entirely (`--iso ERCOT MISO CAISO SPP`, matching the documented "without PJM" scenario) does **not** reproduce the documented Sharpe 0.272 either — it now gives **Sharpe 0.165**, though IC@21d (0.1413) is close to the documented 0.144. Since the PJM cache fix cannot affect non-PJM tickers at all, this points to a second, independent source of drift — most likely `yfinance` equity price data has changed (revised adjusted closes, extended date range, etc.) since the reference numbers were recorded. Not investigated further this session.

### What this means
- **Every Sharpe/Ann-Ret/Max-DD number in `CLAUDE.md`'s "Grid Search Findings" and every prior compact doc's performance table should be treated as unverified until re-run on current data.** The IC numbers (factor quality) seem to survive better than the portfolio-level backtest numbers, but even those don't match exactly.
- The grid-search-winning parameters (`STRESS_SPIKE_PCT=0.97` etc.) and the business-model-arch architecture choice (hard-switch over revenue-mix/dual-track) were selected by comparing Sharpe ratios across configs — if the underlying data was drifting between those comparison runs too, the *ranking* between configs might still hold (relative comparisons on the same data snapshot), but the *absolute* numbers reported as "the winning config's performance" should not be quoted going forward.
- **Recommended next session's first task:** re-run the full grid search and the hard-switch/revenue-mix/dual-track comparison from `docs/superpowers/specs/2026-06-29-business-model-aware-factor-design.md` on today's complete data, before doing further signal development. This was explicitly deferred this session per user direction (proceed with the DC load signal work using today's numbers, flag this prominently) — it hasn't been done yet.

---

## Part 2 — DC Load Signal Feature (this session's planned work)

### Implementation
Executed via `superpowers:subagent-driven-development` against `docs/superpowers/plans/2026-07-06-dc-load-signal-implementation.md`, 8 tasks, each with a fresh implementer subagent + independent task review:

| Task | What | Commits |
|---|---|---|
| 1 | `fetch_zonal_load()` — PJM zone-level metered load | fd8d7cd..f1bd23c |
| 2 | PJM interconnection queue fetch + cleaning (`dc_load_data.py`) | f1bd23c..553a856 |
| 3 | Point-in-time queue membership (`in_queue`, `queued_mw`, `new_mw_since`) | 553a856..06337c8 |
| 4 | `zone_size()` trailing-12-month load denominator | 06337c8..20bd1ed |
| 5 | `compute_dc_load_signal()` — combine level + momentum per ticker | 20bd1ed..817d303 |
| 6 | `fill_with_icr()` — non-PJM ticker fallback | 817d303..c4cca1e |
| 7 | Generalized ICR reporting lag into `regulated_signal_lag_days` param | c4cca1e..7c111a0 |
| 8 | `--regulated-signal icr\|dc-queue` CLI flag, wired into `main.py` | 7c111a0..367ff8f |

All 8 tasks reviewed clean (no unresolved Critical/Important findings). Task 3's review caught a real gap (no test hit the exact boundary date for the safety-critical `in_queue` inequality) — fixed with 3 additional boundary tests before approval; the underlying logic was already correct. Minor findings across tasks (DRY duplication in PJM pagination logic, non-vectorized loop in `fill_with_icr`, a few cosmetic log-message issues) are logged in `.superpowers/sdd/progress.md` but not blocking.

**Process note:** Task 8's original implementer subagent was killed mid-run when the host machine went to sleep during a backgrounded live-verification step. The controller resumed directly (not a fresh subagent) — verified the already-correct diff, ran tests and both live backtests in the foreground, and completed the task normally.

### Results (on today's complete-but-drifting data — see Part 1 caveat)

| Configuration | Sharpe | Ann Ret | Max DD | IC@21d | IC@63d |
|---|---|---|---|---|---|
| Hard switch + ICR | -0.276 | 0.80% | -21.93% | 0.0615 (t=1.261) | 0.1028 (t=2.174) |
| Hard switch + DC load signal | 0.043 | 4.46% | -17.55% | 0.0274 (t=0.715) | 0.0770 (t=2.111) |

**Mixed result.** DC load signal improves Sharpe (+0.32) and Max DD (+4.4pp) over ICR, but has *weaker* IC at both horizons (21d: 0.027 vs 0.061; 63d: 0.077 vs 0.103). This does not cleanly clear the design spec's original success bar ("IC improves vs. baseline") — it trades signal purity for portfolio-level risk-adjusted return. Worth another look once Part 1's data issues are resolved, since both configurations' absolute levels are suspect right now.

### Qualitative check — the actual motivation for this feature
Latest rebalance (2025-12-31) factor scores under `--regulated-signal dc-queue`:

```
AEE   1.50   EVRG  1.49   PEG   1.13  ← long
ETR   1.08   EIX   0.61   EXC   0.40
PPL   0.39   CMS   0.22   XEL   0.10
DTE   0.10   WEC  -0.03   CNP  -0.17
PCG  -0.45   NRG  -0.60   FE   -1.03
AEP  -1.39   D    -1.44   VST  -1.91  ← short
```

- **PEG (PSEG/NJ corridor): scores 1.13, 3rd of 18 — flips to a long candidate.** This validates the core hypothesis: PEG was previously scored on ICR/leverage alone, missing its NJ data-center exposure; the DC load signal now captures it.
- **D (Dominion/Virginia, the single largest DC market in the dataset): scores -1.44, 2nd-worst — does NOT flip.** This contradicts the qualitative motivation that specifically named D as the clearest case (Northern Virginia, ~250 DCs). Not investigated further this session — leading hypothesis is that DOM's zone already has a very large existing load base, diluting the queued-MW/zone-size ratio even if absolute queue activity is substantial. Worth checking `queued_mw(queue, "DOM", as_of)` and `zone_size(zonal_load, ["DOM"], as_of)` directly before trusting this result.

---

## Files Changed

- `grid_resilience/data/grid_data.py` — `fetch_zonal_load()`, `_fetch_pjm_load_by_zone_direct()`
- `grid_resilience/data/dc_load_data.py` (new) — full DC load signal module
- `grid_resilience/config.py` — `REGULATED_SIGNAL`, `DC_QUEUE_MW_MIN`, `DC_QUEUE_PROJECT_TYPES`, `DC_MOMENTUM_WINDOW_DAYS`, `DC_LEVEL_WEIGHT`, `DC_MOMENTUM_WEIGHT`
- `grid_resilience/factor/resilience_score.py` — `regulated_signal_lag_days` param (generalizes the old hardcoded 45-day ICR lag)
- `grid_resilience/main.py` — `--regulated-signal` CLI flag and wiring
- `tests/data/test_zonal_load.py`, `tests/data/test_dc_load_data.py` (new), `tests/factor/test_resilience_score.py` (+1 test)
- `docs/superpowers/specs/2026-07-06-dc-load-signal-design.md` — Evaluation table updated with actual results + data-drift caveat

## Open Items / Next Steps

1. **Priority: re-validate the grid search and business-model-arch comparison on today's complete data** (Part 1). Until this is done, don't quote any Sharpe/Ann-Ret/Max-DD number from before 2026-08-13 as ground truth.
2. Investigate why D doesn't flip to long under the DC load signal despite being the motivating case.
3. The v1 60/40 level/momentum weights and 100MW/90-day thresholds were asserted, not grid-searched (flagged in the design spec's review round) — worth a sensitivity sweep once the data-drift issue is resolved and results are trustworthy again.
4. `MW Capacity` in the interconnection queue reflects PJM's *current* (possibly since-amended) project size, not what was known at each historical `as_of` date — a mild, accepted look-ahead bias flagged in the design spec, unresolved (would need historical queue archives PJM doesn't appear to publish).
5. Minor code-quality findings logged in `.superpowers/sdd/progress.md` across Tasks 1-8 — none blocking, worth a cleanup pass if this branch gets more attention.
6. The parked zone-GSI experiment from the 2026-06-28 session (rejected hypothesis, stashed as `stash@{0}` before this session's Task 1) is still sitting in the git stash, untouched, per earlier user direction.

## How to Resume

```bash
source ~/.zshrc
.venv/bin/python -m pytest tests/ -q  # should be 79 pass
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-queue
# Compare against --regulated-signal icr on the same data
```

Then start with Open Item 1 (re-validate historical baselines) before any further signal iteration.
