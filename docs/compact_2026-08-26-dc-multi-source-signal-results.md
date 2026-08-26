# Session Compact — 2026-08-26: Multi-Source DC Demand Exposure Signal — Backtest Results

**Branch:** res2
**All tests:** 137 passing
**Status:** Feature complete, wired in behind `--regulated-signal dc-multi` (default remains `dc_queue`). **Coverage target missed (9 of ~11 tickers) and the design's "material, not sign-flip" bar is not cleared** — reported plainly below, consistent with how `docs/compact_2026-08-13-dc-load-signal-results.md` and `docs/compact_2026-08-19-peer-group-construction-results.md` treated similar results.

---

## TL;DR

1. Implemented the 3-layer multi-source DC demand signal designed in `docs/superpowers/specs/2026-08-26-dc-demand-exposure-signal-design.md` via 11 reviewed tasks. Layer 2 (PJM Large Load cross-check + FE zone-map fix) and Layer 3 (hyperscaler PPA deal events) both work and are wired into the backtest. **Layer 1 (ERCOT) does not** — it returns zero real ticker coverage in production because ERCOT's public large-load chart, verified across every available deck, plots project *count*, not MW.
2. Live-verified real coverage is **9 tickers, not the ~11 the design spec estimated** — and of those 9, only **1 (VST) is net-new** versus the pre-existing single-source `dc-queue` signal. The other 8 (AEP, D, EXC, FE, PPL, PEG, CEG, TLN) already had real data before this plan started; CEG and TLN simply had their data source swapped from the generation-queue proxy to the (design-intended, cleaner) hyperscaler-deal signal, per an explicit precedence rule.
3. On today's live-verified data, `dc-multi` (9 tickers) beats the immediate predecessor `dc-queue` (8 tickers) on every metric tested — Sharpe, Ann Ret, Max DD, and IC@21d all improve. It also posts a small Sharpe/Ann-Ret/Max-DD improvement over the `icr` fallback baseline, but its IC@21d is **worse** than ICR's, and none of the three configurations come within reach of beating XLU. **This does not clear the design's own pre-stated "material, not sign-flip" bar.**
4. Real, valuable data-quality work happened along the way and should not be discounted just because the backtest result is muted: a genuine FE zone-mapping gap (missing METED/PENELEC) was fixed, three factual errors in the hyperscaler seed data were caught and corrected against primary sources (one of them already marked "confirmed" before closer verification, wrong by ~10x), and a real column-collision architecture bug was found and fixed during integration.

---

## What was built

| Layer | Module | Status |
|---|---|---|
| 1 — ERCOT large-load queue | `grid_resilience/data/ercot_large_load_data.py` | Built, tested, **returns all-NaN in production** — see below |
| 2 — PJM Large Load Adjustment cross-check + FE zone fix | `grid_resilience/data/pjm_large_load_data.py`, `utility_node_map.py` | Built, working — 8/8 tickers agree between the generation-queue signal and PJM's own industry tags |
| 3 — Hyperscaler PPA deal events | `grid_resilience/data/hyperscaler_deals.py`, `grid_resilience/data/seed/hyperscaler_deals.csv` | Built, working — covers CEG, TLN, VST (not NRG — no verified deal found) |
| Combination | `grid_resilience/data/dc_demand_combined.py` (`combine_dc_demand_layers`, `apply_layer_precedence`) | Built, tested — hyperscaler > PJM generation-queue > ERCOT precedence |
| Wiring | `grid_resilience/main.py` — `--regulated-signal dc-multi` | Built, live-verified end-to-end |
| Short-sample harness | `grid_resilience/analysis/short_sample_report.py` (`window_sensitivity_report`, `sample_size_caveat`) | Built, tested, used below |

### Layer 1 (ERCOT) — a real null result, not a minor caveat

ERCOT's public "Large Load Project Distribution - TSP" chart — the source Layer 1 was designed around — prints project **count** on its bars in every deck checked (verified across 3 independent decks during data collection). The seed loader and `compute_ercot_signal` are built and unit-tested, and the seed CSV (`grid_resilience/data/seed/ercot_large_load_monthly.csv`) does contain real system-wide MW rows — but every row has `scope=system_wide` and an empty `tsp` field, so there is no ticker-level MW to attach to AEP or CenterPoint (CNP). Confirmed live: `dc-multi`'s coverage log line names zero ERCOT tickers. This is load-bearing infrastructure with a dead data source, not a rounding error in the ticker count.

### Layer 3 (hyperscaler deals) — real data, with real corrections made along the way

6 deal rows, 3 tickers (CEG, TLN, VST), spanning 2024-03-04 to 2026-01-09. Both rows the plan flagged `needs_verification` were resolved against primary sources (0 rows remain unverified in `hyperscaler_deals.csv` today) — one matched the brief's placeholder exactly (CEG/Meta, 1,121 MW), one didn't (TLN/Amazon: corrected 1,920 MW → 960 MW and the date by 3 days, verified via Talen's own PRNewswire release). While doing that work, a **third** row — already marked `confirmed_primary_source` before this plan started — was found wrong by roughly 10x (1,920 MW → 180 MW, verified against the actual FERC order and Talen's own 8-K) and corrected. A fourth, smaller fix (an `event_type` mislabel that would have misled a reader into thinking a dispute was "resolved" when it wasn't) was caught in post-review and also fixed. No NRG deal was found and verified, so the merchant sleeve is 3 tickers, not the 4 (CEG, TLN, VST, NRG) the design spec named.

### A real architecture bug caught during integration

CEG and TLN are tagged `iso="PJM"` in `TICKER_NODE_MAP`, so they already had real coverage from the pre-existing generation-queue signal before this plan touched anything. Combining that with the new hyperscaler layer via a naive `pd.concat` silently triplicated column names for every one of the 25 universe tickers (not just CEG/TLN) rather than merging them, which crashed the pipeline. Fixed with `apply_layer_precedence()`: hyperscaler (named, disclosed deals) > PJM generation-queue proxy > ERCOT, i.e., the more specific signal wins for any ticker more than one layer covers. A related dtype bug (`pd.NA` from a zero-spread z-score row silently upcasting a column to `object` and crashing a downstream boolean-mask assignment) was also found and fixed at the source (`_zscore_columns`).

---

## Results (live-verified 2026-08-26)

All three configurations run today, same command shape, same universe, same date range (2018–2025):

```
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-multi
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal icr
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-queue
```

| Configuration | Real ticker coverage | Sharpe | Ann Ret | Ann Vol | Max DD | IC@21d (t-stat, N) |
|---|---|---|---|---|---|---|
| **`dc-multi`** (this plan) | 9 (AEP, D, EXC, FE, PPL, PEG, CEG, TLN, VST) | **-0.302** | **0.71%** | 10.91% | **-23.89%** | 0.0104 (t=0.282, N=84) |
| `icr` (fallback only) | 0 real (all 25 tickers on ICR) | -0.331 | -0.30% | 12.99% | -29.08% | **0.0417** (t=0.969, N=84) |
| `dc-queue` (immediate predecessor, pre-this-plan) | 8 (AEP, D, EXC, FE, PPL, PEG, CEG, TLN) | -0.559 | -1.64% | 10.08% | -30.01% | 0.0073 (t=0.200, N=84) |
| XLU benchmark | — | 0.177 | 6.35% | 13.22% | -17.28% | — |
| EW Universe | — | 0.156 | 7.22% | 20.63% | -42.03% | — |

No formal significance test was run comparing Sharpe/Ann-Ret across configurations (no bootstrap CI) — the deltas above should be read as point-estimate comparisons, not statistically established differences. All IC t-stats are well below conventional significance (~2); none of these numbers should be read as "the signal works," only as "which is least bad, on this window."

### `dc-multi` vs. `dc-queue` (the direct predecessor): a real, if modest, win

Every metric tested improves: Sharpe (-0.559 → -0.302), Ann Ret (-1.64% → 0.71%), Max DD (-30.01% → -23.89%), IC@21d (0.0073 → 0.0104). Worth being honest about the mechanism, though: most of this improvement is not from the one net-new ticker (VST) or from Layer 1 (which contributes nothing) — it's substantially from routing CEG and TLN through the hyperscaler-deal signal instead of the noisier generation-queue proxy they used before. That re-routing is exactly what the design intended (named deals are "the cleanest exposure in the universe" per the spec), so this is a legitimate result of the plan's design, not an accident — but it means the *coverage-expansion* half of the plan's stated rationale (§Goal: "adds real... coverage") delivered mostly a data-quality upgrade for two already-covered tickers, not the breadth expansion the ~11-ticker target implied.

### `dc-multi` vs. `icr`: does not clear the "material, not sign-flip" bar

Sharpe improves only marginally (+0.029, both still negative — this is not even a sign flip on the headline risk-adjusted metric). Ann Ret does flip sign (-0.30% → +0.71%), but by under 1 percentage point — the kind of small, untested-for-significance move the design spec's own validation section explicitly warned against treating as evidence ("this project has been burned twice already by short-sample-looking-good results that didn't survive scrutiny"). Max DD improves by a real 5.19pp, the most substantive of the four deltas. But IC@21d — the more direct measure of whether the factor score actually predicts forward returns — gets **worse**, not better (0.0417 → 0.0104), and neither IC is close to statistically significant. This is the same qualitative shape (portfolio-level metrics improve, IC does not) as the single-source `dc-queue` result documented in `docs/compact_2026-08-13-dc-load-signal-results.md` — not a new problem, but not a resolved one either.

**None of the three configurations beat XLU** (Sharpe 0.177, Ann Ret 6.35%). This is consistent with, not contradicted by, `docs/compact_2026-08-26-original-thesis-reexamination.md`'s broader conclusion that no signal formulation or portfolio construction tried to date beats XLU on this backtest window.

---

## Window-sensitivity and sample-size analysis (`short_sample_report.py`)

The pipeline does not currently decompose portfolio return by signal layer (Task 10 confirmed only a combined `pnl.csv` exists) — building that would need real plumbing beyond this task's scope, so two separate, honest things are reported instead.

### 1. Portfolio-level result, monthly-resampled to match the pipeline's own rebalance cadence

`output/pnl.csv`'s daily `strategy` log-returns for `dc-multi`, resampled to month-end (`REBALANCE_FREQ = "ME"`) and trimmed to the first date the strategy actually holds a nonzero position (2018-09-28), giving n=88 — exactly matching the pipeline's own `[ic] dates: 88` count:

| Variant | n_periods | Mean monthly log-return |
|---|---|---|
| full | 88 | 0.0644% |
| drop_first | 87 | 0.0743% |
| drop_last | 87 | 0.0479% |
| drop_first_and_last | 86 | 0.0577% |

`sample_size_caveat(88)` → empty string (88 is well above the 12-period trust floor). The sign doesn't flip under any trim — but **this is a weaker result than it looks**: the new layers built by this plan only contribute real information for a fraction of these 88 months (hyperscaler deals start 2024-03; ERCOT contributes nothing at any point), so most of this stability is inherited from the pre-existing PJM generation-queue/ICR-blend behavior, not fresh evidence for Layers 1 or 3 specifically. Read this as "the combined portfolio-level result isn't a first/last-month artifact," not as "the new layers are validated."

### 2. Per-layer sample sizes — this is where the short-sample caveat actually bites

| Layer | n (independent observations) | `sample_size_caveat()` |
|---|---|---|
| Layer 3 — hyperscaler deal events | 6 events, 3 tickers, 2024-03-04 to 2026-01-09 (~22 months) | *"Only 6 independent period(s) observed (below the 12-period bar treated as minimally trustworthy here) — treat this result as directional, not confirmed."* |
| Layer 2 — PJM LAS cross-check vintages | 2 (Nov 2024, Nov 2025) | Same caveat, at n=2. (The design spec itself already called this "not a standalone backtest" — confirmed here mechanically.) |
| Layer 1 — ERCOT real observations | 0 | Same caveat, at n=0 — the harness correctly flags total absence of data, consistent with the live coverage finding above. |

None of the three new layers, taken alone, clears even the harness's own 12-period minimum-trust floor. The only layer with a real, longer time series is the pre-existing PJM generation-queue proxy (unchanged by this plan, back to 2018).

---

## Honest bottom line

This plan delivered working infrastructure for 2 of 3 designed layers, real (if modest) data-quality corrections to seed data that would otherwise have silently fed wrong numbers into the factor, and a portfolio-level result that beats its immediate predecessor on every metric tested. It did not deliver the ~11-ticker coverage the design spec estimated (9 achieved, and only 1 of those 9 is genuinely new versus what existed before this plan), Layer 1 is dead in production, and the combined signal does not clear its own pre-stated bar for a material edge — Sharpe stays negative, the one metric that flips sign (Ann Ret) moves by under a point, IC gets worse relative to the ICR fallback, and nothing here beats XLU. Treat this the same way `docs/compact_2026-08-13-dc-load-signal-results.md` treated the original DC-load signal: real, shippable engineering work, not a strategy result to build a pitch around.

---

## Recommendation

- **Leave `REGULATED_SIGNAL` default as `dc_queue` in `config.py`** — this task is documentation-only and does not change code, so no default was flipped. That said, since `dc-multi` strictly dominates `dc-queue` on every metric tested here (Sharpe, Ann Ret, Max DD, IC@21d), a future session should consider promoting `dc-multi` to the default and treating `dc-queue`-only as deprecated — the case against doing so is that neither is validated against XLU or ICR by a material margin, so this would only be "pick the less-bad option," not "ship a validated improvement."
- **Do not invest further in Layer 1 (ERCOT) without a new data source.** The chart it was built around structurally cannot be coerced into per-ticker MW; a fix would require ERCOT publishing a different report, not a parsing change.
- **Layer 2's LAS cross-check should keep accumulating annual vintages** (currently 2) before being treated as anything beyond a current-state consistency check, per the design spec's own stated limitation.
- If a future session wants to actually validate Layer 3 in isolation, it would need per-layer return attribution wired into the backtest (not currently available) — flagged here as the concrete next step if the hyperscaler-deal signal specifically is worth pursuing further, rather than re-running the same combined-portfolio comparison and re-deriving the same n=6 caveat.

## Files changed this session (Task 12 — documentation only)

- `docs/compact_2026-08-26-dc-multi-source-signal-results.md` (this file, new)

No code, config, or test files were modified. `docs/superpowers/specs/2026-08-26-dc-demand-exposure-signal-design.md` was checked for a placeholder Evaluation table per the task brief — it has no "Evaluation" section at all (only a "Backtest / validation approach" methodology section), so nothing needed updating there.
