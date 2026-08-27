# Session Compact — 2026-08-26: Multi-Source DC Demand Exposure Signal — Backtest Results

**Branch:** res2
**All tests:** 146 passing
**Status:** Feature complete, wired in behind `--regulated-signal dc-multi` (default remains `dc_queue`). **Coverage target missed (8 of ~11 tickers) and the design's "material, not sign-flip" bar is not cleared** — reported plainly below, consistent with how `docs/compact_2026-08-13-dc-load-signal-results.md` and `docs/compact_2026-08-19-peer-group-construction-results.md` treated similar results.

> **Revised 2026-08-26 (post whole-branch review).** The first version of this document reported a `dc-multi` result that was materially inflated by three defects the final review found, and explained it with a mechanism that does not exist. All five findings were fixed and the pipeline re-run; **every `dc-multi` number below is from the post-fix run and differs from the original document.** `icr` and `dc-queue` were re-run on the same day and are unchanged (neither code path was touched). See "What the post-review fix wave changed" at the end for the before/after.

---

## TL;DR

1. Implemented the 3-layer multi-source DC demand signal designed in `docs/superpowers/specs/2026-08-26-dc-demand-exposure-signal-design.md` via 12 reviewed tasks. Layer 2 (PJM Large Load cross-check + FE zone-map fix) and Layer 3 (hyperscaler PPA deal events) are both built and wired into the backtest. **Layer 1 (ERCOT) is dead in production** — it returns zero real ticker coverage because ERCOT's public large-load chart, verified across every available deck, plots project *count*, not MW. **And Layer 3, though it computes correctly, has no effect on the factor scores of its own target tickers** under the architecture actually backtested — see the next point.
2. **Layer 3 is structurally inert under `--arch hard-switch`.** `build_factor()` routes the DC-demand signal (`icr_z`) only to tickers with `pass_through < 0.5`; Layer 3's merchant sleeve is CEG (0.80), TLN (0.90), VST (1.00), all of which are scored from `beta_z` instead. Their own factor scores never read Layer 3 — before or after this plan. The only way Layer 3 touches the factor today is by shifting the shared `icr_z` cross-section's mean/std, which perturbs *other*, regulated tickers. That is a renormalization side-effect, not the designed mechanism. Recorded as a "Known limitation" in the design spec; fixing it needs a merchant-path integration, which is out of scope here.
3. Live-verified real coverage is **8 tickers, not the ~11 the design spec estimated** — and **none of the 8 is net-new** versus the pre-existing single-source `dc-queue` signal. All 8 (AEP, D, EXC, FE, PPL, PEG, CEG, TLN) already had real data before this plan started. VST, previously reported as the one net-new name, does **not** count: its only disclosed deal is dated 2026-01-09, after `config.BACKTEST_END = "2025-12-31"`, so the hyperscaler layer has no information about it anywhere in the tested window.
4. On today's live-verified post-fix data, `dc-multi` beats its immediate predecessor `dc-queue` on Sharpe, Ann Ret and Max DD but is **worse on IC@21d**, and it is **worse than the plain `icr` fallback** on both Sharpe and Ann Ret. None of the three configurations come within reach of beating XLU. **This does not clear the design's own pre-stated "material, not sign-flip" bar** — and the earlier claim that `dc-multi` "strictly dominates `dc-queue`" is withdrawn; it does not.
5. Real, valuable data-quality work happened along the way and should not be discounted just because the backtest result is muted: a genuine FE zone-mapping gap (missing METED/PENELEC) was fixed, a PJM LAS zone-vocabulary bug was fixed (AEP's 2030 large-load demand was understated by 34%), three factual errors in the hyperscaler seed data were caught and corrected against primary sources (one of them already marked "confirmed" before closer verification, wrong by ~10x), and two real architecture bugs (a column collision and a layer-precedence displacement) were found and fixed.

---

## What was built

| Layer | Module | Status |
|---|---|---|
| 1 — ERCOT large-load queue | `grid_resilience/data/ercot_large_load_data.py` | Built, tested, **returns all-NaN in production** — see below |
| 2 — PJM Large Load Adjustment cross-check + FE zone fix | `grid_resilience/data/pjm_large_load_data.py`, `utility_node_map.py` | Built, working — 8/8 tickers agree between the generation-queue signal and PJM's own industry tags (after the post-review zone-vocabulary fix) |
| 3 — Hyperscaler PPA deal events | `grid_resilience/data/hyperscaler_deals.py`, `grid_resilience/data/seed/hyperscaler_deals.csv` | Built, tested, **no live effect on its own tickers under `hard_switch`** — see TL;DR 2 |
| Combination | `grid_resilience/data/dc_demand_combined.py` (`combine_dc_demand_layers`, `apply_layer_precedence`) | Built, tested — per-date precedence: hyperscaler > PJM generation-queue > ERCOT |
| Wiring | `grid_resilience/main.py` — `--regulated-signal dc-multi` | Built, live-verified end-to-end |
| Short-sample harness | `grid_resilience/analysis/short_sample_report.py` (`window_sensitivity_report`, `sample_size_caveat`) | Built, tested, used below |

### Layer 1 (ERCOT) — a real null result, not a minor caveat

ERCOT's public "Large Load Project Distribution - TSP" chart — the source Layer 1 was designed around — prints project **count** on its bars in every deck checked (verified across 3 independent decks during data collection). The seed loader and `compute_ercot_signal` are built and unit-tested, and the seed CSV (`grid_resilience/data/seed/ercot_large_load_monthly.csv`) does contain real system-wide MW rows — but every row has `scope=system_wide` and an empty `tsp` field, so there is no ticker-level MW to attach to AEP or CenterPoint (CNP). Confirmed live: `dc-multi`'s coverage log line names zero ERCOT tickers. This is load-bearing infrastructure with a dead data source, not a rounding error in the ticker count.

### Layer 3 (hyperscaler deals) — real data, real corrections, and no live effect on its own tickers

6 deal rows, 3 tickers (CEG, TLN, VST), spanning 2024-03-04 to 2026-01-09. Both rows the plan flagged `needs_verification` were resolved against primary sources (0 rows remain unverified in `hyperscaler_deals.csv` today) — one matched the brief's placeholder exactly (CEG/Meta, 1,121 MW), one didn't (TLN/Amazon: corrected 1,920 MW → 960 MW and the date by 3 days, verified via Talen's own PRNewswire release). While doing that work, a **third** row — already marked `confirmed_primary_source` before this plan started — was found wrong by roughly 10x (1,920 MW → 180 MW, verified against the actual FERC order and Talen's own 8-K) and corrected. A fourth, smaller fix (an `event_type` mislabel that would have misled a reader into thinking a dispute was "resolved" when it wasn't) was caught in post-review and also fixed. No NRG deal was found and verified, so the merchant sleeve is 3 tickers, not the 4 (CEG, TLN, VST, NRG) the design spec named.

Two things about this layer must be read together with the above, or the layer looks far more consequential than it is:

- **It is inert for its own tickers under `hard_switch`** (TL;DR 2). CEG/TLN/VST are merchant names; `build_factor` scores them from `beta_z`. Nothing in this plan changed that, and nothing in this plan tested an architecture where it isn't true. `revenue_mix` and `dual_track` would give CEG 20% and TLN 10% weight on the signal — and VST exactly 0%, because `pass_through=1.00` zeroes the `(1 - pt)` term in both. Neither was backtested here.
- **VST contributes nothing to the tested window.** Its single disclosed deal (Meta, 2,176 MW) is dated 2026-01-09, after `BACKTEST_END`. In the original implementation the layer returned 0.0 for pre-disclosure dates, so VST was scored 0.0 on all 96 rebalance dates and — because the layer z-scores within its own 3-member {CEG, TLN, VST} subpopulation — sat near the bottom of that range for most of the recent window (≈ -1.4σ on 16 of the last 22 rebalances). That is not a neutral placeholder; it was a systematic negative tilt on a ticker about which the layer knew nothing. Fixed post-review (see below): pre-disclosure now returns NaN, VST is no longer claimed by the layer at all inside the backtest window, and it falls through to the ICR fallback like any other uncovered ticker.

### Two real architecture bugs caught during integration

**Column collision (caught during the plan).** CEG and TLN are tagged `iso="PJM"` in `TICKER_NODE_MAP`, so they already had real coverage from the pre-existing generation-queue signal before this plan touched anything. Combining that with the new hyperscaler layer via a naive `pd.concat` silently triplicated column names for every one of the 25 universe tickers (not just CEG/TLN) rather than merging them, which crashed the pipeline. Fixed with `apply_layer_precedence()`: hyperscaler (named, disclosed deals) > PJM generation-queue proxy > ERCOT. A related dtype bug (`pd.NA` from a zero-spread z-score row silently upcasting a column to `object` and crashing a downstream boolean-mask assignment) was also found and fixed at the source (`_zscore_columns`).

**Precedence displacement (caught in the final whole-branch review).** That first precedence fix decided ownership per *column*: a layer claimed a ticker outright if it had a value anywhere in the column. Because `compute_hyperscaler_signal` returned 0.0 (not NaN) before a ticker's first disclosure, the hyperscaler layer claimed CEG/TLN for **all 96 rebalance dates back to 2018**, on the strength of deals not announced until 2024 — silently discarding the PJM generation-queue layer's real, dispersed data for those tickers across six years where PJM had information and the hyperscaler layer had none. Fixed: NaN now means "no information" and 0.0 is reserved for a genuinely-zero disclosed sum, and precedence is resolved per date, so the PJM proxy keeps CEG/TLN's pre-disclosure history.

---

## Results (live-verified 2026-08-26, post-review fixes applied)

All three configurations run today, same command shape, same universe, same date range (2018–2025):

```
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-multi
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal icr
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-queue
```

| Configuration | Real ticker coverage | Sharpe | Ann Ret | Ann Vol | Max DD | IC@21d (t-stat, N) |
|---|---|---|---|---|---|---|
| **`dc-multi`** (this plan) | 8 (AEP, D, EXC, FE, PPL, PEG, CEG, TLN) | -0.456 | -0.83% | 10.59% | **-24.61%** | 0.0057 (t=0.157, N=84) |
| `icr` (fallback only) | 0 real (all 25 tickers on ICR) | **-0.331** | **-0.30%** | 12.99% | -29.08% | **0.0417** (t=0.969, N=84) |
| `dc-queue` (immediate predecessor, pre-this-plan) | 8 (AEP, D, EXC, FE, PPL, PEG, CEG, TLN) | -0.559 | -1.64% | 10.08% | -30.01% | 0.0073 (t=0.200, N=84) |
| XLU benchmark | — | 0.177 | 6.35% | 13.22% | -17.28% | — |
| EW Universe | — | 0.156 | 7.22% | 20.63% | -42.03% | — |

No formal significance test was run comparing Sharpe/Ann-Ret across configurations (no bootstrap CI) — the deltas above should be read as point-estimate comparisons, not statistically established differences. All IC t-stats are well below conventional significance (~2); none of these numbers should be read as "the signal works," only as "which is least bad, on this window."

### `dc-multi` vs. `dc-queue` (the direct predecessor): a partial, mechanically-explainable win

Three of four metrics improve — Sharpe (-0.559 → -0.456), Ann Ret (-1.64% → -0.83%), Max DD (-30.01% → -24.61%) — and **IC@21d gets worse** (0.0073 → 0.0057). The earlier claim that `dc-multi` "strictly dominates `dc-queue` on every metric tested" is withdrawn: it does not.

**The mechanism, stated accurately.** Both configurations resolve the *same 8 tickers* with real data, and both feed that data into the factor through the same `icr_z` slot, which under `hard_switch` is only read by tickers with `pass_through < 0.5`. So the delta cannot be — and is not — "CEG and TLN now use the hyperscaler signal instead of the noisier generation-queue proxy." CEG (`pass_through=0.80`) and TLN (`0.90`) are merchant names whose own factor scores are computed from `beta_z` in **both** configurations; neither signal ever reaches them. What actually differs between the two runs is:

1. **The PJM layer's z-score subpopulation shrinks from 8 tickers to 6.** In `dc-queue`, the generation-queue signal is z-scored across all 8 PJM-covered tickers including CEG and TLN. In `dc-multi`, CEG and TLN are claimed by the hyperscaler layer on their post-disclosure dates, so on those dates the PJM layer scores only the 6 regulated names against each other. Removing two members changes every remaining PJM ticker's z-score — and those 6 are precisely the `pass_through < 0.5` names whose scores the factor *does* read.
2. **A renormalization side-effect on the shared `icr_z` cross-section.** The merchant sleeve's values still enter the pooled `icr_z` z-score even though the merchant tickers themselves never read it, so the mere presence (and scale) of the hyperscaler layer shifts the mean/std against which the regulated names are measured. Directly measured during review: shifting the merchant sleeve moved AEP's factor score by -0.67 and PPL's by -0.99. Independently reproduced here: permuting the merchant sleeve's own DC-demand values among themselves — which leaves the pooled `icr_z` distribution intact — changes **nothing**, delta 0.000 for every ticker in the universe, merchant and regulated alike. (Merchant scores can still move when the sleeve is *shifted* rather than permuted, but only because `build_factor`'s closing cross-sectional renormalization propagates the regulated names' changes back to everyone — their pre-renormalization `arch_score` never reads the signal.)

Neither of those is the designed mechanism. (1) is a defensible-but-incidental consequence of the precedence rule; (2) is noise leakage that the design spec never intended and that a merchant-path integration would remove. **Do not read this delta as evidence that Layer 3 works.** The honest summary is that `dc-multi`'s portfolio-level edge over `dc-queue` comes from perturbing the regulated names' cross-section, not from adding information about the merchant names.

### `dc-multi` vs. `icr`: does not clear the "material, not sign-flip" bar — and now loses on two of four metrics

Sharpe is **worse** than the plain ICR fallback (-0.331 → -0.456) and Ann Ret is **worse** (-0.30% → -0.83%). Max DD improves by a real 4.47pp, the one substantive gain. IC@21d — the more direct measure of whether the factor score actually predicts forward returns — is **much worse** (0.0417 → 0.0057), and neither IC is close to statistically significant.

This is the same qualitative shape (a drawdown improvement without an IC improvement) as the single-source `dc-queue` result documented in `docs/compact_2026-08-13-dc-load-signal-results.md` — not a new problem, but not a resolved one either. It is also a sharper negative read than the pre-fix version of this document gave: the earlier text reported a small Sharpe gain and an Ann-Ret sign flip over `icr`, and both of those were artifacts of the defects listed at the end of this document.

**None of the three configurations beat XLU** (Sharpe 0.177, Ann Ret 6.35%). This is consistent with, not contradicted by, `docs/compact_2026-08-26-original-thesis-reexamination.md`'s broader conclusion that no signal formulation or portfolio construction tried to date beats XLU on this backtest window.

---

## Window-sensitivity and sample-size analysis (`short_sample_report.py`)

The pipeline does not currently decompose portfolio return by signal layer (Task 10 confirmed only a combined `pnl.csv` exists) — building that would need real plumbing beyond this task's scope, so two separate, honest things are reported instead.

### 1. Portfolio-level result, monthly-resampled to match the pipeline's own rebalance cadence

`output/pnl.csv`'s daily `strategy` log-returns for `dc-multi`, resampled to month-end (`REBALANCE_FREQ = "ME"`) and trimmed to the first date the strategy actually holds a nonzero position (2018-09-28), giving n=88 — exactly matching the pipeline's own `[ic] dates: 88` count:

| Variant | n_periods | Mean monthly log-return |
|---|---|---|
| full | 88 | -0.0748% |
| drop_first | 87 | -0.0665% |
| drop_last | 87 | -0.0636% |
| drop_first_and_last | 86 | -0.0550% |

`sample_size_caveat(88)` → empty string (88 is well above the 12-period trust floor). The sign doesn't flip under any trim — but it is now consistently **negative** under every trim (the pre-fix version of this table was consistently positive, on numbers the defects inflated). And this is a weaker result than it looks either way: the new layers built by this plan contribute real information for only a fraction of these 88 months (the hyperscaler layer starts 2024-03 and is inert for its own tickers regardless; ERCOT contributes nothing at any point), so most of this stability is inherited from the pre-existing PJM generation-queue/ICR-blend behaviour, not fresh evidence for Layers 1 or 3. Read it as "the combined portfolio-level result isn't a first/last-month artifact," not as "the new layers are validated."

### 2. Per-layer sample sizes — this is where the short-sample caveat actually bites

| Layer | n (independent observations) | `sample_size_caveat()` |
|---|---|---|
| Layer 3 — hyperscaler deal events | 6 events, 3 tickers, 2024-03-04 to 2026-01-09 (~22 months); only 5 events and 2 tickers fall inside the backtest window | *"Only 6 independent period(s) observed (below the 12-period bar treated as minimally trustworthy here) — treat this result as directional, not confirmed."* |
| Layer 2 — PJM LAS cross-check vintages | 1 (the 2025-09-16 LAS posting, the only entry in `LARGE_LOAD_SOURCE_URLS`) | Same caveat, at n=1. (The design spec itself already called this "not a standalone backtest" — confirmed here mechanically.) |
| Layer 1 — ERCOT real observations | 0 | Same caveat, at n=0 — the harness correctly flags total absence of data, consistent with the live coverage finding above. |

None of the three new layers, taken alone, clears even the harness's own 12-period minimum-trust floor. The only layer with a real, longer time series is the pre-existing PJM generation-queue proxy (unchanged by this plan, back to 2018).

---

## Honest bottom line

This plan delivered working infrastructure for 2 of 3 designed layers and real data-quality corrections to seed data and to a PJM zone-vocabulary bug that would otherwise have silently fed wrong numbers into the factor and its diagnostics. It did **not** deliver the ~11-ticker coverage the design spec estimated (8 achieved, none of them net-new versus what already existed), Layer 1 is dead in production, Layer 3 has no effect on the merchant tickers it was designed for under the architecture actually tested, and the combined signal does not clear its own pre-stated bar for a material edge — Sharpe stays negative and is worse than the plain ICR fallback, IC is much worse than ICR's, and nothing here beats XLU. The one durable gain is a ~4.5pp drawdown improvement over `icr` with no accompanying IC improvement, which is exactly the pattern this project has learned not to trust. Treat this the same way `docs/compact_2026-08-13-dc-load-signal-results.md` treated the original DC-load signal: real, shippable engineering work, not a strategy result to build a pitch around.

---

## Recommendation

- **Leave `REGULATED_SIGNAL` default as `dc_queue` in `config.py`.** The earlier recommendation — that a future session "consider promoting `dc-multi` to the default … since it strictly dominates `dc-queue`" — is **withdrawn**. On the corrected numbers `dc-multi` does not dominate `dc-queue` (IC@21d is worse) and loses to the plain `icr` fallback on both Sharpe and Ann Ret. More importantly, even where `dc-multi` does win, the win traces to a z-score-subpopulation change and a renormalization side-effect on the regulated names, not to the multi-source data doing what it was designed to do — which is not a basis for promoting anything to a default.
- **Fix Layer 3's integration before spending any more effort on Layer 3's data.** Under `--arch hard-switch` the merchant sleeve never *reads* the DC-demand signal — CEG/TLN/VST's own pre-renormalization `arch_score` comes from `beta_z` regardless of what the seed table contains. That is not the same as "the seed table could be perfect or empty and their factor scores would be identical": `build_factor`'s closing cross-sectional renormalization still lets a *shifted* seed table move their final scores indirectly (see "the mechanism, stated accurately" above) — it's a noise channel, not the designed signal path, but it is a real one. This needs a merchant-path integration in `build_factor()` (out of scope for the review fix wave) before Layer 3's own data quality can matter on purpose rather than by accident. `revenue_mix`/`dual_track` are a partial workaround for CEG (20%) and TLN (10%) only — VST gets 0% weight in both, because `pass_through=1.00`.
- **Do not invest further in Layer 1 (ERCOT) without a new data source.** The chart it was built around structurally cannot be coerced into per-ticker MW; a fix would require ERCOT publishing a different report, not a parsing change.
- **Layer 2's LAS cross-check should keep accumulating annual vintages** (currently 1) before being treated as anything beyond a current-state consistency check, per the design spec's own stated limitation.
- If a future session wants to actually validate Layer 3 in isolation, it needs (a) the merchant-path integration above and (b) per-layer return attribution wired into the backtest (not currently available) — and even then, the disclosed-deal history inside the backtest window is 5 events across 2 tickers.

---

## What the post-review fix wave changed

The final whole-branch code review raised five findings, all verified against the live pipeline. All five were fixed in this branch; three of them changed the numbers.

| # | Finding | Effect on results |
|---|---|---|
| 1 | Layer 3 is inert for CEG/TLN/VST under `hard_switch` (they are on the `beta_z` path); the original mechanism narrative in this document was false | Documentation only — mechanism section above rewritten, "Known limitation" added to the design spec |
| 2 | VST's only disclosed deal postdates `BACKTEST_END`; it was not "net-new coverage" but a constant zero that the layer's own z-score turned into a systematic negative tilt | Coverage headline corrected 9 → 8; resolved behaviourally by fix #3 |
| 3 | `compute_hyperscaler_signal` returned 0.0 (not NaN) before a ticker's first disclosure, so the hyperscaler layer claimed CEG/TLN for all 96 dates and displaced the PJM layer's real 2018–2024 data | **Changed the numbers.** NaN now means "no information"; precedence is resolved per date; uncovered tickers are reindexed to the full universe before `fill_with_icr` so they get their own ICR reading rather than a flat cross-sectional mean |
| 4 | `cross_check_dc_signal` matched LAS zone codes without normalizing PJM's short codes, dropping every `DAY` (Dayton) row | AEP's `mw_demand_2030` in `output/dc_signal_cross_check.csv`: 9,296.2 → 13,996.2 MW. Diagnostic only, not blended into the factor |
| 5 | The `dc_multi` branch silently substituted an empty DataFrame when the PJM fetch failed, where `dc_queue` warns and falls back to `icr` | No effect on this run (fetch succeeded); `dc_multi` now warns and falls back identically |

`dc-multi` before → after the fix wave (`icr` and `dc-queue` re-run the same day, both unchanged):

| Metric | Pre-fix (originally reported) | Post-fix (reported above) |
|---|---|---|
| Real ticker coverage | 9 | 8 |
| Sharpe | -0.302 | -0.456 |
| Ann Ret | 0.71% | -0.83% |
| Ann Vol | 10.91% | 10.59% |
| Max DD | -23.89% | -24.61% |
| IC@21d | 0.0104 (t=0.282) | 0.0057 (t=0.157) |
| Mean monthly log-return (n=88) | +0.0644% | -0.0748% |

The pre-fix numbers should be treated as void, not as an alternative scenario: they were produced by a configuration in which six years of real PJM data for two tickers had been replaced by a constant zero, and in which 17 uncovered tickers were collapsed onto a flat cross-sectional mean instead of their own ICR readings.

## Files changed

**Task 12 (original, documentation only):**

- `docs/compact_2026-08-26-dc-multi-source-signal-results.md` (this file)

**Post-review fix wave:**

- `grid_resilience/data/hyperscaler_deals.py` — NaN vs 0.0 semantics (finding 3)
- `grid_resilience/data/dc_demand_combined.py` — per-date precedence, NaN-preserving z-score, `ValueError` contract check (findings 3, 7)
- `grid_resilience/data/pjm_large_load_data.py` — LAS zone-code normalization, case-insensitive industry tag match (findings 4, 10)
- `grid_resilience/main.py` — dc_multi fetch-failure warning + ICR fallback, full-universe reindex before `fill_with_icr`, bare `assert` removed (findings 5, 7)
- `tests/data/test_hyperscaler_deals.py`, `tests/data/test_dc_demand_combined.py`, `tests/data/test_pjm_large_load_data.py` — updated/added coverage
- `docs/superpowers/specs/2026-08-26-dc-demand-exposure-signal-design.md` — Layer 3 "Known limitation" note (finding 1)
- `README.md` — `dc-multi` described as per-ticker/per-date selection rather than blending, corrected coverage count (finding 9)
- `docs/compact_2026-08-26-dc-multi-source-signal-results.md` — this revision (findings 1, 2, 6, 8)
