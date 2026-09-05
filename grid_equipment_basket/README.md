# Grid Equipment Suppliers Thematic Basket

Long-only basket of US-listed grid / data-center electrical-infrastructure suppliers —
a direct-revenue bet on the power buildout, as opposed to the utility-demand bets in
`grid_resilience/` and `dc_demand_basket/`.

Spec: ../docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md
Step 1 results (full numbers): ../docs/grid-equipment-basket-step1-results.md
Value-chain reframe spec: ../docs/superpowers/specs/2026-08-29-value-chain-reframe-design.md
Value-chain reframe results (both gates, all caveats): ../docs/grid-equipment-value-chain-results.md

## Composition (as of 2026-08-29)

9 names, all verified on the latest-10-K **business description only** — never on historical
returns. Per-name verification log: `candidate_research.md`.

| Ticker | Company | Bucket |
|---|---|---|
| ETN | Eaton | Electrical equipment — transformers, switchgear, DC power (Electrical Americas + Electrical Global, ~73% of revenue) |
| HUBB | Hubbell | Utility & electrical solutions — grid T&D components, transformers, meters (Utility Solutions ~63% of revenue) |
| GEV | GE Vernova | Grid equipment — HVDC, transformers, switchgear — plus power generation; spun off from GE, vendor price history from 2024-03-27 |
| VRT | Vertiv | Data-center power & thermal management — cleanest single-name expression of the theme |
| PWR | Quanta Services | Electric-power infrastructure EPC — reports remaining performance obligations / 12-month + total backlog |
| MYRG | MYR Group | Transmission & distribution EPC — small-cap (~$2-3B) |
| NVT | nVent Electric | Electrical connection & protection, enclosures, liquid cooling — data-center exposure |
| FLNC | Fluence Energy | Grid-scale battery storage — chronic underperformer, deliberately retained (passes the business test; keeping a known loser limits cherry-picking) |
| PRIM | Primoris Services | Utility / power-delivery infrastructure EPC — borderline on materiality, retained with reasoning in `candidate_research.md` |

Foreign-listed names (ABB, Siemens Energy, Prysmian, Nexans) are logged in
`candidate_research.md` but **excluded** from the backtestable basket — the only US-accessible
lines are thin OTC ADRs with poor adjusted-close quality.

Weighting: equal-weight across the names that have price history as of each rebalance date;
quarterly rebalance ~6 weeks (42 calendar days) after each calendar quarter-end
(~mid-Feb / mid-May / mid-Aug / mid-Nov); 25% single-name cap. Long-only, fully invested,
no leverage, no short leg. Total-return basis (yfinance auto-adjusted closes). Weights are set
on the rebalance date and drift with relative price performance until the next one — no daily
re-equalization. With 8-9 names the equal weight is 11-13%, so the 25% cap never binds at
Step 1; it exists for a future Step 2 tilt. A name with no price history yet (GEV before
2024-03-27) is simply absent from the weight vector until its first rebalance with data, at
which point the next rebalance includes it. Transaction costs are negligible at this cadence
and breadth and are not modeled; rebalance frequency is a documented low-sensitivity
parameter, not an optimized one.

## Benchmarks

| Benchmark | Role |
|---|---|
| **XLI** | Primary. Most of the universe is GICS Industrials, not Utilities — this is the relevant sector benchmark. The decision gate is defined against XLI. |
| **SPY** | Secondary — broad-market reference. |
| **XLU** | Secondary — continuity with the `grid_resilience` / `dc_demand_basket` work, which benchmarks against XLU. |
| **GRID** | Honesty check — First Trust NASDAQ Clean Edge Smart Grid Infrastructure, a rules-based third-party thematic ETF covering nearly this exact theme. |
| **PAVE** | Honesty check — Global X U.S. Infrastructure Development, a broader rules-based US-infrastructure-buildout thematic ETF. |

If a hand-picked basket cannot beat a rules-based thematic ETF that already exists, the edge
is "the theme", not the construction — and the writeup says so plainly.

## Step 1 / Step 2 result

**Recommended basket: equal-weight, no tilt.** Step 1 (the theme) cleared its decision gate;
Step 2 (the backlog tilt) did not beat equal-weight and is not adopted.

### Step 1 — equal-weight theme

Window **2023-01-01 -> 2026-07-31** (~3.56 years, 43 monthly observations). Equal-weight,
no tilt.

Basket **CAGR 59.8%** vs XLI 20.2%; basket **Sharpe 1.36** vs XLI 0.95. The basket beat all
five benchmarks on both CAGR and Sharpe, including GRID (CAGR 23.8%) and PAVE (CAGR 24.5%) —
but at ~2.2x the volatility of XLI (36.5% vs 16.5%) and a deeper drawdown (-41.3% vs -18.5%).
A 2020-2022 prior-regime panel (context only, not a gate) shows the same ordering: basket
CAGR 25.4% / Sharpe 0.69 vs XLI 7.6% / 0.26.

**Decision gate (spec §5): PASS.**

| Check | Values | Pass? |
|---|---|---|
| basket CAGR > XLI CAGR | 59.77% vs 20.20% | PASS |
| basket Sharpe > XLI Sharpe | 1.360 vs 0.954 | PASS |
| basket CAGR within 2 pp of max(GRID, PAVE) CAGR, or higher | 59.77% vs 24.47% | PASS |

Read alongside the mandatory caveats: the window is short (Sharpe standard error ~+/-0.5, so
the gap is directional, not significant), GEV covers only the back ~60% of it, and the
universe was chosen in 2026 knowing which names won. Bias mitigations: inclusion on business
description only, FLNC (a known underperformer) kept in, equal-weight removes weight
cherry-picking, and the GRID/PAVE comparison rules out only the weakest version of the "this
is just the theme" reading — a hand-picked 2026-vintage universe can still out-run a
rules-based one precisely because the winners were already known, so that check does not
neutralise the hindsight bias. Full numbers and caveats:
`../docs/grid-equipment-basket-step1-results.md`.

### Step 2 — backlog-growth weight tilt (evaluated, NOT adopted)

The gate passed, so the Step 2 tilt was built and re-validated (spec §6). The tilt starts
from the Step 1 equal weights, ranks names by trailing-YoY growth in their own
point-in-time-disclosed order backlog / RPO, multiplies the **top-half-ranked names by 1.25x
and the bottom half by 0.75x** (median name and any name with no backlog signal left at
1.0x), then renormalizes and re-applies the 25% cap. The 1.25 / 0.75 constants are fixed in
`config.py`, not fitted. Run it with `--tilt backlog`.

Same 2023-01-01 -> 2026-07-31 window:

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|---|
| BASKET — equal-weight | 59.77% | 36.48% | 1.360 | 1.774 | -41.35% |
| BASKET — backlog tilt | 59.16% | 35.98% | 1.363 | 1.784 | -43.01% |

**§6 re-validation: the tilt must improve BOTH Sharpe and CAGR vs equal-weight. It does not**
— Sharpe is +0.003 (inside the ±0.5 sampling noise) but **CAGR is -0.61 pp lower** and the
max drawdown is 1.7 pp deeper. **Equal-weight remains the recommended basket.** The tilt over-
weighted FLNC (the basket's designated chronic underperformer) through late 2024 / early 2025
on its fast backlog growth and under-weighted MYRG throughout; disclosed-backlog-growth rank
did not track forward return over this window. HUBB and NVT are never tilted (annual-only
backlog disclosure -> NaN signal), and VRT is left untilted at the 2025 rebalances by a
lookback-span guard added for its gappy disclosure (9 rows present, 6 of the 15 quarters
across 2023-26 missing). A companion max-staleness guard (signal -> NaN when the latest
disclosed quarter is more than 200 days before the rebalance) was also added; it changes no
rebalance on this window — wherever VRT's latest row is that stale the span guard already
NaNs it — but hardens the signal against a name that stops disclosing. Full Step 2
table, per-rebalance weight moves, and the keep-or-adopt reasoning:
`../docs/grid-equipment-basket-step1-results.md`.

The Step 2 comparison inherits every Step 1 caveat — same ~43-month, survivorship-biased
window, same ±0.5 Sharpe standard error, same GEV coverage gap.

## Value-chain reframe (2026-08-29)

A third construction option and an intra-theme hedge, from
`../docs/superpowers/specs/2026-08-29-value-chain-reframe-design.md`. Does **not** change the
Step 1 equal-weight basket or the Step 2 tilt path. Thesis: in a capex boom the excess return
accrues to whoever holds the supply-constrained bottleneck **and can price it** — the equipment
**makers** — not to the **contractors / assemblers** who compete on price for volume. The active
bet overweights makers and underweights (long-only) or shorts (market-neutral) the price-takers,
with a margin-change + backlog-coverage-change signal setting within-bucket amounts.

**Frozen buckets** (from 10-K business descriptions, set in `config.py` before any backtest,
never revised from results):

| Makers (pricing power) | Contractors / assemblers (price-takers) |
|---|---|
| ETN, HUBB, GEV, VRT, NVT | PWR, MYRG, PRIM, FLNC |

FLNC is the one borderline call — it sells a manufactured product but assembles third-party
cells, bids competitively, and runs chronic negative gross margin (no pricing power), so the
thesis places it with the price-takers. This 5/4 split lands close to a hand-split the handoff
already flagged as *probably hindsight*; the overlap is real and disclosed. Mitigations: the
split rule is business-model-based and frozen pre-backtest; the prior-regime panel and the
drop-VRT/GEV pass test whether it is merely "we picked the winners"; and PWR (a large winner)
sits in the *short* leg, so shorting it cost money.

### Two constructions

- `--construction value-chain-tilt` — long-only. Step 1 equal weights × maker ×1.25 /
  contractor ×0.75 (base), × within-bucket ×1.10 / ×0.90 by the composite signal rank,
  renormalized, 25% cap re-applied. Constants in `config.py` (`VC_BASE_MAKER`,
  `VC_BASE_CONTRACTOR`, `VC_WITHIN_TOP`, `VC_WITHIN_BOTTOM`), documented, not fitted.
- `--construction pair` — market-neutral, 100% gross long makers / 100% gross short
  contractors, reported standalone, as a 30% overlay on the equal-weight basket, and
  risk-matched to the conditional-QQQ-short comparator.

Both options print the same combined report (`backtest.value_chain_report` /
`value_chain_table`); the gate lines are the operative difference. `--drop-winners` drops
VRT and GEV from the makers bucket for the robustness pass (value-chain constructions only).

### Gate outcomes (primary window 2023-01-01 -> 2026-07-31)

**GATE 1 (long-only tilt, spec §7.4): PASS — but weak.** Tilt Sharpe **1.441** vs equal-weight
**1.360**; tilt CAGR **64.13%** vs **59.77%**. Both legs clear, so by the pre-registered rule the
tilt clears its gate and is available behind the flag. Health warning: the gaps are inside the
±0.5 Sharpe standard error; the pass is reproduced with the fundamental signal switched off (it
is essentially the static maker/contractor bucket split — plateau: signal-off 1.434 vs shipped
1.441); it does **not** survive `--drop-winners` (tilt CAGR 41.53% < equal-weight 42.26%); and
the 2020–2022 prior-regime panel has the tilt losing on both legs (Sharpe 0.678 vs 0.690, CAGR
24.51% vs 25.42%). **Equal-weight remains the headline recommended basket;** the tilt is a
documented alternative, not demonstrated skill.

**GATE 2 (the pair as a hedge, spec §7.4): FAIL — use the conditional QQQ short.** The pair 30%
overlay made the spec §7.3 drawdown episode *deeper* (episodeDD **-43.08%** vs **-41.35%**
un-hedged); the conditional QQQ short cut it (**-37.34%**) at a carry cost of **-0.53%/yr**. Gate 2
needs both protection and carry; the carry leg passes (pair standalone carry 15.74% > conditional
short -0.53%) but the protection leg fails. Per spec §7.4 the recommendation on Gate-2 failure is
"use the conditional QQQ short" (or "size down, no hedge"). **Adopt the conditional QQQ short**
as the intra-theme drawdown hedge. The pair has positive standalone carry and *raised* return as
a 30% overlay (CAGR 69.36% / Sharpe 1.508) — interesting as a return sleeve, but it is not a hedge.
(The risk-matched overlay row now reports `k = NaN` on the primary window: the conditional short
lowers portfolio vol below the un-hedged basket and the pair is vol-additive, so no non-negative
pair weight can match it — see the results doc.)

### New caveats (value-chain reframe — in addition to every Step 1 / Step 2 caveat above)

- **The §4.1 gross-margin signal is live for only PRIM in the primary window** (14/14 rebalances —
  PRIM tags all four discrete quarters). The other eight names are NaN'd by the 300–430-day
  trailing-5-quarter span guard: ETN / GEV never tag a discrete December quarter; NVT / VRT / MYRG /
  PWR tagged discrete Dec quarters through ~2021 then stopped; FLNC skips the September quarter;
  HUBB files none of these concepts as quarterly XBRL. A rebuild that derives Q4 = FY − 9-month
  YTD (spec §9, root `CLAUDE.md`) is the open follow-up. Either way the within-bucket signal is a
  small perturbation on the static bucket split (plateau: signal-off Sharpe 1.434 vs shipped 1.441).
- **Backlog-coverage (§4.2) is live for 8 of 9 names** (all but HUBB) from the 2024-05 rebalance
  (FLNC from 2024-02, GEV from 2025-08). The final review fixed a revenue-tag bug in `margin_data`
  (first-nonempty stopped at the stale legacy `Revenues` tag for MYRG / PRIM / PWR / VRT and never
  read the modern ASC-606 tag); the fix unions the two tags, lifting coverage from 4/9 to 8/9
  names. HUBB alone takes the base bucket multiplier at every rebalance.
- **The 5/4 bucket split overlaps the handoff's hindsight-flagged hand-split.** The `--drop-winners`
  failure shows much of Gate 1's pass rides on VRT and GEV being in the overweight bucket.
- **Gross-margin dilution** for the diversified names (ETN, GEV, PRIM) matters only where §4.1 is
  live (PRIM); company-wide and grid-segment operating-margin variants are logged as next steps
  (spec §9, root `CLAUDE.md`), not built here.
- **Short leg is not costless.** Pair results are gross. A borrow cost of **~16.27 pp/yr** on the
  short leg would erase the pair's carry edge over the conditional short — concentrated in MYRG
  (thin, ~$20–80M/day) and FLNC (hard/expensive to borrow).
- **`--prior-regime` for value-chain constructions:** Gate 2 is pinned to a hard-coded 2024-H2
  drawdown episode, which has no data in a 2020–2022 run. The final review replaced the crash with
  a guard — the panel now runs through the real CLI / `value_chain_report` path and reports
  `GATE 2: n/a (window has no 2024-H2 drawdown episode)`; Gate 1 and the standalone rows still
  compute.

Full tables (primary / prior-regime / drop-winners), both gate evaluations, the borrow-cost
sensitivity, the parameter plateau, and the complete caveat list:
`../docs/grid-equipment-value-chain-results.md`.

## Return convention

All maths here uses **SIMPLE returns** (`pct_change`, `(1+r).prod()`), unlike `grid_resilience`
which uses log returns. A log-based recompute will not tie out exactly — that is expected.

## Run

```
python -m grid_equipment_basket [--start ...] [--end ...] [--prior-regime]
    [--construction {equal-weight,backlog-tilt,value-chain-tilt,pair}]
    [--drop-winners]        # value-chain constructions only: drop VRT, GEV from makers
    [--overlay]             # also print the layer-1 risk overlay (trend gate + vol target) report
    [--overlay-l2]          # also run the layer-2 grid-congestion regime ladder (both windows)
    [--capex-guidance]      # also run the utility capex-guidance revision signal report (Deliverable D)
    [--output DIR] [--no-plot]
```

Examples:

```
python -m grid_equipment_basket --start 2023-01-01                 # primary window, equal-weight (recommended basket)
python -m grid_equipment_basket --prior-regime                     # 2020-2022 panel
python -m grid_equipment_basket --construction backlog-tilt        # Step 2 backlog tilt (built + evaluated; NOT the recommended basket, see below)
python -m grid_equipment_basket --construction value-chain-tilt    # value-chain reframe: makers-vs-contractors tilt + pair/overlay report
python -m grid_equipment_basket --construction pair --drop-winners # same value-chain report, VRT/GEV dropped from the makers bucket
```

`--construction` defaults to `equal-weight`. `--tilt backlog` is retained as a deprecated
alias for `--construction backlog-tilt` (the `--tilt {none,backlog}` flag still parses).
`--drop-winners` only affects the value-chain constructions (`value-chain-tilt`, `pair`).
`--prior-regime` runs the 2020-2022 panel for the value-chain constructions too; Gate 2 is
reported as `n/a` there (its drawdown episode is pinned to a 2024-H2 window with no data in
2020-2022), while Gate 1 and the standalone rows still compute.

Outputs (to `--output`, default `./output_grid_equipment`):

- `equal-weight` / `backlog-tilt`: `metrics.csv`, `basket_returns.csv`, `performance.png`
  (`performance.png` skipped with `--no-plot`)
- `value-chain-tilt` / `pair`: `value_chain_metrics.csv`, `value_chain_gates.json`
- `--overlay-l2`: `regime_metrics.csv` (baselines + per-rung metrics/gate/verdict, both windows),
  `regime_timeline.csv` (per-rung daily exposure multiplier)
- `--capex-guidance`: `capex_guidance_timing.csv` (per-series/horizon rank-IC, control-t, pass flag),
  `capex_guidance_gates.json` (universe/feasibility + de-risk & scaler gate verdicts; when the
  panel is not yet testable, just the not-testable dict)

## Layer-2 grid-congestion regime signal (2026-08-31 — OFF by default, 2026-09-01)

`grid_regime.py` + the `overlay.py` seam (`regime_exposure`, `apply_overlay_l2`,
`exposure_series_l2`; layer-1 functions untouched). Replaces layer-1's *price* trend gate with a
*physical* read — zonal transmission-congestion $ in the DC-heavy PJM zones (DOM, AEP, COMED, PPL),
trailing-3y z-scored, reduced to a monthly basket-exposure multiplier — so
`exposure = regime_multiplier × vol_target_scalar`. Data is the cached `grid_resilience` PJM zone
LMP + zonal load (2018-01→2025-12); **zero network I/O**.

**`config.REGIME_ENABLED = False`.** Adopted as the recommended overlay by PM decision on
2026-08-31, then **reverted 2026-09-01**: the Jan–Aug 2026 out-of-sample update showed it does not
beat a plain price-gate + vol-target OOS (Sharpe 1.03 vs 1.16), and correlation diagnostics found
the overlay's apparent edge was the vol-target de-lever, not congestion-specific information — see
`docs/handoff_2026-09-01-grid-buildout-long-short.md` §3.4/§3.7/§7.2. The code and shipped config
(`grid_regime.shipped_config()` = ladder rung 1) are kept for the negative result; to re-enable:

```python
from grid_equipment_basket import grid_regime, overlay
mult = grid_regime.live_multiplier(end="2025-12-31")            # discrete, PJM-4, congestion-only
overlaid = overlay.apply_overlay_l2(basket_returns, mult)       # x vol target, cap 1.5
```

**Evidence / caveat.** A pre-registered ladder of 6 variants (`config.REGIME_LADDER`; rung 7 RT/DA
deferred) was run with a frozen gate: beat layer-1-only on **Sharpe AND Calmar** on **both** the
primary (2023-01→2025-12) and prior (2020-01→2022-12) windows, on a plateau. **No rung strictly
passed** — every rung beats layer-1-only on Sharpe but not primary-window Calmar (2.16 vs 2.32): it
keeps more upside and takes a ~5pp deeper DeepSeek drawdown (−23% vs −18%), because PJM congestion
was genuinely high in early 2025 and the signal stayed levered in. It *does* clear the prior window
(rung 1 Sharpe 0.61 vs layer-1's 0.20) — the two signals are complementary, not substitutes. Rung 1
was **adopted as the recommended overlay by PM decision** on the generalisation + physical-grid
differentiator, accepting the deeper drawdown; layer 1 stays available via `--overlay`. Full
write-up incl. the adoption rationale: `docs/grid-regime-layer2-results.md`.

Live yfinance fetch was validated during implementation (cold == warm == uncached, 124/124
rows on the price-fetcher check). A fresh clone fetches from yfinance on first run; subsequent
runs read the warm parquet cache and reproduce the metrics exactly. The value-chain
constructions additionally fetch SEC XBRL quarterly fundamentals (`margin_data.fetch_fundamentals`
— revenue unioned across `Revenues` + `RevenueFromContractWithCustomerExcludingAssessedTax`,
deduped per quarter with the earliest filing kept), cached to
`data/cache/fundamentals_<TICKER>.parquet` (an empty parquet is written as a sentinel for a
genuinely factless name so it is not re-fetched every run). This fetch was re-run live on
2026-08-29 after the revenue-tag-union fix; the realized per-name signal coverage is recorded
in the results doc (backlog-coverage now live for 8 of 9 names).

## FTR bid-implied forward-congestion signal (2026-09-02 — FAILED)

`ftr_signal.py`: an alternative to the layer-2 congestion regime, using PJM FTR
(Financial Transmission Rights) *bid* data as a genuinely forward-looking proxy for
congestion expectations (an FTR is itself a multi-year-forward instrument, unlike a
trailing LMP z-score). PJM's actual FTR *clearing* prices require a PJM membership
login; the bid data (`ftr_bids_mnt`) is public but is not filterable by sink location
server-side, so each auction month (150k-600k rows) is downloaded in full and filtered
client-side to the four DC-heavy zones' bare zone-aggregate sinks (`AEP`, `COMED`,
`DOM`/`DOMINION HUB`, `PPL`). MW-weighted mean bid price per zone, equal-weighted,
trailing-z-scored, applied with the feed's own stated 4-month publication lag, feeds
`grid_regime.regime_multiplier`/`overlay.apply_overlay_l2` unchanged (same seam as
layer-2).

**Result: FAILED the same pre-registered gate as layer-2** (`ftr_signal_report()`) —
the multiplier engaged substantially (average exposure 0.91, real time at both 0.6 and
1.25) but tracked layer-1-only almost exactly on both windows; G1 and G3 fail, G2 passes
only marginally. Full write-up: `docs/ftr-bid-signal-results.md`. A live backfill of the
96 months needed (2018-2025) took 106 minutes — PJM's rate limiter throttles hard under
sustained sequential pagination, much heavier than a single-month probe suggests.

## Backlog-coverage-alone signal (2026-09-02 — FAILED)

`basket.coverage_tilt_targets()` + `backtest.coverage_report()`: tests the coverage
half of the value-chain reframe's blended margin+coverage composite in isolation --
tilt by year-over-year change in RPO/backlog coverage (÷ TTM revenue) alone, top/bottom
half at 1.25x/0.75x. Data: `load_backlog_csv()` (fresh for all 9 names) ÷
`margin_data.ttm_revenue()` (8 of 9 -- HUBB has no usable XBRL revenue tag, a known
pre-existing limitation).

**Result: FAILED Gate 1** (tilt must beat equal-weight on Sharpe AND CAGR, both
windows) -- primary window CAGR edges up but Sharpe doesn't and drawdown worsens;
prior window the tilt is *identical* to equal-weight (no name has enough backlog
history that far back for the year-ago comparison to engage). Full write-up:
`docs/backlog-coverage-signal-results.md`.

## Phase 2 — cross-sectional factor (NOT built here)

Only if verification ever lands 8+ names with clean, comparable, segment-level backlog and
multiple years of quarterly history does a cross-sectional backlog-growth-surprise factor
become defensible. That needs its own spec — do **not** reintroduce `build_factor()` /
z-scoring / IC machinery into this basket. This is scoped as a thematic basket, matching
`dc_demand_basket`.

A purely **descriptive** check of as-of backlog growth against each name's own subsequent
63-/126-day return (`backlog_forward_check.py`) was run once to inform that question. On this
tiny sample (4–9 as-of points per name; HUBB / NVT / VRT have too few to include) the signs
are mixed and lean the wrong way — 4 of 6 names show a negative association — so it gives **no
encouragement** to open a Phase-2 spec now. Descriptive only: no IC metric, no z-scoring, no
significance claimed. Full table and read: `../docs/grid-equipment-basket-step1-results.md`.

## Capex-Guidance Revision Signal (Deliverable D) — 2026-09-04

Handoff: `docs/handoff_2026-09-03-transmission-project-filings.md` §6. Design spec:
`docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md`. Results:
`docs/capex-guidance-signal-results.md`.

An aggregate timing signal (not a cross-sectional factor — it never re-weights the 9 basket
names) built from a hand/web-assembled panel of 15 large US electric utilities' forward
multi-year capex-guidance revisions (`grid_resilience/data/seed/utility_capex_guidance.csv`,
loaded by `grid_resilience/data/utility_capex_guidance.py`). Tests the revision-flow ($ and
size-weighted %, both the whole-panel total and the data-center-attributed portion, the latter
with a point-in-time EM-style fill for utilities that only qualitatively mention data centers)
as (1) a monthly rank-IC timing signal against forward basket and long/short-spread returns,
controlling for Δ10y yield, SMH, and (separately) the big-four hyperscaler capex series, and
(2) a one-directional de-risk multiplier / two-sided scaler for the basket's exposure, gated
against layer-1 (price trend gate + vol target) with the same `grid_regime.gate_check`/
`final_verdict` machinery the FTR-bid and grid-congestion signals use.

Run with `python -m grid_equipment_basket --capex-guidance` (writes
`capex_guidance_timing.csv`, `capex_guidance_gates.json` to the output dir).

**RESULT: FAILED — both legs.** Live run 2026-09-05 on the full 14/15-utility panel (NEE excluded: 0 in-window revisions + FPL on record that its plan "was not driven by data centers"). The timing bar (spec §5) fails every cell: the two testable whole-panel revision-flow series (`all_usd`, `all_pct`) show a *negative* rank-IC at all three horizons (−0.12 to −0.40, no |t|≥ 2), and the four data-center-attributed series are untestable because every stated/imputed DC-$ vintage is 2026-dated so the z-score never warms. The de-risk / scaler gate (spec §6) fails G1 across the whole parameter plateau (de-risk +0.03 Sharpe / −0.21 Calmar vs layer-1; scaler −0.06 / −0.55). Attempt #15 against this theme; first test of the *commitment*-data class and it lands like the prior forecast/flow negatives. Nothing wired live. Full write-up, tables and honesty caveats: `docs/capex-guidance-signal-results.md`.
