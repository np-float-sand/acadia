# Screened Electrification — SOW Execution Plan (Design)

**Date:** 2026-09-15
**Status:** approved for implementation planning

## 1. Purpose

`docs/scope-of-work-draft.md` is a real draft engagement (Consultant: Sandhya Persad; Client:
Addenda Capital; June 2026), written around `grid_resilience/` — an 18-utility long/short book
trading a "stress beta" to a Grid Stress Index (GSI). Since then, a different strategy has
emerged as the actual ship candidate: **Screened Electrification Equipment**
(`electrification_strategy/`, `docs/electrification-strategy-proposal-v5.md`) — a 27-name
equipment-supplier screen + vol-target + macro sleeve, framed in RESEARCH-LOG as "thematic beta,
not alpha" after ~29 signal-search attempts failed an honest regime-split test.

This spec reconciles the two: it produces (a) an updated SOW reflecting what will actually be
delivered, and (b) an execution plan that does the Phase 1–3 work for real, including one
genuinely untried piece of research (stress beta re-tested on the supplier universe) rather than
either blindly following the old SOW language or silently dropping it.

**Guiding principle for all of Phase 1:** any free or low-cost power/energy-market data source
(grid congestion costs, LMP volatility, forward power prices, reserve-margin data, etc.) is fair
game to test against the equity signal — but it only ships if it demonstrably improves the gated
result, never for narrative flavor alone. This is the same discipline §3.4 applies to stress beta,
applied project-wide, and it's why §3.2 (forward power) stays descoped rather than added on
spec — consistent with RESEARCH-LOG's ~29 tested-and-mostly-rejected signal candidates.

## 2. Deliverables

1. **`docs/scope-of-work-v2-screened-electrification.md`** (+ HTML/PDF via
   `tools/md2html.py`, matching this repo's convention for versioned proposal docs). Keeps the
   commercial/legal skeleton of v1 unchanged (parties, hourly rate, monthly cap, data-cost
   reimbursement, IP assignment, scope limitations, term, signatures). Rewrites Background and
   the Phase 1–3 bodies around Screened Electrification, including honest, conditional language
   for the stress-beta item (see §3.4) — the client sees the real gate, not a guaranteed feature.
2. **Implementation plan** (via the `writing-plans` skill, next step after this spec) that
   sequences the actual engineering work below, with the stress-beta branch built in as a gate.
3. This design spec, for the record.

## 3. Phase 1 — Data Integration & Signal Development

### 3.1 Data pipeline consolidation + short-interest integration
`electrification_strategy/` already has a working, cached, reproducible pipeline (equity prices
via `grid_equipment_basket.data.prices`, FRED via `fred.py`, parquet caching under
`electrification_strategy/data/cache/`). This task formalizes it as the "established and
validated" Phase 1 deliverable: confirm a clean-environment rebuild works end to end (fresh
cache), and close a real gap — **short interest has zero coverage anywhere in this repo**, in
either `grid_resilience/` or `electrification_strategy/`.

Short interest is added as a **cost/liquidity diagnostic**, not a signal input: it directly
closes the limitation proposal v5 already flags in its Method notes ("Trading costs at scale are
still to be estimated"), by estimating borrow cost / crowding risk for Option B's short leg
(XLI-proxy overlay + the optional DLR/EQIX insurance short). New module:
`electrification_strategy/short_interest.py`, cached like the rest of the pipeline.

### 3.2 Forward power price data — descoped for now
Real prior feasibility work exists (STEO `ELWHU_*` forward series — scoped, not run; ERCOT RT/DA
spread — feasible) but the shipped strategy doesn't need a forward-power input, and the GSI reuse
in §3.4 is realized/spot data, not forward-looking. Per direction: **skip**. Document the
decision and point at `docs/steo-archive-feasibility-results.md` and
`docs/rt-da-spread-feasibility-results.md` as available future work. Reopen only if §3.4's
stress-beta construction turns out to need a forward-looking stress proxy (not expected).

### 3.3 Client risk data — adapter against a placeholder schema
No Addenda risk-model file exists in the repo yet. Build `electrification_strategy/risk_exposures.py`:
ingests a CSV shaped `date, ticker, factor, exposure` (long format — one row per date/ticker/factor),
aligned to the 27-name universe + benchmarks on the strategy's existing 42-day rebalance lag
convention. Ship with a synthetic fixture file and tests proving the adapter is correct
independent of real data, plus a short schema-contract note (in the technical reference, §5) of
exactly what's needed from Addenda so the real file is a drop-in swap.

### 3.4 Stress-beta gate — the enhanced-signal deliverable

This is the literal "build it properly" instruction, done honestly with a pre-registered
pass/fail gate rather than open-ended research.

**Step A — reuse, don't rebuild, but resolve one real mismatch.**
`grid_resilience/signals/grid_stress_index.py` (GSI: LMP z-score 40% / congestion fraction 25% /
reserve tightness 20% / named-event flag 15%, per ISO) and
`grid_resilience/signals/conditional_beta.py` already exist, are tested, and were validated on a
*different* universe (18 utilities with real-time grid exposure, and separately the
merchant-generator panel in `research/narrative_mispricing/`). Nobody has run this against the
Screened Electrification supplier universe (ETN, HUBB, GEV, VRT, PWR, MYRG, NVT, FLNC, PRIM,
etc.) — genuinely untried, and economically distinct: suppliers may plausibly show a *positive*
stress beta (crisis creates emergency demand for their product) rather than the utilities' analog
(negative beta = resilient = long).

`gsi_for_ticker()` assumes a per-ticker home ISO (`ticker_iso_map`) — the right model for a
utility's service territory, but suppliers are national/global manufacturers with no such
mapping. Do **not** force a `ticker_iso_map` for this universe. Instead build one blended
**national GSI** — an equal-weighted average of `build_multi_iso_gsi()`'s per-ISO output across
all ISOs already covered — and apply it uniformly to all 27 tickers. This is a simplification
worth stating plainly in the results writeup, not hiding: it treats grid stress as a national
demand-for-equipment signal rather than a location-specific exposure, which is the economically
correct framing for manufacturers regardless of gate outcome.

**Real-world grounding for the congestion sub-signal:** Reuters (Maguire, 2026-07-20, "America's
power grid is choking on costly congestion") reports PJM transmission congestion cost market
participants $777.8M in June 2026 and a record ~$1B in May 2026 during a regional heat wave —
concentrated in Pennsylvania, Maryland, Northern Virginia and New Jersey, the data-center
epicenter — driven by 10+ year transmission permitting timelines. This corroborates CLAUDE.md's
2026-09-08 finding that PJM's `congestion` GSI component was never broken (0% NaN, real signal):
the congestion sub-signal Step B tests is measuring a real, currently worsening, economically
large phenomenon, not a data artifact. It's a citation for the Phase 2 findings report (§4.3),
not a new data source — PJM congestion is already inside the existing GSI.

**Regime dating — the grid was slack before ~2021.** Both the Reuters reporting above and this
project's own prior work point to a regime break around 2021, not a smooth trend: the ancillary/
scarcity-pricing probe (`docs/ancillary-scarcity-pricing-feasibility-results.md`) already found
**no** timing signal on 2018-2021 (\|t\| < 1.1) and left 2022-2025 untested as the more-promising
window; February 2021 (ERCOT Winter Storm Uri) is the commonly-cited inflection point after which
reserve margins tightened, generator retirements accelerated, and (later) data-center load
compounded it. Requiring a stress-beta effect to also hold during the pre-2021 slack-grid era
would be testing the hypothesis in a regime where it isn't expected to apply — a misspecified
gate, not rigor. So the primary test window is **regime-based, not calendar-AI-boom-based**:

**Step B — pre-registered gate** (this split is fixed now, before any test is run — not chosen
after looking at results):
1. Compute a blended national/cross-ISO GSI over the electrification backtest window
   (2017-06 → present).
2. Each month, rank the 27 suppliers by trailing 12-month rolling `stress_beta`
   (`conditional_beta.py`, unmodified). Form a top-tercile-minus-bottom-tercile spread portfolio.
3. **Gate 1 (statistical), primary window = tight-grid era (2021-01-01 → present):** pooled
   spread-return t-stat \|t\| ≥ 2 within this window. The pre-2021 slack-grid era is computed and
   reported for context (does the effect plausibly strengthen post-2021, consistent with the
   regime story, or is it flat throughout — which would undercut the regime explanation) but is
   **not required to show the same sign** — it's diagnostic, not part of the pass bar. The
   existing `SUBWINDOWS` split in `config.py` (2019-22 / 2023-26) is reported alongside for
   comparability with the rest of the strategy's backtest tables, but the tight-grid split is what
   the gate is judged on.
4. **Gate 2 (practical, only if Gate 1 passes):** wiring `stress_beta` in as a scoring tilt must
   not break the strategy's existing pre-registered winner criteria (`WINNER_MAXDD_MAX`,
   `WINNER_SUBWINDOW_SHARPE_MIN`, `WINNER_FEARED_PNL_MIN` in `config.py`).

**PASS (both gates):** wire `stress_beta` in as a documented scoring component of
`electrification_strategy`'s construction — this is Phase 1 item 4, delivered as designed.

**FAIL (either gate) — the expected base case given ~29 prior attempts, but not a foregone
conclusion since this exact universe was never tested:** do not force it. Instead, write up the
negative result (methodology, gate values, why) as a `docs/*-results.md` entry per project
convention, and reposition the deliverable: formalize proposal v5's three existing built-in
protections — the 15% GLD/IEF macro reserve, the valuation-extension de-risk ladder, and the
20%-target vol-targeting rule — as a "Stress Sensitivity & Grid-Event Response" section, each
rule explicitly mapped to a stress category from v5's own risk table, with the episode-drawdown
table (COVID / 2022 rate shock / DeepSeek / tariff shock / 2026 selloff) as the empirical
evidence. This satisfies the SOW's underlying intent — understand and manage sensitivity to
stress-correlated shocks — without misrepresenting a negative regression result as a working
signal.

**Time-box:** this step gets a hard stop once Gate 1's pooled t-stat and subwindow signs are
computed — no iterating on GSI weights, universe cuts, or lookback windows to chase significance.
One clean run, one verdict, logged either way.

## 4. Phase 2 — Backtest & Validation

1. **Create Strategy** — `electrification_strategy/backtest.py` already maps signal → weights
   (screen, vol-target, valuation overlay, hedge overlay). Extend it to reflect §3.4's outcome
   (either the new stress-beta tilt, or no construction change) and wire §3.3's risk-exposures
   adapter in as a reporting/diagnostic overlay (portfolio-level factor exposure report), not a
   hard constraint, since no real Addenda data exists yet to optimize against.
2. **Rolling backtest** — re-run `backtest.run_comparison` end to end with the above changes,
   confirm reproducibility from a clean cache, produce the standard output bundle
   (`metrics.csv`, `episode_drawdowns.csv`, `plateau.json`, `returns.csv`, `performance.png`).
3. **Written findings report** — a Phase-2-named deliverable distinct from the SOW itself:
   updates proposal v5 (→ v6) or a dedicated findings doc, folding in the stress-beta gate
   outcome, the short-interest cost diagnostic, risk-data integration status (adapter built,
   pending real file), and known limitations (trading costs at scale, single-cycle backtest,
   whichever of §3.4's two branches applies) plus the recommended configuration (Option A vs. B
   and parameters, as already selected in v5). Grounds the physical-need thesis with current
   market evidence (e.g. the Reuters PJM-congestion reporting cited in §3.4) rather than resting
   on the thesis alone.

## 5. Phase 3 — Handoff & Documentation

1. **Code documentation** — docstring pass across `electrification_strategy/` plus any new
   modules from Phase 1 (`short_interest.py`, `risk_exposures.py`, and the reused GSI modules if
   §3.4 passed); a `docs/technical-reference.md` covering each module's inputs/outputs/role.
2. **Operational runbook** — `docs/runbook.md`: environment setup, refreshing caches, running
   `python -m electrification_strategy`, rebalance cadence/lag interpretation, reading output
   files, common failure modes (yfinance rate limits, FRED API key, gridstatus auth/cache
   staleness if GSI is wired in, stale-cache troubleshooting).
3. **Handoff sessions** — logistics depend on Client scheduling; this plan produces the
   materials (technical reference, runbook, a walkthrough outline) that make the sessions
   possible, but can't schedule them autonomously.
4. **Final codebase delivery** — a `docs/deliverable-manifest.md` listing exactly what ships
   (`electrification_strategy/`, any reused `grid_resilience/signals/` modules, relevant `docs/*`,
   `tests/electrification_strategy/`) versus what stays internal (the ~29 FAILED `research/`
   probes, scratch `output_*` directories) — then a literal tagged `git tag` release once the
   Client confirms the manifest.

## 6. Budget note

SOW estimates: Phase 1 50–70h, Phase 2 40–60h, Phase 3 20–30h (110–160h total). Most of Phase 2–3
formalizes work that already exists; the only open-ended item is §3.4's stress-beta test, which
is deliberately time-boxed with a hard pre-registered gate so it can't silently consume the
Phase 1 budget the way earlier signal searches in this project's history did.

## 7. Assumptions

- Addenda's risk-model file schema is unknown; §3.3's placeholder schema (long-format
  `date, ticker, factor, exposure`) is a reasonable default but may need revision once the real
  file arrives — flagged explicitly in the runbook and technical reference, not hidden.
- "Reproducible research environment" (Phase 1 item 1) is interpreted as: a documented, scripted,
  cache-based pipeline runnable from a clean checkout — not a hosted/CI service.
