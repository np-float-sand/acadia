# Project Results — running index

**Purpose:** one place to see the current state of every strategy line in this
repo and what's been tried against each. Every research session should add or
update its own section here when it finishes a piece of work — keep entries
short (a few lines + links to the detailed doc), this file is an index, not
the record itself.

**How to update:** find your strategy's section (or add a new one), add a row
to its attempt ledger if you tested a signal, and update the "current status"
line if it changed. Don't rewrite other sections' history — append, don't
overwrite, unless you're the one correcting a stale status.

---

## Grid-equipment / data-center buildout theme (`grid_equipment_basket/`)

**Current adopted trade**: long a concentrated US + foreign grid/data-center
buildout basket (ETN, GEV, HUBB, VRT, NVT, PWR, MYRG, PRIM, FLNC + ABB/Schneider/
Prysmian/Hitachi), short the residential-solar + EV-charging + storage sleeve
directly (ENPH, SEDG, CHPT, RUN, BLNK, STEM) — beta-hedged, vol-targeted.
**Sharpe 2.28 (2023-26), 1.10 (full 2020-26).** Documented as a crowded,
regime-dependent style trade, not a discovered signal — see
`docs/handoff_2026-09-01-grid-buildout-long-short.md` for the full honesty
disclosure (crowding, 2019-22 negative Sharpe, squeeze risk).

Superseded base case: same long book, short the whole GRID ETF instead
(Sharpe 1.00 / 0.53) — weaker because ~50% of GRID's holdings are the names
you're long. Kept as the "cleaner-variant" comparison.

**Why the DER-short trade worked** (2026-09-02 analysis): not really a stock-picking
edge — a financing-quality/duration factor wearing a sector-pair-trade costume.
Long leg funded by programmatic hyperscaler/regulated-utility capex; short leg
funded by rate-sensitive consumer/growth-stage capital, and hit by adverse
state-level net-metering policy (CA NEM 3.0) at the same time rates rose. See
`docs/handoff_2026-09-01-grid-buildout-long-short.md` §1.3 and the 2026-09-02
session transcript reasoning (not yet its own doc — worth writing up if this
principle gets reused).

**New finding, different in kind — thematic-catalyst detection**: PJM's Load
Analysis Subcommittee publicly named "Data Centers (AEP, APS, Dominion)" as a
load-forecast adjustment category on **2022-11-29** — ~7 months before the
equity market's major re-rating (June-Sept 2023). A real, dated, public leading
indicator for *which theme is starting*, not a timing overlay on an
already-known trade. Caveat: ERCOT's equivalent task force (founded even
earlier, April 2022) was driven by **Bitcoin mining**, not AI data centers — a
naive detector needs to read *which* driver is named, not just that large-load
MW is growing. See `docs/thematic-catalyst-detection-2026-09-02.md`. **Not yet
built into an actual runbook** (source list / cadence / hit-to-trade workflow)
— flagged as open work.

### Signal-attempt ledger (long-running, cross-session; numbering has gotten
fuzzy across docs — reconcile if you can pin down the gaps)

| # | Signal | Verdict | Doc |
|---|---|---|---|
| — | Backlog-growth tilt, value-chain maker/contractor tilt, backlog-surprise event study, grid-demand-sensitivity factor, transmission rate-base compounders | FAILED (5 attempts, pre-2026-08-31) | `docs/backlog-surprise-and-proposals.md`, `docs/transmission-rate-base-negative.md` |
| — | Layer-2 congestion regime (6-rung ladder) | FAILED gate, adopted by PM override, then **reverted 2026-09-01** on OOS evidence | `docs/grid-regime-layer2-results.md` |
| — | Option A (relative DC-vs-rest-of-PJM congestion) | FAILED — prior-window "win" was the COVID demand-collapse coincidence | `docs/grid-regime-layer2-results.md` §3.2 |
| — | Option B (signal-tilted long/short-utilities pair) | FAILED — signal was backwards | `docs/grid-regime-layer2-results.md` §3.3 |
| — | Un-crowd-the-universe diversifier | FAILED — underperforms, deeper drawdown, 0.84 corr | `docs/triage_2026-08-31-differentiation-ideas.md` |
| #12 | FTR bid-implied forward-congestion signal | FAILED — engaged (avg exposure 0.91) but tracked vol-target baseline | `docs/ftr-bid-signal-results.md` |
| #13 | Backlog-coverage-alone tilt (RPO/backlog ÷ TTM revenue, YoY change) | FAILED — identical to equal-weight in the only window with data | `docs/backlog-coverage-signal-results.md` |
| — | Interconnection-queue velocity (PJM Table B-9, YoY Δ forward data-center MW by zone) | Inconclusive — only 2 annual vintages exist (2025, 2026); the one period it mattered, Sharpe went 1.16→1.05 | (session transcript 2026-09-02, not yet its own doc) |
| #15 | Utility capex-guidance revision signal ("Deliverable D") | FAILED — doesn't clear the pre-registered timing bar on any series/horizon | `docs/capex-guidance-signal-results.md` |
| ~#17 | Henry Hub forward-gas signal (proxy for forward power) | FAILED — true long-dated data isn't free; near-curve proxies fail the bar | `docs/hhub-forward-gas-signal-results.md` |

**Still open**: RT/DA spread as a cross-sectional axis (conceptual gap
unresolved — no company in the universe is an identifiable merchant
generator); ERCOT CRR clearing prices (public but needs its own scrape);
backlog pricing-power language (10-K text-mining, untried); quality/
balance-sheet weighting (scoped, paused — data gaps worse than coverage's,
which already failed); options-structured hedge (blocked on data); the
thematic-catalyst-detector runbook (above).

---

## Grid Resilience Strategy (`grid_resilience/`)

Long/short utility strategy trading conditional sensitivity to a Grid Stress
Index (GSI: LMP z-score + congestion fraction + reserve tightness + named
events). 18-ticker universe, 3 long / 5 short, monthly rebalance. See the
repo root `README.md` for the full mechanism.

**Consolidated signal evidence (June–Aug 2026):** `docs/grid-stress-signal-evidence.md`
— IC ≈ +0.14 (t ≈ 1.9) on the clean non-PJM universe, not significant; breaks on PJM
T&D names; whipsaws on VST/NRG; no config beats XLU on corrected data. Real-but-weak
signal, kept as a risk-managed L/S wrapper, not claimed as an edge.

---

## Electrification Strategy (`electrification_strategy/`)

Rules-based **long-only** basket of US electrification/grid-equipment/
data-center-power companies, with a valuation-extension de-risk overlay and an
optional hedge overlay (GLD/short-duration sleeve + conditional DLR/EQIX
short). Self-described as "honest enhanced thematic beta, not alpha — close to
the VOLT ETF on return, with risk control the ETF lacks." See
`docs/electrification-strategy-v1-results.md` and
`docs/handoff_2026-09-05-electrification-strategy.md`.

---

## Other lines (not yet summarized here — add a section when you next touch these)

- `backlog_factor/` — breadth-first backlog/RPO growth-surprise event study across
  ~109 industrials. FAILED (non-monotone, reversing bump). `docs/backlog-surprise-factor-results.md`.
- `transmission_rate_base/` — FERC transmission-rate-base growth rank, ~34 utilities.
  FAILED (rank-IC −0.02). `docs/transmission-rate-base-results.md`.
- `grid_demand_factor/` — return-sensitivity to a grid-demand nowcast. FAILED
  (near-orthogonal). `docs/triage_2026-08-31-proposals-bda.md`.
- `dc_demand_basket/` — design doc exists (`2026-08-28-dc-demand-thematic-basket-design.md`
  under its own folder); status unclear, needs an owning session to summarize.
- Local branches `backlog-surprise-factor`, `res1`, `strategy-triage-bda` exist
  unmerged as of 2026-09-06 — check whether they're superseded or still live
  before assuming `main` has everything.
