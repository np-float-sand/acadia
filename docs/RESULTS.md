# Project Results Ledger — grid / power-buildout equity strategies

**Purpose.** One running record of every strategy, signal, and construction attempt in this
project, with its pre-registered verdict and key numbers. Every working session appends here.
Detailed write-ups stay in their own `docs/*-results.md` files; this doc is the index + synthesis.

**Last updated:** 2026-09-06

---

## How to update this doc (for every future session)

1. When a probe/build reaches a verdict, add one row to the **Attempts ledger** table for its
   phase (chronological, newest last). Keep the row to: date · name · what it tested · verdict ·
   the single most important number · link to the full write-up.
2. If the verdict changes a standing conclusion (something got adopted, reverted, or the
   stop-rule moved), update **Standing conclusion** and **What is shipped** too.
3. Keep verdict language pre-registered and blunt: `PASS` / `FAIL` / `NOT ADOPTED` /
   `NOT YET TESTABLE` / `ADOPTED` / `REVERTED`. No spin — negative results are the main product
   of this project and are logged as first-class outcomes.
4. Attempt numbering across the buildout-theme search is historically fuzzy (sources say
   "~14", "#12", "#15", "~16", "~17" for overlapping things). Don't fight it — use dates as the
   spine and give an approximate running count.

---

## Standing conclusion (as of 2026-09-06)

**No signal-based timing edge has survived a pre-registered gate in the grid/power-buildout
theme after ~15–17 attempts spanning three independent data classes** — demand *forecasts*
(PJM Table B-9), physical *flow* (transmission congestion, FTR prices, LMP spreads), and firm
*commitments* (utility capex-guidance revisions). The commitment class was the last untried
angle and it failed like the others (this session, `capex-guidance-signal-results.md`).

The defensible ways to hold this theme, all **construction / risk-management, not signal**:

- **Equal-weight `grid_equipment_basket`** (9 US names) — cleared its decision gate vs XLI
  (`grid-equipment-basket-step1-results.md`). This is the recommended core.
- **+ Layer-1 risk overlay** (price trend gate + 20% vol target) — the one robust positive
  across every regime tested; the honest read is "hold less when the trend is down and vol is
  high," everything past that is tuned.
- **Concentrated, index-hedged long/short** (long the 9 + foreign ADRs / short the GRID ETF or
  a DER sleeve) — a strong trailing record (Sharpe ~2 in 2023–26 with beta-hedge + vol-target,
  ~1.2–1.35 plain) but pitch it as a **style / relative-value** trade with explicit crowding,
  regime, and short-squeeze risk — not as an edge.
- The **capex-cycle-pair classifier** (4-criterion "customer + 2 of 3" screen) mechanically
  reproduces the long/DER-short book without hindsight name-picking, but does not time or
  generalize it.

**Next-move options** (from `handoff_2026-09-01 §7`, still current): ship the discretionary
thematic position and keep the **county data-center permit capture (Deliverable E)** running
for a ~2028 re-test with real history; or move the signal search to a less-arbitraged theme.

---

## What is shipped / live

| Component | State | Notes |
|---|---|---|
| `grid_equipment_basket` equal-weight core | **recommended** | 9 names, quarterly rebal, 25% cap; gate PASS vs XLI |
| Layer-1 overlay (`--overlay`, `OVERLAY_*` in `config.py`) | **recommended overlay** | trend gate + vol target; robust positive, rest is tuned |
| Layer-2 congestion regime (`REGIME_ENABLED`) | **built, `False`** | strict gate not passed; reverted 2026-09-01 (no OOS edge over price-gate) |
| Backlog-growth tilt (`--tilt backlog`) | built, **not adopted** | didn't beat equal-weight on both Sharpe + CAGR |
| Value-chain tilt / pair (`--construction`) | built, **conditional-short only** | Gate 1 weak-PASS, Gate 2 FAIL |
| `--capex-guidance` (Deliverable D report) | built, **negative result** | signal not wired live anywhere |
| `grid_resilience` business-model-aware factor | built, off-by-default flags | `BUSINESS_MODEL_ARCH`, `USE_ICR`, `PEER_GROUP_CONSTRUCTION` all off pending validation |
| `electrification_strategy` v1 | built | best plain thematic Sharpe ~0.85; see below |

---

## Attempts ledger

### A. Grid-resilience (original regulated-utility long/short) — signal & construction

| Date | Attempt | What it tested | Verdict | Key result | Write-up |
|---|---|---|---|---|---|
| 2026-06/07 | Business-model-aware factor architecture | separate merchant vs regulated factor logic | BUILT | Sharpe 0.331 (pre-data-drift); off-by-default | memory `session-2026-06-analysis` |
| 2026-08-13 | DC load signal (PJM interconnection-queue level + momentum) | queue growth as a regulated-name demand signal | BUILT / MERGED | feature complete; **equity-price data drift discovered** — historical Sharpe 0.331/0.272 not to be trusted until re-validated | `compact_2026-08-13-dc-load-signal-results.md` |
| 2026-08-19 | Peer-group (basket-vs-basket) construction | split universe into merchant/mixed/regulated, build L/S within each to cancel sector beta | FAIL (hypothesis) | core hypothesis does not hold on backtest; `PEER_GROUP_CONSTRUCTION=False` | `compact_2026-08-19-peer-group-construction-results.md` |
| 2026-08-26 | Multi-source DC-demand exposure signal | blend PJM queue + ERCOT + hyperscaler deals into one exposure score | FAIL (bar) | coverage 8/~11 names; "material, not sign-flip" bar not cleared; behind `--regulated-signal dc-multi` | `compact_2026-08-26-dc-multi-source-signal-results.md` |
| 2026-08 | ICR (interest-coverage-ratio) factor component | low-ICR as a short-candidate signal during rate stress | BUILT, unvalidated | `USE_ICR=False`; yfinance history too short pre-2022 | CLAUDE.md "Future Work" |

### B. Grid-buildout / equipment-basket theme — the signal search

Approx. running count in brackets; historically imprecise (see note above).

| Date | Attempt | What it tested | Verdict | Key result | Write-up |
|---|---|---|---|---|---|
| 2026-08-29 | **Equal-weight thematic basket (Step 1)** | 9 US grid/DC-electrical names, equal-weight, vs XLI | **PASS** | CAGR 59.8% vs XLI 20.2%; Sharpe 1.36 vs 0.95 (short window, GEV covers back ~60%) | `grid-equipment-basket-step1-results.md` |
| 2026-08-29 | Backlog-growth weight tilt (Step 2) | tilt by trailing YoY disclosed-backlog growth | NOT ADOPTED | Sharpe +0.003 but CAGR −0.61 pp, deeper DD; over-weighted FLNC | `grid-equipment-basket-step1-results.md` |
| 2026-08-29 | Value-chain reframe (maker-vs-contractor tilt + intra-theme hedge) | pricing-power makers over price-taker contractors | Gate 1 weak-PASS / Gate 2 FAIL | use the conditional short only; built + merged | `grid-equipment-value-chain-results.md` |
| 2026-08-31 | Backlog-*surprise* event-study factor | RPO surprise vs trailing-4Q mean → cross-sectional CAR, 91 names / 1,778 events | FAIL | quintile CARs non-monotone; no Q5−Q1; stop before Phase 2 | `backlog-surprise-factor-results.md` |
| 2026-08-31 | Transmission rate-base compounders (proposal B) | rank ~34 utilities by FERC transmission-rate-base growth | FAIL (all 3 conditions) | rank-IC −0.018 (t −0.35); L/S Sharpe −0.61; no residual alpha. 4th cross-sectional attempt to fail | `transmission-rate-base-results.md` |
| 2026-08-31 | Big-four hyperscaler capex-deceleration de-risk trigger | MSFT/GOOGL/AMZN/META capex YoY roll-over → scale basket back | FAIL | slow annual number can't time a basket trading the live narrative (TDD) | git `3293733`; `triage_2026-08-31-differentiation-ideas.md` |
| 2026-08-31 | **Layer-1 risk overlay** (trend gate + vol target) | hold basket only above MA-100, scale to 20% vol | **ADOPTED** | the one robust positive across regimes; edge = converting the −42% DeepSeek DD to ~−9%; rest is tuned | `handoff_2026-08-29-grid-equipment-basket.md` |
| 2026-08-31 | Layer-2 grid-congestion regime signal | zonal PJM transmission-congestion $ + reserve tightness → monthly exposure multiplier | Gate NOT passed → ADOPTED (PM) → **REVERTED 2026-09-01** | beats layer-1 on Sharpe not primary Calmar (2.16 vs 2.32); OOS shows no edge over plain price-gate; `REGIME_ENABLED=False` | `grid-regime-layer2-results.md` |
| 2026-09-02 | FTR bid-implied forward-congestion signal [~#12] | MW-weighted buy-side FTR obligation bids at DC-heavy PJM zone sinks → exposure multiplier | FAIL | tracks layer-1-only almost exactly on both windows despite real multiplier variation | `ftr-bid-signal-results.md` |
| 2026-09-02 | Backlog-coverage-alone tilt [~#13] | YoY change in backlog coverage (RPO ÷ TTM revenue), unblended | FAIL | no independent information beyond what already failed | `backlog-coverage-signal-results.md` |
| 2026-09-02/03 | Capex-cycle-pair classifier | 4-criterion "customer + 2 of 3" screen to reproduce the grid/DER book without hindsight | PARTIAL | mechanically rebuilds the book (~1.55 long-only / ~1.95 hedged Sharpe); does **not** time or generalize (EV only) | memory `capex-cycle-pair-classifier` |
| 2026-09-03 | PJM MW-revision signal (Table B-9 load-forecast vintages) [~#12] | Δ accepted data-center MW between annual vintages → basket timing / de-risk | FAIL | acting on the 2026 stall cost ~+15% excess / ~+30% absolute; de-risked into a rally; quarterly corr ≈ 0 | `pjm-large-load-vintages-2026-09-03.md` |
| 2026-09-03 | VA transmission-filings probe (Deliverable A) | 72 Dominion VA CPCN cases → DC-driven project-$ fraction as timing / name-weighting | FAIL (timing + weighting) | DC-$ fraction rises 0.27→0.85 (confirmation only, 2022 level shift); quarterly corr ≈ 0; one TO, can't weight | `va-transmission-filings-probe-results.md` |
| 2026-09-03 | Category-demand → maker-rotation probe [~#16] | Census M3 / IP / PPI equipment-category demand × hand-built category→name matrix → cross-sectional tilt | FAIL | works 2019–22 (rank-IC t ≈ 2.4), dead 2023–26 (t = 0.6); 26% of book unmapped | `category-demand-rotation-results.md` |
| 2026-09-04/06 | **Capex-guidance revision signal (Deliverable D)** [~#15] | aggregate 15-utility 5-yr capex-guidance revisions ($ + size-weighted %, whole-panel + DC-attributed w/ point-in-time EM fill) as monthly rank-IC timing signal + de-risk/scaler overlay. First test of **commitment** data. | FAIL (both legs) | feasibility PASS 14/15 (NEE out). Every primary rank-IC negative, none \|t\|≥2 (best −1.54); de-risk leg **not exercised** (composite z never < +0.212); scaler fails G1 across the plateau; DC-attributed cut **not yet testable** (all DC-$ vintages 2026-dated). 9/10 sub-threshold \|t\|≥1.5 cells negative → weak *consistent inverse* sign (crowding read). Nothing wired live. | `capex-guidance-signal-results.md` |
| 2026-09-04/05 | Henry Hub / forward power-price signal [~#17] | free gas/power proxies (NG front-month, UNL 12-mo strip, contango, ETFs) as forward basket signal | FAIL (data-blocked) | no forward relationship survives a regime split or SMH control; kitchen-sink GBM/RF overfits (2023–26 OOF IC vanishes on 2018–26). EIA STEO `ELWHU_*` forward power series noted but not yet backtested | `hhub-forward-gas-signal-results.md` |

### C. Electrification strategy (broader thematic book, newest)

| Date | Attempt | What it tested | Verdict | Key result | Write-up |
|---|---|---|---|---|---|
| 2026-09-06 | Electrification strategy v1 | 12-cell grid (marquee/frozen/thematic × plain/+val/+sleeve/+short) long-only enhanced thematic beta ≈ VOLT | BUILT | best plain Sharpe ~0.85 (thematic), MaxDD ~−28%; value + sleeve trims vol & DD at 2–4%/yr drag | `electrification-strategy-v1-results.md` |
| 2026-09-06 | Short-leg-as-insurance probe | any thematic short sized purely to cut downturn risk at 3–6%/yr drag | PARTIAL | no thematic short works alone (DER sleeve is an anti-hedge in the feared solar-squeeze case). Best structure: GLD/short-duration 15% sleeve + real-yield-gated DLR+EQIX short 0.25× → MaxDD −31→−26, beta 0.76→0.57, Sharpe held; sleeve alone is 90% of it | `electrification-short-leg-insurance-probe-results.md` |

---

## Synthesis — why the signal search keeps failing

1. **Three data classes, same result.** Demand *forecasts* (Table B-9) are slow, annual,
   methodology-constrained, and revised as easily down as up. Physical *flow* (congestion, FTR,
   LMP) is real-time but the equity move front-runs it — the basket trades the live AI-capex
   narrative, not the grid's physics. Firm *commitments* (capex guidance) are forward-obligation
   by construction and still don't lead: by the time utilities are loudly raising 5-yr plans, the
   equipment stocks have already re-rated.

2. **Only one macro cycle.** Every "it worked" sub-result (category-demand, the Table B-9 annual
   lead, the value-chain Gate 1) lives entirely in the 2019–22 pre-AI window and dies in 2023–26,
   or is n≈5–8 with one shared inflection. There is one AI-power capex cycle in the data and
   most signals are just fitting its shape.

3. **The consistent-inverse sign** (capex-guidance §4.4): weak but directionally coherent —
   heavier guidance-raising precedes *lower* forward basket returns. If anything is worth
   revisiting it is a signal read *inverted* (fade the loudest guidance) or as one input in a
   combined/nonlinear model, on infrastructure that now exists — not in the thesis direction.

4. **Construction beats signal here.** The equal-weight basket cleared its gate; the trend-gate
   + vol-target overlay is the one durable positive; the pair classifier removes the hindsight
   objection. The edge, to the extent there is one, is "own the theme with disciplined risk
   management," not "time it."

---

## Still open / untried

- **County data-center permit / zoning / abatement capture (Deliverable E)** — the most-leading
  data considered; assemble the quarterly approved-MW series now, re-test ~2028 with real history.
- **PUC data-center tariff / ESA contracted-MW (Deliverable C)** — reuses the docket stack;
  history barely predates 2024, so feasibility-gate hard before spending on it.
- **EIA STEO `ELWHU_*` forward *power* price series** (~16-mo tail, per hub, 2010–) — a genuine
  free forward power feed; needs the STEO archive for point-in-time + a spark-spread test.
- **Combined / nonlinear model** using the capex-guidance panel + big-four hyperscaler capex as
  *joint* predictors (not one-as-control), and/or tail-only conditioning — cheap follow-up on
  built infrastructure.
- **Deliverable D re-run ~2027** — the DC-attributed half was structurally untestable (all DC-$
  vintages 2026-dated); the machinery is built and waiting for history.
- **grid_resilience off-by-default flags** (`BUSINESS_MODEL_ARCH`, `USE_ICR`,
  `PEER_GROUP_CONSTRUCTION`) — never validated on post-data-drift prices.
