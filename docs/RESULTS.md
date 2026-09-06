# Grid Buildout / Electrification Strategy — Consolidated Results Ledger

**Purpose:** the single source of truth for the AI-power / grid-buildout / electrification research
program. Every conversation updates *this file* rather than spawning another standalone doc.

## Update protocol (for every future chat)

1. **Append, don't rewrite.** Add a row to the relevant ledger table; never delete a past result.
2. Each new probe gets: a one-line ledger row here (name · data class · verdict · `docs/<file>` link)
   **and** its own detailed `docs/<name>-results.md`. This file is the index; the detail doc is the record.
3. Update **§1 Current state** and **§7 Stop-rule counter** if the probe changes them.
4. Keep the memory index (`~/.claude/.../memory/MEMORY.md`) in sync — one line per negative result.
5. Convert relative dates to absolute. Note which conversation / date.

Last updated: **2026-09-06**.

---

## 1. Current state

**What shipped:** `electrification_strategy/` — a rules-based **long-only enhanced-thematic-beta**
strategy. Honest label: *enhanced thematic beta, not alpha.*

- **Universe:** US names in ≥2 of the electrification ETFs {VOLT, ELFY, ZAP, GRID}, filtered to
  GICS Electrical Equipment + Construction & Engineering; point-in-time membership + 42-day lag.
- **Weighting:** equal-weight, 25% single-name cap, quarterly reconstitution.
- **Risk:** 20% annualised vol target (1.5× cap, slack held at rf).
- **Overlay:** 0–15% diversifier sleeve (GLD + short Treasuries) sized by a real-yield +
  HY-credit-spread signal.
- **Backtest 2016–2026:** CAGR 17.3% / Sharpe 0.71 / MaxDD −25.9% / SPY β 0.78. Positive Sharpe
  every sub-window (2016–18 0.29, 2019–22 0.54, 2023–26 1.48). Since VOLT's Dec-2024 launch it
  matches the ETF on return with better risk control — the edge over just buying VOLT is
  discipline + a track that reconstructs to 2016, not return.
- **Primary risk:** a sustained falling-rate, risk-on reversal (washed-out clean-energy rallies,
  the theme de-rates). Docs: `electrification-strategy-proposal-v2.md`, `electrification-strategy-v1-results.md`,
  `handoff_2026-09-05-electrification-strategy.md`.

**Long/short variant (Phase-2, not shipped):** long the ~33 industrial suppliers / short the
consumer clean-energy sleeve (ENPH SEDG RUN NOVA CHPT EVGO BLNK STEM), short weight scaled
0 → 0.55× by the 6-month change in the 10-year real yield. 2013–26: CAGR 14.0% / Sharpe 0.62 /
MaxDD −31% / β 0.70 — same Sharpe & CAGR as long-only, *wider* drawdown. Short-leg +3.4%/yr in
2023–26 (t=2.6) but full-cycle +0.1%/yr (t=0.1). Structure is right; the rate signal alone does
not earn its place — needs more consumer-vs-industrial signals (valuation, revision breadth,
policy calendar, credit).

**Pithy description:** *long the transition's profitable suppliers, short its consumer pure-plays*
— grid/DER version: *long who builds the grid, short who plugs in*.

**Accumulated conclusion:** no signal-based stock-selection or timing edge exists in this theme at
this sample (one macro regime, ~15 yr of grid data, ~3.7 yr of the data-center era). The trade
stands on **structural logic (who the customer is) + trailing performance**, pitched with full
disclosure of crowding + regime risk.

---

## 2. Construction rule — the one constructive methodology output

**The 4-criterion "customer + 2 of 3" classifier** (see memory `capex-cycle-pair-classifier`,
session 2026-09-02/03). Mechanically reproduces the hand-picked book, removing the hindsight
objection.

- **Pool:** listed pure-plays in electrification/grid (10-K Item 1: grid T&D equipment, DC power &
  cooling, grid-scale storage, power EPC, residential/distributed solar, EV charging, BTM storage).
- **4 criteria:** (1) customer — B2B institutional vs retail/household [mandatory axis];
  (2) profitable — op margin > 0 AND FCF > 0; (3) valuation duration — real earnings/P-E vs EV-Sales;
  (4) policy dependence — not subsidy-gated vs 10-K risk factors lean on ITC/45X/NEM/NEVI.
- **Assignment:** pool pure-play + customer axis matches + ≥2 of {profit, duration, policy}.
  LONG = institutional customer; SHORT = mirror. Neither → excluded (e.g. FLNC).
- **Result:** LONG 17 (ETN HUBB NVT VRT GEV PWR MYRG PRIM POWL ATKR WCC ABBNY SBGSY PRYMY HTHIY
  NXT FSLR) / SHORT 8 (ENPH SEDG RUN CHPT BLNK STEM EVGO WBX). Long-only EW Sharpe ~1.55,
  beta-hedged+vol-tgt pair ~1.95 (2023–26) = hand-picked's 1.99. Strict (all 4) ≈ loose.
- **Does NOT:** fix regime dependence (still ~−0.7 pre-2023); generalize (EV only — nuclear/space/
  cannabis/hydrogen/genomics fail; precondition = institutional side in a *real* capex up-cycle);
  qualify as a factor (loads negative on quality, ~40% is PAVE/TAN sector rotation, rest is
  one-regime concentration, t≈1.6 full-sample).

Raw (unhedged) dollar-neutral long/short Sharpe ≈ **1.2–1.4** (2023–26); the ~2.2 from earlier
handoffs required the beta-hedge + 20%-vol-target overlay and one window.

---

## 3. Signal-attempt ledger — timing / demand signals

**All failed.** ~17+ distinct attempts. Verdict shorthand: FAIL = no tradable edge; CONF = works
only as confirmation/dashboard, not a signal.

| # | attempt | data class | verdict | detail doc |
|---|---|---|---|---|
| 1 | disclosed-backlog-growth tilt | fundamental | FAIL (over-weighted FLNC; ≈ equal-weight) | grid-equipment-basket-step1-results.md |
| 2 | value-chain maker-vs-contractor tilt / pair | fundamental | Gate 1 weak-PASS, Gate 2 FAIL | grid-equipment-value-chain-results.md |
| 3 | backlog-surprise factor (1,778-event breadth study) | fundamental | FAIL (non-monotone, ~1wk bump reverses) | backlog-surprise-factor-results.md |
| 4 | grid-demand sensitivity factor (Proposal D) | physical nowcast | FAIL (rank-IC t = −0.47) | triage_2026-08-31-proposals-bda.md |
| 5 | transmission rate-base compounders (FERC Form 1) | regulatory | FAIL (Q5−Q1 Sharpe −0.61, no residual α) | transmission-rate-base-results.md |
| 6 | Proposal A — zone-matched congestion pair | physical | killed at triage (breadth) | triage_2026-08-31-proposals-bda.md |
| 7 | Layer 1 — price trend gate + vol target | price | ADOPTED as overlay (helps primary DD, hurts 2020–22) | grid-regime-layer2-results.md |
| 8 | Layer 2 — grid-congestion regime (6-rung ladder) | physical (LMP congestion) | FAIL (no rung cleared gate); briefly adopted, REVERTED | grid-regime-layer2-results.md |
| 8a | Option A — DC-minus-rest-of-PJM relative congestion | physical | FAIL (erased prior-window edge = COVID coincidence) | grid-regime-layer2-results.md §9 |
| 8b | Option B — signal-tilted basket/short-utilities pair | physical | FAIL (signal backwards) | grid-regime-layer2-results.md §9 |
| 9 | big-four hyperscaler capex-deceleration de-risk | fundamental | FAIL (fires ~a year late; lagging confirmation) | handoff_2026-09-01-grid-buildout-long-short.md §3.5 |
| 10 | backlog *coverage*-alone tilt | fundamental | FAIL (≈ equal-weight; no data pre-2023) | backlog-coverage-signal-results.md |
| 11 | PJM data-center MW-revision timing (Table B-9, 6 vintages) | regulatory forecast | FAIL (de-risked into a rally; quarterly corr≈0) | pjm-large-load-vintages-2026-09-03.md |
| 12 | category-demand → maker-rotation (Census M3 / PPI / IP NAICS 3353) | national industry | FAIL (fwd-1m IC t=0.78; fwd-3m all from 2019–22) | category-demand-rotation-results.md |
| 13 | capex-guidance revision signal (Deliverable D — 15 utilities' 5-yr guidance) | regulatory/transcript | FAIL (rank-IC negative every horizon; scaler misses G1) | capex-guidance-signal-results.md |
| 14 | Henry Hub / forward power-gas signal (+ 25-signal GBM/RF kitchen sink) | commodity | FAIL (2023–26 OOF IC vanishes on 2018–26; overfit) | hhub-forward-gas-signal-results.md |
| 15 | VA transmission-filings project-$ (Deliverable A — 72 Dominion CPCN cases) | regulatory | FAIL as timing (quarterly corr≈0); CONF — DC-$ share 0.27→0.85 | va-transmission-filings-probe-results.md |
| 16 | non-price name selection (RTEP, patents, ISO queues) | regulatory/IP | FAIL — no external dataset reproduces the name list (names customers, not suppliers) | pjm-large-load-vintages-2026-09-03.md |
| 17 | PJM capacity-auction (BRA) clearing prices as timing | market | FAIL — lags equity 18–24mo, inverted, no exit signal | pjm-large-load-vintages-2026-09-03.md |

Related non-strategy negatives (older): FTR-bid signal, RT/DA spread, outage/reserve-margin
(EIA-860 `status` is annual-scale — dead), interconnection-queue velocity (not testable, <2
vintages — start collecting for a ~2028 test).

---

## 4. Name-weighting ledger — cross-sectional tilts on the long book

**All failed to beat equal-weight.** 17 correlated names in one regime is too thin a cross-section.

| tilt | verdict |
|---|---|
| backlog growth (top½ ×1.25 / bottom½ ×0.75) | FAIL — Sharpe 1.363 vs EW 1.360, CAGR lower |
| backlog coverage (YoY Δ) | FAIL — 1.446 vs 1.448, drawdown worse; inert pre-2023 |
| value-chain maker vs contractor | FAIL Gate 2 |
| within-book 6-month cross-sectional momentum (top-6 / top-9) | FAIL — Sharpe 1.07 / 1.29 vs EW 1.41, deeper DD |
| DC-power sub-sector overweight (VRT/GEV/POWL/ETN 2.5×) | marginal — +0.09 Sharpe 2023–26 (in noise), −0.12 pre-2023 |
| national category-demand × category→name matrix | FAIL — see ledger #12 |

Conclusion: run **equal-weight, 25% cap**. Equal-weight already captures the VRT/GEV/POWL run
because it holds them.

---

## 5. Short-leg / hedge ledger — CLOSED (2026-09-05)

**No holdable negatively-correlated leg exists.** ~25 short/hedge candidates tested.

| candidate class | result |
|---|---|
| clean energy (solar / EV / storage / hydrogen — the DER sleeve) | +0.25–0.6 corr, 2023–26 only, anti-hedge in the feared solar-squeeze scenario |
| firm generation (gas turbine / nuclear / IPP) | +0.60 corr, falls *harder* in drawdowns (priced beta, not a hedge) |
| 7 other industries (aero-supply, airlines, etc.) | only aero-supply works standalone; blending destroys it |
| China broad | +0.03 recent corr but 0.85 SPY β, recurring fat left tail, no exit signal |
| retail/office REITs (DC-moratorium redirect thesis) | +0.58 corr, regime-dependent — same pattern as all others |
| genuine negatives | only VIXY (−0.60, −30%/yr carry) and USD (−0.31, faded post-2022) — tactical only |
| best *structure* found | GLD/short-duration 15% sleeve + real-yield-gated DLR+EQIX short 0.25×: MaxDD −31→−26, β 0.76→0.57, Sharpe held 0.83, +13% in the feared scenario, 3.3%/yr drag (but ~15%/yr inside a rate-rising bull). Sleeve-alone is 90% of it with no short. |

Docs: `electrification-short-leg-insurance-probe-results.md`, `electrification-ls-strategy-note.md`
§§6–8, `handoff_2026-09-05-electrification-strategy.md` §§1c, 2.

---

## 6. Open / untried

| item | status | doc |
|---|---|---|
| **Deliverable C** — PUC data-center tariff/ESA contracted-MW (commitment data) | scoped, not built; feasibility-gate on history depth (<2024) | handoff_2026-09-03-transmission-project-filings.md §6 |
| **Deliverable E** — county DC permit / zoning / abatement filings (most-leading) | scoped, not built; also a data-collection project to start now | handoff_2026-09-03-transmission-project-filings.md §6 |
| Deliverable D — utility capex-guidance revisions | **BUILT, FAILED** (ledger #13) | capex-guidance-signal-results.md |
| labor-bottleneck data (BLS OEWS electrician/lineworker wages, job-postings) as backlog-conversion signal | survived the round-2 triage check; needs a cheap correlation check first | triage_2026-09-04-differentiation-ideas-round2.md #7 |
| cat-bond / reinsurance pricing (Artemis.bm) as leading indicator for utility wildfire-liability equity | survived triage; data may be subscription-only; check lead vs lag first | triage_2026-09-04-differentiation-ideas-round2.md #10 |
| interconnection-queue velocity | not testable (<2 vintages) — archive quarterly vintages now, test ~2028 | handoff_2026-09-01-grid-buildout-long-short.md §5.1 |
| EIA STEO ELWHU_* forward wholesale power series | real free forward *power* series (~16mo tail, per hub) — not yet backtested (needs STEO archive for point-in-time + spark-spread test) | hhub-forward-gas-signal-results.md |
| RT/DA (real-time − day-ahead) LMP spread — cross-sectional | never built; low prior (DA congestion showed ~0 equity corr) | handoff_2026-09-01-grid-buildout-long-short.md §8.1 |

Round-2 triage also killed 8 thematic adjacencies as already-crowded/re-rated (crypto-to-AI
hosting, wildfire-liability short, power-semis, gas midstream, nuclear PPA, European grid RV,
demand-response/VPP, water-rights) — see `triage_2026-09-04-differentiation-ideas-round2.md`.

---

## 7. Stop-rule counter

**~17 signal attempts, all FAIL.** Pre-committed stop rule (from the transmission-filings handoff):
if Deliverables **C** and **E** also return quarterly rank-IC ≈ 0 like Table B-9 and the VA probe
→ that's ~19 attempts → **stop looking for a data-center demand signal; ship the trade as a
discretionary thematic position** (`electrification_strategy/` long-only core, per
`handoff_2026-09-05-electrification-strategy.md` §1a) and keep the E + queue-velocity captures
running for a 2028 re-test.

---

## 8. Module & key-doc index

| path | what |
|---|---|
| `electrification_strategy/` | **the shipped strategy** — long-only enhanced-thematic-beta, CLI `python -m electrification_strategy` |
| `grid_equipment_basket/` | earlier 9–17-name theme basket + Layer-1/2 overlays + all the failed signal modules (`grid_regime.py`, `capex_signal.py`, `capex_guidance_signal.py`, `category_demand.py`, `ftr_signal.py`, `backlog_data.py`, `value_chain.py`) |
| `grid_resilience/` | the original GSI / stress-beta utility long/short; also hosts shared data fetchers + the VA / PJM-large-load / transmission seed data |
| `grid_demand_factor/`, `transmission_rate_base/`, `backlog_factor/`, `dc_demand_basket/` | negative-result probe packages (see root README blockquote) |
| `docs/handoff_2026-09-05-electrification-strategy.md` | current master state for the shipped strategy |
| `docs/handoff_2026-09-01-grid-buildout-long-short.md` | the long/short investigation of record (pre-electrification-package) |
| `docs/handoff_2026-09-03-transmission-project-filings.md` | the regulatory-filings data source — Deliverable A (built, failed), C/D/E scope |
| `docs/electrification-ls-strategy-note.md` | the full long/short + hedge investigation (§§1–8) |
| memory `capex-cycle-pair-classifier`, `pjm-mw-revision-signal-negative`, `va-transmission-filings-probe-negative`, `capex-guidance-signal-negative`, `hhub-forward-gas-signal-negative`, `category-demand-rotation-negative`, `electrification-short-leg-insurance-probe` | per-probe memory records |
