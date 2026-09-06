# Research log — AI-power / electrification strategy

**Single source of truth for this line of research.** Every conversation appends to / updates
**this file** rather than spawning another standalone doc. Detailed write-ups live in the linked
docs; this is the index + verdicts + open items.

**Update convention:** add new threads to the ledger (§3) with a one-line verdict + a link to the
detail doc; keep §1 (current strategy) and §4 (open items) current; date every change.

Last updated: **2026-09-06**.

---

## 1. Current strategy — what would ship

**A rules-based, long-only electrification-equipment book. Enhanced thematic beta, not alpha**
(~0.85–0.88 correlated with the VOLT ETF; ~all the return is the theme).

- **Canonical build:** `electrification_strategy/` package (BUILT, 38 tests; spec
  `docs/superpowers/specs/2026-09-06-electrification-strategy-design.md`, plan alongside).
  Shipped variant **`screen +val+sleeve`**: 27-name supplier screen (the
  `capex-cycle-pair-classifier` "institutional customer + ≥2 of {profitable, earnings-valued,
  low-policy-dependence}" rule, supplier side only, **no ETF-membership gate**), equal-weight
  25% cap, 42-day lag, 20% trailing-vol target; + a graduated valuation-extension de-lever; +
  a 15% GLD/IEF sleeve. Full 2017–2026: **Sharpe 0.69 / MaxDD −24% / SPY β 0.59**. VOLT-fair
  window (2025-01→2026-08): **CAGR 27.7% vs VOLT 24.1%, Sharpe 1.35 vs 0.82, MaxDD −17% vs −24%**
  — it beats VOLT because it excludes VOLT's ~43% utilities/midstream/components.
  Docs: `docs/electrification-strategy-v1-results.md`, `docs/electrification-strategy-proposal-v2.md` (+ `.html`).
- **Earlier parallel framing (this session):** ETF-holdings universe (≥2 of {VOLT, ELFY, ZAP,
  GRID} + GICS filter) + 20% vol target + GLD/IEF sleeve. 2016–2026 Sharpe 0.71 / MaxDD −26%.
  **Superseded** by `screen` — the ETF-holdings gate turned out to depend on incomplete
  top-25 holdings for GRID/PAVE/ELFY and pulls in utilities. Kept for its IC proposal + charts:
  `docs/electrification-strategy-proposal.html` (Artifact: claude.ai/code/artifact/4e680fe5-fdb7-48a6-a266-b1f47f250050),
  `docs/electrification-ls-strategy-note.md` (§§1–8, the full long/short + hedge investigation).

**Adopted overlays / tilts**
- **GLD + short-Treasury sleeve** (~15%) — the clean risk reducer; hedges the two macro breaks
  (credit/growth scare → bonds; debasement/geopolitics → gold). Raises Sharpe, cuts MaxDD ~4pp.
- **70/30 grid-maintenance vs greenfield-DC-power tilt** — net-neutral on headline metrics, cuts
  exposure to an idiosyncratic data-center de-rating (moratorium/NIMBY). To be wired as a
  selectable construction. (`docs/electrification-strategy-v1-results.md` follow-ups.)

**Rejected / not durable**
- **Any structural short leg** — see §3 "Short-leg / hedge search". Best case is optional
  insurance (a DLR·EQIX or DER-sleeve short at fixed small notional) costing ~1–5%/yr of drag.
- **Rate-conditional rotation into utilities** — hurts Sharpe at any weight; redundant with the
  IEF sleeve; XLU is a worse asset than the supplier book.
- **Conditional merchant-power short** — every weight lowers Sharpe, deepens MaxDD.

**Primary disclosed risk:** a sustained falling-rate, risk-on reversal — the washed-out
clean-energy complex rallies and the theme de-rates. Not hedged.

---

## 2. Accumulated conclusion

Across **~25 signal attempts** (this project + predecessors), **no tradable timing or
cross-sectional edge has survived an honest out-of-sample / regime split.** Three independent
demand-data classes — forecast (PJM load forecasts), physical-flow (congestion, FTR), and
commitment (utility capex guidance) — have each failed. The defensible product is the
**rules-based long-only book + risk overlays**, pitched as disciplined thematic beta, plus the
**`capex-cycle-pair-classifier` construction rule** (the one constructive output: it mechanically
reproduces the hand-picked book, removing the "you hand-picked winners" objection).

---

## 3. Master research ledger

### Strategy construction
| thread | when | outcome | detail |
|---|---|---|---|
| grid_equipment_basket step 1 (thematic long book) | 2026-08 | BUILT + merged; solid long book, no edge beyond theme | `docs/grid-equipment-basket-step1-results.md` |
| Value-chain reframe (maker-vs-contractor tilt + intra-theme hedge) | 2026-08-29 | BUILT + merged; Gate 1 weak-PASS, Gate 2 FAIL | `docs/grid-equipment-value-chain-results.md` |
| Transmission rate-base compounders (FERC Form 1) — Proposal B | 2026-08-31 | BUILT + merged; gate FAILED (Q5−Q1 Sharpe −0.61) | `docs/transmission-rate-base-results.md` |
| Layer-1 risk overlay (trend gate + vol target) | 2026-08 | BUILT + merged; helps in-sample, hurts 2020–22 (overfit to DeepSeek) | `grid_equipment_basket/overlay.py` |
| Layer-2 grid-congestion regime | 2026-08-31 | BUILT + merged; no rung passed the pre-registered gate; `REGIME_ENABLED=False` | `docs/grid-regime-layer2-results.md` |
| **Electrification long-only strategy (`screen +val+sleeve`)** | 2026-09-06 | **BUILT (package, 38 tests) — the current ship** | `docs/electrification-strategy-v1-results.md` |
| Electrification strategy — ETF-holdings universe framing | 2026-09-05 | superseded by `screen`; IC proposal + charts kept | `docs/electrification-strategy-proposal.html` |

### Timing signals (all FAILED)
| thread | when | outcome | detail |
|---|---|---|---|
| DC-load multi-source signal (FE / generation-queue) | 2026-08 | FAILED — unverifiable, zone attribution ambiguous | `docs/compact_2026-08-*` |
| PJM MW-revision (Load Forecast Table B-9), 6 vintages | 2026-09-03 | FAILED — lagging, de-risked into a rally (cost ~+15pp excess) | `docs/pjm-large-load-vintages-2026-09-03.md` |
| FTR bid-implied forward congestion | 2026-09-02 | FAILED — same gate as layer-2; tracks layer-1 exactly | `docs/ftr-bid-signal-results.md` |
| Big-4 hyperscaler capex-deceleration de-risk | 2026-08-31 | FAILED — fires ~a year late; lagging confirmation | `grid_equipment_basket/capex_signal.py` |
| Utility 5-yr capex-guidance revisions (Deliverable D) — first test of the "commitment" data class | 2026-09-04 → 2026-09-06 | FAILED both legs. Feasibility PASS 14/15 (NEE out, documented). Every primary rank-IC negative, none \|t\|≥2 (best −1.54); de-risk leg **not exercised** (composite z never < +0.212, so the multiplier is flat 1.0 — its "FAIL" was really the vol-target-only baseline); two-sided scaler fails G1 across the plateau; DC-attributed cut **not yet testable** (all DC-$ vintages 2026-dated, z-score never warms) → re-run ~2027. **9 of 10 sub-threshold \|t\|≥1.5 cells are negative** — a weak but *consistent inverse* sign (crowding / already-priced read); if revisited, test inverted / in a combined model, not thesis-direction. Nothing wired live. Built: seed panel + loader + `capex_guidance_signal.py` + `--capex-guidance`. | `docs/capex-guidance-signal-results.md` |
| Category-demand → maker rotation (Census M3 / PPI panel) | 2026-09-03 | FAILED — fwd-1m IC t=0.8; fwd-3m IC 100% pre-AI; dead 2023–26 | `docs/category-demand-rotation-results.md` |
| Henry Hub / forward-gas signal (5 rounds incl. GBM/RF, IS-OOS split) | 2026-09-05 | FAILED — no regime-robust relationship; overfits the 43-mo window | `docs/hhub-forward-gas-signal-results.md` |
| STEO `ELWHU_*` forward power price (~16-mo forecast tail, free) | 2026-09-05 | SCOPED, NOT RUN — needs STEO archive for point-in-time spark-spread test | `docs/hhub-forward-gas-signal-results.md` §Round 3 |
| Backlog-surprise factor (1,778 events) | 2026-08-31 | FAILED — event-study gate FAILED | `docs/backlog-surprise-factor-results.md` |

### Cross-sectional / weighting (all FAILED)
| thread | when | outcome | detail |
|---|---|---|---|
| Disclosed-backlog-growth tilt | 2026-08 | FAILED — over-weights FLNC | `docs/backlog-surprise-and-proposals` |
| Backlog-coverage-alone tilt (RPO ÷ TTM revenue) | 2026-09-02 | FAILED Gate 1 — indistinguishable from equal-weight where it engages | `docs/backlog-coverage-signal-results.md` |
| Grid-demand-sensitivity factor (Proposal D) | 2026-08-31 | FAILED — IC t = −0.47 | `docs/triage_2026-08-31-proposals-bda.md` |
| VA/Dominion transmission-filing work-type tilt | 2026-09-03 | FAILED — mix ~all line/rebuild/substation; static "tilt to EPCs, zero the makers" | `docs/va-transmission-filings-probe-results.md` |
| Cross-sectional price momentum, wide 44-name universe | 2026-09-05 | FAILED — 2023–24 burst only (+2.5 Sharpe), +0.24 since; sector bet on EPCs | `research/xsec_factors_ls/mom_stress.py` |
| Cointegration pairs stat-arb (wide universe) | 2026-09-05 | FAILED — OOS Sharpe ~0.3–0.5, one −0.9 year | `research/xsec_factors_ls/pairs.py` |
| Un-crowd the universe (POWL/ATKR/AZZ/…) | 2026-08-31 | FAILED — underperforms marquee, deeper DD, 0.84 corr | `docs/triage_2026-08-31-differentiation-ideas.md` |

### Short-leg / hedge search — CLOSED
| candidate | outcome |
|---|---|
| DER sleeve (resi solar + EV + BTM storage) | +0.25–0.5 corr; L/S positive **only** 2023–26; negative both pre-AI windows |
| Short TAN (ETF) | same regime pattern, clean 2013→ history |
| Firm generation (gas turbine / nuclear / IPP) | +0.60 corr, falls **harder** in the long book's drawdowns — more of the same beta |
| 7 other-industry supplier/downstream pairs (oilsvc/E&P, semicap/memory, ag, mining, rail, aero/airlines, biotools/biotech) | only **aero-supply/airlines** positive every regime (full Sharpe ~0.80) — **bookmarked as its own book**; blending it away kills it |
| China broad (FXI+MCHI) | least correlated (+0.03 recent), positive every window, **but** 0.85 SPY β, recurring fat left tail when China decouples/rallies, **no exit signal works** |
| Retail / office REITs (Google-AI "alt real estate" idea) | +0.58 corr, flat pre-2019, −5% in the long book's worst months — not a hedge |
| GLD / long-duration / VIX / USD | GLD ~0 corr (uncorrelated diversifier — KEPT in the sleeve); TLT hedges a growth scare but +0.36 in a rate shock; VIXY −0.60 (bleeds carry); USD −0.31 (faded) — tactical only |
| Conditional rate-signal short (short weight ∝ Δ 10y real yield) | full-cycle **wash** (+3.4%/yr 2023–26, −1%/yr otherwise, net +0.1%/yr, t=0.1); β barely moves; DD wider — documented as Phase-2 pending non-rate signals |
| DLR·EQIX short as fixed insurance | runner-up in the package grid — trims MaxDD ~1.4pp for ~1.2pp/yr drag; optional |
Detail: `docs/electrification-ls-strategy-note.md` §§6–8, `docs/electrification-short-leg-insurance-probe-results.md`.

### Idea-generation / other
| thread | when | outcome | detail |
|---|---|---|---|
| capex-cycle-pair classifier ("customer + 2 of 3" rule) | 2026-09-02 | **KEEPER** — mechanically reproduces the book; not a timing rule, doesn't generalize | memory `capex-cycle-pair-classifier` |
| Thematic-catalyst detector (utility load-forecast filings name a new large-load category) | 2026-09-02 | idea-generation tool, not a trading signal — PJM named "Data Centers" 1 day before ChatGPT | `docs/thematic-catalyst-detection-2026-09-02.md` |
| VA/Dominion transmission-project demand DB (Deliverable A) | 2026-09-03 | BUILT (72-case seed + loader + SCC Breeze API) — DC-$ share confirms the buildout; not a signal | `docs/va-transmission-filings-probe-results.md` |
| Data-center commitment signals — Part 2 (C = PUC ESA/tariff MW, E = county permits) | 2026-09-03 | SCOPED; D failed; C + E not yet built | `docs/handoff_2026-09-03-transmission-project-filings.md` §6 |
| Interconnection-queue *velocity* (Δ large-load MW by zone, vintages) | 2026-09-02 → 2026-09-03 | Superseded by the row above — first probed 2026-09-02 with only 2 annual vintages (inconclusive), then a later session pulled 6 vintages (2021–2026, PDF+xlsx Table B-9) and got a definitive FAILED verdict | `docs/pjm-large-load-vintages-2026-09-03.md` |
| Differentiation-idea triages (rounds 1 & 2, ~13 ideas) | 2026-08-31 / 09-04 | 8 of 10 (round 2) dead — already mainstream / re-rated | `docs/triage_2026-09-04-differentiation-ideas-round2.md` |
| RT/DA LMP spread cross-sectional split | 2026-09-01 | pivoted to FTR (failed); the RT/DA idea itself still literally untried | `docs/handoff_2026-08-15-rt-da-spread-signal.md` |
| Earnings-estimate-revision / analyst-breadth momentum | — | NOT ATTEMPTED — needs ≥3yr point-in-time consensus EPS feed | `docs/handoff_2026-09-01…` §8.2 |

---

## 4. Open items / pre-live gates

1. **Point-in-time universe** (both the `screen` judgment columns and, if `thematic`/`frozen`
   kept, a full GRID/PAVE/ELFY holdings feed) — required before any live capital.
2. **Wire the 70/30 grid-maintenance tilt** as a selectable construction in
   `electrification_strategy/universe.py`.
3. **Pre-register** the vol target, valuation lookbacks and sleeve weights via a plateau test;
   lock an OOS start date.
4. **Implementation / capacity study** — borrow, turnover, transaction cost at target AUM.
5. **Moratorium / NIMBY risk monitor** — state/county moratorium status + permit-approval counts
   (Loudoun, Prince William, Central OH, N. Texas) as a de-risk input (Google-AI idea D).
6. **Options-collar** sizing — the only cause-agnostic hedge; blocked on an options-data feed.
7. **Phase-2 conditional short** — only if non-rate consumer-vs-industrial signals (relative
   valuation, earnings-revision breadth, policy calendar, credit) can be built; low prior.
8. **Aero-supply / airlines** — evaluate as a *separate* L/S book (different mandate).
9. **STEO forward-power spark-spread** — pull the STEO monthly archive, build the point-in-time
   PJM-West / ERCOT forward series, test the implied-heat-rate move.
10. Keep the **interconnection-queue-velocity** quarterly capture running for a ~2028 test.
11. **Capex-guidance signal, revisited** — cheap follow-ups on built infrastructure
    (`capex_guidance_signal.py`, the 15-utility panel): (a) test the signal *inverted* (fade the
    loudest guidance-raising) given the consistent negative sign; (b) use the capex-guidance panel
    + big-four hyperscaler capex as *joint* predictors (not one-as-control) and/or tail-only
    conditioning; (c) re-run the DC-attributed cut ~2027 when >1 yr of dated DC-$ disclosure exists.
    Low prior, but the machinery is done. Also scoped but unbuilt: Deliverable C (PUC DC tariff/ESA
    contracted-MW) and E (county DC permits) — `handoff_2026-09-03-transmission-project-filings.md` §6.

---

## 5. Code & doc map

- **Built package:** `electrification_strategy/` (+ `tests/electrification_strategy/`, 38 tests).
  Predecessors still in tree: `grid_equipment_basket/`, `grid_resilience/`, `transmission_rate_base/`,
  `backlog_factor/`, `grid_demand_factor/`.
- **Kept data modules:** `grid_resilience/data/va_transmission_data.py` + seed
  `grid_resilience/data/seed/va_transmission_projects.csv`; `grid_resilience/data/seed/pjm_large_load_b9_vintages.csv`;
  `grid_equipment_basket/data/category_demand.py` (FRED panel — dashboard series, not a signal);
  `grid_equipment_basket/ftr_signal.py` (negative result, kept).
- **Exploratory probes (tracked):** `research/` — `va_transmission_probe/`,
  `category_demand_rotation/`, `forward_gas_power/`, `xsec_factors_ls/` (see `research/README.md`).
  The parallel electrification-strategy session keeps its throwaway scripts in the gitignored
  `scratchpad/` (`crowding_probe`, `backlash_hedge`, `hedge_probe`, `hedge_feasibility`,
  `utility_rotation`, `make_exposure_plot`). None are production; re-run to regenerate.
- **Proposals:** `docs/electrification-strategy-proposal-v2.md` / `.html` (canonical, `screen`),
  `docs/electrification-strategy-proposal.html` (earlier, ETF-holdings).
- **Handoffs (context):** `handoff_2026-09-01-grid-buildout-long-short.md` (the full prior record),
  `handoff_2026-09-03-transmission-project-filings.md`, `handoff_2026-09-05-electrification-strategy.md`.
