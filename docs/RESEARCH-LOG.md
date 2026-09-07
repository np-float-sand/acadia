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
| Layer-2 Option A — DC-minus-rest-of-PJM *relative* congestion | 2026-09-01 | FAILED — still misses primary-window Calmar; removing the common-mode component **erases** the prior-window edge (Sharpe 0.61→0.21), proving the absolute signal's 2020–22 "win" was the COVID demand-collapse coincidence | `docs/grid-regime-layer2-results.md` §7 |
| Layer-2 OOS update (PJM zone LMP Jan–Aug 2026) | 2026-09-01 | congestion overlay did **not** beat the plain price-gate OOS (Sharpe 1.03 vs 1.16); drawdown protection held. Basis for the `REGIME_ENABLED` revert | `docs/grid-regime-layer2-results.md` §9 |
| Utility 5-yr capex-guidance revisions (Deliverable D) — first test of the "commitment" data class | 2026-09-04 → 2026-09-06 | FAILED both legs. Feasibility PASS 14/15 (NEE out, documented). Every primary rank-IC negative, none \|t\|≥2 (best −1.54); de-risk leg **not exercised** (composite z never < +0.212, so the multiplier is flat 1.0 — its "FAIL" was really the vol-target-only baseline); two-sided scaler fails G1 across the plateau; DC-attributed cut **not yet testable** (all DC-$ vintages 2026-dated, z-score never warms) → re-run ~2027. **9 of 10 sub-threshold \|t\|≥1.5 cells are negative** — a weak but *consistent inverse* sign (crowding / already-priced read); if revisited, test inverted / in a combined model, not thesis-direction. Nothing wired live. Built: seed panel + loader + `capex_guidance_signal.py` + `--capex-guidance`. | `docs/capex-guidance-signal-results.md` |
| Category-demand → maker rotation (Census M3 / PPI panel) | 2026-09-03 | FAILED — fwd-1m IC t=0.8; fwd-3m IC 100% pre-AI; dead 2023–26 | `docs/category-demand-rotation-results.md` |
| Henry Hub / forward-gas signal (5 rounds incl. GBM/RF, IS-OOS split) | 2026-09-05 | FAILED — no regime-robust relationship; overfits the 43-mo window | `docs/hhub-forward-gas-signal-results.md` |
| STEO `ELWHU_*` forward power price (~16-mo forecast tail, free) | 2026-09-05 | SCOPED, NOT RUN — needs STEO archive for point-in-time spark-spread test | `docs/hhub-forward-gas-signal-results.md` §Round 3 |
| Backlog-surprise factor (1,778 events) | 2026-08-31 | FAILED — event-study gate FAILED | `docs/backlog-surprise-factor-results.md` |
| PJM capacity-auction (BRA) clearing prices | 2026-09-03 | FAILED — lags equity 18–24 mo, inverted (best at the $29 low), no exit signal | RESEARCH-LOG §6 (2026-09-02→09-06) |
| Book-to-bill scaler (sleeve-aggregate order book) | 2026-09-06 | INFEASIBLE from XBRL — only PWR/MYRG/PRIM have usable RPO; makers report it in MD&A prose only | RESEARCH-LOG §6 (2026-09-02→09-06) |

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
| Layer-2 Option B — signal-tilted long-basket / short-rest-of-PJM-utilities pair | 2026-09-01 | FAILED the pre-registered *spread-informative* check — the basket beat XLU **more** when the signal said congestion was easing (+18 bp/day) than tightening (+7 bp/day); the signal was backwards. Any Sharpe gain over a static pair came from a smaller average short, not signal content | `docs/grid-regime-layer2-results.md` §8 |
| Within-17-name 6-mo momentum + DC-power sub-sector overweight (backward analysis) | 2026-09-06 | FAILED — momentum tilt loses to equal-weight (Sharpe 1.07 vs 1.41); DC-power 2.5× buys +0.09 Sharpe 2023–26 (in noise), −0.12 pre-2023; EW already holds them | RESEARCH-LOG §6 (2026-09-02→09-06) |

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
| DER sleeve SHORT, re-tested on `screen +val+sleeve` (2026-09-06) | **anti-hedge, worse at every size** — full Sharpe 0.69→0.36 (0.15×) → −0.18 (0.40×); MaxDD −24%→−42%→−83%; feared-scenario P&L −36% to −95%. Re-confirms the earlier verdict |
| DER sleeve LONG when real yields falling, 0→10% (2026-09-06) | **mild PASS** — Sharpe 0.69→0.77, CAGR +2.4pp, MaxDD −24→−27, +7% feared, rescues 2019–22. Candidate = open item #17, not wired |
| rate-conditional rotation INTO utilities (XLU) (2026-09-06) | **REJECT** — Sharpe 0.69→0.63 (0.30×), MaxDD deepens; redundant with the IEF sleeve leg; XLU is a worse asset (Sh 0.38 / DD −36%). VOLT's ~30% utility weight is *why its Sharpe is lower* |
| conditional merchant-power short (VST/NRG/CEG/TLN), backlash-triggered (2026-09-06) | **REJECT** — every weight lowers Sharpe (1.00→0.93), deepens MaxDD; merchant doesn't reliably fall in an idiosyncratic DC selloff |
Detail: `docs/electrification-ls-strategy-note.md` §§6–8, `docs/electrification-short-leg-insurance-probe-results.md`.

### Idea-generation / other
| thread | when | outcome | detail |
|---|---|---|---|
| capex-cycle-pair classifier ("customer + 2 of 3" rule) | 2026-09-02 | **KEEPER** — mechanically reproduces the book; not a timing rule, doesn't generalize | memory `capex-cycle-pair-classifier` |
| Classifier generalization (EV / hydrogen / nuclear / space / cannabis / genomics) + multi-pool | 2026-09-06 | EV only (post-mania ~0.7); nuclear inverts, space/cannabis self-reject. grid/DER+EV concurrent book ~1.69 Sharpe; more pools dilute. "2 live pairs, not 5." | RESEARCH-LOG §6 (2026-09-02→09-06) |
| grid/DER spread attribution (is it a factor?) | 2026-09-06 | NO — loads −1 on quality, ≈0 on duration/momentum; ~40 % is PAVE/TAN sector rotation; PAVE/XLI itself is Sharpe −0.51 (2023–26). Residual is one-regime concentration, not alpha. | RESEARCH-LOG §6 (2026-09-02→09-06) |
| US-industrial-policy L/S variant (46 policy suppliers / 13 non-policy industrials) | 2026-09-06 | NOT SHIPPED — β-hedged Sharpe 0.81 (2023–26) / 0.18 (2021–22); one macro bet, regime-dependent, PAVE/XLI benchmark negative | RESEARCH-LOG §6 (2026-09-02→09-06) |
| Non-price name selection (RTEP / patents / ISO queues → the 17-name book) | 2026-09-06 | FAILED — no external dataset reproduces the names; ISO/utility filings name customers not suppliers; patents skew to SOEs and miss all EPCs | RESEARCH-LOG §6 (2026-09-02→09-06) |
| Thematic-catalyst detector (utility load-forecast filings name a new large-load category) | 2026-09-02 | idea-generation tool, not a trading signal — PJM named "Data Centers" 1 day before ChatGPT | `docs/thematic-catalyst-detection-2026-09-02.md` |
| VA/Dominion transmission-project demand DB (Deliverable A) | 2026-09-03 | BUILT (72-case seed + loader + SCC Breeze API) — DC-$ share confirms the buildout; not a signal | `docs/va-transmission-filings-probe-results.md` |
| Data-center commitment signals — Part 2 (C = PUC ESA/tariff MW, E = county permits) | 2026-09-03 | SCOPED; D failed; C + E not yet built | `docs/handoff_2026-09-03-transmission-project-filings.md` §6 |
| Interconnection-queue *velocity* (Δ large-load MW by zone, vintages) | 2026-09-02 → 2026-09-03 | Superseded by the row above — first probed 2026-09-02 with only 2 annual vintages (inconclusive), then a later session pulled 6 vintages (2021–2026, PDF+xlsx Table B-9) and got a definitive FAILED verdict | `docs/pjm-large-load-vintages-2026-09-03.md` |
| Differentiation-idea triages (rounds 1 & 2, ~13 ideas) | 2026-08-31 / 09-04 | round 1: sell-the-overlay-as-insurance FAILED (premium 16–28 pp/yr vs ~10%/yr for puts), un-crowd FAILED, queue-velocity NOT TESTABLE. round 2: 8 of 10 dead — already mainstream / re-rated; survivors #7 labor-bottleneck QCEW data, #10 cat-bond pricing (spike-before-build) | `docs/triage_2026-08-31-differentiation-ideas.md`, `docs/triage_2026-09-04-differentiation-ideas-round2.md` |
| Grid-buildout L/S proposal — long US+foreign buildout / short GRID (or DER sleeve), β-hedged + vol-tgt | 2026-09-01 | DOCUMENTED, not shipped as an edge — DC-era Sharpe ~1.1 vs GRID / ~2.2 vs DER sleeve, **negative pre-2023**; legs are +0.5 correlated (not a hedge); it's a crowded, regime-dependent style trade. Home-bias / sub-sector / sector-neutral decompositions: congestion signal has ~0 return correlation; the spread "lives in the short leg"; US-vs-ex-US premium is grid-specific but a minor contributor | `docs/handoff_2026-09-01-grid-buildout-long-short.md` §1–§3 |
| RT/DA LMP spread cross-sectional split | 2026-09-01 | pivoted to FTR (failed); the RT/DA idea itself still literally untried | `docs/handoff_2026-08-15-rt-da-spread-signal.md` |
| Earnings-estimate-revision / analyst-breadth momentum | — | NOT ATTEMPTED — needs ≥3yr point-in-time consensus EPS feed | `docs/handoff_2026-09-01-grid-buildout-long-short.md` §8.2 |

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
12. **Backlog *pricing-power language* text-mining** — count/score lead-time, price-escalation and
    backlog-margin language in 10-K/10-Q MD&A, per name per quarter. Distinct from backlog *size*
    (growth, surprise, coverage — all FAILED). Untried. (`handoff_2026-09-01…` §8.3b)
13. **Quality / balance-sheet weighting** of the basket (leverage × margin stability × FCF
    conversion) vs equal-weight, with a residual-vs-low-vol-factor control. Scoped, not run;
    low prior, ~half a day. (`handoff_2026-09-01…` §8.4)
14. ~~Round-2 triage survivors — spike before build~~ **RESOLVED 2026-09-06, both DEAD:**
    (#7) labor-bottleneck data (BLS QCEW) — real, DC-specific wage/employment divergence
    confirmed, but no county-level revenue attribution exists for any public equity, and the
    national-aggregate margin-correlation version fails a pre/post-2022 sub-period split;
    one unresolved thread (IESC margin vs. DC-hub wage index, 4-qtr lag) not chased further.
    (#10) cat-bond/reinsurance pricing — zero reaction in SRRIX/ILS to the actual 2026-08-31
    EIX/PCG wildfire-liability crash; wrong peril (physical vs. legislative) entirely. See §6
    "2026-09-04 → 09-06" below and `docs/triage_2026-09-04-differentiation-ideas-round2.md`.
15. **State-level net-metering/DER policy dockets as a short-side entry trigger** (e.g. CA NEM 3.0 —
    dated, public, telegraphed months before the final PUC vote). Untried — everything tried so far
    is a long-side physical/price signal; nothing has used a regulatory-calendar input. (session
    2026-09-02, §6 below)
16. **Hyperscaler capex growth as an entry accelerator**, not the already-failed exit de-risk use
    of the same data (`capex_signal.py` failed as a de-risk trigger — fires ~a year late; never
    tried as "capex reaccelerating past a threshold" to size *into* the spread). Untried. (session
    2026-09-02, §6 below) — **this is the "load surprise done right"**: the demand-data classes all
    failed as *timing* because they lag a 2–4-yr-backlog business, but an *entry accelerator* is
    more tolerant of lag (you're confirming a trend, not calling a turn).
17. **Solar rotation when real yields fall** — a small conditional LONG in the beaten-down DER
    sleeve (ENPH/SEDG/CHPT/RUN/BLNK/STEM), sized 0→10% by "6-mo Δ 10y real yield < 0", on
    `screen +val+sleeve`. Tested 2026-09-06: full Sharpe 0.69→0.77, CAGR 14.4→16.8%, MaxDD
    −24→−27%, +7% in the feared solar-squeeze months, and it *rescues* the 2019–22 sub-window
    (Sharpe +0.51→+0.78). **Candidate, NOT wired** — trigger fires 42% of days (too loose), adds
    disclosed tail risk. Opposite sign of the rejected DER *short* (§3). (session 2026-09-06, §6)

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

---

## 6. Session contributions

Per-conversation log. Each chat appends a dated block here for what it did; the
substance goes in §1–§5, this is just attribution + a pointer.

### 2026-09-04 → 09-06 — Round-2 differentiation triage (10 ideas), #7/#10 spiked to conclusion, independent capex-guidance-signal verification

Brainstormed 10 new differentiation ideas outside the marquee grid-equipment basket
(crypto-miner-to-AI-hosting, wildfire liability, power semis, gas midstream, nuclear
PPA/restart, water utilities, EU grid equipment, demand-response/VPP, labor-bottleneck
data, cat-bond/reinsurance pricing), then live-checked each against September-2026
market state before any build. **8 of 10 dead on arrival** — every sector-adjacency
idea had already re-rated 30–140% YTD with explicit sell-side/retail coverage naming
the same thesis (Siemens Energy +138%, VICR +138%, TeraWulf's $19B Anthropic lease,
EIX −23%/PCG −18% same-day on the 2026-08-31 wildfire-liability bill). Full triage:
`docs/triage_2026-09-04-differentiation-ideas-round2.md`.

- **#7 (labor-bottleneck data) — spiked to a full negative, not just scoped.** Live
  BLS QCEW API (NAICS 2382, county-level, ~5–6 mo lag, free/public) confirmed a real,
  DC-specific wage/employment divergence (Loudoun VA electrical-contractor jobs
  8,672→18,027 2020–26; wage growth 16–18% YoY in DC-hub counties vs 6% national in
  2025Q4). But no county-level revenue attribution exists for any public equipment
  maker or contractor (EMCOR/IES/MYR Group disclose data-center revenue by *end-market
  segment*, not geography), so the tradeable version had to be reframed as a national
  aggregate wage-index vs. EME/IESC/MYRG operating margin. That correlation looked real
  (up to r=+0.52, p=0.002) but **did not survive a pre/post-2022 sub-period split or
  first-differencing** — a shared-secular-uptrend artifact, same failure shape as the
  big-4 capex-deceleration signal. Rebuilding on the actual DC-hub counties (not the
  diluted national number) confirmed the underlying divergence is real (DC-hub-minus-
  national spread 0.27pp pre-2022 → 1.88pp post-2022) but only one of 24 tested
  margin-correlation cells (IESC, 4-quarter lag, t≈4.2/2.7) stayed sign-stable across
  the sub-period split — flagged as the single most credible unresolved thread, not
  validated. Tested directly against the grid-equipment basket's own returns too
  (bypassing the contractor-margin channel entirely): no correlation at any lag
  (n=17 quarters). **Verdict: DEAD as a signal** — the wage divergence is real but
  untradeable in every channel tested.
- **#10 (cat-bond/reinsurance pricing) — spiked to a clean kill.** SRRIX (12+ yr daily
  NAV) and the Brookmont cat-bond ETF (ILS) both showed **zero reaction** to the
  2026-08-31 EIX/PCG wildfire-liability crash, and SRRIX moved *after*, not before,
  the Jan-2025 LA wildfire ignition. Cat bonds trigger on physical wildfire loss; the
  risk actually repricing CA utility equity is legislative/liability-cap risk — a
  different peril entirely with no cross-exposure. **Verdict: DEAD, cleanly** — not a
  data problem, a mechanism mismatch.
- **Independent verification of the capex-guidance signal (Deliverable D) — corroborates
  the FAIL, adds one flagged-and-fixed bug.** Using the data layer directly (built by a
  parallel session this same window), ran the pre-registered §5 methodology
  (HAC/Newey-West rank-IC + Δ10y/SMH control regression) on the primary $ series: **no
  horizon clears both bars** (1mo t=−0.68/−0.61, 3mo t=−1.28/−2.00, 6mo t=−0.96/−1.59)
  — independently consistent with the master ledger's own "best −1.54" FAIL verdict
  below, and every coefficient tested negative, matching the "consistent inverse sign"
  note. Also ran the disclosure-date event study proposed as D's cheap pre-check: no
  abnormal basket reaction at day-0/+1/+5 around any of the 35 real capex-guidance
  revision dates (benchmarked against like-for-like rolling windows, not naive
  daily-vs-multiday) — a genuine null, consistent with (not proof of) the information
  living in the aggregate rather than any single disclosure. **Found and reported a
  real bug** in `capex_guidance_signal.py`'s `guidance_composite` (`.fillna(daily)`
  silently substituted raw dollar values for a legitimate NaN warmup period, corrupting
  any downstream rank-IC) — fixed by the parallel build in `c2e1444` before this
  session's own corrected re-test.

### 2026-08-31 → 09-06 — Proposals B/D/A triage, transmission-rate-base build, Layer-1 overlay

- **Triaged proposals B / D / A** (`docs/triage_2026-08-31-proposals-bda.md`): B
  recommended and built; **A killed at triage** (zone-matched congestion pair —
  loser-side universe too thin, below the breadth floor); **D built then failed**.
- **Built `grid_demand_factor/`** (proposal D — rank stocks by return-β to a
  grid-demand nowcast). Pre-registered gate **FAILED**: rank-IC −0.02 (t −0.47),
  Q5−Q1 Sharpe 0.12, non-monotone, 6/6 variants fail; the nowcast is ~orthogonal
  to equity returns. Negative result, no module shipped as a strategy.
- **Built `transmission_rate_base/`** (proposal B — long fast FERC-transmission-
  rate-base compounders / short flat, ~34 regulated utilities, full 14-task
  gated build). Pre-registered gate **FAILED**: the five signal quintiles all
  earned ~the utility-sector return; Q5−Q1 Sharpe −0.61, rank-IC −0.018 (t −0.35),
  residual α −0.6 %/yr. Merged to `main` (`a1c4cb2`). FERC Form 1 extraction /
  filer→parent-ticker map / HAC-OLS additivity / pre-registered-gate code kept as
  reusable infra. `docs/transmission-rate-base-results.md`.
- **Merged** the earlier `backlog-surprise-factor` and `strategy-triage-bda`
  branches into `main`.
- **Built `grid_equipment_basket/overlay.py`** — Layer-1 risk overlay (trend gate
  MA-100 + 20 % vol target, month-held; `--overlay` CLI; `OVERLAY_*` config; 10
  tests). Primary 2023–26: Sharpe 1.36→1.58, MaxDD −41 %→−18 % (plateau, not
  knife-edge). **Prior 2020–22: HURTS** (Sharpe 0.69→0.20) — overfits the one
  V-shaped DeepSeek drawdown. Merged to `main` (`2949d9a`).
- **Wrote `docs/handoff_2026-08-31-two-layer-overlay.md`** — scoped the Layer-2
  grid-congestion regime brainstorm, incl. the feasibility check that killed
  interconnection-queue *velocity* as backtestable (only ~11 monthly ERCOT
  snapshots existed at the time).
- 2026-09-06: reviewed repo state for a consolidated results doc; found this log
  already established by parallel sessions, so contributed here rather than
  spawning another. Left the L2-overlay memory corrected (adoption reverted
  2026-09-01, `REGIME_ENABLED=False`).

### 2026-08-31 → 09-01 — Layer-2 grid-congestion regime: full build, gate, revert; options A/B; grid-buildout L/S proposal

Picked up `docs/handoff_2026-08-31-two-layer-overlay.md` and built Layer-2 end to end.

- **Built `grid_equipment_basket/grid_regime.py`** — abs `|congestion $|` + reserve-tightness in
  DC-heavy PJM zones (DOM/AEP/COMED/PPL), trailing-756d z-score → monthly exposure multiplier;
  a **frozen 6-rung pre-registered ladder** (`config.REGIME_LADDER`), the gate (beat layer-1-only
  on Sharpe **and** Calmar on both windows, on a plateau), `--overlay-l2` CLI, ~55 tests. Merged
  to `main`. Spec `docs/superpowers/specs/2026-08-31-grid-regime-layer2-design.md`; results
  `docs/grid-regime-layer2-results.md` (§1–§9).
- **Gate result: no rung passed** — every rung beats layer-1 on Sharpe but not primary-window
  Calmar (2.16 vs 2.32). Rung 1 was **adopted by PM override** (`REGIME_ENABLED=True`) on a
  generalisation + physical-differentiator argument, then **reverted 2026-09-01** on the OOS
  evidence (`REGIME_ENABLED=False`).
- **Option A** (DC-minus-rest-of-PJM relative congestion) — FAILED; proved the absolute signal's
  prior-window "win" was the COVID demand-collapse coincidence.
- **Option B** (signal-tilted long-basket / short-rest-of-PJM-utilities pair) — FAILED the
  spread-informative check (signal backwards).
- **Big-4 capex-deceleration de-risk trigger** (`grid_equipment_basket/capex_signal.py`, TDD) —
  FAILED; fires ~a year late, missed the DeepSeek drawdown entirely.
- **Fetched PJM zone LMP Jan–Aug 2026** for a genuine OOS test; congestion overlay did not beat
  the plain price-gate OOS.
- **`docs/triage_2026-08-31-differentiation-ideas.md`** — sell-the-overlay-as-insurance (FAILED),
  un-crowd-the-universe (FAILED), interconnection-queue velocity (NOT TESTABLE).
- **`docs/handoff_2026-09-01-grid-buildout-long-short.md`** — the "Grid Buildout, GRID-hedged" L/S
  proposal + the full prior record + §8 ordered next-conversation work plan; home-bias, sub-sector
  and sector-neutral decompositions showing the congestion signal has ~0 return correlation and the
  spread's return lives in the short leg. Also a 1-page tear-sheet artifact
  (claude.ai/code/artifact/0e5f0591-94b0-48b0-a884-24097940f1e4 — stale, pre-OOS).

### 2026-09-02 → 09-06 — classifier generalization, spread attribution, capacity-auction & book-to-bill probes, DC-commitment-signal scoping

The long conversation that produced the `capex-cycle-pair-classifier` (memory) and the
transmission-project-filings handoff. All negative except the classifier and the handoff scope.

- **Classifier — generalization test.** Applied the "customer + 2 of 3" rule to 6 other themes.
  **EV supply chain** works post-mania (β-hedged+vol-tgt Sharpe ~0.7, 2022–26; −0.7 in the
  2020–21 EV mania). Hydrogen (2000 & 2020), genomics ≈ 0; **nuclear/SMR inverts** (loses 2024–26 —
  still mid-mania); **space & cannabis self-reject** (no institutional-supply sleeve — RKLB / SMG
  are financially indistinguishable from the shorts). Precondition: the institutional side must be
  in a *real* capex up-cycle — only grid/DER (utility + hyperscaler spend) and, weakly, EV clear it.
- **Multi-pool concurrent book.** grid/DER + EV run together (equal risk, vol-tgt): blended
  Sharpe ~1.69 / MaxDD −9 % (2022-06 → 2026-08); pairwise pair-return corr 0.1–0.4 (genuinely
  diversifying) — **but** adding hydrogen / eVTOL / battery-tech (Sharpe 0.1 / 0.2 / −0.6) only
  dilutes. "2 live pairs, not 5."
- **grid/DER spread attribution.** Regressed the dollar-neutral spread on market + quality
  (COWZ−SPY) + duration (TLT) + the PAVE−TAN sector pair. **Not** quality (loads −1), **not**
  duration (≈ 0), **not** momentum. ~40 % is the PAVE/TAN infra-vs-solar sector rotation (R² 0.41);
  residual "alpha" +53 %/yr (t 2.6 primary, 1.6 full-sample) is one-regime thematic concentration,
  not a factor. **PAVE/XLI itself is Sharpe −0.51 over 2023–26** — the ETF version of the trade
  loses money; the edge is entirely the concentrated name selection, on one macro cycle. Raw
  unhedged dollar-neutral Sharpe ≈ 1.2–1.4; the ~2.2 in earlier handoffs needed the β-hedge +
  20 % vol-target overlay and one window.
- **US-industrial-policy L/S variant** (long ~46 policy-capex suppliers across 7 buckets — AI/DC,
  semicap, aero/defense, reshoring E&C, water, grid, thermal — / short ~13 matched non-policy
  quality industrials). β-hedged+vol-tgt Sharpe **0.81** (2023–26), **0.18** (2021–22), **0.45**
  full; trimmed ~35-name version ~0.95. One disclosed macro bet (US industrial policy persists),
  regime-dependent, PAVE/XLI benchmark negative. NOT SHIPPED.
- **Non-price name selection — dead end.** No external dataset mechanically reproduces the
  17-name book. ISO/utility planning data (PJM MW forecast Table B-9, RTEP $11.6 B/window, FERC
  Form 1) names the *customers* (zones, utilities, TOs), not the suppliers. Patents (H02J / H01F /
  H02B) skew to SOEs / Asian conglomerates and miss every EPC contractor; PatentsView's free API
  is decommissioned. The book is a fundamental-character selection — the 4-criterion screen is
  what reproduces it.
- **PJM capacity-auction (BRA) clearing prices as timing — FAILED.** Lags the equity 18–24 months
  (the 10× July-2024 print came after VRT was already +1,500 %), inverted (the play worked best
  when prices were at their $29 low, Dec 2022), and there is **no exit signal** — 2026/27 cleared
  $329, higher. Lagging confirmation, not a trigger. Possible *late*-exit rule: "first BRA to
  clear materially lower YoY" — untested, no AI-era instance.
- **Book-to-bill scaler — INFEASIBLE from XBRL.** Only the 3 EPC contractors (PWR / MYRG / PRIM)
  have a clean deep `RevenueRemainingPerformanceObligation` series; ETN stopped tagging 2024-03,
  VRT / HUBB / NVT never did, GEV starts 2024. Derived `(ΔRPO + rev) / rev` is too noisy to gate
  on (PRIM 0.49 → 1.89 q/q). Equipment makers report book-to-bill in MD&A prose / calls only → a
  multi-quarter hand-collection project, not a build.
- **Retrieved** 6 PJM Load Forecast Table B-9 vintages (2021–2026, PDF + xlsx) →
  `grid_resilience/data/seed/pjm_large_load_b9_vintages.csv`; write-up
  `docs/pjm-large-load-vintages-2026-09-03.md`. Fed the MW-revision (FAILED) and non-price-selection
  work above.
- **Scoped** `docs/handoff_2026-09-03-transmission-project-filings.md` — prelim investigation of
  FERC eLibrary + state PUC filings (they name project scope / cost / driver, **not** the EPC
  contractor or equipment vendor) and **Part 2 (§6)**: data-center *commitment* signals —
  Deliverable C (PUC DC tariff/ESA contracted-MW), D (utility capex-guidance — since built +
  FAILED by a parallel session), E (county DC permit filings) — with pre-registered bars and a
  ~17-attempt stop rule. Plus the §2 weighting-note refinements (zone tilt works for EPCs, not
  equipment makers; work-type → supplier map, needs Order 1000 proposals not final orders).
- **Thematic-catalyst detector — tested.** The process (watch utility load-forecast filings for a
  newly-named large-load category) would have surfaced the *theme* ~7 months before the 2023
  equity re-rating, but not the specific supplier names (needs a second mapping step) and **not
  the DER short at all** (that is a separate rates/hedge decision). n = 1, idea-generation only.

### 2026-09-02 — grid-buildout L/S: DER-sleeve-short verification, by-year performance, ex-ante-trigger brainstorm

Independently verified the handoff's DER-sleeve-short number and asked whether the trade could
have been found systematically rather than picked in hindsight. Built the FTR signal (#12),
backlog-coverage-alone tilt (#13), and the thematic-catalyst-detector finding (above) in the same
session — see their own ledger rows/docs.

- **Reconstructed the DER-sleeve-short spread independently** (long US+foreign buildout, short
  ENPH/SEDG/CHPT/RUN/BLNK/STEM, rolling-beta hedge, 20% vol target): Sharpe 2.28 (2023-01→2026-08,
  matches the handoff's 2.23), 1.10 full 2020-26. By year: 2023 +111.8%/3.36, 2024 +97.0%/2.77,
  2025 +34.1%/1.25, 2026 YTD +22.5%/1.38 — the only variant still positive in 2026 (the GRID-short
  version is −6.3%/−0.55 YTD).
- **Proposed a "financing-quality/duration divergence" narrative** for why it worked (programmatic
  hyperscaler/utility capex vs. rate-sensitive consumer-financed DER, amplified by the 2022-23
  hiking cycle + CA NEM 3.0). **Correction: superseded by the more rigorous spread-attribution
  regression above** (this same §6 block, 2026-09-02→09-06 session) — neither quality nor duration
  load on the spread; ~40% is PAVE/TAN sector rotation, the residual is one-regime concentration,
  "not a factor." Log the financing-quality story as a narrative that didn't survive a formal
  test, not as an established mechanism — don't re-cite it as the reason this worked.
- **Ex-ante systematic-trigger brainstorm** (could this have been found mechanically?): (1)
  thematic-catalyst detector — tested, see above; (2) open items #15–16 below (DER policy-docket
  short trigger; capex-acceleration entry trigger) — both untried; (3) ruled out — real rates
  alone, too coarse (flags "short duration growth" broadly, not DER specifically).
- **Robustness fix**: `_pjm_get` (`grid_resilience/data/grid_data.py`) retried on HTTP 429 but not
  on a transient read-timeout, which crashed a 96-month sequential FTR backfill partway through.
  Now retries both — `tests/data/test_pjm_retry.py`.

### 2026-09-06 — `screen` supplier construction, backlash tilt, hedge/rotation re-tests, behaviour figure

Parallel electrification-strategy conversation. Built the `screen` universe, re-tested the
hedge/rotation menu against it, and produced the v2 proposal + a behaviour figure. Substance in
§1 / §3 / §4; this is attribution.

- **`screen` construction — the current ship.** 4th universe = the `capex-cycle-pair-classifier`
  rule (institutional customer + ≥2 of {profitable, earnings-valued, low-policy-dependence}),
  supplier side only, **no ETF gate** → 27 names, 0% utilities/pipelines/components. Rebuilt
  `etf_membership_2026.csv` from real published holdings (VOLT ~30% utilities / 10% midstream /
  13% components; `thematic` shrank 10→7, `frozen` 23→15; GRID/PAVE/ELFY top-25-visible only).
  Shipped `screen +val+sleeve` by PM override of the pre-registered `marquee plain` (games the
  raw-Sharpe rule via concentration + no protection). Beats VOLT on the fair window: CAGR 27.7
  vs 24.1, Sharpe 1.35 vs 0.82, MaxDD −17 vs −24. Merged to `main`. Package now 38 tests, 4
  constructions in the CLI + comparison grid.
- **Data-center-backlash hedge test** (`scratchpad/backlash_hedge.py`; no in-sample episode,
  proxy = 12 worst DC-power months with SPY flat/up): **grid-maintenance tilt ADOPT** (70/30
  grid-maint/DC-power, net-neutral headline, β 0.36 to the DC-power sleeve, +0.4% vs DC-power
  −7.2%) → open item #2; **conditional merchant-power short REJECT**.
- **Rotation re-tests:** rate-conditional rotation INTO utilities (XLU) — REJECT; DER sleeve
  SHORT re-tested on `screen` — anti-hedge, worse at every size; DER sleeve LONG when real yields
  fall (0→10%) — mild PASS → open item #17. (§3 table rows added.)
- **Behaviour figure** `docs/electrification-strategy-exposure.png` (embedded in the proposal
  HTML) — growth / rolling 1-yr Sharpe (+ SPY, VOLT context) / net supplier-basket weight
  (trimmed <0.85 baseline ~73% of days; valuation rule active ~42%; deepest trim 0.14 Apr-2020).
- **v2 proposal** `docs/electrification-strategy-proposal-v2.{md,html}` (3-tier: like-I'm-5 / -10
  / new-grad) + Artifact `claude.ai/code/artifact/c5db7bcc-11aa-41e5-9048-161308ecb3a7`.
- **Demand-signal follow-ups:** open items **#15** (regulatory-calendar DER short trigger — the
  only untried signal *class*) and **#16** (hyperscaler capex growth as an *entry* accelerator,
  not the failed lagging *exit* de-risk) both stand for whoever picks this up next.
