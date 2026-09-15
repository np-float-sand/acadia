# Research log — AI-power / electrification strategy

**Single source of truth for this line of research.** Every conversation appends to / updates
**this file** rather than spawning another standalone doc. Detailed write-ups live in the linked
docs; this is the index + verdicts + open items.

**Update convention:** add new threads to the ledger (§3) with a one-line verdict + a link to the
detail doc; keep §1 (current strategy) and §4 (open items) current; date every change.

Last updated: **2026-09-09**.

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
  **Proposal v3** (2026-09-07, `docs/electrification-strategy-proposal-v3.html`, Artifact
  `claude.ai/code/artifact/829dc90b-02a4-4caa-8abb-9eb8eed3c198`): the investment-committee
  write-up — "Screened Electrification Equipment", VOLT/GRID/SPY comparison + `SPY+15% GLD/IEF`
  benchmark, a systematic-risk/macro-bet table, and the full 9-candidate short-offset analysis
  concluding **no equity short improves the book** (drawdown control = the built-in overlays +
  an optional cash/T-bill weight; a 25% T-bill weight matches a `−XLI 0.25×` overlay at a higher
  Sharpe). Reproducible build kit: `research/electricity_earnings_exposure/proposal_v3/`.
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

Across **~28 signal attempts** (this project + predecessors), **no tradable timing or
cross-sectional edge has survived an honest out-of-sample / regime split.** Three independent
demand-data classes — forecast (PJM load forecasts), physical-flow (congestion, FTR), and
commitment (utility capex guidance) — have each failed. A fourth framing — **electricity as a
company-specific economic input** (physical power-cost/-revenue exposure → company earnings →
equity mispricing), the "move one layer closer to the market" pivot — also failed its own
pre-registered gate 1 (2026-09-07): a bottom-up fleet×price model tracks merchant-generator
*revenue* but not *operating income* (hedges + non-gen segments sever the link), and a
hand-mapped power-cost-exposure signal across ~18 power-intensive industrials leaves no
fingerprint on gross margin and no factor-neutralised forward-return predictability. The defensible product is the
**rules-based long-only book + risk overlays**, pitched as disciplined thematic beta, plus the
**`capex-cycle-pair-classifier` construction rule** (the one constructive output: it mechanically
reproduces the hand-picked book, removing the "you hand-picked winners" objection).

**One partial exception, watch-list only (2026-09-08):** the narrative-mispricing thread's
follow-on — a GBM over the *combination* of merchant stress-beta + ICR + EDGAR disclosure counts
+ momentum — is the first attempt in the ~28 to clear a shuffled-label null cleanly and survive a
merchant-confound exclusion (Fold B, train 2024-25 / test 2026, OOS AUC 0.758). It still does
**not** survive a regime split (flat pre-2023) and rests on 6 months / ~50 observations, so it is
**not tradable** — logged as §4 item 18 to re-check quarterly, not a change to the conclusion above.
`docs/handoff-narrative-mispricing-portal.md`. The **returns** expression of that model was then
built and tested to a verdict (2026-09-08 → 09-09, "**Dispatch**", `docs/handoff-2026-09-08-dispatch-ls.md`):
as a dollar-neutral cross-sectional L/S it **FAILED the regime split** — even after pruning the
dead features (`gdelt_z`, stale `icr_z`) and widening the universe 13 → 19 names, ~100% of the
Sharpe is the 2024 merchant-vs-regulated melt-up (ex-2024 Sharpe −0.05), and a sector-neutral /
regulated-only cut leaves only a trivial non-fundable signal. So the watch-list exception is
**narrower than it looked**: it survives only as a tercile-*classification*-AUC curiosity on
2026 (§4 item 18, re-check quarterly), not as a tradable book. The one durable output is the
**point-in-time trailing-net-margin screen**, kept as a quality overlay (it lifted the worst
year of every Dispatch variant and of the shipped long-only book).

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
| STEO archive feasibility, reframed as merchant-beta stabilizer | 2026-09-08 | **SCOPED, NOT RUN (corrected)** — PUDL does NOT shortcut this: its actual processed database (421 tables, live-checked) has zero STEO/ELWHU tables; the promising-looking Zenodo record is just a raw-file mirror of the same PDFs. Back to the original ~half-day PDF-extraction estimate. | `docs/steo-archive-feasibility-results.md` |
| Backlog-surprise factor (1,778 events) | 2026-08-31 | FAILED — event-study gate FAILED | `docs/backlog-surprise-factor-results.md` |
| PJM capacity-auction (BRA) clearing prices | 2026-09-03 | FAILED — lags equity 18–24 mo, inverted (best at the $29 low), no exit signal | RESEARCH-LOG §6 (2026-09-02→09-06) |
| ERCOT CRR true clearing prices — feasibility check | 2026-09-08 | **FAILED (feasibility)** — live API exposes only a rolling ~12-month window (confirmed, params don't unlock more), no public bulk archive; needs a special ERCOT data request. | `docs/ercot-crr-feasibility-results.md` |
| Real-time ancillary/scarcity pricing — first honest timing-IC check | 2026-09-08 | **IN PROGRESS** — feasibility PASS (PJM) unchanged; a naive daily-overlapping-window IC looked strong (t up to 4.61) but was an autocorrelation artifact — properly corrected (monthly, HAC), no signal on 2018-2021 (t<1.1). n=44 months thin; 2022-2025 not yet pulled (rate-limited). | `docs/ancillary-scarcity-pricing-feasibility-results.md` |
| Narrative-extremity short (Approach A: EDGAR filing mentions + GDELT news volume composite) | 2026-09-08 | **NOT ADOPTED** — refined both data sources (broader EDGAR keywords incl. "large load"; GDELT self-baseline z-score) which cleanly fixed VST's under-narration measurement gap (0.996-corr-style artifact analog) but rank-IC stayed insignificant at every horizon (best |t|=1.40, 1mo, wrong sign for the thesis). Better data ruled out "just bad measurement." Handoff written for Approach B. | `docs/narrative-mispricing-approach-a-results.md`, `docs/handoff-narrative-mispricing-approach-b.md` |
| Narrative-mispricing Approach C (broadened: narrative + every alignable RESEARCH-LOG feature, walk-forward GBM/RF) | 2026-09-08 | **FAILED** (narrative-as-signal) — pre-registered both-windows bar (\|t\|≥2 + perm p<0.05 in FULL 2018-25 *and* RECENT 2023-25) fails outright: RECENT window clears the bar in several cells (t up to 3.73) but FULL window is flat-to-negative everywhere (\|t\|<1, perm p 0.20-0.92) — the exact Henry Hub GBM overfit shape recurring. Narrative composite never ranks top-3 in feature importance. **Follow-on (two-fold walk-forward, Fold B: train 2024-25 / test 2026): PROMISING, NOT ACTIONABLE** — the *combination* of merchant stress-beta + ICR + EDGAR disclosure counts + momentum gets OOS AUC 0.758 (0.719 regulated-only), sits at the 100th pct of a shuffled-label null, is seed-stable and not one-month-driven, and survives dropping the VST/CEG/TLN/NRG merchant confound — the first result in this line (~28 attempts) to clear a permutation-null bar this cleanly. Still n=48-54 / 6 months / one split, flat pre-2023, GDELT stale in-window → a lead to re-check quarterly, not a signal to wire. Driver is stress-beta (item 3), **not** the narrative/attention data. | `docs/narrative-mispricing-approach-c-results.md`, `docs/ml-two-fold-walkforward-results.md`, `docs/handoff-narrative-mispricing-portal.md` |
| Book-to-bill scaler (sleeve-aggregate order book) | 2026-09-06 | INFEASIBLE from XBRL — only PWR/MYRG/PRIM have usable RPO; makers report it in MD&A prose only | RESEARCH-LOG §6 (2026-09-02→09-06) |
| **Regulatory-calendar DER short trigger** (open item #15 — short resi-solar in windows around scheduled adverse net-metering decisions) | 2026-09-06 | **FAILED** — first test of a regulatory-calendar signal *class*. Event study over 6 adverse milestones: mean DER return **+0.0% vs the book** (major-only −10.9%, but that is the one CA NEM 3.0 cluster; the Dec-2021 *proposal* milestone is +8.7% — short into a rip). Overlay on `screen +val+sleeve`: +0.04 Sharpe but MaxDD deepens 4–7pp, feared-scenario P&L −7% to −18%, timed-short sleeve −1.9%/yr standalone. n≈1 event; telegraphed policy already in the price. `nem_calendar.csv` kept as a reference for the discretionary risk monitor (open item #5), not wired. | `docs/nem-calendar-der-short-results.md` |

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
| **Merchant-generation margin nowcast → 4-name L/S** (Probe A; VST/NRG/CEG/TLN) | 2026-09-07 | **FAILED at the retrodiction gate** — fleet×realised-ISO-price model tracks Δrevenue (pooled Spearman +0.45) but Δoperating-income is +−0 (−0.05, sign-agree 48%; CEG −0.46). Hedge MtM + retail/non-gen segments sever the realised-price→earnings link. A2 returns test not run. | `docs/electricity-earnings-exposure-probe-results.md` |
| **Electricity-cost exposure cross-section** (Probe B; ~18 power-intensive industrials — Al / EAF steel / chlor-alkali / industrial gas / BTC miners) | 2026-09-07 | **FAILED the pre-registered 6-gate tree** — hand-mapped `−intensity·%Δ(state or hub power price)` signal: G1 Δgross-margin IC≈0 (t −0.6 / −0.2), G1 Δop-margin predictive t −1.9 (wrong sign; only *contemporaneous* t +2.2, an accounting identity), G2 fwd 1m/3m IC≈0, G3 FF5+MOM+sector-neutralised coef t +1.7 → +1.0. Only breadth (G4) passes. 2022–26 fwd-3m blip (t 1.96) collapses to t 0.7 on the wholesale-price proxy → proxy noise. | `docs/electricity-earnings-exposure-probe-results.md` |
| **"Dispatch"** — Fold-B walk-forward turned into a dollar-neutral cross-sectional L/S (GBM-ranked, monthly, long-top/short-bottom tercile) + 2026-drag diagnosis + point-in-time profitability screen | 2026-09-08 → 09-09 | **TESTED, FAILED the regime split.** (1) *Drag diagnosis:* the walk-forward L/S bleeds in 2026 (−5.1% partial-yr) entirely on the **long leg, mostly TLN** (−4.3%/mo realised in 2026, held long 9 of 12 months); the model is momentum-dominated in practice (`mom_z` imp ≈0.27 > the stress-beta/ICR/EDGAR trio Fold B is credited to), and two inputs are broken — `icr_z` for TLN is frozen at a **2016** value + ffilled, `gdelt_z` frozen at ~2025-09. (2) *Fix that works (illustrative, thin n):* adding a point-in-time EDGAR **bottom-tercile trailing-net-margin screen to the long leg** (TLN is loss-making by Q4-25: TTM NI −$219M, margin −8.5%, worst 1/13) flips the walk-forward L/S from **+2.1%/yr Sharpe 0.22 MaxDD −16%** to **+9.5%/yr Sharpe 0.74 MaxDD −13%** (2026 −5%→+7%); "no negative TTM NI" is weaker (+3.5%/yr), a crude min-operating-history cut *backfires* (−5%/yr). (3) NOT ADOPTED — long-only XLI/PAVE/VOLT still beat it on raw return (+21/+23/+30 %/yr), universe collapses to 10–13 effective names / 3–4 per side / 42 months. Named, build plan + pre-registered gates written. (4) *"Combine with the narrative-mispricing thread for breadth?" — NO:* Dispatch *is* that thread's Fold-B model as a book (same panel/universe/features). Ran `spike_c_2026_holdout.py` (the thread's widest framing — 25 names, 9 features, 2018-start, continuous rank-IC, train≤2025-12/test-2026 holdout): 2023-25 "strong" cells decay to \|t\|≈1.4 on +2026, **2026 holdout flat-to-negative every cell** (RF 3-month t −3.77 / −2.12, inverted OOS), `narrative_z` importance rank 7–8/9, `own_mom6m` dominates. Same momentum-loading failure on a wider universe + different framing → no breadth of edge from merging; breadth must come from *pruning* dead features (`gdelt_z`, stale `icr_z`) to widen the universe + extend to 2018, plus the profitability screen. (5) *Pruned v2 + sector-neutral v3 (2026-09-09):* pruned `gdelt_z`/`xlu_xng_spread`/`anc_composite`, rebuilt `icr_z` point-in-time from EDGAR (100% cov, frozen-2016-TLN bug fixed), widened `stress_beta_z` via union of the two rolling-beta panels → **breadth 10–13 → 16–19 names/mo**. Headline Sharpe jumps (GBM clf 0.22→0.77) **but it is ~100% the 2024 merchant-vs-regulated melt-up**: ex-2024 (2023+2025+2026) Sharpe **−0.05** (classifier) / +0.07 (regressor); merchant−regulated spread was +68/+127/+35/−27% across 2023–26. Regulated-only leaves a trivial signal (GBM clf ex-2024 Sharpe +0.19, +0.36 with the margin screen, but 2023 still negative). **FAILED the regime split — not a standalone cross-sectional strategy** (~29th tracked attempt, same outcome). Kept: the point-in-time net-margin screen as a quality overlay (lifted the worst year of every variant incl. the shipped long-only book) + the pruned/widened point-in-time panel as reusable infra. | `docs/handoff-2026-09-08-dispatch-ls.md`, `docs/ml-two-fold-walkforward-results.md` |

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
| RT/DA LMP spread — live feasibility check | 2026-09-08 | **ERCOT PASS** (RT already the production fetch, DA is one cheap `get_dam_spp` call away) — **PJM FAIL pre-2024** (full nodal download, 429s within 6 pages even for 1 day; fine for recent dates). VST/NRG (the actual problem names) are both ERCOT. | `docs/rt-da-spread-feasibility-results.md` |
| Earnings-estimate-revision / analyst-breadth momentum | — | NOT ATTEMPTED — needs ≥3yr point-in-time consensus EPS feed | `docs/handoff_2026-09-01-grid-buildout-long-short.md` §8.2 |
| Congestion-spread Phase 1 validation + Phase 2 go/no-go | 2026-09-08 | **NOT RECOMMENDED (Phase 2)** — offline check vs production caches: the "NaN for all ISOs" diagnosis was only ever true for ERCOT (100%→0% NaN after the fix, real signal); PJM's `congestion` column was never broken (0% NaN pre-fix, spread-fill changes 0/2,922 days). No live bug left to fix; GSI's problem is signal strength, not construction. | `docs/untried-tasks-log.md` §1 |
| Interconnection-queue-velocity capture — process check | 2026-09-08 | **NOT RUNNING** — open item #10's "keep running" has no automation behind it (no cron/workflow, one static cached snapshot). Flagged for a build-or-abandon decision, not silently fixed. | `docs/untried-tasks-log.md` §8 |
| ICR retest via EDGAR XBRL (full 25-name universe, 63 median quarters) | 2026-09-08 | **FEASIBILITY PASS, HYPOTHESIS FAIL** — full-sample rank-IC t=-0.95 (wrong sign); the two big-looking 3-yr-window correlations trace to PCG/EIX wildfire-liability and VST/CEG AI-re-rating confounds, not leverage. `USE_ICR` stays off. | `docs/icr-edgar-retest-results.md` |
| Merchant stress-beta orthogonalization, fix #2 (trailing power-price-trend regressor) | 2026-09-08 | **FAILED** — ran the real pipeline via monkeypatch; VST's orthogonalized beta correlates 0.996 with baseline (barely moves), IC@21d/63d both get slightly worse. Trailing realized price is the wrong proxy for the forward-looking narrative repricing; redirects to item 6 (STEO). | `docs/merchant-beta-orthogonalization-results.md` |
| Merchant stress-beta orthogonalization, fix #1 (XLU/XNG correlation exclusion filter) | 2026-09-08 | **NOT ADOPTED** — works exactly as diagnosed (100% precision isolating NRG/VST/CEG/TLN, zero false positives), stress-period Sharpe -0.145→-0.085, but too small to beat XLU. Best of the constructions tried. | `docs/merchant-beta-orthogonalization-results.md` |
| Merchant stress-beta orthogonalization, fix #4 (exponentially-weighted/structural-break-adaptive window) | 2026-09-08 | **FAILED, decisively** — full Sharpe -0.007→-0.448, IC worse at every horizon. Recency-weighting an already-thin stress-day sample trades bias for ruinous variance. | `docs/merchant-beta-orthogonalization-results.md` |

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
15. ~~State-level net-metering/DER policy dockets as a short-side entry trigger~~ — **TESTED
    2026-09-06, FAILED.** First test of a regulatory-calendar signal class. Event study: no
    consistent sign (mean DER +0.0% vs the book across 6 adverse milestones; the Dec-2021 NEM 3.0
    *proposal* is +8.7% — short into a rip). Overlay: +0.04 Sharpe but deepens MaxDD 4–7pp,
    feared-scenario P&L negative, timed short −1.9%/yr standalone. n≈1 event (CA NEM 3.0);
    telegraphed policy is already priced. `electrification_strategy/data/nem_calendar.csv` kept
    as a reference for the discretionary risk monitor (#5). Revisit only with more events
    (~2027–28 state NEM-review wave) + a tight vote→effective window + framed as a scale-down of
    the #17 long-solar tilt, not an outright short. `docs/nem-calendar-der-short-results.md`.
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
18. **Narrative-mispricing Fold B re-check (PROMISING, NOT ACTIONABLE)** — the stress-beta + ICR +
    EDGAR-disclosure-count + momentum *combination* is the first thing in this line (~28 attempts) to
    clear a shuffled-label null cleanly (Fold B, train 2024-25 / test 2026: OOS AUC 0.758, 0.719
    regulated-only; seed-stable; survives the VST/CEG/TLN/NRG merchant-confound exclusion). Blocked
    on sample size (n≈48-54, 6 months, one split; flat pre-2023). Cheap follow-ups: (a) re-run the
    exact Fold B quarterly as 2026 realized 3-mo-forward outcomes accrue — not done; (b) isolate the
    3-feature stress-beta×ICR×EDGAR interaction (drop the other 4) — **done 2026-09-08, mixed**: on
    the full 23-name universe the isolated 3-feature model is much weaker than the 7-feature one
    (AUC 0.567, only 78th-pct vs. its null — the drop-one ablation understated how much the other 4
    features, individually "noise," matter in combination); on the 19 regulated-only names the
    3-feature model holds up close to the full model (AUC 0.645 vs 0.670, both ~96-98th pct, though
    one test month — 2026-06, AUC 1.00 — is doing a lot of the work). **Revises the story**: this is
    not cleanly "3 features are the whole signal, 4 are noise" — the fuller feature set is doing real
    work, at least on the full universe. (c) fix the stale 2026 GDELT leg — not done. Do **not** wire
    live. `docs/handoff-narrative-mispricing-portal.md`, `docs/ml-two-fold-walkforward-results.md`;
    `research/narrative_mispricing/foldb_3feat_isolation.py`; `untried-tasks-log.md` item 9/10.
19. ~~**"Dispatch" — build the Fold-B model into a real cross-sectional L/S book**~~ **— TESTED → FAILED the regime split, 2026-09-09 (v3 below). Closed as a standalone strategy; margin-screen overlay + pruned point-in-time panel kept.**
    2026-09-08 diagnosis + counterfactual: the walk-forward L/S (GBM classifier, monthly, dollar-
    neutral, long-top/short-bottom tercile) loses in 2026 on the long leg (mostly TLN), the model
    is momentum-dominated, and `icr_z` (TLN frozen at a 2016 value) + `gdelt_z` (frozen ~2025-09)
    are broken inputs. Adding a **point-in-time EDGAR bottom-tercile trailing-net-margin screen to
    the long leg** takes the walk-forward from +2.1%/yr Sharpe 0.22 MaxDD −16% → +9.5%/yr Sharpe
    0.74 MaxDD −13% (illustrative only — 42 months, 10–13 effective names, 3–4/side). Build plan
    (`docs/handoff-2026-09-08-dispatch-ls.md`): (a) restore the universe to ~23 by fixing feature
    coverage; (b) formalise the profitability screen point-in-time (net / operating / FCF margin;
    continuous weight vs binary gate); (c) de-bias away from `mom_z` (ablation says dropping it
    *helps*), cap single-name / ISO concentration, add a turnover penalty; (d) test a 3-month hold;
    (e) risk overlays — vol-target ~8–10%, T-bill + GLD/IEF sleeve to a portfolio-vol target,
    reuse the REGIME gate; (f) costs (borrow/turnover/spread); (g) marginal Sharpe vs the shipped
    long-only book. **Pre-registered gates:** ≥20 names with complete features every month
    2019→now; net-of-cost Sharpe ≥0.6 full-sample AND ≥0.4 in each of ≥3 non-overlapping
    sub-windows; beats a 60/40 XLI/T-bill blend on Sharpe with corr-to-XLI ≤0.3; vol-targeted
    variant realised vol within ±20% of target and MaxDD > −15%; every fundamental input
    point-in-time. Reproduce from `research/narrative_mispricing/ml_panel_2022_2026.parquet` +
    `research/electricity_earnings_exposure/edgar.py`.
    *Breadth-by-merging with the narrative-mispricing thread — checked 2026-09-08, NO:* ran
    `spike_c_2026_holdout.py` (25 names / 9 features / 2018-start / continuous rank-IC / proper
    2026 holdout — the thread's widest framing). 2023-25 "strong" cells decay to \|t\|≈1.4 on
    +2026; **2026 holdout flat-to-negative in every cell** (RF 3-month t −3.77 / −2.12, inverted
    OOS); `narrative_z` importance 7–8/9; `own_mom6m` dominates. Same momentum-loading failure on
    a wider universe → breadth comes from *pruning* `gdelt_z` + stale `icr_z` (widens universe,
    extends panel to 2018) + the profitability screen, not from an ensemble of narrative framings.
    *Pruned v2 built + run 2026-09-08:* dropped `gdelt_z` / `xlu_xng_spread` / `anc_composite`
    (dead as cross-sectional rankers), rebuilt `icr_z` point-in-time from EDGAR (100% coverage,
    frozen-2016-TLN bug fixed), widened `stress_beta_z` via the union of the two rolling-beta
    panels → **usable names/month 10–13 → 16–19 from 2024 on**. Headline Sharpe jumps (GBM clf
    0.22→0.77, GBM reg 1.08) **but it is a 2024-only artifact** — every variant returns +24 % to
    +86 % in 2024 and negative-to-flat in 2023 / 2025 / (classifier) 2026; the merchant-minus-
    regulated spread was +68 / +127 / +35 / −27 % across 2023–26, i.e. Dispatch is a
    momentum-timed merchant-vs-regulated sector bet. **No cross-sectional edge survives the
    regime split.** Scratchpad `dispatch_pruned.py`; `docs/handoff-2026-09-08-dispatch-ls.md` §"v2".
    *Sector-neutral v3, run 2026-09-09 — FAILED, Dispatch closed:* stripping the merchant-vs-
    regulated axis kills it — full GBM classifier ex-2024 (2023+2025+2026) Sharpe **−0.05** (from
    +0.77), regressor +0.07 (from +0.87); i.e. ~100% of the Sharpe was the 2024 merchant-IPP
    melt-up. Regulated-only leaves a trivial, non-fundable signal (GBM clf ex-2024 Sharpe +0.19
    raw, +0.36 with the margin screen, but 2023 negative). Sector-neutral-combined and all
    regressor variants are noise outside 2024. **Kept:** the point-in-time trailing-net-margin
    screen as a *quality overlay* (improved the worst year of every variant incl. the shipped
    long-only book — candidate for §4 item 13 / the electrification book); the pruned/widened
    point-in-time panel (`[edgar_z, icr_z(EDGAR XBRL), stress_beta_z(union), mom_z]`, 16–19
    names/mo) as reusable infra. Fold B's AUC watch-item (item 18) is untouched — that is
    tercile-classification AUC, not this returns expression. Scratchpad
    `dispatch_sector_neutral.py`; `docs/handoff-2026-09-08-dispatch-ls.md` §"v3".
20. **Forward power price data (SOW Phase 1 item 2) — DESCOPED for Screened Electrification
    (2026-09-15).** STEO `ELWHU_*` (scoped-not-run) and ERCOT RT/DA spread (feasible) remain
    available future work, but the shipped strategy doesn't need a forward-power input, and the
    stress-beta gate reuse (item below) is realized/spot GSI data, not forward-looking. Revisit
    only if a future signal specifically needs a forward-looking stress proxy. See
    `docs/steo-archive-feasibility-results.md`, `docs/rt-da-spread-feasibility-results.md`,
    `docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md` §3.2.

---

## 5. Code & doc map

- **Untried-tasks log:** `docs/untried-tasks-log.md` — working checklist of electricity-market-data angles from the 2026-09-08 review, worked one at a time; closed items get a row here in §3 too.
- **Built package:** `electrification_strategy/` (+ `tests/electrification_strategy/`, 38 tests).
  Predecessors still in tree: `grid_equipment_basket/`, `grid_resilience/`, `transmission_rate_base/`,
  `backlog_factor/`, `grid_demand_factor/`.
- **Kept data modules:** `grid_resilience/data/va_transmission_data.py` + seed
  `grid_resilience/data/seed/va_transmission_projects.csv`; `grid_resilience/data/seed/pjm_large_load_b9_vintages.csv`;
  `grid_equipment_basket/data/category_demand.py` (FRED panel — dashboard series, not a signal);
  `grid_equipment_basket/ftr_signal.py` (negative result, kept);
  `electrification_strategy/data/nem_calendar.csv` (10 dated state net-metering events — #15 FAILED, kept as a risk-monitor reference).
- **Exploratory probes (tracked):** `research/` — `va_transmission_probe/`,
  `category_demand_rotation/`, `forward_gas_power/`, `xsec_factors_ls/`,
  `electricity_earnings_exposure/` (2026-09-07 — merchant-margin nowcast + power-cost exposure
  cross-section, both FAILED; `edgar.py` = reusable XBRL companyconcept helper) (see `research/README.md`).
  The parallel electrification-strategy session keeps its throwaway scripts in the gitignored
  `scratchpad/` (`crowding_probe`, `backlash_hedge`, `hedge_probe`, `hedge_feasibility`,
  `utility_rotation`, `make_exposure_plot`). None are production; re-run to regenerate.
- **Proposals:** `docs/electrification-strategy-proposal-v3.html` (2026-09-07, investment-committee
  write-up + short-offset analysis; build kit `research/electricity_earnings_exposure/proposal_v3/`
  — `build_figures.py` → `assemble.py [--deploy --light --pdf]`),
  `docs/electrification-strategy-proposal-v2.md` / `.html` (`screen`, 3-tier explainer),
  `docs/electrification-strategy-proposal.html` (earlier, ETF-holdings).
- **Handoffs (context):** `handoff_2026-09-01-grid-buildout-long-short.md` (the full prior record),
  `handoff_2026-09-03-transmission-project-filings.md`, `handoff_2026-09-05-electrification-strategy.md`.

---

## 6. Session contributions

Per-conversation log. Each chat appends a dated block here for what it did; the
substance goes in §1–§5, this is just attribution + a pointer.

### 2026-09-07 — "Electricity intelligence for equity alpha" pivot — two pre-registered spikes, both FAILED

Responded to an external two-part critique ("dislocation alpha" + "Electricity Intelligence
for Equity Alpha"): stop trying to rescue the GSI utility-equity factor; instead move one
layer closer to the power market. Constraint fixed by the user: **output must be an equity
signal — instruments are equities or at most equity options** (no power-market trading), which
rules out the critique's headline RT/DA nodal-dislocation product. Ran the two in-mandate
spikes it reduces to. Both reached pre-registered `FAIL` verdicts. Detail:
`docs/electricity-earnings-exposure-probe-results.md`. Probes: `research/electricity_earnings_exposure/`
(`edgar.py` XBRL companyconcept helper, `probe_a.py`, `universe.py`, `probe_b.py` — throwaway,
tracked).

- **Probe A — merchant-generation margin nowcast** (VST/NRG/CEG/TLN). Hand-coded fleet (GW by
  company×market×fuel, acquisition effective-dates) × cached ERCOT/PJM DA hub price − Henry Hub
  gas → quarterly economic-margin model, vs EDGAR XBRL `Revenues` / `OperatingIncomeLoss`
  (energy filers don't tag `GrossProfit`). **Retrodiction gate FAILED**: Δmodel tracks
  Δrevenue (pooled Spearman **+0.45**, sane) but Δoperating-income **−0.05** (sign-agree 48%;
  CEG −0.46). Hedge mark-to-market + retail / non-generation segments sever the
  realised-price→earnings link. A2 (returns test) not run per the pre-registered fall-through.
  Independently re-confirms `grid-stress-signal-evidence.md` §5 (VST/NRG regime instability)
  from the fundamentals side.
- **Probe B — electricity-cost exposure cross-section** (~18 power-intensive industrials:
  aluminium AA/CENX/KALU, EAF steel NUE/STLD/CMC/ATI, chlor-alkali OLN/WLK, industrial gas
  APD/LIN, BTC/AI-hosting miners MARA/RIOT/CLSK/WULF/CIFR/IREN/BTDR). Hand-coded per-name
  `{state: weight}` exposure vector × electricity-intensity constant; signal
  `−intensity·%Δ(weighted power price)`, z-scored; price = EIA state industrial retail
  (primary) and ERCOT/PJM DA hub (measurement check). **6-gate tree (from the proposal)
  FAILED**: G1 Δgross-margin IC≈0 (t −0.6 / −0.2 across both price proxies), G1 Δop-margin
  predictive t −1.9 (wrong sign — only *contemporaneous* t +2.2, an accounting identity the
  10-Q already carries), G2 fwd-1m/3m return IC≈0, G3 FF5+MOM+sector-neutralised signal coef
  t **+1.7 → +1.0**. Only G4 (breadth, 18 names) passes. The 2022–26 fwd-3m return blip
  (t 1.96 on the retail-price proxy) collapses to t 0.7 on the wholesale proxy — proxy noise,
  not a regime effect.
- **Probe C — input-cost hedge screen** (follow-up, same session). Tested whether the
  `electrification_strategy/` supplier book has hedgeable exposure to its real cost inputs
  — copper (`HG=F`), grain-oriented electrical steel (`CLF` proxy), skilled labour (FRED
  construction AHE + RHI/MAN/KFY/ASGN staffing basket) — over 184 names, all sectors,
  109 months. **Gate 0 FAILED**: book beta to copper +0.02 (t 0.2), e-steel +0.02 (t 0.6),
  labour ≈ 0 — market beta alone gives R² 0.58, the three input factors add nothing (they
  pass costs through / had pricing power in the shortage). Tail check is *backwards*: in the
  12 biggest copper/steel/labour shock months the book returned +3.6% / +5.0% / +5.8% excess
  vs +1.5% typical — it **outperforms** when input costs spike. Screen surfaced only the
  mechanical names (CLF itself, copper miners β≈1–1.7, staffing firms) — most with positive
  book beta (more cyclical risk, not a hedge); the clean-beta ones (RIO/BHP/VALE/GOLD) have
  mediocre standalone Sharpe. **No hedge warranted** — there is no input-cost risk to hedge.
  `research/electricity_earnings_exposure/probe_c_inputcost_hedge.py`.
- **Takeaway.** "Move one layer closer to the market" does not rescue the equity signal —
  the binding constraint was never distance from the market; the equity payoff is a
  multiple/narrative story (merchant gens) or a multi-input-cost margin story (industrials)
  that a power-market model cannot see. Nothing wired. No open item added (the honest read is
  the electricity-alpha research line is closed for the equity mandate; the shipped product
  stays the `electrification_strategy/` long-only book). yfinance quarterly financials depth
  for these names is only ~6–7 quarters — EDGAR XBRL `companyconcept` (the reused helper) is
  the sourcing path for any future fundamentals work here.

### 2026-09-04 → 09-06 — Deliverable D BUILD: utility capex-guidance revision signal, end to end (FAILED)

Built Deliverable D from `handoff_2026-09-03-transmission-project-filings.md §6` — the first
test of the **commitment** data class — via a full 17-task subagent-driven build (brainstorm →
spec `docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md` → plan → per-task
implement/review → final whole-branch review → fix wave → scoped re-review). Committed to `main`
`085735b`…`b3eaa3a` (+ this log entry). Distinct from the parallel session's *independent
verification* block below — this is the build that produced the machinery it checked.

- **What shipped:** `grid_resilience/data/utility_capex_guidance.py` (loader, feasibility count,
  point-in-time EM-style DC-attributed imputation, panel aggregation) + hand/web-researched seed
  `grid_resilience/data/seed/utility_capex_guidance.csv` (44 rows, 15 utilities, 84% H-confidence);
  `grid_equipment_basket/capex_guidance_signal.py` (daily z-scored composite, one-directional
  de-risk multiplier + two-sided scaler, HAC-OLS / rank-IC stats, `feasibility_gate`,
  `timing_report`, `derisk_scaler_report`, `guidance_signal_report`, `guidance_table`);
  `config.py` constants; `--capex-guidance` CLI + `_write_guidance_outputs`; ~54 new tests
  (full suite 458→496 over the window, 0 regressions). Detail: `docs/capex-guidance-signal-results.md`.
- **Verdict — FAILED both legs** (see §3 row). Feasibility PASS 14/15 (NEE excluded, documented).
  Every primary rank-IC negative, none |t|≥2 (best −1.54). **De-risk leg not exercised** — the
  `all_usd` composite z never drops below +0.212, so the multiplier is flat 1.0 across all 9 grid
  combos / both windows; the final review caught the results doc mis-framing that as a merit FAIL
  and it was corrected to "= the vol-target-only baseline." Two-sided scaler fails G1 across the
  plateau. DC-attributed cut **not yet testable** (all DC-$ vintages 2026-dated) → re-run ~2027.
- **§4.4 sub-threshold inventory** (added this session, at user request): 9 of 10 cells at
  |t|≥1.5 are negative — a weak but *consistent inverse* sign. Open item 11 scopes the revisit
  (test inverted / joint with big-4 hyperscaler capex / nonlinear).
- **Final-review fix wave** (`ff32c58`) also fixed: a NaN-denominator bug inflating the `all_pct`
  series ~2.4×→1.0× across the window; 4 empty-Series guards leaving a non-DatetimeIndex (crashed
  `--capex-guidance` on a cold big-four cache); a 50-day look-ahead in the hyperscaler control
  (now aligned to `known_date`).
- The `.fillna(daily)` bug in `guidance_composite` (raw $ values substituted for NaN warmup) was
  flagged both by the parallel verification session and independently by this build's Task-8
  review; reverted in `c2e1444` before any gate ran on it.
- Housekeeping: briefly created a duplicate `docs/RESULTS.md`, then found this log already
  established and folded the content here (`f7ba06b`); added the `## Research Log` rule to
  `CLAUDE.md` pointing every future session at §3.

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

### 2026-09-06 — short-leg-as-insurance probe → crowding fixes → v1 package build → `screen` construction → backlash tilt → rotation re-tests → regulatory-calendar DER short

One long electrification-strategy conversation. Chronological:

1. **Short-leg-as-insurance probe** (`scratchpad/hedge_probe.py`, `hedge_feasibility.py`;
   `docs/electrification-short-leg-insurance-probe-results.md`). Re-opened the CLOSED short-leg
   search on a *risk-reduction* bar (not durable-alpha): scored a 7-item menu (static/conditional
   SPY short, firm-gen, DC-REITs, DER sleeve, crowding basket, GLD/IEF sleeve, de-lever) by
   **insurance efficiency = avg episode-DD reduction ÷ CAGR drag**, with hard gates (no episode
   deepened, feared-scenario P&L ≥ −2%, calm drag ≤ 6%/yr). Result: **no thematic short passes**
   — DER sleeve is an *anti-hedge* (feared-scenario P&L −24% to −110%), firm-gen deepens the 2022
   DD, crowding bleeds −45%/yr carry, composites are poisoned by any DER weight. **GLD/IEF 15%
   sleeve wins** (insurance-efficiency 1.9, raises Sharpe); **conditional DLR·EQIX short** (gated
   on 6-mo real-yield rise) is the runner-up (trims MaxDD ~1.4pp for ~1.2pp/yr drag).
2. **Crowding fixes** (`scratchpad/crowding_probe.py`). Quality broadening (marquee-9 → ~20
   profitable names) — keeps the return, 4pp shallower MaxDD, top weight 11%→5%; net-neutral
   Sharpe. Valuation-extension de-lever (graduated 1.0/0.8/0.6 on price-vs-200dMA + 12m-vs-SPY) —
   mild "sell-when-euphoric" trim, ~0–3pp MaxDD, plateau-stable. Both **ADOPTED** into the build.
3. **v1 strategy package** — brainstorm → spec (`docs/superpowers/specs/2026-09-06-…`) → plan
   (`docs/superpowers/plans/2026-09-06-…`) → 10-task TDD build → merged. Comparison harness:
   {marquee, frozen, thematic} × {plain, +val, +val+sleeve, +val+sleeve+short}. Pre-registered
   winner `thematic +val+sleeve` (Sharpe 0.85 / MaxDD −24%). Two deviations logged: daily
   vol-target (not the month-held `overlay.vol_target_scalar`, which deepened MaxDD ~15pp),
   valuation extension on the plain basket index.
4. **`screen` construction — the current ship.** 4th universe = the `capex-cycle-pair-classifier`
   rule (institutional customer + ≥2 of {profitable, earnings-valued, low-policy-dependence}),
   supplier side only, **no ETF gate** → 27 names, 0% utilities/pipelines/components. Rebuilt
   `etf_membership_2026.csv` from real published holdings (VOLT ~30% utilities / 10% midstream /
   13% components; `thematic` shrank 10→7, `frozen` 23→15; GRID/PAVE/ELFY top-25-visible only).
   Shipped **`screen +val+sleeve`** by PM override of the pre-registered `marquee plain` (which
   games the raw-Sharpe rule via concentration + no protection). Full 2017–2026 Sharpe 0.69 /
   MaxDD −24% / β 0.59; beats VOLT on the fair window (2025-01→2026-08): CAGR 27.7 vs 24.1,
   Sharpe 1.35 vs 0.82, MaxDD −17 vs −24. Merged to `main`; 38 tests.
5. **Data-center-backlash hedge test** (`scratchpad/backlash_hedge.py`; no in-sample episode,
   proxy = 12 worst DC-power months with SPY flat/up): **grid-maintenance tilt ADOPT** (70/30
   grid-maint/DC-power, net-neutral headline, β 0.36 to the DC-power sleeve, +0.4% vs DC-power
   −7.2%) → open item #2; **conditional merchant-power short REJECT** (lowers Sharpe, deepens MaxDD).
6. **Rotation re-tests** (`scratchpad/utility_rotation.py`): rate-conditional rotation INTO
   utilities (XLU) — **REJECT** (worse asset, redundant with the IEF sleeve leg); DER sleeve
   SHORT re-tested on `screen` — **anti-hedge**, worse at every size (Sharpe 0.69→0.36→−0.18,
   MaxDD −24→−42→−83); DER sleeve LONG when real yields fall (0→10%) — **mild PASS**
   (Sharpe 0.69→0.77, +7% feared, rescues 2019–22) → **open item #17**.
7. **Behaviour figure** `docs/electrification-strategy-exposure.png` (embedded in the proposal
   HTML) — growth / rolling 1-yr Sharpe (+ SPY, VOLT context) / net supplier-basket weight
   (trimmed <0.85 baseline ~73% of days; valuation rule active ~42%; deepest trim 0.14 Apr-2020).
8. **v2 proposal** `docs/electrification-strategy-proposal-v2.{md,html}` (3-tier: like-I'm-5 / -10
   / new-grad) + Artifact `claude.ai/code/artifact/c5db7bcc-11aa-41e5-9048-161308ecb3a7`.
9. **Load-surprise review** — confirmed every demand-data class already failed as *timing*;
   linked open items **#15** (regulatory-calendar DER short trigger) and **#16** (capex as an
   *entry* accelerator) as the untried angles.
10. **Regulatory-calendar DER short trigger (open item #15) — TESTED, FAILED.** Built a 10-event
    NEM calendar (`electrification_strategy/data/nem_calendar.csv`). Event study: mean DER return
    +0.0% vs the book across 6 adverse milestones (no consistent sign — the Dec-2021 NEM 3.0
    *proposal* is +8.7%). Overlay on `screen +val+sleeve`: +0.04 Sharpe but MaxDD −24→−28→−31,
    feared-scenario P&L −7% to −18%, timed-short sleeve −1.9%/yr standalone. n≈1 event; policy is
    already priced. Nothing wired; calendar kept as a reference. `docs/nem-calendar-der-short-results.md`.

### 2026-09-08 — full research-log + grid-resilience review; untried-tasks log opened; 2 items closed

Reviewed this entire log plus `docs/grid-stress-signal-evidence.md` and the open handoffs
end to end, at the user's request, to find electricity-market-data angles either genuinely
untried or built-but-never-closed-out. Opened `docs/untried-tasks-log.md` as a working
checklist (8 items, priority-ordered) and worked the first one immediately rather than only
listing it.

- **Congestion-spread Phase 1/2 (item 1) — closed with a verdict.** Offline check against
  production LMP caches (no new fetches) found the original "`congestion_frac` is NaN for
  all ISOs" diagnosis was only ever true for ERCOT — PJM's raw feed already carries a real
  `congestion` component (0% NaN, spread-fill changes 0/2,922 days there). ERCOT's fix works
  exactly as designed (100%→0% NaN, a real non-trivial series). Since the fix predates the
  2026-06-17 grid search, every reported GSI number since already reflects it — there's no
  "before" state left to A/B, and 3+ months of subsequent probes soak-tested it without
  incident. **Phase 2 (component decomposition + 5th sub-signal) marked NOT RECOMMENDED** —
  no live bug left, and the documented constraint is signal strength / sample size, not
  construction. Corrected `CLAUDE.md`'s overbroad claim accordingly.
- **Interconnection-queue-velocity capture (item 8) — process check, negative.** Open item
  #10 says to "keep running" a quarterly capture for a ~2028 retest; verified there is no
  cron, workflow, or versioned snapshot series anywhere in the repo, only one static cached
  file. Flagged as a build-or-abandon decision rather than silently fixing it.
- **Items 2–7 logged, not yet started:** ICR retest via the EDGAR helper built 2026-09-07
  (the yfinance-depth blocker that shelved `USE_ICR` no longer applies); the 5 documented-
  but-unbuilt merchant stress-beta fixes for the VST/NRG whipsaw; the RT/DA LMP spread
  feasibility check (recommended 3 times over a month, never run); ERCOT CRR true clearing
  prices (a stronger instrument than the PJM bid data that already failed); the STEO
  forward-power series reframed as a beta-stabilizer rather than a basket-timing signal
  (its original, more specific purpose); and real-time ancillary/scarcity pricing (ERCOT
  ORDC, PJM sync reserve) — never scoped at all. Detail and next steps for each:
  `docs/untried-tasks-log.md`.

### 2026-09-08 (cont.) — item 2 (ICR-via-EDGAR retest): feasibility PASS, hypothesis FAIL

Picked up untried-tasks-log item 2. Resolved CIKs for all 25 `grid_resilience` universe
tickers via SEC's `company_tickers.json`, then reused `research/electricity_earnings_exposure/edgar.py`'s
`quarterly_flow()` to pull `OperatingIncomeLoss` / `InterestExpense` back to 2008 for most
names — a same-session build (`research/icr_edgar_retest/icr_probe.py`), not a new project.

- **Feasibility: PASS.** Median 63 quarters/ticker vs yfinance's documented ~4–12; 22/25
  tickers covered back to 2019+. The specific limitation CLAUDE.md flagged for `USE_ICR` is
  resolved.
- **Hypothesis: FAIL, on genuinely good data this time.** Full-sample monthly rank-IC
  (regulated-only — ICR's actual production domain in `--arch hard-switch`, excluding
  NRG/VST/CEG/TLN — 102 months, 2019–2026): mean IC −0.025, t = −0.95, 48% of months
  positive — a null result with the wrong sign. The targeted 2022-23 rate-hiking-cycle test
  is equally flat (t = −0.24, regulated-only; t = +0.07, full universe). Two windows show a
  large correlation (t > 4) but both dissolve into single-name confounds already documented
  elsewhere in this line: PCG/EIX (2024–26 wildfire-liability litigation, unrelated to
  interest coverage) anchor the low-ICR end, VST/CEG/D/ETR (AI-power-demand re-rating)
  anchor the high end — the same merchant/idiosyncratic-narrative contamination that has
  broken every other cross-sectional signal tried on this universe
  (`docs/grid-stress-signal-evidence.md` §5).
- **Disposition:** `USE_ICR` stays `False` — now confirmed correct rather than a
  data-availability placeholder. Kept: `CIK_BY_TICKER` (25 names) and the probe script for
  any future fundamentals work on this universe. Not done: wiring EDGAR into production
  `fetch_icr()` — no point, the hypothesis failed on the good data.
  Detail: `docs/icr-edgar-retest-results.md`. Updated `CLAUDE.md`'s ICR and RT/DA-spread
  Future Work notes accordingly (RT/DA spread is now the top open signal candidate).

### 2026-09-08 (cont.) — item 3 (merchant stress-beta orthogonalization): fix #2 tested, FAILED

Continued down the untried-tasks-log priority order. Built
`research/merchant_beta_orthogonalization/probe.py` to test fix #2 of the 5 documented
candidates for the VST/NRG stress-beta whipsaw: add each ISO's power-price trend as a
second OLS regressor so the GSI coefficient is orthogonalized to the market's structural
narrative. Ran it against the **real, unmodified `grid_resilience.main.run()` pipeline**
(monkeypatched only `conditional_beta.compute_stress_betas`) so every other stage — GSI
construction, factor scoring, portfolio backtest, IC test — is identical to the documented
baseline, which was re-confirmed bit-for-bit in the same session (full Sharpe −0.007,
IC@21d t=1.352, matching `grid-stress-signal-evidence.md` §2.2's "with PJM" row).

- **Result: FAILED.** VST's orthogonalized beta correlates 0.996 with the unmodified
  baseline (mean move 7.2%; NRG 11.7%) — the fix barely engages despite the second
  regressor being live for 75-80% of rebalance dates. Every documented flip-window date
  (2023-09 through 2025-06) shows the same beta to within estimation noise in both
  versions. The one metric that matters — IC — moved the wrong way: IC@21d t 1.352→1.027,
  IC@63d t 1.889→1.782. Full Sharpe ticked up (−0.007→+0.046) but that's noise on one
  parameter change over 93 months, not the effect this fix targeted.
- **Diagnosis:** a trailing 252-day *realized* LMP trend is the wrong proxy for what's
  actually diagnosed — a forward-looking, event-driven valuation re-rating (a specific
  restart announcement, a hyperscaler contract), not a slow-moving spot-price statistic.
  Same gap the June-2026 forward-power handoff and the Henry-Hub-proxy failure both hit:
  the informative part of this story lives in the forward segment of the curve.
- **Redirect, not a dead end:** this makes item 6 (EIA STEO `ELWHU_*`, a genuine ~16-month
  forward series) the more theoretically motivated version of the same idea — worth trying
  there instead of re-running this exact regressor choice. Fix #3 (power-price momentum /
  forward-curve regime flag) should likely be folded into that same STEO work rather than
  tested standalone.
- **Not tested this session:** fixes #1 (XLU/XNG correlation filter), #4
  (structural-break-adaptive window), #5 (fundamentally-anchored beta) — item 3 stays
  IN PROGRESS. Detail: `docs/merchant-beta-orthogonalization-results.md`.

### 2026-09-08 (cont.) — item 3 closed out: fixes #1 and #4 tested (3/5 total)

Per user instruction ("finish the spikes you have started"), continued item 3 rather than
moving to the newly-added item 9 (narrative/mispricing, queued behind this).

- **Fix #1 (XLU/XNG correlation filter) — real effect, NOT ADOPTED.** Monkeypatched
  `main.build_rolling_weights` to exclude a ticker from that month's long/short eligibility
  when `corr(stock, ^XNG) > corr(stock, XLU)` for 2 consecutive months. Validity check is
  clean: all 63 flagged ticker-months (of 1,171) are NRG/VST/CEG/TLN, zero false positives
  on the 21 regulated names — the proxy correctly isolates exactly the diagnosed problem.
  Stress-period Sharpe improves −0.145→−0.085, full Sharpe −0.007→+0.050. Note: IC is
  unaffected by construction (`compute_ic()` runs on the pre-filter `rolling_factors`), so
  fix #2's "IC must improve" bar doesn't apply here — judged on realized backtest
  performance instead, which is the correct test for an eligibility filter. Still nowhere
  near beating XLU (0.261) — logged as validated-but-insufficient, not wired.
- **Fix #4 (exponentially-weighted / structural-break-adaptive beta window) — FAILED
  decisively.** WLS with a 63-day-halflife recency weight over the same stress-day window:
  full Sharpe −0.007→−0.448, stress Sharpe −0.145→−0.597, IC@21d t 1.352→0.809. Recency
  weighting an already-thin stress-day-filtered sample amplifies noise faster than it
  captures genuine regime change.
- **Item 3 verdict, 3/5 fixes tested:** one real-but-insufficient (#1), two failed (#2, #4),
  one redirected into item 6 (#3, folded into the STEO forward-power work since it's the
  forward-looking version of what #2 already tested and failed in trailing form), one left
  open pending real data collection (#5, fundamentally-anchored beta — needs hand-collected
  fuel-mix/region/hedge-book data, not a price-data spike). Consistent with the 2026-08-26
  closeout audit: construction fixes on this signal top out at a few points of Sharpe.
  Full detail: `docs/merchant-beta-orthogonalization-results.md`.
- **Also this turn:** added open item 9 to `docs/untried-tasks-log.md` at user request —
  electricity narrative/mispricing (short side), a genuinely new thesis distinct from the
  already-FAILED Probes A/B earnings-exposure channel. Explicitly queued behind finishing
  item 3 and the items-4-7 spikes.

### 2026-09-08 (cont.) — item 4: RT/DA LMP spread feasibility check, run for the first time

Continuing down the priority order after closing item 3. This exact check was recommended
as "the next step" in three handoffs since 2026-08-15 and never executed until now.

- **ERCOT: PASS, and a genuine surprise.** Every prior document assumed "real-time LMP"
  was the missing fetcher (`CLAUDE.md`: "needs a new `ISO_RT_LOCATION_TYPE` config dict and
  a `fetch_lmp_rt()` variant"). Live-checked `gridstatus.Ercot.get_lmp()`'s own docstring
  and behavior: it's already gridstatus's **real-time** (SCED 5-min) product — the exact
  same call already feeding the production GSI's `lmp_z` component. The actually-missing
  leg is **day-ahead** (`get_dam_spp(year)`) — pulled 2018 live in 7.9s, 122,640 rows,
  identical location naming (`HB_NORTH`/`LZ_NORTH`/etc.) to the cached RT data, full history
  2011->present. This is a much smaller build than every prior estimate assumed.
- **PJM: confirmed blocked, precisely.** `market="REAL_TIME_HOURLY"` for a pre-2024 date
  forces a full ~10,000+-pnode download before location filtering regardless of date-range
  width — reproduced live twice (a 1-week and a 1-day range from 2019 both 429'd on page 6
  of 50k-row pagination, `ValueError: No objects to concatenate`). The identical call for a
  recent week (2025-06) succeeded in 4.7s with server-side pnode filtering confirmed in the
  request log. So the blocker is real but only for the ~2018-2024 stretch of the backtest.
- **Why this doesn't matter as much as it sounds:** VST and NRG — the two names the whole
  stress-beta whipsaw traces to (item 3, `grid-stress-signal-evidence.md` §5) — are both
  ERCOT. PJM's pre-2024 blocker mostly affects CEG/TLN, which have short listed histories
  anyway (13 and 9 quarters per the item-2 ICR retest).
- **Disposition:** feasibility-only, as every prior handoff itself recommended. The actual
  signal design/build (does RT-minus-DA predict returns, pre-registered gate) is not started.
  Corrected `CLAUDE.md`'s RT/DA note. Detail: `docs/rt-da-spread-feasibility-results.md`.

### 2026-09-08 (cont.) — item 5: ERCOT CRR feasibility check, FAILED

Continued down the priority order after item 4. Found ERCOT's actual CRR report IDs via
web search (`gridstatus` has no CRR wrapper at all) — Monthly Auction Results (report type
11201) and Long-Term Auction Results (11203). Queried the live document-list API: exactly
12 documents returned, a rolling 2025-09→2026-08 window, unchanged across three different
parameter guesses (`startDate`/`endDate`, `mimicKey`, `size`) meant to unlock older files.
Checked ERCOT's public CRR landing page directly — no historical-archive or bulk-download
section exists. **Verdict: FAILED at feasibility.** Closes the open question left by
`docs/ftr-bid-signal-results.md` ("untested") with a real negative: full 2018-2026 CRR
history needs a special ERCOT client-services data request, not a free API pull — same
shape as every other genuinely-forward-looking instrument in this research line (CME/ICE
power forwards, NERC GADS, PJM's SSO-walled FTR clearing prices). Not pursued further.
Detail: `docs/ercot-crr-feasibility-results.md`.

### 2026-09-08 (cont.) — item 7: ancillary/scarcity pricing feasibility, PJM PASS / ERCOT FAIL

A genuinely new idea, never mentioned anywhere in this research line before this session
(surfaced during the initial review, not carried over from any prior handoff). Checked live:

- **ERCOT candidates both blocked**, same shape as item 5's CRR finding:
  `get_real_time_adders()` (the actual ORDC price adder — the purpose-built real-time
  scarcity signal) and `get_as_reports_dam()` (day-ahead ancillary clearing prices) are
  both served from rolling "2-Day Disclosure"-style reports — a 2019 test for either
  returned nothing, only the most recent (2026-09) document exists via the live API.
- **PJM: PASS, cleanly.** `get_real_time_as_market_results()` — Regulation / Synchronized
  Reserve / Primary Reserve clearing prices — returned full data back to January 2018 in
  1-1.5 seconds per test. This dodges item 4's PJM pnode-count blocker entirely because
  it's a market-level aggregate report, not per-node LMP. Sanity check: the Jan 2018 East
  Coast cold snap shows a real $230/MWh Regulation clearing-price spike vs. a typical
  $5-90 — genuine scarcity-event sensitivity, not noise.
- **Disposition:** mechanistically distinct from every LMP-based signal in this log (a
  system-operator scarcity computation, not a settled energy price) and immediately
  buildable for PJM's regulated names — most of the universe. ERCOT would need the same
  special-request workaround flagged in item 5, or a first pass skips ERCOT entirely.
  Feasibility-only; signal design/gate not started. Detail:
  `docs/ancillary-scarcity-pricing-feasibility-results.md`.

**Items 4, 5, 7 (the three data-feasibility spikes) all now closed.** Item 6 (STEO
forward-power, reframed) remains — the last queued item before the user's item 9
(narrative/mispricing).

### 2026-09-08 (cont.) — item 6: STEO archive feasibility, better path found (not fully closed)

Last of the four queued spikes (items 4/5/6/7). The original scoping assumed scraping ~90
STEO PDF releases directly (~half a day). Checked: `eia.gov/outlooks/steo/archives/`
blocks directory listing (403 with both a default and browser-style User-Agent), but
individual files ARE fetchable once the naming pattern is known
(`archives/{mon}{yy}.pdf`, confirmed live) — and every vintage is PDF-only, no CSV/Excel
companion, meaning real table-parsing across ~90 files. That's harder than the original
estimate, not easier. **Found a better path instead:** Catalyst Cooperative's PUDL project
— already used in this repo for FERC Form 1 (`transmission_rate_base/`) — publishes the
full STEO archive, 1983-2026, as structured Frictionless Data Packages on Zenodo, already
parsed rather than raw PDFs. **Not yet confirmed:** whether PUDL's package actually
includes `ELWHU_PJ`/`ELWHU_TX` and preserves per-vintage (as-of-publication) values rather
than only the latest revision — the point-in-time premise depends on this and wasn't
checked this session. Item 6 stays IN PROGRESS, not closed. Detail:
`docs/steo-archive-feasibility-results.md`.

**All four queued feasibility spikes (items 4, 5, 6, 7) now have a session's worth of real
live-data evidence behind them** — three fully closed (4: mixed pass/fail, 5: fail, 7:
mixed pass/fail), one (6) upgraded to a clearer next step but not finished. Per user
instruction, item 9 (electricity narrative/mispricing) is next.

### 2026-09-08 (cont.) — item 6 correction: PUDL does not shortcut the STEO extraction

User asked to keep working items 6 and 7 while starting item 9. Followed up on item 6's
"PUDL might have this structured" lead from the same session and found it doesn't hold:
live-checked PUDL's actual processed database (the public S3 nightly mirror at
`s3://pudl.catalyst.coop/nightly/`, the same source `transmission_rate_base/` already
reads for FERC Form 1) — 421 tables total, zero matching STEO/ELWHU/wholesale/outlook.
The Zenodo record that looked promising turned out to be PUDL's *raw-input preservation
archive* — a mirror of EIA's original files, confirmed by direct inspection to be
year-zipped raw material ("not a cleaned/processed dataset" per its own description), not
a shortcut past parsing them. **Corrected disposition: back to the original ~half-day
PDF-table-extraction estimate** from `docs/hhub-forward-gas-signal-results.md` — Zenodo is
useful only as a more stable place to fetch the same files (EIA's own archive blocks
directory listing), not as a time-saver on the actual work. Updated
`docs/steo-archive-feasibility-results.md`, `docs/untried-tasks-log.md` item 6 to remove
the overstated "better path found" framing from the prior session block.

### 2026-09-08 (cont.) — item 7 follow-up: a real timing-IC check, and a caught methodology trap

User asked to keep pushing items 6 and 7 while starting item 9. Pulled the full
`PJM_RTO`-wide ancillary clearing-price history for 2018-2021 (175,320 rows; 2022-2025 hit
sustained 429s on a much larger post-market-redesign schema and needs monthly chunking,
not attempted) and built a rolling-756-day z-scored composite to test as a basket-timing
signal (it's ISO-wide, identical across every PJM name on a given day, so it can only work
as an overlay, not a cross-sectional ranker, same shape as the GSI's congestion regime).

**Caught and corrected an inflated first-pass result.** A naive daily-sampled,
overlapping-window Spearman IC looked genuinely strong (21d t=-3.85, 63d-vs-relative
t=+4.61) — but recomputing properly (monthly-resampled, non-overlapping, HAC standard
errors, the same discipline this whole log uses everywhere else) collapsed it to nothing:
1-month t=+1.04, 3-month t=+0.18. The daily-overlap version was inflated by
autocorrelation, not signal — logging this explicitly since it's exactly the kind of trap
this research line has been careful to avoid, and the corrected write-up
(`docs/ancillary-scarcity-pricing-feasibility-results.md`) replaces the overstated framing
from earlier the same day with the honest one.

**Disposition: feasible, not yet promising, not yet killed.** n=44 months is thin and the
2022-2025 DeepSeek-regime window (the one that mattered most for VST/NRG elsewhere) is
still unpulled. Item 7 stays IN PROGRESS.

### 2026-09-08 (cont.) — item 9 opened properly via brainstorming; Approach A run to a verdict

User asked to move to item 9 (electricity narrative/mispricing) while continuing items 6
and 7. Given item 9 has no existing flow in this repo to extend, invoked
`superpowers:brainstorming` rather than jumping straight to a probe. Classified the first
step as a spike ("can 'narrative' even be operationalized in a testable way") before any
design commitment. Checked three candidate data sources live: EDGAR full-text search
(works, free, point-in-time, per-company keyword-mention counts), GDELT news-volume API
(works, free, daily resolution back to at least 2019, but a stricter-than-documented rate
limit needed two rounds of exponential-backoff retries to get usable coverage), and Google
Trends via `pytrends` (dead — returns nothing, matches this repo's recurring experience
with unofficial reverse-engineered APIs). User picked EDGAR + GDELT combined, sequenced
Approach A (narrative-extremity short) then B (event study) then C (narrative-fundamentals
divergence).

- **Approach A run to completion.** Built a monthly composite (z-scored trailing-4Q EDGAR
  "data center" mentions + z-scored trailing-63d GDELT volume) and ranked it cross-sectionally
  across the 25-name universe (22 with usable GDELT coverage). Cross-sectional rank-IC vs
  forward return is right-signed for a short thesis (negative) at every horizon but not
  significant: 1mo t=-0.39, 3mo t=-1.15, 6mo t=-0.55, n≈85-87 months.
- **A real, useful measurement finding, not just a null result:** VST's composite score is
  *consistently negative* (under-narrated relative to peers) all through 2023-2025 — the
  exact window everyone agrees its AI-power narrative dominated its stock — because VST is
  a generation company, not a data-center operator, so it almost never says "data center"
  in its own filings (1 total EDGAR mention across the whole window, vs. Southern's 25,
  Talen's 19, Exelon's 16). GDELT's raw cross-sectional volume for a mid-cap merchant
  generator is also swamped by mega-caps' routine coverage baseline. NRG's composite does
  show a real escalation timed to its 2024-25 re-rating, so the mechanism isn't blind, just
  keyword-narrow and not baseline-normalized.
- **Disposition: not killed, not adopted.** Two concrete, untried fixes logged (broader
  EDGAR keyword set for merchant generators specifically; normalize GDELT by each
  company's own trailing baseline instead of raw cross-sectional volume). Per the user's
  own sequencing, proceeding to Approach B next rather than immediately re-running A with
  fixes. Detail: `docs/narrative-mispricing-approach-a-results.md`.

### 2026-09-08 (cont.) — item 9 Approach A refined and closed out; handoff written for B

User asked for a short handoff for Approach B (to continue in a fresh session) before
refining Approach A with the two fixes flagged in the first pass.

- **Handoff written:** `docs/handoff-narrative-mispricing-approach-b.md` — reuses the
  already-cached EDGAR/GDELT data, names the biggest risk up front (repeating open item
  #15's n≈1/telegraphed-news failure mode), and orders the work: count candidate events
  and gate on n before building any event-study machinery.
- **Approach A refined:** added "large load" as an EDGAR keyword (PJM/ISO planning
  terminology — VST's count went from 1 to 13 combined with "data center"; "hyperscaler"
  was also tried and dropped, near-zero everywhere) and switched GDELT from a
  cross-sectional z-score to each ticker's own trailing-756-day self-baseline z-score.
  **This cleanly fixed the measurement problem** — VST's composite now genuinely tracks
  known events (1.83 at the Sep-2023 narrative build, 1.31 at the Feb-2025 peak) instead of
  sitting permanently negative. **It did not move statistical significance** — rank-IC is
  still |t|<1.4 at every horizon, and the 1-month sign flipped positive (narrative
  intensity behaving like short-term momentum, not overhype), while 3/6-month stayed
  weak-negative. Honest conclusion: the null result survives better data, so it isn't a
  measurement artifact — the relationship (if one exists) is genuinely weak on this
  operationalization.
- **Disposition: Approach A NOT ADOPTED**, closed with a clean writeup. Two reusable
  outputs kept regardless: the EDGAR "large load" + GDELT self-baseline measurement recipe
  (validated, usable for future narrative work on this universe), and the momentum-not-
  overhype sign structure as a note for anyone revisiting narrative intensity later.
  Approach B is queued next via the handoff, not started this session.

### 2026-09-08 (cont.) — handoff written for item 9 Approach C

Wrote `docs/handoff-narrative-mispricing-approach-c.md` (narrative-fundamentals
divergence short) at user request, for continuation alongside the existing Approach B
handoff. Explicitly instructed per the user: don't judge the narrative data on a
standalone linear rank-IC test alone — if that comes back flat, try the narrative
composite and a fundamentals measure (ICR, valuation, margin) as features in a
multivariate RF/GBM alongside this project's other existing signals, since a conjunction
effect can be invisible to a linear test on either variable alone. Paired with an explicit
guardrail against the exact overfit trap this project already hit once (the Henry Hub
"kitchen-sink GBM on 25 pooled signals" result that vanished out-of-sample,
`docs/hhub-forward-gas-signal-results.md`) — pre-register the OOS window and use a
purged/embargoed split given ~90 autocorrelated monthly cross-sections, not random k-fold.
Nothing built yet; both B and C now have handoffs ready for a fresh session.

### 2026-09-08 (cont.) — item 9 Approach C run to a verdict: FAILED (broadened per user request)

Picked up the Approach C handoff. User asked to go beyond the handoff's minimum (narrative +
ICR) and fold in **any feature already built in this research log that might help in tandem,
even a t-stat-~1 one**, using GBM/RF rather than a plain linear divergence sort. Classified as
a spike, named the precedent risk up front (Henry Hub's kitchen-sink GBM overfit), got a nod,
then built a 10-feature monthly panel on the 25-name universe: the narrative composite, ICR,
merchant stress-beta, a **freshly re-run** production GSI factor score (current config, not a
stale cached file), each ticker's own price momentum, and four market-wide conditioning series
(utility capex-guidance revisions, category-demand momentum, PJM ancillary pricing, real-yield
change, gas momentum) — one feature per distinct data class this project has built to date that
could be aligned onto this universe/window without a fresh multi-day fetch.

Pre-registered before looking at results: walk-forward `TimeSeriesSplit(5)` GBM + RF, full vs.
lean feature sets, 1mo/3mo forward return, HAC rank-IC, 60-draw within-month permutation null,
PASS bar = \|t\|≥2 **and** perm p<0.05 in **both** a FULL (2018-25, n=88mo) and RECENT (2023-25,
n≈24mo) window.

**Result: FAILED**, reproducing the Henry Hub overfit shape exactly. RECENT-window cells look
like a real edge (t up to 3.73, perm p as low as 0.000, 6 of 8 full/lean × 1mo/3mo × GBM/RF
cells clear \|t\|≥2); every one of the 8 FULL-window cells is flat-to-negative (\|t\|<1, perm p
0.20-0.92) — the identical "2023-26 flicker that vanishes on 2018-26" pattern already logged in
`docs/hhub-forward-gas-signal-results.md` Round 4. Fails the pre-registered both-windows bar
outright. Feature importances add a second, independent finding: `narrative_z` never ranks in
the top 3 of any full-model cell (share 0.02-0.10) — even the inflated recent-window fits are
driven by price momentum, the production GSI factor, and stress-beta, not narrative content.
So the answer to "does narrative interact with other data" is no, on this test: it earns no
material importance in a model that does find *something* (if only in the window that doesn't
generalize).

**Disposition:** Approach C closed, FAILED. Reusable: `research/narrative_mispricing/approach_c/`
(panel-build + walk-forward GBM/RF + permutation-null harness, kept for reuse on any future
feature combination on this universe). Approach B (event study) is now the only unstarted
thread in item 9. Detail: `docs/narrative-mispricing-approach-c-results.md`. Updated
`docs/untried-tasks-log.md` item 9 accordingly.

### 2026-09-08 (cont.) — ML combined retest (RF/GBM/classifier/quantile), 2022-2025 only

User asked to retest the session's weak/failed signals in combination — RF, GBM,
classifiers, quantile fits — specifically restricted to 2022-2025 (electrification wasn't
a live narrative before ChatGPT) rather than the full 2018-2025 window used everywhere
else this session, and to watch for low-t-stat signals gaining power in a group.

- **Built a combined panel:** ICR (item 2), EDGAR narrative + GDELT self-baseline (item 9),
  merchant stress-beta + the continuous XLU/XNG correlation spread (item 3), 6-month
  momentum, PJM ancillary-scarcity (item 7, 2022-2025 leg still fetching when this pass
  ran — rate limits on the post-market-redesign schema are worse than 2018-2021, confirmed
  again this session; contributes zero importance as a placeholder, not a tested null).
  45 months, ~23 tickers.
- **Pre-registered a single time-based OOS split** (train 2022-2023, test 2024-2025) and a
  shuffled-target sanity check *before* looking at results, specifically because this
  project already hit this exact trap once (Henry Hub kitchen-sink GBM,
  `docs/hhub-forward-gas-signal-results.md`, flagged again in the Approach C handoff two
  turns before this test ran).
- **Result: the trap recurred, caught cleanly this time.** In-sample fit is dramatic (GBM
  rank-IC up to 0.97, GBM classifier AUC = 1.000 on ~107 training rows / 7 features) and
  collapses out-of-sample (rank-IC 0.05-0.19, classifier AUC 0.61-0.66). The shuffled-target
  check confirms it isn't real: refit on random noise, the "important" features
  (`xlu_xng_spread`, `mom_z`) look similarly important — a chunk of the apparent structure
  is an artifact of feature variance/split frequency, not predictive content. **No credible
  evidence of a combined-signal effect** on the six signals tested so far.
- **Ancillary leg completed and folded in** (45 months, 1.96M rows, monthly-chunked
  pull worked around the post-2022 rate limits). Verdict unchanged, now confirmed on all
  7 signals: in-sample fit stays dramatic (GBM classifier AUC 1.000), OOS stays weak
  (-0.07 to +0.22 rank-IC / 0.53 to 0.62 AUC), shuffled-target importances still resemble
  the real ones. Adding ancillary pricing made the 1-month case *worse*, not better
  (GBM OOS rank-IC flips negative, classifier AUC drops to chance) — consistent with more
  overfitting surface, not new information, on a still-tiny training set.
  Detail: `docs/ml-combined-retest-results.md`.

### 2026-09-08 (cont.) — two-fold walk-forward retest (2022-23→2024, 2024-25→2026)

User requested a proper two-fold walk-forward instead of one split, and corrected the
framing: don't label an early fold's weak generalization "overfitting" when the trade
itself wasn't fully relevant yet (pre-2022/early-2022). Extended all seven signals through
2026 to make this possible: EDGAR narrative keywords (now through 2026Q3), PJM ancillary
pricing (2026 YTD pulled fresh, 431,918 rows), stress-beta (full `grid_resilience`
pipeline rerun through 2026-08). **GDELT could not be extended** — the 2026 pull hung on
repeated rate-limit backoffs for 25+ minutes and never completed; proceeded per explicit
user instruction using the stale (frozen ~2025-09) `gdelt_z` value for the 2026 test
window, flagged throughout rather than hidden.

- **Fold A (train 2022-23, test 2024): weak, OOS mostly negative** (regressor rank-IC
  -0.08 to -0.23, classifier AUC 0.42-0.57) — framed as early-regime per the corrected
  instruction, not as overfitting, though the in-sample/OOS gap is still large regardless
  of label.
- **Fold B (train 2024-25, test 2026): the most credible finding of the entire session.**
  GBM classifier (top/bottom-tercile 3-month forward return) gets OOS AUC=0.758 — stable
  across 10 random seeds (std=0.003), at the 100th percentile of a 50-draw shuffled-label
  null distribution (null mean 0.491), and positive in every one of the 6 test months
  (monthly AUC 0.55-1.00, not one-month-driven).
- **Drop-one-feature ablation reveals the real driver isn't narrative data.** Removing
  `stress_beta_z` costs the most (0.758→0.694); removing `mom_z` or `xlu_xng_spread`
  *improves* the AUC; removing the known-stale `gdelt_z` costs only 0.023 (a small but
  nonzero drop from a frozen feature — a data-integrity flag, not a validation).
  **The load-bearing combination is merchant stress-beta (item 3's already-studied,
  individually-weak GSI factor) + ICR + EDGAR disclosure counts** — confirming the user's
  original hypothesis (weak signals gaining power combined) on a different signal than
  the narrative/attention data this whole thread started with.
- **Disposition: real lead, not yet actionable.** n=54 test observations across 6 months
  is too small to trust on its own despite clearing the noise-floor test. Next steps
  logged, none done this session: fix GDELT properly, let the 2026 test window grow,
  isolate the 3-feature (stress-beta/ICR/EDGAR) interaction directly, do not wire
  anything live. Detail: `docs/ml-two-fold-walkforward-results.md`.

### 2026-09-08 (cont.) — item 9/10 cross-session reconciliation: my approach_c ablation + a merchant-exclusion check on the other session's Fold B

A third concurrent session (on the same machine) had independently picked up this same
item 9/10 thread and already produced the two-fold walk-forward finding above
(`docs/ml-two-fold-walkforward-results.md`) plus a fourth session's peer proposal to hand
this session the "2026 holdout" ownership. Reconciled rather than duplicating:

- **This session's own contribution (before the cross-session discovery):** ran the
  Approach C broadened GBM/RF panel to a verdict (FAILED — see the block above), then,
  per user pushback that the full 2018-25 window mismatches when an AI-power narrative
  could plausibly exist, ran a with/without-`narrative_z` ablation confined to
  2022-on/2023-on windows. Result: narrative_z's marginal contribution is **mixed but
  often positive** in the 4-feature "lean" model within those windows (e.g. RF/lean/3mo,
  2023-on: t=1.61 without narrative_z, not significant → t=2.56 with it, significant) —
  not simply inert, contrary to what the original full-model feature-importance ranking
  alone suggested. But the same lean model gets ~0 signal on the full 2018-25 window, and
  n=22-34 months is thin — exactly the regime where the Henry Hub work's own Round 5
  ("split the AI regime, re-test in/out-of-sample") found a similarly fragile positive
  tilt that didn't survive scrutiny. Kept as supporting texture, not a standalone verdict.
- **Discovering the third session's Fold B result** (`docs/ml-two-fold-walkforward-results.md`,
  OOS AUC=0.758, stress-tested, driven by stress-beta+ICR+EDGAR not GDELT) changed the plan:
  rather than re-run a near-duplicate 2026 holdout, ran the one check neither prior analysis
  performed — **does Fold B's signal survive excluding VST/CEG/TLN/NRG**, the four names
  already documented (`docs/grid-stress-signal-evidence.md` §5) as the recurring confound
  behind several other broken cross-sectional signals on this universe (most recently the
  ICR retest). **Result: it survives** — OOS AUC 0.755→0.719 on the remaining 19 regulated
  names, still 100th-percentile vs. its own shuffled-label null, not one-month-driven.
  This raises, not lowers, Fold B's credibility: the standard "it's just those four names"
  explanation that has killed prior signals on this universe does not apply here. Addendum
  written into `docs/ml-two-fold-walkforward-results.md` rather than a new doc. Script:
  `research/narrative_mispricing/foldb_merchant_exclusion.py`.
- **Coordination note:** a peer session ("acadia-9d") had independently built
  `research/narrative_mispricing/spike_c_2026_holdout.py` (a sound, reviewed single-fold
  train≤2025-12/test-2026 design) and offered to hand off "2026 holdout" ownership; once
  the third session's already-completed, more rigorous two-fold+stress-test result surfaced,
  running that single-fold script would have been redundant, so it was not run — kept in
  the repo as a reviewed, reusable alternative framing (continuous rank-IC across all 25
  names, vs. Fold B's tercile-classifier framing on 23) if anyone wants a second cut later.
- **Net read across all three threads today:** the narrative/attention data on its own
  (Approach A, Approach C) still FAILS every pre-registered bar tried. But a real, now
  twice-stress-tested (shuffled-label null + merchant-exclusion), small (n=48-54, 6 months)
  pattern exists in a *combination* of stress-beta, ICR, EDGAR disclosure counts, and price
  momentum for 2024-26 — this is the first result across this whole research
  line's ~28+ tracked attempts to clear a shuffled/permutation-null bar this cleanly. Still explicitly **not
  actionable** on 6 months of data per the original writeup's own conclusion, and GDELT's
  2026 extension remains unresolved (still running, rate-limited, 30+ minutes in).

### 2026-09-08 (cont.) — portal handoff written for the narrative-mispricing thread (session acadia-9d)

User asked for the thread's status to be reflected as "still promising" in the ledger and
distilled into one doc to carry to the main strategy portal. Done, no new analysis:

- **`docs/handoff-narrative-mispricing-portal.md`** — one self-contained page: the full arc
  (Approach A / Approach C-broadened / two-fold walk-forward / merchant-exclusion), the Fold B
  result stated precisely, why it is not actionable, and the four concrete next steps to promote
  it. Pre-registered verdicts: narrative-as-signal **FAILED**; weak-signals-in-combination (Fold B)
  **PROMISING / NOT ACTIONABLE**; merchant-confound explanation **RULED OUT**.
- **§3 ledger row** (Approach C) extended with the Fold B follow-on pointer + verdict; the
  "FAILED (narrative-as-signal)" verdict and prior session blocks left intact.
- **§2** gained a "one partial exception, watch-list only" note; **§4 item 18** added (re-check
  Fold B quarterly; isolate the 3-feature interaction; fix stale GDELT; do not wire live).
- Coordination: this session had launched a duplicate `grid_resilience.run(end=2026-08-31)` +
  narrative extension before discovering acadia-32 / acadia-6c were already on it — killed both,
  verified no shared-file corruption, stood down on data/holdout per the user's instruction that
  acadia-32 owns the holdout. Throwaway scripts left in `research/narrative_mispricing/`
  (`extend_2026.py`, `spike_c_2026_holdout.py`, `_holdout_prices.parquet`).

### 2026-09-08 (cont.) — "Dispatch": Fold-B walk-forward → L/S, 2026-drag diagnosis, profitability-screen counterfactual, build-plan handoff

User asked (a) how the Fold-B model would perform as a tradable L/S (Sharpe / return /
drawdown per year 2023-26 vs XLI, PAVE, VOLT), (b) what causes its 2026 drag, (c) whether
a profitability screen with a reasonable lookback would have filtered the worst name, then
(d) to name it and write a handoff. All from `research/narrative_mispricing/ml_panel_2022_2026.parquet`
+ `research/electricity_earnings_exposure/edgar.py`; throwaway scripts in the session
scratchpad (`bt_twofold.py`, `diag_2026.py`, `tln_profitability.py`, `bt_screen.py`).

- **As a L/S (GBM classifier, expanding walk-forward, monthly rebalance, dollar-neutral,
  long-top / short-bottom tercile, ~3–4 names/side), 2023-03 → 2026-08:** raw = **+2.1%/yr,
  Sharpe 0.22, MaxDD −16.2%**; per year 2023 +14.1% (thin, 10 mo) / 2024 +1.0% / 2025 −1.8%
  / 2026† −5.1%. Benchmarks same window: XLI +21.3%/yr S1.33, PAVE +23.2%/yr S1.17, VOLT
  (Dec-24 inception) +29.5%/yr S1.19. The L/S loses to all three on return and Sharpe.
- **2026 drag = the long leg, mostly TLN.** Short leg ~neutral. TLN realised −4.3%/mo in
  2026 (worst of the 13-name universe) and the model held it long 9 of 12 months
  (−9.6 Mar, −7.4 Jul, −14.7 Aug). Model is momentum-dominated in practice (`mom_z` imp
  ≈0.25–0.29, `xlu_xng_spread` ≈0.20–0.26 > the stress-beta/ICR/EDGAR trio Fold B was
  credited to); from mid-2026 it even rotates *into* high-stress-beta names as they de-rate.
- **Two broken inputs:** `icr_z` for TLN is frozen at a **2016** interest-coverage value and
  forward-filled for the whole 2022-26 panel (pre-bankruptcy company); `gdelt_z` is frozen
  at ~2025-09 (2026 pull never completed). The model had no real leverage/coverage read on
  TLN and a stale news feature contributing 0.11–0.17 importance in-window.
- **Profitability screen counterfactual (point-in-time EDGAR, ~50-day filing lag):** TLN has
  no comparable P&L until 2023-Q3 (emerged from Ch. 11 May 2023); by the Q4-25 filing its
  TTM net income is **−$219M, net margin −8.5%, worst 1/13** on both margin and growth.
  Adding a **bottom-tercile trailing-net-margin exclusion to the long leg** flips the
  walk-forward L/S to **+9.5%/yr, Sharpe 0.74, MaxDD −13.2%** (2026 −5%→+7%, Sharpe
  −0.82→+1.53); "exclude negative TTM NI" is weaker (+3.5%/yr, only catches H2-2026); a
  crude "≥8 quarters of revenue history" cut **backfires** (−5.0%/yr — benches good longs).
  Illustrative only: 42 months, 10–13 effective names, 3–4/side.
- **Named "Dispatch"**, build-plan handoff written: **`docs/handoff-2026-09-08-dispatch-ls.md`**
  (restore universe depth, formalise the point-in-time profitability screen, de-momentum,
  3-month horizon, vol-target + T-bill/gold sleeve + REGIME gate, costs, portfolio fit) with
  pre-registered gates. **Verdict: LEAD — SCOPED, NOT BUILT.** Parent doc's "do not wire
  live" stands. §3 "Cross-sectional / weighting" row + §4 item 19 added; §2 exception
  paragraph extended.

### 2026-09-08 (cont.) — "combine Dispatch + narrative-mispricing for breadth?" → ran spike_c, answer is NO

User asked whether merging Dispatch with the narrative-mispricing thread buys breadth
(more names / observations / signal). Added a note to `docs/handoff-2026-09-08-dispatch-ls.md`
and ran the one outstanding mispricing artifact to settle it.

- **Framing:** Dispatch *is* the mispricing Fold-B model as a book — same panel
  (`ml_panel_2022_2026.parquet`), 23 names, 7 features (`edgar_z`/`gdelt_z` included), GBM.
  Merging does not stack two edges; the narrative features are already inside and are the
  weak/broken part (ablation: dropping `gdelt_z` *raises* AUC; it is stale in-window).
- **Ran `research/narrative_mispricing/spike_c_2026_holdout.py`** (previously "not run,
  redundant") — the thread's widest independent framing: 25 names, 9 features, 2018-start
  panel, continuous rank-IC (regressor, not tercile classifier), GBM & RF × {full,lean} ×
  {1m,3m}, walk-forward + a proper train≤2025-12 / test-2026 holdout with HAC t. Fixed one
  Py3.12-only f-string so it runs on 3.11. Result written to `spike_c_2026_holdout_results.csv`.
  - 2023-25 re-derivation: only lean models clear \|t\|>1 (best lean fwd-3m GBM IC +0.124,
    t +2.25); full models \|t\|<1.
  - +2026 (window B): every cell decays to \|t\|≤1.4.
  - **2026 holdout (window C): flat-to-negative in every cell** — RF 3-month IC −0.167
    (t −3.77) and −0.186 (t −2.12), i.e. *inverted* OOS; GBM cells ≈0.
  - `narrative_z` feature-importance rank **7–8 of 9** (≈0.01–0.04); `own_mom6m` dominates
    (0.28–0.34) — the same momentum-loading failure mode as the 7-feature panel, on a wider
    universe and a different framing.
- **Verdict: no breadth of edge from merging.** The wider net reproduces "worked 2023-25,
  inverts 2026." `spike_c` is not worth folding in as an ensemble member as-is. Breadth work
  for Dispatch stays: prune the dead features (`gdelt_z`, stale/ffilled `icr_z`) to widen the
  feature-complete universe + push the panel back to 2018, add the point-in-time profitability
  screen, de-momentum. Docs updated: §3 Dispatch row (4), §4 item 19, portal doc pointer +
  addendum, `docs/handoff-2026-09-08-dispatch-ls.md`.

### 2026-09-08 (cont.) — "Dispatch v2": prune dead features + widen universe from full-history feature build

User: prune the dead features; use 2018+ data to *build* features but keep the *test* window
at 2023+ (the strat is known not to work in the earlier regime); don't gate on univariate
t-stat — use it only as a loose inclusion check, then let the RF/GBM combine. Script:
scratchpad `dispatch_pruned.py`.

- **Pruned:** `gdelt_z` (stale ~2025-09), `xlu_xng_spread` (single value per month → zero
  cross-sectional info + stale), `anc_composite` (2 levels → near-degenerate). **Rebuilt**
  `icr_z` point-in-time from EDGAR XBRL (TTM EBIT / |interest|, full history, 50-day lag) —
  fixes the frozen-2016-for-TLN bug, 100% coverage. **Widened** `stress_beta_z` = union of the
  two rolling-beta panels (adds NRG/VST/CNP/EIX/PCG, 2018 starts). Final set
  `[edgar_z, icr_z, stress_beta_z, mom_z]`.
- **Breadth achieved:** usable names/month 10–13 → **16–19 from 2024 on** (5–6 per side).
- **Loose feature check (informational):** `stress_beta_z` IC +0.14 (t +3.1), `mom_z` +0.15
  (t +2.7) carry it; `edgar_z` (t −1.4) and `icr_z` (t −0.2) weak, kept for the model.
- **Result — the widening confirms the earlier read, does not overturn it.** Headline Sharpe
  jumps (GBM clf 0.22→0.77; GBM reg 1.08) but it is a **single-regime artifact**: every model
  variant returns **+24 % to +86 % in 2024** and is negative-to-flat in 2023 / 2025 / (classifier)
  2026. The merchant-minus-regulated spread was +68 / +127 / +35 / −27 % across 2023–26 → Dispatch
  is a momentum-timed **merchant-vs-regulated sector bet**, fully loaded (via `stress_beta_z` +
  `mom_z`) just in time for 2024, then rode the 2026 reversal down. Only 4 merchant names, so
  more universe = more regulated shorts, not a diversified bet. The margin screen still helps
  2026 specifically (GBM clf −6.5 %→+2.2 %; RF reg −1.7 %→+7.5 %) but cannot rescue 2023/2025.
- **Verdict: pruning did its job (breadth 13→19); no cross-sectional edge survives the regime
  split — consistent with §2's accumulated conclusion.** One pre-registrable next test before
  abandoning: rank *within* merchant/regulated buckets (or residualize on the merchant-spread
  factor) and check whether anything survives 2023 + 2025 + 2026. If it is all 2024, there is
  no strategy. Docs: `docs/handoff-2026-09-08-dispatch-ls.md` §"v2", §4 item 19.

### 2026-09-09 — "Dispatch v3": sector-neutral test → FAILED the regime split, Dispatch closed as a standalone strategy

Last pre-registered test from the v2 block: strip the merchant-vs-regulated axis and check
whether any cross-sectional edge survives 2023 + 2025 + 2026 (not just the 2024 merchant-IPP
melt-up). Script: scratchpad `dispatch_sector_neutral.py`.

- **~100% of Dispatch's Sharpe is 2024.** Full GBM classifier: 2024 +61.7% (S 2.55), **ex-2024
  (2023+2025+2026) −1.3%/y, Sharpe −0.05** (full-period was +0.77). Full regressor: ex-2024
  Sharpe +0.07 (full-period +0.87).
- **Regulated-only (drop VST/NRG/CEG/TLN):** 2024 collapses to +1.0% — a real but trivial
  cross-sectional signal remains: GBM classifier ex-2024 Sharpe +0.19 raw, **+0.36 with the
  margin screen**, full-period ~0.40. Directionally consistent but **2023 still negative**, so
  it fails a "positive in each non-2024 year" bar and is not fundable standalone. GBM regressor
  regulated-only is negative everywhere (ex-2024 S −0.75).
- **Sector-neutral combined** (rank within each bucket, book bucket-balanced): ex-2024 Sharpe
  +0.12 (classifier) / noise (regressor, 2025 −27% then 2026 +45%).
- **Verdict: FAILED.** Dispatch is a momentum-timed merchant-vs-regulated sector bet
  (`stress_beta_z` + `mom_z` on 4 merchant names) whose one good regime (2024) has passed and
  which rode the 2026 reversal down −27%. No standalone cross-sectional alpha. Consistent with
  §2's accumulated conclusion — this is now the ~29th tracked attempt with the same outcome.
- **Kept:** (a) the point-in-time trailing-net-margin screen as a *quality overlay* — it lifted
  the worst year of every variant, incl. the shipped long-only book; (b) the pruned/widened
  point-in-time panel (`[edgar_z, icr_z(EDGAR XBRL), stress_beta_z(union of both beta panels),
  mom_z]`, 16–19 names/mo) as reusable infra. Fold B's AUC watch-item (§4 item 18) is
  untouched — that is tercile-classification AUC, not this returns expression.
- Docs: `docs/handoff-2026-09-08-dispatch-ls.md` §"v3"; §3 Dispatch row + §4 item 19 updated.
