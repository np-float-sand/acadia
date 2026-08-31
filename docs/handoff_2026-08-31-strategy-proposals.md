# Handoff — Strategy Proposals After the Grid-Equipment Work (2026-08-31)

## TL;DR

Two builds shipped this session, both concluding **negative or weak** on stock selection:

1. **Value-chain reframe + intra-theme hedge** (`grid_equipment_basket/`) — **MERGED to `main`**.
   Gate 1 (long-only maker-vs-contractor tilt) weak-PASS, **Gate 2 (market-neutral pair as a hedge)
   FAIL** → use a conditional-QQQ-short instead. See `docs/grid-equipment-value-chain-results.md`.
2. **Backlog growth-surprise event study** (`backlog_factor/`) — on branch
   **`backlog-surprise-factor`, NOT merged**. Pre-registered gate **FAIL** across 1,778 events / 91
   names: the Q5−Q1 abnormal CAR is non-monotone, is a ~1-week bump that reverses to −1% by 63 days,
   and the signal's direction carries no information. See `docs/backlog-surprise-factor-results.md`.

**The through-line:** three independent attempts to find a cross-sectional stock-selection edge in/around
the AI-power theme have now failed — the disclosed-backlog-growth tilt (grid_equipment_basket Step 2),
the value-chain tilt (Step 1 of this session), and the backlog-*surprise* factor at real breadth. The
only robust *positive* result anywhere is **risk management on the equal-weight basket** (trend gate +
~20% vol target: Sharpe 1.36 → 1.85, DD −41% → −18%, in-sample) — which the client has rejected as too
mainstream (a public Fidelity explainer recommends the same names).

**Open strategic question for the next session:** keep drilling for an edge that may not exist in this
theme at this sample size, or accept "the theme, risk-managed" as the honest answer and move the search
somewhere less picked-over. The three remaining proposals below are the "keep drilling, but
differently" options.

---

## Repo / branch state

| Branch | Contents | Status |
|---|---|---|
| `main` @ `516f05e` | grid_equipment_basket incl. the value-chain reframe (`--construction value-chain-tilt \| pair`, `--drop-winners`), hedges module, results docs | current |
| `backlog-surprise-factor` @ `e1be695` (8 commits off `main`, ff possible) | `backlog_factor/` module: SEC XBRL RPO discovery+fetch, growth-surprise signal, cross-sectional event study w/ month-clustered t-stats + pre-registered gate, 109-name curated universe, 25 tests, negative-result writeup | **decision pending** — merge to keep the negative result + reusable code in history, or shelve |

Working tree also has pre-existing untracked cruft unrelated to this work: `dc_demand_basket/`,
`docs/handoff_2026-08-29-grid-equipment-basket.md`, `grid_equipment_basket/2026-08-28-...design.md`,
and modified `grid_resilience/**/__pycache__/*.pyc`. Leave them.

Full repo test suite: **259 passing** (`.venv/bin/python -m pytest -q`).

---

## What we now believe (accumulated evidence)

- **No cross-sectional stock-selection edge inside the grid/AI-power theme** at the sample we have
  (~3.5–7 yr, one macro cycle). Backlog level, backlog growth, backlog *surprise*, and a
  maker-vs-price-taker value-chain split have all been tried and all failed or come out weak.
- **The theme is fully mainstream.** GE Vernova / Eaton / Quanta are named beneficiaries in a Fidelity
  learning-center article; they sit in every AI-infra ETF.
- **Risk management is the only robust value-add** found: "hold less when the trend is down and vol is
  high." It generalizes better than any fitted stock signal but its entire back-tested edge is turning
  the one −42% DeepSeek drawdown into ~−9%; in a window without a −40% event it reads as drag. Not built
  as a committed module — still just in-sample notes in `docs/handoff_2026-08-29-grid-equipment-basket.md`.
- **The conditional-QQQ-short works better than a clever market-neutral pair.** It's built and validated
  in `grid_equipment_basket/hedges.py` (`conditional_short_mask` / `conditional_short_overlay`).
- **Backlog-surprise, cross-sectionally, is a real "no."** Weak, machinery-only, ~1-week, non-monotone
  positive reaction that reverses. Negative-surprise names drift up *more* than positive-surprise names
  over 63 days. This is now settled at 1,778 events — not a small-sample artifact.

---

## Remaining proposals (raised this session, not built)

Recommended order: **B first** (best breadth × tractability × differentiation, highest data cost),
then **D-reframed** (cheapest — all data exists), then **A** (most differentiated, breadth-capped,
fiddly data). Each gets its own spec → plan → gated build, same discipline as this session.

### B — Transmission rate-base compounders

- **Thesis:** regulated utilities whose FERC-jurisdictional **transmission rate base is compounding**
  (they must build lines for new large loads) are the under-owned, boring expression of the buildout.
  Long the fast transmission-rate-base growers, short the flat / merchant-drag ones.
- **Universe:** ~35–45 US regulated electric & multi-utilities. `grid_resilience`'s ~20-name utility
  universe + zone maps is a starting point; widen to the full regulated set.
- **Signal:** transmission-capex ÷ rate-base growth (trend), or transmission rate base as a rising % of
  total rate base. **Data cost is the crux** — sources: 10-K MD&A segment capex guidance, **FERC Form 1**
  (transmission rate base is a Form 1 line; clunky but structured — FERC eLibrary / Form 1 database
  dumps), rate-case filings for forward rate-base.
- **Breadth:** ~40 names × ~60 quarters (2010–2026); slow-moving signal → quarterly/semi-annual
  rebalance, low turnover.
- **Differentiation:** genuinely under-covered; not in any AI-infra ETF. **Required additivity check:**
  vs a plain utility-beta / dividend-yield / low-vol factor — it must not just be a repackaged bond proxy.
- **Watch-outs:** rate-case outcomes are lumpy and political; regulatory lag; signal is public in 10-Ks
  so partly arbitraged.

### D — Grid-data macro nowcast, reframed as a cross-sectional sensitivity factor

- **Original idea:** use the LMP / congestion / reserve-margin / DC-load signals as a real-time read on
  industrial electricity demand and trade a broad risk-on/off rotation. **As a timer this fails the
  breadth test** (one autocorrelated bet per month).
- **Breadth reframe:** cross-sectional. Rank every stock (or every industrial) by its historical
  return-**sensitivity (beta)** to the grid-demand nowcast, long high-sensitivity / short low-sensitivity,
  monthly rebalance. Hundreds of names × monthly = breadth.
- **Data:** entirely built already — `grid_resilience` (LMP, congestion, reserve margin, multi-ISO GSI),
  `dc_demand_basket` (DC-load signal, hyperscaler deals). Only need a broad price panel.
- **Differentiation:** maximally non-consensus — the grid-stress plumbing as a style factor, not a stock
  picker. **Cheapest to try** (no new data).
- **Watch-outs:** the underlying grid signals validated *mixed-to-negative* on their own in
  `grid_resilience` (see `docs/compact_2026-08-26-original-thesis-reexamination.md`); a factor on a weak
  signal inherits the weakness. "Nowcasting the cycle" is also a crowded macro game. Set an honest
  pre-registered IC bar and expect to fail it.

### A — Zone-matched congestion pair

- **Thesis:** data-center load concentrating in specific ISO zones raises local congestion / capacity
  charges; the incumbent power-hungry industrials in those **same zones** eat the cost and can't pass it
  through. **Long** zone utilities/IPPs absorbing the load, **short** the zone's power-cost-exposed
  industrials (aluminum smelters, chlor-alkali, steel EAF, industrial gas, ammonia).
- **Data:** interconnection-queue MW by zone (`grid_resilience` has queue data +
  `get_raw_interconnection_queue`), LMP/congestion by zone (`grid_resilience` multi-ISO LMP), and the
  fiddly bit — **map each industrial's plants to ISO zones** (EIA-860 plant coordinates, or 10-K
  facility lists).
- **Breadth:** capped — ~6–8 zones with meaningful DC growth × a handful of tradeable names each. This is
  the **weakest on the breadth constraint**. Broaden by making it continuous: rank every industrial by
  "DC-load-growth exposure in its production zones − its power-cost pass-through ability," trade the tails.
- **Differentiation:** high — nobody models the loser side zone-by-zone. Note: the *naked* industrials
  short already failed (+0.3–0.45 corr, fell with the basket); zone-matching against the
  load-capturing utility is the untested twist.
- **Watch-outs:** facility-to-zone mapping is error-prone; power cost is a small line item for many
  industrials; "can't pass through" varies by product.

---

## Reusable infrastructure (built, tested, generic)

- `backlog_factor/data/rpo.py::discover_rpo_filers(quarters)` — SEC `frames` API → any-XBRL-concept
  filer list, SIC-filtered. Generic universe discovery for any tagged concept.
- `backlog_factor/event_study.py` — `event_pairs`, `quintile_car_table`, `clustered_tstat` (month-
  clustered), `monotonic`, `subperiod_split`, `by_industry_signs`, `evaluate_gate`. Drop-in event-study
  harness for any (signal, filing-date) panel.
- `backlog_factor/signal.py::neutralize_cross_section(raw, industry_group, log_mktcap)` — winsor →
  z-score → regress out industry dummies + size → residual. Generic cross-sectional neutralization.
  Also `name_surprise_series` (growth-minus-trailing-mean surprise + history/span/definition-break
  guards) and `staleness_decay`.
- `grid_equipment_basket/hedges.py` — `conditional_short_mask` / `conditional_short_overlay` (the
  QQQ-short that beat the pair), `find_drawdown_episode`, `episode_drawdown`, `risk_match_weight`,
  `simulate_pair`, `pair_overlay`.
- `grid_equipment_basket/margin_data.py` — SEC XBRL *flow*-concept fetch (revenue / gross profit,
  quarterly-fact filtering, tag-union across `Revenues` / ASC-606 tag).
- `grid_resilience/` — multi-ISO LMP, congestion fraction, reserve-margin tightness, GSI, DC-load
  signal, hyperscaler deals, ISO zone maps (`TICKER_NODE_MAP`), interconnection-queue fetch,
  `factor/neutralize.py`, `portfolio/backtest.py::compute_ic`.
- `grid_resilience/data/cache_utils` — monthly-parquet WIDE price cache (used by both basket fetchers;
  `backlog_factor/data/prices.py` is a re-pointed copy of `grid_equipment_basket/data/prices.py`).

---

## Method discipline that has worked (keep it)

- **Pre-register the gate before running.** Both this session's builds had explicit pass/fail criteria
  written into the spec; both failed honestly and were written up as negative results rather than
  massaged. That is the point.
- **Phase / gate expensive data collection.** `backlog_factor` never hand-collected the non-GAAP-backlog
  names because the XBRL-only event study failed first.
- **Breadth check.** The backlog-surprise result is trustworthy *because* it's 1,778 events, not 9 names.
  Apply the same bar to B/D/A — if a proposal can't reach a few hundred independent bets, say so up front.
- **Additivity check for any "new" factor** — regress its returns on the obvious known factors (revision
  momentum, price momentum, quality, utility-beta/yield) and require residual alpha. "Less-mined series"
  is a hollow claim without it.

## Suggested next steps

1. Decide the `backlog-surprise-factor` branch's fate (merge to keep the negative result + reusable
   code, or shelve).
2. Pick ONE of B / D / A and spec it (recommend B; D if data-cost-averse; A only if you accept the
   breadth cap). Same gated, pre-registered discipline.
3. If B/D/A also come up empty, the honest recommendation is "own the theme, risk-managed" (build the
   trend-gate + vol-target overlay as a real module) or take the search to a less-crowded area entirely.
