# Handoff — Two-Layer Risk Overlay for the Grid-Equipment Basket (2026-08-31)

For the next session to **brainstorm layer 2** and decide how the two layers
combine. Layer 1 is already built.

## Context in one paragraph

Five pre-registered cross-sectional stock-selection attempts in the AI-power /
grid theme have all failed (backlog growth, value-chain split, backlog surprise,
grid-demand sensitivity, transmission rate-base — see
`docs/triage_2026-08-31-proposals-bda.md`,
`docs/transmission-rate-base-results.md`). The one robust positive is **risk
management on the equal-weight basket**. The client rejected "own the theme,
risk-managed" as too mainstream *when framed as stock list + trend/vol rules* —
the counter is that the stock list isn't the edge, a **physically-grounded read
on whether the buildout is actually accelerating** is. That read is layer 2.

## The basket (unchanged, `grid_equipment_basket/`, on `main`)

9 equal-weight names (ETN HUBB GEV VRT PWR MYRG NVT FLNC PRIM), quarterly
rebalance, 25% cap, long-only. Primary window 2023-01 → 2026-07: CAGR 59.8%,
Sharpe 1.36, **Max DD −41.3%** (the Nov-2024→Apr-2025 "DeepSeek" efficiency
scare). Beats XLI / GRID / PAVE / SPY on CAGR and Sharpe. Headline risk:
hindsight-selected universe, one macro regime, partly crowded.

## Layer 1 — trend gate + vol target — BUILT (branch `grid-equipment-overlay`)

`grid_equipment_basket/overlay.py` (+ 10 tests, `--overlay` CLI flag,
`OVERLAY_*` constants in `config.py`). Long-only, de-levering, decided at each
month-end and held through the next calendar month:

- `trend_gate(basket_prices, ma_days=100)` → 1.0 while month-end close ≥ its MA,
  else 0.0 (→ cash at rf 4%).
- `vol_target_scalar(basket_returns, lookback=20, target_vol=0.20, max_leverage=1.5)`
  → `min(target_vol / annualised_trailing_vol, 1.5)`.
- `apply_overlay(...)` → `exposure·basket_ret + (1−exposure)·rf_daily`,
  `exposure = gate · vol_scalar`.
- `overlay_report(start, end, prior_regime=False)` / `overlay_table(rep)` →
  basket-only vs +gate vs +gate+vol-target, mechanics, and a 5×3
  **parameter-plateau grid** (MA ∈ {50,75,100,125,150} × target-vol ∈
  {0.15,0.20,0.25}).

### Results (live)

| | primary 2023-01→2026-07 | | | prior regime 2020–2022 | | |
|---|---|---|---|---|---|---|
| config | CAGR | Sharpe | MaxDD | CAGR | Sharpe | MaxDD |
| basket only | 59.8% | 1.36 | −41.3% | 25.4% | 0.69 | −45.4% |
| + trend gate | 51.8% | 1.34 | −28.8% | 15.1% | 0.47 | −45.4% |
| + trend gate + vol-target | 43.6% | **1.58** | **−18.4%** | 6.1% | **0.20** | −41.5% |

Primary-window plateau: Sharpe 1.56–1.84 across all 15 cells — not knife-edge.
Mechanics: avg exposure 0.61, 16 % of days fully gated out, 6 gate flips / 3.5 yr.

### The finding that motivates layer 2

**Layer 1 does not generalise.** On the primary window it halves the drawdown
(−41 %→−18 %) and lifts Sharpe. On the 2020–2022 panel it *hurts* on every
measure — the drawdown barely moves (−45 %→−41 %), Sharpe collapses
(0.69→0.20), CAGR is gutted (25 %→6 %), and the plateau grid is 0.02–0.38
everywhere. The primary-window gain is an **artifact of that window's single,
clean, V-shaped −42 % episode** that the MA cross happened to catch. A price-only
trend/vol rule overfits the one drawdown it was tuned near. Layer 1 is the
honest de-risking baseline; it is not, by itself, an edge.

## Layer 2 — the grid-data regime signal — TO BRAINSTORM

**Thesis:** the buildout's bull case is "the grid is the bottleneck." Whether
that bottleneck is *tightening or easing* is measurable in the physical
electricity market before it shows up in equipment-maker margins — and that read
is not in prices, momentum, or analyst models. Use it to set the basket's
exposure (e.g. 0.5× / 1.0× / 1.5×) and/or the hedge state, layered on / replacing
layer 1's price rule.

### Feasibility of the candidate inputs (checked this session)

| input | backtestable over 2018–2026? | why |
|---|---|---|
| **#2 Zonal congestion / scarcity regime** — congestion fraction, reserve-margin tightness, LMP z-score, RT/DA spread in DC-heavy zones (Dominion/NoVA, AEP-Ohio, ERCOT-West, ComEd) | **Yes** | `grid_resilience` GSI + per-zone LMP/load/congestion data runs ERCOT+PJM from 2018; MISO/CAISO/SPP from 2023. `signals/grid_stress_index.py` already composites most of it; `data/utility_node_map.py` has the zone maps. |
| **#1 Interconnection-queue velocity** — Δ large-load MW in ISO queues (a 2–4 yr forward demand book) | **No (as-is)** | `data/ercot_large_load_data.py` seed = **11 monthly snapshots, 2025-05→2026-03**; `data/pjm_large_load_data.py` = one 2026 vintage; `data/hyperscaler_deals.py` = event dots. No multi-year history. Would need an archive dig (ERCOT TAC reports, PJM LLA-request decks) or years of forward accumulation. Powerful if it can be assembled — flag as a separate data-collection project. |
| **#3 RT/DA spread to split gen-levered (GEV, IPP-exposed) vs T&D/equipment names** | **Partly** | needs a `fetch_lmp_rt()` variant + an `ISO_RT_LOCATION_TYPE` config dict — CLAUDE.md says infra is ~90 % ready, not built. This is a *cross-sectional* use, on a genuinely untested axis (prior splits were all "which company is better-run"). |

### Open design questions for the brainstorm

1. **Signal, not timer.** Layer 1 proved a binary/continuous price rule overfits
   one episode. Does a grid-congestion regime avoid that, or does it just
   overfit the *same* DeepSeek episode from a different angle? Pre-register the
   test on **both** windows (2023–2026 and 2018–2022) from the start.
2. **What does layer 2 output?** A discrete regime label (tight / neutral /
   loose) → exposure multiplier? A continuous 0–1.5 exposure? A gate that only
   *overrides* layer 1 (e.g. "physical data says the bottleneck is intact →
   ignore the price gate and stay invested")?
3. **How do the layers compose?** Options: (a) layer 2 replaces layer 1's trend
   gate, keeping the vol target; (b) `exposure = layer1 × layer2`; (c) layer 2
   is a veto in one direction only; (d) layer 2 picks the hedge (conditional
   QQQ short via `hedges.py`) rather than the exposure.
4. **Which zones, and how are they weighted?** Equal-weight the DC-heavy ISO
   zones, or weight by disclosed data-center load / queue MW (`dc_demand_*`)?
5. **Lead/lag.** Congestion is contemporaneous-to-slightly-leading for prices;
   does it actually lead the *equipment stocks*, or move with them? Check the
   cross-correlation before building.
6. **Breadth honesty.** This is still one exposure decision per month — low
   breadth. The defence is "the client owns the theme regardless; layer 2 only
   sets the weight, and a physical-data reason to set it is worth paying for."
   Make that argument explicit or drop the project.
7. **Pre-registered bar.** e.g. layer-2-gated basket must beat both buy-and-hold
   *and* layer-1-only on a blended 2018–2026 metric (Sharpe and Calmar), on
   both sub-windows, with the regime signal's parameters frozen before the run.

### Reusable infra

- `grid_equipment_basket/overlay.py` — layer 1; `exposure_series()` and the
  `_month_hold` helper are the composition points for layer 2.
- `grid_equipment_basket/hedges.py` — `conditional_short_mask` /
  `conditional_short_overlay` (validated conditional QQQ short), drawdown-episode
  and risk-match helpers.
- `grid_resilience/signals/grid_stress_index.py` — `build_gsi`,
  `build_multi_iso_gsi`, per-zone GSI; `signals/stress_events.py` event catalog.
- `grid_resilience/data/grid_data.py` — LMP / load / congestion fetch per ISO;
  `data/utility_node_map.py` — ticker→ISO-zone maps.
- `grid_resilience/portfolio/backtest.py::compute_ic` — cross-sectional IC (for #3).
- `docs/handoff_2026-08-15-rt-da-spread-signal.md` — the #3 RT/DA design notes.

## Suggested first move for the brainstorm

Scope layer 2 as **#2 only** (zonal congestion/scarcity regime → basket
exposure), pre-registered on both windows, composed with layer 1 as
`exposure = layer1_vol_target × layer2_regime_multiplier` (drop layer 1's price
gate, since the physical signal is meant to replace it). Treat #1 and #3 as
separate later projects. If #2's regime signal can't beat layer-1-only on the
2018–2022 window, the honest answer is "run the basket smaller, no clever
overlay" — and that's a valid Phase-2 deliverable.
