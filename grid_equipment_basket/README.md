# Grid Equipment Suppliers Thematic Basket

Long-only basket of US-listed grid / data-center electrical-infrastructure suppliers —
a direct-revenue bet on the power buildout, as opposed to the utility-demand bets in
`grid_resilience/` and `dc_demand_basket/`.

Spec: ../docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md
Step 1 results (full numbers): ../docs/grid-equipment-basket-step1-results.md

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

## Step 1 result

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
cherry-picking, and the GRID/PAVE comparison rules out the "this is just the theme" reading.
Full numbers and caveats: `../docs/grid-equipment-basket-step1-results.md`.

## Return convention

All maths here uses **SIMPLE returns** (`pct_change`, `(1+r).prod()`), unlike `grid_resilience`
which uses log returns. A log-based recompute will not tie out exactly — that is expected.

## Run

```
python -m grid_equipment_basket --start 2023-01-01      # primary window
python -m grid_equipment_basket --prior-regime          # 2020-2022 panel
python -m grid_equipment_basket --tilt backlog          # Step 2 (only after the gate PASSED; module built in Tasks 6-8)
```

Outputs (to `--output`, default `./output_grid_equipment`): `metrics.csv`,
`basket_returns.csv`, `performance.png`. Live yfinance fetch was validated during
implementation (cold == warm == uncached, 124/124 rows on the Task 2 check; primary and
prior-regime runs reproduce bit-for-bit off the warm cache).

## Phase 2 — cross-sectional factor (NOT built here)

Only if verification ever lands 8+ names with clean, comparable, segment-level backlog and
multiple years of quarterly history does a cross-sectional backlog-growth-surprise factor
become defensible. That needs its own spec — do **not** reintroduce `build_factor()` /
z-scoring / IC machinery into this basket. This is scoped as a thematic basket, matching
`dc_demand_basket`.
