# Triage — Proposals B / D / A feasibility probes (2026-08-31)

Follows `docs/handoff_2026-08-31-strategy-proposals.md`. The handoff recommended
picking ONE of B / D / A and running a full gated build. This session instead
ran cheap feasibility probes on all three first, because three prior full gated
builds in this theme have come up empty and the binding uncertainty for each
remaining proposal is *whether the data/signal supports it at all*.

Probe code is a real repo module on branch `strategy-triage-bda`:
`grid_demand_factor/` (+ `tests/grid_demand_factor/`, 12 tests).

---

## Verdict summary

| Proposal | Data feasible? | Breadth | Recommendation |
|---|---|---|---|
| **B — transmission rate-base compounders** | **YES — confirmed live** (PUDL FERC Form 1 parquet, free, loads in <2 s, 1994–2025) | Borderline (~40 names, annual) | **BUILD — sole survivor.** Spec next, with a hard additivity gate. |
| **D — grid-demand sensitivity factor** | Nowcast exists (ERCOT+PJM) but is near-orthogonal to equity returns | High in principle | **DO NOT BUILD — pre-registered gate FAILED, robustly** (IC t = −0.47; 6/6 variants fail). |
| **A — zone-matched congestion pair** | Winner side yes; loser side structurally thin | **FAILS the breadth floor** | **DO NOT BUILD.** |

---

## B — Transmission rate-base compounders

**Feasibility: CONFIRMED, live-tested this session.**

- `pd.read_parquet("https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/core_ferc1__yearly_plant_in_service_sched204.parquet")`
  returned **298,611 rows × 14 cols in 1.8 s**. No auth, no cost (Catalyst
  Cooperative PUDL, CC-BY-4.0). Fits the repo's existing parquet-cache pattern.
- Columns: `utility_id_ferc1`, `report_year` (**1994–2025**), `ferc_account_label`,
  `ferc_account`, `starting_balance`, `additions`, `retirements`, `ending_balance`,
  `row_type_xbrl` (flags subtotal rows).
- Transmission plant is present at line-item granularity — `towers_and_fixtures_`,
  `overhead_conductors_and_devices_`, `station_equipment_`, `poles_and_fixtures_`,
  `land_and_land_rights_` `…_transmission_plant`, plus a `transmission_plant`
  subtotal (FERC accounts 350–359). "Regional transmission and market operation
  plant" (≈ accounts 380s) is a separate labeled block.
- Companion tables: `core_ferc1__yearly_utility_plant_summary_sched200` (total-plant
  denominator → transmission-as-%-of-rate-base trend);
  `core_ferc1__yearly_transmission_lines_sched422` (circuit miles, physical
  cross-check). PUDL ships `utility_id_pudl.csv` mapping the 432 `utility_id_ferc1`
  values → EIA utility id + name.

**Why FERC Form 1 transmission plant ≈ transmission rate base.** Post-EPAct-2005,
most large IOUs recover FERC-jurisdictional transmission under **formula rates**
keyed directly to booked transmission plant (350–359) net of depreciation, plus
CWIP and less ADIT. So net transmission plant in service tracks the FERC
transmission rate base closely — far more closely than booked plant tracks the
state-jurisdictional distribution/generation rate base.

**The one real manual cost — a subsidiary→parent→ticker map.** IOUs file Form 1 at
the operating-company level (AEP files ~7: Appalachian Power, Ohio Power, Indiana
Michigan Power, AEP Texas, PSO, SWEPCO, Kentucky Power, Transource…). Need a
hand-built static dict of ~40 listed parents × 1–8 filers each — same shape and
effort as the existing `TICKER_NODE_MAP`. This is the gating task and it is
bounded.

**Breadth — borderline, flag it in the spec.** ~40 regulated electric / multi-
utilities × annual data × slow-moving signal ≈ 600 name-years gross but only
~120–160 effectively-independent bets after autocorrelation. Below the "few
hundred" bar the handoff sets. Partial mitigations: 10-Q MD&A capex-guidance
updates add intra-year refreshes; rate-case filings add event-style updates. It
is a low-turnover regulated-utility value strategy, so the strict breadth bar
arguably binds less — but say so explicitly rather than hand-wave it.

**Additivity — the make-or-break gate, NOT resolved by triage.** Must regress the
transmission-growth-sorted long/short return on: utility beta, dividend yield,
low-vol, size, price momentum, and a plain "utility capex intensity" factor;
require residual alpha with t ≥ 2. The DOE National Transmission Needs Study names
AEP, Eversource, NextEra, ITC/Fortis and Berkshire Hathaway Energy as the
transmission leaders — a spread of dividend yields, so the growth dispersion is
*plausibly* not just a yield repackage. Must be shown, not assumed. Caveat: ITC
Holdings, the cleanest listed pure-play, is now a private Fortis subsidiary, so
the long book loses its purest name.

**Recommended pre-registered gate for the full build:**
1. Rank-IC of transmission-rate-base growth vs forward 12-month relative return:
   mean IC ≥ 0.03, t ≥ 2 over 2012–2025.
2. Additivity: residual-alpha t ≥ 2 after the factor panel above.
3. Q5−Q1 (fast vs flat transmission growers) annualized Sharpe ≥ 0.4, monotone
   quintiles.
All three pass → build the strategy module. Any fail → negative result, stop.

---

## A — Zone-matched congestion pair

**Feasibility: the loser side is structurally too thin to trade.**

- DC-load concentration is real but narrow. PJM **Dominion zone** (Northern
  Virginia) dominates — PJM's 2025 load forecast shows **> 20 GW** of data-center
  growth in that one zone by 2037, vs ~5.7 GW in the 2022 forecast. Secondary
  zones: PJM AEP-Ohio, PJM ComEd (Chicago), PJM PPL (eastern PA), ERCOT (diffuse
  and nodal — 198 GW of large-load interconnection *applications* in Q1 2026 alone,
  but spread statewide), and Western pockets (AZ, PNW/Oregon, Nevada). ≈ 6–8
  zones, matching the handoff's estimate.
- **Winner side** (~10–12 tradeable): D, AEP, EXC, PPL, plus IPPs TLN / CEG / VST /
  NRG, and PNW / POR / SO for the Western and Southeast pockets.
- **Loser side collapses** once you require all three of {genuinely power-cost-
  exposed} ∧ {plants in a DC-growth zone} ∧ {weak pass-through}:
  - **Primary aluminum:** only **4 US smelters exist** (2 Century, 2 Alcoa), 2
    idled; located in KY / SC / MO — **not in any DC-growth zone**. Power ≈ 40% of
    cost, but the names are not where the congestion is.
  - **Chlor-alkali (OLN, WLK):** the most power-exposed (~40–50% of cash cost) but
    Gulf Coast (TX / LA) — a secondary ERCOT story at best, not the NoVA / Ohio /
    Chicago corridors.
  - **EAF steel (NUE, STLD, CLF):** power is only ~5–7% of COGS, pass-through via
    the steel price is real, and mills are mostly SERC / non-ISO Southeast.
  - **Industrial gas (LIN, APD):** air-separation is power-heavy but supply
    contracts carry explicit power pass-through (take-or-pay). "Can't pass
    through" fails.
  - **Nitrogen (CF):** uses natural gas as feedstock and fuel, minimal grid power —
    wrong input entirely.
  → ~3–6 weak-fit loser names.
- **Discrete pair universe: ~10 winners × ~5 losers → well under the ~25–30
  independent-pair breadth floor. FAILS.**
- **Continuous version** (rank every industrial by "zone DC-exposure − pass-through
  ability") is *constructible* — EPA GHGRP facility coordinates (~8k large
  emitters, catches most chemical / steel / aluminum plants) + ISO / utility-
  territory shapefiles (HIFLD / EIA) + point-in-polygon + a hand-scored
  pass-through index — but it is **the most expensive data build of the three**,
  for the weakest prior, on a short thesis (power-exposed industrials) that
  **already failed naked** (+0.3–0.45 correlation with the basket, fell with it).

**Recommendation: DO NOT BUILD.** Lowest priority. If ever revisited, only the
continuous version, and only after B (and D, if it survives) are exhausted.

---

## D — Grid-demand sensitivity factor

**Nowcast reality check (from the cached GSI series):**

- Full daily history exists only for **ERCOT + PJM** (2018–2025). MISO starts
  2023, CAISO 2023-02, SPP 2025. A "multi-ISO nowcast" is really a 2-ISO nowcast.
- The GSI is a **weather-driven stress / scarcity index** (LMP z-score +
  congestion fraction + reserve tightness + a near-always-on event flag), *not* an
  industrial-electricity-demand nowcast. `reserve_tightness_z` — the most
  demand-level component — is pinned at 0.0 for ERCOT / MISO / SPP much of the
  time (load-capacity proxy limitation).
- The cached series end 2025-12-31 — stale by 8 months at time of writing.

**Pre-registered gate (written before the run):** D is worth a full gated build
only if BOTH — (1) |mean monthly Spearman rank-IC| ≥ 0.03 and |t| ≥ 2.0; AND
(2) Q5−Q1 annualized Sharpe ≥ 0.35 with monotone quintiles.

**Probe run** (`grid_demand_factor/probe.py`). The S&P 100 yfinance pull was
blocked by hard rate-limiting, so the probe was run against the **dense 101-name
industrial universe already cached** at `backlog_factor/data/cache/prices_*.parquet`
(machinery / electrical-equipment / construction / defense / semi-cap, 2019–2025,
59 monthly cross-sections) — arguably the *right* test population for D, since
these are the electricity-demand-exposed industrials the reframe targets. Method:
36-month rolling OLS β of monthly returns on monthly Δnowcast (ERCOT+PJM GSI mean);
monthly quintiles by β; next-month equal-weight returns.

Result — **GATE FAILED on both conditions:**

| metric | value | bar | |
|---|---|---|---|
| mean monthly rank-IC | **−0.0086** | \|·\| ≥ 0.03 | FAIL |
| rank-IC t-stat | **−0.48** | \|·\| ≥ 2.0 | FAIL |
| rank-IC % months positive | 47% | — | |
| Q5−Q1 annualized Sharpe | **0.12** (t = 0.26) | ≥ 0.35 | FAIL |
| quintile means (Q1→Q5) | +2.18 / +1.97 / +1.71 / +1.99 / +2.39 % | monotone | FAIL (flat) |

**Robustness — the null holds across every reasonable variant:**

| nowcast / window | IC mean | IC t | Q5−Q1 Sharpe | monotone |
|---|---|---|---|---|
| Δnowcast, 36 m, ERCOT+PJM (pre-registered) | −0.0084 | −0.47 | +0.07 | no |
| level, 36 m, ERCOT+PJM | +0.0186 | +0.98 | +0.68 | no |
| Δnowcast, 30 m | −0.0121 | −0.73 | −0.32 | no |
| Δnowcast, 24 m | +0.0007 | +0.04 | −0.15 | no |
| Δnowcast, 36 m, ERCOT-only | −0.0093 | −0.57 | −0.13 | no |
| Δnowcast, 36 m, PJM-only | −0.0083 | −0.50 | +0.20 | no |

All six: |IC t| < 1, no monotone quintile ordering, no Sharpe with t-support.

**Structural reason it fails.** Monthly Δnowcast has **near-zero contemporaneous
correlation with any equity aggregate** — XLI −0.06, ITA −0.10, XHB −0.09,
PAVE −0.01, SOXX +0.01. A style factor built on cross-sectional *sensitivity* to a
regressor that barely co-moves with equities is estimating noise. Compounding
factors: the de-seasonalised nowcast has lag-1 autocorrelation 0.63 (few
independent monthly innovations over the sample → unstable betas), and full daily
GSI history exists only for ERCOT + PJM. Note `month_r2 = 0.072` — the GSI *level*
is **not** mostly calendar-seasonal weather, which was a point in D's favour, but
it does not overcome the missing equity linkage.

**Recommendation: DO NOT BUILD.** This matches the handoff's own prior (the
underlying grid signals validated mixed-to-negative alone; "nowcasting the cycle"
is crowded). If B fails and D is ever revisited, it needs (a) a demand-*level*
nowcast built from actual ISO load / EIA-930, not the stress GSI, and (b) a price
panel from a non-yfinance source.

---

## Recommended next step

**B is the sole survivor.** Spec it through the normal brainstorming → spec → plan
→ gated-build flow, carrying its pre-registered gate (rank-IC t ≥ 2, residual
alpha t ≥ 2 after the known-factor panel, Q5−Q1 Sharpe ≥ 0.4 monotone). Treat
**A** and **D** as closed negative/no-go results.

If B also fails its gate, four cross-sectional stock-selection attempts in this
theme will have come up empty (backlog-growth tilt, value-chain tilt,
backlog-surprise factor, transmission rate-base) plus two no-go's from this
triage — at which point the honest call is "own the theme, risk-managed" (build
the trend-gate + vol-target overlay as a real module) or move the search to a
less-picked-over area.

Branch bookkeeping: this triage lives on `strategy-triage-bda` (off `main`). The
`backlog-surprise-factor` branch fate is still undecided (merge to keep its
negative result + reusable event-study harness in history, or shelve).
