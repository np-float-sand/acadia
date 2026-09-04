# Category-demand → maker-rotation probe — results (FAILED)

**Date:** 2026-09-03
**Scopes:** `docs/handoff_2026-09-03-category-demand-rotation.md`
**Context:** `docs/handoff_2026-09-01-grid-buildout-long-short.md`, `docs/va-transmission-filings-probe-results.md`
**Code:** `grid_equipment_basket/data/category_demand.py` (FRED fetcher, kept) · probe script `scratchpad/catrot/probe.py` (throwaway)

---

## TL;DR

- Built the national equipment-category demand panel (Census M3 electrical-equipment new
  orders / shipments / **unfilled orders** / inventories, Fed IP NAICS 3353, BLS PPI for
  transformers / switchgear / energy wire, core-capex new orders), monthly 1990→2026, from the
  keyless FRED CSV endpoint.
- Hand-built a category→name exposure matrix for the 9 from 10-K segment mix. **26% of the book on
  average — and 45–70% of GEV / VRT / FLNC — is "unmapped"** (gas turbines, data-center cooling,
  storage, solar EPC): no public monthly category series touches it.
- Composite tilt (Signal A) and the transformer-vs-switchgear rotation (Signal B) **fail the
  pre-registered bar.** They add a small amount in 2019–22 (rank-IC t ≈ 2.4, Sharpe +0.08 → +0.26
  as bet size rises) and **nothing in 2023–26** (rank-IC t = 0.6, Sharpe flat at every bet size).
- **Verdict: FAILED — logged, not built.** Kill criteria K2 (regime — works pre-AI, dead in the AI
  regime that matters) and K3 (fwd-1m IC t = 0.78; fwd-3m IC passes pooled but is 100% from the
  2019–22 sub-window). This is ~attempt 16; the accumulated evidence says no reachable data
  cross-sections this book.

---

## 1. Data (kept: `grid_equipment_basket/data/category_demand.py`)

`fetch_category_demand()` → monthly wide frame, month-end index, 1990→2026 (~5-week M3 lag), no NaN gaps:

| column | FRED id | what |
|---|---|---|
| `ee_new_orders` / `ee_shipments` / `ee_unfilled_orders` / `ee_inventories` | `A34SNO` / `A34SVS` / `A34SUO` / `A34STI` | Census M3, Electrical Equipment, Appliances & Components (NAICS 335), $M SA |
| `ip_naics3353` / `ip_naics335` | `IPG3353S` / `IPG335S` | Fed G.17 industrial production (NAICS 3353 = transformers+switchgear+motors+relays) |
| `ppi_transformers` / `ppi_switchgear` / `ppi_energy_wire` | `PCU335311335311` / `PCU335313335313` / `PCU335929335929` | BLS PPI — **only monthly category split, price not volume** |
| `core_capex_new_orders` | `NEWORDER` | macro capex context |
| derived | — | `ee_book_to_bill`, `ee_backlog_months`, `ee_inv_to_ship`, `ppi_tx_vs_sg` |

Confirmed: **no public monthly volume series below NAICS 3-digit** for electrical equipment; the
transformer / switchgear split exists only as PPI.

## 2. Exposure matrix (hand-built, 10-K segment mix, 2026; rows sum to 1)

| name | transformers | switchgear | energy_wire | broad_EE | TD_construction | unmapped |
|---|---|---|---|---|---|---|
| ETN | 0.05 | 0.40 | 0.00 | 0.45 | 0.05 | 0.05 |
| HUBB | 0.15 | 0.15 | 0.20 | 0.35 | 0.10 | 0.05 |
| GEV | 0.20 | 0.12 | 0.03 | 0.15 | 0.05 | **0.45** |
| VRT | 0.00 | 0.20 | 0.05 | 0.20 | 0.00 | **0.55** |
| NVT | 0.00 | 0.10 | 0.15 | 0.70 | 0.05 | 0.00 |
| PWR | 0.00 | 0.00 | 0.05 | 0.03 | 0.87 | 0.05 |
| MYRG | 0.00 | 0.00 | 0.05 | 0.05 | 0.75 | 0.15 |
| PRIM | 0.00 | 0.00 | 0.05 | 0.05 | 0.55 | 0.35 |
| FLNC | 0.00 | 0.00 | 0.00 | 0.25 | 0.05 | **0.70** |

`unmapped` = gas turbines + wind (GEV), data-center cooling & services (VRT), grid storage (FLNC),
solar EPC + pipeline (PRIM), C&I construction (MYRG). It has no public monthly demand series;
score = 0.

## 3. Signals & results

**Signal A — composite.** Per-category z-scored demand momentum (PPI momentum for transformers /
switchgear / wire; `ee_new_orders` momentum for broad_EE; `ee_unfilled_orders` momentum for
TD_construction), lagged 2 months, mapped through the matrix to a per-name score; tilt =
equal-weight + k·(score − mean), clipped ≥ 0, renormalised; monthly rebalance.

| lookback × k | 2019–22 Sharpe (tilt / EW) | 2023–26 Sharpe (tilt / EW) | turnover |
|---|---|---|---|
| 6mo, k=0.03 | +0.93 / +0.89 | +1.44 / +1.43 | 0.3× |
| 6mo, k=0.06 | +0.97 / +0.89 | +1.44 / +1.43 | 0.6× |
| 6mo, k=0.20 | +1.07 / +0.89 | +1.45 / +1.43 | 1.6× |
| 6mo, k=0.35 | +1.15 / +0.89 | +1.47 / +1.43 | 2.3× |

Headline (6mo, k=0.06): CAGR 21.2% vs 19.3% (2019–22), **50.6% vs 51.4% (2023–26 — loses)**.
Biggest average bet: ETN 0.137 vs 0.111 equal-weight.

**Signal B — transformer-vs-switchgear rotation.** tilt ∝ (transformer_expo − switchgear_expo) ·
z(Δlog PPI_transformers − Δlog PPI_switchgear). Same pattern: 2019–22 +0.97–1.01 / +0.89;
2023–26 +1.42–1.44 / +1.43.

### Pre-registered checks

| check | bar | result |
|---|---|---|
| Sharpe, both windows, on a plateau | tilt > EW on both | 2019–22 yes (+0.08); **2023–26 no (+0.01, and loses on CAGR)** |
| rank-IC, score vs fwd-1m return | pooled \|t\| ≥ 2 | **t = +0.78** (n = 98) — FAIL |
| rank-IC, score vs fwd-3m return | pooled \|t\| ≥ 2 | t = +2.17 pooled, **but 2019–22 t = +2.38 / 2023–26 t = +0.57** — the signal is entirely pre-AI |
| not a low-vol repackage | residual alpha \|t\| ≥ 1.5 after LMH + MKT | **t = +0.87** — FAIL (little excess return to explain) |
| force VRT/NVT/FLNC → 1/9 | still beats EW both windows | 2019–22 +0.92/+0.89; 2023–26 +1.43/+1.43 — *not* a "hold-less-VRT" effect, but no edge either |
| turnover ≤ 1.5×/yr | — | ok below k=0.2; the pre-AI edge only appears at k ≥ 0.2 (turnover 1.6–2.3×) |

## 4. Why it fails — and it's the same wall

1. **The AI regime does not reward category rotation.** 2023–26 the whole complex re-rated
   together on the data-center narrative; cross-sectional dispersion among the 9 came from *which
   name had the best data-center-power story* (VRT, GEV), not from which equipment *category* had
   accelerating national orders. Transformer PPI, switchgear PPI, and M3 backlog all rose together
   → no discriminating power (IC t = 0.6).
2. **≈26% of the book — 45–70% of GEV / VRT / FLNC — is unmapped.** No public monthly series
   covers data-center power & cooling or grid storage, so the signal structurally cannot tilt
   toward the names (VRT, GEV) that drove returns. Same structural gap that killed the VA
   work-type tilt.
3. **The small pre-AI signal is plausibly real but useless.** In a normal industrial cycle (2019–22)
   "which category's orders are accelerating" has weak forecast power for the levered makers
   (IC t ≈ 2.4). But 2019–22 is exactly the regime where the *whole trade* has negative Sharpe
   (handoff §1.4), so a within-book tilt that helps there does not help.
4. **Census M3 is national, lags ~5 weeks, and is in every industrials-analyst model** — any
   signal in it is largely priced.

## 5. Recommendation

- **Keep** `category_demand.py` — a clean, reusable electrical-equipment-demand pull (book-to-bill,
  backlog months, transformer/switchgear PPI). It is a fine *dashboard* series ("is national grid-
  equipment demand still expanding?") but **not a signal**.
- **Do not** build the point-in-time exposure matrix, the EEI capex-mix variant, or the 17-name
  extension. The AI-window IC is zero; extending the map won't change that.
- This is ~attempt 16. Fold into the standing conclusion (`docs/handoff_2026-09-01…` §7): ship the
  concentrated long/short as a discretionary style/relative-value position with the monitored
  dashboard; move new research to a less-arbitraged theme. The one avenue still open with a
  non-consensus prior is interconnection-queue *velocity* as a multi-year data-collection project
  (§8.6), plus the other session's data-center *commitment* signals (D/C/E,
  `docs/handoff_2026-09-03-transmission-project-filings.md` §6).
