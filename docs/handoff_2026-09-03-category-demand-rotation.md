# Handoff — category-demand → maker-rotation signal for the grid-equipment book (scope)

**Date:** 2026-09-03
**Context:** `docs/handoff_2026-09-01-grid-buildout-long-short.md` (the grid/DER long/short has no
signal-based edge after ~15 attempts), `docs/va-transmission-filings-probe-results.md` (VA/Dominion
CPCN filings confirm the buildout but fail as a timing signal and can't weight the 9 names —
work-type tilt backtest: Sharpe +0.01 vs equal-weight), memory `capex-cycle-pair-classifier`.

**The idea being scoped:** the physical buildout is not one homogeneous thing — in any given year
the marginal dollar is going disproportionately into *transformers*, or *switchgear*, or *line
construction labour*, or *data-center power & cooling*. If a public **category-level demand series**
can identify which sub-segment is accelerating, and each of the 9 (or the 17-name classifier book)
has a known **revenue exposure** to those sub-segments, that yields a **time-varying cross-sectional
tilt** — "overweight the names levered to the segment that is surging right now." This is the
handoff-§2-use-#3 ("work-type → supplier weighting") idea, moved off the VA dockets (too few
projects, biased to line construction, thin on equipment type, lags) and onto national
equipment-demand data.

---

## TL;DR

- **The category-demand data exists, is free, monthly, and deep** — but the granularity is coarse:
  a NAICS-335 *aggregate* (new orders / shipments / unfilled-orders / inventories, 1992→, ~5-week
  lag) plus a **transformer-vs-switchgear-vs-wire split that is only available as PPI (price),
  not volume**. No public monthly series for data-center power & cooling or grid storage.
- **The exposure map is hand-built** from 10-K segment disclosures that do not line up with NAICS.
  GEV has standalone financials only since April 2024.
- **The structural problem from the VA work-type tilt carries over:** VRT, NVT, FLNC map to
  categories (data-center power, storage) with no public demand series, so the signal can only ever
  rotate 6 correlated names (ETN/HUBB/GEV/PWR/MYRG/PRIM). Every prior within-book rotation
  (maker-vs-contractor, backlog-growth, un-crowd, VA work-type) failed to beat equal-weight.
- **Prior: low.** This is worth ~1–2 days as a probe with hard kill criteria, not a build
  commitment. If it clears both sub-window Sharpe bars *and* the IC bar *and* is not just a low-vol
  repackage, extend it; otherwise it is attempt ~16 and we log it and fall back to the discretionary
  dashboard.

**Relationship to `handoff_2026-09-03-transmission-project-filings.md` §6 (Part 2, C/D/E):** that
list is *demand/timing* signals ("is the theme accelerating" — utility capex-guidance revisions,
PUC data-center contracted-MW, county permits). This doc is orthogonal — *cross-sectional*
weighting ("which of the 9 is better right now"). They can be pursued independently; D (capex-guide
revisions) is the cheapest thing on either list and is the sensible first move overall.

---

## 1. Preliminary data-source investigation — what was checked

| source | series (verified reachable) | freq / history / latency | what it gives | limits |
|---|---|---|---|---|
| **Census M3 via FRED** | `A34SNO` new orders, `A34SVS` shipments, `A34SUO` **unfilled orders (backlog)**, `A34STI` inventories — all **NAICS 335 (Electrical Equipment, Appliances & Components)** | monthly, 1992→, SA, ~5-wk lag | book-to-bill (`NO/VS`), backlog momentum (`UO`, Δ`UO/VS`), inventory-to-shipments | **3-digit aggregate only** — no monthly volume split for transformers vs switchgear vs motors. Includes appliances (335 is broader than grid). |
| **Fed IP (G.17) via FRED** | `IPG3353S` (**NAICS 3353 Electrical Equipment** = transformers + switchgear + motors + relays), `IPG335S` (335) | monthly, 1972→ | production-volume index for grid-relevant equipment; cleaner than 335 (drops appliances) | still a blend of 4 sub-industries; no sub-3353 split. |
| **BLS PPI via FRED** | `PCU335311335311` **power & distribution transformers** (2005: 155 → 2026: **475**, ~3.1×), `PCU335313335313` **switchgear & switchboard apparatus** (158 → **395**, ~2.5×), `PCU335929335929` other energy wire & cable (192 → 361) | monthly, 2005→ (some to 1980s), ~3-wk lag | **the only monthly category split** — price as a scarcity/demand proxy; the transformer super-cycle and switchgear cycle are both visible and *not* synchronous | price, not volume; assumes the category is supply-constrained (true for transformers 2021–24, weaker generally); does not isolate HV vs MV. |
| **EEI "Industry Capital Expenditures" report** | functional split: transmission / distribution / generation, actuals + 5-yr projections. 2024 actual: **$32.6 B transmission, $60.2 B distribution** | annual (updated ~2×/yr), 2000→, free PDF | the utility-capex-mix instrument — rising T-share → EPC + HV transformer names; rising D-share → distribution-component names | annual; IOUs only; a projection, not an order book. |
| **FERC Form 1** | transmission plant additions per utility | annual, ~4-mo lag | bottom-up build of the same T-capex mix, by utility | annual; heterogeneous accounting. |
| **NEMA** | Electroindustry Business Confidence Index (EBCI) — current + 6-mo-ahead diffusion | monthly, public | sentiment lead | granular NEMA shipment indexes are members-only / paid — **not usable**. |
| **ISM Manufacturing** | PMI + "Electrical Equipment, Appliances & Components" industry commentary | monthly | qualitative confirm | not a quant series. |
| **Company 10-Ks** | segment revenue: ETN (Electrical Americas / Electrical Global), HUBB (Utility Solutions / Electrical Solutions), NVT (Enclosures / Electrical & Fastening / Thermal Mgmt), VRT (products vs services; Americas/APAC/EMEA), GEV (Power / Wind / Electrification), PWR (Electric Power Infrastructure), MYRG (T&D vs C&I), PRIM (Utilities / Energy) | annual/quarterly | the exposure-matrix inputs, point-in-time-ish from historical filings | segments ≠ NAICS categories; companies restate; GEV standalone only since 2024-04. |

**Finding:** enough to build a *coarse* category signal — a 335-aggregate demand/backlog series plus
a transformer/switchgear/wire PPI split plus an annual T-vs-D capex mix. Not enough for a
rich monthly volume-by-sub-segment series, and nothing for the data-center-power / storage
sub-segments.

---

## 2. The signal being scoped

### 2.1 Category → name exposure matrix (the crux; hand-built)

Rows = demand categories with a public series; columns = the 9 (extend to the 17-name classifier
book — POWL, ATKR, WCC, NXT, FSLR, ADRs — in phase 2). Each cell ≈ the fraction of the name's
revenue driven by that category, from the latest 10-K segment mix + product commentary. Illustrative:

| category (series) | ETN | HUBB | GEV | VRT | NVT | PWR | MYRG | PRIM | FLNC |
|---|---|---|---|---|---|---|---|---|---|
| Power/dist. transformers (`PCU335311`) | · | ▪ | ▪▪ | · | · | · | · | · | · |
| Switchgear & switchboard (`PCU335313`) | ▪▪ | ▪ | ▪ | ▪ | · | · | · | · | · |
| T&D construction labour (EEI T-capex, `A34SUO`) | · | ▪ | ▪ | · | · | ▪▪▪ | ▪▪▪ | ▪▪ | · |
| Energy wire & cable (`PCU335929`) | · | ▪ | · | · | · | ▪ | ▪ | ▪ | · |
| Distribution components (EEI D-capex) | ▪ | ▪▪▪ | · | · | ▪ | ▪ | ▪ | ▪ | · |
| Data-center power & cooling (**no public series**) | ▪ | · | · | ▪▪▪ | ▪▪ | · | · | · | · |
| Grid-scale storage (**no public series**) | · | · | ▪ | · | · | · | · | · | ▪▪▪ |

The two "no public series" rows are the structural hole (see §3). VRT/NVT/FLNC could be *proxied*
with hyperscaler capex — but `grid_equipment_basket/capex_signal.py` already built that and it
**failed as a signal** (`docs/handoff_2026-09-01…` §3.5: fires a year late).

### 2.2 Candidate signals (rank by expected value)

1. **Category-momentum composite → tilt.** z-score each category series' 3–6-month change (and its
   book-to-bill / backlog momentum for the 335 aggregate); combine into a per-name score via the
   exposure matrix; tilt equal-weight by the score, rebalanced semi-annually. The direct expression
   of the idea.
2. **Transformer-vs-switchgear relative PPI momentum.** When `PCU335311` momentum > `PCU335313`
   momentum, overweight the transformer-levered names (GEV, HV-adjacent HUBB); when reversed,
   overweight switchgear (ETN, HUBB, + POWL in the 17-name book). A clean 2-category rotation that
   sidesteps the volume-data gap.
3. **Utility capex-mix tilt.** From the EEI report (annual) or a quarterly proxy built from the
   top-20 IOUs' 10-Q capex + guidance: rising transmission share → EPC (PWR/MYRG/PRIM) + GEV;
   rising distribution share → HUBB/ETN. Lowest frequency, most economically direct.
4. **335-backlog as a basket-level exposure gauge** (not cross-sectional): scale total book
   exposure by `Δ(A34SUO / A34SVS)`. Low priority — every prior basket-timing signal has failed;
   include only as a control.

### 2.3 Universe

Phase 1: the 9 (`config.UNIVERSE`). Phase 2 (only if phase 1 clears): the 17-name classifier book
(adds POWL, ATKR, WCC, ABBNY, SBGSY, PRYMY, HTHIY, NXT, FSLR) — POWL and PRYMY in particular give
the switchgear and cable categories real cross-sectional dispersion the 9 lack.

---

## 3. Honest limitations

- **Same universe, one macro cycle, thin breadth.** 9–17 names, +0.5 to +0.85 correlated, ~3.7 yr
  of the data-center regime. Prior is low: **maker-vs-contractor tilt, backlog-growth tilt,
  un-crowd-the-universe, and the VA work-type tilt all already failed to beat equal-weight.**
- **The rotation can't reach 3 of the 9.** VRT, NVT, FLNC map to categories with no public demand
  series; they get a static weight (equal, or from a failed proxy). A category signal that can only
  move ETN/HUBB/GEV/PWR/MYRG/PRIM is rotating within a set that historically doesn't reward
  rotation — and it structurally can't tilt toward VRT/GEV, the basket's biggest 2023–26 winners
  (this is exactly why the VA work-type tilt failed).
- **Coarse category data.** The only monthly sub-segment split is *price* (PPI), which is a demand
  proxy only while the category is supply-constrained. The volume series stop at NAICS 3-digit.
- **Census M3 is heavily watched and lags ~5 weeks.** Any signal in it is likely already in
  industrials-analyst models and in price.
- **The exposure matrix is a judgment call** on segment disclosures that don't match NAICS, and it
  is only weakly point-in-time (GEV pre-2024 has no standalone segments; ETN and HUBB have
  restated).
- **PPI-as-demand is regime-specific.** The transformer PPI 3× move is a genuine shortage signal
  2021–24; in a normal market PPI ≈ input-cost pass-through, not demand.

---

## 4. Recommended first step (~1–2 days) + pre-registered kill criteria

### Build

1. **Data pull.** A small cache module (`grid_equipment_basket/data/category_demand.py` or
   `grid_resilience/data/`) fetching from FRED (`A34SNO`, `A34SVS`, `A34SUO`, `A34STI`, `IPG3353S`,
   `PCU335311335311`, `PCU335313335313`, `PCU335929335929`, `NEWORDER`) 2005→present, monthly,
   parquet-cached. Confirm no sub-3353 volume series exists (checked — it doesn't).
2. **Exposure matrix.** Hand-code the category→name matrix for the 9 from the latest 10-Ks; document
   every cell with the segment-revenue basis. Static first pass (as-of 2026), accept the
   look-ahead for the feasibility check only.
3. **Signal + backtest.** Build candidate signals #1 and #2 (§2.2); map to a semi-annual
   equal-weight tilt; backtest vs equal-weight of the same 9 over **both** sub-windows
   (2019-01→2022-12 and 2023-01→2026-08), on a parameter plateau (lookback ∈ {3,6,9,12} mo,
   tilt strength ∈ {0.5, 1.0}).

### Pre-registered bar (must clear ALL — mirror the layer-2 / RT-DA discipline)

- **Sharpe:** the tilt beats equal-weight of the same universe on Sharpe on **both** sub-windows,
  on a plateau (not a single lookback).
- **IC:** pooled rank-IC of the per-name category-momentum score vs forward 3-month return has
  |t| ≥ 2.
- **Not a low-vol repackage:** regress the tilt's excess return on a low-vol factor
  (SPLV−SPY, or the basket's own low-vol residual); require residual t ≥ 1.5.
- **Not just "hold less VRT":** re-run with VRT (and NVT, FLNC) forced to equal-weight; the tilt
  over the remaining names must still beat equal-weight on Sharpe on both windows. *(The VA
  work-type tilt's entire "edge" was underweighting VRT — this check is non-negotiable.)*
- **Turnover** ≤ ~1.5×/yr.

### Kill criteria (check cheapest first; any one → stop, log, fall back)

- **K1 (granularity):** the exposure matrix + available series cannot produce a tilt whose
  time-varying active share vs equal-weight exceeds ~0.10 → the data is too coarse → kill.
- **K2 (regime):** the tilt beats equal-weight on 2023–26 but **not** on 2019–22 → regime artifact
  (every prior tilt died here) → kill.
- **K3 (IC):** pooled rank-IC |t| < 1.5 → kill.
- **K4 (low-vol):** the tilt's excess return is fully explained by low-vol exposure
  (residual t < 1.5) → not a category signal → kill.
- **K5 (VRT check):** fails the "force VRT/NVT/FLNC to equal-weight" re-run → the signal is a
  vol tilt, not a category rotation → kill.

### If it clears

Build the **point-in-time exposure matrix** (from historical 10-Ks), add the **EEI / utility
capex-mix variant** (§2.2 #3), extend to the **17-name classifier book** (POWL/PRYMY give the
switchgear/cable categories real dispersion), and only then consider wiring it as a documented,
low-conviction overlay alongside the trend/vol layer — never as a standalone signal.

### If it fails

Log it in `docs/` as attempt ~16, note "national category-demand data is too coarse / too priced /
can't reach the DC-power names to rotate this book," and fall back to the discretionary concentrated
long/short with the monitored dashboard (`docs/handoff_2026-09-01…` §7). At that point the
accumulated evidence (≈16 honest attempts) is a strong prior that no reachable data times or
cross-sections this trade, and new research effort should move to a less-arbitraged theme.
