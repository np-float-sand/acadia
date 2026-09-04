# Handoff — utility transmission-project filings as a data source (prelim investigation + scope)

**Date:** 2026-09-03
**Context:** the grid/DER long/short (`grid_equipment_basket`, `docs/handoff_2026-09-01-grid-buildout-long-short.md`)
has no signal-based edge after ~14 attempts. The construction rule (4-criterion "customer + 2 of 3"
screen — see memory `capex-cycle-pair-classifier`) is solid but is a fundamental screen. Open
question: can *grid-side regulatory data* either (a) select/weight the "long who builds the grid"
universe, or (b) time entry/exit. This handoff scopes utility transmission-project filings for that.

---

## TL;DR

- **The data is public and real.** FERC eLibrary (1989→present) + state PUC docket systems hold
  every transmission CPCN / rate filing, with scope, cost, route, in-service date, and — usefully —
  the **driving need stated explicitly** (the Dominion "Tributary" 2025 final order names *"includes
  a data center"* as the need).
- **What these filings give cleanly:** a project-level *demand* database — project, transmission
  owner, $, kV, region/zone, driver (data-center / reliability / generation-interconnection),
  filed date, target in-service date. More granular on $ / geography / driver than PJM RTEP alone.
- **What they do NOT give:** the EPC contractor or equipment/transformer vendor. The Commission's
  *final order* covers scope/cost/route/need only; procurement is a post-approval step. Contractor
  names appear inconsistently in filed application exhibits / competitive-proposal decks, at low
  and uneven yield.
- **Better route to the supplier side:** reverse-map from the contractors themselves — PWR / MYRG /
  PRIM / MTZ and the equipment makers name their awards + utility customers in 10-Ks / 8-Ks /
  earnings calls. Cleaner "who's winning the work" signal than NLP over thousands of PUC exhibits.

---

## 1. Preliminary investigation — what was checked

| source | access | contents | names contractors/vendors? |
|---|---|---|---|
| **FERC eLibrary** | no official API; undocumented JSON backend `https://elibrary.ferc.gov/eLibrarywebapi/api/` (can change without notice); community MCP server exists ("ferc-elibrary-mcp"). 2M+ docs, 1989→present. | Order 1000 competitive filings, Form 715 transmission plans, 205 rate filings, settlements | rarely — specs not brands |
| **PJM Order 1000 redacted proposals** | `pjm.com/planning/competitive-planning-process/redacted-proposals` — per-window proposal PDFs (e.g. Transource Carlisle–Rocky River 345 kV) | equipment *parameters* (ratings, impedances, STATCOMs, capacitors), cost commitments, developer identity | equipment *type* yes; vendor sometimes (developers name OEM partners as a credibility signal) |
| **PJM RTEP / TEAC** | already reviewed (`docs/pjm-large-load-vintages-2026-09-03.md`) | project, designated entity (TO or competitive developer), cost, in-service | no |
| **Virginia SCC docketsearch** | direct PDF URLs `scc.virginia.gov/docketsearch/DOCS/…`; Dominion also posts per-project folders at `dominionenergy.com/…/power-line-projects/<name>/pdfs/` | CPCN application, DEQ environmental report, staff testimony, **final order** | final order: **no**. Sampled the Tributary 2025 order (`2025-06-30-pur-2024-000181-final-order.pdf`): scope, 230 kV extension, **$32.3M**, in-service Apr 2027, need "includes a data center" — **no contractor named** |
| Other PUCs (not yet sampled) | Ohio PUCO (AEP), Texas PUC / PUCT (ERCOT — different regime, CCN not CPCN), New Jersey BPU (PSEG), Maryland PSC (data-center corridor) | same structure as VA | expected same — final orders won't name contractors |

**Finding:** these filings are a strong *demand / driver* source and a weak *supplier-identity* source.

---

## 2. Proposed scope

### Deliverable A (primary) — transmission-project demand database

A point-in-time table, one row per project vintage:

`project_id, transmission_owner, iso, state, region_or_zone, kv, cost_usd, driver
(data_center | reliability | generation_interconnection | policy | other), filed_date,
approved_date, target_in_service_date, source_url, doc_type`

- **Sources:** FERC eLibrary (Order 1000 filings) + VA SCC + OH PUCO + NJ BPU + MD PSC + TX PUCT
  dockets. Start with VA (Dominion / Data Center Alley) — highest signal-to-noise.
- **Extraction:** docket-list scrape → filing PDFs → `pdftotext` + regex/NLP for the fields above.
  Driver attribution is the valuable, non-obvious field (RTEP does not attribute).
- **Uses:**
  1. Independent confirmation + $ sizing of the buildout, with a **data-center-driven fraction**
     over time — a cleaner "is the theme accelerating" read than PJM's accepted-MW Table B-9
     (which failed as a timing signal, `docs/pjm-large-load-vintages-2026-09-03.md`).
  2. **Regional weighting input** for the queue-velocity idea: Δ project-$ by zone → tilt the
     builder universe by each name's disclosed regional exposure. (Universe still defined by the
     4-criterion screen; this only weights it.) **Note the sleeve split:** this can tilt the
     **EPC contractors** (PWR/MYRG/PRIM/MTZ — they disclose backlog by customer/segment, so
     project-$-by-zone → dominant utilities → contractors naming those utilities → overweight).
     It **cannot** tilt the **equipment makers** (ETN/HUBB/GEV/VRT/NVT + ADRs sell nationally
     through distributors — regional project-$ is a market-size signal that lifts them equally);
     for those the only regional mapping is Deliverable B (vendor-named-on-project), which is
     ~60-70% complete. Prior is low regardless — backlog-growth, backlog-coverage and
     maker-vs-contractor tilts all already failed to beat equal-weight on this 17-name book.
  3. **Work-type → supplier weighting (best of the weighting ideas).** Tag each project by
     work-type `{new_HV_line, line_rebuild, new_substation, substation_upgrade,
     transformer_addition, HVDC/FACTS/STATCOM, underground_cable, reactive_compensation,
     DC_interconnect}`; tally project-$ by type over a trailing window; upweight names whose
     product mix matches the dominant type via a hand-built work-type→supplier map:
     HVDC/FACTS → HTHIY/SIEGY/GEV; substation/switchgear → ETN/HUBB/POWL; overhead line →
     HUBB/PRYMY + EPCs; cable → PRYMY; transformers → GEV/HTHIY/SIEGY; DC power → VRT/ETN/SBGSY/NVT.
     Unlike the zone tilt, this *does* differentiate the equipment makers (an HVDC-heavy buildout
     is a real reason to overweight Hitachi/Siemens/GEV over Eaton). Needs the *application
     engineering exhibits* + Order 1000 proposals, not just final orders (final orders are thin on
     equipment detail). Lower-effort proxy for the same mix: NEMA electroindustry data + utility
     10-K capex breakdowns. Prior still low (17 correlated names, overlapping product categories)
     but this is the most economically defensible weighting scheme considered.
  4. Backfill vintages now so a Δ-over-time series exists for a ~2028 test (same posture as
     queue-velocity and book-to-bill hand-collection).

### Deliverable B (secondary) — contractor "revealed order book"

Reverse-map the supplier side from the companies:

`ticker, award_date, counterparty_utility, project_name, iso/state, value_usd (if disclosed),
scope (transmission | substation | data_center_electrical | equipment_supply), source
(8-K | 10-K MD&A | press_release | earnings_call), source_url`

- **Universe:** PWR, MYRG, PRIM, MTZ, EME, FIX (EPC) + ETN, HUBB, GEV, VRT, NVT + ADRs (ABB,
  Siemens Energy, Hitachi Energy, Prysmian) US press releases.
- **Extraction:** SEC 8-K/10-Q full-text search (EDGAR FTS API) + company press-release pages +
  transcript scrape.
- **Use:** the actual supplier-selection signal — which listed names are winning the work named in
  Deliverable A's projects. Cross-referencing A×B gives project → contractor coverage without
  digging PUC exhibits.

---

## 3. Honest limitations

- **Contractor names are not systematically in the regulatory filings** — the primary reason a
  naive "parse PUC dockets → supplier map" fails. Deliverable B (reverse-map) is the workaround and
  it's disclosure-dependent (private firms — Burns & McDonnell, PAR, Southwire, S&C — are invisible;
  utilities self-perform some work).
- **No API for eLibrary or most PUCs** — brittle scraping of changing endpoints and heterogeneous
  PDF layouts across states. Budget most of the effort here.
- **Same core problem as every prior physical-data attempt:** this maps the *customer* pipeline
  well; the customer→supplier bridge stays partial. Expect Deliverable A to be usable, Deliverable B
  to be ~60–70% complete.
- **One regime.** Even with a clean database, the equity test still has only 2023→ as the "it
  works" window.

---

## 4. Recommended first step

Build **Deliverable A for Virginia only** (Dominion, ~2019→2026, ~40–80 projects). ~1–2 days.
Checks: (a) does the data-center-driven project-$ fraction actually rise 2020→2026, (b) does it
lead or lag the basket, (c) is per-zone project-$ granular enough to weight names. If (a)/(c) look
useful, extend to the other four states and start Deliverable B. If not, log it and fall back to
the discretionary dashboard from `docs/handoff_2026-09-01…` §7.

---

## 5. Result of the recommended first step (2026-09-03)

Built. Full write-up: **`docs/va-transmission-filings-probe-results.md`**.
Data: `grid_resilience/data/seed/va_transmission_projects.csv` (72 Dominion VA CPCN cases,
2019–2026, ~$9.9 B). Code: `grid_resilience/data/va_transmission_data.py`.

| check | outcome |
|---|---|
| **(a)** DC-driven project-$ fraction rises 2020→2026? | **PASS** — broad fraction ~0.27 (2019) → ~0.85 (2022) → ~0.83 (2026); ~60% of $ DC-driven strict, ~75%+ broad. But it is a **2022 level shift**, saturated ~0.8 since — good *confirmation*, no headroom as a "accelerating now" gauge. |
| **(b)** leads or lags the basket? | **FAIL as timing** — quarterly corr ≈ 0 at every lead (n≈30); the annual "+0.8 one-year lead" is n=7 and one shared 2022–24 inflection. Same failure mode as PJM Table B-9. |
| **(c1)** per-zone granular enough to weight names? | **N/A Virginia-only** — one TO (Dominion), ~one PJM zone (DOM). Zone tilt (§2 #2) needs multiple states/TOs, as predicted. |
| **(c2)** work-type mix → maker weighting (§2 #3)? | **FAIL from CPCN orders** — mix is ~87% line/rebuild/substation (HVDC/FACTS/transformer ≈ $0); implied tilt is a *static* "overweight EPCs, zero ETN/GEV/VRT/NVT" — doesn't vary year to year and zeros the basket's actual winners. The maker-differentiating detail isn't in final orders (needs Order 1000 proposals + engineering exhibits). |

**Decision: logged, not extended.** Do not build OH/NJ/MD/TX or Deliverable B on this basis.
Keep the VA table + SCC fetch helpers as a **monitored input to the discretionary dashboard**
(`docs/handoff_2026-09-01-grid-buildout-long-short.md` §7) — it confirms scale and the DC-driven
share with more resolution than Table B-9, but yields no tradeable timing signal and no defensible
cross-sectional name weighting from Virginia alone.
