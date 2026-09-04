# VA/Dominion transmission-project filings — probe results (Deliverable A, Virginia only)

**Date:** 2026-09-03
**Scopes:** `docs/handoff_2026-09-03-transmission-project-filings.md` §4 (recommended first step)
**Context:** `docs/handoff_2026-09-01-grid-buildout-long-short.md`, memory `capex-cycle-pair-classifier`
**Code:** `grid_resilience/data/va_transmission_data.py` · **Data:** `grid_resilience/data/seed/va_transmission_projects.csv`

---

## TL;DR

- **Built Deliverable A for Virginia/Dominion:** 72 transmission-line CPCN cases (Va. Code
  § 56-46.1), filed 2019→2026, one row each, with cost, kV, work-type, target in-service, and the
  **driving need classified from the Commission's final order / hearing-examiner report**.
  ~$9.9 B of Dominion transmission investment.
- **Data acquisition is easy, not brittle.** The SCC DocketSearch has an undocumented but clean
  Breeze/OData backend (`.../DocketSearchAPI/breeze/CaseDetails`) — case header + full document list
  by case number, PDF bytes at a stable path. No scraping of changing HTML. Extraction of the four
  hard fields (cost / need / kV / in-service) from the final order + HE report worked on ~90% of
  cases first pass; the rest were in the staff report or application.
- **Check (a) — PASS.** The data-center-driven project-$ fraction **rises** 2019→2022
  (~0.27 → ~0.85 on the broad definition) and stays ~0.8 through 2026. By approved-year it climbs
  ~0.0 (2019) → ~0.9 (2026). DC is **~60% of the $ on the strict definition, ~75%+ broad**. This is
  a cleaner, higher-resolution "the buildout is real and DC-driven" read than PJM's accepted-MW
  Table B-9.
- **Check (b) — FAIL as a timing signal.** At quarterly resolution (n≈30) the filed-$ series has
  **~zero correlation with the basket at every lead/lag**. The annual "filings lead the basket by
  one year, +0.8" is n=7 and is one shared 2022–24 inflection showing up in two series, not
  independent lead information. Same failure mode as the PJM MW-revision signal.
- **Check (c) — FAIL for name weighting, from Virginia alone.**
  - (c1) One transmission owner (Dominion), ~one PJM zone (DOM). The finest cut is SCC region
    (NoVA 53% of $, Eastern 19%, Central 14%, Southern/Western ~7% each) and it is lumpy. The
    zone-tilt idea (handoff §2 use #2) **needs multiple states / multiple TOs** — confirmed, as the
    handoff predicted.
  - (c2) The work-type $ mix is **~all overhead-line + rebuild + substation** (36% / 30% / 21%);
    transformer $16 M, HVDC/FACTS/STATCOM $0, cable $0. The handoff's work-type→supplier map
    (§2 use #3) then produces a **static** "overweight HUBB/PWR/MYRG/PRIM, zero ETN/GEV and
    VRT/NVT/FLNC" tilt — it does not vary year to year (per-name yearly weight σ ≈ 0.05–0.07), and
    it zeros the names (VRT, GEV) that actually drove basket returns 2023–26. The
    equipment-maker-differentiating detail (HVDC / FACTS / transformer type) that use #3 depends on
    is **not in CPCN final orders** — it needs Order 1000 competitive proposals + application
    engineering exhibits, a different document set.

**Verdict — log it, do not extend.** Do not build the other four states or Deliverable B on this
basis. Keep `va_transmission_projects.csv` + the SCC fetch helpers as a **monitored input to the
discretionary dashboard** (`docs/handoff_2026-09-01…` §7): it independently confirms scale
(~$10 B), the DC-driven share (~60–80%), and the 2022 regime turn. It is not a tradeable timing
signal and does not yield a defensible cross-sectional name weighting from Virginia.

---

## 1. What was built

`grid_resilience/data/seed/va_transmission_projects.csv` — 72 rows, schema:

| field | notes |
|---|---|
| `case`, `matter_no` | SCC case number + internal matter id |
| `region` | SCC grouping (NoVA / Eastern / Central / Southern / Western) — sub-state area, **not** PJM zone |
| `project_name`, `caption` | short name + full SCC caption |
| `filed_date` | SCC `Case_Established_Date` (≈ application date) |
| `approved_date` | `Final_Order_Date`; blank for the 8 still-pending 2026 cases |
| `status`, `disposition` | e.g. Approved / Adopted / Granted / **Canceled** (PUR-2021-00276, superseded) |
| `kv` | max nominal voltage in the project (69–765) |
| `cost_usd_m` | Commission-stated total estimated conceptual project cost, $M. `cost_quality` H/M/L (L = derived from a route/method figure or a small sample) |
| `isd_year` | target in-service year |
| `work_type` | `|`-delimited: `new_line, rebuild, reconductor, new_substation, substation_upgrade, transformer, conversion, underground, corridor, loop, gen_interconnect` |
| `driver` | `data_center / reliability_aging / load_growth_other / generation_interconnection / policy_undergrounding` |
| `dc_strict` | 1 if a data-center customer / campus / technology park is named in the order's **need findings** |
| `dc_broad` | 1 if `dc_strict` OR the project sits in a DC load pocket and the stated need is area load growth |
| `key_doc_kind`, `key_doc_url` | the richest document used (final order → HE report → staff report → order for notice) + its PDF URL |
| `scc_case_url` | DocketSearch deep link |

Loader + live helpers: `grid_resilience/data/va_transmission_data.py`
(`load_va_transmission_projects`, `dc_dollar_fraction`, `fetch_case_detail`,
`fetch_case_documents`, `document_url`, `download_document`).

### Population caveats

- Source list is the SCC's public "Transmission Line Projects" index, Dominion rows, case-year
  2019–2026. It is the SCC's curated notice list, not a guaranteed-exhaustive docket query — a
  handful of small closed 2019–2020 cases may be missing. 72 is inside the handoff's expected
  40–80.
- 8 cases (all PUR-2026-…) are pre-order; their cost / kV / need come from the application + order
  for notice, so `dc_strict` is biased **low** for 2026 (thin need text — e.g. the $874 M
  Morrisville–Wishing Star 500 kV line is `dc_strict=0`, `dc_broad=1`). Use `dc_broad` for recent
  years.
- `cost_usd_m` is the **as-stated conceptual** figure at CPCN — not point-in-time-vintaged and not
  the final booked cost. Fine for share/trend; not for precise $ levels.
- One large outlier: PUR-2021-00142 (Coastal VA Offshore Wind onshore transmission) at $1,148 M —
  the CVOW *generation* project is $9.8–21.5 B; only the onshore transmission facilities are in the
  table, tagged `generation_interconnection`.

---

## 2. Check (a) — DC-driven project-$ fraction over time  ·  PASS

By **filed year** (ex-canceled):

| filed year | n | total $M | DC $M (broad) | frac (broad) | frac (strict) | DC cases (strict) |
|---|---|---|---|---|---|---|
| 2019 | 6 | 464 | 124 | 0.27 | 0.04 | 1 |
| 2020 | 5 | 257 | 0 | 0.00 | 0.00 | 0 |
| 2021 | 9 | 1,552 | 129 | 0.08 | 0.07 | 2 |
| 2022 | 8 | 1,418 | 1,204 | **0.85** | 0.85 | 7 |
| 2023 | 10 | 503 | 234 | 0.47 | 0.38 | 2 |
| 2024 | 15 | 2,316 | 2,056 | **0.89** | 0.83 | 11 |
| 2025 | 10 | 1,875 | 1,506 | **0.80** | 0.73 | 7 |
| 2026 | 8 | 1,503 | 1,253 | **0.83** | 0.12* | 3* |

\* 2026 strict understated — pending cases, thin need text.

By **approved year**: frac_broad 0.00 → 1.00 → 0.12 → 0.16 → 0.75 → 0.55 → 0.79 → **0.91** (2019→2026).

- Filed-year broad fraction: slope **+11.9 pp/yr**, corr(t) **+0.79**. Strict: +7.1 pp/yr, +0.47.
- **But the rise is a 2022 level shift, not a smooth ramp.** 2019–2021 DC is a minor share; from
  2022 it is the dominant driver and has been pinned ~0.8 for four years. As *confirmation* the
  buildout is DC-driven and large, it is strong. As a "theme accelerating **now**" gauge it has no
  headroom left — it saturated in 2022.
- Driver mix over the whole window: **data_center $5.87 B**, reliability/aging $1.96 B, other load
  growth $0.85 B, generation interconnection $1.15 B, undergrounding $0.06 B.

---

## 3. Check (b) — lead / lag vs the 9-name buildout basket  ·  FAIL (as timing)

Basket calendar-year total return: 2019 +28% · 2020 +41% · 2021 +43% · 2022 −2% · 2023 +77% ·
2024 +52% · 2025 +52% · 2026 +18%.

**Annual** (n≈7), corr(filed-year metric, basket return shifted by k):

| metric | k=−2 | k=−1 (filings lead) | k=0 | k=+1 (basket leads) | k=+2 |
|---|---|---|---|---|---|
| DC-$ strict | −0.27 | **+0.70** | −0.01 | +0.28 | −0.19 |
| DC-$ broad | −0.14 | **+0.81** | −0.14 | +0.26 | −0.19 |
| total $ | −0.30 | **+0.88** | −0.09 | −0.17 | +0.18 |
| frac broad | +0.11 | +0.53 | −0.22 | +0.46 | 0.00 |

The k=−1 "+0.8" looks like a one-year lead, but the YoY path shows what it is: filed DC-$ jumps
+833% in 2022 (basket −2% that year, +77% next) and +779% in 2024 (basket +52% that year *and*
next). Both series inflect once, together, in the 2022–24 data-center boom. n=7, one shared event.

**Quarterly** (filed-quarter DC-$ broad, n≈30 — the resolution you would actually act on):

| lag k | −4 | −2 | −1 | 0 | +1 | +2 | +4 |
|---|---|---|---|---|---|---|---|
| corr | −0.16 | −0.00 | +0.10 | −0.03 | +0.40 | +0.26 | −0.24 |

Essentially zero at every lead. The only non-trivial value (+0.40 at k=+1) is the **basket leading
the filings** — the wrong direction for a signal, and weak (n=16). CPCN filings are a slow,
administratively-paced series (application → 8–12 months → order); they confirm the buildout, they
do not lead the equities, which price the capex cycle 6–12 months ahead. Same conclusion as
`docs/pjm-large-load-vintages-2026-09-03.md`.

---

## 4. Check (c) — can this weight among the 9 / 17 names?  ·  FAIL from Virginia alone

### (c1) Per-zone — not testable with one TO

All 71 active cases are Dominion in ~one PJM zone (DOM). Region share of $: **NoVA 53%**, Eastern
19%, Central 14%, Southern 7%, Western 7% — and lumpy by year (Eastern is 76% of 2021 solely from
the one CVOW line; NoVA dominates 2022/2024/2026). The handoff's zone-tilt (project-$ by zone →
dominant utilities → contractors naming those utilities) needs **≥ several states / TOs** to create
cross-sectional dispersion. Virginia alone maps everything to "Dominion / DOM."

### (c2) Work-type → supplier weighting (handoff §2 use #3) — the mix is too monotone, and too shallow

Work-type $ mix, full period (project $ split evenly across its tags):

| new_line | rebuild | new_substation | gen_interconnect | corridor | conversion | reconductor | underground | transformer |
|---|---|---|---|---|---|---|---|---|
| $3.60 B | $3.03 B | $2.09 B | $0.45 B | $0.36 B | $0.13 B | $0.08 B | $0.07 B | $0.02 B |

- **~87% of the $ is `new_line` + `rebuild` + `new_substation`.** HVDC/FACTS/STATCOM $0, cable $0,
  transformer-addition $16 M. So the map collapses to "line/EPC work + basic substation."
- Implied 9-name tilt from that mix (vs equal 0.111): **HUBB 0.264, PWR 0.206, MYRG 0.187,
  PRIM 0.184** over; **ETN 0.092, GEV 0.067** under; **VRT / NVT / FLNC → 0.00** (nothing in VA
  transmission filings maps to data-center power/cooling or storage). Active share vs equal-weight
  0.40 — but it is almost entirely "tilt into the 4 line names, zero the 3 DC-equipment names."
- Year-by-year the tilt barely moves (per-name yearly weight σ ≈ 0.05–0.07; the only real swings
  are 2022 all-substation → ETN up, and 2021 CVOW → PWR/GEV up). It is a **structural** bet, not a
  time-varying signal.
- It also zeros VRT and GEV — two of the basket's biggest 2023–26 winners — because their product
  (DC power & cooling, large gas/HVDC equipment) does not show up in Dominion CPCN line filings.
  That is the opposite of what worked.
- Per the handoff itself, the maker-differentiating detail needs Order 1000 proposals + application
  engineering exhibits. The **final orders are confirmed thin on equipment type** — you can tag
  line vs rebuild vs substation and no finer.

---

## 5. Data-source notes (for anyone who does extend this)

- **SCC Breeze/OData** — `https://www.scc.virginia.gov/DocketSearchAPI/breeze/CaseDetails`
  - `GetDetail?$filter=Case_Number eq 'PUR-YYYY-NNNNN'` → `MATTER_NO`, `Caption`, `Status`,
    `Disposition`, `Disposition_Date`, `Case_Established_Date`, `Final_Order_Date`, `Closed_Date`.
  - `GetDocuments?$filter=MATTER_NO eq <n>&$select=DocID,FileName,Document_Name,Date_Filed`.
    **The `$select` is mandatory** — the underlying keyless EF view otherwise identity-collapses
    every row to a single `$ref:"1"`.
  - Also available: `GetActivities`, `GetParticipants`, `GetServiceList`, `GetStaff`.
- **PDF bytes** — `http://www.scc.virginia.gov/docketsearch/DOCS/<FileName>` (FileName like
  `86$701!.PDF`; URL-encode `$` and `!`). No auth. Some application volumes are >60 MB.
- **Richest fields-per-page:** hearing-examiner report → final order → staff report. The cost
  sentence is near-canonical: *"the (total) estimated (conceptual) cost of the (Rebuild) Project …
  is approximately $X million."* The need is in the HE report's "NEED FOR THE PROJECT" section and
  restated as numbered findings in the final order.
- **Driver attribution is genuinely manual-grade** but fast at this N — the order's need findings
  are explicit ("to serve a new data center customer", "to address aging infrastructure at the end
  of its service life", "to interconnect the CVOW Project"). ~1–2 hrs of reading for 72 cases.
- To go multi-state you would re-implement per PUC: Ohio PUCO (DIS), NJ BPU, MD PSC, TX PUCT — each
  a different docket system, mostly without a Breeze-style API. Budget most of the effort there,
  per the handoff.

---

## 6. Recommendation

1. **Keep** `va_transmission_projects.csv` + `va_transmission_data.py` as a monitored dashboard
   input. Refresh 1–2×/yr (new PUR-2026/2027 cases; fill pending-case costs when orders issue).
   It answers "is Dominion's transmission buildout still large and DC-driven?" with more resolution
   than Table B-9.
2. **Do not** extend to OH/NJ/MD/TX or start Deliverable B on the strength of this probe — the
   name-level uses (b timing, c weighting) do not survive, and that was the point.
3. Cross-check once against PJM RTEP DOM-zone baseline `b#` costs to see whether the SCC total
   ($9.9 B) and the RTEP DOM total agree — a cheap validation of the database, still worthwhile
   even though the signal is dead.
4. Fall back to the discretionary concentrated long/short with the monitored dashboard
   (`docs/handoff_2026-09-01-grid-buildout-long-short.md` §7) as the standing recommendation.
