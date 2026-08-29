# Grid Equipment Suppliers — Candidate Verification Log

Spec: `docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md` (§2 Candidate universe, §7 bias mitigations).

**Inclusion rule.** A name is included only if a plain reading of its **most recent annual filing (10-K)**
says it derives *material* revenue from (a) selling grid / data-center electrical equipment, or
(b) engineering / constructing electric-power infrastructure. Inclusion is **never** decided on historical
stock returns. FLNC (a chronic underperformer) is deliberately retained because it passes the business test —
keeping a known loser limits cherry-picking (§7). Foreign-listed names are logged here for the record but
excluded from the backtestable basket (US-accessible lines are thin OTC ADRs only).

**Method.** CIKs resolved from `https://www.sec.gov/files/company_tickers.json`. Latest annual filing located
via `https://data.sec.gov/submissions/CIK##########.json`. Business + Backlog / Remaining-performance-obligation
(RPO) sections read from the primary filing document. XBRL `RevenueRemainingPerformanceObligation` availability
checked per-name via `https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/RevenueRemainingPerformanceObligation.json`.
Listing / first-clean-daily-history and average dollar volume from yfinance `history(period="max")`.
All SEC requests sent with User-Agent `acadia-research sand.gh1902@gmail.com`. Research date: 2026-08-29.

`backlog_disclosure` vocabulary: `xbrl_rpo` (files the GAAP RPO concept, XBRL-tagged, filing-dated) /
`nongaap_backlog_total` (a single company-defined backlog $ figure) /
`nongaap_backlog_segment` (company-defined backlog broken out by reportable segment) /
`book_to_bill_only` / `none`.

---

## Verified candidates

### ETN — Eaton Corp plc
- CIK: 1551182
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-02-26 (accession 0001551182-26-000007).
  https://www.sec.gov/Archives/edgar/data/1551182/000155118226000007/etn-20251231.htm
- business_verdict: **include** — "Eaton's Electrical sector helps customers manage power in a way that's
  reliable, efficient, safe and sustainable. From the grid to homes, buildings, data centers and industrials –
  Eaton plays a vital role in modernizing infrastructure and accelerating the electrification of society."
  Electrical Americas ($13,276M) + Electrical Global ($6,815M) = $20.1B of $27.4B FY2025 net sales (**73%**);
  the balance is Aerospace / Vehicle / eMobility. Grid + data-center electrical equipment is the majority of
  the company.
- backlog_disclosure: `nongaap_backlog_segment` — MD&A "Performance metrics" tables give per-segment Backlog
  and Book-to-bill: Electrical Americas backlog $13,246M (b/b 1.2), Electrical Global $2,034M (b/b 1.0),
  Aerospace $4,316M (b/b 1.1); a total "firmly committed" backlog of ~$19.8B is also stated in the
  revenue-recognition note. (ETN *did* XBRL-tag `RevenueRemainingPerformanceObligation` from 2020 through
  2024-Q1 and then discontinued it — consistent with the design-spec note that ETN "does not tag this
  concept" going forward; Step 2 will need the MD&A figures, not the XBRL API, for ETN.)
- listing: NYSE:ETN. Clean daily history since the 1970s (yfinance max start 1972-06-01); full coverage of
  both the primary (2023-present) and prior-regime (2020-2022) windows. 1-yr avg daily $ volume ~$965M.
- decision: **include** — grid/data-center electrical equipment is ~73% of revenue and is described as
  Eaton's core mission; segment-level backlog is disclosed.

### HUBB — Hubbell Incorporated
- CIK: 48898
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-02-12 (accession 0001628280-26-007500).
  https://www.sec.gov/Archives/edgar/data/48898/000162828026007500/hubb-20251231.htm
- business_verdict: **include** — "Hubbell is a world-class manufacturer of electrical and utility solutions …
  We provide utility and electrical solutions that enable our customers to operate critical infrastructure
  reliably and efficiently." Utility Solutions segment (**63% of 2025 revenue**) "consists of businesses that
  enable the grid to conduct, communicate and control energy … utility transmission & distribution (T&D)
  components such as arresters, insulators, connectors, anchors, bushings, enclosures, cutouts and switches";
  the remaining ~37% is the Electrical Solutions segment. Essentially the entire company is electrical /
  grid equipment.
- backlog_disclosure: `nongaap_backlog_total` — Item 1 "Backlog": "The backlog of orders believed to be firm
  at December 31, 2025 was $2,159 million compared to $1,898 million at December 31, 2024." Shipment timing is
  described per segment but the dollar figure is company-total. (No live XBRL RPO series — the concept was
  tagged only sporadically in 2018-2020.)
- listing: NYSE:HUBB. Clean daily history since the 1970s (yfinance max start 1972-06-05); covers both
  windows. Single share class since the 2015 reclassification of the former HUBA/HUBB structure. 1-yr avg
  daily $ volume ~$266M.
- decision: **include** — a pure-play electrical / grid-components manufacturer; total firm backlog disclosed.

### GEV — GE Vernova Inc.
- CIK: 1996810
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-01-29 (accession 0001996810-26-000015).
  https://www.sec.gov/Archives/edgar/data/1996810/000199681026000015/gev-20251231.htm
- business_verdict: **include** — GE Vernova is "a developer, manufacturer, and service provider of power
  generating and decarbonizing solutions." Three segments, all power / grid: Power $19,767M (gas, nuclear,
  hydro, steam turbines), Wind $9,110M, Electrification $9,642M of $38,068M FY2025 revenue. The
  Electrification segment's Grid Solutions business (FY2025 revenue $6,620M) supplies "high-voltage direct
  current transmission (HVDC) and alternating current substation solutions, power transformers, switchgear,
  synchronous condensers, and … grid automation." 100% of revenue is power-generation or grid equipment /
  services.
- backlog_disclosure: `xbrl_rpo` — "As of December 31, 2025 … approximately $94.4 billion in remaining
  performance obligations (RPO)" is stated in the filing and the company actively XBRL-tags
  `us-gaap:RevenueRemainingPerformanceObligation` (companyconcept API returns a continuous quarterly series;
  the total figure filed against the tag at 2025-12-31 is ~$150.2B, which includes long-term service
  agreements). Step 2 can pull GEV RPO structurally.
- listing: NYSE:GEV. Spun off from GE; **regular-way trading began 2024-04-02** (yfinance shows a handful of
  when-issued prints from 2024-03-27). First clean daily history 2024-04-02 — this **postdates the Step 4
  price-check window (2023-01-01→2023-06-30) and ~40% of the primary backtest window**; per spec §4 GEV simply
  enters the equal-weight basket on its first available date and is absent before then. This is expected and
  acceptable, not a data defect. 1-yr avg daily $ volume ~$2.3B.
- decision: **include** — entire business is grid + power-generation equipment; RPO is XBRL-tagged. Short
  price history is handled by the basket construction rule, not by exclusion.

### VRT — Vertiv Holdings Co
- CIK: 1674101
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-02-13 (accession 0001674101-26-000008).
  https://www.sec.gov/Archives/edgar/data/1674101/000167410126000008/vrt-20251231.htm
- business_verdict: **include** — "We are a global leader in the design, manufacturing and servicing of
  critical digital infrastructure technology that powers, cools, deploys, secures and maintains electronics
  that process, store and transmit data. We primarily provide this technology to data centers, communication
  networks and commercial and industrial environments worldwide." The cleanest single-name expression of the
  data-center electrical-infrastructure theme.
- backlog_disclosure: `nongaap_backlog_total` — Item 1 "Backlog": "Vertiv's estimated combined order backlog
  was $15.0 billion and $7.2 billion as of December 31, 2025 and 2024, respectively." Company-total figure;
  Vertiv also reports book-to-bill each quarter on earnings calls. **Vertiv does not XBRL-tag
  `RevenueRemainingPerformanceObligation`** (companyconcept API returns HTTP 404 for the base concept and the
  Current/Noncurrent variants) — Step 2 must hand-collect the disclosed backlog $ / book-to-bill for VRT.
- listing: NYSE:VRT. Public via the GS Acquisition Holdings SPAC merger completed 2020-02-07 (ticker was
  GSAH before then; yfinance max start 2018-08-02 is the SPAC IPO). Clean daily history for the operating
  company from ~2020-02-07 — full coverage of the primary window; the 2020-2022 prior-regime panel is missing
  its first ~5 weeks (Jan–early-Feb 2020). 1-yr avg daily $ volume ~$1.6B.
- decision: **include** — textbook data-center power/thermal supplier; total order backlog disclosed
  (hand-collect for Step 2).

### PWR — Quanta Services, Inc.
- CIK: 1050915
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-02-19 (accession 0001050915-26-000006).
  https://www.sec.gov/Archives/edgar/data/1050915/000105091526000006/pwr-20251231.htm
- business_verdict: **include** — the Electric segment (historical Electric Power Infrastructure Solutions +
  Renewable Energy Infrastructure Solutions) serves utilities investing "through multi-year grid modernization
  and reliability programs … high demand for new and expanded transmission, substation and distribution
  infrastructure needed to reliably transport power." Electric-power infrastructure EPC is the core business
  (the other segment is Underground Utility & Infrastructure).
- backlog_disclosure: `xbrl_rpo` — actively XBRL-tags `us-gaap:RevenueRemainingPerformanceObligation`
  (companyconcept returns a continuous quarterly series; total RPO $23.76B at 2025-12-31). PWR **additionally**
  reports a non-GAAP "backlog" by reportable segment in MD&A (Electric backlog $36.17B total / Underground
  $7.81B total at 2025-12-31, with 12-month splits), reconciled to RPO. Best structured coverage in the
  universe.
- listing: NYSE:PWR. Clean daily history since 1998; covers both windows. 1-yr avg daily $ volume ~$620M.
- decision: **include** — electric-power infrastructure EPC; RPO XBRL-tagged and segment backlog reconciled.

### MYRG — MYR Group, Inc.
- CIK: 700923
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-02-25 (accession 0000700923-26-000007).
  https://www.sec.gov/Archives/edgar/data/700923/000070092326000007/myrg-20251231.htm
- business_verdict: **include** — "We are a holding company of specialty electrical construction service
  providers … we serve the electric utility infrastructure, commercial and industrial construction markets …
  a leading specialty contractor serving the electric utility infrastructure." Two segments: Transmission &
  Distribution (T&D) and Commercial & Industrial (C&I) — both electrical construction.
- backlog_disclosure: `xbrl_rpo` — actively XBRL-tags `us-gaap:RevenueRemainingPerformanceObligation`
  (continuous quarterly series; ~$2.52B at 2025-12-31). MYRG **also** presents a non-GAAP backlog table by
  segment in Item 1 (T&D $1,018.1M, C&I $1,806.2M, total $2,824.3M at 2025-12-31, with 12-month splits).
- listing: NASDAQ:MYRG (NasdaqGS). Clean daily history since 2008; covers both windows. Market cap ~$4.6B.
  **Average daily dollar volume: ~$82M (trailing 1-yr); it ran lower (~$20-40M/day) earlier in the backtest
  window.** Adequate for an equal-weight position in a 9-name basket (max equal weight ~11%); this is the
  smallest and least-liquid name and is noted as such. No liquidity-based exclusion.
- decision: **include** — electric utility T&D EPC; RPO XBRL-tagged plus segment backlog. Liquidity adequate
  but the thinnest in the universe — flagged.

### NVT — nVent Electric plc
- CIK: 1720635
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-02-17 (accession 0001628280-26-008608).
  https://www.sec.gov/Archives/edgar/data/1720635/000162828026008608/nvt-20251231.htm
- business_verdict: **include** — "nVent is a leading global provider of electrical connection and protection
  solutions … We design, manufacture, market, install and service high performance products … [a] portfolio
  of bus systems, cable management, control buildings, cooling solutions, both liquid and air, electrical
  connections, enclosures, equipment protection, power connections and power management solutions, and
  switchgear systems." Data-center exposure is explicit: "We are an enclosures and liquid cooling leader in
  the U.S. and globally," serving "the infrastructure vertical, including power utilities and data centers."
  ~100% electrical equipment.
- backlog_disclosure: `nongaap_backlog_segment` — Item 1 "Backlog of Orders by Segment": Systems Protection
  $2,118.9M, Electrical Connections $231.0M, total $2,349.9M at 2025-12-31 (vs $749.3M prior year; the jump
  is the Electrical Products Group acquisition plus "the growth of our data centers business"). Beginning
  2025-12-31 nVent also discloses company-total RPO ($2,349.9M — the same figure) in the revenue-recognition
  note, having dropped the one-year practical expedient; a live XBRL RPO series is not yet available
  (companyconcept returns only stale 2018 data), so treat NVT as segment non-GAAP for Step 2.
- listing: NYSE:NVT. Spun off from Pentair; trading since 2018-04-30 (yfinance max start 2018-04-24). Clean
  daily history covers both windows. 1-yr avg daily $ volume ~$275M.
- decision: **include** — electrical connection/protection/enclosure + data-center liquid-cooling supplier;
  segment-level backlog disclosed.

### FLNC — Fluence Energy, Inc.
- CIK: 1868941
- Latest annual filing: 10-K for FY ended 2025-09-30, filed 2025-11-25 (accession 0001868941-25-000081).
  https://www.sec.gov/Archives/edgar/data/1868941/000186894125000081/flnc-20250930.htm
  (A 10-K/A for FY2024 was filed 2025-04-18; the FY2025 10-K above is the latest annual report.)
- business_verdict: **include** — "Fluence is a global market leader delivering intelligent energy storage
  and optimization software for renewables and storage. Our energy storage solutions and operational services
  are designed to help create a more resilient grid …" "We sell configurable energy storage solutions with
  integrated hardware, software, and digital intelligence." Grid-scale battery energy-storage systems are
  grid electrical equipment; the business test is met. **Included on the business test alone, independent of
  its poor stock performance and going-concern-adjacent noise, per spec §2 / §7.**
- backlog_disclosure: `xbrl_rpo` — "As of September 30, 2025, the Company had $5.3 billion of remaining
  performance obligations related to our contractual commitments, which we refer to as our backlog." Actively
  XBRL-tags `us-gaap:RevenueRemainingPerformanceObligation` (continuous quarterly series). FLNC also reports a
  GW-denominated "contracted backlog" (9.1 GW storage; 7.0 GW O&M) in Item 1 — non-dollar, not used for
  Step 2.
- listing: NASDAQ:FLNC (NasdaqGS). IPO 2021-10-28. Clean daily history covers the full primary window; the
  2020-2022 prior-regime panel only from 2021-10-28. Market cap ~$2.0B; 1-yr avg daily $ volume ~$141M —
  adequate.
- decision: **include** — grid-scale battery-storage equipment supplier; RPO XBRL-tagged. Retained despite
  weak returns by design.

### PRIM — Primoris Services Corporation
- CIK: 1361538
- Latest annual filing: 10-K for FY ended 2025-12-31, filed 2026-02-24 (accession 0001104659-26-018677).
  https://www.sec.gov/Archives/edgar/data/1361538/000110465926018677/prim-20251231x10k.htm
- business_verdict: **include (borderline — see both sides)**.
  - *For inclusion:* "We are a leading provider of critical infrastructure services … through our two
    segments: Utilities and Energy. The Utilities segment … specializes in … the installation and maintenance
    of new and existing natural gas and electric utility distribution and transmission systems, and
    communications systems." Utilities revenue was $2,691.7M of $7,574.9M total FY2025 (**~36%**) and is
    entirely utility distribution/transmission work (electric power delivery, gas, communications). The Energy
    segment's growth was "primarily due to increased renewable energy … activity" — Primoris is one of the
    largest US utility-scale solar + storage EPC contractors, and its customer list leads with "solar
    facility developers, power producers, gas and electric utilities." Electric-power / grid infrastructure
    construction is comfortably a *material* share of revenue on a plain reading.
  - *Against inclusion:* Unlike PWR (Electric segment) or MYRG (T&D segment), Primoris has **no reportable
    segment that is unambiguously and exclusively electric-power infrastructure**. Utilities blends electric
    with gas distribution and telecom; Energy blends renewables with petroleum/petrochemical and state-DOT
    road work. A strict "must have a clean electric-power segment" reading would exclude it.
  - *Call:* include. The 10-K's own segment description puts electric utility T&D at the top of the Utilities
    segment, and utility-scale renewable-generation EPC is the stated growth driver of the larger segment;
    together these clear the "material electric-power infrastructure" bar. Flagged as the one genuinely
    borderline name in the universe.
- backlog_disclosure: `nongaap_backlog_segment` — MD&A presents Fixed Backlog + MSA Backlog by reportable
  segment (Utilities: $2,000.9M next-12-mo / $6,423.4M total; Energy: $3,290.5M / $5,521.9M — at
  2025-12-31). Primoris also XBRL-tags `us-gaap:RevenueRemainingPerformanceObligation` (companyconcept
  returns a quarterly series, ~$5.3B at 2025-12-31), so a structured RPO pull is possible too, but the
  segment split is the richer disclosure.
- listing: NYSE:PRIM. Clean daily history since 2008 (earlier years on NASDAQ under the same ticker); covers
  both windows. Market cap ~$4.0B; 1-yr avg daily $ volume ~$149M — adequate.
- decision: **include (flagged borderline)** — material electric utility T&D + utility-scale renewable EPC
  revenue; segment-level backlog disclosed.

---

## Logged but excluded (foreign-listed)

Excluded from the backtestable basket by design (spec §2): the only US-accessible lines are thin OTC ADRs
with poor adjusted-close quality and low volume, not comparable to the primary listings. These names are
recorded so a future session does not re-research them. None appears in the SEC `company_tickers.json`
domestic-issuer map; each files a 20-F (or nothing) with the SEC, not a 10-K.

| Ticker | Company | Thesis fit | Reason for exclusion |
|---|---|---|---|
| ABB | ABB Ltd | Strong — electrification, motion, transformers, switchgear, EV charging | Swiss primary listing (SIX: ABBN) / Nasdaq Nordic; US access only via OTC ADR **ABBNY** — thin volume, imperfect adjusted-close continuity. Not comparable daily history. |
| SIEGY | Siemens Energy AG | Strong — grid technology (transformers, HVDC, switchgear), power generation | Frankfurt primary listing (ENR.DE); US access only via OTC ADR **SIEGY**. OTC ADR only. |
| PRYMY | Prysmian S.p.A. | Strong — power transmission & distribution cables, grid connections | Milan primary listing (PRY.MI); US access only via OTC ADR **PRYMY**. OTC ADR only. |
| NEXNF | Nexans S.A. | Strong — power cables, grid interconnections, HV systems | Euronext Paris primary listing (NEX.PA); US access only via OTC grey-market **NEXNF** — essentially no reliable daily history. OTC grey-market only. |

---

## Final basket universe

`ETN, HUBB, GEV, VRT, PWR, MYRG, NVT, FLNC, PRIM`

**Count: 9.** All nine US-listed candidates pass the business-description test on their latest 10-K. This
equals the spec §2 research starting list — verification confirmed it rather than trimming it. The count sits
at the top of the spec's "expected 6-9" range.

- Backlog-disclosure coverage for Step 2 (if the §5 gate is reached): `xbrl_rpo` (structured, filing-dated) —
  **GEV, PWR, MYRG, FLNC** (and PRIM also tags it); `nongaap_backlog_segment` — **ETN, NVT, PRIM**;
  `nongaap_backlog_total` — **HUBB, VRT**. Definitions are **not** coerced to a common metric (spec §6); each
  name's disclosure is captured as-is. VRT and ETN(current) will need hand-collection from the filings /
  transcripts.
- Price coverage: on a clean fetch (`fetch_prices(..., use_cache=False)`), all eight non-GEV names plus all
  five benchmarks have 100% daily history over 2023-01-01→2025-07-31. GEV has data from 2024-04-02 only
  (~52% of the primary window) — expected, handled by the equal-weight "present on the date" rule (spec §4)
  and consistent with the spec §5 caveat that GEV covers less than half the window.
