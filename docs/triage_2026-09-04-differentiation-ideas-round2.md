# Triage — 10 candidate differentiators for the AI-power/data-center theme (2026-09-04)

**Context:** follow-up to `docs/triage_2026-08-31-differentiation-ideas.md`, which concluded (inside
the existing 9-19-name grid-equipment basket) "no demonstrated edge beyond thematic beta" and
recommended moving the search to a less picked-over area. This session brainstormed 10 ideas
outside that basket — new universes, new mechanisms — then live-checked each against current
(Sept 2026) news/analyst coverage before any build commitment, in the same spirit as the data
checks elsewhere in this repo (e.g. the interconnection-queue check in
`docs/handoff_2026-09-01-grid-buildout-long-short.md`).

**Bottom line: 8 of 10 are dead — already mainstream, already re-rated, in most cases by
double- or triple-digit YTD moves with explicit sell-side/fund-manager coverage naming the exact
thesis.** The "less picked-over area" does not exist one thematic layer out from the marquee
basket either; 18+ months of the AI-power narrative has been enough for retail media (Motley
Fool, Yahoo, Benzinga), specialist newsletters (SemiAnalysis), and sell-side (VanEck, Bernstein,
Mizuho, BMO) to cover every natural sector adjacency. Only the two ideas that are **alternative-data
mechanisms rather than sector picks** survive the check — and even those survive only at the
"nobody visibly reports doing this" level, not as validated signals.

---

## Dead on arrival (already crowded / already priced)

| # | Idea | Evidence it's already crowded |
|---|---|---|
| 1 | Crypto-miner-to-AI-hosting repricing (CIFR, CORZ, IREN, HUT, WULF) | $70B+ cumulative AI/HPC contracts announced sector-wide; TeraWulf $19B Anthropic lease, Core Scientific $24B+ contracted revenue, Cipher $9.3B AWS/Google-backstopped backlog, Galaxy $4.5B CoreWeave deal. VanEck and Bernstein publish sizing notes; Motley Fool tells retail which one to buy; Visible Alpha already models 71% of IREN/CORZ 2026 revenue as HPC. |
| 2 | Wildfire-liability short book (PCG, EIX, etc.) | EIX -23%, PCG -18% on 2026-08-31 (EIX's largest single-day move in 25+ years) on a CA wildfire liability bill; Mizuho/BMO cut price targets same week; options IV at multi-year highs. This is a live, heavily-traded, litigation/legislative-headline-driven trade, not a hidden risk factor. |
| 4 | Power-semiconductor layer (VICR, MPWR, ON, ALGM) | VICR +137.8% YTD, MPWR +49.2%, ADI +44.1%, TXN +73.6% YTD. SemiAnalysis runs a dedicated newsletter naming every competitor ("Energizing AI: Power Delivery Competition"). Explicit analyst quote: "the market appears crowded with increasing competition." |
| 5 | Gas-turbine-driven midstream (WMB, KMI, EQT) | WMB +33% YTD at 52-week highs, KMI +27% YTD, on the literal "data centers make pipelines great again" thesis (Fortune headline). WMB trades a 34 P/E vs 23 for KMI specifically because of this narrative — priced, not hidden. |
| 6 | Nuclear PPA/restart basket (CEG, TLN, VST) | Microsoft/Constellation/Three Mile Island is one of the most widely covered corporate energy deals of the decade (CNBC, Utility Dive, World Nuclear News since Sept 2024); analysts note Constellation secured the deal at ~2x standard wholesale rate — already priced at a premium. |
| 8 | European grid-equipment relative value (Prysmian, Siemens Energy, Nexans, Schneider) | Siemens Energy +138% YTD (more than most US marquee names), Prysmian +41% YTD. "The AI trade in Europe is about data centres and power" is a literal Yahoo Finance headline; fund managers already quoted on the exact thesis. |
| 9 | Demand-response/VPP enrollment (Sunrun, Stem) | Sunrun +26% in a single day on the Tesla/Renew Home 16.8GW VPP announcement; explicitly headline/momentum-driven ("VPP headlines move the stock quickly" — Benzinga). |
| 3 | Water-rights/cooling-water exposure (AWK, WTRG) | Not yet re-rated as hard as the others (no comparable stock-move evidence found), but actively being pitched to retail right now — Motley Fool: "Is This Sector the Hidden Bottleneck of the AI Boom?", AWWA has published a white paper specifically on data-center water demand, Simply Wall St running "3 US utility stocks tied to AI data center growth" listicles. **Closing fast, not virgin.** |

## Survivors — mechanism-level, not sector-level

| # | Idea | Why it survives the check | Open feasibility questions |
|---|---|---|---|
| 7 | Labor-bottleneck data (BLS OEWS electrician/lineworker wages, job-posting volume, by metro) as a leading signal for which grid-equipment makers/contractors can actually convert backlog to revenue on time | The **narrative** is thoroughly reported (McKinsey: 130k more electricians needed by 2030; Microsoft's president calls it the single biggest AI-buildout bottleneck; Insurance Journal, EnergyNow, trade press all cover it) — but no search result describes anyone running it as a **systematic, ticker-level quantitative signal**. This is exactly the class of idea the repo's own backlog-coverage/backlog-surprise attempts (both FAILED, using *financial* backlog data) never tried: capacity constraint, not order-book size. | BLS OEWS is annual/metro-level, not weekly — real point-in-time panel construction is nontrivial (same posture as the hand-assembled `va_transmission_projects.csv`). Causal chain (metro wage inflation → maker's specific backlog-conversion delay) is unproven and needs a cheap correlation check before any build, same as everything else in this repo. |
| 10 | Cat-bond/reinsurance pricing (Artemis.bm wildfire cat-bond issuance/pricing) as a **leading** indicator for utility wildfire-liability equity risk, feeding the same short book idea #2 already showed is crowded as a headline trade | ILS/cat-bond pricing is a genuinely separate, specialist market from equity research — Artemis.bm's 2026 coverage ($5.18B wildfire-exposed cat bond issuance YTD, FAIR Plan Golden Bear Re raises) is about the cat-bond market itself, not cross-referenced with equity positioning anywhere found. If cat-bond investors reprice wildfire risk ahead of equity analysts, that lead time is tradable and orthogonal to "trade the legislative headline" (idea #2, already dead as a standalone). | Clean historical spread/pricing time series may be subscription-only (Artemis.bm's free content is mostly news, not a panel); need to check whether cat-bond repricing actually *leads* equity moves or is coincident/lagging (cat bonds reprice on scheduled resets, not continuously) before assuming a lead-time edge exists. |

---

## Recommendation

- **Don't build 1, 2, 4, 5, 6, 8, 9 as differentiators.** They're legitimate thematic-beta sleeves
  (same category as the existing grid-equipment basket) but not edges — building any of them now
  is buying an already-re-rated trade, with the same "well-run thematic sleeve, not a
  differentiated one" verdict the 2026-08-31 triage already reached for the core basket.
  If more thematic diversification is wanted for its own sake (uncorrelated timing, not alpha),
  water (#3) and European grid (#8) are the least-baked of the dead group — but that's a portfolio
  decision, not a signal-discovery one.
- **Spike #7 and #10 properly before committing to a build** — cheap correlation/data-availability
  checks in the style of `docs/triage_2026-08-31-differentiation-ideas.md`'s "Data check" sections,
  not full specs yet. Both could still fail the way outage/reserve-margin and RT/DA spread did.
- **Deliverable D (`docs/superpowers/plans/2026-09-04-capex-guidance-signal.md`) remains the only
  fully-designed, ready-to-build item in the queue** — it was pre-registered and approved before
  this session's search, and nothing found here changes that. It's attempt #15 in the core basket
  theme; #7 and #10 above would be attempts #16-17 in the wider adjacent-sector search, if either
  survives its own spike.

---

## Follow-up spikes (same session, after user pushback)

### #7 — labor-bottleneck data: PARTIALLY SURVIVES, needs reframing

**Data check (live, not just literature search):** BLS QCEW Open Data API
(`https://data.bls.gov/cew/data/api/{year}/{qtr}/industry/2382.csv`, NAICS 2382 = Electrical
Contractors) is fully public, no auth, ~5-6 month release lag (2026Q1 data already available in
Sept 2026). County-level, quarterly, with a pre-built `oty_avg_wkly_wage_pct_chg` column.

**Signal check:** pulled 2020Q1 → 2026Q1 for Loudoun County VA, Prince William County VA,
Maricopa County AZ, Douglas County GA, Tarrant County TX vs. US total. Loudoun (densest DC market
in the US) electrical-contractor employment roughly doubled (8,672 → 18,027) and wage growth hit
16.1% YoY in 2025Q4 vs. 6.2% national; Douglas County GA hit 18.3% YoY same quarter. Real,
measurable, cross-sectionally differentiated (Maricopa and Prince William show much less
divergence) — not noise.

**The problem:** no attribution path to the marquee basket (VRT/GEV/ETN/PWR don't disclose
county-level revenue). **Reframe required:** target labor-intensive EPC/contractor names that
*do* report segment exposure — EMCOR (EME), IES Holdings (IESC), MYR Group (MYRG) — as a
margin/earnings-surprise signal, not a tilt on the existing equipment-maker basket. Same ~5-6
month lag risk that killed the outage/reserve-margin signal applies; untested whether it's fatal
here given the causal claim (quarterly margin surprise) is narrower than a basket-timing claim.
**Verdict: worth a real design pass on the reframed (EPC-contractor) version; the original
(basket-tilt) framing is not buildable as stated.**

### #10 — cat-bond/reinsurance leading indicator: DEAD, cleanly

**Data check:** SRRIX (Stone Ridge Reinsurance Risk Premium fund) has 12+ years of daily NAV via
yfinance; ILS (Brookmont Catastrophe Bond ETF) has data since April 2025. Data access was never
the obstacle.

**Mechanism check — two live tests:**
1. Around the actual Jan 2025 LA wildfires (Palisades ignition 2025-01-07): SRRIX declined
   starting 2025-01-10, **three days after ignition** — coincident/lagging, not leading.
2. Around the 2026-08-31 EIX (-23%) / PCG (-18%) crash: **SRRIX and ILS show zero reaction** —
   both instruments trended smoothly upward through the entire event with no visible break.

**Why it fails:** cat bonds trigger on *physical* wildfire-peril losses; the Aug 31 crash was
driven by *legislative* liability-cap risk (a CA wildfire bill), a completely different risk
channel that reinsurance pricing has no exposure to. The single biggest wildfire-liability equity
event available to test against produced no cat-bond market reaction whatsoever. **Verdict: dead,
not a data problem — a structural mismatch between the peril cat bonds price and the peril
(legal/political) that's actually moving these stocks.**

### Deliverable D — flagged, not yet re-scoped

User pushback: D (as designed) is "just a theme, not a continued signal, and it's crowded." Spec
review confirms both concerns are grounded in the spec's own text: §1 states explicitly D is
"one aggregate time-series signal (not cross-sectional)" — architecturally a single monthly dial,
same mechanism class as the FTR signal (FAILED) and the big-four capex-deceleration trigger
(FAILED — "fires roughly a year late," lagging confirmation not a leading signal). Given this
session's finding that every adjacent expression of the utility-capex-for-DC narrative has
already re-rated 30-140% YTD with sell-side coverage, D's premise (aggregate capex-guidance
revisions are still unpriced) is weaker than it looked when the spec was approved this morning —
capex guidance is typically raised live on an earnings call, an instantly-priced event, unlike
the buried-in-a-10-Q hyperscaler capex data that produced the lagging big-four signal.
**Proposed next step (not yet actioned):** add a cheap event-study pre-check (does the basket
move on the actual disclosure dates already in the seed panel?) before running the full
pre-registered backtest — reuses existing data, sequences the cheapest disconfirming test first.
A genuinely cross-sectional ("continued") version would need a new design (maker/EPC
customer-exposure map), tracked separately as a possible "D-2," not a patch to the approved spec.

### #7 — full spike result: DEAD (attribution reframe also fails)

Following the geographic-attribution kill (contractors disclose data-center revenue by
**end-market segment**, not geography — no county-level breakdown exists for EME/IESC/MYRG), the
reframed version was spiked: national QCEW NAICS 2382 (Electrical Contractors) YoY wage growth
(49 quarters, 2014Q1-2026Q1, `data.bls.gov/cew/data/api/{y}/{q}/industry/2382.csv`, own_code=5,
area_fips=US000) vs. quarterly operating margin for EME/IESC/MYRG (SEC XBRL companyfacts,
CIK 105634/1048268/700923; revenue tags unioned across `Revenues` /
`RevenueFromContractWithCustomerExcludingAssessedTax` / `SalesRevenueServicesNet` — same
first-nonempty→union fix already applied to the value-chain signal per CLAUDE.md).

**Full-sample correlation looked real:** IESC +0.52 (p=0.002) at a 4-quarter lag, EME +0.42
(p=0.009) at lag 0 — but **wrong sign** for the original margin-compression thesis (positive, not
negative: higher wage growth associates with *higher* margins, consistent with both being driven
by the same demand boom rather than a cost squeeze).

**Did not survive the standard robustness checks:**
- IESC pre-2022 corr +0.08 (p=0.73) vs. post-2022 corr -0.05 (p=0.88) — both ~zero split apart.
- EME pre-2022 +0.43 (p=0.038) flips to post-2022 -0.30 (p=0.315) — sign instability.
- First-differenced (trend removed): IESC -0.23 (p=0.17), EME +0.31 (p=0.066) — neither survives.
- MYRG: no relationship at any lag, full sample or otherwise.

**Verdict: DEAD.** The full-sample relationship was a shared-secular-uptrend artifact (both wage
growth and contractor margins have risen since 2021-2022 for unrelated reasons), not a genuine
lead-lag relationship — same failure shape as the big-four capex-deceleration trigger
(`capex_signal.py`, FAILED). Two mechanism changes were required to get here (county→national
geography, then this robustness failure) — same conclusion as the "Stop here" option in the
pre-spike AskUserQuestion, now confirmed empirically rather than assumed. Not pursuing further.

### #7 — addendum: DC-hub-specific wage index (refined, still not validated)

Re-ran using an employment-weighted index of the actual DC-hub counties (Loudoun VA, Prince
William VA, Douglas GA, Maricopa AZ, Tarrant TX) instead of the diluted national NAICS 2382
aggregate, since the interesting divergence was always at the county level. Confirms the
divergence is real and DC-specific, not generic wage inflation: **DC-hub-minus-national spread
averaged 0.27pp pre-2022 vs. 1.88pp post-2022** (~7x widening).

Re-tested against EME/IESC/MYRG margins (2 constructions × 4 lags × 3 tickers = 24 cells): EME
and MYRG remain weak and sign-unstable across the 2022 split (e.g. EME flips +0.27 → -0.71 at
lag 0) — same noise pattern as the national-aggregate version. One cell is notably stronger and
sign-stable: **IESC at a 4-quarter lag, corr +0.56 (level) / +0.43 (spread), positive in both the
pre- and post-2022 sub-periods** (+0.20→+0.52). This is the single most encouraging result in the
whole #7 line of inquiry, but it is one cell out of 24 tested — treated as likely
multiple-comparisons noise per this project's own standard, not a discovery, unless it survives a
genuine pre-registered out-of-sample holdout. **Not pursued further without that test; logged as
the one thread worth revisiting if someone wants to chase it properly later.**

---

## D's event-study pre-check (run, per the proposed cheap-first-test)

Used the actual Deliverable-D data layer now under active build elsewhere in the repo
(`grid_resilience/data/utility_capex_guidance.py::load_capex_guidance`, 35 revision events across
14 utilities, 2024-2026) against real grid-equipment-basket prices (9 names, `UNIVERSE` in
`grid_equipment_basket/config.py`, pulled directly via yfinance after the repo's cached
`equity_prices.fetch_prices` hit rate-limit gaps in this environment).

**Method:** for each of the 35 individual capex-guidance disclosure dates, computed the
equal-weight basket's return at day 0, cumulative 0-1, and cumulative 0-5 trading days, and
compared each against a like-for-like benchmark (the full-sample distribution of returns/rolling
windows of the same length — not a naive daily-vs-multiday comparison, which would spuriously
inflate the multi-day result via compounding).

**Result: no abnormal reaction at any horizon.** Event-day |return| 1.85% vs. all-day mean 1.70%
(p=0.65); 0-1 day 2.17% vs. 2.46% (p=0.48); 0-5 day 4.36% vs. 4.13% (p=0.70). Revision size
(signed or absolute) does not rank-correlate with reaction size at any horizon either
(Spearman |ρ| ≤ 0.18, all p>0.3).

**Interpretation — does not kill D, does not confirm it either:** this is a genuine null, not
evidence of "already fully priced same-day" (which would show up as an abnormally large,
revision-size-correlated reaction and doesn't). It's consistent with D's own premise that
information value lives in the *aggregate* trend across 15 utilities' disclosures rather than any
single utility's announcement day — a single utility is only one of many customers behind a
9-name diversified basket, so no single disclosure should be expected to move it much. **The cheap
pre-check clears D to proceed to its full pre-registered test; it doesn't pre-validate it.**
Caveat: n=35 events, equal-weight dilution across 9 names — likely underpowered to detect a small
real effect even if one exists.

## Wage-divergence signal, tested directly against the basket (not just contractors)

Tested the DC-hub wage index and DC-minus-national spread directly against the grid-equipment
basket's own forward quarterly returns (2022-2026 overlap, n=17 quarters) at lags 0/1/2/4 —
bypassing the EME/IESC/MYRG margin channel entirely, since that's a more direct test of whether
this data has any timing value for the actual trade. **Result: no correlation at any lag**
(|corr| ≤ 0.30, all p>0.24; most lags near zero). Combined with the margin-channel result
(one thin, likely-spurious cell for IESC at lag 4), this closes out the wage-divergence line of
inquiry: the underlying labor-market divergence is real and DC-specific, but no channel tested
(contractor margins, basket returns) shows a validated trading signal from it.
