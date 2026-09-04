# Design Spec — Grid Equipment Suppliers Thematic Basket

**Status:** Draft spec, not yet built.
**Suggested location:** new top-level directory `acadia/grid_equipment_basket/`, sibling to `grid_resilience/` and `acadia/dc_demand_basket/`.

## Context and rationale for a new direction

Every utility-level cross-sectional signal tried so far (stress-beta, ICR, peer-group construction, DC-demand multi-source) has failed to clear a "material edge over XLU" bar — see `docs/compact_2026-08-19-peer-group-construction-results.md` and `docs/compact_2026-08-26-dc-multi-source-signal-results.md`. A recurring problem across all of them: the underlying demand signal (grid stress, DC load) had to be *inferred* from indirect proxies (queue MW, stress-beta) because utilities themselves don't cleanly disclose the thing being measured.

Grid equipment suppliers are a structurally different bet on the same underlying trend (multi-year grid buildout driven by AI power demand, electrification, and reshoring), but with a data advantage: order **backlog** and **book-to-bill ratio** are standard, quarterly-disclosed, quantitative figures in these companies' own SEC filings and earnings calls — not something that has to be inferred from a third party's interconnection queue. This spec is scoped to test whether that data quality advantage translates into a usable signal, starting as a thematic basket (matching the DC-demand basket's scope) with an explicit note on where it could later graduate to an actual factor if the cross-section proves large enough.

## Candidate universe (research starting point — verify each, do not assume inclusion)

Companies with plausible direct exposure to grid/data-center power infrastructure buildout:

- **Transformers / switchgear:** Hubbell (HUBB), Eaton (ETN), GE Vernova (GEV), nVent Electric (NVT), ABB (foreign-listed — confirm reporting availability), Siemens Energy (foreign-listed).
- **Transmission cable / HVDC:** Prysmian, Nexans (both foreign-listed — confirm US reporting/ADR availability before including).
- **Data-center power & thermal infrastructure:** Vertiv (VRT) — direct data-center power/cooling supplier, arguably the cleanest single-name expression of this theme.
- **Grid-scale storage:** Fluence Energy (FLNC).
- **EPC / grid construction & engineering:** Quanta Services (PWR), MYR Group (MYRG).

**Do not add any of these to the working dataset without independently confirming**: (a) it actually reports backlog or book-to-bill as a disclosed metric, (b) its backlog is meaningfully attributable to grid/data-center-related work specifically rather than generic industrial/construction backlog, and (c) US-listed with standard SEC reporting (drop or flag foreign-listed names lacking clean quarterly comparables).

## Step 1 — Backlog/book-to-bill data collection

**New module:** `acadia/grid_equipment_basket/backlog_data.py`

- Source: each company's own 10-Q/10-K "backlog" disclosure and earnings-call book-to-bill commentary. This is self-reported and not standardized across companies (some report total backlog, some segment-level, some don't report book-to-bill at all) — **do not force a uniform metric**; capture what each company actually discloses and note the inconsistency explicitly rather than papering over it with an assumed-comparable number.
- Point-in-time discipline: backlog figures are as-of-quarter-end and typically disclosed ~4-6 weeks after quarter close — use the filing/call date, not the quarter-end date, as the availability date for any forward-looking use, to avoid look-ahead bias.
- Where available, capture segment-level backlog (e.g., Eaton's Electrical segment specifically) rather than company-wide, since company-wide backlog dilutes the grid-specific signal with unrelated business lines — flag per-company whether segment data is available or only total.
- Secondary/corroborating data (not primary): industry capex forecasts (EEI annual reports, DOE grid modernization reports, transformer lead-time trackers) — useful for narrative context, not for the quantitative dataset.

## Step 2 — Start as a thematic basket, not a factor (matching the DC-demand basket's scope)

Given the candidate universe above is likely 6-10 names even before verification narrows it further, this is still probably too small for a robust cross-sectional z-score approach, so default to the same construction as `acadia/dc_demand_basket/`:

- Long-only, no short leg.
- Inclusion rule: verified backlog/book-to-bill disclosure with plausible grid/DC attribution.
- Weighting: consider ranking by book-to-bill trend (accelerating vs. decelerating) or backlog growth rate as a *tiebreaker/sizing* input rather than a z-scored factor — e.g., equal-weight the qualifying names, then tilt modestly toward those with the strongest recent backlog growth, with an explicit, documented tilt magnitude rather than an opaque score.
- Position caps and rebalance cadence: same approach as the DC-demand basket spec — quarterly review aligned to earnings season (this data literally only updates quarterly, unlike event-driven deal announcements), explicit max single-name weight.

## Step 3 — Note the factor potential as an explicit phase 2, not this spec's deliverable

Unlike the DC-demand deal data (inherently sparse, event-based), backlog/book-to-bill is a recurring quarterly time series across (potentially) 6-10+ names going back several years for most of these companies — meaningfully thicker than the DC-demand basket's 2-3 name, 2-year history. **If** verification lands on 8+ names with clean, comparable segment-level data and several years of quarterly history, a cross-sectional factor (rank by backlog-growth surprise, e.g.) becomes methodologically defensible in a way it wasn't for the DC-demand data. This spec does not build that — it's flagged here so a future session doesn't have to rediscover the option, and so nobody quietly reintroduces factor machinery into what's scoped here as a basket.

## Validation

- **Backlog-growth vs. forward-return descriptive check**, not a formal backtest: for each name, plot backlog growth rate against subsequent 1-2 quarter forward return. This is exploratory — meant to inform whether Step 3's factor option is worth pursuing later, not to produce a Sharpe/IC number to defend now.
- **Basket-level track record since inception**, same as the DC-demand basket — track forward from construction date against a relevant benchmark. XLI (Industrials) is likely a better benchmark than XLU here, since most of this universe isn't utility-classified; confirm sector classification per name before picking the benchmark.
- Explicitly avoid retrofitting an 8-year historical backtest as the primary validation — same lesson as the DC-demand and resilience-factor work: don't let backtest availability substitute for asking whether the data existed and was disclosed in a comparable form that far back (transformer/grid-capex-driven backlog growth is itself a recent phenomenon; pre-2023 backlog data may reflect an entirely different demand environment and isn't necessarily informative about the current thesis).

## Deliverables

- `acadia/grid_equipment_basket/candidate_research.md` — per-candidate verification log (metric disclosed, segment-level or total, grid/DC attribution assessment, include/exclude decision with reasoning).
- `acadia/grid_equipment_basket/data/backlog_quarterly.csv` — collected backlog/book-to-bill time series, per company, with filing-date-based availability timestamps.
- `acadia/grid_equipment_basket/basket_construction.py` — builds the basket per Step 2.
- `acadia/grid_equipment_basket/README.md` — current composition, weights, inception date, chosen benchmark and why, and the Step 3 phase-2 note so the factor option isn't lost or silently attempted without the stated preconditions.
