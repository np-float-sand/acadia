# Design Spec — DC-Demand Named-Deal Thematic Basket (Small-N, Not a Factor)

**Status:** Draft spec, not yet built.
**Suggested location:** new top-level directory `acadia/dc_demand_basket/`, sibling to `grid_resilience/`, not inside it. Reasoning below.

## Why this is a new directory, not a `grid_resilience/` extension

`grid_resilience/` is built around cross-sectional factor construction (`build_factor()`, z-scoring, `hard_switch`/`revenue_mix`/`dual_track` architectures, an 8-year backtest against XLU). Per `docs/compact_2026-08-26-dc-multi-source-signal-results.md`, that machinery is a structurally bad fit for this thesis:

- The named-deal universe is currently 3 tickers (CEG, TLN, VST), 2 inside the backtest window — too thin for a z-score to carry information (`_zscore_columns` collapses to sign-only, ±1, at n≤2, demonstrated directly in that doc).
- The one architecture (`hard_switch`) actually backtested routes the DC-demand signal only to `pass_through < 0.5` tickers, which structurally excludes the merchant names (CEG 0.80, TLN 0.90, VST 1.00) the signal was built for. Fixing that is a real architecture change, scoped out of this spec.
- The backtest window (2018–2025) is 77–99% pre-data for every ticker in the sleeve (VST's only deal postdates `BACKTEST_END` entirely).

Rather than force a 2–3 name thesis through 8-year cross-sectional factor plumbing, this spec treats it as what it actually is: a small set of named, dated, quantifiable corporate catalysts, sized and held as a long-only conviction basket. No z-scoring, no short leg, no `build_factor()` integration, no full-history backtest as the primary validation.

## Goal

1. Widen the verified named-deal universe well past 3 tickers by expanding data collection beyond the original 4-name target list (CEG, TLN, VST, NRG).
2. Build a conviction-weighted, long-only basket construction and rebalancing process driven by disclosed deal MW, not a cross-sectional score.
3. Validate with methods that fit small-N event data (per-deal event studies, basket-level track record since inception) rather than a long historical backtest.

## Step 1 — Widen the named-deal universe

**Candidate tickers to research (not pre-verified — confirm each independently before adding):**

Beyond the existing CEG/TLN/VST, plausible candidates based on the same disclosure pattern (named counterparty, quantified MW/GW, dated announcement) include regulated and quasi-regulated utilities with publicly reported hyperscaler/data-center agreements — e.g., Southern Company (SO) / Georgia Power large-load agreements, Dominion (D) Virginia "Data Center Alley" interconnection queue and disclosed customer commitments, Entergy (ETR) data-center service agreements and any disclosed SMR/nuclear deals, NextEra (NEE) corporate PPAs, Xcel (XEL), PPL, and AEP subsidiary-level disclosures. **Do not assume any of these has a qualifying deal — this list is a research starting point, not a confirmed set.** Apply the exact same bar as the existing seed data: a named counterparty, a quantified MW figure, and a traceable primary source.

**Source hierarchy (reuse the point-in-time discipline from `docs/2026-08-26-dc-demand-exposure-signal-design.md`, Layer 3):**

1. Company press releases, earnings call transcripts, investor day decks, SEC 8-K/10-Q filings — primary, and the only acceptable source for `first_disclosure_date`.
2. Trade press specifically covering data-center/hyperscaler infrastructure (e.g. Data Center Dynamics, Utility Dive) as a discovery mechanism to find deals faster than manually scanning transcripts — every hit must still be traced back to a primary source before entering the dataset.
3. ISO/regulatory filings (PJM LAS, ERCOT large-load queue, state PUC filings) as **corroboration only**, never as the `first_disclosure_date` source, per the existing point-in-time rule — if a filing is genuinely the first public mention (no prior press), flag it as a distinct leading-indicator case rather than folding it into the general rule.

**Schema (extend the existing `hyperscaler_deals.csv` schema, do not redesign it):**

`ticker, counterparty_hyperscaler, mw, first_disclosure_date, source_url, source_type (press_release | earnings_call | sec_filing | trade_press_corroborated | iso_filing_leading_indicator), verification_status (confirmed_primary_source | needs_verification), notes`

**Verification bar:** every row must reach `confirmed_primary_source` before it counts toward basket construction — this project has already found one previously-"confirmed" row wrong by ~10x (per the prior compact doc), so re-verify existing rows against primary sources during this pass too, not just new ones.

**Target:** no fixed ticker-count target. Report whatever the verified count turns out to be, honestly — the prior spec's mistake was pre-committing to "~11 tickers" and then reporting the miss as a headline failure. State the search process and the count found; do not treat a small number as a failed deliverable if the underlying verification was done correctly.

## Step 2 — Basket construction (not a factor)

- **Long-only.** No short leg. The thesis ("named DC-demand exposure is under-recognized") doesn't have a natural short side the way a resilience-spread thesis might; forcing one adds risk without a clear economic rationale.
- **Inclusion rule:** a ticker enters the basket once it has at least one `confirmed_primary_source` deal.
- **Weighting:** conviction-weighted by cumulative disclosed MW under contract, normalized by market cap (MW/$B market cap) to avoid the basket being dominated by whichever company happens to have the largest single deal in absolute MW terms. Report both the raw-MW and market-cap-normalized weights side by side — don't silently pick one.
- **Position caps:** apply a max single-name weight (e.g., 25–30%, exact figure is a judgment call to make explicit and document, not bury in code) given the small basket size will otherwise concentrate hard in 1–2 names.
- **Rebalance trigger:** event-driven (a new verified deal, or a name-changing update to an existing deal) rather than calendar-driven, since the underlying information arrives as discrete events, not a continuous flow. Also allow a calendar backstop (e.g., quarterly review) to catch verification-status changes even with no new deals.

## Step 3 — Validation (fit the method to the data, not vice versa)

Two separate, honestly-scoped reports, not one blended backtest:

1. **Per-deal event study.** For each `confirmed_primary_source` deal, compute short-window cumulative abnormal return (CAR) around `first_disclosure_date` (e.g., [-1, +5] trading days, benchmark-adjusted against XLU or SPX). Report each event's CAR individually — do not average across events and present a single number as if it were a stable estimate; with ~6–15 events this is descriptive, not inferential.
2. **Basket-level track record since inception.** Once the basket exists, track its actual return stream forward from construction date against XLU and SPX. This is the only honest test of "does this help going forward" — do not attempt to backtest the basket construction methodology over 2018–2025, since the underlying data doesn't exist for most of that window (same problem the prior factor attempt ran into).

**Explicitly out of scope for this spec:** cross-sectional z-scoring, `build_factor()`/`hard_switch` integration, IC@21d or any factor-style predictive-power metric, and any claim of statistical significance given the sample size. If someone later wants to revisit the factor approach, that requires the merchant-path integration fix flagged in the prior design doc — not something to smuggle back in here under a different name.

## Deliverables

- `acadia/dc_demand_basket/deal_research.md` — running log of candidate tickers researched, sources checked, and verification outcome (including negative results — "checked X, no qualifying deal found" — so future sessions don't re-research the same names).
- `acadia/dc_demand_basket/data/hyperscaler_deals_v2.csv` — widened, re-verified deal table.
- `acadia/dc_demand_basket/basket_construction.py` — builds the weighted basket from the CSV per Step 2.
- `acadia/dc_demand_basket/event_study.py` — per-deal CAR calculation per Step 3.1.
- `acadia/dc_demand_basket/README.md` — states the basket's current composition, weights, inception date, and the explicit non-goals above, so a future reader doesn't mistake this for a backtested factor.
