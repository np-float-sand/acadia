# Handoff — Outage / Reserve-Margin Signal for Merchant Generator Differentiation

**Status:** Idea only, least developed of the three follow-on ideas from the 2026-08-15 session. Needs data-source research before any real brainstorming can happen — this is earlier-stage than the RT/DA spread handoff.

**Start a fresh conversation with:** "Let's explore the outage/reserve-margin signal idea — read docs/handoff_2026-08-15-outage-reserve-margin-signal.md" and use the `superpowers:brainstorming` skill from scratch.

---

## Why this exists — the diagnosis that led here

Full context in `docs/compact_2026-08-13-dc-load-signal-results.md`. Short version: current signals underperform XLU; long book rides sector beta, short book has no edge over naive selection (see `docs/handoff_2026-08-15-pairs-basket-construction.md` for the diagnostic numbers). This idea, like the RT/DA spread idea, targets *signal quality* rather than portfolio construction — specifically, it targets a known gap: **the existing stress-beta signal treats all merchant generators as interchangeable**, when in reality their exposure to a given stress event depends on their own generation fleet's availability at that moment.

When asked "what else can we do with the power grid info," this was one of four ideas offered (the user chose to pursue #1, #3, and #4 — this is #4):

> Bring in outage/reserve-margin data (EIA-860/861 planned outages, or PJM/ERCOT capacity auction clearing prices) to differentiate which merchant generators actually benefit from an upcoming stress period, rather than treating VST/NRG as interchangeable. Addresses a real gap (today's stress-beta signal doesn't distinguish generators by their own outage/reserve exposure) but is more data-integration effort.

## The core idea (not yet fleshed out)

Today, `grid_resilience/signals/conditional_beta.py::compute_stress_betas()` estimates a rolling OLS "stress beta" per ticker — how much a stock's excess return responds to system-wide grid stress (the GSI). This treats VST and NRG (the only two merchant generators in the universe) symmetrically except for whatever their historical price-return relationship happens to show. It doesn't account for **which generator has more available (non-outaged) capacity to actually capture a stress event's price spike** — a generator with a unit down for maintenance during a heat wave captures less upside than one running at full capacity, even if their historical stress-betas look similar.

Candidate data sources (none verified for availability/quality yet):
- **EIA-860/861** — annual/monthly generator-level capacity and (for 861) some outage-adjacent data. Need to check actual granularity and lag (annual data would likely be too stale to matter for a signal meant to differentiate short-term stress-event capture).
- **NERC GADS** (Generating Availability Data System) — the standard industry source for generator outage/availability rates, but access may be restricted/paid — needs checking, this is not something already used elsewhere in this codebase.
- **PJM/ERCOT capacity market clearing prices** (PJM capacity auction, ERCOT ORDC scarcity pricing) — these are forward-looking, market-priced reflections of expected reserve margin/scarcity, which might be a cleaner and more available proxy than trying to get unit-level outage data directly. Worth checking whether `gridstatus` exposes anything here before assuming custom scraping is needed.

## Open questions to work through when this gets picked up (none explored yet)

1. **Does usable data even exist at the right granularity and lag?** This needs to be answered before any design work — if outage data is only available annually or with a multi-month lag, it can't differentiate a specific stress event's capture, and the idea doesn't work as stated. This is the first thing to check, not something to assume.
2. Is this really only relevant to the 2-name merchant group (VST, NRG), or could a similar reserve-margin/scarcity signal also matter for the regulated path (e.g., utilities whose service territory sits behind a chronically tight reserve margin face different regulatory/rate dynamics)? Scope this explicitly rather than assuming merchant-only.
3. How does this relate to the existing Grid Stress Index, which already has a "reserve tightness" sub-component (`grid_resilience/signals/grid_stress_index.py`, 20% weight in GSI per the CLAUDE.md doc) — is that already capturing some of what this idea is after, at the system level rather than the generator level? Read that code before assuming this is entirely new.
4. Given the merchant group is only 2 names, is a new signal here worth the data-integration effort at all, versus just accepting VST/NRG's existing stress-beta differentiation as adequate and spending effort elsewhere (peer-group construction, RT/DA spread)? This is worth an honest gut-check early, given the "more data-integration effort" cost flagged when this idea was first raised.

## Recommendation for whoever picks this up

Don't start with a design doc — start with a scoped research question: "what generator-level outage/availability/reserve-margin data can we actually get, at what granularity and lag, and does gridstatus already expose any of it?" If the answer is "nothing usable at the needed granularity," this idea should be shelved in favor of the other two. This is the most speculative of the three follow-on ideas and should be evaluated cheaply before real investment.
