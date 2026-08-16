# Handoff — RT/DA LMP Spread Signal

**Status:** Idea only. Not yet brainstormed in depth — this is the least developed of the three follow-on ideas from the 2026-08-15 session. Needs a real brainstorming pass (clarifying questions, approach comparison) before a design doc.

**Start a fresh conversation with:** "Let's brainstorm the RT/DA LMP spread signal — read docs/handoff_2026-08-15-rt-da-spread-signal.md" and use the `superpowers:brainstorming` skill from scratch (this handoff is background, not a resumable design-in-progress like the peer-group construction one).

---

## Why this exists — the diagnosis that led here

Full context in `docs/compact_2026-08-13-dc-load-signal-results.md`. Short version: the strategy's current signals (stress beta, ICR, the newly-built DC load signal) all underperform simply holding XLU (Sharpe 0.162). Diagnosis showed the long book rides sector beta and the short book has no edge over a naive short of the whole universe — see `docs/handoff_2026-08-15-pairs-basket-construction.md` for the numbers. That handoff addresses the *portfolio construction* side of the problem (removing sector-beta exposure). This handoff is about the *signal quality* side — whether a different/better underlying signal could add real alpha, especially once it's not being diluted by sector-beta exposure.

When asked "what else can we do with the power grid info," this was one of four ideas offered (the user chose to pursue #1, #3, and #4 — this is #3):

> Finish the RT/DA LMP spread signal (already ~80% scoped per the roadmap in CLAUDE.md). Real-time vs. day-ahead price spread cleanly separates generators (benefit from RT spikes) from T&D utilities (hurt by them) — cheap to complete, but on its own it's still a cross-sectional factor and would face the same sector-beta problem unless paired with the peer-group construction redesign.

## What's already known/scoped

From `CLAUDE.md`'s "Future Work — Signal Improvements" section (already in the repo, written in an earlier session):

> **RT/DA LMP spread (5th GSI sub-signal):** Real-time vs day-ahead price spread cleanly separates generators (benefit from RT spikes) from T&D utilities (hurt by congestion charges). Infrastructure ~90% ready — needs a new `ISO_RT_LOCATION_TYPE` config dict and a `fetch_lmp_rt()` variant. Highest-impact addition after ICR is validated.

Also referenced: `docs/superpowers/specs/2026-05-28-congestion-spread-design.md` (an earlier spec on inter-zonal spread — related but not identical; that one is about zone-to-zone spread within day-ahead prices, this one is about real-time-vs-day-ahead spread at the same location). Read both before designing, to make sure this doesn't duplicate or conflict with the existing spread infrastructure (`daily_spread_summary`, `fill_congestion_from_spread` in `grid_resilience/data/grid_data.py`, already used for congestion-fraction imputation).

## The core idea (needs real brainstorming, not yet fleshed out)

Real-time (RT) LMP diverging from day-ahead (DA) LMP at the same node indicates unexpected grid stress (unplanned outages, forecast errors, extreme weather that wasn't priced in ahead of time). The hypothesis: **merchant generators benefit from RT spikes** (sell into the spot market at the higher price) while **T&D utilities are hurt by them** (pass-through congestion costs, reliability costs). This would be a genuinely different signal from the existing GSI (which is built entirely from day-ahead data) — it captures *unexpected* stress specifically, not just high average stress.

## Open questions to work through when this gets picked up (none decided yet)

1. Does this fit as a 5th sub-component of the existing Grid Stress Index (`grid_resilience/signals/grid_stress_index.py::build_gsi()`), alongside LMP z-score/congestion fraction/reserve tightness/event flag? Or is it cleaner as an independent signal that only applies to the merchant path (paralleling how ICR/DC-load only applies to the regulated path in the hard-switch architecture)?
2. What's the actual data availability? `gridstatus` needs to support fetching real-time LMP with sufficient granularity/history for PJM and ERCOT (the two ISOs with meaningful merchant-generator exposure in this universe: VST/NRG in ERCOT). Needs a live-data verification step before designing further — don't assume `gridstatus` behaves as hoped without checking, per this session's repeated experience that `gridstatus` has version-specific bugs and undocumented quirks (e.g., `get_interconnection_queue()` was broken, PJM's `get_load_metered_hourly` short-code vocabulary didn't match assumptions).
3. Should this feed into the peer-group basket construction (`docs/handoff_2026-08-15-pairs-basket-construction.md`) as one of the ranking inputs for the merchant group specifically, given that group is small (VST, NRG) and needs the best possible differentiator between the two names?
4. Reporting/computation lag: RT LMP is only knowable after the fact — confirm there's no look-ahead risk in how this gets used at each rebalance date (similar care to what was needed for the DC load signal's point-in-time reconstruction).

## Recommendation for whoever picks this up

Start with a live data-availability check (does `gridstatus` cleanly return RT LMP for PJM/ERCOT at the needed granularity, over the full 2018-2025 backtest window) before any design work — this determines whether the "~90% ready" estimate in CLAUDE.md is actually accurate. Then run the standard brainstorming flow: clarifying questions, 2-3 approaches, design doc.
