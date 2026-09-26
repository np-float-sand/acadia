# Handoff — commercial electricity price (CEP) as a data-center-REIT signal (not started)

**Date:** 2026-09-16
**Status:** SCOPED, NOT RUN.

## Handback — results (fill in when run)

*Pending — not yet run.* When this is picked up, record here: date run, script(s), CEP series
used (national vs. state-weighted), linear-spec result, non-linear-spec result, 2023-onward vs.
2026-YTD consistency, standalone verdict, and the portfolio verdict — specifically, whether
gating/scaling the existing `hedge_overlay.py` DLR/EQIX short sleeve on CEP (alone, or jointly
with the current real-yield gate) improves the combined `screen +val+sleeve` book's Sharpe/
Calmar/MaxDD versus the real-yield-only gate already shipped — plus a link to the full results
doc once written.

## Portfolio role — this signal has a direct home in the shipped book already

Unlike most items in this project's log, `DLR`/`EQIX` are **already part of the shipped
strategy** — as the real-yield-gated short sleeve in `electrification_strategy/hedge_overlay.py`
(`docs/electrification-short-leg-insurance-probe-results.md`). So this is not a satellite-sleeve
question, it's a direct **enhancement-of-an-existing-component** question: does adding CEP as a
second gate condition (either replacing or supplementing the current "6-mo real yield rising"
trigger) improve that sleeve's own contribution to the combined book? Test at minimum: (a) CEP
gate alone vs. real-yield gate alone, (b) CEP AND real-yield (both must fire) vs. either alone,
(c) CEP as a continuous scaler (0-1x notional) rather than a binary gate. Report the effect on
the **combined book's** Sharpe/Calmar/MaxDD and on the feared-scenario P&L
(`docs/electrification-strategy-proposal-v3.html`'s stress scenarios), not just the sleeve's own
standalone return — a gate change that helps the sleeve in isolation but hurts the combined
book's drawdown profile is not an improvement.

## Where this comes from

User-pasted AI-chat summary of: Khaliq & Boufous, "Do energy price shocks affect U.S. data
center REIT returns?", *Finance Research Letters* 104 (2026),
sciencedirect.com/science/article/abs/pii/S1544612326007191. **Partially verified**: the
abstract is confirmed via `ideas.repec.org/a/eee/finlet/v104y2026ics1544612326007191.html`
(WebFetch, 2026-09-16) — ScienceDirect itself 403'd. Confirmed from the abstract: monthly data,
**Dec 2015 – Dec 2025**, regresses data-center REIT returns on commercial electricity price
(CEP), natural gas price, crude oil price, geopolitical risk, market volatility, market return,
and 10-yr Treasury yield (GS10) changes. Finding: **higher CEP → lower REIT returns**; NGP/COP
lose significance once financial controls are added; market return positive, GS10 negative (as
expected for a rate-sensitive real-asset sector). **Not confirmed** (not in the abstract, would
need full-text): the exact REIT list, whether CEP is a national average or state/region-weighted
series, and the regression's exact functional form (levels vs. changes, lag structure).

## Why this is a different test from what this project has already run

This project already has a DC-REIT short sleeve (`DLR`/`EQIX`, gated on 6-month rising real
yield — `electrification_strategy/hedge_overlay.py`,
`docs/electrification-short-leg-insurance-probe-results.md`) and already tested an
electricity-cost-exposure signal on power-intensive industrials (Probe B, FAILED, see
`docs/electricity-earnings-exposure-probe-results.md`), but **Probe B's universe did not
include data-center REITs** — it covered aluminum/steel/chlor-alkali/industrial-gas/crypto
names only, no REITs. So this is a genuinely untried combination: CEP-as-signal ×
data-center-REIT-as-target. It is close in spirit to Probe B (same data class, same failed-once
mechanism idea — power costs → margin/return), so it inherits Probe B's prior: a
similarly-constructed cost-exposure signal already failed 5 of 6 gates on a different sector.
This should be weighed as a real, not token, prior against a fresh positive result.

## Proposed test design (draft — not pre-registered)

1. **Universe**: `DLR`, `EQIX` at minimum (both already used in this project's hedge sleeve);
   optionally widen to `IRM` (already in `electrification_strategy/data/universe_seed.csv` as
   `data_center_reit`) and any other public pure-play DC REIT (e.g. `COR`/`QTS` are
   delisted/acquired — check current public float before widening).
2. **Signal**: national EIA average commercial retail electricity price (`sectorid=COM` on the
   same `electricity/retail-sales` endpoint this project already uses with `sectorid=IND` in
   `research/electricity_earnings_exposure/probe_b.py::eia_state_industrial_price` — trivial
   change) OR a REIT-portfolio-weighted average of the states each REIT actually operates
   data centers in (more faithful to the mechanism, more work — the seed CSV doesn't currently
   carry a state-exposure map for REITs the way Probe B's universe does for industrials).
3. **Target**: the paper's own horizon is monthly — replicate that first (own-horizon
   replication) before testing this project's preferred 1–3mo forward-return framing.
4. **Controls**: match the paper's own control set (NGP, COP, GPR, MV, MR, GS10) at minimum so
   a first pass is a real replication, not a redesign that's already different from what was
   claimed to work.
5. **Test both a linear and a non-linear specification, not just a t-test:**
   - Linear: the paper's own OLS/regression replication (4 above), plus this project's usual
     Fama-MacBeth / pooled-t rank-IC convention.
   - Non-linear: a decile or tercile spread on CEP level/change (extreme-CEP months vs. the
     rest) analogous to the stress-beta decile follow-up, and/or a tree-based fit (GBM/RF) over
     CEP + the paper's own control set. Any non-linear result must be checked out-of-sample
     before being taken seriously (see the crypto-miner handoff's note on this project's
     overfitting history with non-linear fits).
6. **Required windows, both reported, not just one:** 2023-01-01 → present, and 2026-01-01 →
   present (2026-YTD) as a separate, more recent check, in addition to the paper's own full
   Dec 2015–Dec 2025 replication window. A result that holds only in the longer window or only
   in the recent one is flagged as regime-fragile, not adopted.
7. **Pre-register before running**: window(s), CEP series (national vs. state-weighted — pick
   one), and a gate threshold for each spec above, in writing, before looking at output — same
   discipline as `docs/electrification-stress-beta-decile-preregistration.md`.
8. **Combination test** (see "Portfolio role" above): the (a)/(b)/(c) gate-design comparisons on
   the existing DLR/EQIX sleeve, evaluated on the combined book, over the same 2023-onward and
   2026-YTD windows.

## Feasibility

- **CEP data**: high — this project's EIA retail-sales fetcher already works for the sibling
  `IND` sector code; `COM` is confirmed to exist as a sector code on the same endpoint (EIA
  publishes residential/commercial/industrial/transportation retail price series
  identically-shaped). No new infrastructure needed.
- **REIT price/return data**: trivial (`fetch_prices`, universe already partly seeded).
- **State-level DC-REIT exposure weights** (if the more faithful version is wanted): **not
  currently in this repo** — would need hand-collection (10-K property lists) the way Probe B's
  industrial `{state: weight}` map was hand-coded. Skip this for a first pass; start with the
  national CEP series.

## Statistical-power note

n≈2-3 REIT names is thin for any cross-sectional test — this is a **time-series** signal
(CEP level/change → REIT-portfolio-average or single-name return), not a cross-sectional
ranking signal like most of this project's other work, so the relevant "n" is months
(2015–2025 ≈ 120), not names. That's a real strength relative to most of this project's
thin-cross-section problems, but it also means the natural application (if it works) is a
**timing/overlay signal on the existing DLR/EQIX short sleeve**, not a stock-picking factor —
scope it that way from the start rather than forcing a cross-sectional framing it doesn't fit.

## Not yet pre-registered

Scoping only. A run requires locking universe, CEP series (national vs. weighted), window, and
gate threshold in writing first.

## References

- Khaliq & Boufous (2026), *Finance Research Letters* 104,
  sciencedirect.com/science/article/abs/pii/S1544612326007191 (abstract confirmed via
  ideas.repec.org; full text not accessed).
- `docs/electricity-earnings-exposure-probe-results.md` §"Probe B" — closest prior art, same
  data class, FAILED on a different (non-REIT) universe.
- `electrification_strategy/hedge_overlay.py`,
  `docs/electrification-short-leg-insurance-probe-results.md` — the existing DLR/EQIX short
  sleeve this signal would most naturally gate/scale, if it works.
- `research/electricity_earnings_exposure/probe_b.py::eia_state_industrial_price` — the EIA
  retail-sales fetcher to adapt (`sectorid=IND` → `COM`).
