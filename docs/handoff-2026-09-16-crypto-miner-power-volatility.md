# Handoff — electricity-price volatility as a crypto-miner-stock signal (RUN, FAILED)

**Date:** 2026-09-16
**Status:** RUN 2026-09-16, FAILED (both arms). See "Handback — results" below and
`docs/electricity-price-volatility-miner-results.md`.

## Handback — results

**Date run: 2026-09-16.** Pre-registered first (`docs/electricity-price-volatility-miner-preregistration.md`,
Decisions 1-7 — hub universe widened to all 19 cached ERCOT/PJM hubs with split-sample
selection, miner set widened to include MARA/RIOT/IREN/BTDR as robustness-only, and a parallel
opposite-sign generator arm added on VST/NRG/CEG/TLN), then run via
`research/electricity_price_volatility/run_probe.py` (loaders: `hub_loader.py`, `data.py`;
regressions: `regressions.py`, `hub_selection.py`).

**Hub actually used:** the Mid-C question above is moot — Mid-C/CAISO were out of scope from
the pre-registration (not fetchable in this repo). Hub selection was done by split-sample scan
of all 19 ERCOT+PJM hubs on a 2023-2024 train window only. CIFR and WULF both selected
**ERCOT:HB_PAN** in-sample. CIFR's pick is geographically plausible (100% ERCOT/TX fleet);
**WULF's is a red flag** — WULF's disclosed fleet is 100% NY(NYISO)/PA(PJM), zero ERCOT
exposure, so an ERCOT hub is not a defensible proxy for it at all (worse than the
"known-weak PJM proxy" this doc anticipated). VST cleared **zero of 19 hubs** on the sign
filter (no hub selected, an explicit result, not a fallback).

**Linear-spec result (gate-bearing, 2025-present test window):** FAIL across the board. Neither
CIFR nor WULF clears \|t\|≥2.0 (best \|t\|=0.47); the parallel generator arm also fails (VST no
hub to test; NRG/CEG/TLN all \|t\|<1). Non-linear GBM specs show interesting-looking OOS rank
correlation (Spearman ρ up to 0.7) but feature-importance checks show 55-80% of that is carried
by the BTC control, not the hub variable — the exact Probe-B-style bloc-move pattern this
project has seen before, now visible even inside a spec that explicitly controls for it.

**2023-onward vs. 2026-YTD consistency:** not applicable to a pass/fail call since nothing
passed the 2025-present gate to begin with; both windows are uniformly null/insignificant for
the primary names.

**Standalone verdict: FAIL** (both primary miners, both robustness/widened miners against the
predicted sign, and all four primary generators — see results doc §7 for the full breakdown).

**Portfolio-combination verdict: NOT RUN** — per this doc's own instruction and the task's
constraint, the combined-portfolio backtest against `screen +val+sleeve` is only exercised if a
primary name passes standalone. None did.

**Full results doc:** `docs/electricity-price-volatility-miner-results.md`.

## Portfolio role — this is a satellite sleeve, not a tilt on the shipped book

`CIFR`/`WULF` (and the wider crypto-miner set) are **not held** in the shipped
`screen +val+sleeve` electrification book (`electrification_strategy/data/universe_seed.csv`
has no crypto-miner names) — so this cannot be scoped as a within-book tilt the way the
stress-beta decile follow-up was. The only two ways this can be useful are: (a) **a standalone
satellite sleeve traded alongside the shipped book** at some fixed or signal-scaled weight
(evaluate combined Sharpe/Calmar/MaxDD and correlation-to-book, not the sleeve's own Sharpe in
isolation — a low-Sharpe, low-correlation sleeve can still raise combined Sharpe), or (b) if the
volatility-stabilization mechanism is real, **a risk-overlay input** (e.g. a curtailment-hedging
signal could inform a vol-target or de-risk gate the way the L1 trend-gate already does). Report
both the sleeve's standalone numbers and its effect on the combined portfolio — a standalone
FAIL does not automatically mean "not useful," and a standalone PASS does not automatically mean
"add it" if it's highly correlated with something already held.

## Where this comes from

User-pasted AI-chat summary of a paper claimed to be: "Do electricity prices predict the
returns and volatility of cryptocurrency mining stocks?", *Finance Research Letters*,
Sept 2026 (sciencedirect.com/science/article/abs/pii/S1544612326012626). **The article itself
is paywalled — WebFetch returned HTTP 403 on the ScienceDirect abstract page, and a web search
did not turn up an independently-indexed abstract or working-paper version.** Everything below
about the paper's claims is therefore sourced only from the pasted secondary summary, not
verified against the primary source. Treat every specific below as unconfirmed until the paper
itself (or its supplementary data) can be read.

Claimed mechanism (per the pasted summary, unverified): AR-X GARCH + LSTM/GRU over 1,054
trading days; for pure-play miners (Cipher Mining `CIFR`, TeraWulf `WULF`), Mid-Columbia
("Mid-C") hub electricity price returns displace SPY as the dominant predictor of daily stock
returns; lagged electricity-price volatility predicts *lower* subsequent equity variance
("operational curtailment hedging" — miners power down and resell pre-purchased power when
wholesale prices spike, stabilizing cash flow).

## A specific red flag to resolve before building anything

**Mid-C is a Pacific Northwest / WECC hub, and it does not obviously match either named
company's actual fleet.** Per public disclosure, Cipher Mining's sites (Odessa/Alborz, Bear,
Chief, Barber Lake) are Texas/ERCOT; TeraWulf's are Lake Mariner (NY, NYISO) and Nautilus
(PA, PJM). Neither company has a disclosed Pacific Northwest footprint. Either (a) the pasted
summary garbled the paper's actual hub choice, (b) the paper uses Mid-C only as a generic
liquid-market wholesale-price *proxy* rather than claiming it's these companies' settlement
price (plausible — Mid-C is a commonly-used reference hub in academic papers for exactly this
reason), or (c) the paper's own attribution is off. **This must be resolved by reading the
actual paper before any build** — if the mechanism really is site-specific settlement price
(not a generic proxy), the right US hub for `CIFR`/`WULF` specifically is ERCOT North/Houston
and NYISO/PJM-West, not Mid-C, and this project already has ERCOT + PJM hub data (see below).

## Directly relevant prior art in this project — read before building

`research/electricity_earnings_exposure/` (Probe B, 2026-09-07, FAILED,
`docs/electricity-earnings-exposure-probe-results.md`) already tested a related mechanism on
an overlapping universe that **includes the same miners** (`MARA, RIOT, CLSK, WULF, CIFR,
IREN, BTDR`, `research/electricity_earnings_exposure/universe.py`): a hand-mapped
electricity-cost-exposure signal (`−intensity · %Δ power price`) across 18 power-intensive
industrials, using both EIA state-retail and ERCOT/PJM wholesale-hub price proxies. Headline
result: a fwd-3m return cell that looked real (2022–26, retail price, IC t=1.96) **was entirely
a "crypto miners vs. aluminum vs. gas" sector bet** — once sector-neutralized, the sign flipped
(NW-t −2.12) and the effect died (`probe_b_focus.py`, confirmatory test). **This is a direct
warning for the new probe**: crypto-miner stock returns in this project's sample are already
known to move as a bloc, largely on BTC price, in a way that can masquerade as an
electricity-price relationship in a mixed cross-section. The new mechanism (volatility, not
level; miners-only, not cross-sector) is different enough to be worth a fresh look, but any
test **must isolate power-price variation from BTC-price variation** (e.g. control for BTC
return/vol explicitly, or difference two miners' spreads against each other) or it will likely
reproduce the same false-positive shape Probe B already caught once.

## Proposed test design (draft — not pre-registered)

1. **Resolve the hub question first** (see red flag above) — either get the paper's actual hub
   choice, or pick ERCOT (CIFR) / PJM-West or NYISO (WULF) as the defensible site-specific
   proxy, stated explicitly and justified in writing before looking at any results.
2. Daily or weekly (not monthly — the claimed mechanism is short-horizon) GARCH-style variance
   regression: regress `CIFR`/`WULF` return variance on lagged wholesale-price return variance,
   **with BTC return/variance as an explicit control**, not just SPY.
3. **Test both a linear and a non-linear specification, not just a t-test:**
   - Linear: the GARCH-style variance regression coefficient / its t-stat (as in 2 above), plus
     a plain OLS of forward return on lagged power-price return.
   - Non-linear: a tree-based fit (GBM/RF, classifier on sign or regressor on magnitude) over
     lagged power-price level/return/volatility + BTC controls, and/or a tail-only cut (extreme
     power-price-move days vs. the rest) analogous to the stress-beta decile follow-up. Any
     non-linear result must be checked out-of-sample (train/test split, not just in-sample fit)
     before being taken seriously — this project's history (Henry Hub GBM, narrative-mispricing
     Approach C) shows non-linear fits routinely "find" a relationship in a short window that
     does not survive a full-sample or held-out check.
4. **Required windows, both reported, not just one:** 2023-01-01 → present, and 2026-01-01 →
   present (2026-YTD) as a separate, more recent check. A result that holds in one window but
   not the other is flagged as regime-fragile, not adopted, consistent with this project's
   established pattern of full-window-vs-recent-window discrepancies (Henry Hub, narrative
   Approach C, Dispatch v2's 2024-only artifact).
5. Pre-register the gate the same way Task 5 did: a specific |t| or AUC bar for each spec above,
   and an explicit sign prediction (lagged power-price vol → *lower* subsequent equity vol, per
   the curtailment-hedging mechanism) — a result of the wrong sign is a clean FAIL, not
   "interesting."
6. Small universe (2 pure-play names named in the summary; a slightly wider set — `MARA`,
   `RIOT`, `CLSK`, `IREN`, `BTDR` — already exists in `research/electricity_earnings_exposure/
   universe.py` if breadth is wanted) — flag n up front rather than let it surface as a surprise
   after running, per this project's decile-follow-up precedent
   (`docs/electrification-stress-beta-decile-preregistration.md`).
7. **Combination test** (see "Portfolio role" above): once a standalone verdict is reached,
   backtest a fixed-weight blend of this sleeve with `screen +val+sleeve` and compare combined
   Sharpe/Calmar/MaxDD to the book alone, over the same 2023-onward and 2026-YTD windows.

## Feasibility

- **ERCOT/PJM hub data**: already cached and fetchable in this repo
  (`research/electricity_earnings_exposure/probe_b.py::_wholesale_hub_quarterly`, though that's
  quarterly — a daily version would need to hit the ERCOT/PJM fetchers at native (hourly/daily)
  resolution instead of resampling to quarterly).
- **Mid-C hub data**: **not currently fetchable in this repo.** No existing fetcher; EIA
  publishes Mid-C as one of its tracked wholesale hubs on eia.gov/electricity/wholesale, but
  whether that's available through the `api.eia.gov/v2` REST API used elsewhere in this
  project (`grid_resilience/data/eia_data.py`) or only as a downloadable file dump is
  **unconfirmed — first thing to check if this hub turns out to be the right one.**
- **Stock/BTC price data**: trivial (`fetch_prices`, already used everywhere in this repo; BTC
  price via any standard source).

## Not yet pre-registered

This doc is scoping only. Per this project's discipline (see
`docs/electrification-stress-beta-decile-preregistration.md` for the pattern), a run requires
locking the hub choice, window, controls, and gate threshold in writing *before* looking at any
output.

## References

- Pasted summary citing: sciencedirect.com/science/article/abs/pii/S1544612326012626
  (paywalled, not independently verified — see caveat above).
- `docs/electricity-earnings-exposure-probe-results.md` §"Probe B" — the directly relevant
  prior FAIL on an overlapping universe.
- `research/electricity_earnings_exposure/universe.py`, `probe_b.py`, `probe_b_focus.py` —
  reusable crypto-miner universe + EIA/hub fetchers.
