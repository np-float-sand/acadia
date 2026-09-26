# Handoff — industrial electricity usage growth as a broad-market timing overlay (not started)

**Date:** 2026-09-16
**Status:** SCOPED, NOT RUN.

## Handback — results (fill in when run)

*Pending — not yet run.* When this is picked up, record here: date run, script(s), lag
structure used, linear-spec result, non-linear-spec result, 2023-onward vs. 2026-YTD
consistency, own-horizon (broad-market) replication verdict, point-in-time feasibility finding,
and the portfolio verdict — whether layering this as a third overlay (alongside L1 trend-gate
and L2 regime) on `screen +val+sleeve` raises the combined book's Sharpe/Calmar or lowers
MaxDD versus the existing two-layer overlay stack — plus a link to the full results doc once
written.

## Portfolio role — a candidate third overlay layer, evaluated in combination

This is explicitly **not** proposed as a standalone strategy — its own-horizon target is the
broad market (SPY), which this project doesn't trade directly. The only useful outcome is
whether it **enhances the shipped book when layered on top of what's already there**: test it
(a) alone as an exposure gate (for comparability with the paper's own framing), and (b) stacked
with the existing L1 trend-gate + vol-target and (if `REGIME_ENABLED`) L2 congestion-regime
overlays, reporting the combined book's Sharpe/Calmar/MaxDD at each stage
(none → L1 → L1+L2 → L1+L2+this). A signal that fails standalone but measurably improves the
combined stack (lower correlation to the existing gates' trigger timing) is still worth noting;
one that passes standalone but is redundant with L1's price-trend gate (same trigger months) is
not adding anything and should be reported as such rather than double-counted.

## Where this comes from

User-pasted AI-chat summary citing Da, Huang & Yun, "Industrial Electricity Usage and Stock
Returns," *Journal of Financial and Quantitative Analysis* 52(1), 2017. **Verified** (WebSearch,
2026-09-16, multiple independent sources incl. Cambridge Core, Notre Dame News, CNBC): real,
peer-reviewed, well-established paper — not a 2026 result, despite how the pasted summary
presented it. Confirmed finding: growth in industrial electricity usage predicts *future*
**broad market** (not sector, not single-name) stock returns up to 1 year ahead, R²≈9%,
1956–2010, US/Japan/UK; a 1pp rise in usage growth → ≈0.92pp *decline* in forward 1-yr market
return (countercyclical risk-premium proxy — usage tracks the most cyclical parts of the real
economy in real time, and high current output typically precedes lower forward returns).

## Why this is a different animal from everything else in this project

Every other item in `docs/RESEARCH-LOG.md` is either (a) a **cross-sectional** signal — rank
electrification-supplier or grid-stress names against each other, or (b) an **overlay on this
project's own basket** — trend-gate, vol-target, congestion-regime, all scaling *this specific
book's* exposure. This is neither: it's a **broad-market** (SPY-level) timing signal with a
well-established, decades-old academic result behind it, orthogonal to sector selection. The
natural application is not "which of our 27 names to overweight" — it's a macro exposure gate,
structurally most similar to the existing **L1 trend-gate + vol-target overlay**
(`docs/grid-equipment-overlay.md`, ADOPTED) but driven by a different (non-price) input. Scope
it as a candidate **third overlay layer** on the shipped book, not a new cross-sectional factor.

## Proposed test design (draft — not pre-registered)

1. **Own-horizon replication first**: industrial electricity usage growth (probably YoY or a
   smoothed short lag) → forward 12-month SPY (or S&P 500 total return) return, full available
   history. Check the sign and rough magnitude match the paper's own numbers before doing
   anything else — if the replication doesn't reproduce the published relationship on the
   original asset (broad market), there's no basis for applying it to this project's narrower
   book.
2. **Then and only if (1) holds**: test the same signal as a scalar gate on `screen +val+sleeve`
   net exposure (0–100%, or a discrete on/off band), the same way the L1 trend-gate is
   structured, and compare Sharpe/MaxDD/Calmar to the existing overlay stack with and without
   it layered on.
3. **Point-in-time discipline — the likely failure mode**: EIA electricity-usage data is
   revised after initial release (preliminary → final vintages). The original paper's 1956–2010
   sample almost certainly used final, not point-in-time, data — standard for an academic paper
   of that era, but a known trap for *this* project specifically (see the PJM load-forecast
   vintage work, `docs/pjm-large-load-vintages-2026-09-03.md`, and the capex-guidance panel's
   point-in-time EM-fill discipline). A backtest using the final revised series would overstate
   what was actually knowable in real time. If EIA vintage/revision history for
   `electricity/retail-sales` sales volumes isn't available, this needs to be flagged as a
   real (not cosmetic) limitation, not silently assumed away.
4. **Test both a linear and a non-linear specification, not just a t-test/R²:**
   - Linear: the paper's own OLS regression (usage growth → fwd 1-yr return), replicated as-is.
   - Non-linear: a decile/tercile cut on usage-growth extremes (top decile usage-growth months
     vs. bottom, analogous to the stress-beta decile follow-up) and/or a tree-based fit (GBM/RF)
     over usage growth + standard macro controls (rates, credit spreads). Any non-linear result
     must be validated out-of-sample, not just fit in-sample — this project's history (Henry Hub
     GBM, narrative-mispricing Approach C) shows non-linear specs are prone to finding a
     short-window relationship that doesn't generalize.
5. **Required windows, both reported, not just one:** 2023-01-01 → present, and 2026-01-01 →
   present (2026-YTD) as a separate, more recent check, alongside the own-horizon full-history
   replication in step 1. Given the mechanism is a 1-year-ahead forecast, the 2026-YTD window
   will necessarily have very few independent (non-overlapping) observations — report that
   n explicitly rather than let a small-sample result look more confident than it is.
6. **Pre-register before running**: lag structure (usage growth window), whether US-only or the
   paper's full US/Japan/UK panel (US-only is much cheaper and this project only trades US
   equities), the exposure-gate functional form, and a gate threshold for each spec above
   (Sharpe/Calmar delta vs. the existing overlay stack) — in writing, before looking at output.
7. **Combination test** (see "Portfolio role" above): the none → L1 → L1+L2 → L1+L2+this
   comparison, over the same 2023-onward and 2026-YTD windows.

## Feasibility

- **US industrial electricity usage/sales**: very likely available from the same
  `api.eia.gov/v2/electricity/retail-sales` endpoint this project already uses for *price*
  (`research/electricity_earnings_exposure/probe_b.py`), which also carries a sales-volume
  field in the same dataset — **needs a live check of the exact field name and whether a
  ready national aggregate exists or must be summed across states**, but no new data source
  or credential is required.
- **Point-in-time / vintage data**: **unconfirmed** — this is the real feasibility question,
  not the raw data pull. Check whether EIA exposes a revision history for this series (it does
  for some series via its "annual" vs. "monthly early release" distinction) before assuming a
  realistic backtest is possible at all.
- **SPY / broad-market returns**: trivial, already fetched everywhere in this repo.

## Statistical-power note

This is the opposite problem from most of this project's work: n is *not* thin (a monthly or
even annual US time series back to the 1950s–60s if EIA history goads that far, or at minimum
back to whenever this project's usual start date is) — the risk here is a different one,
regime-dependence over a 70-year sample (the "countercyclical risk premium" mechanism may not
be stable across the low-rate 2010s / AI-capex-driven 2020s vs. the mid-century industrial
economy the original sample was built on). A pre-registered regime split (matching this
project's usual 2021-tight-grid-era convention, or a pre/post-2000 split) is the right check,
same discipline as everywhere else in this project.

## Not yet pre-registered

Scoping only. A run requires locking the lag structure, series (US vs. panel), point-in-time
treatment, and gate threshold in writing first — and confirming the own-horizon replication
before ever touching this project's own book.

## References

- Da, Z., Huang, D., & Yun, H. (2017). "Industrial Electricity Usage and Stock Returns."
  *Journal of Financial and Quantitative Analysis*, 52(1), 37–69. Cambridge Core /
  ideas.repec.org/a/cup/jfinqa/v52y2017i01p37-69_00.html (verified via independent secondary
  sources, not paywalled full text).
- `docs/grid-equipment-overlay.md` — the existing L1 trend-gate + vol-target overlay this
  signal would most naturally extend or compete with.
- `docs/pjm-large-load-vintages-2026-09-03.md` — this project's own prior encounter with the
  point-in-time-vs-revised-data trap on a different EIA/grid dataset; the cautionary precedent.
- `research/electricity_earnings_exposure/probe_b.py::eia_state_industrial_price` — the EIA
  retail-sales fetcher to adapt (price field → sales/volume field).
