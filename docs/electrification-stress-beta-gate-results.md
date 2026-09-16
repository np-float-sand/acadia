# Stress-beta gate — Screened Electrification supplier universe

**Date:** 2026-09-15
**Spec:** docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md §3.4

## Methodology

Reused grid_resilience's GSI (LMP z-score 40% / congestion fraction 25% / reserve tightness
20% / named-event flag 15%) and conditional_beta.py unmodified (electrification_stress_beta/
national_gsi.py), blended equal-weighted across ERCOT/PJM/MISO/CAISO/SPP into one national
series (suppliers have no home ISO, unlike a utility's service territory), and applied
uniformly to the 27-name `screen` universe. Monthly top-tercile-minus-bottom-tercile
stress_beta spread portfolio, pooled t-stat on the forward monthly spread return.

Gate 1 (pre-registered): |t| >= 2.0 on the 2021-01-01-onward window (tight-grid era — see
below), pre-2021 reported for context only.

**Data coverage note (independently verified on a second run of this script):** of the
31-ticker seed pool, THR (Thermon Group Holdings) returned 100% NaN from the price fetch.
Direct yfinance probe confirms this is not a transient fetch error (`Quote not found for
symbol: THR`, reproducible on repeated queries against both historical and 5-day windows,
while other tickers fetch normally in the same run) — a web check found THR was acquired by
CECO Environmental and delisted from the NYSE on 2026-06-01, and Yahoo Finance no longer
serves any historical quote for it post-delisting. The gate therefore ran on the remaining 30
of 31 tickers; monthly tercile coverage was otherwise high (67 of a possible 68 primary-window
months, 41 pre-2021 months), so this does not materially affect statistical power or the
verdict below.

## Regime dating

Reuters (Maguire, 2026-07-20, "America's power grid is choking on costly congestion")
reports PJM congestion cost $777.8M in June 2026 and ~$1B in May 2026, concentrated in
PA/MD/Northern Virginia/NJ. This project's own ancillary/scarcity-pricing probe
(docs/ancillary-scarcity-pricing-feasibility-results.md) already found no timing signal on
2018-2021 (|t| < 1.1). February 2021 (ERCOT Winter Storm Uri) is the commonly-cited
tightening inflection.

## Result

| window | t-stat | n (months) |
|---|--:|--:|
| 2021-01 → present (primary) | 0.144 | 67 |
| pre-2021 (context) | -1.042 | 41 |

**Gate 1: FAIL** (sign: positive)

## Verdict

Grid stress does not show a statistically significant relationship with supplier returns on this universe/window. The primary-window t-stat of 0.144 is nowhere near the pre-registered |t| >= 2.0 threshold (n=67 months), and falls well short of the minimum bar for correlation significance. This result is consistent with ~29 prior signal-search attempts on this project, which have generally returned null or marginally-significant results. Task 6b now applies: formalize the strategy's existing built-in protections (volatility targeting + forward momentum screen) as the deliverable — see docs/electrification-stress-sensitivity-response.md.
