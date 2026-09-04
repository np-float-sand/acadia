# Backlog-coverage-alone signal — results (2026-09-02)

**Status: FAILED.** Attempt #13 against this theme's search for a signal-based edge
(see `docs/handoff_2026-09-01-grid-buildout-long-short.md` for the full prior record).

## What this was

Handoff §8.3(a): a tilt on year-over-year change in backlog *coverage*
(RPO/backlog ÷ TTM revenue), distinct from the backlog-*growth* signal that already
failed (`docs/backlog-surprise-and-proposals.md` — that one over-weighted FLNC).
`value_chain.py` already had `coverage_ratio`/`coverage_change_signal` built for
the value-chain reframe's blended margin+coverage composite; this tested coverage
*alone*, unblended, to see whether it carries independent information.

## Construction

- Data: `load_backlog_csv()` (unions SEC XBRL RPO + hand-collected `nongaap_backlog_total`,
  fresh for all 9 names through 2025-12/2026-06) ÷ `margin_data.ttm_revenue()`
  (works for 8 of 9 — **HUBB has no usable XBRL revenue tag, a known, documented,
  pre-existing data limitation, not a bug to fix**).
- Signal: `coverage_change_signal` = coverage now − coverage a year ago, ranked;
  top/bottom half get `top=1.25`/`bottom=0.75` equal-weight multipliers
  (`basket.coverage_tilt_targets`, new function — mirrors `backlog_tilt_targets`'s
  mechanics with the coverage signal in place of raw backlog growth).
- Gate: same as the value-chain reframe's Gate 1 — tilt must beat equal-weight on
  **both** Sharpe and CAGR, on both the primary and prior windows.
- Code: `grid_equipment_basket/basket.py` (`coverage_tilt_targets`),
  `grid_equipment_basket/backtest.py` (`coverage_report`). 5 new tests.

## Live result

| Window | Equal-weight Sharpe / CAGR / MaxDD | Coverage-tilt Sharpe / CAGR / MaxDD | Gate |
|---|---|---|---|
| Primary (2023-01 → 2025-12) | 1.448 / 61.0% / −41.4% | 1.446 / 62.3% / −43.4% | **FAIL** — CAGR edges up, Sharpe doesn't, drawdown worsens |
| Prior (2020-01 → 2022-12) | 0.690 / 25.4% / −45.4% | 0.690 / 25.4% / −45.4% | **FAIL** — tilt is *identical* to equal-weight: no name had a full year of backlog history that far back, so the signal never engaged |

**Verdict: FAILED.** Coverage-alone is not the useful half of the value-chain
reframe's blended margin+coverage composite — on its own it's indistinguishable
from equal-weight in the window where it has any data at all, and marginally worse
risk-adjusted in the window where it does engage.

## What's still open in §8.3

- The *text-mining* half (score lead-time/price-escalation language in 10-K/10-Q
  MD&A — a pricing-power signal, distinct from both backlog growth and coverage)
  was not attempted here.
- Quality/balance-sheet weighting (§8.4) and options-structured hedge (§8.5, blocked
  on data) remain untried.
