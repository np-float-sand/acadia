# REIT-breadth probe — Screened Electrification

**Date:** 2026-09-15
**Source:** ssrn-5736562.pdf (Mamkhezri, Sun & Yang, 2026, "The Hidden Cost of the Cloud") +
user request. The paper gives a causal PJM congestion-price estimate (data-center entry -> ~70%
/ $2.49/MWh increase in the transmission-induced price component); it says nothing about REITs
directly. This probe tests the user's hypothesis that data-center-exposed REITs could add
breadth to the 27-name supplier universe.

## Candidates and scoping

DLR/EQIX excluded -- already `config.HEDGE_SHORT_TICKERS` (the conditional short leg); can't
sensibly be long and short the same names. Tested: IRM (Iron Mountain), AMT (American Tower) --
both have data-center segments and institutional tenants. Both pass the existing `screen`
criteria's "customer_institutional + >=2 of 3" bar (see universe_seed.csv comments for the
REIT-specific FFO-based reading of `earnings_valued`).

## Method

New `screen_reit` construction = the existing `screen` pool (provably unchanged -- REIT rows
are explicitly excluded from it) union IRM/AMT. Ran the full backtest comparison grid
(marquee/frozen/thematic/screen/screen_reit x plain/+val/+val+sleeve/+val+sleeve+short).
Effective names: `screen` 27.0, `screen_reit` 29.0 (confirms both new names are unconstrained
by the 25% single-name cap). Real run: `poetry run python -m electrification_strategy --universe
all --no-plot --output /tmp/reit_probe`, 2017-06-01 to 2026-08-31, full universe including
fresh yfinance fetches for IRM/AMT. One unrelated warning (`THR` 404'd on this particular
yfinance fetch, "possibly delisted") did not affect results: THR is a permanent, known gap
(acquired by CECO Environmental, delisted 2026-06-01, 100% NaN -- see
docs/electrification-stress-beta-gate-results.md), so the 252-day listing-eligibility gate
drops it. `screen`'s effective-N of 27.0 is therefore exactly as expected -- it is the 28
rule-passing seed tickers *minus* THR, not a sign that THR was still in the book.

## Pre-registered gate

For each of the four overlay-stack cells, `screen_reit`'s Sharpe must not be >0.05 below the
matching `screen` cell, and MaxDD must not be >3pp worse.

| stack | screen Sharpe | screen_reit Sharpe | delta | screen MaxDD | screen_reit MaxDD | delta |
|---|--:|--:|--:|--:|--:|--:|
| plain | 0.705 | 0.720 | +0.015 | -28.19% | -27.35% | +0.84pp |
| +val | 0.667 | 0.692 | +0.025 | -28.19% | -27.35% | +0.84pp |
| +val+sleeve | 0.693 | 0.716 | +0.023 | -23.99% | -23.24% | +0.75pp |
| +val+sleeve+short | 0.655 | 0.685 | +0.030 | -23.17% | -21.78% | +1.39pp |

All four cells clear both legs of the gate, and not marginally: `screen_reit` beats `screen` on
*every single* Sharpe and MaxDD figure in the grid, not just clears the "not worse than" bar.

### Run reconciliation (two runs, two number sets)

The table above reports the original Task 7 probe run (`/tmp/reit_probe`). Task 9 later
re-ran the whole grid independently from a cold cache (`output_electrification_v2/metrics.csv`)
and produced slightly different *absolute* Sharpe/MaxDD figures for the same cells — normal
run-to-run variance from cache staleness for individual tickers between the two fetches, not a
methodology change. Both number sets are disclosed here rather than silently reconciled to one.

| stack | Task 7 Sharpe delta | Task 9 cold-run Sharpe delta | Task 9 `screen` / `screen_reit` Sharpe | Task 9 MaxDD delta |
|---|--:|--:|--:|--:|
| plain | +0.015 | +0.013 | 0.713 / 0.726 | +1.03pp |
| +val | +0.025 | +0.021 | 0.675 / 0.696 | +1.03pp |
| +val+sleeve | +0.023 | +0.020 | 0.708 / 0.728 | +0.78pp |
| +val+sleeve+short | +0.030 | +0.026 | 0.662 / 0.688 | +1.91pp |

**The ADOPTED verdict is robust to this.** Every `screen_reit - screen` Sharpe delta stayed
positive in both runs, and every MaxDD delta stayed favorable in both runs; no cell comes close
to the gate's "Sharpe not >0.05 below / MaxDD not >3pp worse" bar in either. The deltas — which
are what the gate actually tests — are stable to within ~0.004 Sharpe across the two runs; only
the absolute levels moved.

**Verdict: ADOPTED**

`screen_reit` clears the gate on all four cells with margin. Kept as a documented, available
construction (`universe.CONSTRUCTIONS` now includes `"screen_reit"`).

Does it overtake `pick_winner`'s overall pre-registered winner? **No** -- the winner stays
`('marquee', 'plain')` (Sharpe 0.772), unchanged by this addition; `marquee`'s 9-name
concentrated book still has the highest raw Sharpe in the full 20-cell grid, and `pick_winner`
doesn't look inside the `screen`/`screen_reit` family specifically. Restricting to the cells
that clear the `pick_winner` risk gates (MaxDD >= -27%, both subwindow Sharpes >= 0.4, feared-PnL
>= -2%, drag <= 6%): the same two stacks pass for both constructions (`+val+sleeve`,
`+val+sleeve+short`), and in both of those `screen_reit` strictly dominates `screen`
(`+val+sleeve`: Sharpe 0.716 vs 0.693, MaxDD -23.2% vs -24.0%; `+val+sleeve+short`: Sharpe 0.685
vs 0.655, MaxDD -21.8% vs -23.2%). Since the strategy's actual shipped lineage is
`screen +val+sleeve` (RESEARCH-LOG.md §1's "Current strategy" -- not the mechanically-selected
`marquee plain`), this is a real, if modest, improvement to the shipped construction if IRM/AMT
are folded in: same risk profile, uniformly better numbers, +2 names of breadth (27->29, +7%).

Not treated as a reason to revisit `marquee`'s standing as the pre-registered `pick_winner`
output -- that comparison is unaffected by this probe (`marquee` doesn't touch the seed CSV or
the REIT rows at all).
