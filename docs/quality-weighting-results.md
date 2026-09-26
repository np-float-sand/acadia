# Quality / balance-sheet weighting — results

**TL;DR: FAIL on both pre-registered gates. NOT ADOPTED.** RESEARCH-LOG open item 13
(`docs/handoff_2026-09-01-grid-buildout-long-short.md` section 8.4, re-scoped to the shipped
27-name `screen` universe). Pre-registration: `docs/quality-weighting-preregistration.md`
(locked before any backtest ran).
Quality-weighted `screen +val+sleeve` beats equal-weight in only 1 of 2 sub-windows (2019-22:
+0.005 Sharpe; 2023-26: -0.052 Sharpe) -- gate (a) requires both, so it fails outright.
Gate (b) is moot given (a)'s failure, but is reported anyway: the (quality minus equal-weight)
excess return has essentially zero relationship to a low-vol factor (beta=-0.003, t=-0.32) and
essentially zero mean (alpha=-0.08%/yr, t=-0.25) -- not a low-vol repackage, but also not a
source of return at all.
The honest read: this is a near-no-op weighting scheme (mean tilt factor stays within roughly
[0.76, 1.18] most rebalances, full range [0.54, 1.31] across the sample) whose small full-sample
drag (Sharpe 0.706 -> 0.696) is concentrated in exactly the regime (2023-26 AI-buildout) where
the strategy's returns actually come from, because the composite systematically underweights
VRT (mean composite z = -1.04, the most penalized name in the universe, driven by leverage) --
one of the book's strongest 2023-26 performers.

## 1. Reconciling the 27-name task description with this worktree's actual 28-name `screen`

This worktree's checked-out `electrification_strategy/data/universe_seed.csv` still includes
THR (Thermon Group) in the `screen` pool -- 28 names, not the 27 the task and
`docs/RESEARCH-LOG.md` describe. Verified directly: the worktree's `screen` pool minus THR
equals the 27-name list this probe uses, character-for-character. This looks like a version-sync
artifact between this isolated worktree checkout and the main repo (plausibly a real corporate
action -- THR does not appear in SEC's current `company_tickers.json` and yfinance returns
"possibly delisted" for it in this session, consistent with an acquisition sometime in 2026 that
a later main-repo commit removed THR for).

In practice this didn't matter: THR's price download failed entirely this session ("$THR:
possibly delisted; no timezone found"), so it never enters `available` inside
`simulate_basket` regardless of which universe definition is used. The as-shipped-in-this-
worktree sanity check (calling `electrification_strategy.backtest.run_comparison` unmodified,
`screen +val+sleeve`) and this probe's own 27-name-restricted equal-weight baseline produced
**identical** results (Sharpe 0.706, CAGR 14.59%, MaxDD -23.91% both), confirming both
constructions traded the same 27 live names this session. Both are reasonably close to the
documented main-repo reference (Sharpe 0.69, MaxDD -24%, beta 0.59) -- the small gap is
attributable to this worktree's slightly different seed/price-cache vintage, not a harness bug.

## 2. Formula and gate (as pre-registered, unchanged)

Composite z = mean(z(-leverage), z(margin_stability), z(fcf_conversion)) across the three
quality inputs, computed only for names with all three known as of a given date (else neutral,
factor 1.0). Weight factor = 1 + 0.25 * clip(composite_z, -2, 2), applied to the plain 1/N share,
renormalized, then the same 25% single-name cap the shipped book uses. Full detail:
`docs/quality-weighting-preregistration.md` section 3.

- Leverage: Debt/Equity from SEC XBRL instant facts (debt = union of five common debt tags;
  equity = StockholdersEquity, falling back to the NCI-inclusive variant).
- Margin stability: -stdev of trailing 8 quarterly gross margins (min 6 required), reusing
  `grid_equipment_basket/margin_data.py`'s revenue/gross-profit fetch unchanged, re-pointed at
  the 27-ticker CIK map built this session for the sibling backlog-pricing-power probe.
- FCF conversion: (CFO - Capex) / Revenue, annual (10-K) XBRL facts. Revenue pulled via the same
  Revenues-union-RevenueFromContract union `margin_data.py` uses for the same anti-stale-tag
  reason (see section 5 bug note below).

Gate (a): quality-weighted Sharpe beats equal-weight Sharpe in BOTH `config.SUBWINDOWS`
("2019-22", "2023-26"). Gate (b): |t(alpha)| >= 1.5 from OLS-regressing the monthly
(quality - equal-weight) excess return on a monthly SPLV-minus-SPY low-vol factor. PASS requires
both.

## 3. Results

### 3.1 Sharpe / CAGR / MaxDD by window

| window | cell | CAGR | Ann. vol | Sharpe | MaxDD |
|---|---|--:|--:|--:|--:|
| full (2017-06 -> 2026-09) | equal-weight | 14.59% | 15.30% | 0.706 | -23.91% |
| full | quality-weight | 14.50% | 15.43% | 0.696 | -23.91% |
| 2019-22 | equal-weight | 11.71% | 16.95% | 0.503 | -22.72% |
| 2019-22 | quality-weight | 11.81% | 16.94% | 0.508 | -22.51% |
| 2023-26 | equal-weight | 32.37% | 15.25% | 1.655 | -18.85% |
| 2023-26 | quality-weight | 31.97% | 15.57% | 1.603 | -18.91% |

**Gate (a): FAIL.** Quality-weighted beats equal-weight only in 2019-22 (+0.005 Sharpe, a
statistically meaningless difference on 4 years of daily data) and loses in 2023-26 (-0.052
Sharpe) -- the sub-window that actually carries the strategy's return (Sharpe 1.6+ vs 0.5).
Full-sample Sharpe is worse, not better (0.706 -> 0.696).

### 3.2 Low-vol-control regression (gate b)

Monthly OLS, n=112 months: `excess_r = alpha + beta * (SPLV_r - SPY_r) + eps`.

| stat | value |
|---|--:|
| alpha (monthly) | -0.00007 (-0.08%/yr annualized) |
| t(alpha) | -0.25 |
| beta | -0.003 |
| t(beta) | -0.32 |
| R-squared | 0.001 |

**Gate (b): FAIL** (|t(alpha)|=0.25, needs >=1.5) -- but not for the "low-vol repackage" reason
the gate was designed to catch. Both the intercept AND the slope are statistically
indistinguishable from zero. This is not a disguised low-vol tilt masquerading as quality; it is
a weighting scheme whose net effect on returns is, within noise, nothing at all -- R-squared of
0.001 means the low-vol factor explains essentially none of the (tiny) excess-return series'
variance either.

## 4. Mechanism -- why it's a near-no-op, and why the small effect it does have is a drag

**Data coverage.** Of the 27 names: leverage available for 25, FCF conversion for 25, margins
for 24. But the composite requires all three simultaneously, which is materially thinner and
grows over the sample: 0 of ~22-24 eligible names have a complete composite at the first two
rebalances (2018-08, 2018-11, before any 10-K FCF data exists), rising to 21-23 of 26-27 eligible
by 2023-26. Averaged across all 33 quarterly rebalances: mean 25.3 eligible names, mean 18.2 with
a complete (non-neutral) composite -- roughly a quarter of the book is weight-neutral (factor
1.0) at any given time purely from missing fundamentals, not from a genuinely middling quality
score. Four of 27 names (CARR, HUBB, JCI, MTZ) never once get a complete composite across the
whole 2019-2026 panel: HUBB and JCI have zero facts under any of the debt/equity/CFO/capex/
revenue tags this probe queries (confirmed directly against the raw SEC XBRL companyconcept
endpoint, not a code bug -- both return a genuinely empty `units.USD` object for every equity
tag tried); CARR (spun off 2020) has only 2 quarters of leverage data; MTZ's margin history has
just 4 disclosed quarters total, under the 6-quarter minimum this probe requires for a stability
estimate.

**Weight-factor range.** Mean per-rebalance factor range is [0.76, 1.18]; the global extremes
across the whole sample are [0.54, 1.31] -- real, non-degenerate dispersion (the mechanism is
working as designed, not silently defaulting to equal weight), but modest relative to the +-25%
top/bottom-half tilt this project's existing rank-tilt signals (`backlog_tilt_targets`,
`coverage_tilt_targets`) apply, because most names cluster within one standard deviation of the
cross-sectional mean on this particular composite.

**Per-name composite (2019-2026 quarterly mean, where computable).** Most rewarded: AME (+1.17,
but only 6/31 quarters -- thin, ignore), AYI (+0.57, 30/31 quarters -- a real, persistent result),
EME (+0.40), NVT (+0.39), ETN (+0.38), EMR (+0.38). Most penalized: VRT (-1.04, 19/31 quarters),
APG (-0.31), PRIM (-0.26), ENS (-0.24).

**VRT is the mechanism.** Vertiv carries meaningfully more balance-sheet leverage than the rest
of the universe (private-equity/LBO heritage prior to its 2020 SPAC listing) and is systematically
the most underweighted name in the quality-weighted book by a wide margin over any other name.
VRT is also one of the strongest individual performers in the AI-data-center-buildout window
(2023-26) -- exactly the regime driving this strategy's return (2023-26 Sharpe 1.6+ vs 2019-22's
0.5). A weighting scheme that penalizes leverage without regard to what the leverage financed
mechanically fights the book's own best-performing regime: it is systematically light exactly
the name the AI-buildout bull market rewards most, and that cost outweighs whatever
diversification/stability benefit the tilt provides elsewhere. This is the same "quality-vs-
growth regime mismatch" mechanism that shows up in equity factor investing generally, not
something specific to this universe or formula choice.

## 5. Data-sourcing note: a bug caught and fixed before scoring, not after

While building the FCF-conversion fetch, revenue was initially pulled with a first-nonempty
strategy across `{Revenues, RevenueFromContractWithCustomerExcludingAssessedTax}` -- and it
reproduced exactly the failure mode `margin_data.py`'s own docstring warns about: ETN's stale
pre-ASC-606 `Revenues` tag has data for 2010-2016 only, so first-nonempty stopped there and never
reached the modern tag, silently truncating ETN's FCF-conversion history to 7 annual rows ending
2016 (with nonsensical `fcf_conversion=inf` where the merge degenerated). Fixed to union both
tags (same pattern `margin_data.py` already uses for the identical reason) **before any gate was
evaluated** -- this is a data-pipeline correctness fix, not a post-hoc formula change, and the
locked pre-registration's section 2(c) ("Revenue = same union as (b)") already specified the
correct behavior; the bug was in the first draft of the implementation catching up to what was
already pre-registered.

## 6. Disposition

**FAIL / NOT ADOPTED.** Close RESEARCH-LOG open item 13. Neither pre-registered gate passes:
quality-weighting beats equal-weight in only 1 of 2 required sub-windows, and the excess return
it does produce is statistically indistinguishable from zero (not a low-vol repackage, but not a
source of alpha either). The original scoping's "prior: low" was directionally right, though the
realized outcome is more precisely "no measurable effect, tilted slightly negative by a
leverage-vs-AI-buildout regime mismatch on one name (VRT)" rather than "small positive effect
that doesn't clear the bar." Do not retry with a re-tuned tilt coefficient, alternate leverage
metric (e.g. Debt/EBITDA), or alternate lookback on this same panel -- that would be exactly the
after-the-fact tuning the pre-registration exists to prevent. If quality/balance-sheet weighting
is revisited, the open question worth testing first is whether the VRT-specific mismatch is
generic (any quality tilt fights the AI-buildout regime, because the regime specifically rewards
levered growth) or idiosyncratic to Debt/Equity as the leverage proxy -- untested here, out of
scope for this probe.

## 7. Files

- `docs/quality-weighting-preregistration.md` -- locked formula and gate, written before any
  backtest ran.
- `research/quality_weighting/cik_map_27.py` -- 27-ticker CIK map (copied from
  `research/backlog_pricing_power/cik_map.py`'s data; that module lives only in the main repo's
  untracked working tree and isn't visible from this worktree checkout).
- `research/quality_weighting/fetch_fundamentals.py` -- XBRL pulls for leverage (new instant-fact
  fetch), FCF conversion (new annual-flow fetch), and margin stability (reuses
  `grid_equipment_basket/margin_data.py`'s helpers unchanged). Own cache under
  `research/quality_weighting/_cache/`.
- `research/quality_weighting/signals.py` -- point-in-time visibility filter, winsorize/z-score,
  composite formula.
- `research/quality_weighting/run_backtest.py` -- sanity check, equal-weight vs quality-weighted
  comparison, sub-window Sharpes, low-vol-control regression, gate evaluation. Outputs under
  `research/quality_weighting/output/` (`metrics.csv`, `returns.csv`, `gate_summary.csv`,
  `quality_weight_diagnostics.csv`, `run_log.txt`).

Throwaway probe. Not wired into `electrification_strategy/` or `grid_equipment_basket/`.
