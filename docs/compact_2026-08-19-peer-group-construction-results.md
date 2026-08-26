# Peer-Group (Basket-vs-Basket) Construction — Implementation Results

**Status:** Implemented and validated. **The design's core hypothesis does not hold on backtest data** — reported honestly below, not spun.

**Spec:** `docs/superpowers/specs/2026-08-15-peer-group-construction-design.md` (approved design, now built).
**Code:** `grid_resilience.portfolio.construction.build_grouped_weights()`, `build_rolling_weights(grouped=True)`, `--peer-group` CLI flag. `PEER_GROUP_CONSTRUCTION=False` by default — matches the `BUSINESS_MODEL_ARCH`/`USE_ICR` pattern of off-until-validated. 12/12 unit tests pass (`tests/portfolio/test_construction.py`); full suite (98 tests) unaffected.

## Why this was built

`docs/handoff_2026-08-15-pairs-basket-construction.md` diagnosed the strategy's underperformance vs XLU as architectural: ranking the whole 18-name universe and going long top-3/short bottom-5 exposes both legs to sector beta (long book corr to XLU +0.76, underperforming buy-and-hold XLU on raw return; short book Sharpe no better than a naive equal-weight short of the whole universe). The fix: split the universe into business-model peer groups (merchant/mixed/regulated, from `utility_node_map.TICKER_NODE_MAP`) and build long+short baskets *within* each group, on the theory that legs sharing a similar regulatory/rate/beta profile would cancel sector beta while preserving diversification.

## What was tested

Full pipeline, `--arch hard-switch --regulated-signal dc-queue`, Apr 2018–Dec 2025, 18-ticker/5-ISO universe (same config as `output_dcqueue/dc_executive_summary.html`), run twice: once with `--peer-group`, once without. Same underlying factor scores in both runs (IC is identical: 0.0282 @ 21d, 0.0615 @ 63d) — only the portfolio-construction step differs.

## Result: risk dropped meaningfully, Sharpe did not improve, and the sector-beta diagnosis was not confirmed at the whole-strategy level

| | Whole-universe (baseline) | Peer-group |
|---|---|---|
| Sharpe | −0.276 | **−0.343** (worse) |
| Ann. Return | 1.48% | 1.69% |
| Ann. Vol | 9.13% | **6.73%** (−26%) |
| Max Drawdown | −19.39% | **−13.72%** (−29%) |
| Strategy corr. to XLU | +0.104 | +0.198 (worse) |
| Long-book corr. to XLU | +0.777 | +0.840 |
| Short-book corr. to XLU | −0.803 | −0.776 |

Neither configuration beats XLU (Sharpe 0.162, Ann. Return 7.90%).

**Why Sharpe got worse despite lower volatility:** the risk-free rate (4%) exceeds both strategies' raw annualized return (~1.5–1.7%), so excess return is negative in both configurations. Dividing a negative excess return by a *smaller* volatility produces a *more negative* Sharpe ratio — the metric penalizes the very risk reduction the redesign achieved. Peer-group construction did what it mechanically set out to do (cut vol ~26%, cut max drawdown ~29%), but that's not the same as generating positive excess return, and Sharpe conflates the two.

**Why the sector-beta diagnosis didn't hold at the whole-strategy level:** a per-sleeve breakdown (computed directly from `build_grouped_weights()`, not just the aggregate) shows within-sleeve dollar-neutrality *is* substantially cancelling each sleeve's own beta — net sleeve-to-XLU correlation is +0.16 (regulated), +0.15 (mixed), +0.02 (merchant), each much lower than that sleeve's long leg alone (+0.78, +0.66, +0.56 respectively). The mechanism works exactly as designed *within* each peer group. But the whole-universe baseline's net strategy correlation to XLU (+0.104) was already low — apparently by chance, from how its specific top-3-long/bottom-5-short picks happened to offset across the full universe — and peer-group construction's aggregate (+0.198, capital-weighted across sleeves that are each only partially cancelled) didn't beat that. Business-model tagging (merchant/mixed/regulated) is a reasonable regulatory proxy but evidently not a tight enough empirical beta match within a group to fully net out at the portfolio level.

## Implementation is not in question

- 12 unit tests (`tests/portfolio/test_construction.py`) verify: total book sums to exactly ±0.5, each sleeve is independently dollar-neutral, correct tickers are selected per sleeve given known scores, capital scaling matches the group-size-proportional design (2/18, 5/18, 11/18), and `build_rolling_weights(grouped=True)` produces the same `[date, ticker, weight]` shape as the ungrouped path.
- A live per-sleeve diagnostic (reconstructing weights from `factor_scores.csv` and slicing by `business_model`) confirms the construction behaves exactly as specified — the negative result is a genuine finding about the strategy's underlying names and factor scores, not a bug in `build_grouped_weights()`.

## Honest bottom line

Peer-group construction is a real, measurable win on risk (lower vol, lower drawdown) but does not close the gap to XLU, and the specific mechanism the design predicted (lower whole-strategy correlation to XLU) did not materialize — the whole-universe baseline's correlation was already low. The deeper problem flagged in the original diagnosis stands: the factor's picks, in aggregate, don't generate enough raw excess return to beat a 4% risk-free hurdle, let alone XLU's 7.90% annual return, regardless of how the book is constructed. Fixing that likely requires improving the underlying signal's return-generating power (not just risk-shaping the wrapper around it) — the RT/DA spread and outage/reserve-margin signal ideas (`docs/handoff_2026-08-15-rt-da-spread-signal.md`, `docs/handoff_2026-08-15-outage-reserve-margin-signal.md`) remain the more promising next lever, now that this construction question has been tested and answered rather than left as an open hypothesis.

## Recommendation

Leave `PEER_GROUP_CONSTRUCTION=False` by default — it doesn't yet justify becoming the default path. Keep it available via `--peer-group` since the volatility/drawdown reduction may be independently useful (e.g. if paired with a signal that does generate positive excess return, the same construction could then also deliver a better Sharpe). Don't invest further in tuning the peer-group split (e.g. grid-searching book sizes) before the underlying signal's excess-return problem is addressed — per the sleeve diagnostics above, better peer-group tuning would only sharpen beta-cancellation, which was not the binding constraint here.
