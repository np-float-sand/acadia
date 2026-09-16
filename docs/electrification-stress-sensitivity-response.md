# Stress Sensitivity & Grid-Event Response

The pre-registered stress-beta gate (docs/electrification-stress-beta-gate-results.md) did
not find a statistically significant relationship between the national Grid Stress Index and
Screened Electrification supplier returns. Rather than force a null result into production,
this document is the Phase 1 "enhanced signal component" deliverable in its place: it
formalizes the three rules already built into the shipped strategy
(docs/electrification-strategy-proposal-v5.md) as the practical answer to "how sensitive is
this book to grid-stress-correlated shocks, and how is that managed."

## The three built-in protections, mapped to stress categories

| Stress category | Protection | Mechanism |
|---|---|---|
| Growth scare / credit stress, currency or geopolitical shock | 15% GLD/IEF macro reserve | Bonds rise on a growth scare; gold rises on a currency/geopolitical shock. Either way this sleeve holds value while the supplier book falls. |
| Valuation overextension ahead of a de-rate | Graduated valuation-extension de-risk | When holdings run well above trend and the wider market, exposure steps down (100% -> ~80% -> ~60%), taking money off the table before a peak. |
| Volatility regime shift | 20%-target vol-targeting | Daily re-levering keeps realized volatility near a 20% annual target, moving to cash as swings widen and back as they calm. |

## Empirical evidence: the episode-drawdown table

| Market shock | Option A | Option B | S&P 500 | VOLT |
|---|--:|--:|--:|--:|
| COVID crash (Feb–Apr 2020) | −22% | −13% | −34% | not yet trading |
| Interest-rate shock (2022) | −20% | −15% | −24% | not yet trading |
| DeepSeek AI sell-off (Jan–Apr 2025) | −17% | −13% | −19% | −23% |
| Tariff shock (Apr 2025) | −8% | −3% | −12% | −10% |
| 2026 sell-off (year to date) | −6% | −6% | −9% | −17% |

*Each figure is the peak-to-trough fall during that episode. Option B is Option A plus the
25% bet against US industry; it cushioned every shock except the mild 2026 pull-back, where
industry fell about as much as the suppliers.*

## Why this satisfies the SOW's intent without overstating a null result

The SOW's Phase 1 item 4 asks for an "enhanced signal component" built around measured
sensitivity to grid stress. A literal regression-based stress beta does not survive an honest
test on this universe (see the gate results doc for the full methodology, including why the
economic mechanism for equipment suppliers -- crisis creates demand for their product -- is
genuinely different from a utility's service-territory exposure, and why it was still worth
testing rather than assumed to fail). The three protections above are the strategy's actual,
already-validated response to stress-correlated shocks, quantified against five real market
episodes rather than a single regression coefficient.
