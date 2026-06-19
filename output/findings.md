# Grid Resilience Strategy — Research Findings

Accumulated findings from analysis session, June 2026. Raw notes for follow-up.

---

## 1. Benchmark Comparison — Raw vs Risk-Adjusted

**Finding:** XLU buy-and-hold returned +113% over the same backtest period (Apr 2018–Dec 2025), versus the strategy's +90%. The strategy underperforms on raw cumulative return.

**But:** XLU's max drawdown was −36.1% vs −13.7% for the strategy. XLU's annual volatility was 20.6% vs 13.2%. On a return-per-unit-of-risk basis the strategy is comparable — and the drawdown protection is substantial.

**Honest framing:** This is a risk-managed long/short strategy, not a long-only vehicle. The right comparison is risk-adjusted, not absolute. The strategy is a complement to long-only utility holdings, not a replacement.

**Follow-up question:** At what portfolio allocation does adding this strategy to a long-only utility book improve the overall Sharpe meaningfully?

---

## 2. Stressed vs Calm IC — The Chart is Misleading

**Finding:** The backtest chart's IC panel colours bars orange (stressed) and green (calm) based on whether the 21-day average GSI before each rebalance date exceeds 0.5. The visual impression is that the factor works better in stressed periods.

**Reality:** When computed properly (mean GSI window, matching backtest code exactly):
- Stressed months (n=64): mean IC **+0.137**, pct positive 55%
- Calm months (n=26): mean IC **+0.169**, pct positive 62%
- Difference: −0.032 (calm is *slightly better*)
- Welch t-test p = 0.856 — not significant at any conventional level

**Implication:** The stressed/calm IC split in the writeup is misleading and should be removed or reworded. The factor predicts cross-sectional returns at a similar rate in both regimes. The IC improvement narrative is not supported by the data.

**Root cause of visual confusion:** The GSI threshold of 0.5 is too low — 65 of 90 rebalance dates fall above it, so almost all bars are orange regardless of whether the grid is genuinely stressed.

**Follow-up question:** What GSI threshold would give a more economically meaningful stressed/calm split? Consider the 75th or 90th percentile rather than an absolute 0.5.

---

## 3. 2024 IC Breakdown — The AI Data Center Narrative

**Finding:** 2024 mean IC was −0.023 (near zero, effectively useless). The factor score at end of 2023 had a −0.727 Pearson correlation with 2024 full-year returns — nearly a perfect inversion.

**Cause:** VST (Vistra) and NRG both had large negative factor scores entering 2024 (VST: −1.43, NRG: −1.34) — the model was short them. But VST returned +262% and NRG +79% in 2024.

**Why:** The AI data center buildout created a structural demand for firm, always-on power delivered under long-term bilateral contracts. Data centers cannot run on intermittent renewables. Regulated utilities (Duke, Dominion, etc.) are structurally unable to offer commercial bilateral deals at scale — their rates, capacity commitments and capital allocation are all controlled by state regulators.

VST sat in exactly the right place: it owns large dispatchable gas and nuclear plants in ERCOT and PJM — the two most capacity-constrained grids in the country — with no regulatory obligation to serve at regulated rates. It could sign a 15-year bilateral contract with a hyperscaler without regulatory approval. When Microsoft announced a deal to restart Three Mile Island specifically to supply its data centers, it validated the thesis: tech companies would pay a premium for firm, low-carbon power, and VST was one of very few entities that could deliver it.

**The inversion:** Every grid stress headline that previously sold VST off (emergency fuel costs, reliability penalties) now bought it — because tighter grids meant more pricing power for the only generator willing to sign firm deals. The stress beta estimated on 2018–2023 data was obsolete, but the model had no way to know that the market's interpretation of the same physical signal had reversed.

**Follow-up question:** Would excluding VST and NRG from the universe (they are competitive/merchant generators, not regulated utilities) improve IC consistency? Or should they have a separate model?

---

## 4. How VST Flipped to a Positive Factor Score — The Mechanical Story

**Finding:** VST's factor score flipped from deeply negative (−1.4 to −2.2) to strongly positive (+2.25 to +2.5) in May 2024, staying positive through April 2025.

**Mechanical explanation:** The factor estimates stress beta via rolling OLS: regress each stock's daily returns against the composite GSI. If a stock tends to go *up* on high-GSI days, the beta is negative — the model classifies it as resilient — and assigns it a high positive factor score (long candidate).

By mid-2024, enough observations of VST rising on grid stress news had accumulated in the rolling window that the OLS estimate flipped sign. The model was descriptively correct — VST *was* behaving like a resilient stock. But the cause was the narrative (AI/power demand), not physical grid resilience. The model cannot distinguish between the two.

**Key insight:** The factor is a behavioural signal, not a fundamental one. It measures *how the stock responds to stress*, not *why*. When the "why" changes, the signal eventually follows — but with a lag that creates whipsaw losses.

**Follow-up question:** Could a momentum filter help — e.g., don't flip a stock's score direction unless the new beta estimate persists for N consecutive months? This would reduce false flips but also slow adaptation to genuine regime changes.

---

## 5. Why VST Was Lackluster in Q1 2025

**Finding:** The factor entered 2025 long VST at +2.2 score. VST returned −0.6% in Q1 2025. January and February IC were −0.69 and −0.50 — the worst months in the 2025 data.

**Cause:** DeepSeek. On January 27 2025, DeepSeek R1 demonstrated frontier-level AI performance at a fraction of the assumed compute cost. The entire AI infrastructure trade sold off (Nvidia −17% in a single day). VST had been re-rated almost entirely on the AI/data center power demand narrative. When that premise was questioned, VST gave back a chunk of its premium while boring regulated utilities (DTE, WEC, CMS) quietly ground higher.

**The factor's error:** It was long VST expecting it to behave like a resilient outperformer during grid stress, based on mid-2024 learned behaviour. But the stock was now reacting to AI compute demand economics, not grid conditions. The GSI had nothing to say about DeepSeek.

**Follow-up question:** This suggests the factor needs a "narrative override" or at minimum an awareness that competitive gen names are driven by power price macro more than grid stress topology. One proxy: track implied vol or correlation of VST with energy commodity indices vs. utility sector ETF. A sudden shift in that correlation might flag that a name has left the "utility" regime.

---

## 6. Why VST's Beta Flipped Negative Just as It Began to Deliver (H2 2025)

**Finding:** The factor flipped VST back to a strongly negative score (−1.4 in May 2025, deepening to −2.6 by July) just before VST had a +65% Q2 2025. Then in H2 2025, when the factor was short VST, VST began to underperform — and IC recovered strongly (Oct: +0.80, Nov: +0.63).

**Mechanical explanation:** By May 2025 the rolling OLS window had absorbed Q1 2025's VST weakness (flat/negative returns during elevated-GSI periods). Flat VST during high-GSI looked like positive stress beta again — stock goes nowhere when grid is stressed — so the model re-classified it as exposed and flipped it back to short.

VST then had a massive Q2 driven by earnings beats, nuclear asset revaluations, and renewed long-term power contract confidence — factors entirely outside the GSI framework. The model was short a stock that was rallying for idiosyncratic reasons.

**Then the model was right:** By Q3–Q4 2025, the AI power demand narrative had normalised, VST's multiple compressed, and the factor being short it worked again (IC +0.38, +0.80, +0.63 in Sep–Nov).

**The whipsaw pattern in summary:**
- Late 2023: model short VST → VST starts rallying on AI narrative → IC suffers
- May 2024: model flips long VST → correct for mid/late 2024 → IC recovers
- Jan 2025: DeepSeek hits, VST flat → model still long → IC suffers
- May 2025: model flips short VST → VST has one more AI rally (Q2) → IC suffers
- H2 2025: narrative cools, VST underperforms → model short → IC recovers

**Root finding:** Two narrative flip-flops in 18 months caused four IC hits. The rolling OLS always lags the narrative shift by roughly one quarter.

---

## 7. Universe Contamination — VST and NRG Are Not Regulated Utilities

**Finding (cross-cutting):** Most of the factor's IC instability in 2023–2025 can be traced to VST and NRG, both of which are competitive/merchant generators with equity behaviour driven by power price macro and energy market narratives rather than regulated utility fundamentals.

The original universe design included them because they operate in stressed grid regions (ERCOT, PJM) and have physically meaningful stress exposure. That rationale is correct. But their *equity* behaviour is that of commodity-linked energy stocks in bull cycles, not regulated utilities — they can go up 262% in a year or down 23%.

**Possible fixes:**
1. Exclude VST and NRG entirely and test whether IC stabilises
2. Keep them but cap their factor score magnitude so they can't dominate the portfolio
3. Build a separate sub-model for competitive gen names that incorporates power price momentum alongside stress beta
4. Add a correlation filter: if a stock's rolling correlation with XNG (oil & gas E&P ETF) exceeds its correlation with XLU, flag it as having "left the utility regime"

**Follow-up question:** Run the full backtest excluding VST and NRG. Hypothesis: Sharpe improves, max DD similar, and IC is more stable — but raw return likely drops because VST/NRG contributed significantly when the model was correctly long them (mid-2024).

---

## 8. The Factor Works — But in the Right Universe

**Overall conclusion from the analysis:**

The stress beta signal is real and physically motivated. The IC is positive in both stressed and calm regimes (+0.14 overall). The strategy generates a Sharpe of 0.31 vs XLU's 0.24 (using the backtest's own method) with 2.6× lower max drawdown.

The instability is not in the signal itself but in the universe composition. Two names (VST, NRG) that straddle the utility/commodity-gen boundary account for a disproportionate share of both the wins (2023, mid-2024) and the losses (2024 H1, 2025 H1). A cleaner universe of pure regulated utilities would likely show a smoother IC series at the cost of some upside.

The strategy as currently constructed is best understood as: **a grid stress factor applied to a mixed utility/competitive-gen universe**, where the competitive-gen component introduces macro energy narrative risk that the factor cannot price.


---

## 9. Would ICR Help With the VST Case? If Not, What Might?

**Finding:** The Interest Coverage Ratio factor component would **not** have helped with the VST case and would likely have made it worse.

ICR measures financial fragility — a company with weak interest coverage is more likely to get hurt when stress events drive up its costs. VST's problem in 2024 was the opposite: the factor was correctly identifying it as financially exposed to grid stress (high fixed costs, commodity fuel risk), but the *market* stopped caring about that exposure and started pricing in a structural growth narrative. ICR would have reinforced the short thesis on VST, not contradicted it. Lower ICR = more short signal = even worse 2024 IC.

ICR is the right tool for identifying utilities that will be *hurt* by stress events. It has nothing to say about whether the market is in a regime where stress events are being interpreted as a positive for that name.

---

**What might actually help:**

**1. Stock-to-sector correlation shift (most actionable)**

Monitor each name's rolling 60-day correlation with XLU (utilities ETF) vs. XNG or XLE (energy/commodity ETF). When a stock starts behaving more like a commodity name than a utility, it has left the factor's domain of validity — the stress beta signal is no longer measuring what it thinks it is.

Implementation: if `corr(stock, XNG) > corr(stock, XLU)` for two consecutive months, exclude the name from the scored universe that month. Re-admit when the condition reverses.

This would have flagged VST and NRG as having "left the utility regime" by mid-2023, before the worst IC damage in 2024.

**2. Power price momentum**

The factor treats all high-GSI periods the same regardless of *why* the grid is stressed. A grid stressed by a one-off polar vortex is different from a grid structurally stressed by persistent load growth. If long-dated power forwards in ERCOT or PJM are trending upward for structural reasons (not just seasonal), merchant generators are in a regime where high stress beta is an asset, not a liability.

A simple version: if the 12-month rolling average of ERCOT day-ahead hub prices is rising year-over-year, flip the sign interpretation for merchant generators — their high stress beta becomes a positive score rather than a short signal.

**3. Forward power curve slope**

If the long-dated power forward curve is in contango (future prices above spot), the market is pricing in scarcity rents for generators with firm capacity. Stress beta estimated from historical spot data will lag this regime shift. Tracking whether the forward curve is in backwardation or contango for the relevant ISO could act as a regime flag for competitive gen names specifically.

**Prioritisation:**

The correlation shift filter (option 1) is the most immediately buildable — it uses only equity price data already in the pipeline, requires no new data sources, and provides a clean binary flag. Options 2 and 3 require forward curve data (available via EIA or CME) but would add a more fundamental explanation for *why* the correlation shift is happening.

**Follow-up question:** Run a backtest variant that applies the XLU/XNG correlation filter and excludes VST/NRG when they fail it. Hypothesis: eliminates the 2024 IC crater and most of the 2025 H1 drag, with modest reduction in 2023/mid-2024 upside.


---

## 10. How to Change the Beta / Grid Stress Factor to Cope With VST's Changing Behaviour

**Design question — no new data needed.**

The root problem is that the rolling OLS estimates *how the stock behaves* during grid stress, which is correct when the market's interpretation of that stress is stable, but breaks when the interpretation flips. There are a few ways to address this at the beta level:

**Option 1: Anchor to operational exposure, not equity returns (most robust)**

Instead of estimating beta from rolling equity-vs-GSI regressions, derive it from fundamental data: what fraction of the company's revenue is spot-price exposed vs. contracted or regulated? What grid region? What fuel mix? These don't change when investor narratives shift. VST's physical exposure to ERCOT merchant pricing is the same in 2023 and 2024 — what changed is how investors *value* that exposure, not the exposure itself. A fundamentally-anchored beta wouldn't have flipped. The cost: you lose the ability to detect genuine changes in operational exposure over time (asset sales, new contracts). You'd want to update it annually rather than monthly.

**Option 2: Orthogonalize the beta to power price returns (most elegant)**

Before running OLS of stock returns on GSI, partial out the effect of ERCOT/PJM power price returns from both variables. The residual beta measures "stress sensitivity beyond what's explained by spot power prices." In 2024, VST's equity was almost entirely driven by power price expectations — removing that component would have kept its stress beta closer to zero rather than flipping it. This is cleaner than a regime flag and doesn't require a binary in/out decision. It doesn't require classifying names as merchants or not — it just removes the power price component from everyone and lets the residual stress sensitivity speak.

**Option 3: Stability-weighted beta**

Measure the month-to-month variance of the rolling beta estimate. Names with highly unstable betas (like VST post-2023) get their score magnitude damped — you trust the estimate less when it's jumping around. Stable names (WEC, DTE) get full weight. This wouldn't fix the direction error but would have reduced how heavily VST featured in the portfolio during the uncertain period.

**Option 4: Adaptive window length with structural break detection**

Rather than a fixed rolling window, use adaptive window length based on structural break detection. If the Chow test detects a break in the beta series, reset and use only post-break data. This would have caught the mid-2024 flip sooner and the May 2025 flip sooner — reducing lag in both directions. Still doesn't solve the direction problem, just reduces the lag.

Of these, option 2 is probably the most elegant because it doesn't require a classification decision about which names are "merchants."

---

## 11. Is It All Merchants That Would Need Some Correction?

Not all merchants equally — it's a spectrum based on what fraction of EBITDA is spot-price exposed rather than locked in under regulation or long-term contracts.

The useful taxonomy for the current universe:

| Name | Character | Needs correction? |
|------|-----------|-------------------|
| VST, NRG | Primarily merchant/competitive gen — majority of revenue is spot-price exposed | Yes, most severely |
| ETR, EXC | Mostly regulated with merchant generation subsidiaries | Partial — contamination risk in bull power markets |
| DUK, SO, NEE, D | Fully regulated or mostly contracted | No — stress beta is purely cost risk, signal works as designed |
| WEC, DTE, CMS, AEE | Fully regulated, no meaningful merchant exposure | No |

The distinguishing variable isn't "merchant yes/no" — it's whether the stock's primary *valuation driver* is power prices vs. regulated rate base. When power prices are rising structurally, VST and NRG get re-rated like energy commodity companies. ETR and EXC might show mild contamination. The others won't.

A practical screen you could operationalize: if more than 40% of EBITDA comes from unregulated/merchant generation (available from annual reports or S&P Capital IQ), flag the name as "regime-sensitive" and apply the orthogonalization or operational beta approach rather than the rolling equity OLS. That gives you a principled rule rather than a hard exclusion.

The harder question is whether VST and NRG *should* be in the universe at all. They add genuine cross-sectional dispersion and when the model has them right (mid-2024 to mid-2025) they add significant return. Excluding them entirely probably stabilizes IC but leaves alpha on the table. The orthogonalization approach would let you keep them while making the signal more robust.

