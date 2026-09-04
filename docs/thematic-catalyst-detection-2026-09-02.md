# Utility load-forecast filings as a thematic-catalyst detector — evidence (2026-09-02)

**What this is:** not a trading signal — a different kind of tool. Everything else
built this session (FTR bids, congestion regime, queue-velocity) is a *timing
overlay* on an *already-identified* trade. This is evidence that the trade's
underlying catalyst (AI-driven data-center demand) was **publicly, specifically
named by a grid regulator months before the equity market re-rated on it** — i.e.
a candidate **idea-generation** signal: watch utility/ISO load-forecast filings
for a brand-new named large-load category, as an early flag that a theme is
starting, not as a timing dial on a theme already known.

## 1. The PJM evidence

PJM's Load Analysis Subcommittee, **November 29, 2022** ("2023 Preliminary PJM
Load Forecast," Andrew Gledhill, Resource Adequacy Planning — slide 3,
`.../las/2022/20221129/item-04a---2023-preliminary-pjm-load-forecast.pdf`):

> "Forecast Adjustments – **Data Centers (AEP, APS, Dominion)**; NRBTMG to DR
> (AEP, ATSI, PL); Peak Shaving Adjustment (EKPC)"

This is an explicit, dated, named line item — "Data Centers" called out as a
distinct large-load adjustment category, three specific utility zones attached.
Notably, this is **one day before ChatGPT's public launch** (Nov 30, 2022) —
the PJM filing did not anticipate the AI narrative; it was tracking data-center
interconnection requests that were already in the utility pipeline independent
of it.

## 2. Real lead time, measured against actual price action

Reconstructed the long buildout basket (US-9 + 4 foreign ADRs, equal-weight)
vs. SPY, monthly, 2022-06 through 2023-12:

| Month | Basket cum. index | SPY cum. index |
|---|---|---|
| 2022-10 | 0.785 | 0.756 |
| **2022-11** | **0.907** | 0.814 |
| 2022-12 | 0.975 | 0.863 |
| 2023-03 | 1.092 | 0.840 |
| 2023-06 | 1.197 | 0.901 |
| **2023-07** | **1.342** | 0.950 |
| 2023-09 | 1.484 | 0.967 |

A modest divergence is already visible in Nov-Dec 2022 (the same window as the
PJM filing), but that period is confounded with a broad market bottom, so it
isn't cleanly attributable. The unambiguous, large re-rating happens
**June-September 2023** — basket goes from ~0.90x to ~1.48x SPY-relative in
three months. That's **roughly 7 months after the PJM filing**. A watcher of
utility load-adjustment filings would have had a genuine multi-month head start
on the equity move, not a hindsight-only story.

## 3. Cross-ISO check — the important caveat

ERCOT's Large Flexible Load Task Force (LFLTF) was founded **even earlier**
(first meeting April 14, 2022; charter approved June 15, 2022) — its own
stated founding rationale: *"Texas electric utilities were experiencing a high
volume of large load interconnection requests for **data centers, crypto
mining facilities, and green hydrogen sites**."*

But the actual September 2022 LFLTF deck available
(`ercot.com/files/docs/2022/09/23/LFLTF Deck Sept 26.pdf`, presented by the
Texas Blockchain Council) is entirely about **Bitcoin mining** — companies
operational, MW energized (700-1,000 MW in 2021 → 1,500-2,000 MW in 2022),
median facility size — with no data-center-specific content in the material
actually reviewed.

**This matters**: a naive keyword detector watching ERCOT alone in 2022 would
have flagged "large load boom" and pointed toward Bitcoin miners (RIOT, MARA,
CLSK) — a related but genuinely different trade (crypto-price-driven, not
AI-capex-driven) with a very different risk profile. PJM's Nov 2022 language
is the cleaner, more specific hit for the actual trade that worked. A working
detector needs to **distinguish sub-categories within "large load"** (data
center / crypto mining / hydrogen / EV-fleet charging, etc.), not just flag
"large load MW is growing" — the raw MW figure looks similar across very
different underlying themes.

## 4. Implication

Utility/ISO load-forecast-adjustment filings are a genuine, public,
**months-ahead-of-price** leading indicator for *which theme is starting* —
but only when the filing specifically names the demand driver. This is best
used as a periodic **thematic scan** (watch PJM LAS / ERCOT TAC / MISO / other
ISO filings for a newly-named large-load category, at what zones, what
early MW) rather than a backtestable trading signal in the usual sense — there
is no repeated time series to gate on, just a small number of discrete
"a new theme just got named publicly" events per decade.

## 5. Not yet done

- Check whether ERCOT's *later* 2023-2024 materials specifically separate
  "data center" from "crypto mining" as the AI narrative diverged from crypto
  (crypto-mining growth actually slowed/consolidated in Texas by 2023 per the
  Sept 2022 deck's own "end of the gold rush" framing, while data centers kept
  accelerating) — would sharpen the cross-ISO comparison.
- No systematic backtest is possible here (one historical instance) — the
  value is as a forward-looking watch process, not a validated factor.
