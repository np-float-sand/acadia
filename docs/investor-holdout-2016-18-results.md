# Investor strategy — slack-grid holdout, 2016–2018 (2026-09-25)

Research record only — **not** for the investor proposal. The lighter version of the holdout scoped in
`docs/investor-rule-robustness-results.md` §Addendum. No rule was tuned on these years. Reproduce:
`python research/electrification_strategy_probes/holdout_fetch_hits.py` (EDGAR wording hits for 10-Ks
filed 2012–16, 115 names, 0 failures → `electrification_strategy/data/vocabulary_hits_2012_2016.csv`),
then `python research/electrification_strategy_probes/holdout_2016_18.py`
(→ `output_html/img/tophalf/holdout_2016_18.json`).

Everything is rebuilt point-in-time from filings as published: membership (rules 1–5), the ranking,
the short score, supplier growth and rule 11. The strategy code is `investor_book.build`, unchanged,
with 10bp on every trade.

## Verdict: **FAIL** — the strategy does not survive a slack grid, and rule 11's growth condition is the main reason

| 2016–18 | Sharpe | Calmar | max DD | return a year | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|
| L0 rules 1–5, equal weight, no overlay, no short | **0.64** | 0.58 | −25.0% | **14.6%** | 39.5% | 26.8% | −14.8% |
| L1 + risk overlay | 0.55 | 0.49 | −25.8% | 12.7% | 29.3% | 32.0% | −16.2% |
| L2 + ranking, top half | 0.53 | 0.47 | −27.1% | 12.7% | 28.2% | 33.0% | −16.1% |
| L3 + hold buffer | 0.48 | 0.41 | −28.9% | 11.8% | 31.0% | 30.5% | −18.2% |
| L4 + short $0.25 | 0.40 | 0.38 | −26.4% | 10.0% | 28.7% | 23.3% | −16.2% |
| L5 + rule 11, real yield only | 0.37 | 0.36 | −26.4% | 9.5% | 26.9% | 23.3% | −16.2% |
| **L6 + rule 11, both conditions (the proposal)** | **−0.11** | 0.06 | −26.4% | **1.5%** | **1.3%** | 23.0% | −16.2% |
| GRID ETF | 0.22 | 0.22 | −29.7% | 6.5% | 22.7% | 27.6% | −22.8% |
| PAVE ETF (from Mar 2017) | −0.41 | −0.16 | −27.8% | −4.4% | — | 17.5% | −19.2% |
| S&P 500 | 0.43 | 0.47 | −19.4% | 9.2% | 12.0% | 21.8% | −4.6% |
| XLI | 0.38 | 0.37 | −24.1% | 8.9% | 20.0% | 24.1% | −13.3% |
| XLU | 0.51 | 0.68 | −15.6% | 10.6% | 16.1% | 12.1% | 4.0% |

- **The raw screen (L0) held up well** — Sharpe 0.64, above GRID, the S&P and XLI. The core idea
  (capital-light suppliers that say so in their filings) survived a slack grid.
- **Every added rule lowered Sharpe out of sample**, the reverse of 2019–26. The overlay (0.64 → 0.55),
  the short (0.48 → 0.40) and the buffer (0.53 → 0.48) all cost something; the ranking was roughly
  neutral (0.55 → 0.53).
- **Rule 11's growth condition did the damage.** Suppliers' real revenue was negative from 2016 Q1 to
  2016 Q4 (−5% at worst; the oil-crash industrial recession), so the strategy sat in T-bills from
  1 March 2016 to 28 February 2017 — through a year the screen rose 27–39%. The real-yield-only rule
  (min −0.06%, never below zero on a day that mattered) would have stayed invested: Sharpe 0.37 against
  −0.11. **Revenue lags prices:** the stocks recovered on expectations while trailing sales were still
  falling, so a revenue-based switch exits late and re-enters late.
- **Nothing protected against 2018.** The Q4-2018 selloff took the strategy down 26% — more than its
  worst fall in the whole 2019–26 backtest. Rule 11 stayed on (real yields and supplier growth both
  positive), and the volatility target did not cut fast enough.
- **The ranking hurt only mildly and not where the thesis fears.** Ranking minus equal weight: +1.4% a
  year while rule 11 was on, −2.2% while off; by year −0.8%, +0.6%, +0.1%. Consistent with "no edge in a
  slack grid", not with active harm.

## Caveats

- Only 3 years, one of which rule 11 spent in T-bills.
- The long book was thin — 5.8 names on average (as few as 3) against ~14 since 2023 — because few
  companies passed the rules then. The short averaged 12 names on 8 of its 14 signals (power betas
  need ISO data from mid-2017).
- Survivorship bias: the universe is today's listed companies. It flatters the long side and removes
  failed shorts, so the true holdout result is probably worse, not better.
- Prices start 2015, so 2016's trailing windows (volatility target, stretch cut) are short.

## What this implies (not acted on)

1. **Revert rule 11 to real yield only.** The growth condition was adopted today because it matched the
   regime strip; its first out-of-sample test cost 25 points in one year. Real yield alone tested the
   same in 2019–26 and far better here.
2. **The stack's value is regime-specific.** Out of sample the plain screen beat every elaboration. The
   case for the rules rests on 2023+ (and the drawdown control in 2020–22), so the proposal should not
   imply the full stack is proven across regimes.
3. **The 2018 drawdown is a live risk the proposal does not show.** A −26% fall with nothing triggering
   is the realistic slack-grid downside.
4. The full 2010–18 build would add the 2012–13 negative-real-yield episode and a longer slack sample.
