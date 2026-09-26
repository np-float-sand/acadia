# Proposal: Capex-Intensity Long/Short

**Date:** 2026-09-22
**Status:** proposal — implemented in `capex_intensity_strategy/`, not live
**Evidence:** `docs/capex-intensity-rule-results.md` (discovery),
`docs/algorithmic-long-short-books-results.md` §6 (the audit that forced it)

---

## 1. The proposal

Run a long/short book over the US power / electrification complex, ranked on one number:
**trailing-twelve-month capital spending divided by trailing-twelve-month revenue.**
Long the lightest quartile, short the heaviest quartile at half size.

> **Buy the companies that sell into the power buildout. Short the companies that are the buildout.**

Equipment makers, electrical contractors and fuel-cycle firms spend little of their own
money on plant — they convert someone else's capital programme into revenue. Utilities,
pipelines and miners spend enormous amounts to earn a regulated or commodity return on it.
Both are exposed to the same buildout; only one captures it without funding it.

## 2. Why this and not the alternatives

The shipped `electrification_strategy` book is excellent but its universe is a 33-name
list assembled in 2026, of which 30 of 33 pass the screen — the selection lives in the
list, not the rule. Every attempt this session to make a power-market book *mechanical*
failed until this one:

| construction | 2023-26 Sh | 2025-26 Sh | why rejected |
|---|---|---|---|
| GSR static (grid-stress physics) | 1.48 | 1.00 | hand-classified, 0.0x turnover; not a rule |
| Grid-stress-beta L/S | 0.56 | 0.66 | thesis **inverts** — goes long fossil, shorts the firm-capacity names it exists to own |
| Profitability-filter L/S | 0.85 | 0.78 | works, but weaker on every metric |
| **Capex-intensity L/S** | **1.23** | **1.12** | — |

The discovery was not a guess. Eleven point-in-time metrics were screened for their
ability to separate a hand-classified target list from the rest of the pool:

| metric | 2019-22 (fit) | 2023-26 |
|---|---|---|
| **capex intensity** | **t = −15.08** | **t = −15.20** |
| size | −7.20 | 11.27 |
| market beta | 4.89 | 13.66 |
| **momentum 12-1** | **−0.20** | 11.09 |
| **grid-stress beta** | 2.49 | **0.50** |

Capex intensity is the **only** metric that separates strongly in the fit window *and*
holds afterwards, and it wins by more than 2x over the next best. Momentum separates only
*ex post* (−0.20 → 11.09) — that is the outcome, not a predictor. Grid-stress beta, the
metric the original thesis was built on, barely separates and decays to nothing.

## 3. The rule

| step | definition |
|---|---|
| **Universe A** | every US 10-K filer whose SEC **SIC code** is in the electrical/power set (equipment, switchgear, **power-line construction**, utilities, generation, gas transmission) |
| **Universe B** | any filer whose **prior-year** annual report matches the fuel-cycle lexicon — *"uranium concentrates"*, *"nuclear fuel cycle"*, *"uranium enrichment"*, *"yellowcake"*, *"enriched uranium"* |
| **Liquidity** | trailing 1-yr median dollar volume > $10M, re-checked monthly |
| **Score** | point-in-time TTM capex ÷ TTM revenue, from EDGAR `companyfacts` keyed on each fact's `filed` date |
| **Book** | long lightest 25%, short heaviest 25% at 0.5x, monthly rebalance, 20% vol target, 10bp costs |

**Universe B is not decoration.** SIC alone cannot find the nuclear fuel cycle: CCJ and
UEC file under 1090 (Misc Metal Ores), LEU and UUUU under 1400 (Nonmetallic Minerals),
sitting among hundreds of gold miners. What distinguishes them is their own filing
language. Adding the lexicon recovered all four, plus peers nobody picked — DNN, NXE, URG,
EU, ASPI, LTBR, NNE. Separately, **SIC 1623 is literally "Power Line Construction"** and
recovers the electrical contractors.

Deliberately excluded: 3674 (semiconductors — would drag in AMD, AVGO, ARM), 1311 (crude
oil & gas, ~900 filers) and 1040 (metal mining, mostly gold).

## 4. Performance

RF = 4%, 10bp costs, 20% vol target. **2023 onward is out-of-sample for the discovery**,
which used only 2019-2022 to choose the metric.

| window | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| 2019-2022 *(fit)* | 11.0% | 21.1% | 0.41 | −29.1% | 0.38 |
| **2023-2026 *(oos)*** | 31.8% | 21.1% | **1.23** | −22.7% | **1.40** |
| **2025-2026 *(oos)*** | 29.5% | 21.6% | **1.12** | −22.6% | **1.30** |
| full 2019-2026 | 20.6% | 21.1% | 0.80 | −29.1% | 0.71 |

~22 long / 22 short · turnover 1.0x/yr · universe 100 (2019) → 159 (2025) names.
SPY over the same windows: 0.48 / 1.14 / 0.82 / 0.72.

**Current book (2026-08-31).**
*Long* — AEP, AME, BTU, CGNX, CMI, CTRI, ED, ESAB, FELE, FLNC, FTV, GE, IESC, IEX, KAI,
KEYS, MKSI, MTZ, MWA, NVT, ONTO, PNR, POWL, RRX, SPB, SYM, TRMB, UUUU, WTS.
*Short* — AEE, AES, CMS, CNP, CPK, DUK, ENS, EOSE, ES, ETR, EVRG, EXC, IDA, KNTK, LEU,
LNT, NFG, NI, NRGV, NWE, ORA, PCG, POR, PPL, SERV, SLDP, SR, VG, WMB.

Asset-light suppliers long; capital-heavy regulated utilities and cash-burning developers
short. The book reads the way the thesis says it should.


![Growth, time under water and worst drawdown against the cheaper alternatives](img/capex_growth.png)

![Risk against return](img/capex_risk.png)

## Head-to-head: the two windows that matter

Both books and every alternative on one basis — production `compute_metrics`, RF = 4%,
same price data, same run. **Bold marks the best value in each column.**

### 2023-2026

| | CAGR | Vol | **Sharpe** | **MaxDD** | **Calmar** |
|---|---|---|---|---|---|
| **Screened Electrification +short** | 28.3% | 14.3% | 1.54 | **-12.3%** | **2.31** |
| **Screened Electrification (no short)** | 31.0% | 15.5% | **1.56** | -17.0% | 1.82 |
| **Capex-intensity L/S 0.5x** | 31.8% | 21.1% | 1.23 | -22.7% | 1.40 |
| GRID (smart-grid ETF, 0.57%) | 25.4% | 20.4% | 1.02 | -22.2% | 1.15 |
| PAVE (infrastructure ETF, 0.47%) | 26.1% | 20.4% | 1.04 | -27.9% | 0.94 |
| SPY (S&P 500, 0.09%) | 21.2% | 14.8% | 1.11 | -18.8% | 1.13 |
| XLI (industrials, 0.08%) | 20.1% | 16.4% | 0.95 | -20.2% | 0.99 |
| XLU (utilities, 0.08%) | 7.8% | 16.5% | 0.30 | -16.8% | 0.47 |

### 2025-2026

| | CAGR | Vol | **Sharpe** | **MaxDD** | **Calmar** |
|---|---|---|---|---|---|
| **Screened Electrification +short** | 27.6% | 14.7% | **1.46** | **-11.9%** | **2.32** |
| **Screened Electrification (no short)** | 26.7% | 16.2% | 1.30 | -17.0% | 1.57 |
| **Capex-intensity L/S 0.5x** | 29.5% | 21.6% | 1.12 | -22.6% | 1.30 |
| GRID (smart-grid ETF, 0.57%) | 24.8% | 23.6% | 0.89 | -21.1% | 1.17 |
| PAVE (infrastructure ETF, 0.47%) | 15.2% | 22.2% | 0.57 | -23.7% | 0.64 |
| SPY (S&P 500, 0.09%) | 17.4% | 16.9% | 0.80 | -18.8% | 0.93 |
| XLI (industrials, 0.08%) | 14.3% | 18.8% | 0.59 | -18.2% | 0.79 |
| XLU (utilities, 0.08%) | 6.5% | 16.2% | 0.22 | -12.6% | 0.51 |

**Reading it.** The shipped `+short` book takes Sharpe and Calmar in both windows, and it
does so at roughly **two-thirds the volatility** of the capex-intensity book and **half
the drawdown**. Capex-intensity delivers the highest raw CAGR in 2023-2026 (31.8%) and
2025-2026 (29.5%), but pays for it in vol (21%) and depth (−23%).

Against the cheap alternatives: **both books beat GRID, PAVE, SPY, XLI and XLU on Sharpe
and Calmar in both windows** — so this is not a case where the ETF quietly wins. XLU, the
naive "buy power" trade, is the worst thing on the page in both windows, which is the
thesis stated as a negative control.

## 5. Robustness

| objection | test | result |
|---|---|---|
| "It's just short utilities" | drop **all 44 utilities** from the pool | 2023-26 **1.10**, 2025-26 **1.05** — survives |
| "The short leg is decoration" | short weight 0 → 0.25 → 0.5 → 1.0 | 1.07 → 1.20 → **1.26** → 1.15; monotone to 0.5x |
| Transaction costs | 5 / 10 / 25 / **50** bp | 1.27 / 1.26 / 1.22 / **1.14** |
| Quantile choice | q = 0.15 / 0.20 / 0.25 / 0.33 | 1.29 / 1.21 / 1.26 / 1.14 |

**This is the first construction in this line where an algorithmic short earns its place.**
The earlier books needed their shorts shrunk to 0.15-0.25x to avoid damage — shorting
low-quality names during a thematic bull market lost money, because the junk basket
*outperformed* (+47.8% vs +37.4% CAGR in 2023-26). Here the heavy-capex basket is a
genuine underperformer, so the short adds monotonically out to half size.

## 6. Risks, and what would falsify it

1. **It is a capex-cycle bet.** Fit-window Sharpe is 0.41. It pays while the buildout
   runs. **Kill condition:** if the spread between light and heavy capex-intensity baskets
   goes flat for four consecutive quarters, the cycle has turned and the book should be
   stood down — do not wait for a drawdown to decide.
2. **Metric selection used ex-post labels.** Eleven metrics were screened against a
   hand-classified list. Once stated the rule needs no labels, and the backtest never
   references them — but *which* metric was chosen carries selection risk that only
   forward out-of-sample resolves. **Date-stamp the rule now.**
3. **Asset-light is a known factor.** This overlaps documented quality/capital-discipline
   factors. The claim is a correctly specified factor for this universe with a matching
   economic rationale — not a new anomaly. **Test it against a generic asset-light factor
   outside this universe** before sizing.
4. **Borrow.** Regulated utilities are easy to short; EOSE, SLDP, SERV and NRGV are not.
   Assume the short leg is partially unfillable and check the book still works.
5. **Survivorship — unfixed, and it affects the shipped book equally.** 2,168 of 2,535
   filers in these SIC codes have no ticker today, and yfinance serves no delisted prices.
   Long legs are flattered; short legs are understated (bankruptcies would have been
   profitable shorts). Every absolute level here should be read as optimistic until a
   delisted-inclusive price source exists.

## 7. What shipping would involve

`capex_intensity_strategy/` is complete and tested (18 tests, offline against synthetic
filings). Pre-live gates, in order:

1. Date-stamp and freeze the rule; begin forward OOS tracking.
2. Source delisted-inclusive prices (CRSP or equivalent) and re-run — this is the single
   largest unknown.
3. Generic-factor control: does asset-light work as well outside the power universe?
4. Borrow and capacity study on the short leg.
5. Decide position sizing against the shipped book — correlation is not yet measured, and
   both books are long the same theme.
