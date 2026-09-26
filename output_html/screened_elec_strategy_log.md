# Screened Electrification: Rule Book

This is the detailed companion to *How the portfolio is built* in the investor write-up. The write-up
explains each rule in plain terms. This rule book gives the exact thresholds, timing and data source
behind each one, so the portfolio can be rebuilt from public data.

All figures in the write-up (including the Sharpe ratio of 1.85 from 2023 on, after 10bp trading costs)
come from the rules exactly as written here.

---

## Long Book: Screen

The screen is re-run every month. Each rule uses only information that was public on that date.

### Supplier Test

| Rule | Detail | Source |
|---|---|---|
| Capital spending is low relative to sales | Trailing-12-month capital spending ÷ trailing-12-month revenue. Companies are ranked on this ratio among the liquid candidates, and the **lightest 60%** pass. | SEC EDGAR XBRL filings: capital spending (purchases of property, plant and equipment) and revenue |
| The company describes the buildout in its own words | At least one annual report (10-K) filed in the **previous four years** uses one of these exact phrases: *data center(s), substation, switchgear, transmission and distribution, grid modernization, electric utilities*. | SEC EDGAR full-text search of 10-K filings |

### Profitability

| Rule | Detail | Source |
|---|---|---|
| Earnings | Trailing-12-month net income above zero, using only figures already filed with the SEC on that date. | SEC EDGAR XBRL filings (net income) |
| Not subsidy-dependent | A company is **excluded** if any 10-K filed in the previous four years uses: *investment tax credit, production tax credit, advanced manufacturing production credit* or *net energy metering*. One mention in one filing is enough to exclude it. | SEC EDGAR full-text search of 10-K filings |

### Liquidity

| Rule | Detail | Source |
|---|---|---|
| Trading volume | Median daily value traded of at least **$10 million** over the past year (252 trading days, at least 120 of them with data), checked monthly. | Daily prices and volumes (Yahoo Finance) |
| Price history | At least **400 trading days** of price history to be considered, and at least **252 trading days** since listing to be held. | Daily prices (Yahoo Finance) |

**Point-in-time data.** Every accounting figure becomes usable only from the day it was filed with the
SEC. If a company later files a corrected figure for a past quarter, the correction is used only from
its own filing date.

---

## Long Book: Rank

| Rule | Detail | Source |
|---|---|---|
| Capital intensity | Trailing-12-month capital spending ÷ trailing-12-month revenue. Lower is better. | SEC EDGAR XBRL filings |
| Revenue acceleration | Year-on-year growth in trailing-12-month revenue this quarter, minus the same figure last quarter. Higher is better. Revenue is taken as filed with the SEC at the time. | SEC EDGAR XBRL filings |
| Combining the two | Each month both measures are scored across the eligible companies (extreme values are trimmed at the 5th and 95th percentiles, then each measure is standardised), and the two scores are averaged. A company missing a measure scores zero on it. | n/a |
| Who enters | At each **quarterly** rebalance, new names come from the **top half** of the companies that pass Stage 1. | n/a |
| Who stays (buffer) | A name already held stays as long as it remains in the **top 65%**. This stops names near the cut-off from being bought and sold repeatedly. | n/a |

These two measures were chosen from fourteen candidates tested. They were the only two with a
consistent relationship to future returns. The combination was fixed on 2026-09-24 and has not been
tuned since.

The write-up's *Ranking alone* figure (in Return Attribution) compares the top half against every
qualifying name at equal weight. That test holds the top half outright, without the 65% buffer.

---

## Long Book: Weight

| Rule | Detail |
|---|---|
| Equal weight | Each selected name gets the same weight at each rebalance. Weights then drift with prices until the next rebalance. |
| Single-name cap | No name above **25%** of the long book. |

---

## Long Book: Risk Control

| Rule | Detail |
|---|---|
| Volatility target | Exposure is set daily to target 20% annual volatility, measured over the previous month (21 trading days) up to *yesterday*, and never exceeds 1.5×. |
| No-trade band | The book is resized only when the target moves more than **20 points** of exposure from where the book stands (for example from 80% to below 60% or above 100%). Large moves, such as a volatility spike, still go straight through. |
| Stretch de-lever | Exposure steps down to 80% when the book is more than 15% above its 200-day average or far ahead of the market over the past year, and to 60% at the extremes. Uses the *previous* month's reading. |
| Gold / Treasury sleeve | 15% in GLD and IEF, reset to 15% monthly. |

---

## Short Book

| Rule | Detail | Source |
|---|---|---|
| Universe | Companies in the SEC's electrical-equipment and power industry codes, plus nuclear fuel-cycle names found by keyword search of filings, trading at least $10m a day (the same liquidity rule as the long book). About 200 names. | SEC EDGAR (industry codes, full-text search); Yahoo Finance |
| On-theme filter | Only companies whose 10-Ks used the long book's buildout wording in the previous four years can be shorted. This removes banks, insurers and a gold miner that reached the universe through keywords, and industrials with no real grid business. | SEC EDGAR full-text search of 10-Ks |
| Not held long | Any name the long book holds is excluded. | n/a |
| Score | Fourteen signals, each trimmed at the 5th and 95th percentiles, standardised across the universe and averaged with equal weight, at each month-end from information available that day. **Filings (6):** capital spending ÷ revenue, revenue acceleration, trailing-12-month revenue growth, revenue surprise, earnings surprise, growth in capital spending. Surprises are measured against the company's own history (the same quarter a year earlier, plus drift), not analyst estimates. **Prices (2):** 12-month momentum skipping the latest month, 1-month reversal. **Power markets (6):** each stock's rolling 36-month sensitivity to shocks in wholesale energy prices, congestion and scarcity pricing (controlling for the S&P 500), and that sensitivity multiplied by the latest 3-month shock. | SEC EDGAR XBRL filings; Yahoo Finance; PJM and ERCOT day-ahead prices (energy: PJM hubs and ERCOT hub average; congestion: PJM zones; scarcity: 99th-percentile hourly ERCOT hub price) |
| Basket | The bottom fifth by that score, equally weighted, re-formed at each quarter-end. 16 names on average, never fewer than 12. | n/a |
| Size | $0.25 short per $1 of long book, reset to $0.25 at each month-end; weights drift with prices in between. | n/a |
| Costs | 10bp on every change in the basket or its size; borrow of 1% a year on the short notional, charged daily. | n/a |

---

## Step-Aside Rule

| Rule | Detail | Source |
|---|---|---|
| Real yield | 10-year inflation-protected Treasury (TIPS) yield above zero, using the *previous* day's reading. | FRED series DFII10 |
| Real supplier growth | Median trailing-12-month revenue growth, as filed and less CPI inflation, of the names that pass the long-book screen, above zero. Uses the latest filed revenue and the latest published CPI. | SEC EDGAR XBRL filings; CPI (FRED series CPIAUCSL, used from the month after publication) |
| Switch | When either turns negative, the whole portfolio (long, short and sleeve) moves into 3-month T-bills the next day, and returns when both are positive again. Each switch trades about 1.25× capital at 10bp. | FRED series DTB3 (T-bill return) |

---

## No Look-Ahead

Every decision in the backtest uses only information that existed on the day it was made:

- industry codes as filed;
- 10-K wording only from filings published *before* the year being decided;
- earnings, capital spending and revenue only from the date each figure was filed with the SEC;
- a ranking or score from the latest month-end on or before the trade;
- position size from volatility up to *yesterday*;
- the stretch cut from the *previous* month's reading;
- the step-aside rule from the previous day's real yield and the latest filed revenue and published CPI.

What a backtest cannot remove is **research hindsight**. The long ranking, the short score, the on-theme
filter on the short, the $0.25 size and the step-aside rule were all chosen in September 2026 after
comparing alternatives on this same history, so the backtest flatters them to some degree.

---

## Decision Log

| Date | Decision |
|---|---|
| 2026-09-24 | Selection ranking fixed: capital intensity + revenue acceleration, equally weighted. |
| 2026-09-25 | Subsidy exclusion list confirmed as four terms, including the advanced manufacturing production credit. |
| 2026-09-25 | Point-in-time handling as described above (corrections used from their own filing date) accepted. |
| 2026-09-26 | Investor write-up's *Portfolio Construction* section rebuilt as stage tables (screen, rank, weight, risk control) for both books; numbered rules dropped (rule 11 is now the step-aside rule); the no-look-ahead section moved here from the write-up. |

---

## INTERNAL: remove before sharing

**Build that produced the published numbers:**
`research/electrification_strategy_probes/build_tophalf_proposal.py` → `investor_book.build()`, on the
`screen_auto` A1+A4 membership from `build_a1a4_proposal.py`.

**Overrides of `config.py` defaults.** The A1+A4 build sets `LOOKBACK, QUANTILE = 4, 0.60`
(`build_a1a4_proposal.py:43`), but `electrification_strategy/config.py` still has `AUTO_LOOKBACK_YEARS = 2`
and `AUTO_CAPEX_QUANTILE = 0.5`. The package CLI with its defaults does **not** reproduce the published book.

**Where each rule lives in the code:**
- Capex-light cut: `auto_screen.light_capex_mask`. XBRL tags: `PaymentsToAcquirePropertyPlantAndEquipment`, then `PaymentsToAcquireProductiveAssets`; revenue from `Revenues` / ASC-606 tags (`capex_intensity_strategy/edgar.py`).
- Vocabulary: `config.AUTO_CUSTOMER_TERMS`, `config.AUTO_POLICY_TERMS`, `auto_screen.vocabulary_years`. Hits are cached in `electrification_strategy/data/vocabulary_hits.csv`.
- Liquidity: `capex_intensity_strategy/config.py` (`LIQUIDITY_*`, `MIN_PRICE_HISTORY_DAYS`), `electrification_strategy/config.LISTING_MIN_DAYS`.
- Point-in-time: `capex_intensity_strategy/fundamentals.py:75` uses the latest filing as of each date.
- Composite: `research/xsec_alpha_probe/tilt.composite`. Signals are in `research/xsec_alpha_probe/signals.py`.
- Selection, buffer, band, short, rule 11: `research/electrification_strategy_probes/investor_book.py` (`LONG_HOLD = 0.65`, `BAND = 0.20`, `SHORT_FREQ = "Q"`).
- Cap: `config.MAX_SINGLE_NAME_WEIGHT = 0.25`.

**Open: short-size calibration (2026-09-26).** The $0.25-per-$1 short size is not yet explained in the write-up. Plan: test a range of fixed sizes (and beta-matched) since 2023 and on the full period, then add a rule-of-thumb line ("$4 long, $1 short: a quarter of the shorted stocks' move comes off the return").

**Resolved write-up vs code discrepancies (2026-09-26):** the write-up's old rule 6 said "ranks are averaged" (the code averages
winsorized z-scores) and old rule 3 said "as first published, never as later restated" (the code uses the latest filing
as of each date). The new stage tables say "scores" and "figures already filed at the time", which match the code.

**Wording note:** the write-up used to call the sixth filing signal "asset growth". The code measures growth in trailing-12-month
capital spending (`capex_restraint`), so the tables now say "growth in capital spending".

**Short-side code:** `investor_book.short_basket`, 14-signal composite `research/xsec_alpha_probe/short_compare.all14_short`
(`composites.py`, `signals.py`, `power.py`). Borrow `NAME_BORROW = 0.01`, size `FIXED_K = 0.25`, `SHORT_FREQ = "Q"`.
Step-aside: `RATE_SIGNAL = "DFII10"`, `CASH_SERIES = "DTB3"`, `supplier_growth()` in `build_tophalf_proposal.py`.

**Stats behind the write-up's control tables (recomputed 2026-09-26, exact match to published figures):** long-book stages from
`build_tophalf_proposal.build_book` (`raw`, `vol_only`, `core`, `book`), before costs; short/step-aside stages from `investor_book.build`
with `short=None` + real yield clipped high (long only, always on), default short with the real yield clipped (plus short), and the
full strategy, all at 10bp.

**Style-factor check (write-up §Style Factors, run 2026-09-26):** OLS of daily strategy returns (`investor_book.build`, full
strategy at 10bp) on factors, 2023-01-01 to 2026-08-31 (918 days). Academic runs use excess returns over the Fama-French RF.
ETF styles are Yahoo Finance adjusted closes. t-statistics are plain OLS; Newey-West (5 lags) gives nearly identical values.
Scripts were not saved in the repo. Logged in `docs/RESEARCH-LOG.md` §3.
