<!--
  Turn this file into HTML / PDF with tools/md2html.py (from the repo root):

    poetry run python tools/md2html.py docs/electrification-strategy-proposal-v5.md --theme light --pdf

  Options:  --theme auto | dark      --no-toggle      --toc      -o PATH
  Figures (docs/pv3-*.png) are inlined, so the HTML and PDF are self-contained.
-->

# Screened Electrification Equipment

## Summary

Building and upgrading the electricity network is a long-running
trend that is likely to continue for years as it rests on physical need  (to power AI data centres, to electrify heating, supply electric vehicles, and to bring factories back to the United States) rather than investor enthusiasm. 

This strategy buys shares in the profitable,
well-run companies that make the equipment and do the construction work for that build-out.
It follows a fixed set of rules rather than a manager's judgement, and it has produced a
better balance of return and risk than simply buying an index fund for the same trend.

There are two versions:

- **Option A — buy-only.** The portfolio only buys shares; it never places bets that a
  price will fall (and pays no borrowing cost). It still moves a fair amount with the
  overall stock market: (about 0.6 as much as the market) but this is far less than the VOLT electrification index fund which has about 1.0 beta to market.
- **Option B — buy the suppliers, bet against general industry.** For a lower market beta, on top of Option A, add
  a position that profits if a broad basket of US industrial companies falls. This is a
  bet that the chosen electrical-equipment suppliers do better than industry as a whole.
  It cuts how much the portfolio moves with the market roughly in half — to about 0.37 —
  but it adds the fees and risks of betting against shares.

---

## What the strategy buys

In this small test case, the portfolio holds roughly 27 companies that sell into the electricity build-out:
makers of power equipment (transformers, switchgear, the electronics that control power
flow), companies that build large batteries for the grid, and electrical engineering and
construction firms. It holds suppliers — not the utilities that run the wires, and not
the companies that own power plants. 

A company is included only if:

1. it sells mainly to large institutional buyers — utilities, cloud-computing companies,
   industrial customers — not to consumers; and
2. it meets at least two of these three tests:
    - it makes a profit today,
    - its share price is set by its earnings (a reasonable P/E ratio) rather than by a story about the future, and
    - it does not depend heavily on government subsidies.


For the simple example, every holding gets the same weight, capped at 25% each. 

---

## Built-in protections

Three rules adjust the portfolio automatically. Each one addresses a specific risk.

**Growth Scare, Rates Shock, Currency Shock, Geopolitical Risk** A 15% reserve is held in gold and short-term US government bonds.
If there is a scare about economic growth or company debt, the bond part of the sleeve will rise providing some protection. If
there is a currency or geopolitical shock, the gold rises. Either way, that portion holds
its value while shares fall.

**Prices get too far ahead of profits.**
If the holdings have risen well above their own long-term trend (prices too far ahead of profits) and well above the wider
market, the strategy sells some of them in steps — down to about 80% invested, then about
60%. This takes money off the table when prices look stretched, so the portfolio is not
fully exposed at a peak.

**Choppy Markets**
When day-to-day price swings grow, the strategy moves part of the money to cash; when
markets calm down, it moves back in. The aim is to keep the yearly size of the ups and
downs near 20%.

---

## How it compares to the VOLT electrification index fund

The strategy covers the same trend as VOLT, a listed index fund for electrification. It is
different in that it **leaves out the parts of VOLT that hold the return back**: regulated
electric utilities (about 30% of VOLT), oil and gas pipelines (about 10%), and basic
components (about 13%). Those businesses are sensitive to interest rates, grow slowly, and
are only loosely connected to the equipment build-out.

Over VOLT's live history (January 2025 to August 2026) the strategy delivered a **similar
total gain** (about +30%) but with a **better Sharpe** (1.49 versus 1.05), a
**smaller worst fall** (−17% versus −23%), and **less movement with the market** (about
0.6 beta versus about 1.0).

---

## Performance

![Growth of $1 from January 2025 to August 2026 for the strategy, VOLT, GRID and the S&P 500](pv3-growth-fair.png)

*January 2025 to August 2026. The strategy keeps pace with the electrification index funds
but with much smaller swings — it does not jump to VOLT's 1.8-times level in mid-2026, and
it does not give that back afterwards.*

![Growth of $1 from January 2019 to August 2026 for the strategy, PAVE and the S&P 500](pv3-growth-full-linear.png)

*January 2019 to August 2026. The strategy lagged from 2019 to 2022.  But since the electrification phase of the economy began, the strategy has outperformed to finish level with the infrastructure index fund and ahead of
the overall market — with smaller swings and a smaller drawdown than either.*

![The fall from each prior high point, for the strategy, VOLT and the S&P 500](pv3-drawdown.png)

*The fall from each prior high point. In the 2020 COVID crash the strategy fell 22% while
the market fell 34%. In the 2022 interest-rate shock it fell 20% versus the market's 24%.
Its worst point in the whole test period is −23%.*

![The strategy's Sharpe over rolling 12-month windows, versus the S&P 500](pv3-rollsharpe.png)

*Sharpe ratio over each rolling 12-month window — same 3.8% risk-free rate as the tables.
Both the strategy and the
market went negative in 2022; the strategy is ahead of the market for most of the record.*

### Return and risk by period

Four summary measures, each for three time spans. Sharpe (with risk free at avg 3m Tsy over the test period), Max Drawdown, Compound Annual Growth Rate, and market sensitivity measured as beta to S&P500.

*Full history · January 2019 – August 2026*

| Strategy | Sharpe | Max DD | CAGR | S&P Beta |
|---|--:|--:|--:|--:|
| **Option A** — buy-only | **1.05** | −23% | +20.8% | 0.61 |
| **Option B** — with the bet against industry | 1.02 | **−16%** | +16.6% | **0.37** |
| VOLT electrification index fund | — | — | — | — |
| VOLT + 15% gold/bond reserve | — | — | — | — |
| VOLT + 15% reserve + vol-target | — | — | — | — |
| US stock market (S&P 500) | 0.71 | −34% | +17.5% | 1.00 |
| S&P 500 + the 15% gold/bond reserve | 0.75 | −29% | +15.9% | 0.86 |

*AI-boom period · January 2023 – August 2026*

| Strategy | Sharpe | Max DD | CAGR | S&P Beta |
|---|--:|--:|--:|--:|
| **Option A** — buy-only | **1.80** | −17% | +31.5% | 0.73 |
| **Option B** — with the bet against industry | **1.80** | **−13%** | +26.3% | **0.50** |
| VOLT electrification index fund | — | — | — | — |
| VOLT + 15% gold/bond reserve | — | — | — | — |
| VOLT + 15% reserve + vol-target | — | — | — | — |
| US stock market (S&P 500) | 1.25 | −19% | +22.6% | 1.00 |
| S&P 500 + the 15% gold/bond reserve | 1.32 | −16% | +20.7% | 0.87 |

*VOLT's live history · January 2025 – August 2026*

| Strategy | Sharpe | Max DD | CAGR | S&P Beta |
|---|--:|--:|--:|--:|
| **Option A** — buy-only | **1.49** | −17% | +27.7% | 0.65 |
| **Option B** — with the bet against industry | 1.43 | **−13%** | +22.1% | **0.43** |
| VOLT electrification index fund | 1.05 | −23% | +30.5% | 0.99 |
| VOLT + 15% gold/bond reserve | 1.12 | −19% | +28.4% | 0.86 |
| VOLT + 15% reserve + vol-target † | 0.33 | −9% | +7.5% | 0.37 |
| US stock market (S&P 500) | 0.88 | −19% | +18.9% | 1.00 |
| S&P 500 + the 15% gold/bond reserve | 1.01 | −16% | +18.8% | 0.87 |

*VOLT began trading in December 2024, so it has no figures for the two earlier spans.
† The vol-target on VOLT is computed by looking, each day, at VOLT's price swing over prior 20 days, annualizing and setting invested faction as 20% divided by that number, capped at 100%.  The uninvested rest earns the cash rate.  Then belnd 85% of that with the 15% gold/bond reserve. (A 90d lookback for vol is worse). Note that, 
even the best VOLT version "VOLT +
reserve" row (1.12) stays well below Option A (1.49) over the same window — the gap is the
choice of companies, not the wrappers.*

- Option A has the best Sharpe in every span, and a smaller worst fall than the
  market.
- Option B matches Option A's Sharpe during the AI-boom period while cutting the
  worst fall to −13% to −16% and cutting market sensitivity to about 0.4.
- Adding the gold/bond reserve to the plain market index or to VOLT (bottom row) gives a smoother
  ride but nowhere near the strategy's Sharpe — the difference of performance in our strategy relative to the market comes from the
  choice of companies, not from the reserve sleeve.

### How it held up in specific market shocks

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

---

## What the strategy is betting on, and how each risk is handled

| Risk | How covered | What it means, and what the strategy does |
|---|---|---|
| Investors mark down the whole electrification group | Partly | AI-data-centre spending slows, data-centre bans spread, or grid spending flattens. As prices fall and swings widen, the two cutback rules move money into cash, so the strategy grows more defensive the longer it lasts. |
| Interest rates fall and beaten-down clean-energy shares jump | Lag only | The strategy still makes money — the bond half of the reserve gains and the suppliers rise with the recovery. It may just trail the fastest-bouncing solar names for a while. We do not bet against those names. |
| A broad stock-market fall | Reduced | The portfolio moves about 0.6-for-1 with the market. The reserve and the choppy-market rule cut the loss — down 22% in the 2020 COVID crash versus the market's 34%. |
| Interest rates spike quickly | Mixed | It hurts twice: these are still more than average growth so share prices for our sector will likely fall and the bond part of the reserve also falls. The vol-targeting rule helps. |
| The US repeals clean-energy tax credits | Reduced | The "not dependent on subsidies" test keeps out the companies that would be hit hardest. |
| Too much riding on one company | Limited | 27 companies, none over 25%; equal weight caps any single failure near 4–8% of the portfolio. Deliberately concentrated, not a 200-stock fund. |
| Rising costs of copper, electrical steel and labour | Not a real risk | Tested directly: the portfolio barely moves with any of the three. These suppliers raised their own prices to cover it, and the portfolio did better than usual when those costs rose most. |

---

## Method notes

- **Test period.** The full test runs June 2017 to August 2026; all figures use January
  2019 onward. This is only one economic cycle but it is the period during which this electrification theme is relevant. 
- **Sharpe** = (yearly return − 3.8% risk-free cash rate) ÷ the yearly size of the ups and
  downs. The 3.8% is roughly the average US 3-month Treasury-bill rate over the test period,
  and is used the same way in the tables and in the rolling-window chart.
- **How often it trades** Company holdings change about monthly so share turnover is low.   The cash level and the gold/bond mix are recalculated daily, and both move slowly.  Trading costs at scale are still to be estimated.
- **Prices** are adjusted for dividends and splits. VOLT began trading  in December 2024.

