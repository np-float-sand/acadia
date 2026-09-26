# Screened Electrification Equipment

> **In brief.** Electricity demand is growing faster than the grid can carry it. Regulated utilities
> pay for the new wires, transformers and substations; the equipment makers and contractors that supply
> them are paid as the work is delivered. The strategy holds the most capital-light, fastest-accelerating
> of those suppliers and shorts the weakest fifth of electrification companies, today mostly the
> utilities and pipelines doing the paying.
>
> The grid stress that took hold in 2023 looks durable. Utilities have committed to multi-year capital
> plans, equipment lead times run to years, and real yields remain elevated. The suppliers' own
> filings show no slowdown: their median revenue growth is 14% a year and still accelerating, the
> fastest since early 2023. Over that period the long/short has delivered a Sharpe ratio of 1.77 with
> a beta of 0.50 to the S&P 500.

*Backtest window: January 2019 – August 2026 · Trading and borrow costs included where stated ·
Management fee not assumed*

---

## Performance at a glance

![Sharpe ratio of the strategy, GRID, PAVE, the S&P 500 and VOLT](img/tophalf/inv_sharpe.png)

*Sharpe ratio — return earned per unit of risk. Bars: the full period (light) and since the grid
became tight in 2023 (dark). Lines: Sharpe over the trailing 12 months. The shading in the lower panel
marks the market environment behind each stretch: the strategy lags when money is free and the market
rewards subsidy stories, and leads when real yields are positive and the grid is tight. The dashed
amber line (right axis) is the median revenue growth of the suppliers that pass the long book's rules,
read from their filings as published — a proxy for the buildout, in the suppliers' own numbers.
\*VOLT (an electrification ETF) launched in December 2024; its bar and line cover only its own
history.*

### Since the grid became tight (January 2023 – August 2026)

| | **Long/short (the strategy)** | GRID ETF | PAVE ETF | S&P 500 | VOLT ETF* |
|---|---|---|---|---|---|
| **Sharpe ratio** | **1.77** | 0.93 | 0.91 | 1.16 | 0.78 |
| **Calmar ratio** | **2.63** | 1.11 | 0.87 | 1.20 | 0.97 |
| **Max drawdown** | **−11.5%** | −20.8% | −26.2% | −18.8% | −23.4% |
| **Volatility** | **13.2%** | 20.3% | 20.6% | 15.0% | 25.5% |
| **Beta to the S&P 500** | **0.50** | 1.10 | 1.09 | 1.00 | 1.00 |
| **Return a year** | **30.4%** | 23.0% | 22.8% | 22.6% | 22.8% |
| **Max time under water** | 5.3 months | 7.6 months | 7.9 months | **4.2 months** | 5.3 months |
| Return / Sharpe after trading costs | 29.4% / 1.71 | *fees already in price* | | | |

*\*VOLT: December 2024 – August 2026 only (21 months). Over that same window the strategy's Sharpe
was 1.25, its max drawdown −11.5% and its return 20.9% a year.*

### Before the grid was tight (January 2019 – December 2022)

| | **Long/short (the strategy)** | GRID ETF | PAVE ETF | S&P 500 |
|---|---|---|---|---|
| **Sharpe ratio** | 0.40 | **0.77** | 0.61 | 0.48 |
| **Calmar ratio** | 0.52 | **0.58** | 0.44 | 0.39 |
| **Max drawdown** | **−17.3%** | −40.6% | −44.1% | −33.7% |
| **Volatility** | **13.9%** | 27.3% | 29.7% | 22.5% |
| **Beta to the S&P 500** | **0.28** | 1.06 | 1.13 | 1.00 |
| **Return a year** | 8.9% | **23.6%** | 19.2% | 13.1% |
| **Max time under water** | 16.8 months | 13.3+ months | 11.8+ months | 11.9+ months |
| Return / Sharpe after trading costs | 8.1% / 0.34 | *fees already in price* | | |

**This is a grid-tightness strategy.** It does not work well when the grid has ample spare capacity:
in 2019–22 the suppliers' median revenue growth fell to −6% a year, and the strategy trailed the theme
ETFs. There is no exit signal, but the strategy de-risks on its own: position size falls automatically
as volatility rises or prices run far above trend (to a quarter of full size in March 2020 and a third
in April 2025), and the ranking rotates toward suppliers whose sales are still growing.

*Sharpe uses a flat **4.0% risk-free rate**. Returns are total returns, dividends reinvested. The
strategy is net of 1% a year borrow on the short and 10bp on its re-formation and resizing; the "after
trading costs" row also charges 10bp on the long side's trading ([Costs](#what-it-costs-to-run)). Max time under water is the
longest stretch below a previous peak within the window; "+" means the drawdown was still unrecovered
at the end of 2022.*

**Since 2023 the strategy leads every benchmark on every measure but one** — the S&P 500 recovered
from the 2025 tariff selloff about a month faster. **Before 2023 it did not:** it had by far the
smallest drawdown and volatility at a quarter of the market's beta, but a lower Sharpe and return than
the theme ETFs, and its longest spell under water (December 2019 – May 2021) came as a clean-energy
rally lifted exactly the stocks the rules exclude.

![Growth of $1](img/tophalf/inv_growth.png)

*VOLT's line starts on the strategy's line at VOLT's launch, so the two can be compared from
that date. The dashed amber line (right axis) is the suppliers' median revenue growth — the buildout proxy — as above. Over the whole period GRID and PAVE, carrying the full theme at 1.1× the market's beta,
compounded faster than the strategy; since 2023 the strategy compounded fastest.*

---

## How the portfolio is built

The gap in capital intensity between the utilities and their suppliers is a fact in the accounts
rather than a judgment call, which is what lets the whole portfolio be built by rule — nine rules for
the long book and one for the short side.

### The long book: which companies qualify

**1. Suppliers to the grid, not owners of it.** We keep only companies whose capital spending is low
relative to their sales. This separates a contractor from a utility without anyone deciding which is
which.

**2. The company says so in its own filings.** A firm qualifies only if its annual report (10-K) used
the words *data center, substation, switchgear, transmission and distribution, grid modernization* or
*electric utilities* in the previous four years. A company's own legally binding disclosure is more
precise than a sector label.

**3. Profitable, on the numbers known at the time.** Requiring real reported earnings screens out
story stocks promising grid revenue that never arrives. We use earnings *as first published*, never as
later restated.

**4. Not dependent on subsidies.** If a firm's filings lean on the investment tax credit, production
tax credit or net metering, an investor is buying an election result as well as electricity demand.

**5. Liquid enough to trade.** At least $10m of stock traded on a typical day over the past year, and a
full year of price history.

### The long book: which of them we hold

**6. The better half, on two facts from the accounts.**

- **How little the company spends on capital, relative to its sales.** Within an already
  capital-light list, the lightest keep doing better: they turn the buildout into cash now.
- **Whether its revenue growth is speeding up.** This quarter's year-on-year growth in
  trailing-twelve-month revenue minus last quarter's, on revenue as filed.

Each month both facts are ranked across the eligible electrification universe and the two ranks are
averaged. At each quarterly rebalance the book holds the top half of the names that pass rules 1–5,
equally weighted, with no name above 25%. We tested fourteen candidate ranking signals — earnings and
revenue surprises, momentum, reversal, and sensitivity to wholesale power prices, congestion and
scarcity pricing — and these two were the only ones with a consistent relationship to future returns.

### The long book: how we control risk

**7. We cut size when the book gets volatile.** Exposure is scaled each day to target 20% annual
volatility, measured over the previous month, and never exceeds 1.5×.

**8. We cut size again when the book gets stretched.** When it has run more than 15% above its 200-day
average, or far ahead of the market over the past year, exposure steps down to 80%, and to 60% at the
extremes.

**9. We hold 15% in gold and Treasuries** (GLD and IEF), uncorrelated with the rest of the book when
that matters most.

![What each layer of risk control does](img/tophalf/overlay.png)

| stage | Sharpe | return a year | worst drawdown |
|---|---|---|---|
| Selected names, no risk control | 1.00 | 33.6% | **−47.3%** |
| + volatility target | 1.05 | 27.5% | −28.8% |
| + stretch de-lever | 1.04 | 25.2% | −28.8% |
| **+ gold/Treasury sleeve (the long book)** | **1.08** | **23.1%** | **−24.1%** |

### The short side

**10. We short $0.25 for every $1 in the long book, spread across the weakest fifth of the
electrification companies that describe the buildout in their own filings.**

- **Universe.** The same electrification population the long book is drawn from — companies in the
  SEC's electrical-equipment and power industry codes, plus nuclear fuel-cycle names found by keyword —
  that trade at least $10m a day. About 200 names.
- **Filter.** Only companies that pass rule 2 — the buildout wording in their own 10-K — can be
  shorted. This keeps the short on-theme: it removes banks, insurers and a gold miner that reached the
  universe through keywords, and industrials with no real grid business.
- **Score.** Each month-end, fourteen signals from information available that day, trimmed,
  standardised and averaged with equal weight: from filings, capital spending relative to sales,
  revenue acceleration, revenue growth, revenue and earnings surprises and asset growth; from prices,
  12-month momentum and 1-month reversal; and each stock's sensitivity to wholesale power prices,
  congestion and scarcity pricing.
- **Basket.** The bottom fifth by that score, equally weighted, re-formed monthly: 19 names on average,
  never fewer than 13, 27 today.
- **Size.** A fixed $0.25 short per $1 of long book.

### No look-ahead

Every decision in the backtest uses only information that existed on the day it was made: industry
codes as filed; 10-K wording only from filings published *before* the year being decided; earnings,
capital spending and revenue as first reported; a ranking or score from the latest month-end on or
before the trade; position size from volatility up to *yesterday*; the stretch cut from the *previous*
month's reading.

What a backtest cannot remove is **research hindsight**: the long ranking, the short score, the rule-2
filter on the short and the $0.25 size were all chosen in September 2026 after comparing alternatives
on this same history, so the backtest flatters them to some degree.

---

## Performance

### Year by year

![Calendar-year returns](img/tophalf/inv_years.png)

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|---|
| **Long/short** | 45.7% | −5.3% | 7.4% | −5.0% | 38.0% | 31.7% | 28.0% | 12.9% |
| GRID ETF | 42.2% | 48.8% | 27.7% | −13.9% | 21.6% | 15.2% | 29.7% | 17.1% |
| PAVE ETF | 32.6% | 19.7% | 36.4% | −7.2% | 31.0% | 17.9% | 19.4% | 14.7% |
| S&P 500 | 31.1% | 18.3% | 28.7% | −18.2% | 26.2% | 24.9% | 17.7% | 13.1% |
| Long/short Sharpe | 2.25 | −0.67 | 0.30 | −0.61 | 2.20 | 1.75 | 1.59 | 1.37 |
| Long/short worst drawdown | −6.7% | −13.9% | −9.7% | −11.9% | −8.6% | −6.8% | −11.5% | −5.7% |

The strategy beat GRID in 2022–2024 and trailed it in 2025 and 2026, when the theme itself rallied
hard. 2020 and 2021 were weak: few names, and a market that rewarded the clean-energy and subsidy
stocks the rules exclude.

### Drawdowns

![Drawdown](img/tophalf/inv_drawdown.png)

---

## Where the return comes from

A regression of each book's daily return on GRID's (or the S&P's): whatever is left after accounting
for exposure to the theme is alpha. A t-statistic above 2 means the result is unlikely to be luck.

| alpha a year (t-statistic) | full period | 2019–22 | **2023 on** |
|---|---|---|---|
| **Long/short vs GRID** | **10.7% (2.6)** | 3.4% (0.6) | **17.6% (3.3)** |
| Long/short vs S&P 500 | 11.9% (2.8) | 5.4% (0.9) | 16.6% (2.9) |
| *of which: the long ranking (rule 6), vs holding every qualifying name equally* | 1.5% (0.8) | −0.4% (−0.1) | 3.9% (1.6) |

- **The alpha survives the short.** About 17.6% a year against GRID since 2023, at a beta of only
  0.31 to GRID over the full period: the short removes theme exposure, not stock-selection skill.
- **The alpha is real and statistically strong since 2023** (t above 3), and small but positive in
  2019–22.
- **Most of it comes from how the long side is built** — capital-light suppliers, the volatility target
  and the gold/Treasury sleeve. The ranking adds about 4 points a year since 2023 (t ≈ 1.6): suggestive,
  not proven.

---

## The short side in detail

### What is being shorted

![The short basket against the long book and PAVE](img/tophalf/inv_short_leg.png)

The short basket rose **11.3% a year** over the full period, against 23.1% for the long book and 20.8%
for PAVE, with a worst fall of −49%. It falls with the theme but has trailed it — which is what a short
leg should do.

**The basket today (27 names, formed 31 July 2026):**

| group | names | why they rank low |
|---|---|---|
| Electric utilities (8) | BKH · CMS · CNP · EIX · ES · OGE · PCG · TXNM | Own the wires: capital spending a quarter to half of sales |
| Gas utilities (7) | ATO · NFG · NJR · NWN · OGS · SR · SWX | Own the pipe networks |
| Pipelines and LNG (5) | EPD · KNTK · LNG · OKE · TRGP | Own pipelines and terminals |
| Power producer (1) | XIFR | Owns generation |
| Nuclear developers (2) | NNE · SMR | Spending ahead of revenue |
| Industrials and contractors (4) | IESC · PRIM · ROP · XYL | Slowing sales, estimate misses, weak momentum |

About half the basket, on average since 2019, is utilities, power producers and pipelines — the
companies paying for the buildout. The rest are electrification industrials that the buildout is
passing by.

### PAVE as a simple alternative

**PAVE is the simple alternative.** A $0.25 short of the PAVE ETF (US infrastructure and electrical
companies — closest in make-up to the long book) gives almost the same result: Sharpe 1.01 over the
full period and 1.80 since 2023, with a slightly smaller worst fall (−16.2% against −17.3%). It is one liquid line
($13.5bn fund, about $82m traded a day, listed options), normally cheap to borrow — no single-name
borrow to arrange. We prefer the single-name short because its edge is in stock selection rather than
index tracking, and because it fared better in 2019–22 (0.40 against 0.31); an investor who values
simplicity gives up very little with PAVE.

---

## What the rules hold today

![Names held long and short](img/tophalf/inv_breadth.png)

**Long (17 names, equally weighted; rebalanced 11 August 2026):**

> BWXT · DY · ETN · FELE · FIX · HUBB · JCI · KEYS · MIR · MOG-A · MYRG · NVT · PWR · RRX · WCC ·
> WTS · WWD

**Short (27 names, equally weighted, $0.25 per $1 of long; formed 31 July 2026):** listed
[above](#what-is-being-shorted).

The thinnest long name, Franklin Electric, trades about $23m a day; the median long holding trades
$191m. The long book held as few as one name at a 2019 rebalance and has held 14 on average since 2023.

---

## What it costs to run

| source of trading | turnover a year |
|---|---|
| Changing long names at each quarterly rebalance | 164% |
| Resizing the long side daily (volatility target and stretch cut) | 577% |
| Rebalancing the gold/Treasury sleeve | 33% |
| Re-forming the short basket monthly ($0.25 per $1) | 264% |
| **Total** | **~1,040%** |

*Turnover is the value of everything bought plus everything sold over a year, as a percentage of the
portfolio's capital: each day, the absolute change in every position's weight is added up, and the
daily sums are totalled and annualised. 1,040% means trading about ten times the portfolio's capital
a year, counting both sides of each trade.*

The short's trading and borrow are already charged in every figure above. More than half of all
trading is the daily resizing of the long side, which could be done more cheaply with index futures
than by trading every name.

| net of the long side's trading | return a year | Sharpe | max drawdown |
|---|---|---|---|
| Full period, @ 2bp *(overlay in futures)* | 18.5% | 1.02 | −17.4% |
| Full period, @ 10bp | 17.8% | 0.98 | −17.7% |
| Full period, @ 20bp | 16.9% | 0.92 | −18.2% |
| Full period, @ 10bp **+ 1% management fee** | 16.6% | 0.90 | −18.3% |
| Full period, @ 10bp **+ 2% management fee** | 15.4% | 0.83 | −19.0% |
| **2023 on, @ 10bp** | **29.4%** | **1.71** | −11.7% |
| *GRID ETF, 2023 on (net of its fee)* | *23.0%* | *0.93* | *−20.8%* |

Net of realistic costs the strategy keeps a clear lead over GRID in the grid-tight period.
