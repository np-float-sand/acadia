# Screened Electrification Equipment

> **In brief.** Electricity demand is growing faster than the grid can carry it. Regulated utilities
> pay for the new wires, transformers and substations; the equipment makers and contractors that supply
> them are paid as the work is delivered. The strategy holds the most capital-light, fastest-accelerating
> of those suppliers and shorts the weakest fifth of electrification companies, today mostly the
> utilities and pipelines doing the paying.
>
> The grid stress that took hold in 2023 looks durable. Utilities have committed to multi-year capital
> plans, equipment lead times run to years, and real yields remain elevated. The suppliers' own
> filings show no slowdown: their median revenue growth is 10.5% a year after inflation and still
> accelerating, the fastest since early 2023. Over that period the long/short has delivered a Sharpe
> ratio of 1.85 after trading costs, with a beta of 0.56 to the S&P 500. It steps aside into T-bills
> whenever real yields turn negative or suppliers' sales stop growing after inflation — the one
> environment in which it has failed.

*Backtest window: January 2019 – August 2026 · All strategy figures are after trading costs (10bp on
every trade) and borrow · Management fee not assumed*

---

## Performance at a glance

![Sharpe ratio of the strategy, GRID, PAVE, the S&P 500 and VOLT](img/tophalf/inv_sharpe.png)

<div markdown="1" style="font-size:0.84em;color:var(--ink-soft);line-height:1.5">

- **Bars:** Sharpe ratio (return per unit of risk, after trading costs) over the full period (light) and
  since the grid became tight in 2023 (dark).
- **\*VOLT** (an electrification ETF) launched in December 2024; its bar and line cover only its own
  21-month history.

</div>

<p><img src="img/tophalf/inv_rolling.png" alt="Rolling 12-month Sharpe with the regime indicators" style="width:min(72rem,94vw);max-width:none;position:relative;left:50%;transform:translateX(-50%)"></p>

<div markdown="1" style="font-size:0.84em;color:var(--ink-soft);line-height:1.5">

- **Lines:** Sharpe over the trailing 12 months. The shaded labels mark the market environment behind
  each stretch.
- **Solid blue:** the strategy, shown where its trailing year was fully invested.
- **Dashed blue:** the same strategy had rule 11 kept it invested, shown where the real strategy was
  partly or wholly in T-bills (a year mostly in cash has no meaningful Sharpe). It lagged GRID through
  2020–21 and turned negative in 2022 — what rule 11 stepped around — and joins the solid line once the
  strategy has been fully invested for a year.
- **Hatched grey:** time in T-bills (1 November 2019 – 29 April 2022, plus three brief spells in 2019).
- **Amber dotted line (right axis):** median revenue growth, after inflation, of the suppliers that pass
  the long book's rules, from their filings as published — a proxy for the buildout. It measures
  demand, not valuations.
- **Red dotted line (right axis):** the 10-year real yield.
- **Zero lines:** the Sharpe, supplier-growth and real-yield axes share one zero line, so both dotted
  lines above it means the regime is green.
- **Regime strip:** green when both the real yield and the suppliers' real revenue growth are above
  zero — the only time rule 11 keeps the strategy invested — amber when one is, red when neither. Amber
  or red from late 2019 to spring 2022; green without a break since.

</div>

### Since the grid became tight (January 2023 – August 2026)

| | **Long/short (the strategy)** | GRID ETF | PAVE ETF | S&P 500 | VOLT ETF* |
|---|---|---|---|---|---|
| **Sharpe ratio** | **1.85** | 0.93 | 0.91 | 1.16 | 0.78 |
| **Calmar ratio** | **2.78** | 1.11 | 0.87 | 1.20 | 0.97 |
| **Max drawdown** | **−11.9%** | −20.8% | −26.2% | −18.8% | −23.4% |
| **Volatility** | **13.8%** | 20.3% | 20.6% | 15.0% | 25.5% |
| **Beta to the S&P 500** | **0.56** | 1.10 | 1.09 | 1.00 | 1.00 |
| **Return a year** | **33.2%** | 23.0% | 22.8% | 22.6% | 22.8% |
| **Max time under water** | 5.0 months | 7.6 months | 7.9 months | **4.2 months** | 5.3 months |
| Return / Sharpe before trading costs | 33.8% / 1.89 | | | | |

*\*VOLT: December 2024 – August 2026 only (21 months). Over that same window the strategy's Sharpe
was 1.28, its max drawdown −11.9% and its return 23.1% a year.*

### Before the grid was tight (January 2019 – December 2022)

| | **Long/short (the strategy)** | GRID ETF | PAVE ETF | S&P 500 |
|---|---|---|---|---|
| **Sharpe ratio** | 0.55 | **0.77** | 0.60 | 0.48 |
| **Calmar ratio** | **0.98** | 0.58 | 0.43 | 0.39 |
| **Max drawdown** | **−9.0%** | −40.6% | −44.1% | −33.7% |
| **Volatility** | **8.8%** | 27.4% | 29.7% | 22.5% |
| **Beta to the S&P 500** | **0.12** | 1.06 | 1.13 | 1.00 |
| **Return a year** | 8.8% | **23.5%** | 19.1% | 13.1% |
| **Max time under water** | 33.1 months | 13.3+ months | 11.8+ months | 11.9+ months |
| Return / Sharpe before trading costs | 9.2% / 0.59 | | | |

**This is a grid-tightness strategy.** It does not work well when the grid has ample spare capacity:
in 2020–21 the suppliers' median revenue growth fell to −8% a year after inflation, and the strategy
trailed the theme ETFs. There is no exit signal for the buildout itself, but the strategy de-risks on
its own. It steps aside into T-bills unless both the 10-year real yield and the suppliers' real revenue
growth are above zero (rule 11 — it held T-bills from November 2019 to April 2022). Within the book, position size falls automatically as
volatility rises or prices run far above trend (to a third of full size in April 2025), and the
ranking rotates toward suppliers whose sales are still growing.

*Sharpe uses a flat **4.0% risk-free rate**. Returns are total returns, dividends reinvested. The
strategy's figures are after 10bp on every trade — long and short name changes, resizing, sleeve and
short resets, and each rule-11 switch — and 1% a year borrow on the short ([Costs](#what-it-costs-to-run));
the "before trading costs" row removes the 10bp. ETF returns are net of their own fees. Max time under water is the longest stretch below a previous peak
within the window; "+" means the drawdown was still unrecovered at the end of 2022.*

**Since 2023 the strategy leads every benchmark on every measure but one** — the S&P 500 recovered
from the 2025 tariff selloff about a month faster. **Before 2023 it was a capital-preservation
strategy:** a quarter of the drawdown of the theme ETFs, the highest Calmar ratio, and almost no market
exposure — but a lower Sharpe and return than GRID and PAVE, and a long spell under water (July
2019 – May 2022), most of it spent in T-bills while a clean-energy rally lifted exactly the stocks the
rules exclude.

<p><img src="img/tophalf/inv_growth.png" alt="Growth of $1 with the regime strip" style="width:min(72rem,94vw);max-width:none;position:relative;left:50%;transform:translateX(-50%)"></p>

<div markdown="1" style="font-size:0.84em;color:var(--ink-soft);line-height:1.5">

- **VOLT's line** starts on the strategy's line at VOLT's launch, so the two can be compared from that
  date.
- **Hatched grey:** time in T-bills under rule 11. **Dashed blue:** the strategy's path had rule 11
  kept it invested, starting from its level on the day it moved to T-bills. It fell through the COVID
  crash and ended the spell close to where the strategy stood in T-bills — rule 11 gave up little return
  and avoided the swings.
- **Amber and red dotted lines (right axes):** the buildout proxy and the 10-year real yield, as
  above; their zero lines are aligned, so where both sit above the shared zero the regime is green.
- **Regime strip:** as above.
- Over the whole period GRID and PAVE, carrying the full theme at 1.1× the market's beta, compounded
  faster than the strategy; since 2023 the strategy compounded fastest.

</div>

---

## How the portfolio is built

The gap in capital intensity between the utilities and their suppliers is a fact in the accounts
rather than a judgment call, which is what lets the whole portfolio be built by rule: nine rules for
the long book, one for the short side and one for when to step aside.

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
averaged. At each quarterly rebalance new names enter from the top half of the names that pass rules
1–5, and a name already held stays as long as it remains in the top 65% — a buffer that stops names
near the cut-off from churning in and out. Holdings are equally weighted, with no name above 25%. We tested fourteen candidate ranking signals — earnings and
revenue surprises, momentum, reversal, and sensitivity to wholesale power prices, congestion and
scarcity pricing — and these two were the only ones with a consistent relationship to future returns.

### The long book: how we control risk

**7. We cut size when the book gets volatile.** Exposure is set each day to target 20% annual
volatility, measured over the previous month, and never exceeds 1.5×. To avoid trading on small
changes, the book is resized only when the target moves more than 20 points of exposure away from
where the book stands (for example from 80% to below 60% or above 100%); large moves, such as a
volatility spike, still go straight through.

**8. We cut size again when the book gets stretched.** When it has run more than 15% above its 200-day
average, or far ahead of the market over the past year, exposure steps down to 80%, and to 60% at the
extremes.

**9. We hold 15% in gold and Treasuries** (GLD and IEF), reset to 15% monthly, uncorrelated with the
rest of the book when that matters most.

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
- **Basket.** The bottom fifth by that score, excluding any name the long book holds, equally weighted
  and re-formed each quarter: 16 names on average, never fewer than 12, 24 today.
- **Size.** $0.25 short per $1 of long book, reset to $0.25 at each month-end.

### When the strategy steps aside

**11. We hold the strategy only while the regime is green: the 10-year real yield is above zero and
the suppliers' revenue is still growing after inflation.** The first is the 10-year inflation-protected
Treasury (TIPS) yield; the second is the median trailing-twelve-month revenue growth, as filed and less
CPI inflation, of the names that pass rules 1–5. When either turns negative, the whole portfolio —
long, short and sleeve — moves into 3-month T-bills the next day, and returns when both are positive
again. See [Real yields and the buildout](#real-yields-and-the-buildout).

### No look-ahead

Every decision in the backtest uses only information that existed on the day it was made: industry
codes as filed; 10-K wording only from filings published *before* the year being decided; earnings,
capital spending and revenue as first reported; a ranking or score from the latest month-end on or
before the trade; position size from volatility up to *yesterday*; the stretch cut from the *previous*
month's reading; rule 11 from the previous day's real yield and the latest filed revenue and published CPI.

What a backtest cannot remove is **research hindsight**: the long ranking, the short score, the rule-2
filter on the short, the $0.25 size and rule 11 were all chosen in September 2026 after comparing alternatives
on this same history, so the backtest flatters them to some degree.

---

## Performance

### Year by year

![Calendar-year returns](img/tophalf/inv_years.png)

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|---|
| **Long/short** | 30.3% | 0.4% *(T-bills)* | 0.0% *(T-bills)* | 7.0% | 41.0% | 34.1% | 30.9% | 14.8% |
| *Long/short had rule 11 kept it invested* | *40.9%* | *−0.4%* | *7.8%* | *−1.4%* | *41.0%* | *34.1%* | *30.9%* | *14.8%* |
| GRID ETF | 42.2% | 48.8% | 27.7% | −13.9% | 21.6% | 15.2% | 29.7% | 17.1% |
| PAVE ETF | 32.6% | 19.7% | 36.4% | −7.2% | 31.0% | 17.9% | 19.4% | 14.7% |
| S&P 500 | 31.1% | 18.3% | 28.7% | −18.2% | 26.2% | 24.9% | 17.7% | 13.1% |
| Long/short Sharpe | 1.67 | *in T-bills* | *in T-bills* | 0.32 | 2.28 | 1.92 | 1.67 | 1.35 |
| Long/short worst drawdown | −9.0% | 0.0% | 0.0% | −7.5% | −8.1% | −6.2% | −11.9% | −7.3% |

The strategy beat GRID in every year from 2022 to 2025 and trails it so far in 2026, when the theme
itself has rallied hard. It held T-bills from November 2019 to April 2022, so it missed GRID's 49%
and 28% in 2020 and 2021 — a rally in the clean-energy and subsidy stocks the rules exclude — but also
the COVID crash, and was up 7.0% in 2022, when the S&P lost 18%. *A year spent in T-bills has no
meaningful Sharpe. The hatched blue bars show the return had rule 11 kept the strategy invested, for
each year it spent partly or wholly in T-bills (2019: the last two months; 2022: the first four).*

Over its one episode rule 11 was roughly a wash on return — it gave up 11 points in late 2019 and 8
in 2021, and saved 8 in 2022 — but it held the strategy flat through the COVID crash and the 2022
rate shock, which is what cut the worst drawdown from −15.9% to −11.9%.

### Drawdowns

![Drawdown](img/tophalf/inv_drawdown.png)

---

## Real yields and the buildout

**Rule 11 exists because the strategy has one clear weakness: a market that no longer rewards what
it holds.** Two conditions mark that market, and the strategy stays invested only while both hold.

- **Real yields above zero.** When the 10-year real yield is negative, money is effectively free, and
  the market pays up for distant earnings, subsidy-dependent growth and story stocks — everything the
  rules exclude. When it is positive, capital has a cost, and suppliers that are paid now are worth
  more than companies spending heavily to be paid later.
- **Suppliers' revenue still growing after inflation.** When the median supplier's real sales are
  shrinking, the buildout is not reaching the companies the strategy owns, and the ranking has little
  to separate winners from losers.

| | Sharpe, full | Sharpe, 2019–22 | Sharpe, 2023 on | max drawdown, full | Calmar, full | return a year | time invested |
|---|---|---|---|---|---|---|---|
| **Rule 11: both conditions (the strategy)** | **1.28** | **0.55** | 1.85 | **−11.9%** | 1.66 | 19.8% | 67% |
| Real yield alone | 1.27 | 0.55 | 1.85 | −11.9% | **1.67** | 20.0% | 70% |
| Always invested | 1.13 | 0.49 | 1.85 | −15.9% | 1.31 | **20.8%** | 100% |

- **It costs nothing in the grid-tight period.** Both conditions have held since spring 2022, so rule
  11 has not been triggered since; every 2023-on figure is identical with or without it.
- **It limits the damage when the environment turns.** Over the full period the worst fall drops from
  −15.9% to −11.9% and the Calmar ratio rises by more than a quarter, for about a point a year less
  return. The dashed line in the rolling-Sharpe chart shows what it stepped around.
- **Requiring both conditions performs the same as real yields alone** and steps aside a few months
  earlier — in November 2019, as supplier sales began to shrink, rather than in March 2020. We use both
  because each describes a different half of the thesis: the cost of capital and the buildout itself.
- **Rising real yields also help the short.** Utilities trade like bonds: their monthly returns move
  against changes in the real yield (correlation −0.4 to −0.7 in every period since 2013), and they
  fell 2–3% on average in months when real yields jumped more than 0.2 points.
- **Where it stands now.** The 10-year real yield was **2.44% at the end of August 2026**, close to its
  post-2008 high of about 2.5%, and suppliers' real revenue growth is 10.5% and rising. Both conditions
  are well clear of triggering.

**The caveat.** The rule has been tested on essentially one episode — late 2019 to spring 2022 — and it
was designed after seeing that episode. We adopt it because the mechanism is clear and the weakness is
known, not because the backtest proves it; it will be judged on the live record. Moving to T-bills
also lengthens recovery: the strategy's July 2019 peak was not regained until May 2022.

---

## Where the return comes from

A regression of each book's daily return on GRID's (or the S&P's): whatever is left after accounting
for exposure to the theme is alpha. A t-statistic above 2 means the result is unlikely to be luck.

| alpha a year (t-statistic) | full period | 2019–22 | **2023 on** |
|---|---|---|---|
| **Long/short vs GRID** | **13.4% (3.5)** | 6.5% (1.5) | **18.7% (3.5)** |
| Long/short vs S&P 500 | 14.3% (3.7) | 7.1% (1.6) | 17.5% (2.9) |
| *of which: the long ranking (rule 6), vs holding every qualifying name equally (measured without the buffer)* | 1.5% (0.8) | −0.4% (−0.1) | 3.9% (1.6) |

- **The alpha survives the short.** About 18.7% a year against GRID since 2023, at a beta of only
  0.22 to GRID over the full period: the short removes theme exposure, not stock-selection skill.
- **The alpha is real and statistically strong** (t above 3 over the full period and since 2023).
  In 2019–22 it is positive but not statistically reliable, and part of it is rule 11 keeping the
  strategy out of a market it would have lagged.
- **Most of it comes from how the long side is built** — capital-light suppliers, the volatility target
  and the gold/Treasury sleeve. The ranking adds about 4 points a year since 2023 (t ≈ 1.6): suggestive,
  not proven.

---

## The short side in detail

### What is being shorted

![The short basket against the long book and PAVE](img/tophalf/inv_short_leg.png)

The short basket rose **7.2% a year** over the full period, against 20.8% for PAVE, with a worst fall
of −47%. It falls with the theme but has trailed it — which is what a short
leg should do.

**The basket today (24 names, formed 30 June 2026):**

| group | names | why they rank low |
|---|---|---|
| Electric utilities (7) | BKH · CNP · EIX · ES · PCG · POR · TXNM | Own the wires: capital spending a quarter to half of sales |
| Gas utilities (5) | ATO · NJR · NWN · SR · SWX | Own the pipe networks |
| Pipelines and LNG (5) | EPD · KNTK · LNG · OKE · TRGP | Own pipelines and terminals |
| Power producer (1) | XIFR | Owns generation |
| Nuclear developers (2) | NNE · SMR | Spending ahead of revenue |
| Industrials and contractors (4) | CARR · ENR · IESC · XYL | Slowing sales, estimate misses, weak momentum |

About half the basket, on average since 2019, is utilities, power producers and pipelines — the
companies paying for the buildout. The rest are electrification industrials that the buildout is
passing by.

### PAVE as a simple alternative

**PAVE is the simple alternative.** A $0.25 short of the PAVE ETF (US infrastructure and electrical
companies — closest in make-up to the long book) gives almost the same result under the same rules:
Sharpe 1.20 over the full period and 1.79 since 2023, with a worst fall of −13.4% against −11.9%. It is one liquid line
($13.5bn fund, about $82m traded a day, listed options), normally cheap to borrow — no single-name
borrow to arrange. We prefer the single-name short because its edge is in stock selection rather than
index tracking; an investor who values simplicity gives up very little with PAVE.

---

## What the rules hold today

![Names held long and short](img/tophalf/inv_breadth.png)

**Long (17 names, equally weighted; rebalanced 11 August 2026):**

> BWXT · DY · ETN · FELE · FIX · HUBB · KEYS · MIR · MKSI · MOG-A · MYRG · NVT · PWR · RRX · WCC ·
> WTS · WWD

**Short (24 names, equally weighted, $0.25 per $1 of long; formed 30 June 2026):** listed
[above](#what-is-being-shorted).

The thinnest long name, Franklin Electric, trades about $23m a day; the median long holding trades
$191m. The long book held as few as one name at a 2019 rebalance and has held 14 on average since 2023.

---

## What it costs to run

| source of trading | turnover a year while invested, two-way | one-way |
|---|---|---|
| Changing long names at each quarterly rebalance (with the hold buffer) | 154% | 77% |
| Resizing the long side (volatility target and stretch cut, 20-point band) | 188% | 94% |
| Resetting the gold/Treasury sleeve monthly | 15% | 8% |
| Re-forming the short basket quarterly ($0.25 per $1) | 126% | 63% |
| Resetting the short to $0.25 monthly | 12% | 6% |
| **Total, while invested** | **~495%** | **~250%** |
| **Total, averaged over 2019–2026** (67% of the time invested under rule 11, including its six switches in and out) | **~430%** | **~215%** |
| *For comparison, one-way as reported by each fund: GRID 26%, PAVE about 5–13%, SPY 3%* | | |

*Turnover measures how much is traded in a year relative to the portfolio's capital. **Two-way**
counts every purchase and every sale: each day, the absolute change in every position's weight is
added up, and the daily sums are totalled and annualised. **One-way** — the convention in fund
prospectuses, including the ETF figures above — counts roughly one side, the lesser of purchases or
sales, so it is about half the two-way figure. On the one-way convention the strategy trades like an
active fundamental long/short fund or a systematic factor fund (typically 100–600% a year), not like
an index ETF.*

| trading cost per trade | return a year | Sharpe | max drawdown |
|---|---|---|---|
| None *(before trading costs)* | 20.3% | 1.32 | −11.8% |
| 2bp *(resizing done in index futures)* | 20.2% | 1.31 | −11.8% |
| **10bp (the figures on this page)** | **19.8%** | **1.28** | **−11.9%** |
| 20bp | 19.3% | 1.24 | −12.1% |
| 10bp **+ 1% management fee** | 19.0% | 1.22 | −12.1% |
| 10bp **+ 2% management fee** | 18.2% | 1.17 | −12.3% |
| **2023 on, 10bp** | **33.2%** | **1.85** | −11.9% |
| *GRID ETF, 2023 on (net of its fee)* | *23.0%* | *0.93* | *−20.8%* |

*Full period unless stated. Borrow on the short (1% a year) is charged in every row.* Even at twice our
assumed trading cost, or with a 2% management fee, the strategy keeps a Sharpe well above every
benchmark's.
