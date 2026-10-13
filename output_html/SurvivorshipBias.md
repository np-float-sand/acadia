# Survivorship bias check — one-score long/short

*Tested 2026-10-02 · backtest window January 2019 – August 2026*

## The question

The backtest draws its stocks from SEC filers in power and electrical-equipment industry codes, priced from a
data source that only carries **companies still listed today**. Companies that went bankrupt, were taken over or
delisted during the backtest are missing. If the strategy would have bought losers or shorted takeover targets
among them, the reported results are too good.

To test this we found the missing companies, priced them, put them back in the pool, and re-ran the strategy
unchanged.

## Headline

**Putting the missing companies back barely changes the result.** Sharpe moves by 0.07 or less in every
period, with or without the step-aside rule.

| 2023 – Aug 2026 | Reported (survivors only) | With missing names added |
|---|---|---|
| Sharpe ratio | **2.05** | **2.00** |
| Annual return | 36.8% | 35.8% |
| Volatility | 13.8% | 13.8% |
| Maximum drawdown | −10.2% | −10.4% |
| Calmar ratio | 3.62 | 3.46 |
| Sortino ratio | 2.92 | 2.83 |
| Alpha vs GRID ETF | 21.7% a year (t 3.86) | 20.8% a year (t 3.72) |
| Beta to S&P 500 | 0.54 | 0.54 |

The step-aside rule does not trigger after 2022, so these 2023+ figures are the same with and without it.

### Whole period and 2019–22

| Version | Period | Sharpe | Annual return | Max drawdown |
|---|---|---|---|---|
| **With step-aside** (the strategy) | 2019 – Aug 2026 | 1.34 → 1.36 | 20.8% → 20.5% | −11.1% → −10.4% |
| | 2019 – 2022 | 0.44 → 0.49 | 7.9% → 8.1% | −11.1% → −8.6% |
| **Without step-aside** | 2019 – Aug 2026 | 1.38 → 1.40 | 25.1% → 25.2% | −13.9% → −16.5% |
| | 2019 – 2022 | 0.79 → 0.86 | 15.4% → 16.2% | −13.9% → −16.5% |

*Reported → with missing names added.* The one material difference is drawdown without the step-aside rule:
the extra short positions hurt in 2020–22, the stretch the rule sits out. With the rule on, the drawdown is
slightly smaller.

## Which companies were missing

Of 2,571 SEC filers in the strategy's industry codes, 2,196 have no stock ticker today. Most disappeared
before 2016. **229 dropped out during the backtest years**; we could match and price **57** of them that were
still trading after January 2019. Those 57 were added to the pool.

### What the strategy did with them

The added names pass through the same rules as every other stock — trading-volume floor, screen, fourteen-signal
score — so most never qualified.

- **21 of 57** passed the $10 million-a-day trading-volume floor at some point.
- **8 were actually traded**: 2 on the long side, 7 on the short side (Barnes on both, at different times).
- Average names held moved from 9.9 to 10.3 (long) and 16.4 to 17.6 (short).

| Ticker | Company | How it left | Long | Short |
|---|---|---|---|---|
| B | Barnes Group | Taken private by Apollo, Jan 2025 | ✅ ~2½ years | ✅ 7 quarters |
| ESL | Esterline Technologies | Acquired, Mar 2019 | ✅ ~3 months | |
| ALE | ALLETE | Taken private, Dec 2025 | | ✅ 10 quarters |
| ARCH | Arch Resources | Merged into Core Natural Resources, Jan 2025 | | ✅ 9 quarters |
| AGR | Avangrid | Taken private by Iberdrola, Dec 2024 | | ✅ 7 quarters |
| DCP | DCP Midstream | Acquired, Jun 2023 | | ✅ 5 quarters |
| CEQP | Crestwood Equity | Acquired, Nov 2023 | | ✅ 5 quarters |
| SJI | South Jersey Industries | Taken private, Feb 2023 | | ✅ 3 quarters |

**Qualified but never picked (13):** Equitrans Midstream, EQM Midstream, EnLink Midstream, EnLink Midstream
Partners, TC PipeLines, Covanta, Pattern Energy, TerraForm Power, Calpine, Altra Industrial Motion, Anixter,
Control4, Hyzon Motors.

**Added but never qualified (36)** — too thinly traded, or gone before 2019: Houston Wire & Cable, Aegion,
Briggs & Stratton, FARO, MTS Systems, Rudolph Technologies, IntriCon, Maxwell, UQM, Goldfield, Velodyne,
Berkshire Grey, Snap One, Enable Midstream, Summit Midstream, Rattler Midstream, CNX Midstream, Archrock
Partners, Altus Power, Heliogen, Emeren, Just Energy, EnSync, Westmoreland Coal, Westmoreland Resource
Partners, and eleven micro-caps and blank-check companies.

### Why the effect is small

Almost every missing company that qualified was a utility, pipeline or coal producer. They spend heavily on
plant, so they fail the long book's capital-light screen and land in the short pool. Most of them later
**left through a takeover at a premium**, which costs a short seller money. So survivorship here mainly removes
takeover targets from the short book, not bankruptcies from the long book. The long book picked only Barnes,
which made money.

## Companies we could not include

| Company | How it left | Would it have qualified? |
|---|---|---|
| **Sunnova Energy** | Bankrupt, Jun 2025 | **Likely** — heavily traded 2020–25; a probable short candidate |
| QualTek Services | Bankrupt, 2023 | Borderline, probably too thinly traded |
| Sunworks | Bankrupt, 2024 | Probably not — too thinly traded |
| Cloud Peak Energy | Bankrupt, 2019 | Probably not |
| Southcross, Electriq Power, XT Energy, Arotech | Acquired / delisted | Probably not |

No free price history exists for these. **Sunnova is the one that matters.** If the strategy had shorted it,
a rough estimate is +1% a year and +0.05 to +0.07 Sharpe for 2023+, which would put the result at roughly
**2.00–2.07 against the reported 2.05**. This estimate is from approximate prices and was not run.

## Caveats

- **Prices for delisted companies** come from Tiingo. Each series is cut at the company's exit to avoid
  ticker re-use (for example, "B" now belongs to Barrick Gold). Barnes's trading-volume figure looks inflated
  by this, but Barnes qualifies either way.
- **Exits during 2026 were not checked.** A company that filed its 2025 annual report and was then taken over
  this year falls outside both lists.
- **Re-registered companies** (a new holding company or a blank-check merger) can look alive under a new SEC
  number while their old history is lost.
- **A full fix needs a survivorship-free database** (CRSP or equivalent) that includes delisting returns.

## How it was done

1. Pulled every annual-report filer in the strategy's 28 industry codes from SEC EDGAR, and kept those with no
   ticker today that filed an annual report in 2016 or later.
2. Recovered each one's old ticker from its last annual report, checked against Tiingo's listing dates, and
   downloaded daily prices and trading volume.
3. Added them, with their SEC financial data and annual-report wording, to a separate copy of the data. The
   production data was not changed.
4. Re-ran the whole strategy unchanged on that copy, with and without the step-aside rule.
5. **Control:** the same re-run *without* the added companies reproduces the reported figures exactly
   (Sharpe 1.341 / 0.436 / 2.053), so every difference above comes from the added companies.

*Detail: `docs/survivorship-audit-results.md` · code and data: `research/survivorship_audit/`.*
