<!--
  Turn this file into HTML / PDF with tools/md2html.py (from the repo root):

    poetry run python tools/md2html.py docs/electrification-strategy-proposal-v3.md --theme auto --pdf

  -> writes docs/electrification-strategy-proposal-v3.html  (light/dark auto, corner toggle)
     and   docs/electrification-strategy-proposal-v3.pdf    (light-locked, A4)
  Figures (docs/pv3-*.png) are inlined as data URIs, so both files are self-contained.

  Options:  --theme light | --theme dark   lock the theme     --no-toggle   drop the button
            --toc   add a table of contents                    -o PATH      choose the output path
-->

# Screened Electrification Equipment

**Addenda Capital · Systematic Equity — Proposal v3 · 2026-09-07**

A rules-built long book of the profitable equipment and engineering suppliers to the
electricity build-out — the picks-and-shovels of electrification, with the theme's ballast
stripped out and three risk controls layered on.

Full backtest Jan 2019 – Aug 2026: **+348%** total return, **0.61** market beta, **−23%**
max drawdown, Sharpe **1.05**. This note also documents an exhaustive search for a short
offset — nine candidates — and concludes that none of them improves the book.

---

## 1. What the book holds

The book is long roughly **27 companies that sell into the electricity build-out**: power
equipment (transformers, switchgear, power electronics), grid-scale storage, and electrical
engineering & construction. It holds *suppliers* — not utilities, not power generators.
Representative names: Eaton, GE Vernova, Quanta Services, Vertiv, Hubbell, nVent, Fluence,
Primoris.

### The selection rule

A company enters the book only if it (a) sells to institutional buyers — utilities,
hyperscalers, industrial customers, not consumers — and (b) meets at least two of three
fundamental tests: **profitable today**, **valued on earnings** rather than on story, and
**low dependence on subsidies or policy**. Names are held in **equal weight**, capped at
25% each. There is no cap-weighting, so the book does not concentrate in a few mega-caps
the way a thematic index does.

### Three risk controls

- **Volatility scaling.** The book automatically reduces its invested percentage when
  markets turn choppy and restores it when they calm, aiming to keep annual volatility
  near 20%.
- **Valuation trim.** After the book has run up well ahead of its own long-run trend and
  the broad market, it steps exposure down in stages — to about 80%, then 60% — taking
  risk off into euphoria.
- **A 15% reserve sleeve.** One-seventh of the portfolio is permanently held in gold and
  short-term US Treasuries instead of equities, as a shock absorber: a growth or credit
  scare pushes Treasuries up; a currency or geopolitical shock pushes gold up.

### Is this just VOLT? In what way is it smarter?

It is the same theme. The book runs about **0.86 correlated** with VOLT, the
electrification-theme ETF, and essentially all of its return is the theme. It is "picks and
shovels." It is smarter than buying the ETF in four concrete ways:

- **It strips the ballast.** No regulated utilities (~30% of VOLT), no pipelines or
  midstream (~10%), no commodity components (~13%). Pure equipment and EPC.
- **It screens on fundamentals.** Only profitable, earnings-valued, low-policy-dependence
  names — so it drops the pre-revenue, subsidy-reliant and story-stock exposure a thematic
  index sweeps in.
- **Equal weight, 25% cap** instead of cap weight — materially less single-name
  concentration.
- **Risk controls the ETF has none of** — the three above.

Over the only fair comparison window — January 2025 to August 2026, VOLT's live history —
the book returns **Sharpe 1.49 vs VOLT's 1.05**, a **−17% max drawdown vs −23%**, at
**63% of VOLT's volatility**, for slightly less raw return (CAGR 27.7% vs 30.5%). Over the
full 2019–2026 backtest it earns Sharpe 1.05 at **0.61 market beta**: it captures the theme
with roughly half the market sensitivity of the infrastructure ETFs.

---

## 2. Performance

![Growth of $1, Jan 2025 to Aug 2026: the book, VOLT, GRID and SPY](pv3-growth-fair.png)

*Fair-comparison window. The book keeps pace with the electrification ETFs through mid-2026
with a much smoother path — it does not spike to VOLT's 1.8× in May–July 2026, and it does
not give it back.*

![Growth of $1 on a log scale, Jan 2019 to Aug 2026: the book, PAVE and SPY](pv3-growth-full.png)

*Full backtest. The book trails through 2019–2022, then compounds through the
electrification up-leg to finish level with the infrastructure ETF and ahead of the market
— at lower volatility and a shallower worst drawdown than either.*

![Drawdown from prior peak](pv3-drawdown.png)

*Drawdown from prior peak. COVID: book −22% vs SPY −34%. 2022 rate shock: −20% vs −24%. The
book's worst point in the backtest is −23%.*

![Rolling 12-month Sharpe of the book vs SPY](pv3-rollsharpe.png)

*Rolling 12-month Sharpe. Both go negative in 2022; the book leads the market for most of
the record and through the 2023–2026 up-leg.*

### Headline statistics

| Strategy | Total ret | CAGR | Ann vol | Sharpe | Max DD | SPY β | Fair Sharpe | Fair DD |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **Screened Electrification Equipment** (the book) | **+348%** | +20.8% | 16.3% | **1.05** | **−23%** | 0.61 | **1.49** | **−17%** |
| VOLT — electrification ETF | —¹ | — | 25.5% | — | — | 1.01 | 1.05 | −23% |
| GRID — grid-infrastructure ETF | +397% | +23.3% | 24.2% | 0.81 | −41% | 1.07 | 1.06 | −20% |
| PAVE — US infrastructure ETF | +327% | +20.9% | 25.7% | 0.67 | −44% | 1.12 | 0.77 | −23% |
| SPY — S&P 500 | +243% | +17.5% | 19.3% | 0.71 | −34% | 1.00 | 0.88 | −19% |
| XLI — S&P industrials | +207% | +15.8% | 21.5% | 0.56 | −42% | 0.97 | 0.87 | −18% |
| SPY + 15% GLD/IEF sleeve | +222% | +15.9% | 16.3% | 0.75 | −29% | 0.86 | 1.01 | −16% |

Full backtest Jan 2019 – Aug 2026 unless noted. Fair window = Jan 2025 – Aug 2026 (VOLT's
live history). Sharpe = (CAGR − 3.76%) / annualised volatility. ¹ VOLT listed in 2024, so
its full-window statistics are not meaningful.

### Behaviour in named stress episodes

| Episode | Book | SPY | VOLT |
|---|--:|--:|--:|
| COVID crash (Feb–Apr 2020) | −22% | −34% | — |
| Rate shock (2022) | −20% | −24% | — |
| DeepSeek selloff (Jan–Apr 2025) | −17% | −19% | −23% |
| Tariff shock (Apr 2025) | −8% | −12% | −10% |
| 2026 selloff (YTD) | −6% | −9% | −17% |

---

## 3. Two-year split

Sharpe / total return / worst drawdown, computed separately for the 2023–2024 and
2025–2026 sub-periods. The *Why / what it is* column explains why each row is on the table.

| Row | Why / what it is | Sh 23–24 | Ret 23–24 | DD 23–24 | Sh 25–26 | Ret 25–26 | DD 25–26 |
|---|---|--:|--:|--:|--:|--:|--:|
| *The book & its variants* | | | | | | | |
| **Screened Electrification Equipment** (the book) | Profitable equipment & engineering suppliers to the grid build-out, equal-weighted.<br>Automatically shrinks exposure when markets get volatile or prices outrun their trend.<br>Permanently holds 15% in gold and short-term Treasuries as a shock absorber. | **2.08** | +86% | **−10%** | **1.49** | +52% | **−17%** |
| · raw book, no overlays | The same names in equal weight, with none of the risk controls.<br>Shown to isolate what the overlays cost in return and save in drawdown. | 1.87 | +114% | −16% | 1.27 | +62% | −22% |
| · + built-in insurance short | Adds a small conditional short in data-centre landlords (Digital Realty, Equinix).<br>Engages only while the 10-year real yield is rising; carried as paid tail insurance. | 1.81 | +69% | −8% | 1.66 | +53% | −12% |
| · thematic universe + controls | Same risk controls, but names chosen by "held by ≥2 electrification ETFs".<br>Broader, more index-like — drops the profitability screen. | 1.29 | +61% | −14% | 1.29 | +49% | −19% |
| *Book + a short offset* | | | | | | | |
| book + short · XLI 0.25× | Hold the book, short industrials at ~¼ of book size.<br>Aim: cancel the book's general market and cyclical swings, keep the stock picks. | 2.13 | +72% | −7% | 1.43 | +41% | −13% |
| book + short · China FXI 0.15× | Hold the book, short Chinese large-caps at ~15%.<br>A cyclical exposure that historically moves least in step with the book. | 1.97 | +80% | −9% | 1.42 | +47% | −18% |
| book + short · TAN solar 0.15× | Hold the book, short solar and distributed-energy stocks at ~15%.<br>The theorised natural offset — names that should gain from high power prices and green policy. | 2.81 | +105% | −7% | 1.35 | +42% | −15% |
| book + short · merchant gen 0.15× | Hold the book, short Vistra / NRG / Constellation / Talen at ~15%.<br>The listed names most directly long the wholesale electricity price. | 1.32 | +49% | −10% | 1.58 | +42% | −12% |
| *Benchmarks* | | | | | | | |
| VOLT | Electrification-theme ETF, the closest off-the-shelf comparable; ~40% is utilities, pipelines and commodity components. | — | — | — | 1.02 | +55% | −23% |
| GRID | Grid-infrastructure ETF — T&D equipment plus utilities, weighted by size. | 0.83 | +40% | −21% | 1.04 | +52% | −20% |
| PAVE | Broad US infrastructure ETF (construction, materials, machinery) — the "just buy infrastructure" alternative. | 1.05 | +54% | −13% | 0.75 | +37% | −23% |
| SPY | The S&P 500 — the overall US stock market. | 1.65 | +58% | −10% | 0.85 | +33% | −19% |
| XLI | The S&P industrials sector — the book's parent sector. | 0.95 | +39% | −13% | 0.85 | +35% | −18% |
| SPY + 15% GLD/IEF sleeve | The market with the book's exact reserve sleeve bolted on — isolates the sleeve from the stock selection. | 1.70 | +52% | −9% | 1.01 | +34% | −16% |

Sharpe = (CAGR − 3.76%) / annualised vol, same risk-free rate both periods. Max DD is the
worst peak-to-trough *within* each period. VOLT has no 2023–24 history.

> **Read the shorts row-by-row.** XLI and FXI look strong in 2023–24 (Sharpe 2.1–2.0) — but
> that is the window the book beat its own sector by 2×, so shorting the laggard flatters
> the ratio. In 2025–26 every short cuts return and most cut Sharpe. TAN solar's 2.81 in
> 2023–24 is the solar crash, not a hedge: it adds nothing in 2025–26 and is an anti-hedge
> in the scenario the book most fears (see §6).

---

## 4. Systematic risks & macro bets

What the book is implicitly betting on, and how — or whether — the three overlays address
each one.

| Risk / macro bet | Treatment | What it is, and how the strategy handles it |
|---|---|---|
| Electrification theme de-rating | Partial | AI-capex decelerates, data-centre moratoria spread, or grid capex plateaus and the whole theme re-prices down. Volatility scaling de-grosses on the way down and the valuation trim lowers exposure after run-ups; the sleeve is unaffected. Not eliminated — this is the primary disclosed risk. |
| Falling-rate, risk-on reversal | Not hedged | Rates fall, the washed-out clean-energy complex rallies, and the theme de-rates in relative terms. By design there is no hedge: the Treasury leg of the sleeve gains as rates fall and gold is roughly neutral, but a solar short would *worsen* this — which is why there is none. |
| Broad equity drawdown | Cushioned | A market-wide selloff; the book carries 0.61 market beta. The 15% sleeve and volatility scaling both reduce the hit — the book fell −22% in COVID vs the market's −34%. |
| Rate shock / duration | Mixed | Yields spike fast (2022-style) and long-duration growth multiples compress. The Treasury leg of the sleeve *loses* in a rate spike; volatility scaling helps; the book still drew −20% in 2022 (vs the market's −24%). |
| Industrial / cyclical recession | Limited | Construction and capital-equipment orders slow. Grid capex is backlog-driven and partly regulated — more defensive than broad industrials — but not immune. No explicit hedge. |
| Policy / subsidy reversal | Mitigated | IRA rollback or investment-tax-credit repeal. Mitigated by construction — the screen's "low policy dependence" test deliberately excludes the subsidy-reliant names. |
| Single-name concentration | Bounded | 27 names, 25% cap; a blow-up in a top holding. Equal weight caps any one name near 4–8% of risk. It is not a 200-stock book — the concentration is deliberate and disclosed. |
| Input-cost inflation (copper · electrical steel · labour) | Not a risk | The book's real cost inputs spike. Tested directly: the book's beta to all three is ~0 — these suppliers pass costs through and had pricing power, and the book *outperformed* in the biggest cost-shock months. |

---

## 5. When it can fail

- **A sustained rates-down, risk-on reversal.** The washed-out clean-energy complex rallies
  and the whole electrification theme de-rates. The overlays soften the path; they do not
  stop a fundamental repricing. This is the single biggest risk and it is not hedged.
- **A data-centre / AI-capex air pocket.** Hyperscaler capex decelerates, or local moratoria
  (Loudoun, Prince William, Central Ohio, North Texas) spread. Equipment backlogs cushion
  the first few quarters; a multi-year slowdown would not be cushioned.
- **The book is the theme.** ~0.86 correlated with VOLT, and essentially all of the return
  is thematic beta — enhanced beta, not alpha. If the theme is wrong, the book is wrong.
- **A 2022-style rate shock hurts twice** — growth multiples compress and the Treasury leg
  of the sleeve loses. The book still drew −20% then.
- **One macro cycle of evidence.** The backtest is 2019–2026, dominated by a single long
  electrification up-leg (2023–26). Regime diversity is limited.
- **Point-in-time universe not yet locked.** The screen's judgment columns carry some
  hindsight until a strict as-of-date version is built — a pre-live gate.

---

## 6. Short offsets: considered, not adopted

We tested **nine** short candidates as overlays on the book. **None improves risk-adjusted
return.** The three most worth discussing are below; the rest are summarised after.

### First, the book on its own

| For | Against |
|---|---|
| Captures the electrification theme at ~0.6 market beta | Still ~0.86 correlated with the theme — no explicit hedge for a theme de-rating |
| Best drawdown of any option here — −23% full, vs −34% to −44% for the ETFs and the market | Concentrated: 27 names, deliberately |
| All the risk control is portable and cheap: no shorting, no borrow, ≤100% gross | One-cycle backtest; point-in-time universe not yet locked |
| Beats every benchmark on Sharpe *and* drawdown in both sub-periods | The reserve sleeve's Treasury leg loses in a rate spike |

### Short industrials (XLI), 0.25× — *broad-beta hedge*

**Why we considered it.** Of all the candidates it "looked good": it cuts the full-period
max drawdown by ~7 points and, in 2023–24, lifted Sharpe to 2.1. The idea is to short a
slice of the industrial sector to cancel the book's general market and cyclical swings and
leave the stock selection.

**Sharpe** 1.05 (book 1.05) · **Total return** +238% (book +348%) · **Max DD** −16% (book
−23%) · **Feared scenario** +2.2% (book +3.1%)

| For | Against |
|---|---|
| Real drawdown reduction: −23% → −16% over the full backtest | Costs ~110 points of compound return (+348% → +238%) |
| Sharpe is flat, not down — 1.05 either way | The 2023–24 Sharpe of 2.1 is regime luck — that is the window the book beat its sector by 2× |
| | A live bet that industrials keep lagging the book; ~125% gross, borrow cost, dividends paid on the short |
| | **A 25% cash / T-bill weight gives the same return and drawdown at a *higher* Sharpe (1.06)** — this is de-levering, not hedging |

### Short China large-cap (FXI), 0.15× — *diversifier*

**Why we considered it.** It is the least-correlated global-industrial cyclical available as
one liquid instrument, and it is the only candidate with a *positive* Sharpe delta (+0.02).

**Sharpe** 1.06 (book 1.05) · **Total return** +326% (book +348%) · **Max DD** −20% (book
−23%) · **Feared scenario** +2.5% (book +3.1%)

| For | Against |
|---|---|
| Cheapest of the lot — only ~22 points of full-period return given up | +0.02 Sharpe is inside the noise — not a real improvement |
| Small drawdown help (~3 points); Sharpe nominally the best of any overlay | Adds a China-decoupling tail: a fat left tail whenever China rallies hard |
| | No exit rule; does not hedge the actual feared scenario (+2.5% vs the book's +3.1% in those months) |

### Short solar / DER (TAN), 0.15× — *thematic mirror, the "natural" offset*

**Why we considered it.** It is the textbook offset: short the names that should *benefit*
from high power prices and green policy — the economic mirror of the long book.

**Sharpe** 0.96 (book 1.05) · **Total return** +259% (book +348%) · **Max DD** −20% (book
−23%) · **Feared scenario** +0.3% (book +3.1%)

| For | Against |
|---|---|
| In 2023–24 the combined Sharpe was 2.8 — superficially the best-looking result in the whole study | That 2.8 *is* the 2023–24 solar crash, not a hedge — in 2025–26 it adds nothing (1.35 vs 1.49) |
| | **It is an anti-hedge in the feared scenario**: book +3.1% → +0.3% in those months, because "theme de-rates while clean energy rallies" is exactly when solar rips |
| | Prior tests at larger size took the book's Sharpe from 0.69 to −0.18 and its drawdown from −24% to −83% |

### The other six, briefly

- **Short SPY 0.25×** — the blunt beta hedge. Sharpe −0.11; same de-levering point as XLI,
  with more tracking error.
- **Short PAVE 0.25×** — Sharpe −0.05, drawdown −7 points; a slightly worse XLI.
- **Short GRID 0.25×** — the purest theme-relative construction. Sharpe −0.15; you are
  shorting your own thesis.
- **Short the merchant generators (VST/NRG/CEG/TLN) 0.15×** — the names literally long the
  power price. Sharpe −0.22 *and* drawdown slightly *worse*: they run +0.6 correlated with
  the book, so this shorts the book's own beta.
- **Short a crowded-adjacents basket 0.15×** — Sharpe −0.18; bleeds borrow and carry.
- **Short data-centre REITs (DLR/EQIX) 0.10×** — the runner-up "insurance" leg. Sharpe
  −0.06, drawdown unchanged; as a *conditional* short (only while real yields rise) it is
  the one variant that helped in 2025–26, at a cost in 2023–24.

![Change in Sharpe and in max drawdown for each short candidate applied to the book](pv3-shorts-bars.png)

*Every candidate, full backtest. Left: all but one reduce the book's Sharpe. Right: most
trim drawdown by 1–7 points — the exception, shorting merchant generators, makes it worse.*

### Verdict — no short offset

No equity short improves the book on a risk-adjusted basis. The candidates that reduce
drawdown do it by **shrinking net exposure** — and a cash or T-bill weight does that more
cheaply, at a higher Sharpe, with no borrow, no dividends paid on a short, and half the
gross exposure.

| Full backtest | Sharpe | Total ret | Max DD |
|---|--:|--:|--:|
| Book, 100% | 1.05 | +348% | −23% |
| Book + short XLI 0.25× | 1.02 | +238% | −16% |
| **75% book + 25% short-term Treasuries** | **1.06** | +240% | −18% |

If the mandate requires a lower drawdown, the sanctioned dial is **75–85% book + short-term
Treasuries**, not a short. Otherwise drawdown control lives in the three built-in overlays —
volatility scaling, the valuation trim, and the 15% reserve sleeve.

---

### Method & caveats

**Window.** Backtest runs Jun 2017 – Aug 2026; all headline statistics and figures use
Jan 2019 onward, before which the screen holds fewer than eight listed names and the series
is a near-empty warm-up.

**Sharpe.** (CAGR − 3.76%) / annualised volatility; 3.76% ≈ the mean 3-month US T-bill over
the backtest. The same risk-free rate is used in both sub-periods, which marginally flatters
the 2023–24 figures.

**Construction.** Book returns are the `screen +val+sleeve` variant of the
`electrification_strategy` package. Overlays and the reserve sleeve are applied as
daily-rebalanced constant weights. Short overlays subtract `w ×` the instrument's daily
return; broad-index candidates at `w = 0.25`, thematic baskets at `w = 0.15`, data-centre
REITs at `w = 0.10`.

**Feared scenario.** The 15 calendar months in which solar (TAN) most outran the market — a
proxy for "the theme de-rates while clean energy rallies."

**Data.** Prices from Yahoo Finance adjusted close. VOLT listed in 2024, so its full-window
statistics are omitted and its fair window is Jan 2025 – Aug 2026.

*Simulated historical performance. Past simulated results are not indicative of future
performance. This document is research, not investment advice.*
