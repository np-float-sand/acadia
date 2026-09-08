<!--
  Turn this file into HTML / PDF with tools/md2html.py (from the repo root):

    poetry run python tools/md2html.py docs/electrification-strategy-proposal-v4.md --theme auto --pdf

  -> writes docs/electrification-strategy-proposal-v4.html  (light/dark auto, corner toggle)
     and   docs/electrification-strategy-proposal-v4.pdf    (light-locked, A4)
  Figures (docs/pv3-*.png) are inlined as data URIs, so both files are self-contained.

  Options:  --theme light | --theme dark   lock the theme     --no-toggle   drop the button
            --toc   add a table of contents                    -o PATH      choose the output path
-->
<!--
  Turn this file into HTML / PDF with tools/md2html.py (from the repo root):

    poetry run python tools/md2html.py docs/electrification-strategy-proposal-v3.md --theme auto --pdf

  -> writes docs/electrification-strategy-proposal-v3.html  (light/dark auto, corner toggle)
     and   docs/electrification-strategy-proposal-v3.pdf    (light-locked, A4)
  Figures (docs/pv3-*.png) are inlined as data URIs, so both files are self-contained.
  ...
-->

# Screened Electrification Equipment

**Summary**
The proposed strategy is to take a position in one of the next decade's structural mandates -- the expansion of grid capacity for AI and electrification -- into a systematic book that rewards profitable, well-run companies building that infrastructure and does so with a demonstrably better risk profile than buying the theme passively. 

The first option is a smart-beta long-only one.  While we do avoid borrow costs, the beta to the market is still strong (0.6 compared with VOLT's 1).  The second option is to take a short position in industrials to favorably select an electrification sleeve over general industrials which lowers beta to market (0.37 beta to market) but will incur costs/risks of shorting. 

**Screening**

The book is long roughly **27 companies that sell into the electricity build-out**: power
equipment (transformers, switchgear, power electronics), grid-scale storage, and electrical
engineering & construction. It holds *suppliers* — not utilities, not power generators.
Representative names: Eaton, GE Vernova, Quanta Services, Vertiv, Hubbell, nVent, Fluence,
Primoris.



A company enters the book only if it (a) sells to institutional buyers — utilities,
hyperscalers, industrial customers, not consumers — and (b) meets at least two of three
fundamental tests: **profitable today**, **valued on earnings** rather than on story, and
**low dependence on subsidies or policy**. Names are held in **equal weight**, capped at
25% each. There is no cap-weighting, so the book does not concentrate in a few mega-caps
the way a thematic index does.

**Macro Risks**

The strategy recognizes the kinds of macro risk to which this strategy is exposed.  Eg. a gorwth or credit scare anda currency or geopolitical shock.  To partially offset  these risks we add a (15%) gold/short-Treasure sleeve whose two legs pay off in precisely those states.  One-seventh of the portfolio is permanently held in gold and
  short-term US Treasuries instead of equities, as a shock absorber: a growth or credit
  scare pushes Treasuries up; a currency or geopolitical shock pushes gold up.

To cope with the risk of getting ahead of fundamentals and leaving the book fully invested in a euphoric peak/bubble,  we add a valuation trim that steps exposure down in stages once prices have stretched above their own long-run trend and the market.  After the book has run up well ahead of its own long-run trend and
  the broad market, it steps exposure down in stages — to about 80%, then 60% — taking
  risk off into euphoria.

To cope with risking market turbulance and drawdown regimes amplified by the book's equity beta,  we add a volatility scaling that automatically de-grosses the book when realized volatility climbs and restores it as markets settle, keeping a near  20% volatility target .  The book automatically reduces its invested percentage when markets turn choppy and restores it when they calm, aiming to keep annual volatility near 20%. 

Although this is similar to the VOLT etf, it removes the regulated utilities, midstream etc that are rate-senssitive, slower growing and only loosely tied to the equipment build out.  Roughly 30% of VOLT is regulated utilities. 10% midstream, 13% components.  These would dilute the theme and drag and therefore drag the risk-adjusted return down. Compared with the VOLT (over its live history Jan 2025-Aug 2026) we get similar return (~30%) but with materially higher sharpe (1.49 vs 1.05), a shallower drawdown (-17% vs -23%) and a lower market beta (0.6 vs 1.0).  

An alternative version of this book could take a short position in industrails.  While it has additional costs, it does reduce the beta to market by roughly half , from 0.6 to 0.3, with sharpe essentially unchanged. 

**Performance**

*Full history · Jan 2019 – Aug 2026*

| Strategy | Sharpe | MaxDD | CAGR |
|---|--:|--:|--:|
| **Option A** — book | **1.05** | −23% | +20.8% |
| **Option B** — book + 0.25× XLI short | 1.02 | **−16%** | +16.6% |
| VOLT | — | — | — |
| SPY | 0.71 | −34% | +17.5% |
| SPY + 15% Tsy/gold sleeve | 0.75 | −29% | +15.9% |

*AI period · Jan 2023 – Aug 2026*

| Strategy | Sharpe | MaxDD | CAGR |
|---|--:|--:|--:|
| **Option A** — book | **1.80** | −17% | +31.5% |
| **Option B** — book + 0.25× XLI short | **1.80** | **−13%** | +26.3% |
| VOLT | — | — | — |
| SPY | 1.25 | −19% | +22.6% |
| SPY + 15% Tsy/gold sleeve | 1.32 | −16% | +20.7% |

*VOLT live history · Jan 2025 – Aug 2026*

| Strategy | Sharpe | MaxDD | CAGR |
|---|--:|--:|--:|
| **Option A** — book | **1.49** | −17% | +27.7% |
| **Option B** — book + 0.25× XLI short | 1.43 | **−13%** | +22.1% |
| VOLT | 1.05 | −23% | +30.5% |
| SPY | 0.88 | −19% | +18.9% |
| SPY + 15% Tsy/gold sleeve | 1.01 | −16% | +18.8% |

*Sharpe = (CAGR − 3.76%) / annualised volatility. MaxDD = worst peak-to-trough within the
window. VOLT listed Dec 2024, so it has no full-history or AI-period figures.*

**Total return, full period (Jan 2019 – Aug 2026):** Option A +348% · Option B +238% ·
SPY +243% · SPY + sleeve +222%.

- Option A leads Sharpe in every window and on drawdown against SPY
- Option B matches A's Sharpe in AI period while taking the drawdown to -13-16% and market beta to 0.37.
- The SPY+sleeve row shows that bolting the book's reserve sleeve onto the index buys a smoother ride but nowhere near the book's risk-adjusted return.






---




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


---
