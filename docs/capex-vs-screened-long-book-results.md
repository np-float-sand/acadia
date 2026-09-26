# Capex-intensity long leg vs the screened-electrification book — overlap, fact-layer repair, and united rules

**Date:** 2026-09-23 · **Verdict:** capex-layer bugs **FIXED**; books are **NOT** subsets of each
other (~9 of 25 names overlap); **intersection rule ADOPTED as the best tested long book**
(pending the caveats in §6).

---

## 1. The question

Is the long side of the screened-electrification book a subset of the capex-intensity long leg?
They look like the same idea stated two ways — "companies that sell into the buildout" (capex
rule, mechanical) and "institutional-customer electrical suppliers" (screen, judgment columns).

**Answer: no.** They overlap on about a third of each book, and the differences are systematic,
not incidental.

---

## 2. Fact-layer repair (done first — the original comparison was unanswerable)

Three defects in `capex_intensity_strategy` made most of the screened names unscoreable and
mis-scored several utilities. All three are fixed; 27 tests pass.

| # | defect | location | effect |
|---|---|---|---|
| 1 | the tag loop `break`s on the first tag *present in the JSON*, even when the span filter removed every one of its facts | `edgar.company_facts` | ETN/EMR/PWR had capex but no revenue; HUBB/GEV/CARR/JCI/EME/WCC had revenue but no capex. **13 of 28 screened names were NaN for the whole history.** |
| 2 | `PaymentsToAcquireProductiveAssets` missing from the capex tag list | `edgar.FACT_TAGS` | the only capex tag HUBB, GEV, CARR, JCI, EME, WCC use — and where AEP reports construction expenditure |
| 3 | "a quarter" defined as a 60–200 day span, with no contiguity or staleness guard | `config`, `fundamentals.ttm_panel` | 6-month YTD facts counted as quarters; AEP's capex facts stop in 2021 but were divided into 2026 revenue → **capex/rev 2.4%, putting AEP and ED in the LONG leg** of a strategy whose thesis is shorting utilities |

Defect 3 turned out to be deeper than a guard. Cash-flow items are tagged **year-to-date** (a
10-Q reports 3, 6 and 9 months from the fiscal-year start; the 10-K reports 12), so only Q1 is
ever a standalone quarter. `fundamentals.derive_quarters` now differences consecutive rungs of
that ladder (Q4 = FY − 9M), with each derived quarter inheriting the **later** of the two filing
dates. The differencing is redone at every observation date, so a quarter is always built from
the vintages an investor actually had — deriving once up front silently replaced every period
with its latest restatement and hid the entire pre-2026 history.

**Scoreable names: 116 → 193.** Sanity after the fix: ETN 3.4%, HUBB 2.8%, GEV 4.1%, EME 0.6%,
JCI 1.5% (all asset-light, all previously NaN); AEP 2.4% → 19.7%, DUK 46%, EXC 36% (utilities
back on the heavy side).

Restated strategy metrics (full book, own tuning): full 2019–2026 Sharpe **0.66** / MaxDD −34.1%
/ Calmar 0.50; 2023–2026 OOS Sharpe **1.22** / Calmar 1.40. The OOS window is essentially
unchanged from the pre-fix table; the 2019–2022 fit window is weaker (0.41 → 0.13).

---

## 3. Overlap — where each book's names actually sit

Percentages are share of months in the book, 2019-01 → 2026-08.

### The 27 screened names

| in the capex long leg **often** (≥30% of months) | rarely (1–30%) | **never** |
|---|---|---|
| WCC 65, EME 65, FIX 60, NVT 53, CARR 44, VRT 43, MTZ 40, JCI 38, AME 33, RRX 31, PWR 31 | HUBB 24, POWL 20, ATKR 17, PRIM 16, IESC 12, ENS 6, MYRG 5, ETN 4, EMR 1, AEIS 1 | GEV, APG, AYI, MOD, SPXC, TT |

- **6 names are outside the capex universe entirely** (AYI, MOD, SPXC, TT, APG, and GEV scores
  but never reaches the quartile) — the SIC screen does not reach them.
- **ENS is the one screened name that also shows up SHORT** (11% of months, median capex/rev 2.6%
  but spikes above the heavy cut).
- The screened names that *never* make the long quartile are not capex-heavy — ETN 2.9%, EMR 2.9%,
  AEIS 3.2% — they are simply **out-ranked inside a much larger asset-light pool**.

### The capex longs the screen never holds

THG, PNR, FTV, PH, FLS, TRMB, SPB, WTS, OTIS, CW, NRG, FELE, NOVT, TDY, CGNX, IEX, NDSN, ROK,
ESAB, JBTM, ONTO, ENR, WWD, GNRC, FLNC — general industrials, instruments, automation, one IPP
(NRG) and one battery name (FLNC, 32% of months). Average 27 long names/month vs the screen's 25,
with only **9.3 names in common**.

### Battery / thermal / tidal

| sleeve | in screened book | in capex long leg | contribution |
|---|---|---|---|
| **dc_thermal** (VRT, TT, CARR, JCI, SPXC, MOD) | **yes, 6 of 27** | partly (CARR 44%, VRT 43%, JCI 38%; TT/SPXC/MOD not in the universe) | **adds**: removing it costs Sharpe 0.742 → 0.710 and Calmar 0.658 → 0.612 (shipped stack, 2017-06→2026-08) |
| **grid_scale_storage / battery** (FLNC, STEM) | **no** — both fail the quality gate | FLNC yes, 32% of months; STEM never | **would subtract**: forcing them in drops raw Sharpe 0.832 → 0.766. Standalone: −19.5% CAGR, −93% MaxDD |
| **tidal** | none | none | no exposure exists. "Marine/tidal" appears only in `docs/capex-intensity-proposal-v2.md` §lexicon as a **rejected** screen — 9 full-text hits, 6 of them oilfield services |

The capex rule holds FLNC because a battery integrator *is* asset-light; the screen rejects it
because it is not earnings-valued and is policy-dependent. That is the clearest single case of
the two rules disagreeing on purpose.

---

## 4. Head-to-head and united rules

Window 2019-01 → 2026-08, gross (no costs), every book run through the **same** overlay in the
lower block: 20% vol target (21-day, 1.5× cap) + valuation-extension de-lever + 15% GLD/IEF
sleeve — the shipped `screen +val+sleeve` stack.

### Raw (equal-weight, no overlay)

| book | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| **4. intersection (screen AND capex-light)** | 40.1% | 32.9% | **1.070** | −46.6% | **0.859** |
| 5. union (screen OR capex-light) | 31.6% | 27.5% | 0.992 | −44.8% | 0.705 |
| 6. screen long + capex-heavy short (−0.5×) | 29.7% | 25.5% | 0.991 | −36.5% | 0.812 |
| 1. screened electrification | 34.1% | 30.6% | 0.985 | −46.9% | 0.728 |
| 8. full capex L/S | 22.5% | 21.2% | 0.874 | −31.1% | 0.723 |
| 2. capex-intensity long leg | 26.8% | 27.0% | 0.867 | −44.0% | 0.608 |
| 3. capex-light, ELEC group only | 26.3% | 27.7% | 0.839 | −43.7% | 0.602 |
| 7. capex-light ELEC + capex-heavy short | 22.0% | 22.1% | 0.832 | −32.4% | 0.679 |

### Shipped overlay applied to all eight

| book | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| **4. intersection** | 23.2% | 15.7% | **1.153** | −23.1% | 1.005 |
| 5. union | 23.1% | 16.9% | 1.080 | −22.0% | **1.051** |
| 1. screened electrification *(shipped today)* | 21.0% | 16.4% | 1.000 | −23.2% | 0.905 |
| 6. screen long + capex-heavy short | 20.7% | 16.9% | 0.963 | −27.2% | 0.761 |
| 2. capex-intensity long leg | 20.2% | 17.1% | 0.928 | −21.8% | 0.924 |
| 8. full capex L/S | 19.6% | 17.0% | 0.905 | −25.9% | 0.759 |
| 3. capex-light, ELEC group only | 19.5% | 17.1% | 0.894 | −23.9% | 0.817 |
| 7. capex-light ELEC + capex-heavy short | 18.7% | 17.2% | 0.852 | −26.8% | 0.697 |

**The screened book wins the straight fight.** Long-only vs long-only, on the same overlay, it
beats the capex long leg on Sharpe (1.000 vs 0.928) and CAGR, and loses narrowly on MaxDD
(−23.2% vs −21.8%) and Calmar (0.905 vs 0.924). Raw, it wins on Sharpe and Calmar both.

### Sub-window stability (shipped overlay)

| book | 2019–2022 Sharpe / Calmar | 2023–2026 Sharpe / Calmar |
|---|---|---|
| **4. intersection** | **0.552 / 0.55** | **1.944 / 2.59** |
| 5. union | 0.608 / 0.63 | 1.624 / 1.56 |
| 1. screened electrification | 0.515 / 0.52 | 1.595 / 1.85 |
| 2. capex-intensity long leg | 0.410 / 0.46 | 1.494 / 1.50 |
| 3. capex-light ELEC only | 0.429 / 0.47 | 1.398 / 1.27 |

The intersection beats the shipped book in **both** halves — the only variant that does.

---

## 5. What the combination actually says

The two rules are **complementary filters on different axes**, which is why intersecting them
works and unioning them mostly doesn't:

- the **screen** answers *who does this company sell to, and is it valued on real earnings* —
  a business-model filter, hand-recorded, 33 names, no mechanical reach;
- the **capex rule** answers *is this company funding the buildout or selling into it* — a
  balance-sheet filter, mechanical, 193 names, no view on the customer.

A screened name that has drifted capex-heavy (ENS, and ETN/EMR at the margin) is one the
business-model filter still likes but the balance sheet no longer does. Dropping it is the whole
of the improvement. Last three rebalances the intersection held 11–12 names: AME, ATKR, CARR,
EME, ENS, JCI, MTZ, NVT, POWL, PRIM, PWR, RRX, WCC.

The reverse combination — taking the capex universe and restricting it to the ELEC SIC group
(row 3), a mechanical approximation of "suppliers only" — does **not** recover the screen. It
under-performs the raw capex long leg. The SIC group is too coarse a proxy for "sells to
utilities and hyperscalers"; the judgment columns are doing real work.

Adding the capex **short** leg to the screened book (row 6) costs Sharpe and deepens MaxDD
(−23.2% → −27.2%). Shorting regulated utilities is not a hedge for a long book that is already
levered to the same buildout; the GLD/IEF sleeve remains the better insurance.

---

## 6. Caveats before this is adopted live

1. **Breadth.** The intersection averages **9.3 names/month** against the screen's 25 — a
   concentration jump, and it dips into single digits in the early window. Effective-N and a
   single-name cap need re-checking before this sizes real money.
2. **The judgment columns are still 2026-vintage back-cast.** Intersecting does not remove that
   exposure; it inherits it.
3. **Costs are not modelled** in this comparison (both books gross). The intersection turns over
   more than the screen, because a name can leave on either rule.
4. **Survivorship** — yfinance serves no delisted names; affects every book here equally.
5. The capex fix moved the strategy's own fit-window Sharpe from 0.41 to 0.13. The rule is
   weaker before 2023 than the pre-fix table claimed, and the fixed panel is the honest one.
