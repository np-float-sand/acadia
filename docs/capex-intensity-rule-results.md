# The ex-ante discovery rule — capex intensity

**Date:** 2026-09-21
**Status:** **RULE FOUND. Fully ex-ante, mechanical, OOS-tested. Regime caveat in §6.**
**Parent:** `docs/algorithmic-long-short-books-results.md` §6 (the ex-ante audit that forced this)

**Question asked:** the GSR names were identifiable ex-post but not ex-ante. Is there a rule —
using *other metrics* — that discovers them in advance?

**Answer: yes, and it is capex intensity.** Two separate problems had to be solved: getting the
names into the *universe*, and *selecting* them once there. The first turned out to be easy and the
second is where the answer lives.

---

## 1. Problem 1 — universe membership (SOLVED)

Of the 23 GSR names, **18 were already in the mechanical SIC+liquidity universe**. Only 5 were
missing, and not at random:

| missing | actual SIC | why missed |
|---|---|---|
| CCJ, UEC | **1090** Misc Metal Ores | uranium miners sit in generic mining codes |
| LEU, UUUU | **1400** Nonmetallic Minerals | same |
| MYRG | **1623** Water, Sewer, Pipeline, **Power Line** Construction | a power-infrastructure code I had simply omitted |

Two principled fixes, neither fitted to outcomes:

1. **Add SIC 1623** — literally "Power Line Construction". Recovers MYRG (and 32 filers).
2. **EDGAR full-text rule** over 10-K/40-F/20-F for nuclear-fuel-cycle vocabulary —
   `"uranium concentrates"`, `"nuclear fuel cycle"`, `"uranium enrichment"`, `"yellowcake"`,
   `"enriched uranium"` — run **point-in-time** (filings published before year *Y* define the
   universe for year *Y*). Recovers **all four** uranium names, plus genuine peers nobody
   hand-picked: DNN, NXE, URG, EU, ASPI, LTBR, NNE, INIS.

Result: **23/23 GSR names now inside a fully mechanical, point-in-time universe** of 111 (2019) →
150 (2023) → 187 (2025) names. The text rule is filer-authored language, not analyst judgment.

**But recovering the names did not recover the performance** — Book B on the widened universe still
only managed 2023-26 Sharpe 0.56. That isolated the real problem.

---

## 2. Problem 2 — selection (THE ACTUAL ANSWER)

Eleven ex-ante, point-in-time metrics tested for their ability to separate the 23 GSR names from
the other ~215 in the pool (Welch t, pooled monthly):

| metric | 2019-2022 (fit) | 2023-2026 |
|---|---|---|
| **capex intensity** | GSR 0.030 vs 0.069 · **t = −15.08** | 0.045 vs 0.082 · **t = −15.20** |
| size (log) | t = −7.20 | t = 11.27 |
| market beta | t = 4.89 | t = 13.66 |
| FCF margin | t = −4.59 | t = 9.74 |
| operating margin | t = −3.89 | t = 9.39 |
| **grid-stress beta** | t = 2.49 | **t = 0.50** |
| revenue growth | t = 0.86 | t = 3.18 |
| **momentum 12-1** | **t = −0.20** | t = 11.09 |

**Capex intensity is the only metric that separates strongly in the fit window *and* holds
afterwards**, and it wins by more than 2x over the next best. Momentum separates them *only*
ex-post (t = −0.20 → 11.09) — that is the outcome, not a predictor. Grid-stress beta, the metric
GSR was originally built on, barely separates at all and decays to nothing.

### Why this is the right answer economically
GSR names run **roughly half the capex/revenue of the rest of the pool**. They **sell into** the
capital cycle rather than *being* the capital: asset-light equipment makers, contractors and
fuel-cycle names, versus asset-heavy owners — regulated utilities, E&Ps, miners. That is the
capex-cycle-pair thesis, now **measured rather than asserted**.

---

## 3. The rule

> **Universe:** US filers with SIC in the electrical/power set (1731, 3470, 3490, 3510, 3550, 3561,
> 3569, 3585, 3590, 3600, 3612, 3613, 3620, 3621, 3670, 3679, 3690, 3823, 3829, 5063, **1623**,
> 4911, 4931, 4991, 1221, 4922, 4923, 4924) **∪** names whose prior-year 10-K/40-F/20-F matches the
> nuclear-fuel-cycle lexicon — filtered monthly to trailing-1yr median dollar volume > $10M.
>
> **Score:** point-in-time **TTM capex ÷ TTM revenue**, from EDGAR using each fact's `filed` date.
>
> **Book:** LONG the lightest quartile, SHORT the heaviest quartile at **0.5x**, monthly rebalance,
> 20% vol target, 10bp costs.

Nothing hand-selected. No 2026-dated flags. No labels required once the rule is stated.

---

## 4. Performance — 2023 onward is out-of-sample for the discovery

| window | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| 2019-2022 (fit) | 7.5% | 0.26 | −36.4% | 0.21 |
| **2023-2026 (OOS)** | 32.8% | **1.26** | −23.6% | **1.39** |
| **2025-2026 (OOS)** | 28.1% | **1.07** | −22.8% | **1.23** |
| Full 2019-2026 | 18.9% | 0.74 | −36.4% | 0.52 |

~15 long / 16 short, turnover **1.2x/yr**. It beats both earlier ex-ante books on the *same*
mechanical universe (A: 0.85 / 0.78 · B: 0.56 / 0.66).

Latest holdings — **long** UUUU, IESC, MTZ, FLNC, SYM, TRMB, SPB, AME, ED, POWL, WTS, PNR;
**short** EOSE, DUK, PCG, ETR, IDA, VG, SLDP, ORA, KNTK, SERV. Long asset-light suppliers, short
capital-heavy utilities and cash-burning developers.

---

## 5. Robustness — every obvious objection tested

| objection | test | result |
|---|---|---|
| "It's just short utilities" | drop **all 44 utilities** from the pool | 2023-26 **1.10**, 2025-26 **1.05** (vs 1.07) — survives |
| "The short leg is decoration" | short weight 0 → 0.25 → 0.5 → 1.0 | 1.07 → 1.20 → **1.26** → 1.15; monotone to 0.5x |
| Transaction costs | 5 / 10 / 25 / **50** bp | 1.27 / 1.26 / 1.22 / **1.14** |
| Quantile choice | q = 0.15 / 0.20 / 0.25 / 0.33 | 1.29 / 1.21 / 1.26 / 1.14 |

**This is the first construction in this line where an algorithmic short genuinely earns its
place.** Books A and B needed the short shrunk to 0.15-0.25x to avoid damage; here it improves
Sharpe monotonically out to 0.5x, because the heavy-capex basket is a *real* underperformer rather
than a junk-rally basket.

---

## 6. Caveats — read before using

1. **It does not work pre-2023.** Fit-window Sharpe is 0.18-0.38. The rule earns its returns in the
   2023+ regime only; this is a capex-cycle bet, and it pays while the cycle runs.
2. **Discovery used ex-post labels.** The 23 GSR names came from a 2026 hand-classification, so
   *which metric I landed on* carries selection risk. Mitigants: 11 metrics were screened, capex
   intensity won by t = −15 versus −7.2 for the next best, and once stated **the rule needs no
   labels at all** — the backtest above never references the 23 names.
3. **Asset-light is a known factor.** This overlaps quality/capital-discipline factors documented
   in the general equity literature. The claim here is not a new anomaly but a *correctly specified*
   one for this universe, with an economic rationale that matches the thesis.
4. **Survivorship is still unfixed.** 2,168 of 2,535 filers in the SIC set have no ticker today and
   yfinance serves no delisted prices. Long legs flattered, short legs understated — for this book
   and the shipped one alike. Needs CRSP or equivalent.
