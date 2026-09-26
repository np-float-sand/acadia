# Algorithmic long/short versions of both books

**Date:** 2026-09-21
**Goal:** versions of GSR-Expanded and the shipped Screened Electrification book with a long **and**
short component, both **algorithmically determined** (no hand-picked membership).
**Status:** **A = SURVIVES the ex-ante audit (degraded: Sharpe ~0.8 both windows). B = FAILS — thesis inverts (§4) AND fails ex-ante (§6).**

Motivation: GSR as first built was a **static hand-classified basket** (0.0x turnover, zero
rule-driven membership changes) — see `docs/grid-stress-relief-results.md` §6. This rebuilds both
books so membership is derived by rule at each date.

---

## 1. Inputs — all point-in-time

| input | source | look-ahead control |
|---|---|---|
| Fundamentals | **EDGAR companyfacts**, 55,272 quarterly facts, 114 tickers | uses each fact's **`filed`** date; TTM = last 4 quarter-ends *known at that date* |
| Grid stress | ERCOT hourly zonal prices (2016-2026) + EIA daily fuel-type generation | rolling 252d beta, SPY-controlled, 5-day lag |
| Universe | 115 names: 47 electrification/industrial + 57 power complex | curated **by category, not performance** (from the classifier's own candidate list) |

This replaces two look-ahead inputs in the shipped book: `profitable_2026` is a **2026-dated static
flag** applied across all history, and the 33-name seed pool passes **31 of 33** names — the screen
barely discriminates.

---

## 2. Book A — Screened Electrification, algorithmic L/S

**Rule.** Mechanises the capex-cycle-pair classifier's own operational test
(`research/capex_cycle_pair_classifier/long_screen.py`: PE>0, operating margin>0, FCF>0) as a
point-in-time filter. **LONG** = names passing TTM operating margin >0 **and** net margin >0 **and**
FCF margin >0. **SHORT** = names failing any. Short sized at **0.25x**, 20% vol target, 10bp costs,
monthly.

| window | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| 2019-2022 | 3.9% | 21.5% | 0.10 | −31.9% | 0.12 |
| **2023-2026** | 33.1% | 21.1% | **1.27** | −19.0% | **1.74** |
| **2025-2026** | 23.4% | 21.3% | **0.91** | −19.0% | **1.24** |
| Full 2019-2026 | 17.0% | 21.3% | 0.66 | −31.9% | 0.53 |

Avg **20 long / 8 short**, turnover **1.0x/yr**. Latest: long ACM, AME, ANET, APG, APH, AZZ, CIEN,
CW, DOV, FIX, GGG, HEI, HWM, IESC, J, MOD, MTZ, MWA, NDSN, NVT, PH, PNR, POWL, ROK, STRL, TDG, TRN,
WAB; short ATKR, COHR, FELE, HXL, SHLS. **Holdings are economically coherent** — long cash-generative
industrials, short the unprofitable/marginal. Cost-robust (25bp → Sharpe 1.23 / 0.87).

**The short leg is small for a reason.** At full weight it destroys the book (2023-26 Sharpe
1.25 → **−0.03**): the low-quality basket *outperformed* the high-quality one (+47.8% vs +37.4%
CAGR in 2023-26) — a junk rally inside a thematic bull. Only a 0.15–0.25x short adds value, and it
adds it to **Calmar** (1.53 → 1.74) more than Sharpe.

---

## 3. Book B — Grid-Stress Relief, algorithmic L/S

**Rule.** LONG top-quartile / SHORT bottom-quartile of the 57-name power complex by **rolling
grid-stress beta**, short at 0.25x, same overlays.

| window | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| 2019-2022 | 20.2% | 21.1% | 0.79 | −28.2% | 0.71 |
| **2023-2026** | 36.8% | 21.7% | **1.37** | −15.0% | **2.45** |
| **2025-2026** | 32.2% | 23.2% | **1.15** | −15.0% | **2.15** |
| Full 2019-2026 | 27.9% | 21.4% | 1.07 | −28.2% | 0.99 |

Avg **17 long / 12 short**, turnover 4.0x/yr. On the numbers alone this **beats the static
hand-picked GSR** (2023-26 Calmar 2.45 vs 1.96; 2025-26 Sharpe 1.15 vs 1.00 **and** Calmar 2.15 vs
1.27) — i.e. making it algorithmic *helped*.

---

## 4. Why Book B must NOT ship as a grid-stress strategy

Its economic content **inverts**. Mean percentile rank of each thesis bucket under the signal
(1.00 = most grid-stress exposed):

| thesis bucket | 2023-24 | 2025-26 |
|---|---|---|
| FIRM — uranium / nuclear fuel | **0.70** ✓ | **0.30** ✗ |
| DISP — merchant generation | **0.63** ✓ | **0.27** ✗ |
| INTERMITTENT — solar | **0.09** ✓ | **0.62** ✗ |

In 2023-24 the signal reproduced the physics exactly: firm and dispatchable high, intermittent at
the very bottom. **In 2025-26 it says solar is the most grid-stress-exposed sector and uranium /
merchant generation the least** — the precise opposite of the thesis the book is named after. Latest
holdings make this concrete: **long** AES, AMR, AR, BTU, CNR, CNX, DVN, FANG, FSLR, MUR, OXY, TRGP
(oil, gas and coal producers, plus solar); **short** BWXT, CCJ, LEU, UUUU, VST, NRG, TLN (the entire
firm-capacity complex GSR exists to own).

It still earns Sharpe 1.15 / Calmar 2.15 in that window. **So its recent returns are unexplained by
its stated mechanism.** A rolling return-beta to a stress index is a *statistical rotation signal*,
not an implementation of grid physics; it tracks whatever recently co-moved with the index. That is
the same failure shape as the Layer-2 congestion "common-mode" edge and the geographically
implausible miner-hub selections — performance without mechanism.

**Verdict: do not ship B as "Grid-Stress Relief".** Either ship it under an honest name
(*grid-stress-beta rotation*) with the inversion disclosed, or rebuild the long leg from
**measured point-in-time firm capacity** (EIA Form 860 dispatchable MW per operator) rather than
return-betas. The second is the thesis-faithful fix and is untried — with the known limitation that
the DELIV bucket (grid equipment) owns no generation and cannot be derived from any power dataset.

---

## 5. Where this leaves the comparison

| book | 2023-26 Sh / Cal | 2025-26 Sh / Cal | membership |
|---|---|---|---|
| Shipped `screen +val+sleeve+short` | **1.54** / 2.31 | **1.46** / **2.32** | rule + 2026-dated flags |
| Book A — algorithmic L/S | 1.27 / 1.74 | 0.91 / 1.24 | **fully point-in-time** |
| Book B — algorithmic L/S | 1.37 / **2.45** | 1.15 / 2.15 | fully point-in-time, **thesis inverts** |
| GSR static | 1.48 / 1.96 | 1.00 / 1.27 | hand-picked, 0.0x turnover |

The shipped book still wins on Sharpe in both windows. **Book A is the honest algorithmic
alternative**: lower headline numbers, but no hand-picked membership, no 2026-dated flags, coherent
holdings, 1.0x turnover, and it survives 25bp costs.

---

## 6. EX-ANTE AUDIT — is membership discoverable in advance, or only in hindsight?

The scores in §2–3 are point-in-time. The **candidate universe was not** — it was assembled in 2026
from a theme-curated list. This section replaces it with a mechanical rule and re-runs both books.

### 6.1 How bad was the selection?
Enumerating every 10-K filer in the relevant SIC codes via EDGAR:

| | count |
|---|---|
| Total filers, initial SIC set | **3,560** |
| Still listed today | 515 |
| **No ticker today** (delisted / acquired / private) | **3,045** |
| Of the listed, in my hand-assembled universe | **57** |
| Listed but missed | **458** |

My universe captured **11% of currently-listed filers**. (Caveat: broad codes — 3674 semiconductors
at 505 filers, 1311 oil & gas at 900, 1040 metal mining at 523 — are off-thesis, and many of the
3,045 are shells or long-dead microcaps, so neither number should be read as a pure indictment.)

### 6.2 A mechanical, ex-ante universe
**Rule:** all US 10-K filers with SIC in the electrical/power set
(ELEC: 1731, 3470, 3490, 3510, 3550, 3561, 3569, 3585, 3590, 3600, 3612, 3613, 3620, 3621, 3670,
3679, 3690, 3823, 3829, 5063 · POWER: 4911, 4931, 4991, 1221, 4922, 4923, 4924), **liquidity-filtered
each month** on trailing-1yr median dollar volume > $10M. Nothing hand-selected.
Corrected set: **2,535 filers, 367 listed**; universe size **106 (2019) → 137 (2023) → 170 (2025)**.

It admits names no hand-picker would have chosen — **BLNK, CHPT, FCEL, BLDP, EOSE, ENVX, BEEM,
AMPX** (EV charging, fuel cells, batteries), many of which collapsed. That is the point.

*A first attempt used a narrower SIC list that wrongly excluded 3590/3600/3670/3679 — i.e. ETN,
EMR, GEV, HUBB, VRT, the core grid-equipment names — and produced an unfairly harsh 0.78 / 0.70.
Those figures are superseded by the corrected set below.*

### 6.3 Result
| book | window | hand-assembled | **ex-ante universe** |
|---|---|---|---|
| **A** | 2023-2026 | Sh 1.27 / Cal 1.74 | **Sh 0.85 / Cal 0.93** |
| **A** | 2025-2026 | Sh 0.91 / Cal 1.24 | **Sh 0.78 / Cal 0.88** |
| **A** | 2019-2022 | Sh 0.10 | **Sh 0.28** |
| **A** | full 2019-2026 | Sh 0.66 | **Sh 0.56 / Cal 0.52** |
| **B** | 2023-2026 | Sh 1.37 / Cal 2.45 | **Sh 0.70 / Cal 0.83** |
| **B** | 2025-2026 | Sh 1.15 / Cal 2.15 | **Sh 0.71 / Cal 1.18** |
| **B** | 2019-2022 | Sh 0.79 | **Sh −0.70** |
| **B** | full 2019-2026 | Sh 1.07 | **Sh −0.05 / Cal 0.01** |

A: avg 17 long / 9 short, 0.9x turnover. B: avg 10 long / 8 short, 4.9x turnover.

### 6.4 Verdict
**Roughly one third to one half of measured performance was ex-post universe selection.**

- **Book A SURVIVES, degraded.** Sharpe ~0.8 in both target windows, **positive in all four
  windows**, full-sample 0.56, low turnover. Its edge is smaller than first reported but real and
  genuinely discoverable in advance.
- **Book B FAILS.** Full-sample Sharpe **−0.05**, 2019-2022 **−0.70**. It works only in the recent
  window. Combined with the thesis inversion in §4 (the signal flips to long-fossil / short-firm in
  2025-26), B should not ship in any form.

### 6.5 The limitation I cannot fix
**Survivorship.** 2,168 of 2,535 filers in the corrected SIC set have no ticker today, and yfinance
serves no prices for delisted names. **Every backtest in this repo — these books and the shipped
book alike — runs on survivors only.** Direction of bias: the long leg is flattered (failed
companies absent), while the short leg is *understated* (bankruptcies would have been profitable
shorts). Removing it needs a point-in-time price database with delisted securities (CRSP or
equivalent). Until then, treat all absolute levels here as optimistic.
