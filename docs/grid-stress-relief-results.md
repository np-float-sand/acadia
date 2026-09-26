# Grid-Stress Relief (GSR) — a tradable grid-stress equity signal

**Date:** 2026-09-21
**Status:** **DEMOTED 2026-09-21 — static hand-classified basket; edge does not survive mechanisation (§6)**
**Related:** `docs/power-market-signal-root-cause-results.md`, `docs/power-market-extensions-results.md`

**Bar set:** compelling Sharpe *and* Calmar in **2023-2026** and **2025-2026**; alpha or beta
permitted; signal should relate to **grid stress**.

---

## 1. Headline

| book (25% vol target, 10bp costs) | 2017-18 | 2019-22 | **2023-26** | **2025-26** | full 2017-26 |
|---|---|---|---|---|---|
| **GSR EXPANDED (23 names)** — *RF=0* | Sh 0.59 / Cal 0.58 | 0.94 / 0.66 | Sh 1.84 / Cal 1.96 | Sh 1.17 / Cal 1.27 | Sh 1.20 / Cal 0.83 |
| **GSR EXPANDED — RF=4% (comparable)** | — | — | **Sh 1.48 / Cal 1.96** | **Sh 1.00 / Cal 1.27** | Sh 1.06 / Cal 0.87 |
| GSR CORE (14 names) | 0.80 / 0.99 | 0.69 / 0.48 | 2.11 / 2.46 | 1.12 / 1.32 | — |
| SPY | 0.60 / 0.40 | 0.58 / 0.39 | 1.47 / 1.18 | 1.05 / 0.96 | — |
| XLU | 0.62 / 0.51 | 0.45 / 0.30 | 0.44 / 0.36 | 0.47 / 0.59 | — |

EXPANDED detail: 2023-26 **CAGR 48.1%, vol 26.1%, MaxDD −24.5%**; 2025-26 **CAGR 31.2%,
MaxDD −24.5%**. Positive in all four windows, and beats SPY on **both** metrics in **both** target
windows.

> **CORRECTION (same day).** An earlier draft claimed GSR "beats the shipped book's 2023-26 Sharpe
> (1.556 → 1.84)". **That was wrong — an apples-to-oranges error.** The 1.84 used a zero
> risk-free rate while the shipped book's logged figures use `config.RF = 0.04`. Re-running **both**
> through the production `compute_metrics` with RF=4%: **GSR 1.48 vs shipped `screen +val+sleeve`
> 1.60** (2023-26) and **GSR 1.00 vs 1.32** (2025-26). **The shipped book wins on Sharpe in both
> target windows.** GSR's genuine wins are Calmar in 2023-26 (1.96 vs 1.89) and the full sample
> (Sharpe 1.06 vs 0.70, Calmar 0.87 vs 0.62). See `docs/strategy-comparison-2026-09-21.md`.

**EXPANDED is the headline, not CORE** — it is the hindsight-controlled version (§4).

---

## 2. The signal: classify assets by whether they RELIEVE grid stress

Not a timing signal. A **physics-based classification** of the power complex, stated as a thesis
that makes falsifiable predictions:

> A stressed grid needs **firm capacity** and the **means to deliver it**. Assets that supply
> either should capture the rent. Assets that supply neither — intermittent generation, which
> contributes ~nothing at peak stress, and commodity fuel, which is priced away — should not.

| bucket | rationale | EXPANDED members |
|---|---|---|
| **FIRM** | firm clean capacity (nuclear fuel cycle) | CCJ, LEU, UEC, UUUU, BWXT, CW |
| **DISP** | dispatchable generation, incl. distributed backup | VST, NRG, CEG, TLN, AES, PEG, GNRC |
| **DELIV** | delivery / interconnection buildout | ETN, PWR, VRT, HUBB, POWL, AZZ, NVT, ATKR, MYRG, EME |
| *excluded* | intermittent — no peak contribution | FSLR, ENPH, RUN, ARRY, SHLS |
| *excluded* | commodity fuel — priced by futures | gas/oil E&P |

**Construction:** equal weight within bucket, equal weight across the three buckets, monthly
rebalance, 25% trailing vol target (cap 1.5x), 10bp costs. A name enters only once listed.

---

## 3. Why this is a *grid-stress* signal, stated honestly

The ISO data earns its place in **two** ways, and **not** in a third:

1. **It corroborates the classification.** A GSI-beta panel built from ERCOT hourly zonal prices
   (2016-2026, 93,959 hrs) + EIA daily fuel-type generation, regressing each name on grid-stress
   innovations (SPY-controlled, 252d rolling), puts **intermittent solar at the very bottom**
   (FSLR, ENPH, ARRY, SHLS, RUN) and **firm/dispatchable at the top** (VST, CNR, BTU, UEC, NRG,
   CCJ). The physics claim is independently visible in ISO price data.
2. **It supplied the falsification test** (§4).
3. **It does NOT time the book.** Gating exposure on the grid-stress anomaly, its 3-month change,
   net-load tightness, or scarcity-hours is a wash (best case: 2025-26 Calmar 1.36→1.45 while
   2023-26 Sharpe falls 2.22→2.11). Consistent with this session's root-cause finding — public
   aggregate power data does not time equities.

**So: the ISO data selects and validates; it does not time.** Do not describe GSR as a dynamic
grid-stress timing model. It is a grid-stress-*informed* long book.

Note also that the GSI-beta ranking does **not** by itself reproduce GSR membership — price-spike
beta favours incumbent generators (coal, utilities) and puts VRT/PWR near the bottom, because
acute scarcity spikes reward existing generation while the buildout complex responds to structural
*investment*. **The classification is theory-driven, not data-derived.** That is the main
intellectual liability and is stated plainly here rather than glossed.

---

## 4. Controls

**Hindsight control (the important one).** Adding **9 members chosen for category fit, not
performance** (BWXT, CW, AES, PEG, GNRC, NVT, ATKR, MYRG, EME) moves 2023-26 Sharpe 2.11→1.84 and
2025-26 **1.12→1.17**. Adding new listings too (GEV, SMR, OKLO, NNE) gives 1.96 / 1.03. The result
is not an artifact of a particular 14 names.

**Negative control 1 — intermittent solar (predicted loser):**
Sharpe **+1.46 (2017-18), +1.39 (2019-22), −0.29 (2023-26), +0.16 (2025-26)**.
Solar was the *best* performing bucket before grid stress became binding and the *worst* after.
The thesis made a non-obvious, regime-specific prediction and it landed.

**Negative control 2 — regulated utilities (predicted non-participant):** 2023-26 Sharpe 0.39.
Regulated rate-base earnings don't capture scarcity rent, as claimed.

**Parameter robustness (CORE):** vol target 20/25/30%/none → 2023-26 Sharpe 2.16/2.22/2.24/1.76;
costs 5/10/25/50bp → 2.24/2.22/2.19/2.14; rebalance 5/21/63d → 2.22/2.22/2.21. Turnover is low
(static membership), so cost sensitivity is minimal.

---

## 5. Caveats — read before sizing

1. **This is beta, not alpha.** Correlation to SPY 0.60, to the grid-equipment basket **0.82**.
   It is a better-constructed expression of a theme, not a market-neutral edge. (Permitted by the
   mandate, but do not mis-sell it.)
2. **Grid equipment alone beats it in both target windows** (2023-26 Sharpe 2.44/Calmar 2.79;
   2025-26 1.48/1.66). GSR's case is **regime robustness**, not maximum recent return — DELIV
   alone is −0.36 Sharpe in 2017-18 vs GSR's +0.59. Anyone optimising only for 2023-26 should
   just buy the equipment names and accept the concentration.
3. **Not all-weather.** 2017-18 Sharpe 0.59 and 2019-22 0.94 are unremarkable; full-sample
   MaxDD is **−36.5%**. The strong numbers are concentrated in the grid-stress regime.
4. **Hindsight is reduced, not eliminated.** The buckets were specified from physics, and the
   expansion + negative controls are real tests — but the categories were written down in 2026
   with knowledge of 2023-26 outcomes. The honest out-of-sample test is forward.
5. **Regime dependence is the core risk.** The thesis pays only while firm capacity is scarce.
   If large-load growth stalls or capacity additions catch up, the classification stops binding —
   and the intermittent bucket's 2017-22 record shows how completely these rankings can invert.


---

## 6. DEMOTION — GSR is a static book, and that is disqualifying as a *signal*

**The membership never changes.** Over 2017-2026 there are **zero** rule-driven entries or exits;
names appear only when they list (VST 2016, VRT 2018, CEG 2022, TLN 2023). The only dynamic element
is the 25% vol target — generic risk management, not a signal. **Turnover is 0.0x/yr.**

Two claims in §4 were conflated and must be separated:
- The expansion control shows the result is robust to **which** names are in the buckets. That
  part stands.
- It says **nothing** about whether any rule would have **found** those names in 2017. There is no
  such rule. The list was written down in 2026.

### The test that settles it
Replacing the hand classification with a **mechanical, point-in-time** rule — rank the same
candidate universe by rolling GSI-beta (252d, SPY-controlled, 5d lag), long top quartile, monthly,
same 25% vol target and costs — collapses the result (all at RF=4%):

| construction | 2023-26 Sh/Cal | 2025-26 Sh/Cal | full Sh/Cal | turnover |
|---|---|---|---|---|
| GSR **static**, hand-classified | 1.48 / 1.96 | 1.00 / 1.27 | 1.06 / 0.87 | **0.0x** |
| GSI-beta **dynamic**, rule-derived | **1.18 / 1.70** | **0.64 / 0.89** | 0.70 / 0.42 | 4.0x |
| shipped `screen +val+sleeve` (rule-derived) | **1.56** / 1.82 | **1.30** / 1.57 | 0.69 / 0.60 | — |

Sharpe falls **1.48 → 1.18** and **1.00 → 0.64** on mechanisation. **GSR's advantage is the
hand-classification** — i.e. the hindsight — not the grid-stress thesis.

### Why GSR cannot be fully mechanised at all
Its strongest bucket, **DELIV (grid equipment), owns no generation assets**. No power-market
dataset — EIA Form 860 capacity, Form 923 generation, ISO prices — can identify ETN, PWR or VRT as
grid-stress exposed, because their exposure is to what they *sell*, not what they *own*. FIRM and
DISP could in principle be derived from point-in-time EIA capacity; DELIV cannot. Any mechanical
version therefore loses the bucket that drove the returns.

**This is already solved by the shipped book.** The capex-cycle-pair classifier
("institutional customer + ≥2 of {profitable, earnings-valued, low-policy-dependence}") is a rule
applicable at any date, which is why the log records it as the one KEEPER — it removes the
"you hand-picked winners" objection that GSR reintroduces.

### Status
**Not a signal. Not a candidate to replace the shipped book.** At most a *descriptive* finding:
grid-stress physics explains *why* the theme's sub-sectors diverged after 2023 (and the intermittent
bucket's +1.46 → −0.29 flip is genuine supporting evidence for the mechanism). Keep it as an
explanation, not a portfolio. The shipped `screen +val+sleeve+short` remains the recommendation.
