# Three extensions to the root-cause finding — results

**Date:** 2026-09-21
**Parent:** `docs/power-market-signal-root-cause-results.md`
**Status:** Ext 1 **FAILED** · Ext 2 **PROMISING, NOT ADOPTED (watch-list)** · Ext 3 **FAILED**

The parent doc established that **energy futures efficiently price public power/gas
fundamentals**, so aggregate public data cannot predict power equities. Its constructive
implication was to look **where no futures market prices the information**. That produced three
ranked extensions, tested here in order.

---

## Extension 1 — Nodal basis for a named fleet: **FAILED**

**Thesis.** Hub prices have liquid futures; individual pnodes do not. A generator's realised
capture price versus the hub is company-specific and plausibly unpriced.

### 1a. True nodal is INFEASIBLE with free data
ERCOT resource-node day-ahead LMPs (report 12331, 1,104 nodes) exist only in a **~30-day rolling
window**: 2026-09-19/-18/-14/-07, 2026-08-31 and 2026-08-22 all return data; **2026-08-15 and
every older date tested return `NoDataFoundException`**. No public bulk archive. This is the same
wall the log already hit on ERCOT CRR (rolling ~12mo) and pre-2024 PJM nodal LMP.

> **Note the irony, and take it seriously:** the information is plausibly unpriced *partly
> because it is not publicly archived*. "Unpriced" and "unobtainable" are strongly correlated
> here. Any future plan resting on free nodal history should be treated as blocked until a paid
> or special-request feed is secured.

### 1b. Zonal fallback — built and tested
- ERCOT day-ahead zonal/hub prices, **2016-2026, 93,959 hourly obs x 15 locations**
  (`get_dam_spp`, ~7s/yr).
- EIA generator inventory via `gridstatus.EIA.get_generators()` — monthly point-in-time
  snapshots **with Latitude/Longitude**, 414k rows, 23,476 ERCOT.
- Thermal (gas/coal/nuclear/oil) plants mapped to ERCOT load zones by coordinates, then
  capacity-weighted per owner, point-in-time per snapshot.

Fleet map is economically sensible and the two names genuinely differ:

| company | ERCOT thermal | zone split |
|---|---|---|
| VST (Luminant/LaFrontera) | 13,809 MW | NORTH 10,320 · WEST 1,779 · SOUTH 1,710 |
| NRG (NRG Texas Power) | 8,275 MW | HOUSTON 6,615 · NORTH 1,660 |

Mean capture-vs-hub basis: VST **+1.29** $/MWh (sd 2.18), NRG **+1.71** (sd 3.99), n=3,184 days.

**Result — no usable signal.** Forward return regressed on basis level and basis change, HAC,
controlling SPY and the hub price change:

| | 2019-22 | 2023-26 | 2025-26 |
|---|---|---|---|
| VST basis level (21d/63d) | −0.42 / −0.34 | −0.10 / +1.11 | −0.83 / +0.72 |
| VST basis chg (21d/63d) | −1.24 / +0.43 | −0.84 / −0.32 | +0.05 / +0.54 |
| NRG basis level (21d/63d) | −1.37 / −0.24 | −0.21 / −0.14 | +0.73 / −0.25 |
| **NRG basis chg (21d/63d)** | **−1.84 / −2.50** | **−1.84 / −0.76** | **−1.49 / −1.90** |

NRG's basis-change is the only consistently-signed cell — but the sign is **negative**, i.e. the
*opposite* of the thesis (better relative capture predicting worse returns), it is one name of
two, and |t| is mostly below 2. Reported as a curiosity, not a pass. **FAILED.**

---

## Extension 2 — Plant-level generation nowcast: **PROMISING, NOT ADOPTED**

**Thesis.** EIA Form 923 monthly plant-level generation, aggregated to owner, is a within-quarter
nowcast of fleet utilisation. Test against *revenue/output*, not EPS — the log's Probe A showed
hedge MtM severs the price->operating-income link.

**Built.** 86,181 rows of monthly plant-level generation 2018-01 -> 2026-06, joined to owners
(VST/NRG/CEG/TLN) via the EIA generator inventory; 102 months coverage each. Magnitudes sanity-check
(CEG ~50 TWh/mo — largest US nuclear fleet; VST ~29; NRG ~17; TLN ~11).
**Publication lag modelled strictly**: month *M* treated as public only at **end of month M+2**;
signal = last 3 *published* months vs the same 3 months a year earlier. 88 usable earnings events.

### The finding that reframes Probe A
| relationship | result |
|---|---|
| nowcast -> **EPS surprise** | t=0.88 (pooled) — **no link**, as Probe A predicted |
| **EPS surprise -> announcement return** | t=1.17, **R²=0.016** — *EPS barely moves these stocks* |
| nowcast -> **announcement return** | **t=2.57** pooled, R²=0.071 |

For IPPs the market reacts to **physical output**, not accounting EPS. Probe A's failure was a
**wrong dependent variable**, not a dead mechanism. Sign is correct (more generation -> better
reaction) and it survives a momentum control (t=2.54).

### Robustness — where it weakens
| check | result |
|---|---|
| holdout 2023-2026 | t=**2.81**, R²=0.128 |
| 2025-2026 | t=**3.21**, R²=0.284 |
| within-quarter cross-section | t=2.04 |
| terciles (low/mid/high) | −1.89% / +0.38% / **+1.37%** — monotone |
| **train 2018-2022** | **t=0.22** — no effect; fails the regime split |
| **excluding NRG** | **t=0.59** — largely a single-name result |
| high-minus-low spread | +3.26%/event but **t=1.58, p=0.121** — not significant |

### Tradable version (quarterly, long top-half / short bottom-half of the 4 IPPs, 3-day hold)
| window | quarters | mean/event | total | hit | t |
|---|---|---|---|---|---|
| 2018-2022 | 4 | **−2.84%** | −11.4% | 25% | −0.68 |
| **2023-2026** | 15 | **+2.86%** | **+42.8%** | 67% | 1.64 |
| **2025-2026** | 7 | **+1.65%** | **+11.6%** | 71% | 0.58 |

**Verdict: NOT ADOPTED.** It *is* profitable in both requested windows, but (1) it fails the
2018-2022 regime split, (2) ex-NRG it collapses to t=0.59, (3) neither tradable window is
statistically significant (t=1.64 / 0.58), (4) capacity is tiny — 4 names, ~2 long / 2 short,
4 trades a year, and 2026Q1 alone was −13.6%.

**Unresolved look-ahead risk:** the EIA API serves *revised* generation values, not point-in-time
vintages. Form 923 is revised after first publication, so some of the measured edge may not have
been available live. Quantifying this needs a vintage archive and was not done.

**Watch-list, re-check quarterly** — same status as the Fold-B narrative result. Genuinely the
most promising finding in this whole line, on the strength of the EPS-vs-output reframe.

---

## Extension 3 — Volatility rather than direction: **FAILED**

**Thesis.** Returns are unpredictable here; variance generally is not.

**True options test is blocked** — no historical implied-vol data is wired into this project, and
yfinance exposes only current chains, no history. Tested the free half instead: does power-market
stress predict **forward 21d realised volatility** of power-equity baskets, *controlling for
volatility's own lag and VIX* (the control that makes or breaks such a test)?

Baseline R² from own-lag + VIX alone: ipp 0.307 · gas_ep 0.544 · utility 0.317 · gridequip 0.197 ·
uranium 0.255 — volatility is already largely self-predicted.

Incremental t-stats on power-market stress (2019-22 / 2023-26 / 2025-26):

| target | ERCOT price vol | ERCOT price dispersion | gas vol |
|---|---|---|---|
| ipp | +0.29 / −0.38 / +1.81 | **−2.51 / −2.33 / −0.83** | −1.33 / −0.31 / −0.14 |
| gas_ep | −0.11 / +0.85 / +2.02 | +0.51 / −1.59 / −0.33 | −1.96 / +1.37 / +1.17 |
| utility | −2.49 / +1.94 / +2.62 | −1.76 / +0.66 / +0.37 | −1.42 / +0.12 / +0.43 |
| gridequip | −0.12 / +0.36 / +1.74 | −0.83 / −1.43 / −0.91 | −0.39 / +0.51 / +0.94 |
| uranium | −0.77 / −2.34 / +1.75 | −0.96 / −1.60 / +2.60 | +0.97 / −1.66 / +0.98 |

Signs flip across windows almost everywhere (utility: −2.49 -> +1.94 -> +2.62). The only
semi-consistent cell — ERCOT price dispersion -> IPP vol — is **wrong-signed** (more power-price
dispersion predicting *lower* equity volatility). **FAILED.** A real test of the variance-risk-premium
idea requires purchased historical options data and remains genuinely untested.

---

## Net conclusion

Of the three, only Extension 2 produced anything, and its contribution is conceptual rather than
tradable: **for merchant generators, physical output predicts the earnings reaction while EPS
surprise does not** (R²=0.016). That corrects a mis-specification in Probe A. It does not yet
clear a regime split or a significance bar, and it is capacity-constrained to ~4 names.

Extensions 1 and 3 are closed. Extension 1's nodal branch is blocked on data availability, not on
hypothesis quality — reopen only with a paid/special-request nodal feed. Extension 3 is blocked on
options data.

---

## Follow-on (same day): does the Ext 2 reframe rescue the Dispatch L/S? — **NO**

Ext 2's actionable claim was: *merchant-generator work should use output/volume, not EPS.* The
cheapest test is whether a generation-volume feature rescues the already-built **Dispatch**
cross-sectional L/S, which FAILED its regime split on 2026-09-08 (ex-2024 Sharpe −0.05).

**Built.** EIA Form 923 plant-level monthly generation for **all 23 Dispatch names**, via an
operator→ticker map over the EIA generator inventory (**1,693 plants**). The map validates against
real corporate structure — EXC shows only **233 MW** (Exelon spun generation to CEG in Feb 2022),
ES **70 MW** and ED **850 MW** (both T&D-only), while DUK 59.6 GW / SO 51.3 GW / CEG 30.4 GW are
right-sized. Same strict publication lag (month *M* public at end of *M+2*); feature = last 3
published months vs the same 3 a year earlier, cross-sectionally z-scored per month. 100% coverage
across 1,288 name-months.

### Univariate — no monthly predictive power, and merchants are wrong-signed
| subset | full 2022-26 | ex-2024 | 2023-26 | 2025-26 |
|---|---|---|---|---|
| ALL 23 | IC +0.019 (t 0.68) | +0.013 (0.42) | +0.007 (0.22) | −0.009 (−0.20) |
| **MERCHANT (4)** | **−0.053 (−0.78)** | −0.051 (−0.67) | **−0.106 (−1.41)** | −0.049 (−0.43) |
| REGULATED (19) | +0.030 (0.98) | +0.027 (0.77) | +0.022 (0.65) | +0.012 (0.23) |

The merchant subset — the only place the mechanism should live — carries the **opposite sign** to
Ext 2's event-window result.

### Portfolio — adding the feature does not help
Walk-forward GBM, monthly, long-top/short-bottom tercile:

| variant | full 2023-26 | ex-2024 | 2025-26 | 2026 |
|---|---|---|---|---|
| BASE (`edgar_z, icr_z, stress_beta_z, mom_z`) | Sharpe **1.06** | 0.67 | **0.45** | +6.0%/yr |
| BASE + `gen_z` | 0.82 | 0.83 | 0.26 | **−7.4%/yr** |
| `gen_z` alone | −0.04 | 0.18 | 0.48 | −10.9%/yr |

Hurts full-sample and the recent window; helps only the ex-2024 cut (n=26 months, within noise).

> **Methodology caveat, stated explicitly:** this probe's BASE replication gets ex-2024 Sharpe
> **0.67**, versus the **−0.05** recorded for Dispatch v2 on 2026-09-08. Different panel start,
> train window and labelling — so **absolute levels here are NOT cross-comparable to the logged
> Dispatch numbers**. The verdict rests only on the **within-probe** comparison (BASE vs
> BASE+`gen_z`, identical implementation), which is unambiguous. Same caveat shape as the
> merchant-fundamental-beta probe.

### Verdict and what it means
**The reframe does not generalise.** Ext 2's effect is an **event-window** phenomenon — roughly
3 days, 4 times a year, when output information is actually revealed and reacted to. Diluting it
into monthly forward returns across *all* months, most of which contain no earnings event at all,
washes it out entirely. That is a **scope limitation, not a contradiction**: both results can be
true at once.

So the corrected guidance is narrower than the 2026-09-21 §2 note first implied:
**"use output, not EPS" applies to event-window studies of merchant generators. It does NOT make
generation volume a monthly cross-sectional factor, and it does not rescue Dispatch.**
