# Strategy note — rules-based electrification long/short (and why long-only is the core)

**Date:** 2026-09-05
**Status:** design note. Not built as a module. Supersedes the ad-hoc `grid_equipment_basket`
long/short analysis for the *strategy* question (`docs/handoff_2026-09-01-grid-buildout-long-short.md`).
**Backtest code:** `scratchpad/xsec/ls_strategy.py`, `mom_stress.py`, `pairs.py` (throwaway).

---

## 1. TL;DR

- **~20 attempts** across this project to find a data-driven *signal* to time or cross-sectionally
  weight this theme have all failed an honest out-of-sample / in-sample-out-of-sample split
  (`docs/hhub-forward-gas-signal-results.md`, `docs/category-demand-rotation-results.md`,
  `docs/pjm-large-load-vintages-2026-09-03.md`, and the D/C/E commitment-signal thread).
- What *is* reproducible and rules-based: a **long-only wide electrification-equipment/EPC basket**,
  universe defined by ETF-holdings membership, equal-weight + cap, 20 % vol target, quarterly
  reconstitution. Through-cycle Sharpe **~0.82** (2013→2026), **1.37** (2023–26), **1.08** (2025–26),
  positive in every sub-window, ~−48 % worst drawdown. It is honest enhanced beta (SPY β ~1.2–1.4).
- The **long/short** version (short the residential-solar / EV-charging / behind-the-meter-storage
  sleeve, or short TAN) **only adds Sharpe in 2023–26** — the post-2022-rate-shock + AI-capex window.
  In 2013–18 and 2019–22 the short leg is a drag (spread Sharpe −0.3 to +0.3 vs long-only +0.3 to
  +0.75). It is a **conditional overlay**, not the core.
- The self-referential trend filter ("hold the spread only while its own trailing-12m return > 0")
  cut full-period drawdown from −67 % to −25 % but **parks in cash for three multi-hundred-day
  stretches** (717 d, 432 d, 286 d) — not acceptable as a standalone rule.

---

## 2. The enduring hypothesis — why long suppliers / short DER could persist

The two legs are not "grid stocks vs solar stocks." They are:

| | Long leg (grid T&D equipment + electrical EPC) | Short leg (residential solar + EV charging + BTM storage) |
|---|---|---|
| **Buyer** | Regulated utilities, hyperscalers, industrial reshoring | Households & small businesses |
| **Purchase type** | A physical necessity — you cannot add data-center / EV / electrification load or interconnect *any* generation (including utility-scale solar) without transformers, switchgear, conductor, substations | A discretionary elective upgrade |
| **Funding** | Rate base (utilities *earn a regulated return on capital deployed* — a structural incentive to spend) or hyperscaler free cash flow | Consumer credit / 20-year solar loans / leases |
| **Rate sensitivity** | Low, arguably *inverted* — a bigger rate base is better for utilities; equipment cash flows are short-duration | High and adverse — long-duration consumer cash-flow assets bought on financing; demand falls as rates rise |
| **Policy sensitivity** | Minimal — grid-reliability spend happens regardless of the ITC or administration | Existential — net-metering reform (California NEM 3.0 cut resi-solar demand ~40 %), ITC step-downs, tariffs; 10-Ks lead with these as risk factors |
| **Unit economics** | Profitable, positive FCF, supply-constrained (transformer lead times 2–4 yr, skilled-labour shortage) → pricing power | Commoditised panels/inverters, thin-to-negative margins, capital-markets-dependent |

**The durable claim:** this is a **quality / profitability / low-policy-dependence tilt expressed
within one theme** — long the derived, non-cyclical, rate-base-incentivised demand for the physical
grid; short the pro-cyclical, rate-suppressed, subsidy-gated consumer slice. Quality-minus-junk is a
documented persistent factor; here it happens to line up with a sector boundary.

**The honest limit:** the *magnitude* 2023–26 (spread Sharpe ~1.9) is inflated by a one-time 2022
rate shock that crushed long-duration consumer clean-tech plus a simultaneous AI-capex boom. The
structural hypothesis justifies a **modest positive expected spread through the cycle (~0.3–0.5
Sharpe), lumpy and regime-amplified** — *not* a persistent ~2.0. In a sustained falling-rate / risk-on
environment the washed-out short leg (ENPH −86 %, SEDG −89 %, SPWR bankrupt) mean-reverts hard
(squeeze risk) and the spread can be flat-to-negative for years.

---

## 3. Fixing the lookahead

The quick backtest used **today's ETF holdings applied back to 2017** and **hand-picked short
names known in hindsight to have collapsed**. Real fixes, in order of preference:

1. **Short an index, not names.** Short **TAN** (Invesco Solar, live since 2008) or a
   residential-solar sub-basket — removes 100 % of short-side name selection. Tested: L-equipment /
   S-TAN gives the same regime pattern with clean 2013→ history.
2. **Rule-based point-in-time universe** for the long leg: US-listed, market cap > $500 M, GICS
   sub-industry in {Electrical Components & Equipment, Heavy Electrical Equipment, Construction &
   Engineering}. GICS history is point-in-time in Compustat; a free proxy is to freeze the list
   as-of each year-end from first-listing dates + a manual sub-industry map.
3. **Frozen-2020 universe, test 2020→ only**, and label 2013–19 explicitly as illustrative /
   survivorship-biased.
4. Cross-check membership against the published ETF holdings each quarter with the **42-day
   reporting lag** (`config.REBALANCE_LAG_DAYS`) — holdings are filed weeks after quarter-end;
   rebalancing on same-day data is lookahead.

The dynamic parts (β, vol target, any structural sizing signal) are **price/-yield-based and
knowable same day** — they carry no lookahead. Lookahead risk lives entirely in the constituent list.

---

## 4. Fixing "fully in cash for 288 days"

Replace the binary `spread OR cash` rule with two changes:

1. **Fallback is the long leg, not cash.** When the relative trade is off, hold the equal-weight
   long-leg basket (theme beta) instead of T-bills. Long-only EW equipment/EPC was +0.75 Sharpe in
   2019–22 — vastly better than cash, still fully rules-based.
2. **Continuous structural sizing, never zero.** Size exposure to the *spread* (0.2×–1.3×, never 0)
   by a real-time, economically-motivated composite that follows the enduring hypothesis:
   - **10-year real yield, 6-month change** (FRED `DFII10`) — rising real yields favour the
     institutional/short-duration long leg over the consumer-financed short leg.
   - **High-yield OAS credit-stress z-score** (FRED `BAMLH0A0HYM2`) — wide/widening credit closes
     the capital markets the short leg depends on.
   - (extensible: policy calendar for ITC/NEM events; long-leg-minus-short-leg aggregate FCF margin.)

Tested (`scratchpad/xsec`, L-equip / S-TAN, 2013→2026):

| rule for adverse regime | 2019–22 Sharpe | full Sharpe | max DD |
|---|---|---|---|
| → cash (old) | −0.21 | 0.63 | −27 % |
| → hold long leg | +0.70 | 0.78 | −51 % |
| **structural sizing (0.4–1.2×) + long-leg fallback, never in cash** | +0.20 | 0.71 | −44 % |
| long-leg EW only (reference) | **+0.75** | **0.82** | −48 % |

The structural-sizing version never goes to cash (min exposure 0.40×, 3 % of days below 0.5×) and
lifts 2019–22 from −0.32 to +0.20 — but it still does **not beat holding the long leg alone**
through the cycle.

---

## 5. What to actually build

**Core (ship this):** rules-based **long-only wide electrification basket.**
- Universe: US-listed names in ≥ 2 of {VOLT, ELFY, ZAP, GRID} published holdings (point-in-time,
  42-day lag), filtered to GICS {Electrical Equipment, Construction & Engineering, and the
  data-center-power names}. ~35–45 names.
- Equal-weight, 25 % single-name cap, quarterly reconstitution.
- 20 % annualised vol target on the basket (trailing 21-day realised vol, 1.5× cap, slack at rf).
- Pre-registered live rule; disclose it as **enhanced thematic beta (β ~1.2–1.4)**, not alpha.

**Optional overlay (regime sleeve):** the long/short.
- Short **TAN** (or a resi-solar/EV/BTM sub-basket), beta-hedged to that instrument on trailing
  126-day beta.
- Exposure to the spread sized 0.2×–1.3× by the real-yield + credit composite of §4; freed capital
  held in the long leg, never cash.
- **Disclosed as conditional**: positive expected contribution only while real yields are
  rising/high and credit is not tight; ~0 to negative otherwise; squeeze risk on a risk-on
  reversal because the short names are washed out.

**Still discretionary / open:**
- The exact universe rule and the GICS split need to be frozen and coded from point-in-time data.
- The §4 composite weights (0.5 real-yield / 0.5 credit) are a first guess — need a pre-registered
  plateau test, not optimisation.
- One macro cycle (2022→) for the part that makes money. The structural hypothesis is the reason to
  expect *some* persistence; it is not proof.
