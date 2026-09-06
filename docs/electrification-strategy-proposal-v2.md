# Electrification Strategy — proposal (v2)

**Date:** 2026-09-06
**Backs onto:** `electrification_strategy/` package (BUILT, 38 tests),
`docs/electrification-strategy-v1-results.md`,
`docs/electrification-short-leg-insurance-probe-results.md`, spec
`docs/superpowers/specs/2026-09-06-electrification-strategy-design.md`.
**Shipped variant:** `screen +val+sleeve` — our supplier-screen universe (27 names, no ETF
gate) + valuation de-lever + gold/Treasury sleeve. **vs VOLT on a fair window
(2025-01→2026-08, since VOLT's Dec-2024 launch): CAGR 27.7% vs 24.1%, Sharpe 1.35 vs 0.82,
MaxDD −17.0% vs −24.4%.** Full 2017–2026 backtest: Sharpe 0.69 / MaxDD −24% / SPY β 0.59.

---

## Like you're 5

- We buy equal slices of ~27 companies that make the electrical parts for the AI and
  data-center power boom, and tidy the basket up a few times a year.
- When those stocks jump up too fast we hold a bit less, and we keep some gold and
  government bonds on the side as a cushion for when the boom stumbles.
- We only pick the *suppliers* — the shovel-makers — and leave out the power companies and
  pipelines that the big electrification ETF mixes in; that's why we've been beating it.

## Like you're 10

- It's a rules-based basket of ~27 US electrification / grid-equipment stocks chosen by
  **our own screen** — a real business customer (utilities, hyperscalers), profitable, and
  not dependent on subsidies — equal-weighted, rebalanced quarterly, dialed to a steady
  20% risk level. Since the VOLT ETF launched (Dec 2024) we've returned 28%/yr at a much
  smaller worst-drop (−17% vs −25%).
- Two shock absorbers: a rule that trims the basket when it has run far above its own trend
  or far ahead of the market, and a 15% sleeve of gold + Treasuries that pays off in the
  two macro ways the story breaks — a credit/growth scare (bonds rise) or a
  government-spending / currency-debasement episode (gold rises).
- The screen keeps us a pure *supplier* book: 0% regulated utilities, pipelines, or
  chip-component makers, which are ~43% of VOLT. That's the whole reason we've outrun it —
  and the risk is that if utilities lead again (rates fall), we give that cushion up.

## Like a first-year university grad

- **Construction.** `screen` universe — sub-industry-mapped electrification/grid pool
  filtered by an institutional-customer axis + ≥2 of {profitable, earnings-valued (P/E not
  EV/Sales), low policy-dependence}, i.e. the capex-cycle-pair-classifier rule restricted
  to the supplier side; **no ETF-membership gate**. Equal-weight (25% cap), 42-day
  reporting lag, 20% trailing-vol target (1.5× cap). 27 names, top weight ~4%. Full
  2017–2026 Sharpe 0.69 (mid-pack vs marquee/frozen 0.77); the edge is the recent
  supplier-vs-utility divergence + a method that removes the "you're just VOLT" and
  "you hand-picked winners" objections. Enhanced thematic beta, not alpha (corr to VOLT 0.88).
- **Overlays.** (i) Graduated valuation-extension de-lever — multiplier 1.0/0.8/0.6 on
  price-vs-200dMA (15%/35%) and 12-month return vs SPY (25%/50%), a mild top-trim worth
  ~0–3pp MaxDD; (ii) 15% GLD/IEF sleeve hedging the financing-tightening (duration) and
  debasement/geopolitical (gold) tails of the industrial-policy regime that funds the
  buildout — +13% contribution across the "rates fall while AI-capex unwinds" months, and
  it raises Sharpe.
- **Idiosyncratic-backlash handling.** A 70/30 revenue tilt toward grid-reliability / T&D /
  broad-electrification names (β 0.36 to the greenfield-DC-power sleeve), net-neutral on
  headline metrics and roughly flat in the tested idiosyncratic-DC-selloff months
  (DC-power −7.2% / grid-maintenance +0.4% mean across 12 months); a conditional
  merchant-power short was tested and rejected; index/name puts are the only cause-agnostic
  hedge and carry a premium cost not in the backtest.

---

## vs VOLT — why we've been ahead

VOLT (Tema Electrification) real sector weights: **utilities 30% · energy/midstream 10% ·
tech-components 13% · industrials 47%.** Its top 10 includes NextEra, AEP, Idacorp, OGE
(utilities), Energy Transfer (pipeline MLP), Amphenol & Bel Fuse (components). Our `screen`
book holds **none of that** — it is 100% equipment / EPC / thermal / distribution suppliers.

| 2025-01 → 2026-08 (VOLT-comparable) | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| VOLT ETF | 24.1% | 0.82 | −24.4% |
| `screen` plain | 32.5% | 1.18 | −22.3% |
| `screen +val+sleeve` (shipped) | 27.7% | 1.35 | −17.0% |

The distinction is **compositional and methodological, not statistical** — 0.88 correlated
with VOLT, so this is outperformance *within* the theme, regime-dependent (suppliers over
utilities), on one cycle of evidence.

## Backlash test (2026-09-06, `scratchpad/backlash_hedge.py`, throwaway)

No historical data-center-backlash episode in the 2017–2026 sample; proxy = the 12 worst
months for the DC-power sleeve (VRT, GEV) where SPY was flat/up (≥ −1%).

| | DC-power | grid-maint | merchant | SPY |
|---|---|---|---|---|
| mean across the 12 proxy months | **−7.2%** | **+0.4%** | +0.3% | +2.8% |
| β to the DC-power sleeve | 1.00 | 0.36 | 0.35 | — |

- **Grid-maintenance tilt — adopt.** 70/30 grid/DC construction is net-neutral on
  full-period Sharpe/MaxDD but materially cuts exposure to an idiosyncratic DC de-rating.
- **Conditional merchant-power short — reject.** Every short weight lowers Sharpe
  (1.00 → 0.93 at 0.25×) and deepens MaxDD. Merchant power doesn't reliably fall in an
  idiosyncratic DC selloff.
- **Puts** — the only cause-agnostic hedge; ~3–8%/yr premium; blocked on options data.

## Open / pre-live

- Make the four judgment columns time-varying, and (only if `thematic`/`frozen` are kept)
  obtain a full point-in-time holdings feed for GRID/PAVE/ELFY. `screen` needs neither.
- Wire the 70/30 grid-maintenance tilt as a selectable construction in `universe.py`.
- Options-collar sizing once an options-data feed exists.
