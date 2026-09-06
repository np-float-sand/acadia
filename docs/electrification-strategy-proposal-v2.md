# Electrification Strategy — proposal (v2)

**Date:** 2026-09-06
**Backs onto:** `electrification_strategy/` package (BUILT), `docs/electrification-strategy-v1-results.md`,
`docs/electrification-short-leg-insurance-probe-results.md`, spec
`docs/superpowers/specs/2026-09-06-electrification-strategy-design.md`.
**Shipped variant (pre-registered winner):** `thematic +val+sleeve` — Sharpe 0.85 / MaxDD −24% /
SPY β 0.59 over 2017–2026 (VOLT: 0.58 / −25%).

---

## Like you're 5

- We buy equal slices of the companies that make the electrical parts for the AI and
  data-center power boom, and tidy the basket up a few times a year.
- When those stocks jump up too fast we hold a bit less, and we keep some gold and
  government bonds on the side as a cushion for when the boom stumbles.
- We lean toward the "keep-the-grid-running" companies and away from the
  "build-brand-new-data-centers" ones, so if towns start blocking data centers we get
  hurt less.

## Like you're 10

- It's a rules-based basket of ~10 US electrification / grid-equipment stocks (the ones
  the electrification ETFs agree on), equal-weighted, rebalanced quarterly, and dialed to
  a steady 20% risk level — a disciplined way to own the theme, not a stock-picking bet;
  2017–2026 test Sharpe 0.85, worst drop −24%, vs the VOLT ETF's 0.58 and −25%.
- Two shock absorbers: a rule that trims the basket when it has run far above its own
  trend or far ahead of the market, and a 15% sleeve of gold + Treasuries that pays off in
  the two macro ways the story breaks — a credit/growth scare (bonds rise) or a
  government-spending / currency-debasement episode (gold rises).
- For the risk the sleeve can't cover — a political backlash (data-center moratoriums,
  permit denials) with no recession — we tilt ~70/30 toward grid-maintenance names over
  greenfield data-center-power names; in the months tested where data-center stocks fell
  on their own, the grid-maintenance names were roughly flat.

## Like a first-year university grad

- **Construction:** electrification-equipment universe by ≥2-of-4 thematic-ETF overlap +
  GICS sub-industry filter + 252-day listing gate, equal-weight (25% cap), 42-day
  reporting lag, 20% trailing-realised-vol target (1.5× cap) — 2017–2026 Sharpe 0.85 /
  MaxDD −24% / SPY β 0.59 (VOLT: 0.58 / −25%). Enhanced thematic beta, explicitly not
  alpha; pre-live gate is a point-in-time membership + profitability rebuild.
- **Overlays:** (i) graduated valuation-extension de-lever — multiplier 1.0/0.8/0.6 on
  price-vs-200dMA (15%/35%) and 12-month return vs SPY (25%/50%), a mild top-trim worth
  ~0–3pp of MaxDD; (ii) 15% GLD/IEF sleeve hedging the financing-tightening (duration) and
  debasement/geopolitical (gold) tails of the industrial-policy regime that funds the
  buildout — +13% contribution across the "rates fall while AI-capex unwinds" months, and
  it raises Sharpe.
- **Idiosyncratic-backlash handling:** a 70/30 revenue tilt toward grid-reliability / T&D
  / broad-electrification names (β 0.36 to the greenfield-DC-power sleeve), net-neutral on
  headline metrics and ~flat in the tested idiosyncratic-DC-selloff months (DC-power
  −7.2% / grid-maintenance +0.4% mean across 12 months); a conditional merchant-power
  short was tested and rejected (deepens MaxDD, no reliable payoff); index/name puts are
  the only cause-agnostic hedge and carry a premium cost not in the backtest.

---

## Backlash test (2026-09-06, `scratchpad/backlash_hedge.py`, throwaway)

No historical data-center-backlash episode exists in the 2017–2026 sample; the proxy is
the 12 worst months for the DC-power sleeve (VRT, GEV) where SPY was flat/up (≥ −1%).

| | DC-power | grid-maint | merchant | SPY |
|---|---|---|---|---|
| mean across the 12 proxy months | **−7.2%** | **+0.4%** | +0.3% | +2.8% |
| corr with VRT (monthly) | +0.98 | +0.60 | — | — |
| β to the DC-power sleeve | 1.00 | 0.36 | 0.35 | — |

- **Grid-maintenance tilt — adopt.** 70/30 grid/DC construction is net-neutral on
  full-period Sharpe/MaxDD (1.02 vs 1.00) but materially cuts exposure to an idiosyncratic
  DC de-rating. Trades a little rate-shock protection for it (Rate-22 episode −26% vs −21%).
- **Conditional merchant-power short — reject.** Trigger (DC-power underperforming
  grid-maint over 126d) fires 34% of the time; every short weight lowers Sharpe
  (1.00 → 0.93 at 0.25×) and deepens MaxDD (−25% → −28%). Merchant power does not
  reliably fall in an idiosyncratic DC selloff.
- **Puts** — the only cause-agnostic hedge; ~3–8%/yr premium drag; blocked on options data.

## Open / pre-live

- Rebuild universe membership + the profitability screen point-in-time (spec §1, §3.3).
- Wire the 70/30 grid-maintenance tilt as a selectable construction in
  `electrification_strategy/universe.py`.
- Options-collar sizing once an options-data feed exists.
