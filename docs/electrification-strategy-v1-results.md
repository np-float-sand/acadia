# Electrification Strategy v1 — results

**Date:** 2026-09-06
**Spec:** docs/superpowers/specs/2026-09-06-electrification-strategy-design.md
**Plan:** docs/superpowers/plans/2026-09-06-electrification-strategy.md
**Command:** `python -m electrification_strategy --start 2017-06-01 --end 2026-08-31`
**Package tests:** 36 passing (`tests/electrification_strategy/`).

## The 12-cell grid

```
cell                            CAGR     Vol  Sharpe    MaxDD   beta    drag  feared corrVOLT
marquee plain                 19.5%  20.6%    0.77 -26.8%   0.71   0.0%   0.0%     0.83
marquee +val                  16.5%  18.2%    0.71 -26.8%   0.65   3.0%   3.2%     0.80
marquee +val+sleeve           15.3%  15.5%    0.74 -22.6%   0.56   4.2%  12.9%     0.81
marquee +val+sleeve+short     14.2%  14.8%    0.70 -22.5%   0.46   5.3%  12.9%     0.73
frozen plain                  18.8%  20.4%    0.75 -27.3%   0.76   0.0%   0.0%     0.89
frozen +val                   17.8%  18.4%    0.77 -27.3%   0.71   1.0%   4.1%     0.85
frozen +val+sleeve            16.4%  15.8%    0.79 -23.2%   0.61   2.4%  12.9%     0.86
frozen +val+sleeve+short      15.3%  14.9%    0.76 -21.8%   0.51   3.5%  12.9%     0.78
thematic plain                21.3%  20.5%    0.85 -28.4%   0.75   0.0%   0.0%     0.88
thematic +val                 18.9%  18.0%    0.83 -28.4%   0.68   2.4%   4.5%     0.83
thematic +val+sleeve          17.3%  15.4%    0.85 -24.1%   0.59   4.0%  12.4%     0.85
thematic +val+sleeve+short    16.2%  14.6%    0.82 -22.7%   0.49   5.2%  13.4%     0.77

effective #names: marquee 9.0  frozen 23.0  thematic 10.0

benchmarks (CAGR / Sharpe / MaxDD):
  VOLT    16.8%   0.58   -24.9%
  PAVE    20.4%   0.72   -43.8%
  GRID    20.8%   0.76   -40.6%
  SPY     14.5%   0.61   -33.7%
  XLI     15.6%   0.61   -42.0%
```

`drag` = full-period CAGR give-up vs the same universe's plain book. `feared` = cumulative
hedge P&L over the months where the plain book fell and TAN rose (the "solar squeezes
while AI-power de-rates" divergence).

## Winner (pre-registered rule, spec §6.3)

**Winner: `thematic +val+sleeve`** — Sharpe 0.85, MaxDD −24.1%, full-period drag 4.0%/yr,
feared-scenario P&L +12.4%, SPY β 0.59, corr vs VOLT 0.85.
**Runner-up: `thematic +val+sleeve+short`** — Sharpe 0.82, MaxDD −22.7%, drag 5.2%/yr, β 0.49.
**8 of 12 cells clear the bar** (MaxDD ≥ −27%, both sub-window Sharpes ≥ 0.4, feared ≥ −2%,
drag ≤ 6%/yr).

Reading:

- The harness picked **no short leg**. The GLD/short-duration sleeve alone is the clean
  winner on risk-adjusted return; adding the DLR·EQIX short trims MaxDD a further ~1.4pp
  but costs ~1.2pp/yr of drag and 0.03 of Sharpe — it lands as the runner-up, i.e. optional
  insurance, exactly as the 2026-09-06 probe concluded.
- **Universe:** `thematic` (10 names, ≥2-of-4 thematic-ETF consensus) posts the highest
  Sharpe at every stack; `frozen` (23 names, our quality screen) is ~0.06 Sharpe behind but
  cuts single-name concentration hard (effective #names 23 vs 9) and has the deepest
  benchmark out-performance vs a plain plain-book drawdown. `marquee` (the 9, incl. FLNC)
  is the weakest.
- **vs VOLT:** every non-plain strategy cell beats VOLT on Sharpe (0.58) and most beat it on
  drawdown; the winner is +0.27 Sharpe / +0.8pp MaxDD better than the ETF. Still ~0.85
  correlated — enhanced thematic beta, not decorrelation.
- **vs the probe:** `marquee plain` here is Sharpe 0.77 / MaxDD −26.8% vs the probe's
  ~0.84 / −31% for the same 9 names + a 20% daily vol target; the residual gap is the
  candidate-column set and cache vintage. `+val+sleeve` MaxDD −22.6% and feared +12.9%
  match the probe's −22% / +13%.

## Valuation plateau

```
  x0.8  Sharpe  0.72  MaxDD  -27.3%  CAGR   16.3%
  x1.0  Sharpe  0.77  MaxDD  -27.3%  CAGR   17.8%
  x1.2  Sharpe  0.75  MaxDD  -27.3%  CAGR   17.8%
```
Stable across ±20% threshold scaling (Sharpe 0.72–0.77, MaxDD flat) — not knife-edge.

## Implementation notes / deviations from spec

- **Daily vol targeting**, not `grid_equipment_basket.overlay.vol_target_scalar` (spec §2.2
  named the latter). That helper re-levers only monthly and deepened MaxDD ~15pp through
  fast crashes (marquee plain −48% vs −27%); the 2026-09-06 probe — the source of every
  design number — re-levered daily. `backtest._vol_scalar` matches the probe.
- **Valuation extension measured on the plain basket index** (cumprod of pre-vol-target
  returns), per the spec §4.1 self-review note — avoids circularity with `exposure`; the
  probe used the vol-targeted series and the difference is second-order.
- `THR` (Thermon) 404s from yfinance in this run; immaterial — it is in zero ETFs so it is
  never a member of `frozen` or `thematic`.
- Tasks 3+4 and 7+8 were committed together (each pair is one atomic code unit).

## Caveats (carry into any pitch)

- **Enhanced thematic beta, not alpha** — ~all the return is the electrification theme.
- **Universe is 2026-vintage, back-cast.** Membership, the ETF snapshot, and
  `profitable_2026` are not point-in-time; only the listing-date gate is. **Pre-live gate:**
  rebuild membership from point-in-time holdings / GICS vintages and make the profitability
  screen time-varying before any live capital (spec §1 non-goals, §3.3).
- **Valuation overlay is a mild trim**, not a crash shield (~0–3pp MaxDD).
- **The DLR·EQIX short's** evidence is two rate episodes (2022, DeepSeek); it carries
  borrow + ~2.5%/yr dividend cost and an ugly ride in rate-rising bull markets. It is the
  runner-up here, not the winner.
- One macro cycle for the part that reduces drawdowns — structural logic, not proof.

## Follow-up (2026-09-06): data-center-backlash hedge test

Can a *political/NIMBY* de-rating (moratoriums, permit denials) with **no macro downturn**
be hedged? The GLD/Treasury sleeve cannot help there (no growth scare, no debasement move).
No historical backlash episode exists in-sample; proxy = the 12 worst months for the
DC-power sleeve (VRT, GEV) where SPY was flat/up (>= -1%). Throwaway: `scratchpad/backlash_hedge.py`.

| | DC-power | grid-maint | merchant | SPY |
|---|---|---|---|---|
| mean across the 12 proxy months | **-7.2%** | **+0.4%** | +0.3% | +2.8% |
| monthly corr with VRT | +0.98 | +0.60 | -- | -- |
| beta to the DC-power sleeve | 1.00 | 0.36 | 0.35 | -- |

- **Grid-maintenance tilt -- ADOPT.** 70/30 grid/DC construction (grid-maint = ETN HUBB PWR
  MYRG PRIM EMR AME RRX POWL ATKR AEIS; DC-power = VRT GEV) is net-neutral on full-period
  Sharpe/MaxDD (1.02 vs 1.00 in the scratchpad harness) but cuts exposure to an
  idiosyncratic DC de-rating; trades ~5pp of Rate-22 protection for it. To be wired as a
  selectable construction in `universe.py`.
- **Conditional merchant-power short -- REJECT.** Trigger (DC-power underperforming
  grid-maint over 126d) fires 34% of days; every short weight lowers Sharpe (1.00 -> 0.93 at
  0.25x) and deepens MaxDD (-25% -> -28%). Merchant power doesn't reliably fall in an
  idiosyncratic DC selloff.
- **Puts** -- only cause-agnostic hedge; ~3-8%/yr premium; blocked on options data.

Proposal (3-tier, IC review): `docs/electrification-strategy-proposal-v2.md` /
`docs/electrification-strategy-proposal-v2.html`.
