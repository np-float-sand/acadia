# Electrification Strategy (v1)

Rules-based long-only basket of US electrification / grid-equipment / data-center-power
companies, with a valuation-extension de-risk overlay and an optional hedge overlay
(GLD/short-duration sleeve + a conditional DLR.EQIX short). Honest **enhanced thematic
beta**, not alpha -- close to the VOLT ETF on return, with risk control the ETF lacks.

Spec: `../docs/superpowers/specs/2026-09-06-electrification-strategy-design.md`
Results: `../docs/electrification-strategy-v1-results.md`
Research provenance: `../docs/electrification-short-leg-insurance-probe-results.md`

## Run

```
python -m electrification_strategy --start 2017-06-01 --end 2026-08-31 --output ./output_electrification
```

Prints the 12-cell comparison grid ({marquee, frozen, thematic} universes x {plain,
+valuation, +valuation+sleeve, +valuation+sleeve+short}) and the pitch winner by the
pre-registered rule. Writes `metrics.csv`, `episode_drawdowns.csv`, `plateau.json`,
`returns.csv`, `performance.png`.

## Universes

| name | inclusion | notes |
|---|---|---|
| `marquee` | `grid_equipment_basket.config.UNIVERSE` (the 9) | reference |
| `frozen` | seed pool & sub-industry map & (in >=1 of VOLT/ELFY/ZAP/GRID/PAVE) & `profitable_2026` | our quality screen (~20 names) |
| `thematic` | in >=2 of {VOLT, ELFY, ZAP, GRID}, filtered to sub-industry + listing gate | thematic-ETF consensus (~10 names), no profitability screen |

All three: equal-weight, 25% single-name cap, quarterly reconstitution, 42-day reporting
lag, 252-trading-day listing gate.

## Overlays (frozen params -- `config.py`)

- **Vol target** -- 20% annualised, trailing 21-day realised, 1.5x cap, slack at rf
  (reuses `grid_equipment_basket.overlay.vol_target_scalar`).
- **Valuation / extension** -- monthly multiplier 1.0 / 0.8 / 0.6 on `%-above-200d-MA`
  (15% / 35%) and `12-month return vs SPY` (25% / 50%). A graduated sell-when-euphoric
  trim, **not a crash shield**.
- **Hedge** -- 15% NAV in EW GLD+IEF (the clean winner); optional -0.25x EW DLR+EQIX,
  engaged only while the 6-month change in DFII10 > 0 (5-day lag). The short is a levered
  rising-real-yield bet -- 2 names, ~2.5%/yr dividend paid short, value concentrated in
  the 2022 and DeepSeek drawdowns, ~15%/yr drag in a rate-rising bull.

## Data & update path

- Prices: yfinance via the reused `grid_equipment_basket` monthly-parquet cache.
- DFII10: `fred.py` -- FRED CSV endpoint, parquet cache, committed `data/fred_DFII10.csv`
  offline fallback (refresh the CSV monthly).
- `data/universe_seed.csv`, `data/etf_membership_2026.csv` -- 2026 snapshots.

## Caveats (carry into any pitch)

- **Enhanced thematic beta, not alpha.** ~All the return is the electrification theme.
- **Universe is 2026-vintage, back-cast.** Membership, the ETF snapshot, and
  `profitable_2026` are not point-in-time; only the listing-date gate is. **Pre-live
  gate:** rebuild membership from point-in-time holdings / GICS vintages and make the
  profitability screen time-varying before any live capital.
- **Valuation overlay is a mild trim**, not a crash shield.
- **The DLR.EQIX short** carries borrow + dividend cost, 2-name concentration, and an
  ugly ride in rate-rising bull markets; its evidence is two rate episodes.
