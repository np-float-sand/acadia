# Electrification Strategy (v1)

Rules-based long-only basket of US electrification / grid-equipment / data-center-power
companies, with a valuation-extension de-risk overlay and optional hedge overlays
(GLD/short-duration sleeve + a conditional DLR.EQIX short + a conditional solar-rotation
long). Honest **enhanced thematic beta**, not alpha -- close to the VOLT ETF on return,
with risk control the ETF lacks.

Spec: `../docs/superpowers/specs/2026-09-06-electrification-strategy-design.md`
Results: `../docs/electrification-strategy-v1-results.md`
Research provenance: `../docs/electrification-short-leg-insurance-probe-results.md`

## Run

```
python -m electrification_strategy --start 2017-06-01 --end 2026-08-31 --output ./output_electrification
```

Prints the 25-cell comparison grid ({marquee, frozen, thematic, screen, screen_reit}
universes x {plain, +valuation, +valuation+sleeve, +valuation+sleeve+short,
+valuation+sleeve+solar}) and the pitch winner by the pre-registered rule. Writes
`metrics.csv`, `episode_drawdowns.csv`, `plateau.json`, `returns.csv`, `performance.png`.
Also writes `risk_exposure_report.csv` (the raw long-format table from
`risk_exposures.load_risk_exposures()` -- not yet weighted to the winning cell's
portfolio) and `short_interest_diagnostic.csv` (`short_interest.py` crowding snapshot on
the hedge short names) when the underlying data is present -- both are reporting-only
diagnostics, not backtest inputs, and are silently skipped (no crash) while no real
`data/risk_exposures.csv` or cached short-interest history exists yet. A failed
short-interest fetch is likewise non-fatal: the CLI warns and finishes the run rather than
discarding a completed backtest. The diagnostic carries a `data_quality_flag` column, set
when a row shows the stale-field signature (0%/missing short-%-of-float alongside a
near-zero share count), so an implausible provider reading isn't reported as a clean zero.

**Note on the pre-registered winner:** adding `+val+sleeve+solar` to the grid changes what
the unmodified `pick_winner()` rule mechanically selects -- `frozen +val+sleeve+solar`
(Sharpe 0.85) now beats the previously-documented `marquee plain` (Sharpe 0.77) on every
gate. This wasn't forced; the rule just prefers it once it's a candidate. The actual
shipped construction is still `screen +val+sleeve` (a discretionary PM override of
`pick_winner`'s raw output, same as before) -- see
`../docs/electrification-strategy-v1-results.md` and `../docs/RESEARCH-LOG.md` §1/§4 item 17
before treating the new mechanical winner as a recommendation.

## Universes

| name | inclusion | notes |
|---|---|---|
| `marquee` | `grid_equipment_basket.config.UNIVERSE` (the 9) | reference |
| `frozen` | seed pool & sub-industry map & (in >=1 of VOLT/ELFY/ZAP/GRID/PAVE) | quality screen, ETF-gated (~15 names) |
| `thematic` | in >=2 of {VOLT, ELFY, ZAP, GRID}, filtered to sub-industry + listing gate | thematic-ETF consensus (~7 names); depends on incomplete ETF-holdings data |
| **`screen`** (recommended) | seed pool & sub-industry map & `customer_institutional` & (`earnings_valued` OR `low_policy_dependence`), **`data_center_reit` rows explicitly excluded** | **our supplier rule, NO ETF gate** (27 names). Picked on stated properties. 0% utilities / pipelines / components -- the thesis distinction from VOLT (which is ~43% those). Beats VOLT on a fair window (2025-01->2026-08): CAGR 33% vs 24%, Sharpe 1.18 vs 0.82. |
| **`screen_auto`** (mechanical) | **no seed list, no judgment column**: SEC SIC population + fuel-cycle lexicon + $10M liquidity, then TTM capex/revenue in the light half, then buildout vocabulary in the prior-year 10-K, then TTM `NetIncomeLoss` > 0, then ITC/PTC/NEM/45X exclusion, then SIC group POWER dropped. ~19 names/mo. Beats `screen` on Sharpe in all three windows (full 1.132 vs 1.000) but **misses the pre-registered gate in 2023-26 on MaxDD by 0.55pp** and overlaps `screen` on only 53% of names. **Caveat (2026-09-23):** those figures are measured on the 48-ticker `MARQUEE ∪ seed` price panel, which drops 17 of the 31 names this rule actually picks — hence the "~14 effective names". On its own mechanical universe it holds **18.5 names/mo / 29 effective** and scores **Sharpe 1.049 / MaxDD -20.4%**, so ~0.08 of the quoted Sharpe edge comes from the hand seed list still filtering the universe. See `../docs/screen-auto-breadth-results.md`. **Fixed 2026-09-25:** `run_comparison` and the CLI now price `universe.candidate_tickers(constructions)`, which adds every `screen_auto` member when that construction is requested; through the package, `screen_auto +val+sleeve` (default 2-yr lookback / 0.5 quantile membership) is now **Sharpe 1.009 / Calmar 0.977 full, 1.495 / 1.612 from 2023** (was 1.132 on the truncated panel). The published A1+A4 book (4-yr / 0.60) is 1.050 / 1.118 full, 1.55 / 1.69 from 2023. Selectable via `--universe screen_auto`; **not** in the default grid, because it needs a build artifact. See `../docs/screen-auto-mechanical-universe-results.md`. |
| `screen_reit` | same rule as `screen`, but including `data_center_reit` rows (IRM, AMT) | **REIT-breadth probe (Task 7), ADOPTED as an available construction** -- `screen`'s pool ∪ qualifying data-center REITs (29 names). Beats `screen` on Sharpe and MaxDD in all overlay stacks (e.g. `+val+sleeve`: Sharpe 0.716 vs 0.693, MaxDD -23.2% vs -24.0%); does not change the overall pre-registered winner. Not the shipped default -- `screen` still is. See `../docs/electrification-reit-breadth-probe-results.md`. |

All five: equal-weight, 25% single-name cap, quarterly reconstitution, 42-day reporting
lag, 252-trading-day listing gate. `screen` is the recommended/shipped universe (see
`../docs/electrification-strategy-v1-results.md`); `screen_reit` is a tested, available
alternative, not the default.

## Overlays (frozen params -- `config.py`)

- **Vol target** -- 20% annualised, trailing 21-day realised, 1.5x cap, slack at rf
  (reuses `grid_equipment_basket.overlay.vol_target_scalar`).
- **Valuation / extension** -- monthly multiplier 1.0 / 0.8 / 0.6 on `%-above-200d-MA`
  (15% / 35%) and `12-month return vs SPY` (25% / 50%). A graduated sell-when-euphoric
  trim, **not a crash shield**.
- **Hedge sleeve** -- 15% NAV in EW GLD+IEF (the clean winner).
- **Hedge short** -- optional -0.25x EW DLR+EQIX, engaged only while the 6-month change in
  DFII10 > 0 (5-day lag). A levered rising-real-yield bet -- 2 names, ~2.5%/yr dividend
  paid short, value concentrated in the 2022 and DeepSeek drawdowns, ~15%/yr drag in a
  rate-rising bull.
- **Solar rotation** (`+solar`, open item #17) -- optional +0.10x EW
  ENPH/SEDG/CHPT/RUN/BLNK/STEM long, layered on top of the `+val+sleeve` book, engaged
  only while the same 6-month DFII10 change is *negative* (falling real yields, ~42-44%
  of days). The opposite-sign trade to the hedge short: a convex cushion for the feared
  "rates fall -> washed-out solar squeezes up -> AI-power suppliers de-rate" scenario.
  Tested 2026-09-06 on `screen +val+sleeve`: full Sharpe 0.69->0.77, CAGR 14.4->16.8%,
  MaxDD **worsens** -24.0%->-26.8% (report this honestly -- Sharpe/CAGR improve but the
  raw drawdown is deeper, since the DER sleeve is itself a ~-90% asset), +7% P&L in the
  feared months, and it rescues the weak 2019-22 sub-window (Sharpe 0.51->0.78). Wired
  2026-09-18 reproduces this closely on today's data/cache: Sharpe 0.79, CAGR 16.6%,
  MaxDD -24.9% (shallower than originally tested), feared P&L +8.9% vs the `+val+sleeve`
  ship. The exact 2026-09-06 throwaway script that produced the original table
  (`scratchpad/utility_rotation.py` in that session) only persisted the XLU-rotation
  half; the DER-sleeve-return construction here uses this package's existing
  equal-weight-of-available-tickers convention (`hedge_overlay._ew_returns`, the same
  pattern already used for the GLD/IEF sleeve and the DLR/EQIX short), not a byte-for-byte
  copy of an unrecovered script. **Disclosed tail risk, not the shipped default** -- the
  trigger fires on ~42-44% of days (arguably too loose) and this is a levered
  mean-reversion bet with squeeze risk both ways. See
  `../docs/electrification-strategy-v1-results.md` and `../docs/RESEARCH-LOG.md` §4 item 17.

## Data & update path

- Prices: yfinance via the reused `grid_equipment_basket` monthly-parquet cache.
- DFII10: `fred.py` -- FRED CSV endpoint, parquet cache, committed `data/fred_DFII10.csv`
  offline fallback (refresh the CSV monthly).
- `data/universe_seed.csv`, `data/etf_membership_2026.csv` -- 2026 snapshots.
- `data/vocabulary_hits.csv`, `data/screen_auto_members.csv` -- **build artifacts** for
  `screen_auto`, not snapshots: regenerate with
  `python -m electrification_strategy.build_auto_screen [--refresh]`. `--refresh` re-queries
  EDGAR full-text search (~30 min, one request per candidate filer per phrase).

## Caveats (carry into any pitch)

- **Enhanced thematic beta, not alpha.** ~All the return is the electrification theme.
- **Universe is 2026-vintage, back-cast.** The seed pool, the ETF snapshot, and the three
  judgment columns (`customer_institutional`, `earnings_valued`, `low_policy_dependence`)
  are not point-in-time; only the listing-date gate is. The ETF snapshot is
  top-25-only for GRID/PAVE/ELFY. **Where the exposure actually sits:** an ablation
  (2026-09-22) showed the binding criterion is `customer_institutional` -- a stable
  structural fact about who a firm sells to -- and that 30 of 33 seed names pass it.
  So the back-cast risk is concentrated in **seed-list membership**, not in the filter.
  **Pre-live gate:** a point-in-time seed list, and (for `thematic`/`frozen`) a full
  point-in-time holdings feed. `screen` has no ETF dependency.
  **`screen_auto` (2026-09-23) retires the seed-list half of this gate in its *rule*** -- it
  derives membership mechanically from SEC filings with no hand-recorded input -- but **not yet in
  its *evaluation***: `run_comparison` prices only `MARQUEE_UNIVERSE | load_seed()`, so the seed
  list still acts as a universe filter on the results, and it is worth ~0.08 Sharpe
  (`../docs/screen-auto-breadth-results.md`). Closing this gate properly means re-running the
  pre-registered gate on the mechanical price panel. Concentration is NOT the cost it appeared to
  be (29 effective names on that panel, not 14); the 2023-26 MaxDD gate miss stands. It reaches 27 of 28
  `screen` names by vocabulary but keeps only 53% of them after the other filters, so it is
  a better book rather than a mechanical restatement of this one.
- **`profitable_2026` was removed on 2026-09-22.**
  *We used to do:* `customer_institutional` AND >=2 of 3 of {`profitable_2026`,
  `earnings_valued`, `low_policy_dependence`}. *We now do:* `customer_institutional`
  AND (`earnings_valued` OR `low_policy_dependence`).
  **It was look-ahead** -- one profitability verdict recorded in 2026 and applied
  backwards to every date from 2017 on, so a firm loss-making in 2019 but profitable
  by 2026 counted as profitable in 2019. **And it was not helpful** -- the
  `screen`/`screen_reit`/`frozen` pools are identical with and without it (28/30/15
  names, zero adds, zero drops) and returns byte-identical (max abs diff 0.00e+00 over
  2,421 days). Removing it cost nothing and removed the one year-stamped input.
- **Valuation overlay is a mild trim**, not a crash shield.
- **The DLR.EQIX short** carries borrow + dividend cost, 2-name concentration, and an
  ugly ride in rate-rising bull markets; its evidence is two rate episodes.
- **The solar rotation** is a levered, one-cycle, squeeze-risk-both-ways bet on a ~-90%
  volatility asset; it deepens MaxDD in raw terms even though it raises Sharpe/CAGR, and
  its trigger fires on ~42-44% of days (looser than ideal). Candidate overlay, not shipped
  by default.
