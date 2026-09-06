# Claude Code Instructions

## File Writing

- Always write markdown files using Bash (`cat > file.md << 'EOF' ... EOF`) rather than the Write tool.

## README Maintenance

- After any code change, check whether it invalidates anything in README.md (API signatures, default values, CLI flags, cache behaviour, data sources). If so, update README.md in the same session using Bash.

## Future Work (Tracked)

- **GSI congestion signal — Phase 2:** Add LMP component-based extraction (energy/congestion/loss decomposition from gridstatus) AND promote inter-zonal spread to a separate 5th GSI sub-signal with rebalanced weights. Deferred until the spread signal is validated in backtesting. See `docs/superpowers/specs/2026-05-28-congestion-spread-design.md`.

## Grid Search Findings (2026-06-17)

A parallelised grid search over 243 parameter combinations (`grid_search.py`) identified the optimal config.  Key findings:

### Winning parameters (now live in `config.py`)
| Parameter | Value | Notes |
|---|---|---|
| `STRESS_SPIKE_PCT` | 0.97 | 97th-percentile LMP spike threshold |
| `STRESS_CONG_THRESHOLD` | 0.30 | 30% congestion fraction |
| `STRESS_CONG_MIN_DAYS` | 10 | 10 consecutive days required |
| `PORTFOLIO_LONG_N` | 3 | Concentrated long book |
| `PORTFOLIO_SHORT_N` | 5 | Wider short book |
| `XLU_HEDGE` | False | Individual shorts, not XLU |

**Result:** Sharpe 0.319, Ann Return 8.1%, Max DD -13.7% vs XLU Sharpe 0.235, Max DD -36.1%.

### Key structural finding: XLU hedge destroys alpha
Every `xlu_hedge=True` configuration produced **negative Sharpe** (ranks 157–243 out of 243).  The long book's top-scored names do not outperform XLU by enough to make a sector-relative trade work.  The strategy's alpha comes from the short book picking individual losers, not from long-only stock selection.  Do not re-enable `XLU_HEDGE` without re-validating IC on the long book first.

### Congestion threshold insight
`cong_threshold=0.30` (lower than the original 0.50 design) combined with `cong_min_days=10` (the original duration floor) wins.  The duration filter carries the noise-reduction burden — a lower fraction threshold captures more congestion signals while the 10-day minimum filters out short-lived blips.

### Portfolio concentration
3 long names outperforms 5 or 7 at the same stress config.  The factor has enough cross-sectional dispersion that concentrating in the top 3 adds return without a proportional increase in drawdown.

### Full results
See `output/grid_search_results.csv` for all 243 combinations ranked by Sharpe.

## Future Work — Signal Improvements

- **Interest Coverage Ratio (ICR) factor component:** Implemented in `equity_prices.fetch_icr()` and wired into `resilience_score.build_factor()` as a 15% weight (beta drops to 70%, renewables 15%). Currently **off by default** (`USE_ICR = False` in `config.py`, `--icr` / `--no-icr` CLI flag). Turn on to validate: does low-ICR identify better short candidates during rate stress events? Limitation: yfinance only returns ~4–12 quarters of history per ticker, so ICR contribution is weakest in the pre-2022 backtest window. Consider EDGAR as a historical data source before drawing firm conclusions.

- **RT/DA LMP spread (5th GSI sub-signal):** Real-time vs day-ahead price spread cleanly separates generators (benefit from RT spikes) from T&D utilities (hurt by congestion charges). Infrastructure ~90% ready — needs a new `ISO_RT_LOCATION_TYPE` config dict and a `fetch_lmp_rt()` variant. Highest-impact addition after ICR is validated.

- **Value-chain signal — Q4 derivation + operating-margin variants:** the shipped signal uses
  company-wide gross-margin change. Spec 2026-08-29 §9 logs two untried refinements: company-wide
  operating margin (XBRL, all 9 names) and grid-segment operating margin (hand-collected). Try
  after the gross-margin version is judged; check whether they sharpen or muddy the diversified
  names (ETN, GEV, PRIM). Final-review status (2026-08-29, `docs/grid-equipment-value-chain-results.md`):
  (1) **the revenue-tag selection bug is fixed** — `margin_data` now unions `Revenues` +
  `RevenueFromContractWithCustomerExcludingAssessedTax` instead of first-nonempty, which lifted
  backlog-coverage (§4.2) from 4/9 to 8/9 names; (2) **still open** — the gross-margin component
  (§4.1) is live for only PRIM in the primary window because *most US filers* stopped tagging (or
  never tagged) the December quarter as a discrete 3-month period (PRIM tags all four quarters;
  FLNC tags Dec but skips Sept), so the trailing-5-quarter span guard NaN's the rest. The rebuild
  needs to derive Q4 = FY − 9-month YTD rather than rely on a discrete Q4 fact, especially for the
  genuinely annual-only names. `value_chain_report` no longer crashes under `--prior-regime` (Gate 2
  is reported `n/a` for a window with no 2024-H2 episode).

## Results Ledger (keep updated every session)

`docs/RESULTS.md` is the canonical running record of every strategy / signal / construction
attempt in this project and its pre-registered verdict. **Whenever a probe or build reaches a
verdict, append a row to the relevant table in `docs/RESULTS.md`** (date · name · what it tested ·
verdict · key number · link to the full `docs/*-results.md`), and update its Standing Conclusion /
What-Is-Shipped / Still-Open sections if the verdict moves them. Blunt pre-registered verdict
language only (`PASS` / `FAIL` / `NOT ADOPTED` / `NOT YET TESTABLE` / `ADOPTED` / `REVERTED`) —
negative results are first-class here. The "How to update this doc" section at the top of
`docs/RESULTS.md` has the format.
