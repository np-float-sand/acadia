# Claude Code Instructions

## File Writing

- Always write markdown files using Bash (`cat > file.md << 'EOF' ... EOF`) rather than the Write tool.

## README Maintenance

- After any code change, check whether it invalidates anything in README.md (API signatures, default values, CLI flags, cache behaviour, data sources). If so, update README.md in the same session using Bash.

## Future Work (Tracked)

- **GSI congestion signal — Phase 2: NOT RECOMMENDED (validated 2026-09-08).** The Phase-1
  spread fallback was checked offline against production caches: the "arrives as NaN for all
  ISOs" diagnosis was only ever true for **ERCOT** (raw feed has no congestion component —
  100% NaN before the fix, fully filled after, real non-trivial signal). **PJM's `congestion`
  column was never broken** — 0% NaN before any fix, and the spread-fill changes 0 of 2,922
  days there. Since PJM's component data is already correct and the GSI's documented problem
  is signal strength / sample size (not construction, per the 2026-08-26 closeout audit),
  decomposing further or promoting the spread to a 5th sub-signal has no live bug to fix and
  low prior. See `docs/untried-tasks-log.md` §1, `docs/superpowers/specs/2026-05-28-congestion-spread-design.md`.

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

- **Interest Coverage Ratio (ICR) factor component: RETESTED, STAYS OFF (2026-09-08).** Implemented in `equity_prices.fetch_icr()` and wired into `resilience_score.build_factor()` as a 15% weight (beta drops to 70%, renewables 15%). `USE_ICR = False` in `config.py` — confirmed correct, not just a data-availability placeholder. The yfinance-depth limitation this note used to flag is resolved (EDGAR XBRL gives a median 63 quarters/ticker vs yfinance's 4-12), but retesting with full depth still fails: full-sample monthly rank-IC t=-0.95 (wrong sign), and the two three-year windows that look like a big effect are PCG/EIX wildfire-liability and VST/CEG AI-re-rating confounds, not a leverage relationship. See `docs/icr-edgar-retest-results.md`, `docs/untried-tasks-log.md` item 2.

- **RT/DA LMP spread (5th GSI sub-signal): FEASIBILITY CHECKED 2026-09-08.** ERCOT is fully feasible and easier than assumed — `ercot.get_lmp()` (already the production fetch) IS the real-time leg; only day-ahead (`get_dam_spp(year)`, full year in ~8s, 2011->present) needs adding, no new RT fetcher required. PJM is confirmed blocked for real-time-hourly LMP before ~2024 (forces a full ~10,000-pnode download per request, 429s within 6 pages even for a single day) but fine for recent dates. VST/NRG — the two names actually driving the stress-beta whipsaw — are both ERCOT, so the useful part of this doesn't need PJM's blocked history. See `docs/rt-da-spread-feasibility-results.md`, `docs/untried-tasks-log.md` item 4. Signal design/build itself not yet started.

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

## Research Log (keep updated every session)

`docs/RESEARCH-LOG.md` is the single canonical running record of every strategy / signal /
construction attempt in this line of research and its pre-registered verdict — one file, not a
new standalone doc per session. **Whenever a probe or build reaches a verdict, add a row to the
relevant table in `docs/RESEARCH-LOG.md` §3** (thread · when · outcome · link to the full
`docs/*-results.md`), and update §1 (current strategy), §2 (accumulated conclusion), and §4 (open
items) if the verdict moves them. Blunt pre-registered verdict language only (`PASS` / `FAIL` /
`NOT ADOPTED` / `BUILT` / `NOT YET TESTABLE` / `SCOPED, NOT RUN` / `ADOPTED` / `REVERTED` /
`KEEPER`) — negative results are first-class here. The "Update convention" line at the top of
`docs/RESEARCH-LOG.md` has the format.
