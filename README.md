# Grid Resilience Strategy

A quantitative long/short equity strategy that trades U.S. utility stocks based on their sensitivity to electricity grid stress events.

> **Sibling strategy modules:** `grid_equipment_basket/` — a long-only thematic basket of grid / data-center equipment suppliers (see its own README and `docs/grid-equipment-basket-step1-results.md`).
>
> Beyond the equal-weight theme basket it also carries a value-chain reframe (`--construction value-chain-tilt` | `pair`): overweight the equipment makers, underweight/short the price-taking contractors, with a market-neutral pair + conditional-QQQ-short hedge — Gate 1 (long-only tilt) passes weakly, Gate 2 (pair as hedge) fails in favour of the conditional short. See `docs/grid-equipment-value-chain-results.md`.
>
> `grid_demand_factor/` — a **triage probe** (not a shipped strategy) for "rank stocks by return-sensitivity to a grid-demand nowcast, trade the tails". Pre-registered gate **FAILED** robustly: monthly Δnowcast is near-orthogonal to equity returns, so the sensitivity sort carries no cross-sectional information. Run: `python -m grid_demand_factor.probe --price-glob 'backlog_factor/data/cache/prices_*.parquet'`. See `docs/triage_2026-08-31-proposals-bda.md`.

## Executive Summary

Most utility quant factors (valuation, yield, regulatory lag) are calendar-driven and well-arbitraged. This strategy exploits a different edge: **utilities with operations in stressed grid regions are systematically mispriced around grid stress events**, and their degree of sensitivity varies predictably by service territory, generation mix, and transmission exposure.

The strategy builds a daily **Grid Stress Index (GSI)** for each ISO using real-time LMP prices, load data, and a curated catalog of named stress events (polar vortices, heat domes, hurricanes, congestion crises). It then estimates each utility's **conditional stress beta** — how much its stock moves on high-GSI days relative to sector. Utilities with low stress betas (resilient to grid disruption) go long; utilities with high stress betas (amplified exposure) go short. The portfolio rebalances monthly.

**Key hypothesis:** Grid stress betas are predictable from physical grid topology and generation mix, are not priced by consensus utility analysts, and provide a return edge that is orthogonal to rate cycles and regulatory calendars.

**Universe:** 18 U.S. utility tickers across ERCOT, PJM, MISO, CAISO, SPP, ISO-NE, and NYISO.

**Portfolio:** 3 long / 5 short, equal-weighted within each book, rebalanced monthly.

**Backtest window:** 2018-01-01 – 2026-05-27.

---

## How It Works

The pipeline runs in 7 sequential steps:

```
[1] Equity prices      →  yfinance (adjusted daily closes for universe)
[2] Grid data          →  gridstatus (hourly LMP + load per ISO)
[3] Stress events      →  named catalog + algo-detected LMP spikes
[4] Grid Stress Index  →  composite score per ISO per day [0, 1]
[5] Stress betas       →  OLS of stock returns on GSI, rolling window
[6] Factor scores      →  cross-sectional rank of negative stress beta (+
                           renewable quality adjustment)
[7] Backtest           →  rolling weights → P&L → Sharpe / drawdown metrics
```

The GSI blends four sub-signals: LMP z-score (40%), congestion fraction (25%), reserve tightness (20%), and named event flag (15%).

---

## Caching

All external data is cached to `grid_resilience/data/cache/` as parquet files.

| Cache file | Source | Key |
|---|---|---|
| `equity_prices.parquet` | yfinance | single file for all tickers |
| `{iso}_lmp_{loc_type}.parquet` | gridstatus | one file per ISO + location type |
| `{iso}_load.parquet` | gridstatus | one file per ISO |
| `{iso}_fuel_mix.parquet` | gridstatus | one file per ISO |
| `ercot_lmp_zone.parquet` | gridstatus | ERCOT load-zone LMPs for inter-zonal spread (congestion signal) |

**Gap-filling:** fetchers check what is already on disk and download only the missing date ranges. A new ticker or extended date range triggers a targeted fetch, not a full re-download.

**`force_refresh=False` (default):** use the cache if the file exists; write it if it doesn't.  
**`force_refresh=True`:** skip the cache, re-download the full requested range, overwrite.

---

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- Free API keys (see below)

### 1. Install dependencies

```bash
poetry install
source .venv/bin/activate
```

### 2. Set API keys

**ERCOT** (required for any run):

Register at [developer.ercot.com](https://developer.ercot.com) — free, no approval required. Set credentials in `.env`:

```
ERCOT_USERNAME=your-email@example.com
ERCOT_PASSWORD=your-password
ERCOT_SUBSCRIPTION_KEY=your-subscription-key
```

The bearer token is fetched automatically at runtime by `grid_resilience/data/ercot_auth.py` — you do not need to set `ERCOT_API_KEY` manually.

**EIA** (required for renewable quality adjustment in factor scores):

Register at [eia.gov/opendata](https://www.eia.gov/opendata/register.php) — key emailed immediately. Set in `.env`:

```
EIA_API_KEY=your-key
```

**PJM Data Miner** (only needed if running `--iso PJM`)

Add to `.env`:
```
PJM_API_KEY=your-subscription-key
```

Note: basic DataMiner2 endpoints are publicly accessible without auth (used in the pitch scripts). The `grid_resilience` module passes the key to gridstatus for authenticated bulk pulls. Non-members require a manual approval step:

1. Register for a PJM Tools account at [accountmanager.pjm.com](https://accountmanager.pjm.com/accountmanager/pages/public/new-user.jsf) — fill out the contact form and set a password within 4 hours of the confirmation email.
2. Email **accountmanager@pjm.com** with the exact statement: *"I confirm that the PJM Data will be used for internal business purposes only."* Include your Account Manager username and the email address you registered with. PJM will provision your account for Data Miner non-member access.
3. Once approved, log in to [apiportal.pjm.com](https://apiportal.pjm.com), go to **Profile**, and copy your **Primary Key** under Subscriptions. That is the value to set as `PJM_API_KEY`.

Allow 1–2 business days for approval. The key is passed as the `Ocp-Apim-Subscription-Key` header.

### 3. Run

**Single-ISO quick demo (fastest, ERCOT only, reduced book size):**
```bash
python -m grid_resilience.main --iso ERCOT --start 2020-01-01 --long 2 --short 1
```

**Minimum viable full-portfolio run (3 ISOs, 11 tickers):**
```bash
python -m grid_resilience.main --iso ERCOT PJM MISO --start 2020-01-01
```

**Full strategy (all 5 ISOs, default settings):**
```bash
python -m grid_resilience.main
```

**All CLI options:**
```
--iso              ERCOT PJM MISO CAISO SPP   ISOs to include (default: all 5)
--start            YYYY-MM-DD                 Backtest start (default: 2018-01-01)
--end              YYYY-MM-DD                 Backtest end (default: 2026-05-27)
--long             N                          Long book size (default: 3)
--short            N                          Short book size (default: 5)
--xlu-hedge / --no-xlu-hedge                  Replace short book with -0.5 XLU hedge (default: off)
--icr / --no-icr                              Include interest coverage ratio as a factor component (default: off)
--zone-gsi / --no-zone-gsi                    Per-ticker PJM zone GSI instead of shared system-hub GSI (default: on)
--arch             hard-switch|revenue-mix|dual-track
                                               Business-model-aware signal architecture (default: none — original
                                               stress-beta-only behaviour). Auto-enables --icr when set.
--regulated-signal icr|dc-queue|dc-multi      Signal used for the regulated path of --arch hard-switch
                                               (default: dc-queue). dc-queue proxies data-center load growth from PJM's
                                               interconnection queue for PJM tickers, falling back to ICR elsewhere —
                                               see docs/superpowers/specs/2026-07-06-dc-load-signal-design.md.
                                               dc-multi does not blend its layers: it SELECTS one per ticker per
                                               rebalance date, by precedence, from three layers — PJM
                                               generation-queue proxy, ERCOT TSP-level large-load data, and
                                               disclosed hyperscaler colocation deals (each cross-sectionally
                                               z-scored within its own covered tickers first) — falling back to
                                               ICR for tickers none of the three cover on that date. Where more
                                               than one layer has real data for the same ticker on the same date,
                                               hyperscaler deals win over the PJM generation-queue proxy, which
                                               wins over ERCOT TSP data. Precedence is resolved per date, so a
                                               ticker can be sourced from the PJM proxy before its first disclosed
                                               deal and from the hyperscaler layer after it. Currently resolves
                                               real (non-ICR) values for 8 tickers (6 PJM regulated names plus
                                               CEG/TLN; VST's only disclosed deal postdates BACKTEST_END, and
                                               ERCOT's public TSP data doesn't yet yield usable per-ticker MW
                                               figures). If the PJM queue or zonal-load fetch comes back empty the
                                               whole option warns and falls back to `icr` for that run, exactly as
                                               dc-queue does. Also writes `dc_signal_cross_check.csv` (see Outputs)
                                               — see
                                               docs/superpowers/specs/2026-08-26-dc-demand-exposure-signal-design.md.
--peer-group / --no-peer-group                 Build long/short baskets within business-model peer groups
                                               (merchant/mixed/regulated) instead of ranking the whole universe
                                               (default: off). Intended to cancel sector-beta exposure that
                                               whole-universe ranking carries on both legs — see
                                               docs/superpowers/specs/2026-08-15-peer-group-construction-design.md.
                                               Reduces volatility/drawdown materially in testing but has not been
                                               shown to beat XLU on Sharpe — see
                                               docs/compact_2026-08-19-peer-group-construction-results.md.
--no-plot                                     Skip matplotlib charts
--output           PATH                       Output directory (default: ./output)
```

### 4. Outputs

All files are written to `./output/` (or `--output` path):

| File | Contents |
|---|---|
| `stress_events.csv` | Full event calendar (named + algo-derived) |
| `gsi_ercot.csv` (etc.) | Daily GSI per ISO with sub-signal breakdown |
| `factor_scores.csv` | Rolling factor scores per ticker per rebalance date |
| `pnl.csv` | Daily strategy P&L |
| `backtest_performance.png` | Equity curve + drawdown + stress event overlay |
| `dc_signal_cross_check.csv` | (`--regulated-signal dc-multi` only) Diagnostic: flags tickers where the PJM generation-queue DC signal is positive but PJM's own industry-tagged Large Load data doesn't call the driving zone(s) a data center — not blended into the factor score |

---

## How Many ISOs Do You Need to Test the Strategy?

| Goal | ISOs | Command |
|---|---|---|
| Validate pipeline end-to-end | 1 (ERCOT) | `--iso ERCOT --long 2 --short 1` |
| Run with default 5L/5S book | 3 (ERCOT + PJM + MISO) | `--iso ERCOT PJM MISO` |
| Full production backtest | 5 (all) | *(no flags needed)* |

With ERCOT alone you only get 3 LMP-mapped tickers (NRG, VST, CNP), which is insufficient for the default 5/5 portfolio. ERCOT + PJM + MISO together give 11 tickers — the minimum to run the default book. Adding CAISO and SPP brings the universe to 14 tickers and fills in California wildfire/heat and Plains wind congestion regimes.

SERC (Southern Company) and FRCC (NextEra) are in the universe but have no granular LMP data — they receive sector-average imputation from the factor layer regardless of which ISOs you run.

---

## ISO Coverage and Tickers

| ISO | Tickers | Key Stress Events |
|---|---|---|
| ERCOT | NRG, VST, CNP | Winter Storm Uri, summer heat emergencies |
| PJM | AEP, EXC, PPL, FE | Polar vortex 2019, Winter Storm Elliott |
| MISO | ETR, WEC, DTE, CMS | Polar vortex, Hurricane Ida |
| CAISO | PCG, EIX | CA heat dome Sep 2022, PSPS wildfire events |
| SPP | XEL | Winter Storm Uri (SPS zone) |
| ISO-NE | ES | Nor'easters, winter gas supply stress |
| NYISO | ED | Heat waves (Zone J NYC) |
| SERC/FRCC | SO, NEE | EIA-417 outage data only (no LMP) |

*ISO-NE and NYISO tickers are in the universe map but not in the default `SUPPORTED_ISOS` run list — add them via `--iso` if desired.*

### How the 18 tickers were chosen

The universe was built bottom-up from 10-K service territory filings and FERC Form 1 submissions. Each ticker had to meet three criteria:

1. **Geographic node mapping** — the company's primary service territory or generation fleet maps to at least one named ISO hub or load zone where hourly LMP data is publicly available. This rules out holding companies with purely regulatory (non-market) rate structures.
2. **Sufficient float and liquidity** — large-cap names only, to avoid microstructure noise distorting the stress-beta estimates.
3. **ISO span** — at least one name per major ISO interconnection so the cross-sectional factor has dispersion across physically distinct grid regimes (not just ERCOT correlation).

Two names (SO and NEE) are included even though their ISOs (SERC and FRCC) lack granular nodal LMPs. They are kept in the universe because they are among the largest U.S. utilities by market cap and their stress signal is imputed from EIA-417 outage data and sector-average GSI. Their node lists are intentionally empty in the mapping file, which triggers the imputation path automatically.

The sub-groupings (Generators, Wires-Only, Integrated) mirror the distinctions used by sell-side utility analysts and are used by the portfolio layer to enforce sector-neutral construction — preventing the long book from being a pure merchant-generation tilt.

---

## Data Sources

| Data | Source | Credentials |
|---|---|---|
| Equity prices | yfinance | None |
| LMP + load (most ISOs) | [gridstatus](https://github.com/kmax12/gridstatus) | None (ERCOT needs token — auto-fetched) |
| EIA generation mix | EIA Open Data API | Free key required — add to `.env` |
| PJM grid data | PJM Data Miner API | Non-member email approval required — add to `.env` |
