# Grid Resilience Strategy

A quantitative long/short equity strategy that trades U.S. utility stocks based on their sensitivity to electricity grid stress events.

## Executive Summary

Most utility quant factors (valuation, yield, regulatory lag) are calendar-driven and well-arbitraged. This strategy exploits a different edge: **utilities with operations in stressed grid regions are systematically mispriced around grid stress events**, and their degree of sensitivity varies predictably by service territory, generation mix, and transmission exposure.

The strategy builds a daily **Grid Stress Index (GSI)** for each ISO using real-time LMP prices, load data, and a curated catalog of named stress events (polar vortices, heat domes, hurricanes, congestion crises). It then estimates each utility's **conditional stress beta** — how much its stock moves on high-GSI days relative to sector. Utilities with low stress betas (resilient to grid disruption) go long; utilities with high stress betas (amplified exposure) go short. The portfolio rebalances monthly.

**Key hypothesis:** Grid stress betas are predictable from physical grid topology and generation mix, are not priced by consensus utility analysts, and provide a return edge that is orthogonal to rate cycles and regulatory calendars.

**Universe:** 18 U.S. utility tickers across ERCOT, PJM, MISO, CAISO, SPP, ISO-NE, and NYISO.

**Portfolio:** 5 long / 5 short, equal-weighted within each book, rebalanced monthly.

**Backtest window:** 2018–present (covers Uri, CA heat dome, multiple polar vortex events).

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

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- Free API keys (see below)

### 1. Install dependencies

```bash
poetry install
```

### 2. Set API keys

**ERCOT** (required for any run):
```bash
export ERCOT_API_TOKEN="your-token"
```
Register at [developer.ercot.com](https://developer.ercot.com) — click "Sign Up", verify your email, then generate a token under "My Profile". Free, no approval required.

**EIA** (required for renewable quality adjustment in factor scores):
```bash
export EIA_API_KEY="your-key"
```
Register at [eia.gov/opendata](https://www.eia.gov/opendata/register.php) — fill in the short form and your API key is emailed immediately. Free, no approval required.

**PJM Data Miner** (only needed if running `--iso PJM`):
```bash
export PJM_API_KEY="your-subscription-key"
```
PJM uses their **Data Miner API** portal, and non-members require a manual approval step:

1. Register for a PJM Tools account at [accountmanager.pjm.com](https://accountmanager.pjm.com/accountmanager/pages/public/new-user.jsf) — fill out the contact form and set a password within 4 hours of the confirmation email.
2. Email **accountmanager@pjm.com** with the exact statement: *"I confirm that the PJM Data will be used for internal business purposes only."* Include your Account Manager username and the email address you registered with. PJM will provision your account for Data Miner non-member access.
3. Once approved, log in to [apiportal.pjm.com](https://apiportal.pjm.com), go to **Profile**, and copy your **Primary Key** under Subscriptions. That is the value to export as `PJM_API_KEY`.

Note: approval is not instant — allow 1–2 business days. The key is passed as the `Ocp-Apim-Subscription-Key` header in API calls.

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
--iso    ERCOT PJM MISO CAISO SPP   ISOs to include (default: all 5)
--start  YYYY-MM-DD                 Backtest start (default: 2018-01-01)
--end    YYYY-MM-DD                 Backtest end (default: today)
--long   N                          Long book size (default: 5)
--short  N                          Short book size (default: 5)
--no-plot                           Skip matplotlib charts
--output PATH                       Output directory (default: ./output)
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
| LMP + load (most ISOs) | [gridstatus](https://github.com/kmax12/gridstatus) | None (ERCOT needs token) |
| EIA generation mix | EIA Open Data API | Free key required |
| PJM grid data | PJM Data Miner API | Non-member email approval required |
