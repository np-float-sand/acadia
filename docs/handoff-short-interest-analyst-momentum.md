# Handoff: Adding Short Interest & Analyst Revision Momentum Signals

**Branch:** res2  
**Date:** June 2026  
**Context:** Grid Resilience Strategy — `grid_resilience/` Python package

---

## Why We Are Adding These Signals

### The VST Problem

The strategy's core signal is a **stress beta** estimated by rolling OLS: how much does each utility stock move when the Grid Stress Index (GSI) rises? Low stress beta → long. High stress beta → short.

This works well for stable regulated utilities (WEC, DTE, CMS etc.) where the relationship between grid stress and equity returns is physically determined and consistent. It breaks down for **competitive/merchant generators** — specifically VST (Vistra) and NRG — because their equity can be driven by a power price narrative that is independent of any individual stress event.

**What happened in 2024:**

- VST entered 2024 with a deeply negative factor score (−1.43) → model was **short** VST
- VST returned **+262%** in 2024; NRG returned +79%
- Pearson correlation between Dec-2023 factor scores and 2024 full-year returns: **−0.727** — near-perfect inversion
- The 2024 IC was −0.023 (useless)

**Why:** The AI data center buildout created demand for firm, always-on power under long-term bilateral contracts. Regulated utilities structurally cannot sign these (rates and capacity are regulator-controlled). VST owns dispatchable gas and nuclear in ERCOT and PJM and can sign bilateral deals freely. Microsoft's Three Mile Island restart deal validated the thesis. Grid stress headlines that previously sold VST (emergency fuel costs) now *bought* it (pricing power for firm capacity). Same physical fact, opposite equity interpretation.

**The rolling OLS eventually adapted** — by May 2024, enough observations of VST rising on high-GSI days had accumulated that the model flipped VST to a strongly positive score (+2.25). But this was 4 months late. The model then flipped VST back to short in May 2025 (after Q1 2025 DeepSeek-driven weakness), right before VST had a +65% Q2.

**Root cause:** The rolling OLS measures *how the stock behaves* during stress but cannot detect *why*. When the narrative driving VST shifts — from "grid stress hurts generators" to "grid stress proves generators have pricing power" — the model lags by roughly one quarter in each direction.

### What Short Interest and Analyst Revisions Would Tell Us

**Short interest** is a direct market signal for narrative positioning. If large investors are covering shorts in VST (short interest falling), they are no longer betting on VST being hurt by stress events — the bearish narrative is unwinding. Conversely, rising short interest signals growing conviction that VST is overvalued relative to fundamentals. This is information the GSI cannot provide.

**Analyst revision momentum** captures whether professional sell-side analysts are raising or lowering earnings estimates. Sustained upward revisions in late 2023 would have signalled that analysts had recognised VST's power-demand opportunity and were embedding it into forward earnings. This is another early indicator that a stock has shifted regime — well before the rolling stress beta catches up.

Together, these two signals act as a **narrative regime detector**: they flag when a stock's investment thesis is shifting, allowing the factor to either adapt faster or reduce conviction in the stress beta signal for that name.

---

## Scope

These signals should apply to **all stocks in the universe**, not just VST and NRG — but they will have the most impact on the merchant names where the stress beta is unstable. For fully regulated utilities (WEC, DTE, CMS), analyst revisions and short interest tend to be low-volatility and will contribute a small stabilising nudge.

---

## Proposed Integration

### Option A — Additive factor components (recommended starting point)

Add short interest and analyst revision momentum as two additional weighted components in `build_factor()` in `grid_resilience/factor/resilience_score.py`, alongside the existing stress beta, renewable quality, and ICR.

Current weights:
```
stress_beta:       70%
renewable_quality: 15%
ICR:               15% (off by default)
```

Proposed weights (when all signals active):
```
stress_beta:            60%
renewable_quality:      10%
short_interest_signal:  15%
analyst_revision_mom:   15%
```

Short interest signal: **lower short interest = more positive score** (falling shorts = narrative turning bullish). Cross-sectionally z-scored, winsorized.

Analyst revision signal: **positive EPS revision momentum = more positive score**. Compute as: (current consensus EPS estimate − estimate N months ago) / |estimate N months ago|. Cross-sectionally z-scored, winsorized.

### Option B — Regime filter (more targeted, more complex)

Use these signals as a scaling factor on the stress beta weight specifically for merchant names. If short interest is falling AND analyst revisions are positive → increase weight on the current stress beta (trust the flip faster). If both are negative → reduce weight (dampen the short signal even if stress beta is high).

Start with Option A. If IC improvement is minimal, try B.

---

## Data Sources

### Short Interest

**Source 1: FINRA (free, 2-week delay)**
- URL: https://www.finra.org/investors/learn-to-invest/advanced-investing/short-selling/regsho/short-interest-statistics
- Published twice monthly (mid-month and end-of-month settlement dates)
- CSV download available programmatically
- Contains: ticker, settlement date, short interest (shares), days-to-cover
- **Limitation:** 2-week lag means you won't have the latest reading at rebalance time

**Source 2: yfinance (free, point-in-time only)**
- `yf.Ticker("VST").info["shortRatio"]` — short ratio (days to cover)
- `yf.Ticker("VST").info["shortPercentOfFloat"]` — % of float shorted
- **Critical limitation:** yfinance only returns the *current* snapshot, not history. Useless for backtesting.

**Recommended approach for backtesting:**
Download the full FINRA historical CSVs (available going back to 2013), parse into a per-ticker time series, cache locally. At each monthly rebalance date, use the most recently available FINRA reading (applying the 2-week lag).

**Signal construction:** Use the 3-month change in short interest as % of float (i.e., is the short position growing or shrinking?). Falling short interest = bullish signal = positive score contribution.

### Analyst Revision Momentum

**Source: yfinance (free)**
- `yf.Ticker("VST").eps_revisions` — DataFrame of EPS estimate revision history by quarter
- `yf.Ticker("VST").analyst_price_targets` — current price target summary (mean, high, low, count)
- `yf.Ticker("VST").recommendations_summary` — aggregate buy/hold/sell counts by period

**Limitation:** `eps_revisions` returns recent revision history but depth varies by ticker — typically 2–4 quarters of data, not the full backtest window. For deep historical revisions, a paid source (Bloomberg, FactSet, Refinitiv) would be needed.

**Recommended approach for backtesting (free):**
Use `recommendations_summary` to compute a buy-minus-sell ratio at each monthly snapshot. Track changes in this ratio month-over-month as the revision signal. Less precise than EPS revision magnitude but available for the full history via yfinance.

Alternative proxy: use `analyst_price_targets` — the change in mean price target relative to current stock price as a sentiment signal.

**Signal construction:** `(strong_buy_count + buy_count − sell_count − strong_sell_count) / total_count`, then compute 3-month momentum of this ratio. Rising ratio = analysts getting more bullish = positive score contribution.

---

## Codebase Integration Points

### Where to add the data fetching

Add two new fetch functions to `grid_resilience/data/equity_prices.py` (alongside existing `fetch_prices`, `fetch_returns`, `fetch_icr`):

```
fetch_short_interest(tickers, start, end) → DataFrame indexed by date, columns = tickers
fetch_analyst_revisions(tickers, start, end) → DataFrame indexed by date, columns = tickers
```

Follow the existing monthly parquet caching pattern in that file. Cache files would be:
- `grid_resilience/data/cache/short_interest_{hash}.parquet`
- `grid_resilience/data/cache/analyst_revisions_{hash}.parquet`

### Where to add signal construction

Add two new signal-building functions to `grid_resilience/factor/resilience_score.py` or a new `grid_resilience/factor/narrative_signals.py`:

```
build_short_interest_signal(short_interest_df) → Series indexed by ticker
build_analyst_revision_signal(revisions_df) → Series indexed by ticker
```

Each returns a cross-sectionally z-scored, winsorized series ready to be weighted into `build_factor()`.

### Where to wire into the factor

Modify `build_factor()` in `grid_resilience/factor/resilience_score.py`:
- Add `short_interest` and `analyst_revisions` optional parameters (matching the `icr` pattern)
- Add weight constants `_W_SHORT` and `_W_ANALYST` (suggested 0.15 each, reduce `_W_BETA` from 0.70 to 0.60, `_W_RENEW` from 0.15 to 0.10)
- Add new config flags `USE_SHORT_INTEREST` and `USE_ANALYST_REVISIONS` to `config.py` (default False, matching `USE_ICR` pattern)
- Add `--short-interest` / `--analyst-revisions` CLI flags to `main.py` (matching `--icr` pattern)

### Where to wire into the rolling factor

Modify `build_rolling_factor()` in `resilience_score.py` to pass the new signals through, with the same date-alignment pattern used for ICR (`_icr_at_date` → add `_short_interest_at_date`, `_analyst_revisions_at_date`).

### Where to wire into main.py

In `grid_resilience/main.py`, fetch the new data alongside existing fetches (equity prices, ICR) and pass through to `build_rolling_factor()`.

---

## Config Additions Needed

In `grid_resilience/config.py`, add:

```python
USE_SHORT_INTEREST:      bool = False  # enable short interest signal
USE_ANALYST_REVISIONS:   bool = False  # enable analyst revision momentum signal

SHORT_INTEREST_WINDOW:   int  = 3      # months of short interest change to measure
ANALYST_REVISION_WINDOW: int  = 3      # months of revision momentum window
```

---

## Validation Plan

Once implemented:
1. Run full backtest with both signals off (baseline — should match current results exactly)
2. Run with `USE_SHORT_INTEREST=True` only — measure IC change vs baseline
3. Run with `USE_ANALYST_REVISIONS=True` only — measure IC change vs baseline
4. Run with both on — measure IC and Sharpe vs baseline
5. Specifically examine 2024 and H1 2025 IC to confirm the VST/NRG whipsaw is reduced

**Key hypothesis to test:** Does adding these signals cause VST to receive a less negative (or positive) score entering 2024, given that short interest was falling and analysts were upgrading VST throughout 2023?

---

## Known Limitations Going In

1. **Short interest historical depth:** FINRA data requires manual parsing and download. yfinance alone cannot backtest this signal. Confirm FINRA history is accessible before building the fetcher.

2. **Analyst revision depth:** yfinance `eps_revisions` history is shallow (~2–4 quarters). Full backtest from 2018 may require a paid source or a proxy (recommendations_summary ratio). Test what yfinance actually returns before committing to EPS revisions vs recommendations proxy.

3. **Small universe:** With only 12–18 names having valid factor scores, cross-sectional z-scoring of short interest and analyst signals will be noisy. A signal that is genuinely informative for VST may wash out across the cross-section.

4. **Reporting lag:** Apply a minimum 2-week lag to short interest (FINRA publication delay) and 1-month lag to analyst revisions (to avoid look-ahead bias on recommendation changes).

5. **Interaction with stress beta flip:** The goal is for these signals to smooth the VST flip timing. But if short interest and analyst signals are positively correlated with the stress beta (they often will be — when VST is doing well, shorts cover AND analysts upgrade AND stress beta turns positive), they may amplify rather than lead the existing signal. Measure the cross-signal correlation before drawing conclusions.

---

## Files To Read Before Starting

- `grid_resilience/factor/resilience_score.py` — full factor construction, ICR integration pattern to follow
- `grid_resilience/data/equity_prices.py` — fetcher pattern including `fetch_icr()` to replicate
- `grid_resilience/config.py` — all parameters, where to add new flags
- `grid_resilience/main.py` — CLI wiring, how `USE_ICR` / `--icr` flag is handled
- `output/findings.md` — sections 3, 4, 6, 9, 10, 11 for full analytical context

