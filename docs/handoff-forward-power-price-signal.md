# Handoff: Forward Power Price Signal — Separating Structural Scarcity from Grid Stress

**Branch:** res2  
**Date:** June 2026  
**Context:** Grid Resilience Strategy — `grid_resilience/` Python package

---

## The Core Distinction This Work Addresses

The strategy currently uses one signal to explain utility stock returns during grid conditions: the **Grid Stress Index (GSI)**. The GSI measures short-term, event-driven stress — a polar vortex, a heat dome, a congestion crisis happening *today*. It is built from same-day LMP prices, reserve tightness, and a named event catalog.

**Forward power prices measure something completely different:** the market's expectation of what electricity will cost *one to three years from now*, reflecting structural views on long-run supply and demand. These are traded on exchanges (CME, ICE) as annual strips — e.g. "ERCOT on-peak 2027" — and move based on factors like data center load growth announcements, new generation investment, EPA regulations, and fuel cost outlooks.

**Why the distinction matters:**

When a polar vortex hits ERCOT, both the GSI *and* spot power prices spike simultaneously. In that moment they look like the same thing. But in 2024, ERCOT forward prices for 2026–2028 were rising steadily because investors believed AI data center load growth would create structural generation scarcity — completely independent of whether any stress event was happening on any given day. VST's stock was tracking those long-dated forward prices, not the daily GSI.

The current single-variable model:

```
VST_return = α + β × GSI
```

cannot separate these two effects. The β ends up absorbing both — it becomes unstable because sometimes VST moves with GSI (stress event regime) and sometimes it moves with forward power prices (structural scarcity regime). The rolling OLS chases whichever effect dominated most recently.

The proposed two-variable model:

```
VST_return = α + β₁ × GSI + β₂ × ΔForward_Power_Price
```

assigns each driver its own coefficient. β₁ cleanly measures the pure grid-stress response. β₂ captures the structural power price component. The factor score is built from the full predicted return — both terms — and IC/Sharpe are still measured against actual stock returns.

**Why forward prices specifically, not spot:**

Spot power prices (day-ahead LMP hub prices) are already partially captured inside the GSI via the LMP z-score sub-signal (40% weight). Adding spot prices as a second predictor would create collinearity — on a high-stress day, both GSI and spot price spike for the same reason. Forward prices (annual strips, 12–36 months out) are largely orthogonal to the GSI because they reflect multi-year structural expectations, not today's weather or demand conditions. On any given non-stress day the GSI is low, but forward prices for 2027 can be rising steadily.

---

## What This Would Have Done in 2024

ERCOT Cal-26 and Cal-27 forward prices were rising throughout 2023 and 2024 as the AI data center narrative built. Under the two-variable model, β₂ × ΔForward_Power_Price would have been large and positive for VST entering 2024, pulling the predicted return strongly positive — even while β₁ × GSI was still estimating a slightly negative stress response from 2018–2023 data.

The model would have predicted VST goes up and positioned it as a long candidate several months before the rolling stress beta alone caught up in May 2024. The 2024 IC crater (−0.023) would likely have been significantly shallower.

---

## What This Would and Would Not Fix

| Situation | GSI-only model | Two-variable model |
|-----------|---------------|-------------------|
| VST rising because of AI power demand (forward prices up) | Blind — no signal | Captured by β₂ — helps significantly |
| VST rising because of grid stress event (spot prices spike) | Captured by β₁ | Same — captured by β₁ |
| VST falling on DeepSeek (narrative shock, not prices) | Blind | Partially — if forward prices also fell on DeepSeek day, β₂ captures it; if idiosyncratic, still blind |
| VST rising on specific contract win or earnings beat | Blind | Blind |
| Regulated utilities (WEC, DTE, CMS) | Works well | β₂ ≈ 0, no change to these names |

The two-variable model eliminates the chronic structural problem (forward prices driving merchant names) but does not eliminate idiosyncratic event risk (specific corporate announcements, DeepSeek-type shocks).

---

## Data Sources

### True Forward Power Curves (market prices) — Paid

**CME Group — Electricity Futures**
- Products: ERCOT North Hub Real-Time Peak (NP), ERCOT North Hub Day-Ahead Off-Peak, PJM Western Hub, etc.
- Delivery: monthly, quarterly, and annual strips up to 3–5 years forward
- Access: CME DataMine subscription (~$50–200/month depending on tier). Free delayed quotes visible on CME website but not downloadable for backtesting.
- URL: https://www.cmegroup.com/markets/energy/electricity.html

**ICE (IntercontinentalExchange) — Power Futures**
- Broader product range than CME for North American power
- Access: ICE Data Services subscription — more expensive than CME, typically institutional
- URL: https://www.theice.com/products/Electricity

**Wood Mackenzie / Genscape**
- Power market intelligence including forward curve history
- Expensive, institutional pricing

**Bottom line on true forwards:** Historical forward curve data with sufficient depth (2018–2025) for backtesting requires a paid subscription. CME DataMine is the most accessible starting point.

---

### Practical Free Proxies

Because true forward curve data is not freely available, the following proxies can substitute for backtesting purposes:

**1. Natural Gas Futures — Henry Hub (recommended, free via yfinance)**

In ERCOT and PJM, gas-fired generation is typically the marginal unit that sets the clearing price. Long-dated natural gas prices are therefore the closest freely available proxy for long-dated power prices in these grids.

- Tickers available via yfinance: `NG=F` (front month), `NGG25`, `NGH25`, etc. (specific contract months)
- For a structural signal, use the **12-month forward contract** (e.g. gas for delivery one year from now) rather than the front month, to avoid conflation with current-season demand
- Monthly frequency is sufficient for a monthly-rebalancing strategy
- Full history available from yfinance back to the 1990s

Signal construction: 3-month change in the 12-month Henry Hub forward price. Rising = structural power price pressure building = positive signal for VST/NRG.

**2. ERCOT Day-Ahead vs Real-Time Spread (already partially built)**

Per `CLAUDE.md`, RT/DA LMP spread infrastructure is ~90% built in the codebase (`ISO_RT_LOCATION_TYPE` config needed). This measures short-term forward premium — whether the real-time market is consistently above day-ahead prices, indicating persistent tightness. Less useful than annual strips but free and already partially implemented.

**3. Renewable Energy Penetration Rate**

Rising renewable penetration compresses power prices at the margin (merit order effect) while increasing volatility. This can be tracked free via EIA API (monthly generation by fuel type). An increasing renewable share in ERCOT puts long-run downward pressure on average prices but upward pressure on price volatility — nuanced but informative.

**4. EIA Short-Term Energy Outlook (STEO)**
- Free from EIA API: https://api.eia.gov/
- Published monthly, includes 12–24 month electricity price forecasts for major regions
- These are EIA projections, not market prices — less precise than traded forwards but free and available historically
- API endpoint: `https://api.eia.gov/v2/steo/` — includes `ERCOT_PRICE` and similar series

---

## Recommended Approach

**Phase 1 — Validate the concept using Henry Hub gas futures (free)**

1. Fetch monthly 12-month-forward Henry Hub prices via yfinance back to 2018
2. Add to the return model as a second predictor alongside GSI for VST and NRG specifically
3. Measure IC improvement in 2024 and full backtest vs. baseline
4. If improvement is meaningful, proceed to Phase 2

**Phase 2 — Replace gas proxy with true ERCOT/PJM forward prices (paid)**

Subscribe to CME DataMine at the minimum tier needed for ERCOT and PJM electricity annual strips. Re-run backtest with true market-implied forward prices. Compare IC and Sharpe vs. gas-proxy version.

The gas proxy may be sufficient — if the IC improvement from Phase 1 is already significant, the marginal value of true power forwards may not justify the data cost.

---

## Codebase Integration Points

### New data fetcher

Add `fetch_power_forward_prices()` to `grid_resilience/data/equity_prices.py` or a new `grid_resilience/data/power_forwards.py`:

```python
def fetch_gas_forward_proxy(start: str, end: str) -> pd.Series:
    """
    Fetch 12-month Henry Hub natural gas forward price as a proxy
    for structural long-dated power price expectations.
    Returns monthly series indexed by date.
    """
```

Use yfinance to fetch the relevant NG futures contract, roll forward monthly to maintain a consistent 12-month-ahead series. Cache in the standard monthly parquet pattern.

### New beta estimation

Modify `grid_resilience/factor/conditional_beta.py` (or wherever stress betas are computed — check `compute_stress_betas()`) to support an optional two-variable OLS:

```python
def compute_stress_betas(
    returns: pd.DataFrame,
    gsi: pd.Series,
    power_forward: pd.Series | None = None,   # NEW
    ...
) -> pd.DataFrame:
```

When `power_forward` is provided, run:
```
stock_return = α + β₁ × GSI + β₂ × ΔPowerForward + ε
```

Return both β₁ (for existing factor scoring logic) and the full predicted return including β₂ component.

### Factor score construction

The factor currently uses β₁ alone as the primary component. With the two-variable model, the factor score for a given stock should be based on the **full predicted return** (`β₁ × GSI_today + β₂ × ΔPowerForward_today`), not just β₁.

This is a design decision: see the options below.

**Option A — Use full predicted return as factor score**
Score = predicted return from two-variable model. This directly answers "what does the model expect this stock to do?" IC and Sharpe measured against actual returns. Most theoretically clean.

**Option B — Use β₁ for ranking, β₂ as a correction term**
Keep the existing β₁-based ranking but add a forward-price adjustment that shifts scores when structural power prices are moving. More conservative, easier to explain.

Recommend Option A for a clean implementation.

### Config additions

In `grid_resilience/config.py`:

```python
USE_POWER_FORWARD_SIGNAL: bool = False  # enable forward power price second predictor
POWER_FORWARD_PROXY:      str  = "henry_hub"  # "henry_hub" | "ercot_da" | "eia_steo"
POWER_FORWARD_WINDOW:     int  = 12  # months ahead for forward price (if using futures roll)
```

### CLI additions

In `grid_resilience/main.py`, add `--power-forward` flag matching the `--icr` pattern.

---

## Files To Read Before Starting

- `grid_resilience/factor/resilience_score.py` — existing factor construction, how stress beta is currently used
- `grid_resilience/data/equity_prices.py` — fetcher and cache pattern to follow
- `grid_resilience/config.py` — where to add new flags
- `grid_resilience/main.py` — CLI wiring pattern
- `output/findings.md` sections 3, 4, 6, 10, 11 — full VST analytical context and the two-variable model reasoning
- `CLAUDE.md` — notes the RT/DA spread (5th GSI sub-signal) as ~90% built; coordinate with that work to avoid overlap

---

## Key Validation Tests

1. Fetch Henry Hub 12-month forward series for 2018–2025 and plot against ERCOT spot prices — confirm they diverge during the 2024 AI narrative period (forwards rising while individual stress events are not elevated)
2. Run baseline backtest (power forward off) — confirm results match existing output
3. Run with power forward enabled for VST and NRG only — measure 2024 IC specifically
4. Run with power forward enabled for all stocks — measure full-period IC and Sharpe
5. Examine β₂ estimates over time for VST — confirm they are stable (not flipping) unlike β₁

**Key hypothesis:** β₂ (gas/power forward sensitivity) for VST should be consistently positive throughout 2018–2025 — VST has always benefited from rising structural power prices. β₁ (pure GSI stress response) should be the unstable one. If this holds, the two-variable model cleanly separates the two effects.

---

## Known Limitations Going In

1. **Gas ≠ power in all regimes.** Henry Hub is a strong proxy for ERCOT/PJM power prices when gas is the marginal fuel. In periods when renewables set the clearing price (increasingly common in ERCOT), the gas-power correlation weakens. The proxy will be imperfect.

2. **Front month vs. forward roll.** yfinance gas futures require careful rolling logic to maintain a consistent 12-months-forward series. Front-month gas is highly seasonal and correlated with near-term weather — exactly what we want to avoid. Confirm the rolling methodology before building the backtest.

3. **β₂ is also estimated from rolling data.** If the gas-to-VST relationship itself shifts over time (e.g. VST hedges more of its output under long-term contracts), β₂ will also lag. This is a second-order problem but worth monitoring.

4. **Collinearity with existing GSI sub-signals.** The GSI LMP z-score sub-signal already partially captures spot power price movements. Running a VIF check on the two-variable model (GSI, ΔForward) is recommended to confirm the predictors are sufficiently orthogonal for stable coefficient estimation.

5. **Regulated utilities.** For names like WEC, DTE, and CMS, the power forward signal should have β₂ ≈ 0 because their revenues are not exposed to market prices. If β₂ is non-zero for regulated names, it is likely spurious correlation and the signal should be suppressed for those names (use the merchant classification from findings.md section 7).

