# Outage/Reserve-Margin Availability Signal — Design

**Status: DEAD (2026-08-22).** Superseded by a spike test that disproved the underlying data
source's usefulness for this purpose — see
`docs/handoff_2026-08-15-outage-reserve-margin-signal.md` for the full account. Kept here only
for the record: EIA-860's `status` field is annual-scale (retirements/mothballing), and showed
*zero* change for VST or NRG through Winter Storm Uri (Feb 2021) — it does not capture the
days-to-weeks forced outages this design needed. Do not implement this spec. An independent
review before the spike also found the modulation formula (`beta_z * availability_ratio`) was
directionally wrong for negative-beta cases and silently inert under the `dual_track`
architecture — both moot now given the data-source finding, but preserved here in case a future
attempt reuses this formulation with a different data source.

## Background

Full research trail: `docs/handoff_2026-08-15-outage-reserve-margin-signal.md`. Summary: the
existing merchant-path stress-beta signal (`grid_resilience/signals/conditional_beta.py::compute_stress_betas()`)
treats VST and NRG — the only two merchant generators in the universe, both ERCOT-exposed — as
interchangeable, based purely on their historical price-return relationship. It doesn't account
for which generator actually has available (non-outaged) capacity to capture a given stress
event's price spike.

A 2026-08-20 data-availability research pass found:
- **ERCOT's hourly resource-outage report (`gridstatus.Ercot.get_hourly_resource_outage_capacity`)
  is not usable** — live-tested, it's a rolling operational report with no historical archive
  (earliest available document was from the prior month; nothing from 2018–2025).
- **EIA's `operating-generator-capacity` dataset (Form 860/860M) is usable** — monthly,
  generator-level, 2008–2026 coverage, with a `status` facet (`OP`/`OS`/`SB`/`OA`). This resolves
  the signal to monthly granularity, not event-day granularity — acceptable given the strategy
  rebalances monthly.
- Fixed a standing bug in `grid_resilience/data/eia_data.py::fetch_plant_capacity()` along the
  way: it was calling a dead API route (404), the wrong frequency (`annual` → HTTP 400 on this
  dataset), and would have silently truncated a full year's data past the API's 5000-row cap
  (ERCOT alone is ~10,600 rows/year). All three fixed and covered by `tests/data/test_eia_data.py`.
- **Entity-name mapping requires manual curation, not fuzzy matching.** Live-verified against
  ERCOT's Jan-2025 EIA data: NRG appears under 3 entity names (`NRG Energy Inc`,
  `NRG Texas Power LLC`, `NRG Cedar Bayou Development Company LLC`); Vistra appears *only* as
  `Luminant Generation Company LLC` — a subsidiary brand with no "Vistra" substring at all. A
  naive string match would silently miss Vistra entirely.
- **Reporting lag is ~90 days, not the 45 days ICR uses.** Live-checked 2026-08-20: the most
  recent available EIA period was 2026-05.

## Decisions

1. **Combine via modulating stress beta, not blending as a separate weighted component.** The
   z-scored stress beta (`beta_z` in `resilience_score.py::build_factor()`) is multiplied by an
   availability ratio (0–1), rather than added as an independent signal with its own weight. This
   directly captures the thesis — availability changes how much a historical stress-beta
   relationship should be trusted right now — without introducing a new weight to tune, and
   without touching `compute_stress_betas()` itself (lower risk than recomputing beta on
   outage-adjusted data).
2. **Availability ratio = 3-month trailing average of `OP_mw / total_mw`**, not a single-month
   snapshot and not a level+momentum pair (the DC-load-signal's two-component shape was
   considered and rejected — there's no clear "momentum" thesis for outages the way there was for
   interconnection-queue growth; a plant coming off `OS` status isn't a directional signal the
   way growing queued MW was). The 3-month average smooths single-month EIA categorization
   quirks without adding real complexity.
3. **Lag: 100 days** (conservative round number off the live-measured ~90-day gap). Point-in-time
   correctness: only EIA months with `period <= as_of - 100 days` are considered knowable as of
   any given rebalance date.
4. **Entity mapping is a manually curated dict**, `TICKER_EIA_ENTITIES`, flagged in-code as
   needing 10-K/ownership validation — the same documented-but-accepted caveat already on
   `TICKER_NODE_MAP`'s own module docstring, not a new risk pattern for this codebase.
5. **Merchant-only, ERCOT-only** — VST and NRG are the only tickers with `pass_through >= 0.5`
   (merchant path) in the active universe, and ERCOT is the only ISO with usable outage data per
   the research pass. Regulated/mixed tickers are entirely untouched by this signal.
6. **Off by default**, matching every other experimental signal in this codebase
   (`BUSINESS_MODEL_ARCH`, `USE_ICR`, `PEER_GROUP_CONSTRUCTION`).

## Architecture

New file `grid_resilience/data/outage_data.py` (parallel in role to `dc_load_data.py`):

- `TICKER_EIA_ENTITIES: dict[str, list[str]]` — curated ticker → EIA `entityName` list:
  ```python
  {
      "NRG": ["NRG Energy Inc", "NRG Texas Power LLC", "NRG Cedar Bayou Development Company LLC"],
      "VST": ["Luminant Generation Company LLC"],
  }
  ```
- `compute_availability_ratio(capacity_df, entity_names, as_of, lookback_months=3, lag_days=100) -> float`
  (`as_of` is the rebalance date being scored):
  1. Filter `capacity_df` to rows where `entityName` is in `entity_names`.
  2. Parse each row's `period` (`"YYYY-MM"`) to that month's end date, and keep only rows whose
     month-end `<= as_of - lag_days`.
  3. Take the most recent `lookback_months` distinct qualifying periods.
  4. For each period, ratio = sum(`nameplate-capacity-mw` where `status == "OP"`) / sum(all
     `nameplate-capacity-mw` across `OP`/`OS`/`SB`/`OA`) for that period; a period with zero total
     MW contributes `NaN` rather than raising (defensive only, not expected in practice).
  5. Return the NaN-skipping mean ratio across those periods (a single anomalous/zero-MW period
     doesn't zero out the whole result). Returns `NaN` if no qualifying periods exist at all
     (e.g. before EIA data starts, or entity list empty).
- `compute_merchant_availability_signal(capacity_df, rebalance_dates, tickers) -> pd.Series` —
  loops `compute_availability_ratio()` per ticker (restricted to `TICKER_EIA_ENTITIES` keys) per
  rebalance date, mirroring the per-date loop shape already used by
  `dc_load_data.py::compute_dc_load_signal()`.
- No new fetch/cache layer — `eia_data.fetch_plant_capacity()` (already fixed) already caches per
  `(iso, year)`; this module calls it once per calendar year spanned by the backtest window and
  concatenates.

**Factor integration** — `grid_resilience/factor/resilience_score.py`, at the point `beta_z` is
computed (currently line 96, upstream of all three arch branches):

```python
beta_z = scores["signed_stress_beta"]

if merchant_availability is not None:
    avail = merchant_availability.reindex(scores.index).fillna(1.0)  # no data → no dampening
    beta_z = beta_z * avail
```

New optional param `merchant_availability: pd.Series | None = None` on `build_factor()` and
`build_rolling_factor()`, threaded through the same way `renewable_share`/`icr`/`pass_through`
already are — `None` = today's exact behavior. Because the multiply happens before the
hard_switch/revenue_mix/dual_track branches, no arch-specific code changes are needed; it flows
through automatically. `fillna(1.0)` means regulated tickers (never in `TICKER_EIA_ENTITIES`) and
any merchant ticker with a data gap are simply unmodified.

## Config changes

`grid_resilience/config.py`:
```python
USE_OUTAGE_SIGNAL: bool = False
OUTAGE_SIGNAL_LOOKBACK_MONTHS = 3
OUTAGE_SIGNAL_LAG_DAYS = 100
```

`main.py` CLI: `--outage-signal` / `--no-outage-signal` (`BooleanOptionalAction`, default
`USE_OUTAGE_SIGNAL`), mirroring the `--icr`/`--peer-group` pattern. When set, fetches EIA capacity
data for ERCOT (via `eia_data.fetch_plant_capacity()`, looped per year), builds the
`merchant_availability` Series via `compute_merchant_availability_signal()`, and passes it to
`build_rolling_factor()`.

## Data flow

```
fetch_plant_capacity("ERCOT", year) × each backtest year
        │  (already cached per-year, already fixed route/frequency/pagination)
        ▼
pd.concat → full-window ERCOT generator-capacity history
        │
        ▼
compute_merchant_availability_signal(capacity_df, rebalance_dates, tickers)
        │  per ticker (VST, NRG only) × per rebalance date:
        │  compute_availability_ratio() → 3-month trailing OP-ratio, lagged 100 days
        ▼
merchant_availability: pd.Series (ticker-keyed per date)
        │
        ▼
build_rolling_factor(..., merchant_availability=merchant_availability)
        │  beta_z *= merchant_availability.reindex(...).fillna(1.0)
        ▼
factor score (flows into hard_switch/revenue_mix/dual_track unchanged downstream)
```

## Error handling / edge cases

- EIA fetch fails entirely (network, API key, schema drift): `fetch_plant_capacity()` already
  returns an empty DataFrame per-year on failure (existing behavior, unchanged). Downstream,
  `compute_availability_ratio()` returns `NaN` for affected tickers/dates, `fillna(1.0)`
  neutralizes it — factor computation silently falls back to unmodified stress beta, never
  crashes.
- Ticker not in `TICKER_EIA_ENTITIES` (any regulated/mixed ticker, or a merchant ticker added
  later without curation): `fillna(1.0)`, no dampening, no error.
- No qualifying EIA periods for a given `as_of` (e.g. very early backtest dates before EIA-860
  coverage or before the lag window clears): `NaN` → `fillna(1.0)`, same graceful no-op.
- All-zero total MW for an entity in a given period (shouldn't occur in practice, defensive only):
  ratio computation guards divide-by-zero, returns `NaN` for that period rather than raising.

## Testing

- `tests/data/test_outage_data.py`:
  - `compute_availability_ratio()`: synthetic capacity_df with known OP/OS/SB/OA rows across
    several months — verify ratio arithmetic, verify the lag window correctly excludes
    too-recent periods, verify only curated `entity_names` rows are counted (unrelated entities
    in the same BA are ignored), verify `NaN` on no qualifying data.
  - `compute_merchant_availability_signal()`: produces one value per (ticker, rebalance date)
    pair for VST/NRG only — no rows for regulated tickers.
- `tests/factor/test_resilience_score.py` additions:
  - `build_factor()` with `merchant_availability` provided: a merchant ticker with
    availability=0.5 gets its `beta_z` exactly halved before arch blending.
  - A ticker absent from `merchant_availability` is unmodified (fillna(1.0) no-op).
  - `merchant_availability=None` reproduces today's exact output — regression safety, same
    pattern as the `grouped=False` peer-group tests.

## Out of scope

- Non-ERCOT merchant exposure — there is none in the active universe today (VST/NRG are ERCOT-
  only), so this is moot unless the universe changes.
- Regulated/mixed tickers — entirely untouched; this signal only ever reads
  `TICKER_EIA_ENTITIES`, which contains VST/NRG only.
- NERC GADS — deprioritized per the research handoff (subscription-restricted, no live check
  performed, EIA-860 already provides a usable free alternative).
- Reserve-margin/scarcity-market data (PJM/ERCOT capacity auctions) — a separate, forward-looking
  thesis from generator-level outages; `gridstatus` exposes some methods
  (`Ercot.get_capacity_forecast`, `PJM.get_operational_reserves`, etc.) not evaluated in this
  pass. Independent idea if pursued later.
- RT/DA LMP spread — separately scoped, `docs/handoff_2026-08-15-rt-da-spread-signal.md`.
- Parameter sweep on lookback-months/lag-days — asserted from reasoning and one live
  measurement, not grid-searched. Candidate future work once the signal is validated in
  backtesting, same pattern as other unswept params flagged elsewhere in this codebase.
