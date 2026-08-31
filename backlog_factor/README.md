# Backlog Growth Surprise — Cross-Sectional Event Study (Phase 1)

Spec: `../docs/superpowers/specs/2026-08-31-backlog-surprise-factor-design.md`
Results: `../docs/backlog-surprise-factor-results.md`

## Status: negative result — the pre-registered gate FAILED. No factor was built.

## What this is

A breadth-first test of a single idea: does a name's quarterly backlog / RPO growth **surprise**
— the part of this quarter's backlog change that a naive extrapolation of the company's own recent
order trend would not predict — drift into industry-adjusted stock returns over the following weeks?

`surprise_t = g_t − mean(g_{t-4..t-1})`, where `g_t = log(RPO_t / RPO_{t-1})`. This is the
differentiated version of the `grid_equipment_basket` Step-2 tilt, which ranked names on *raw*
backlog growth (mostly the trend the market already extrapolated) and failed.

## Universe

**109 US-listed order-driven industrials** that tag XBRL `RevenueRemainingPerformanceObligation`,
curated from 207 in-scope filers discovered via the SEC `frames` API: machinery, aerospace & defense,
engineering & construction, electrical equipment, semiconductor capital equipment, building products.
Medical / lab names (whose RPO is service-contract deferred revenue), pre-revenue SPACs, autos, and
software-heavy names were dropped — see `candidate_research.md` for every decision.

## Result

Window 2019-01-01 → 2026-07-31. 91 of 109 names contribute **1,778 scored (name, filing-date) pairs**
after the ≥6-quarter history + clean-span guards — the intended cross-sectional breadth.

The pre-registered gate (spec §5: Q5−Q1 abnormal CAR positive **and monotone**, month-clustered t > 2,
holds in both sub-periods, right sign in ≥ 3 of 5 industry groups) **fails**:

- Q5−Q1 abnormal CAR is **+1.27% at 5 days** but **not monotone** across quintiles, and it **reverses**
  to −1.03% by 63 days.
- Direction carries no information — negative-surprise names drift up *more* over 63 days than
  positive-surprise names.
- Only machinery shows a clean positive Q5−Q1 (t 2.82); electrical and semi-cap equipment are negative.

Per spec §5 this ends the project: no monthly factor, no Phase-2 hand-collection. Full numbers, the
positive/negative split, the by-industry breakdown, and the caveats are in the results doc.

## Run it

```
python -m backlog_factor --event-study [--start 2019-01-01] [--end YYYY-MM-DD] [--output DIR]
```

Fetches RPO (SEC XBRL, cached per CIK) and prices (yfinance, monthly-parquet cache), computes the
abnormal-CAR quintile table and the gate verdict, and writes `event_pairs.csv`,
`quintile_car_table.csv`, `gate.json` to the output dir.

## Layout

`config.py` (universe + constants) · `data/rpo.py` (XBRL discovery + fetch) · `data/prices.py`
(price fetcher, mirrors `grid_equipment_basket`) · `signal.py` (surprise series + guards +
cross-sectional neutralize/decay helpers) · `event_study.py` (CAR windows, quintile CARs,
month-clustered t, gate) · `__main__.py` (CLI) · `candidate_research.md` (curation log).
