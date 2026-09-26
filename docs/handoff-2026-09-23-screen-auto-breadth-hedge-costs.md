# Handoff — `screen_auto`: breadth, a funded hedge, and net-of-cost numbers

**Opened:** 2026-09-23 · **Status:** scoped and approved, not started
**Prereq reading:** `docs/screen-auto-mechanical-universe-results.md`,
`docs/book-master-comparison-results.md`, `docs/capex-vs-screened-long-book-results.md`

---

## 0. Where things stand

`screen_auto` is BUILT and mechanical — no seed list, no judgment column (see
`electrification_strategy/auto_screen.py`, 13 tests; 597 tests pass repo-wide). It beats the
shipped `screen` on Sharpe in all three windows (full 1.130 vs 1.000) but:

- **missed the pre-registered gate in 2023–2026** (MaxDD −3.55pp against a 3pp bar);
- **is concentrated** — ~19 names/month, **14 effective**, and only **4 names in January 2019**;
- **overlaps `screen` on 53% of names** — it is a better book, not a reconstruction of that one.

The best book in the master comparison is `screen_auto` ∩ capex-light *quartile*
(Sharpe 1.234 / Calmar 1.172) but it averages ~10 names and has never been gated.

**Two settled negatives to not re-litigate:**
1. **The capex-heavy leg is not a hedge.** It has positive drift and correlates **+0.66** with
   `screen_auto` (**+0.60 on the long book's own down days**). Shorting it at 0.25× or 0.5× lowers
   Sharpe for every long book tested. A hedge that funds itself needs **negative alpha**, not
   capital intensity.
2. **No long/short beat the best long-only** in any window, and **every short-only book lost money
   in every window** (Sharpe −0.31 to −0.90, MaxDD to −57%).

## 0.1 Cached inputs (regenerate only if stale)

| what | where | cost to rebuild |
|---|---|---|
| EDGAR facts (revenue, capex, net income; YTD ladder) | `capex_intensity_strategy/data/cache/edgar_facts.parquet` | ~10 min |
| capex score + liquidity mask + prices | `pipeline.build(start, end)` | ~4 min |
| vocabulary hits (1,252 rows) | `electrification_strategy/data/vocabulary_hits.csv` | ~30 min (`build_auto_screen --refresh`) |
| `screen_auto` membership (1,735 rows) | `electrification_strategy/data/screen_auto_members.csv` | ~4 min (`build_auto_screen`) |

---

## Part A — breadth (do first; all inputs cached)

Measure each lever **alone and stacked**, C2-style: names/month, Sharpe, Calmar, MaxDD on the
shipped overlay, full window plus both sub-windows. **Keep only levers that do not cost Sharpe.**

| id | lever | config | why | note |
|---|---|---|---|---|
| **A1** | lookback 2 → 4 years | `AUTO_LOOKBACK_YEARS` | a 2017 10-K then covers universe-years 2018–2021; targets the 4-name Jan-2019 book directly | no refetch needed — `vocabulary_years` re-derives from the cached hit table |
| **A3** | policy exclusion fires at ≥2 hits | new `AUTO_POLICY_MIN_HITS` | one passing mention of "investment tax credit" is not a dependency; recovers ~2.5 names/mo (S3 22.1 → S4 19.6) | cheap |
| **A4** | capex quantile 0.5 → 0.6 | `AUTO_CAPEX_QUANTILE` | widens the light cut | cheap |
| **A2** | forms "10-K" → "10-K,10-Q" | `AUTO_FORMS` | 4× the filings, so vocabulary becomes visible sooner | **~30-min refetch; 10-Q language is looser and may add false positives. Only pay for this if A1/A3/A4 leave breadth short.** |

**Breadth target:** no month below ~10 names, and effective N above ~20 at the last rebalance
(currently 14). Note from the C4 diagnostics that a naive ≥10-name *trading* guard COSTS Sharpe
(1.128 → 1.078) — the point of Part A is to earn breadth from the rule, not to gate the book.

## Part B — explore a funded hedge (no pass bar yet, by decision)

Characterise each candidate before setting a bar. For each, report: standalone CAGR/Sharpe/MaxDD,
correlation to `screen_auto`, **correlation on the long book's own down days**, drawdown
contribution, and what swapping 5pp of the GLD/IEF sleeve for it does to the combined book.

| id | candidate | rationale | prior |
|---|---|---|---|
| **B1** | capex-heaviest ∩ persistently unprofitable | firms funding heavy capex out of losses — a negative-alpha screen, newly possible now that `NetIncomeLoss` is in the point-in-time panel | untested |
| **B2** | policy-dependent ∩ unprofitable (ITC/PTC/NEM hits) | the names the long screen discards | **bad prior** — `docs/electrification-short-leg-insurance-probe-results.md` found the DER sleeve was an *anti*-hedge in the feared solar-squeeze scenario and bled −45%/yr as a crowded short |
| **B3** | heaviest quartile with *deteriorating* capex coverage | a trend version of B1 rather than a level | untested |

Current insurance to beat: **15% GLD/IEF sleeve**, ~3.3%/yr drag (≈15%/yr inside a rate-rising
bull). The goal is the same drawdown protection at less give-up, letting the sleeve shrink to ~10%.
**Set the pass bar with the user once B1–B3 are characterised.**

## Part C — net of costs (do last; membership must settle first)

1. Turnover per book (annualised, both legs) for every row of `docs/book-master-comparison-results.md`.
2. Apply `capex_intensity_strategy.config.COST_PER_TURNOVER` (10bp each way) and re-run the master
   table **net**. Expect the intersection books to lose most — they turn over on either rule.
3. Add a **hysteresis** variant: a name must fail the screen twice before it leaves. Report the
   turnover saved and the Sharpe cost.

---

## Open decisions for the user

- **Part A:** whether to pay the ~30-minute refetch for A2 if A1/A3/A4 fall short on breadth.
- **Part B:** the pass bar, to be set after B1–B3 are characterised.
- **Part C:** whether to also gate the ~10-name intersection book, which currently tops the
  comparison but has never been through a pre-registered gate.

## Conventions to preserve

- Every filter reads only filings published **before** the universe-year it decides (`[Y-2, Y-1]`,
  or `[Y-4, Y-1]` under A1).
- Network access stays injected so `auto_screen` tests run offline.
- `screen_auto` stays **out of** `universe.CONSTRUCTIONS` (the default grid) and in
  `ALL_CONSTRUCTIONS`, so a plain CLI run never depends on its build artifact.
- TDD: a failing test before each production change. Update `docs/RESEARCH-LOG.md` §3 at every
  verdict, blunt pre-registered language only.
