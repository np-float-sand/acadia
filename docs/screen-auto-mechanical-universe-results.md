# `screen_auto` — a mechanical replacement for the hand-recorded seed list

**Date:** 2026-09-23 · **Verdict:** **BUILT + PASSES the pre-registered gate on 2 of 3 windows**
(full and 2019–2022), **FAILS 2023–2026 on the MaxDD leg by 0.55pp while beating it on Sharpe**.
It does **not** reproduce the `screen` names (53% overlap) — it produces a different, better book.

---

## 1. What was replaced

`screen` picks names from a 33-row CSV whose membership and three boolean columns were recorded
by hand in 2026 and applied backwards to 2017. The columns are nearly a no-op — 30 of 33 names
pass `customer_institutional` — so the real hand-work is **seed-list membership**, which
`electrification_strategy/README.md` already flags as a pre-live gate.

| hand-recorded input | mechanical replacement | module |
|---|---|---|
| the 33-name seed list | SEC SIC population + fuel-cycle lexicon + $10M liquidity | `capex_intensity_strategy.universe` |
| (implicitly) "not a utility" | TTM capex/revenue in the light half | `auto_screen.light_capex_mask` |
| `customer_institutional` | buildout vocabulary in the prior-year 10-K | `auto_screen.vocabulary_years` |
| `earnings_valued` | TTM `NetIncomeLoss` > 0, `filed`-date-aware | `auto_screen.profitable_mask` |
| `low_policy_dependence` | ITC / PTC / NEM / 45X vocabulary **flags** a name | same |

Every filter reads only filings published **before** the universe-year it decides — the
`[Y-2, Y-1]` convention already used by `FUEL_CYCLE_LEXICON`.

---

## 2. C1 — can the vocabulary reach the screened names?

**Recall: 27 of 28.** Only THR (Thermon) is missed. **Precision is the problem:** 53 non-screen
filers also hit, including the general industrials the screen exists to exclude.

Tightening the vocabulary trades recall for precision roughly one-for-one and never gets clean:

| variant | names kept | screen names kept | extras | missed |
|---|---|---|---|---|
| any customer term, ≥1 filing | 80 | **27/28** | 53 | 1 |
| any customer term, ≥3 filings | 63 | 22/28 | 41 | 6 |
| grid-practitioner terms only (substation / switchgear / T&D) | 39 | 16/28 | 23 | 12 |
| grid term AND data-centre term | 33 | 16/28 | 17 | 12 |

Grid-practitioner vocabulary drops exactly the `dc_thermal` and equipment names (AEIS, AME, ATKR,
AYI, CARR, EMR, JCI, MOD, NVT, SPXC, TT) while still keeping utilities and industrials. So the
broad vocabulary was kept and the discrimination left to the other filters — which is the
`capex-intensity-proposal-v2.md` lesson restated: "data center" is a *customer* word, not
practitioner vocabulary, and is only usable in combination.

---

## 3. C2 — does each filter keep the Sharpe?

Shipped overlay (20% vol target + valuation de-lever + 15% GLD/IEF sleeve), 2019-01 → 2026-08.
**Every filter adds Sharpe, monotonically.**

| stage | names/mo | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| S0 universe (capex-eligible, EW) | 108.9 | 17.5% | 0.787 | −22.8% | 0.766 |
| S1 + capex-light half | 54.5 | 20.1% | 0.918 | −22.8% | 0.881 |
| S2 + customer vocabulary | 29.2 | 21.1% | 0.981 | −22.0% | 0.959 |
| S3 + profitable (TTM NI > 0) | 22.1 | 23.6% | 1.109 | −21.0% | 1.124 |
| **S4 + policy exclusion = `screen_auto`** | 19.6 | 24.0% | **1.128** | −20.6% | **1.165** |
| — shipped `screen` (target) | 24.7 | 21.0% | 1.000 | −23.2% | 0.905 |

Through the package's own `run_comparison` path, with POWER-group names excluded:
`screen_auto +val+sleeve` = CAGR 23.5%, **Sharpe 1.132**, MaxDD −19.8%, Calmar 1.188.

---

## 4. C3 — the pre-registered gate

Bar agreed before the run: **Sharpe within 0.05 of `screen`, MaxDD no more than 3pp worse.**

| window | `screen_auto` Sharpe / MaxDD | `screen` Sharpe / MaxDD | ΔSharpe | ΔMaxDD | verdict |
|---|---|---|---|---|---|
| full | 1.128 / −20.6% | 1.000 / −23.2% | +0.128 | +2.59pp | **PASS** |
| 2019–2022 | 0.623 / −19.7% | 0.515 / −23.2% | +0.108 | +3.47pp | **PASS** |
| 2023–2026 | 1.700 / −20.6% | 1.595 / −17.0% | +0.105 | −3.55pp | **FAIL** |

The 2023–2026 failure is the MaxDD leg missing by 0.55pp, in a window where `screen_auto` earns
a higher Sharpe and 4.3pp more CAGR. Reported as a fail because that is what was pre-registered.

---

## 5. C4 — point-in-time sanity, and what it exposed

| | 2019-01 | 2020-01 | 2021-01 | 2022-01 | 2023-01 | 2025-01 | 2026-07 |
|---|---|---|---|---|---|---|---|
| names | **4** | 12 | 13 | 20 | 22 | 28 | 30 |

Two real problems:

1. **The early book is too thin to trade.** Four names in January 2019, 9–13 through 2021. This is
   honest point-in-time behaviour — "data center" was rare vocabulary in 2017–18 filings, so the
   rule genuinely could not see these names then — but a 4-name book is not investable.
2. **Midstream leakage.** The January 2019 book contains ET and OKE: pipelines that are
   capex-light in a thin pool and whose 10-Ks mention data centres as customers.

Overlap with `screen` averages **10.3 of 19.6 names (53%)**; only GEV is never picked. So this is
not a reconstruction of the hand list — it is a different book that happens to be better.

### Diagnostics (not pre-registered)

| variant | full Sharpe | 2019–22 Sharpe | 2023–26 Sharpe |
|---|---|---|---|
| `screen_auto` as pre-registered | 1.128 | 0.623 | 1.700 |
| + exclude POWER-group SIC | **1.130** | 0.652 | 1.678 |
| + require ≥10 names to trade | 1.078 | 0.483 | 1.700 |
| + both | 1.036 | 0.429 | 1.678 |

- **POWER exclusion is free and principled** — Sharpe 1.128 → 1.130. Adopted: utilities,
  pipelines and generators are off-thesis by construction, not by performance.
- **The breadth guard COSTS Sharpe** (1.128 → 1.078 full; 0.623 → 0.483 in 2019–22). The thin
  early book was *helping*. That is a concentration-luck warning, not an endorsement: the
  2019–2022 edge rests partly on a 4–13 name book. The guarded version (Sharpe 1.036–1.078) is
  the conservative number to trust, and it still clears `screen`'s 1.000.

Effective N at the last rebalance: **`screen_auto` 14.0 vs `screen` 27.0.**

---

## 6. What ships

`screen_auto` is an **available construction, not the default** — it is excluded from
`universe.CONSTRUCTIONS` (the default comparison grid) and listed in `ALL_CONSTRUCTIONS`, so a
plain CLI run never depends on its build artifact.

- `electrification_strategy/auto_screen.py` — the filters, network injected, 13 tests
- `electrification_strategy/build_auto_screen.py` — `python -m electrification_strategy.build_auto_screen [--refresh]`
- `data/vocabulary_hits.csv` (1,252 hits) and `data/screen_auto_members.csv` (1,735 rows) — **build
  artifacts**, reproducible from filings, committed only to avoid a 30-minute EDGAR round trip
- `capex_intensity_strategy/edgar.py` — `filer_mention_dates` / `parse_mention_dates` (CIK-filtered
  full-text search, one request per filer-phrase instead of one per year), `NetIncomeLoss` tag
- `capex_intensity_strategy/fundamentals.py` — `ttm_panel` is now item-agnostic

597 tests pass.

---

## 7. Caveats

1. **The gate failed in 2023–2026.** On the agreed bar, this is not an unqualified pass.
2. **Concentration.** 14 effective names vs 27, and the early-window edge disappears under a
   breadth guard.
3. **The vocabulary list is a choice.** The seven customer terms were fixed before the run and not
   tuned against results, but "which terms" carries selection risk the same way the capex metric
   choice does.
4. **Not the same names.** 53% overlap with `screen`. Anyone expecting a mechanical restatement of
   the hand list will not find it here.
5. **Survivorship** — yfinance serves no delisted names; affects every book in this repo equally.
