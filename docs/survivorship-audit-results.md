# Survivorship audit — which names dropped out of the universe (2026-10-02)

**Verdict: BUILT (candidate list) · impact NOT YET TESTABLE (no delisted prices).**

## What was done
- Re-pulled every 10-K filer under the 28 SIC codes in `capex_intensity_strategy/data/sic_codes.csv`
  from EDGAR: **2,571 filers, 2,196 with no ticker today** (log item 33 previously said 2,168 / 2,535).
- For each no-ticker filer, read the SEC submissions record: last 10-K date, delisting forms
  (25 / 25-NSE), deregistration (15-12B/G, 15-15D), and 8-K items (1.03 bankruptcy;
  2.01 + 5.01 completed acquisition / change of control). Sized by max `dei:EntityPublicFloat`.
- Kept only filers whose last 10-K is **2016 or later** and who filed no 10-K in 2026 —
  i.e. names that existed inside the backtest window and are missing from it.

Code + output: `research/survivorship_audit/` (`pull.py`, `analyze.py`, `survivorship_candidates.csv`).

## Result
Most of the 2,196 died **before 2016** and are irrelevant to the 2019-26 backtest. Only **229**
dropped out inside the window; **~99** had public float ≥ $100M (plausible liquidity-screen passers).
By SIC family: 107 utilities/pipelines (49xx), 50 electrical equipment (36xx), 28 machinery (35xx),
20 instruments (38xx), 8 coal, 9 fabricated metal, 4 wholesale electrical, 3 power-line construction.

| exit type | count | notable in-window names |
|---|---|---|
| Bankruptcy | 30 | **QualTek Services** (SIC 1623 power-line contractor, 2023), **Briggs & Stratton** (standby generators, 2020), Sunnova (2025), Sunworks (2024), EnSync (2019), Westmoreland Coal / Cloud Peak (2019) |
| Acquired | 111 | **Anixter** (→WESCO 2020), Houston Wire & Cable (2021), Altra Industrial Motion (2023), Aegion (2021), KEMET, AVX, Esterline, CIRCOR, Barnes, Cubic, Velodyne; utilities SCANA, El Paso Electric, ALLETE, Avangrid, South Jersey Inds, Great Plains; Covanta, Pattern, ~20 midstream MLPs |
| Delisted / deregistered / stopped filing | 88 | mostly MLP roll-ups, SPACs and micro-caps |

**Direction of bias.** Acquisitions outnumber bankruptcies ~111 : 30, and acquired names carry
missing takeover premiums — so the bias is not purely "losers removed". Bankruptcies are
concentrated in small solar / contractor / coal names (short-book territory), so the clearest
effect is still an **understated short leg**; the long book (suppliers) loses mostly *winners*
via takeover (Anixter, Altra, HWCC, KEMET).

Several commonly cited failures (SunPower, Enviva, Li-Cycle, Charge Enterprises) are **not** in
the list — they file under SIC codes outside the universe, so they were never eligible and cause
no bias here.

## Caveats
- Exit labels are inferred from 8-K item codes and are imperfect: Calpine, Dynegy and Arch show as
  "bankruptcy" from old/subsidiary events (actual exits: take-private / mergers); NorthWestern and
  FREYR (now T1 Energy) re-registered under new CIKs and still trade; some entries are subsidiaries.
- Float values contain XBRL unit errors (e.g. QualTek ~$113tn) — size is indicative only.
- No price history for these names, so the return impact is not measured. Fix still needs a
  delisted-inclusive source (CRSP or equivalent) — log item 33.

---

## Part 2 — rebuild with the dropouts in the pool (2026-10-02)

**Verdict: BUILT · survivorship effect on the one-score long/short is SMALL (|ΔSharpe| ≤ 0.07 in every window), sign mixed.**

### Method
- Old tickers: 10-K `dei:TradingSymbol` + a hand map (`research/survivorship_audit/hand_tickers.csv`), each accepted only if
  Tiingo's listing end-date falls within 2 years of the filer's last 10-K. Prices from Tiingo (free tier keeps delisted
  symbols). Each series is cut at the exit (8-K/Form-25 date if inside the window, else 13 months after the last 10-K,
  and at the first >10-day trading gap — guards against re-used tickers such as B→Barrick, SE→Sea, DYN→Dyne).
- **57 dropouts priced and alive after 2019-01** (40 acquired, 10 delisted, 7 "bankruptcy"-labelled — mostly mergers in
  fact). Added to a *shadow* copy of every cache (`_shadow/`): SIC candidates, prices + dollar volume, EDGAR facts,
  10-K vocabulary hits. Production caches untouched.
- The whole production chain is rebuilt on the shadow (`run_compare.py`): SIC/liquidity universe → screen_auto →
  A1+A4 members → 14-signal score (fundamental panel and power betas recomputed) → buffered top-half long, rule-2
  bottom-quintile short, overlays, step-aside. One change vs production, applied to both runs: a name must have a real
  (not forward-filled) price on the rebalance day to be picked, so a dropout can't be chosen after it stops trading.
- **Control:** the same rebuild with no dropouts (`_shadow_base/`) reproduces production **exactly** (Sharpe 1.341 /
  0.436 / 2.053 with step-aside, 1.376 / 0.788 / 2.053 without; members pickle 0 mismatched cells).

### Result (control → with dropouts)
| book | window | Sharpe | CAGR | MaxDD | Calmar |
|---|---|---|---|---|---|
| with step-aside (production) | full | 1.341 → 1.356 | 20.8% → 20.5% | −11.1% → −10.4% | 1.88 → 1.98 |
| | 2019-22 | 0.436 → 0.494 | 7.9% → 8.1% | −11.1% → −8.6% | 0.71 → 0.94 |
| | 2023+ | 2.053 → 2.001 | 36.8% → 35.8% | −10.2% → −10.4% | 3.62 → 3.46 |
| without step-aside | full | 1.376 → 1.398 | 25.1% → 25.2% | −13.9% → −16.5% | 1.81 → 1.52 |
| | 2019-22 | 0.788 → 0.858 | 15.4% → 16.2% | −13.9% → −16.5% | 1.11 → 0.98 |
| | 2023+ | 2.053 → 2.001 | 36.8% → 35.8% | −10.2% → −10.4% | 3.62 → 3.46 |

Alpha vs GRID 2023+: 21.7%/yr (t 3.86) → 20.8%/yr (t 3.72). Shorted basket return (full) 6.4% → 4.9%/yr.
Names held: long 9.9 → 10.3, short 16.4 → 17.6.

### Which dropouts were picked
- **21 of 57** became eligible at some point; almost all are utilities, pipelines and coal — capex-heavy, so they fail
  the long screen and feed the **short** pool.
- **Long book:** only **Barnes Group** (628 days, taken private by Apollo Jan-2025, ~+17% summed contribution) and
  Esterline (63 days, ~0). Anixter, Altra, Aegion etc. were never long — they fail the screen/vocabulary rules.
- **Short book:** ALLETE (10 quarters), Arch (9), Avangrid (7), Barnes (7), DCP (5), Crestwood (5), South Jersey (3).
  Most were later **taken over at a premium** (ALLETE, Avangrid, SJI, DCP, Crestwood) — survivorship here removes
  takeover targets from the short, not bankruptcies.
- **Step-aside matters for the bias:** with it, the dropouts *lower* the 2019-22 drawdown (−11.1% → −8.6%); without
  it, they *deepen* it (−13.9% → −16.5%) — the extra short names hurt in the 2020-22 stretch the rule sits out.

### Gaps (direction: understates the short leg's gain from failures)
Not priced: **Sunnova** (Tiingo serves the pink-sheet NOVA), **QualTek, Sunworks, Cloud Peak, Azure Midstream**
(no Tiingo history), TECO / Targa RP (reused or missing tickers). These are the real bankruptcies in the window,
so the measured short-side effect is a lower bound on what failures would have added. GTI excluded (wrong issuer).

Code: `research/survivorship_audit/` (`map_tickers.py`, `finalize_map.py`, `fetch_prices.py`, `build_shadow.py`,
`run_compare.py`; outputs `compare_base.json`, `compare_shadow.json`).

---

## Part 3 — rerun on the cleaned screen (2026-10-13)

**Verdict: BUILT · 2023+ effect SMALL (2.277 → 2.231); 2019-22 effect LARGE and negative (0.391 → −0.338) — cause not yet decomposed.**

### Method
Same 57 priced dropouts, rebuilt from today's production caches, which carry both screen fixes (commit 4bb636f).
`build_clean_shadow.py` writes `_shadow_clean_base/` (control, no dropouts) and `_shadow_clean/` (with dropouts).
The production data-center exclusion list never saw the dropouts' own filings, so every dropout "data center(s)" hit
was checked against that 10-K's Item 1 (`clean_dc_dropout_audit.csv`: 65 dropped, 17 kept, 1 kept for lack of
Item 1). The 10-K fetch window was widened to 2012 for the dropouts only (production's `tenk.filings` starts at
2017-06; Barnes's 2017 hit would otherwise have admitted it for 2019). All 13 Barnes hits now fail, so Barnes is out
of both books. Gas T&D dropouts (DCP, Crestwood, South Jersey; SIC 4922-4924) fall out of the short pool via
`RULE2_EXCLUDED_SICS`. Control reproduces production exactly (1.433 / 0.391 / 2.277).

### Result (control → with dropouts)
| book | window | Sharpe | CAGR | MaxDD | Calmar |
|---|---|---|---|---|---|
| with step-aside (production) | full | 1.433 → 1.139 | 21.3% → 16.3% | −12.1% → −11.7% | 1.76 → 1.40 |
| | 2019-22 | 0.391 → −0.338 | 7.4% → 1.0% | −12.1% → −11.7% | 0.61 → 0.09 |
| | 2023+ | 2.277 → 2.231 | 38.6% → 35.8% | −8.5% → −8.5% | 4.57 → 4.24 |
| without step-aside | full | 1.450 → 1.191 | 25.8% → 20.7% | −17.1% → −17.6% | 1.51 → 1.18 |
| | 2019-22 | 0.774 → 0.362 | 15.1% → 8.4% | −17.1% → −17.6% | 0.88 → 0.48 |
| | 2023+ | 2.277 → 2.231 | 38.6% → 35.8% | −8.5% → −8.5% | 4.57 → 4.24 |

Alpha vs GRID 2023+: 24.5%/yr (t 4.30) → 22.9%/yr (t 4.19). Shorted basket return (full) 4.1% → 2.3%/yr.
Names held: long 8.5 → 8.5, short 13.3 → 13.9.

### Which dropouts were picked
- 21 of 57 eligible at some point. **Long:** only Esterline (63 days in 2019, ~0 contribution).
- **Short:** Avangrid (7 quarters, 2019-22), ALLETE (8; 2018-20 and 2024), Arch (9; 2018-20 and 2023-24), Pattern
  Energy (1, 2019-Q1).
- Long-screen membership changes in only 9 cells (Esterline 2019, AZZ 2025), so the 2019-22 damage does not come from
  the screen. Candidates: the dropouts change the cross-sectional 14-signal scores (re-ranking live names in both
  books) and take short-basket slots in 2019-20. Not yet separated.

### Comparison with Part 2
The original-screen audit showed 2019-22 *up* (0.436 → 0.494). That gain leaned on names the clean screen removes:
Barnes long (~+17% summed) and the gas-pipeline shorts (DCP, Crestwood, South Jersey). Without them the dropouts'
net effect on 2019-22 is strongly negative. The 2023+ conclusion (small) holds on both screens.

### Gaps (unchanged)
Unpriced real bankruptcies: Sunnova, QualTek, Sunworks, Cloud Peak. All four carry grid-customer wording every
relevant year (they pass rule 2), so each was short-eligible subject to liquidity; Sunnova (liquid 2020-25) most
likely. Being short them only helps, so the short-side effect here is a lower bound.
