# Grid-Stress (PJM / ERCOT ISO) Signal — Consolidated Evidence

**Compiled:** 2026-09-06
**Scope:** the ISO grid-stress signal at the core of the `grid_resilience/` strategy —
LMP prices + congestion + reserve tightness + named stress events (ERCOT, PJM, MISO,
CAISO, SPP; ISO-NE / NYISO in the universe) → a per-ISO **Grid Stress Index (GSI)** →
each utility's **conditional stress beta** → cross-sectional factor score.

**Purpose:** one place that pulls together every piece of evidence gathered on this
signal (June 2026 → August 2026), previously scattered across `output/findings.md`,
three session compacts, `docs/pjm-signal-handoff.md`, `CLAUDE.md`, and memory.

---

## 0. Bottom line (the result)

**The signal is real and physically motivated but weak, and it has not survived an
honest test.** It does not clear a bar that would justify shipping it as a standalone
edge:

- Cross-sectional **IC ≈ +0.14 (t ≈ 1.9)** on the clean non-PJM universe — positive but
  not significant at any conventional level, and no better in stressed months than calm.
- Adding **PJM** (six rate-regulated T&D names) collapses it to **IC ≈ 0.07, Sharpe ≈ 0**;
  a per-ticker zone-GSI fix made it *worse*, not better.
- On **corrected data** (post the 2026-08-13 PJM cache-gap fix) **no configuration beats
  XLU** — Sharpe ≈ −0.21 to −0.28 vs XLU 0.16.
- The 2023–2025 instability traces almost entirely to **two merchant generators (VST,
  NRG)** whose equity left the "utility" regime and got re-rated on the AI-power
  narrative; the rolling-OLS beta lagged each flip by ~1 quarter and whipsawed.
- The 2026-08-26 closeout audit tried **every angle** — signal formulation, portfolio
  construction, universe, factor validation — and concluded the constraint is
  **sample size / signal strength**, not construction or data engineering: the 2018–2025
  window has too few genuine stress regimes for a monthly cross-sectional factor to work.

**Status:** not adopted as a live edge. The strategy code keeps it (grid-search-frozen
params, `--arch hard-switch` routing PJM names to an ICR/DC-queue signal instead), but
the honest framing is "risk-managed L/S with a weak underlying signal," not "validated
grid-stress alpha."

---

## 1. What the signal is

Pipeline (`grid_resilience/`, see repo-root `README.md` for the full mechanism):

```
[1] Equity prices     → yfinance (adjusted daily closes, ~18-name utility universe)
[2] Grid data         → gridstatus (hourly LMP + load per ISO)
[3] Stress events      → named catalog + algo-detected LMP spikes
[4] Grid Stress Index  → composite score per ISO per day, [0, 1]
[5] Stress betas       → rolling OLS of stock excess return on GSI
[6] Factor scores      → cross-sectional rank of negative stress beta (+ renewable adj.)
[7] Backtest           → monthly rebalance, 3 long / 5 short, P&L → Sharpe / drawdown
```

**GSI sub-signals and weights:** LMP z-score **40%**, congestion fraction **25%**,
reserve tightness **20%**, named-event flag **15%**.

**Universe:** ~18 U.S. utilities across ERCOT, PJM, MISO, CAISO, SPP, ISO-NE, NYISO
(expanded to 20 with CEG / TLN in the 2026-08-26 session; merchant sleeve 2→4).

**Frozen params** (grid-search optimised 2026-06-17, live in `config.py`):
`STRESS_SPIKE_PCT=0.97`, `STRESS_CONG_THRESHOLD=0.30`, `STRESS_CONG_MIN_DAYS=10`,
`PORTFOLIO_LONG_N=3`, `PORTFOLIO_SHORT_N=5`, `XLU_HEDGE=False`, `USE_ICR=False`.
Backtest window `2018-01-01` → `2025-12-31`.

---

## 2. Evidence — IC statistics

### 2.1 Overall cross-sectional IC (`output/findings.md` §2, §8)

| Metric | Value |
|---|---|
| Overall mean IC | **+0.14** |
| Stressed months (n=64, 21-day mean GSI > 0.5) | mean IC **+0.137**, 55% positive |
| Calm months (n=26) | mean IC **+0.169**, 62% positive |
| Stressed − calm difference | **−0.032** (calm is *slightly* better) |
| Welch t-test (stressed vs calm) | **p = 0.856** — not significant |

The original pitch's "the factor works better in stress" claim is **not supported** — the
0.5 GSI threshold was too low (65 of 90 rebalance dates exceed it), which created a
misleading all-orange IC panel. The stressed/calm split was dropped from the executive
summary.

### 2.2 IC by config (`docs/compact_2026-06-28.md`, `docs/compact_2026-07-05.md`)

| Configuration | Sharpe | Ann Ret | Max DD | IC@21d | IC t-stat |
|---|---|---|---|---|---|
| Without PJM (ERCOT/MISO/CAISO/SPP) | **0.272** | 7.99% | −15.41% | **0.144** | **1.918** |
| Hard-switch arch, **PJM included** | **0.331** | 9.38% | −19.53% | 0.1434 | 1.865 |
| Revenue-mix arch, PJM included | 0.185 | 6.64% | −14.70% | 0.1340 | 1.702 |
| Dual-track arch, PJM included | 0.031 | 4.44% | −19.02% | 0.1295 | 1.652 |
| With PJM — hub-only GSI (`--no-zone-gsi`) | −0.027 | 3.69% | −15.67% | 0.067 | 1.367 |
| With PJM — per-ticker zone GSI | −0.009 | 3.89% | −16.89% | 0.077 | 1.608 |

**Read:** the best IC t-stat anywhere is **1.9** (below the conventional |t| ≥ 2 bar).
Business-model routing (hard switch) recovers headline Sharpe to 0.331 by sending the
PJM T&D names to a *different* signal (ICR / DC-queue) — it does **not** rescue the
grid-stress signal on those names.

### 2.3 2024 IC breakdown (`output/findings.md` §3–§7)

| Metric | Value |
|---|---|
| 2024 mean IC | **−0.023** (effectively zero) |
| Corr(factor score end-2023, 2024 full-year returns) | **−0.727** (near-perfect inversion) |
| VST factor score entering 2024 | −1.43 (model short) → VST **+262%** in 2024 |
| NRG factor score entering 2024 | −1.34 (model short) → NRG **+79%** in 2024 |

### 2.4 Price-only vs blended GSI (`docs/compact_2026-08-26...` thread 5)

| | Blended GSI | Price-only GSI (100% LMP z-score) |
|---|---|---|
| Full Sharpe | −0.321 | **−0.585** |
| Long Sharpe | +0.201 | **+0.014** |
| IC@21d | −0.0003 | −0.0026 |
| IC@63d | −0.0351 | −0.0383 |

Isolating the price leg is **worse** — congestion / reserve / event components *add*
information to the blend rather than diluting a cleaner price signal (opposite of the
hypothesis that motivated the test).

### 2.5 Orthogonality check (`docs/compact_2026-08-26...` thread 7)

Factor score (per-ticker average, 20-name universe) vs standard factors:

| Factor | corr | p-value |
|---|---|---|
| ROE | −0.141 | 0.553 |
| Debt/Equity | +0.058 | 0.807 |
| Dividend yield | −0.132 | 0.590 |
| Market beta | +0.287 | 0.219 |

Not a repackaged quality / leverage / yield / low-vol factor (n=20 rules out a *strong*
repackaging, not a moderate one under low power).

---

## 3. Evidence — performance scoreboards (all configs tried)

### 3.1 PJM integration (`docs/pjm-signal-handoff.md`)

| State | Sharpe | Ann Ret | Max DD |
|---|---|---|---|
| Without PJM (ERCOT/MISO/CAISO/SPP) | **0.315** | 8.03% | −13.66% |
| With PJM — broken data (rate-limit cache gaps) | −1.677 | −5.92% | −22.66% |
| With PJM — data fixed | −0.043 | 3.53% | −22.25% |
| Grid-search peak (no PJM) | **0.319** | 8.1% | −13.7% |

### 3.2 Corrected-data re-examination (`docs/compact_2026-08-26...`)

Post the 2026-08-13 PJM cache-gap fix, starting state trailed XLU outright:
**Sharpe ≈ −0.21 to −0.28 vs XLU 0.162.**

| Angle tried | Baseline | Result | Verdict |
|---|---|---|---|
| Peer-group (basket-vs-basket) construction | −0.276 | −0.343 | worse; vol/DD improved but excess return still negative |
| Universe expansion (merchant sleeve 2→4: +CEG, +TLN) | −0.299 | −0.321 | no improvement |
| Price-only GSI | −0.321 | −0.585 | worse (see §2.4) |
| Outage / reserve-margin availability signal | — | killed | EIA-860 status codes are annual-scale; `OP` ratio = 1.0 for VST/NRG even in Feb 2021 (Uri) |
| RT/DA LMP spread | — | not built | ERCOT practical; PJM blocked (downloads full nodal network, 429s) — and PJM is where the names are |
| FTR clearing price | — | incomplete | PJM feed is bid-level only; needs a bespoke report scrape |

**Closeout conclusion (verbatim intent):** none of these beat XLU; the short book has no
edge over a naive short in every configuration; the underlying grid-stress signal is
**real but weak**, and 2018–2025 contains **too few genuine stress regimes** for a
monthly cross-sectional factor — a sample-size / signal-strength problem, not a
construction, mislabeling, or fixable-data problem.

### 3.3 Benchmark framing (`output/findings.md` §1)

| | Strategy | XLU buy-and-hold |
|---|---|---|
| Cumulative return (Apr 2018 – Dec 2025) | +90% | +113% |
| Max drawdown | **−13.7%** | −36.1% |
| Annual vol | **13.2%** | 20.6% |
| Sharpe (backtest method) | **0.31** | 0.24 |

Risk-adjusted it edges XLU with 2.6× less drawdown — but as a *complement* to a long-only
utility book, not a standalone alpha source.

---

## 4. The PJM problem (why the ISO signal breaks on T&D names)

- `config.py:ISO_BENCHMARK_NODES["PJM"]` pulls four system hubs; `build_multi_iso_gsi()`
  builds **one** GSI for all of PJM, so AEP / D / EXC / FE / PPL (+ PEG) all see the
  identical stress series. The only differentiator is `stress_beta`, and PJM is a
  13-state RTO — system-wide events hit all utilities similarly → betas cluster → factor
  score std 0.718 (vs ERCOT's 1.292) → broken rankings.
- **Per-ticker zone GSI** (each PJM name uses its home-zone LMP) was built and tested
  (`docs/compact_2026-06-28.md`) — **hypothesis rejected**: IC@21d went 0.146 → 0.077.
  Root cause: PJM names are **rate-regulated T&D**, revenue set by rate cases, not zone
  LMP. There is no pass-through mechanism, unlike ERCOT's VST/NRG merchant exposure.
- **Fix that shipped:** business-model-aware routing (`--arch hard-switch`) — `pass_through
  < 0.5` names get an ICR / DC-queue signal instead of stress beta. Recovers headline
  Sharpe to 0.331 but concedes the grid-stress signal doesn't work on ~⅓ of the universe.

---

## 5. The VST / NRG merchant-contamination problem

The single biggest source of IC instability (`output/findings.md` §3–§11, memory
`session-2026-06-analysis.md`):

- VST / NRG are **competitive/merchant generators**, not regulated utilities — equity
  driven by power-price macro and the AI-data-center narrative, not grid topology.
- The **same physical fact** (large dispatchable fleet in capacity-constrained ERCOT/PJM)
  went from *liability* (reliability penalties, fuel-cost spikes) to *asset* (pricing
  power for firm bilateral hyperscaler contracts; Three Mile Island restart) with no
  change in the physical exposure — only in how the market valued it.
- The rolling-OLS stress beta is a **behavioural** estimator: it measures *how* the stock
  responds to stress, not *why*. When the "why" flips, the beta follows **~1 quarter
  late** → four IC hits from two narrative flip-flops in 18 months (late-2023, May-2024,
  Jan-2025 DeepSeek, May-2025).
- **Documented candidate fixes (none adopted):**
  1. XLU/XNG correlation filter — exclude a name when `corr(stock, XNG) > corr(stock,
     XLU)` for 2 consecutive months (most buildable, price-data only).
  2. Orthogonalise beta to ERCOT/PJM power-price returns before the OLS (most elegant —
     no merchant/regulated classification needed).
  3. Power-price momentum / forward-curve slope regime flag for merchant names.
  4. Stability-weighted or structural-break-adaptive beta window.
  5. Fundamentally-anchored beta (revenue % spot-exposed, fuel mix, region) updated
     annually.
- ICR would **not** have helped VST — it would have *reinforced* the (wrong-for-2024)
  short.

---

## 6. Grid-search findings (`CLAUDE.md`, `output/grid_search_results.csv`)

Parallelised search over **243 parameter combinations** (2026-06-17):

- **Winning config:** `STRESS_SPIKE_PCT=0.97`, `STRESS_CONG_THRESHOLD=0.30`,
  `STRESS_CONG_MIN_DAYS=10`, long 3 / short 5, **no XLU hedge**.
  Result: **Sharpe 0.319, Ann 8.1%, Max DD −13.7%** vs XLU Sharpe 0.235 / DD −36.1%.
- **XLU hedge destroys alpha:** *every* `xlu_hedge=True` config had **negative Sharpe**
  (ranks 157–243 of 243). The long book's top names don't beat XLU by enough for a
  sector-relative trade — the alpha such as it is comes from the **short book picking
  individual losers**, not long-only selection.
- **Congestion threshold:** `0.30` (lower than the original 0.50 design) + the `10`-day
  minimum wins — the duration filter carries the noise-reduction burden.
- **Concentration:** 3 long names beat 5 or 7 at the same stress config.

---

## 7. Congestion sub-signal — Phase 2 (designed, not yet validated)

`docs/superpowers/specs/2026-05-28-congestion-spread-design.md` (status: Approved):

- The 25%-weight `congestion_frac` component arrives **NaN for all ISOs** because
  gridstatus doesn't reliably expose the LMP congestion component — `build_gsi()`
  silently falls back to `lmp_z`, giving price an effective **65%** weight.
- **Fix (built for ERCOT + PJM):** inter-zonal price spread — `spread = max_zone_lmp −
  min_zone_lmp`, normalised by `|mean_lmp|`. ERCOT gets a zone-level fetch
  (`ercot_lmp_zone.parquet`, LZ_NORTH/SOUTH/WEST/HOUSTON); PJM reuses the zone LMPs it
  already fetches. MISO/CAISO/SPP unchanged (still fall back to `lmp_z`).
- **Deferred:** promoting inter-zonal spread to a separate **5th GSI sub-signal** with
  rebalanced weights, and LMP component-based decomposition (energy/congestion/loss).
  Blocked on the spread signal being validated in backtesting first (`CLAUDE.md` "Future
  Work").

---

## 8. Source-document index

| Document | What it holds |
|---|---|
| `output/findings.md` | **Primary evidence writeup** (June 2026) — IC stats, stressed/calm test, 2024 breakdown, VST/NRG mechanical story, "works in the right universe," fix menu (§§9–11) |
| `docs/compact_2026-06-28.md` | Per-config IC@21d + t-stats; per-ticker zone-GSI built and **rejected** (IC 0.146→0.077) |
| `docs/compact_2026-07-05.md` | Business-model-aware architecture (hard-switch Sharpe 0.331, IC 0.143 t 1.87); revenue-mix / dual-track comparison |
| `docs/compact_2026-08-26-original-thesis-reexamination.md` | **Closeout audit** — every signal/construction/universe/validation angle; "no config beats XLU; signal real but weak; sample-size problem" |
| `docs/pjm-signal-handoff.md` | PJM integration: Sharpe 0.319 → −1.68 → −0.04; one-GSI-for-all-PJM root cause; three-part proposed fix |
| `CLAUDE.md` "Grid Search Findings (2026-06-17)" + `output/grid_search_results.csv` | 243-combo search, winning params, "XLU hedge destroys alpha," congestion-threshold insight |
| `docs/superpowers/specs/2026-05-28-congestion-spread-design.md` (+ plan `2026-05-28-congestion-spread.md`) | Inter-zonal congestion-spread sub-signal design (Phase 2) |
| `docs/superpowers/plans/2026-06-19-pjm-data-quality-fix.md` | PJM rate-limit / cache-gap data-quality fix |
| `docs/superpowers/specs/2026-06-29-business-model-aware-factor-design.md` | Hard-switch / revenue-mix / dual-track design |
| `docs/superpowers/specs/2026-08-20-outage-availability-signal-design.md` + `docs/handoff_2026-08-15-outage-reserve-margin-signal.md` | Outage/reserve-margin availability signal — designed, spiked, **killed** (EIA-860 status is annual-scale) |
| `docs/handoff_2026-08-15-rt-da-spread-signal.md` | RT/DA LMP spread — ERCOT practical, PJM blocked; never built |
| `docs/ftr-bid-signal-results.md` | FTR bid-implied forward congestion — FAILED (tracks the vol-target baseline) |
| Memory: `session-2026-06-analysis.md`, `codebase-overview.md` | IC findings, VST/NRG problem, business-model arch, GSI weights, frozen params |
| repo-root `README.md` | Full 7-step mechanism, GSI weights, caching, CLI flags |

---

## 9. Verdict

**KEEP the mechanism, do not claim it as an edge.**

The grid-stress (PJM / ERCOT ISO) signal is a physically-motivated, non-repackaged
cross-sectional factor with **IC ≈ +0.14 (t ≈ 1.9)** on a clean universe — but that is
below significance, does not beat XLU on corrected data, breaks on rate-regulated T&D
names, and whipsaws on the two merchant generators that carry most of its dispersion. The
2018–2025 sample has too few genuine stress regimes to power a monthly cross-sectional
factor.

It survives in `grid_resilience/` only as a **risk-managed L/S wrapper** (grid-search
params frozen, PJM names routed to a different signal via `--arch hard-switch`). Any
future attempt should target the two documented weak points — **merchant-name regime
contamination** (§5 fix menu, orthogonalisation preferred) and the **NaN congestion leg**
(§7) — before re-testing IC, and should pre-register a |t| ≥ 2 / beats-XLU bar.
