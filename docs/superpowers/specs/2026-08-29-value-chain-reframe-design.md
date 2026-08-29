# Design Spec — Grid Equipment Basket: Value-Chain Reframe + Intra-Theme Hedge

**Status:** Approved design, not yet built.
**Location:** extends the existing `grid_equipment_basket/` module (sibling to `grid_resilience/`, `dc_demand_basket/`).
**Builds on:** `docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md` (Step 1 basket + Step 2 backlog
tilt, both shipped on branch `grid-equipment-basket`). This spec does **not** change the Step 1 equal-weight
basket or the Step 2 tilt path; it adds a third construction option and a hedge.
**Origin:** `docs/handoff_2026-08-29-grid-equipment-basket.md` — Open need 1 (a differentiated edge beyond
"own the theme") and Open need 2 (a working hedge). This is spec **#1 of 3** planned follow-ups; the other two
(crowding-aware weighting; sister-project demand-signal timing overlay) get their own specs later.

---

## 1. Thesis — the reframe

The Step 1 basket cleared its decision gate (CAGR 59.8% vs XLI 20.2%, Sharpe 1.36 vs 0.95) but has no edge
beyond thematic beta: the theme is mainstream, the marquee names are partly crowded, the only active decision
was the 2026-vintage universe, and the Step 2 backlog-growth tilt failed (it over-weighted FLNC and lost to
equal-weight).

**A more defensible explanation of where the return comes from:** in a capex boom the excess return accrues to
whoever occupies the supply-constrained bottleneck **and can price it** — not to whoever posts the fastest
revenue growth. Large power transformers, HVDC converters, medium-voltage switchgear, and gas turbines have
lead times that stretched from roughly one year to two-to-four years. That is a **margin-expansion** story for
the firms that manufacture that apparatus. The firms that engineer and construct the installations bid
competitively and pass through labor; the firms that assemble systems from third-party components compete on
price. Those price-takers get **volume, not margin**.

**The active bet:** overweight the equipment makers, underweight (long-only) or short (market-neutral) the
price-takers, with a fundamental signal — margin trajectory plus order-book coverage trajectory — setting the
amounts within each group.

**Why this is not the failed Step 2 tilt relabeled:** Step 2 ranked names by raw YoY backlog *growth*, which
put FLNC (fast growth off a tiny base, negative margins) at the top. This spec ranks on **margin change** and
**coverage change** (backlog relative to revenue), inside **frozen business-model buckets**, so a fast-growing
price-taker cannot migrate into the overweight leg.

**Why the edge and the hedge are one mechanism:** if the makers extract scarcity rents and the price-takers do
not, then long-makers / short-price-takers is simultaneously the differentiated active bet and a structural
hedge — the one scenario that de-rates the long book (hyperscaler AI-capex slowdown de-rating the whole
picks-and-shovels chain at once) hits both legs, so the spread is far less exposed to it than the naked long.

---

## 2. What gets built — two products, reported side by side

1. **Value-chain-tilted long-only basket** (`--construction value-chain-tilt`). The existing 9-name basket,
   equal-weight start, tilted toward makers and away from contractors with the §4 signal setting within-bucket
   amounts. Reported head-to-head against the Step 1 equal-weight basket and the XLI / GRID / PAVE benchmarks.

2. **Market-neutral pair** (`--construction pair`). Long the makers bucket, short the contractors bucket,
   dollar-neutral. Reported standalone, as a 30% overlay on the equal-weight long basket, and against the §7.2
   comparator.

Neither product replaces the Step 1 equal-weight basket unless it clears its §7 gate.

---

## 3. The buckets — frozen before any backtest

Assigned once, from each name's most recent 10-K business description only (the same evidentiary standard as
universe inclusion in the base spec §2). **Frozen in `config.py` before the first backtest run and not revised
based on results.**

Split rule: **"manufactures electrical apparatus and sets its own price"** (maker) vs **"bids competitively for
engineering/construction work or assembles systems from third-party components"** (price-taker).

| Makers (pricing power) | Contractors / assemblers (price-takers) |
|---|---|
| **ETN** Eaton — transformers, switchgear, DC power (73% electrical) | **PWR** Quanta Services — electric-power infrastructure EPC |
| **HUBB** Hubbell — T&D components, transformers, insulators, arresters | **MYRG** MYR Group — specialty electrical construction |
| **GEV** GE Vernova — HVDC, power transformers, switchgear, turbines | **PRIM** Primoris Services — utility infrastructure construction |
| **VRT** Vertiv — data-center power & thermal gear | **FLNC** Fluence Energy — battery-system integrator |
| **NVT** nVent — connectors, enclosures, liquid cooling, switchgear | |

**FLNC is the one borderline call and is flagged as such:** it sells a manufactured product but assembles
third-party cells, bids competitively against Tesla / LG / Samsung, and has run chronic negative gross margin —
no pricing power. The thesis places it with the price-takers. (This mirrors how the base spec flagged PRIM as
the borderline inclusion call.)

**Hindsight disclosure:** this 5/4 split lands close to a hand-split the handoff already flagged as *probably
hindsight* (its `{ETN, GEV, HUBB, NVT, VRT}` did Sharpe 1.55 standalone vs `{PWR, MYRG, PRIM}` at 1.20). The
overlap is real and stated plainly in the results writeup. Mitigations: the split rule is business-model-based
and frozen pre-backtest; §7's prior-regime panel and drop-VRT/GEV robustness pass both test whether the effect
is merely "we picked the winners"; and the short leg is *conservative*, not hindsight-flattered — Quanta, a
large winner over the window, sits in the short leg, so shorting it **cost** money.

`config.py`:
```
BUCKET_MAKERS      = ["ETN", "HUBB", "GEV", "VRT", "NVT"]
BUCKET_CONTRACTORS = ["PWR", "MYRG", "PRIM", "FLNC"]
```

---

## 4. The signal

Two components, combined by rank-averaging. **No z-scoring, no `build_factor()`** — base spec §12 forbids that
machinery in this module; the signal is plain arithmetic on documented constants.

### 4.1 Pricing-power read — TTM gross-margin change

For each name, from SEC XBRL `companyconcept`:

- Pull `us-gaap:GrossProfit` and `us-gaap:Revenues` (or `RevenueFromContractWithCustomerExcludingAssessedTax`).
- **Fallback** where `GrossProfit` is not tagged (common for EPC filers): derive it as
  `Revenues - CostOfRevenue` (or `- CostOfGoodsAndServicesSold`). All still XBRL, all filing-dated.
- Quarterly gross margin = `GrossProfit / Revenues`. Signal = **YoY change in the trailing-four-quarter
  average gross margin** (TTM margin now minus TTM margin four quarters prior), so one-off quarters do not
  dominate.
- Availability date = the filing date of the 10-Q / 10-K the figure comes from (never the period-end date).

Company-wide operating margin and grid-segment operating margin are **logged as next-step refinements** (§9),
not built here.

### 4.2 Order-book read — backlog-coverage change

- Coverage = (disclosed backlog / RPO from `data/backlog_quarterly.csv`, already collected in Step 2) divided
  by that name's TTM revenue (from the §4.1 XBRL revenue series).
- Signal = **YoY change in coverage**.
- Same point-in-time rule (availability date from `backlog_quarterly.csv`, which already carries it).
- Names whose backlog disclosure is annual-only (HUBB, NVT — noted in the Step 2 results) update coverage once
  a year; the most recent annual figure is carried forward between updates. Flagged in the writeup.

### 4.3 Combine

Within each bucket, on each rebalance date: rank the names on the §4.1 signal and on the §4.2 signal
(ascending, higher = stronger), then **average the two ranks** into a composite. A name missing one component
is ranked on the other alone; a name missing both takes no within-bucket adjustment (§5).

---

## 5. Construction 1 — value-chain-tilted long-only basket

Start from the Step 1 equal weights on each rebalance date (same universe, same ~6-week-lagged quarterly
calendar, same 25% cap). Apply one multiplier per name:

| Step | Multiplier |
|---|---|
| Base bucket | maker **×1.25**, contractor **×0.75** |
| Within-bucket, by §4.3 composite | top half **×1.10**, bottom half **×0.90** |

So a top-ranked maker ≈ ×1.375, a bottom-ranked contractor ≈ ×0.675. Then **renormalize to sum to 1** and
**re-apply the 25% single-name cap**.

- All four constants live in `config.py` (`VC_BASE_MAKER`, `VC_BASE_CONTRACTOR`, `VC_WITHIN_TOP`,
  `VC_WITHIN_BOTTOM`). **Not optimized.** The results writeup reports the surrounding plateau — e.g. base split
  1.20/0.80 through 1.30/0.70 giving materially similar metrics — the same way the handoff overlay work
  reported its MA / vol-target plateau.
- A name with no §4.3 composite (GEV before it has five post-spin quarters; a name with neither signal
  component) takes only the base bucket multiplier.
- Odd bucket count (contractors = 4, makers = 5): the median-ranked maker takes ×1.00 within-bucket.

---

## 6. Construction 2 — market-neutral pair

- **Long leg:** the makers, weighted proportional to §4.3 composite rank (best maker largest).
- **Short leg:** the contractors, weighted proportional to *inverse* composite rank (weakest contractor
  largest short).
- **Dollar-neutral:** 100% gross long / 100% gross short (`PAIR_GROSS = 1.00`). Quarterly rebalance on the
  Step 1 calendar.
- Evaluated three ways:
  1. **standalone** — the pair's own return, vol, Sharpe, max drawdown, and annualized carry;
  2. **30% overlay** on the equal-weight long basket (`PAIR_OVERLAY_WEIGHT = 0.30`): 100% long basket − 30%
     contractors + 30% makers-tilt, i.e. the pair scaled to 0.30 and added;
  3. **risk-matched overlay** — the pair overlay rescaled so that "long basket + pair overlay" and "long
     basket + conditional short" (§7.2) have equal annualized volatility over the primary window, for an
     apples-to-apples drawdown comparison.

Short-side costs are **not modeled** (consistent with the rest of the module). Instead the writeup carries a
named sensitivity: how many percentage points per year of borrow cost on the short leg would erase the pair's
edge, plus an explicit caveat that **MYRG is thin (~$20–80M/day)** and **FLNC is expensive and hard to borrow**
(already heavily shorted, going-concern noise).

---

## 7. Validation

### 7.1 Windows

- **Primary:** 2023-01-01 → 2026-07-31 (unchanged from Step 1; 896 trading days, ~43 monthly obs).
- **Prior-regime panel:** 2020-01-01 → 2022-12-31, context only — **not** a second gate. GEV absent the whole
  panel; FLNC only from its 2021-10 IPO; VRT's early quotes are the predecessor SPAC. Same limitations Step 1
  documented.
- **Robustness pass:** drop **VRT and GEV** from the makers bucket, re-run the primary window. Question:
  does "makers beat contractors" survive without the two largest winners?

### 7.2 The comparator (Construction 2 only)

**Conditional QQQ short.** Short 30% QQQ (`COND_SHORT_TICKER`, `COND_SHORT_WEIGHT = 0.30`) whenever, at the
**prior month-end**, both hold:

- the basket closed **below its 100-day moving average** (`COND_SHORT_MA_DAYS = 100`), and
- the basket's **trailing-20-day realized vol** (`COND_SHORT_VOL_DAYS = 20`) exceeded its **trailing-252-day
  median** realized vol (`COND_SHORT_VOL_REF_DAYS = 252`).

Off otherwise. Overlaid on the same equal-weight long basket, decision re-checked monthly. This is the cheap,
no-new-data hedge the pair must beat.

### 7.3 The drawdown episode both hedges are scored on

From the equal-weight basket's own equity curve over the primary window: take the **peak** as the curve's
highest point in 2024-H2, and the **trough** as its lowest point between that peak date and 2025-06-30 (the
DeepSeek efficiency scare, the −42% event the handoff names). Fix those two dates. For each hedged portfolio
(long + pair overlay; long + conditional short) compute the max drawdown **over that same fixed calendar
span**. That is the protection number both hedges are compared on.

### 7.4 Pre-registered gates

| Gate | Passes iff | On failure |
|---|---|---|
| **1 — long-only tilt** (Construction 1) | Over the primary window, the tilt beats the Step 1 equal-weight basket on **both** Sharpe **and** CAGR (the Step 2 §6 bar) | Equal-weight remains the recommended long book; the tilt is written up as a negative result and left available behind the flag |
| **2 — the pair** (Construction 2) | The pair 30% overlay protects the §7.3 episode drawdown **at least as much as** the conditional-QQQ-short overlay (§7.2), **and** the pair's standalone annualized carry is **no worse than** the conditional short's annualized carry over the same window | Recommendation is stated as "use the conditional QQQ short" or "size down, no hedge" — both listed as acceptable outcomes in the handoff |

Gates are evaluated on the primary window only. Point-estimate comparisons — the ±0.5 Sharpe standard error
(§8) means none of these gaps is statistically significant; the gates are a pre-committed decision rule, not a
significance claim.

### 7.5 Hindsight controls (stated in the writeup)

1. Bucket rule business-model-based and frozen pre-backtest (§3).
2. Prior-regime panel and drop-VRT/GEV pass (§7.1).
3. Short leg is conservative — Quanta (winner) is in it (§3).
4. Gross-margin dilution for ETN / GEV / PRIM acknowledged, with the operating-margin refinement logged for a
   later spec, not silently included now (§9).

---

## 8. Known biases and caveats

**Carried unchanged from Step 1 / Step 2 (base spec §5, §7):**

- ~3.56-year, ~43-monthly-observation window; one macro regime (AI bull with sharp corrections).
- Standard error on an annualized Sharpe from this sample is ~±0.5 (≈ ±1.0 at 95%). Every gate comparison and
  every reported gap is a **directional point estimate, not a significant result**.
- GEV has vendor price history only from 2024-03-27 and covers only the back ~60% of the window.
- **Headline survivorship / hindsight bias:** the universe was chosen in 2026 knowing VRT, GEV, PWR won.

**New to this spec:**

- The 5/4 bucket split overlaps the handoff's hindsight-flagged hand-split (§3; mitigations §7.5).
- Gross margin is diluted for the diversified names (ETN aerospace/vehicle; GEV wind/power/services; PRIM road
  work). Operating-margin and grid-segment-margin variants are §9 next steps, not in this build.
- The short leg is not costless — MYRG thin, FLNC hard/expensive to borrow. Pair results are **gross**, with a
  borrow-cost sensitivity (§6).
- The pair adds fitted-looking degrees of freedom (base bucket split, within-bucket multipliers, three
  conditional-short thresholds). The writeup reports the **parameter plateau**, not a tuned point, and does not
  optimize any of them.

---

## 9. Out of scope / logged next steps

- **Company-wide operating-margin variant** of §4.1 — try after gross margin; log whether it sharpens or
  muddies the signal on the diversified names.
- **Grid-segment operating-margin variant** — hand-collected, same effort tier as the Step 2 backlog data.
  Only worth it if the gross-margin signal looks too noisy on ETN / GEV / PRIM.
- Spec #2 (crowding-aware weighting) and spec #3 (sister-project demand-signal timing overlay) — separate
  specs, later sessions.
- Cross-sectional z-scoring, `build_factor()` / `hard_switch`, IC-style predictive-power metrics, any
  significance claim, an 8-year historical backtest as primary validation — all remain out of scope
  (base spec §12).
- Options-structured hedges (put spreads financed by a call overwrite) — need an options data source; not this
  spec.

---

## 10. Module layout

```
grid_equipment_basket/
  config.py            # + BUCKET_MAKERS / BUCKET_CONTRACTORS (frozen),
                       #   VC_BASE_MAKER / VC_BASE_CONTRACTOR / VC_WITHIN_TOP / VC_WITHIN_BOTTOM,
                       #   VC_SIGNAL_LOOKBACK_Q = 4,
                       #   PAIR_GROSS = 1.00, PAIR_OVERLAY_WEIGHT = 0.30,
                       #   COND_SHORT_TICKER / _WEIGHT / _MA_DAYS / _VOL_DAYS / _VOL_REF_DAYS
  margin_data.py       # NEW — XBRL companyconcept: GrossProfit / Revenues / CostOfRevenue,
                       #   derive-gross fallback, point-in-time TTM gross-margin series.
                       #   Mirrors backlog_data.py structure + monthly-parquet caching.
  value_chain.py       # NEW — bucket lookup; §4.3 composite rank;
                       #   value_chain_tilt_targets()  -> (date x ticker) weight frame (Construction 1);
                       #   pair_weights()               -> long/short weight frame (Construction 2).
                       #   Plain arithmetic on config constants. No z-scoring, no build_factor.
  hedges.py            # NEW — conditional_qqq_short_overlay(); pair_overlay();
                       #   episode_drawdown(peak, trough) and annualized_carry() scoring helpers.
  basket.py            # unchanged (Step 1 + Step 2 tilt paths stay as-is)
  backtest.py          # + report rows: tilt vs equal-weight vs XLI/GRID/PAVE;
                       #   pair standalone / 30% overlay / risk-matched;
                       #   pair vs conditional-short on the two §7.4 gate-2 metrics.
  __main__.py          # + --construction {equal-weight | backlog-tilt | value-chain-tilt | pair}
  README.md            # + the reframe, the frozen buckets, both gate outcomes, §8 caveats
docs/
  superpowers/specs/2026-08-29-value-chain-reframe-design.md   # this spec
  grid-equipment-value-chain-results.md                        # results writeup (both gates, all caveats)
```

Reuse: `margin_data.py` follows `backlog_data.py` (SEC XBRL `companyconcept` fetch, `User-Agent`
`acadia-research sand.gh1902@gmail.com`, hand-collected-fallback discipline, monthly-parquet cache). No changes
to `grid_resilience/` or `dc_demand_basket/`.

---

## 11. Testing

Test-first (TDD). All offline, deterministic fixtures, no network — matches the existing 41-test suite.

- `tests/grid_equipment_basket/test_value_chain.py`
  - bucket assignment is exactly the frozen 5/4 split; an unknown ticker raises
  - TTM gross-margin and its YoY change on a constructed quarterly fixture (incl. the derive-from-cost
    fallback path)
  - backlog-coverage = backlog / TTM-revenue and its YoY change
  - §4.3 composite = mean of two ascending rank vectors; one-component and no-component names handled
  - Construction 1: base ×1.25 / ×0.75 → within-bucket ×1.10 / ×0.90 → renormalize → 25% cap; weights sum to
    1; a no-signal name takes base multiplier only; odd-count median maker takes ×1.00 within-bucket
  - Construction 2: long = makers composite-weighted, short = contractors inverse-weighted; gross long = gross
    short = 1.00; net ≈ 0
- `tests/grid_equipment_basket/test_hedges.py`
  - conditional-short overlay engages **only** when the 100-DMA condition **and** the vol condition both hold
    on the decision date; disengages when either fails
  - `episode_drawdown` over a fixed peak→trough span matches a hand-computed value
  - `annualized_carry` matches a closed-form value on a constructed series
- `tests/grid_equipment_basket/test_margin_data.py`
  - XBRL parse with a mocked `companyconcept` payload; filing-date (not period-end) used as availability date
  - point-in-time: a filing dated after a rebalance is excluded from that rebalance
  - cache hit / miss

Live XBRL fetch validated once manually during implementation; the result and the realized per-name signal
coverage recorded in the README.

---

## 12. Deliverables

- `grid_equipment_basket/margin_data.py`, `value_chain.py`, `hedges.py`
- `config.py`, `backtest.py`, `__main__.py` edits per §10
- Test suite per §11 (value_chain + hedges + margin_data), existing 41 tests still green
- `grid_equipment_basket/README.md` updated: reframe, frozen buckets, Gate 1 and Gate 2 outcomes, §8 caveats
- `docs/grid-equipment-value-chain-results.md`: primary + prior-regime + drop-VRT/GEV numbers for both
  constructions, the two gate evaluations, the borrow-cost sensitivity, the parameter plateau, all §8 caveats
- One-line pointer added to the repo-root `README.md`
- Repo-root `CLAUDE.md` "Future Work" note for the §9 operating-margin variants
