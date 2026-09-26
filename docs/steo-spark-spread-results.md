# STEO forward-power spark-spread as a merchant stress-beta stabilizer (results, 2026-09-19)

**Status: PARTIAL as a ranking input (§1-6); FAILED as a sizing input too (§9,
2026-09-20 — both tested variants). NOT ADOPTED as-is; flagged as the most
promising lead in the series, but every operationalization tried so far
(ranking-swap, symmetric vol-sizing, de-lever-only vol-sizing) fails the
project's Sharpe-does-not-worsen gate.**
Untried-tasks-log item 6. Probes: `research/steo_spark_spread/probe_steo_stabilizer.py`,
`probe_steo_sizing.py`.

---

## TL;DR

- Built a **point-in-time forward power-price spark-spread** for PJM and ERCOT from EIA's
  Short-Term Energy Outlook (STEO), and tested it as the stress-beta regressor for
  VST/NRG/CEG/TLN in place of the trailing-realized-LMP proxy fix #2 already tried and
  failed.
- **Feasibility beat the original estimate.** The original scoping (`docs/steo-archive-
  feasibility-results.md`) assumed a ~half-day PDF-table-extraction job across ~90 monthly
  releases, since EIA's own archive is PDF-only. This session found a better path: EIA also
  publishes a machine-readable XLSX companion workbook at a fixed, overwritten-every-month
  URL, and the Wayback Machine has opportunistically crawled that exact URL 63 times since
  December 2016 — each crawl is a genuine point-in-time vintage in structured form, no PDF
  parsing needed at all.
- **Got 63 point-in-time vintages** (2016-12 → 2026-08, opportunistic coverage with gaps —
  not every month), covering `ELWHU_PJ`, `ELWHU_TX`, and Henry Hub gas (`NGHHUUS`) forecast
  values, forecast horizon ~12-18 months per vintage.
- Built a simple, deliberately unelaborate spark spread (constant 7.5 MMBtu/MWh heat rate —
  a level assumption that doesn't affect the vintage-to-vintage *change* the signal actually
  uses) and a z-scored trend per ISO, used as a drop-in replacement regressor in the same
  stress-beta orthogonalization harness fixes #1/#2/#4 used.
- **Result: IC@21d t-stat improves from 1.352 → 2.427** (crosses |t|≥2.0 for the first time
  anywhere in this 5-fix series) and **IC@63d improves from 1.889 → 1.965** (just short of
  the bar). **Max drawdown improves sharply, −31.4% → −15.9%.** But **full-sample and
  stress-period Sharpe both get slightly worse** (−0.007→−0.085 full; −0.145→−0.211
  stress), so the strict "(a) IC improves AND (b) Sharpe does not get worse" gate used for
  every other fix in this series is not fully cleared.
- **Disposition: NOT ADOPTED as a live change, but the first fix in the whole series with a
  genuine, gate-clearing IC improvement** — worth a closer look at *why* the improved
  ranking information isn't translating into better realized L/S returns before writing
  the mechanism off.

---

## 1. Data: how the STEO archive was actually obtained

`docs/steo-archive-feasibility-results.md` and `docs/untried-tasks-log.md` item 6 had
already ruled out two shortcuts: PUDL's processed database has zero STEO/ELWHU tables, and
a Zenodo record that looked like a pre-parsed archive turned out to be a raw-file mirror of
the same PDFs. The standing estimate was **~half a day of real PDF-table extraction** from
each of ~90 monthly STEO releases.

**This session found a materially easier path, not previously checked:** EIA's *current*
STEO release also publishes a machine-readable companion workbook at

```
https://www.eia.gov/outlooks/steo/xls/STEO_m.xlsx
```

— a fixed URL that gets overwritten every month. The Internet Archive's Wayback Machine
has opportunistically crawled that exact URL many times since 2016. Querying the CDX API
for every 200-status snapshot and downloading each capture's raw bytes (bypassing
Wayback's HTML wrapper) yields **63 genuine point-in-time XLSX vintages**, spanning
2016-12-10 through 2026-08-10, each one a structured spreadsheet — no PDF-table extraction
required at all. Coverage is opportunistic (gaps of 2-6 months are common; this is a
crawler's incidental capture, not a designed archive), consistent with the task's own
instruction to prioritize a working panel over exhaustive completeness.

Extracted series per vintage (`extract_steo_series.py`, Table 7a of the workbook):
`ELWHU_PJ` (PJM-West wholesale power price forecast), `ELWHU_TX` (ERCOT wholesale power
price forecast), `NGHHUUS` (Henry Hub natural gas price forecast) — 11,592 rows in the
long-format panel (`data/steo_panel.parquet`), forecast periods spanning 2012-01 through
2027-12 across all vintages combined.

## 2. Spark-spread construction

Industry-standard gross-margin proxy for a gas-fired generator:

```
spark_spread ($/MWh) = power_price − HEAT_RATE (MMBtu/MWh) × gas_price ($/MMBtu)
```

`HEAT_RATE = 7.5` is an assumed constant (a round mid-efficiency CCGT-fleet number, not
calibrated per-ISO or per-year) — deliberately simple, per the "pre-register nothing
elaborate" instruction. This is a *level* assumption only: since the regressor that
matters is how the forward view *changes* vintage to vintage, a constant heat-rate offset
does not change the sign or shape of that change, only its absolute level.

For each vintage, the "forward view" scalar is the average of `ELWHU_{PJ,TX}` and
`NGHHUUS` over only the periods that vintage's own release flagged as forecast (not
historical) — i.e., each release's own ~12-18-month-ahead outlook, not a single noisy
month. Vintages are compared strictly using each one's own Wayback capture date as the
"known as of" anchor — never a later, revised vintage backdated onto an earlier date. The
resulting per-ISO series is z-scored into a daily trend (`data/steo_trend_{ERCOT,PJM}.
parquet`) in the same style fix #2's trailing-LMP trend used, so it's a like-for-like,
drop-in replacement regressor in the stress-beta orthogonalization harness.

## 3. Results

Same harness, same universe, same window as fixes #1/#2/#4 (`docs/merchant-beta-
orthogonalization-results.md`) — this baseline reproduces those fixes' exact baseline
numbers (full Sharpe −0.007, IC@21d t=1.352), confirming the comparison is apples-to-apples
(unlike the parallel fix #5 fundamentals-beta run, which used a different, shorter default
window and is flagged as non-comparable in its own results doc).

| Metric | Baseline (GSI only) | STEO spark-spread regressor |
|---|---|---|
| Full Sharpe | −0.007 | −0.085 |
| Full ann. return | 3.9% | 3.23% |
| Full max DD | **−31.4%** | **−15.9%** |
| Stress-period Sharpe | −0.145 | −0.211 |
| Calm-period Sharpe | 0.379 | 0.17 |
| IC@21d (mean / t-stat) | 0.0628 / 1.352 | 0.1006 / **2.427** |
| IC@63d (mean / t-stat) | — / 1.889 | 0.0792 / 1.965 |

Conditional IC by GSI regime (new cut, not reported for the other fixes): IC@21d is
stronger in stressed periods (t=2.103, n=79) than calm ones (t=1.320, n=10) — the
improvement is not just a calm-period artifact, though the calm sub-sample is thin.

(Full metric dump: `research/steo_spark_spread/probe_steo_stabilizer.py` run output;
underlying data in `research/steo_spark_spread/data/`.)

## 4. Gate evaluation

Same two-part gate as fix #2/#5, for direct comparability:

**(a) IC@21d t-stat improves vs. baseline — PASSES, decisively, for the first time in this
5-fix series.** 1.352 → 2.427, crossing the |t|≥2.0 bar this whole line of work has been
measuring itself against. IC@63d also improves (1.889 → 1.965) though it doesn't itself
cross 2.0.

**(b) Full-sample Sharpe does not get worse — FAILS**, by a small margin. −0.007 → −0.085
is a decline, even though both are solidly negative and economically close to each other.
Stress-period Sharpe likewise declines slightly (−0.145 → −0.211).

**Net: gate not fully cleared.** Under the strict letter of the rule used for every other
fix in this series, this is a FAIL on condition (b) despite a clean PASS on (a) — reported
plainly as a split result, not rounded up to a pass or down to a clean fail.

## 5. Why the IC improves but realized Sharpe doesn't

The pattern is unusual enough to flag rather than wave away: **the new regressor makes the
*rank* of predicted vs. realized returns meaningfully more accurate (higher IC) and cuts
drawdown nearly in half, but does not improve annualized Sharpe.** A likely mechanism:
the STEO-based beta materially reduces variance (visible in the much shallower max DD),
and the resulting lower-vol book's *ann. return* also falls slightly (3.9%→3.23%) — Sharpe
being a ratio, a proportionally larger vol reduction would have been needed to show up as
a Sharpe improvement even though the underlying ranking signal genuinely got better. This
is different from every prior fix in the series (#1/#2/#4), none of which showed any real
IC movement at all — this is the first candidate where the *information content* measurably
improved, just not (yet) the realized-return metric this project uses as its primary gate.

## 6. Disposition

- **NOT ADOPTED as a live change** — the primary gate (Sharpe does not worsen) is not met,
  and this project's discipline treats a split result as a hold, not a promotion.
- **Flagged as the most promising lead in the whole 5-fix series** — the only one with a
  genuine, gate-clearing IC improvement and a large drawdown reduction. Worth a follow-up
  that asks a narrower, different question: can the improved *ranking* information be
  captured in a construction that doesn't require the annualized-Sharpe metric to also
  improve in the same run — e.g., using the STEO-orthogonalized beta only for portfolio
  *sizing*/vol-targeting (where the drawdown improvement is the whole point) rather than
  for the *ranking* the current harness measures Sharpe against. Not built or tested here;
  logged as the natural next step rather than spun up unprompted.
- **Closes the "half-day PDF extraction" assumption as outdated** — the Wayback-Machine
  XLSX-vintage approach (`fetch_steo_vintages.py`) is faster and more reliable than PDF
  parsing and is reusable for any future STEO-series need in this project.

## 7. Follow-up (2026-09-19): can the 2023-onward Sharpe be improved? No.

Tried to salvage the full-sample IC gain on the window that actually matters most for a
live decision (2023-onward). Two variants tested against a freshly-run 2023-01-01→
2025-12-31 baseline (same universe, same equal-weight/dollar-neutral construction):

| Metric | Baseline (2023-25) | STEO both-ISO | STEO ERCOT-only |
|---|---|---|---|
| Full Sharpe | **0.120** | 0.054 | −0.049 |
| Full ann. return | 6.4% | 5.1% | 3.0% |
| IC@21d t-stat | 0.34 | 0.41 | 0.37 |
| IC@63d t-stat | 1.21 | 1.65 | 1.61 |

**Both variants make 2023-onward Sharpe worse than baseline, not better** — restricting
the regressor to ERCOT only (the ISO VST/NRG actually sit in) makes it worse still, not
better, ruling out "the PJM leg is diluting a real ERCOT signal" as an easy fix. IC still
improves slightly on this window (both horizons, both variants) but nowhere near the
full-sample's t=2.427, confirming the full-sample IC gain is concentrated in the
2018-2022 portion of the sample — the same full-window-vs-recent-window fragility
pattern this project has hit on nearly every other signal (Henry Hub, narrative Approach
C, category-demand rotation). **Do not pursue a beta-regressor swap further on this
window; the sizing/vol-target application floated in §6 remains the only untried angle
that doesn't depend on the pre-2023 period repeating.**

## 8. Follow-up (2026-09-20): does smoothing the vintage step-function help? No — it hurts.

Hypothesis: the STEO regressor is a step function (constant between the 63 irregularly-
spaced vintage releases, jumping discretely at each new release) — maybe the discrete
jumps inject noise that a causal EMA smoothing pass (applied to the daily forward-filled
level, before computing the 252-day trend — still point-in-time safe, no future data used)
would clean up.

| Metric | Unsmoothed (full-sample) | EMA halflife=21d | EMA halflife=63d |
|---|---|---|---|
| IC@21d t-stat | **2.427** | 0.889 | 1.059 |
| IC@63d t-stat | 1.965 | 1.937 | 2.123 |
| Full Sharpe | −0.085 | −0.072 | −0.068 |
| Max DD | **−15.9%** | −30.2% | −30.2% |

| Metric | Baseline (2023-25) | Unsmoothed STEO | EMA hl=21 (2023-25) | EMA hl=63 (2023-25) |
|---|---|---|---|---|
| Full Sharpe | **0.120** | 0.054 | 0.030 | 0.085 |
| IC@21d t-stat | 0.34 | 0.41 | 0.32 | 0.35 |

**Result: smoothing makes it worse, not better.** It destroys the one standout full-sample
number (IC@21d t 2.427→0.89 at hl=21) and gives back nearly all of the drawdown benefit
(−15.9%→−30.2%, back near baseline's −31.4%). IC@63d ticks up marginally at hl=63
(1.965→2.123) but every other metric degrades alongside it, and 2023-onward Sharpe still
doesn't beat baseline at any halflife tested.

**Read:** the discrete jump at each STEO release is more likely the actual information
event than noise around one — EIA's own outlook updates in discrete steps, and the
market's re-rating plausibly happens at the same discrete points. Diffusing that jump
over 21-63 days smears real signal across time rather than removing noise from around it.
Closes this follow-up thread; do not pursue further smoothing variants without a
different rationale.

## 9. Follow-up (2026-09-20): sizing instead of ranking — still FAILED

§6 floated a narrower question: since the STEO-variant book is calmer (MaxDD −15.9%
vs. baseline −31.4%) without beating baseline Sharpe, could that calmness be harvested
as a **vol-target/sizing** input — leaving the long/short **selection** exactly as the
baseline (GSI-only) ranking, and only scaling how big the baseline book's bet is,
day to day? This tests that, for the first time.

**Construction** (`probe_steo_sizing.py`): reran the unmodified baseline pipeline,
capturing its own `weights_matrix`/`returns`/`events_df` (the literal inputs
`run_backtest` uses) via a wrapper — no change to GSI, beta, or ranking. Took the
already-cached STEO-variant full-sample return series purely as a volatility gauge:
daily exposure scalar = (STEO-variant's own full-sample annualized vol) / (its
trailing 21d annualized vol), shifted 1 day to avoid lookahead. Two variants of the
scalar were tested:

- **Symmetric** — clipped to [0.5×, 1.5×] (lever up when STEO-implied vol is below its
  own average, delever when above).
- **De-lever-only** — clipped to [0.5×, 1.0×] (never lever up; the purely defensive
  reading of "the drawdown improvement is the whole point").

Same pre-registered two-part gate as the rest of this series: Sharpe improves AND
MaxDD does not worsen.

| Metric | Baseline | Symmetric [0.5, 1.5] | De-lever-only [0.5, 1.0] |
|---|--:|--:|--:|
| Full Sharpe | **−0.007** | −0.018 | −0.088 |
| Full ann. return | 3.90% | 3.72% | 2.87% |
| Full max DD | −31.41% | −30.38% | **−27.80%** |
| Stress-period Sharpe | −0.145 | −0.221 | −0.259 |
| Calm-period Sharpe | 0.379 | 0.434 | 0.340 |

**Result: both FAIL gate (a).** Sharpe gets worse in both variants, monotonically
worse the more defensive the construction (symmetric −0.018, de-lever-only −0.088).
Gate (b) (MaxDD) technically passes both times — MaxDD improves modestly (symmetric)
to meaningfully (de-lever-only, −31.4%→−27.8%) — but under this project's AND-gate,
a Sharpe decline fails the pair regardless.

**Why, mechanically:**
- The **symmetric** scalar spent 48% of days at its 1.5× leverage-up ceiling and only
  5.9% at the 0.5× floor (mean 1.235) — STEO's return series is right-skewed (calm most
  of the time, occasional sharp vol spikes around stress events), so its full-sample
  average vol sits *above* its typical trailing vol. A self-referential target built
  this way spends most days levering up on the ordinary/calm side, not delevering into
  the rare bad days — it's closer to "buy more when STEO looks calm" than "cut
  exposure when STEO looks risky," which is the opposite of the intended defensive use.
- The **de-lever-only** variant fixes that skew problem directly (mean scalar 0.930,
  74% of days at the neutral 1.0) and does cut MaxDD more (−27.8% vs. baseline
  −31.4%), but return falls proportionally more than vol does (ann. return 3.9%→2.9%,
  a −26% relative cut) — it's deleveraging into periods that, ex post, weren't bad
  enough to justify the return given up, the classic vol-targeting failure mode when
  the underlying trailing-vol signal isn't crisply timed to genuine risk-off regimes.

**Disposition:** closes the sizing angle floated in §6 — not just the ranking-swap,
but the two most natural sizing operationalizations (symmetric and de-lever-only vol
targeting) also fail the Sharpe leg of the gate. Three independent uses of the STEO
signal (ranking-swap §1-6, symmetric sizing, de-lever-only sizing) now all show the
same pattern: real, measurable drawdown/IC improvement that does not convert into a
Sharpe improvement under this project's gate. **NOT ADOPTED.** No further
sizing variant recommended without a different underlying rationale (e.g. gating on
the STEO trend's *level* rather than the orthogonalized book's realized vol — untried,
low prior given this pattern).

## 10. Follow-up (2026-09-20): STEO regime multiplier on the actual SHIPPED book — FAILED, more decisively

§1-9 all tested STEO against the `grid_resilience` package's legacy merchant stress-beta
factor book (Sharpe −0.007 baseline, 2018-2025 full sample) — a research pipeline that
was never the shipped strategy. This follow-up redirects the same "use STEO for sizing,
not ranking" idea onto the **actual shipped `electrification_strategy` book**
(`screen +val+sleeve`, the 27-name equipment-supplier basket + valuation overlay + 15%
GLD/IEF sleeve, full 2017-06→present sample, documented Sharpe 0.69).

**Construction** (`probe_steo_electrification_sizing.py`): reused
`grid_equipment_basket.grid_regime.regime_multiplier` **unchanged** — the same function
and same default discrete thresholds (`REGIME_THRESH=0.5`, `HI=1.25`, `LO=0.6`,
20-day month-lookback) already calibrated for the grid-congestion Layer-2 overlay that
failed its own gate on this same book (`docs/grid-regime-layer2-results.md`) — no new
parameter search, just a new input series (mean of the PJM/ERCOT STEO spark-spread
z-trend). The regime multiplier was layered into the book's existing vol-target
exposure calc (`exposure = steo_mult × valuation_mult × vol_scalar`, capped at the
existing 1.5× ceiling), touching only the core equity leg — the 15% GLD/IEF sleeve and
the 27-name selection were both left untouched.

**Mechanism caveat, stated before running:** STEO's spark spread was built to predict
PJM/ERCOT *merchant generator* profitability (VST/NRG/CEG/TLN) — this book is equipment
*suppliers* (transformers/switchgear/etc.), a more indirect link, and the two prior
forward-power-market regime overlays tried on this exact book (grid-congestion, FTR-bid)
had both already failed their gate. Prior was low going in.

| metric | baseline (shipped) | STEO-regime-sized |
|---|--:|--:|
| Sharpe | **0.687** | 0.528 |
| CAGR | 14.34% | 11.54% |
| Max DD | −23.99% | −23.99% (identical) |
| Ann. vol | 15.43% | 15.33% |

**Result: FAILS gate (a) decisively** (Sharpe −0.16, a much larger decline than either
legacy-book sizing variant in §9) **and gate (b) is a wash, not a win** — MaxDD is
*exactly* unchanged, because the STEO composite only starts 2021-01-19 (Wayback
coverage), so the multiplier is neutral for everything before that date — and the
book's own worst drawdown episode sits in that pre-2021 window. The overlay gave up
real return in-window with zero offsetting drawdown protection where it actually
mattered. Confirms the mechanism worry going in: a signal built around merchant
generator profitability does not transfer usefully to an equipment-supplier book. The
regime multiplier itself was active, not degenerate (mean 0.963; 28.6% of days at the
1.25× ceiling, 27.0% at the 0.6× floor, 44.4% neutral) — this is a real test of the
signal, not a no-op.

**Disposition: NOT ADOPTED. Closes the STEO thread against the shipped book as well as
the legacy one.** Four independent operationalizations now tried across both books
(legacy-book ranking-swap, legacy-book symmetric sizing, legacy-book de-lever-only
sizing, shipped-book regime multiplier) — none clears its gate; the shipped-book test
is the clearest and most decisive failure of the four.
