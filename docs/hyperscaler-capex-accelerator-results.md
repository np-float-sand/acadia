# Hyperscaler capex-reacceleration as an entry accelerator — results

**TL;DR: PASS on the literal pre-registered gate (full-sample Sharpe 0.687→0.754, MaxDD
unchanged at -23.99%), but the MaxDD result is a coverage-timing artifact, not genuine risk
reduction — the book's one worst-ever drawdown (Dec 2018) predates the capex data's warm-up
entirely, so the multiplier was mechanically neutral (1.0x) throughout it. Sub-window analysis
shows the signal is a genuine, cost-free win in 2019-2022 (Sharpe 0.505→0.615, MaxDD tied) but
in 2023-2026 — the actual AI-capex-supercycle window this signal was built for, and where it is
pegged at 1.25x almost permanently (86.6% of 2024, 100% of 2025) — Sharpe is slightly *worse*
(1.595→1.528) and MaxDD deepens by ~4.2pp (-17.0%→-21.3%). Net effect is close to "add
structural leverage during an already-strong regime," not "confirm reacceleration with good
timing." Disposition: **caveated PASS, not recommended for adoption without addressing the
2023-26 behavior** (this task's own instructions rule out wiring it in regardless of the
result — it stays a probe). See §5 for the honest breakdown behind the headline number.**

Pre-registration (written and committed before this script ran): `docs/hyperscaler-capex-accelerator-preregistration.md`.
Probe script: `research/hyperscaler_capex_accelerator/probe_capex_accelerator.py`.
Open item #16, `docs/RESEARCH-LOG.md` §4.

## 1. Context

`grid_equipment_basket/capex_signal.py` uses point-in-time Big-4 hyperscaler
(MSFT/GOOGL/AMZN/META) quarterly capex — a `yoy` (trailing-4-quarter growth) series, stamped
with a `known_date` (quarter-end + 50 days) — as a **one-directional exit de-risk trigger**:
cut exposure to 0.5x when `yoy < 0.15` and `decel2 < 0`. That use **FAILED** (RESEARCH-LOG line
137: "fires ~a year late; lagging confirmation"), the same pattern as every other demand-data
timing signal tried in this line — they all lag a 2-4-year hyperscaler-guidance-to-spend cycle
too much to call a turn.

This probe tests the opposite framing named in RESEARCH-LOG open item #16: the same `yoy`
series as a long-only **entry accelerator** — size UP when capex growth is reaccelerating,
because an entry accelerator is confirming a trend already held, not calling a turn, so lag
should be more tolerable.

## 2. Re-acceleration rule (pre-registered, unchanged from §2 of the pre-registration)

`yoy > yoy.shift(1).rolling(4).mean()` — capex growth exceeds its own trailing 4-quarter
average of the *prior* 4 quarters (current quarter excluded from its own baseline). A plain
moving-average crossover on the growth rate, mirroring `overlay.trend_gate`'s existing
price-vs-MA convention. No new window parameter (4q matches `yoy`'s own window); no threshold
search.

Multiplier: `capex_accel_mult = REGIME_HI (1.25)` when reaccelerating, else `1.0`. `1.25` is
`grid_equipment_basket.config.REGIME_HI`, an existing constant reused unchanged (the
grid-congestion Layer-2 overlay's discrete "up" multiplier), not tuned for this test.
One-way up only — the mirror image of `capex_derisk_multiplier`'s one-way down.

## 3. Construction

Exactly the STEO-regime-multiplier template (`docs/steo-spark-spread-results.md` §10), applied
to the actual shipped `screen +val+sleeve` book, core equity leg only (15% GLD/IEF sleeve and
the 27-name screen selection untouched):

```
exposure = clip(capex_accel_mult x valuation_mult x vol_scalar, upper=VOL_MAX_LEVERAGE)
```

Full sample 2017-06-01 (`config.START_DEFAULT`) → 2026-09-21 (today).

## 4. Sanity check

Harness reproduces the documented shipped baseline before trusting the variant: full-sample
Sharpe **0.687** (RESEARCH-LOG §1 documents 0.69; also matches the STEO probe's independently
computed baseline of 0.687 exactly), CAGR 14.34%, MaxDD -23.99%. Harness validated.

## 5. Results

### 5.1 Full sample (2017-06-01 → 2026-09-21)

| metric | baseline (shipped) | capex-accelerator |
|---|--:|--:|
| Sharpe | 0.687 | **0.754** |
| CAGR | 14.34% | 16.62% |
| Max DD | -23.99% | -23.99% (identical) |
| Ann. vol | 15.43% | 17.03% |

**Gate (a) Sharpe improves: PASS** (0.754 > 0.687).
**Gate (b) MaxDD does not worsen: PASS** (tied, not worse).
**Literal verdict: PASS.**

### 5.2 Sub-window breakdown — the honest picture behind the headline

| window | baseline Sharpe | accel Sharpe | baseline CAGR | accel CAGR | baseline MaxDD | accel MaxDD |
|---|--:|--:|--:|--:|--:|--:|
| 2019-22 | 0.505 | **0.615** | 11.81% | 14.49% | -23.17% | -23.17% (identical) |
| 2023-26 | 1.595 | **1.528** (worse) | 31.53% | 34.78% | -17.03% | **-21.26%** (worse, +4.2pp) |
| VOLT-fair 2025-01→2026-08 | 1.354 | 1.324 (worse) | 27.66% | 32.24% | -17.03% | **-21.26%** (worse, +4.2pp) |

**2019-22 is a genuine, cost-free win**: Sharpe improves, MaxDD is exactly unchanged (not just
non-worse by luck of the gate — the actual worst drawdown of that sub-window is unaffected).
The multiplier is at 1.25x 61.7% of 2020 and 63.3% of 2021 (the COVID-recovery/2021 capex
reacceleration), correctly neutral through all of 2022 (the capex-deceleration/rate-hike bear
market) — this is the signal doing exactly what it was designed to do.

**2023-26 — the actual AI-capex-supercycle window this signal targets — is a loss on both
legs.** The multiplier is pegged at 1.25x 86.6% of 2024 and literally 100.0% of 2025 (mean
multiplier by year: 2024 = 1.217, 2025 = 1.250). At that point it isn't confirming a trend so
much as applying near-permanent structural leverage: CAGR rises (capex growth genuinely never
stopped reaccelerating in this window, so a levered book of a rising asset outperforms) but
Sharpe falls and MaxDD deepens ~4.2pp, because a 1.25x book takes a proportionally larger hit
in every pullback inside an otherwise-strong regime, and there were no pullbacks left for the
signal to sit out (it was already at its ceiling).

### 5.3 Why the full-sample MaxDD gate passes despite this

The book's single worst drawdown across the whole 2017-2026 sample is the Dec-2018 selloff
(-23.99%, both baseline and variant — same date, same magnitude, confirmed by direct
inspection of the drawdown series: `dd.idxmin()` = 2018-12-24 in both). The capex-accelerator
multiplier requires four known quarterly `yoy` prints before its first trailing-average
observation exists (~2020-05-20); everything before that is neutral (1.0x) by construction, so
the multiplier was flatly inert through the entire 2018 window. **Gate (b) passing full-sample
is a coincidence of when the worst-ever drawdown happened to fall relative to the capex data's
availability, not evidence the signal reduces risk** — the same class of artifact flagged in
`docs/steo-spark-spread-results.md` §10 for STEO's pre-2021 coverage gap. Once the signal is
actually live and doing work (2023-26), MaxDD gets measurably worse, not neutral.

### 5.4 Multiplier mechanics

Mean 1.097 full-sample; 38.6% of days at the 1.25x ceiling, 61.4% at 1.0x neutral.
By year (mean multiplier / % of year at 1.25x): 2017-19 = 1.00 / 0%, 2020 = 1.15 / 61.7%,
2021 = 1.16 / 63.3%, 2022-23 = 1.00 / 0%, 2024 = 1.22 / 86.6%, 2025 = 1.25 / 100.0%,
2026 (partial) = 1.16 / 65.9%. Active, non-degenerate signal — not a no-op — but its "on" state
concentrates almost entirely in the two multi-year AI-buildout legs (2020-21, 2024-25), each
held near-continuously rather than toggling in and out.

## 6. Mechanism: does the entry-accelerator framing actually escape the exit-trigger's failure?

Partially, and only in one regime. The exit-trigger failed because a de-risk signal built from
a 2-4-year-lagging demand series arrives too late to call a turn — it protects nothing because
the drawdown it's reacting to has already happened. The entry-accelerator framing changes what
lag costs (a late "on" signal misses early upside rather than causing a late exit), and in the
2019-22 window that reasoning holds up: the signal added real, cost-free Sharpe by confirming
an already-underway recovery.

But the same lag mechanism reappears in a different form once capex growth stops
mean-reverting and simply stays elevated for years (2023-26): a trailing-average crossover
that's "on" 87-100% of the time is no longer *timing* anything — it's a near-permanent leverage
multiplier that only differs from a flat 1.25x from-day-one book at the margins. A book that is
essentially always levered up during its strongest, most levered-already-by-being-long-the-theme
window takes proportionally larger drawdowns in that window's pullbacks, which is exactly what
happened. **The entry-accelerator framing successfully escapes the exit-trigger's specific
failure mode (late confirmation of an already-priced move) in the recovery-from-trough case, but
does not escape a structural risk that's specific to sizing rules built on slow-moving,
persistently-elevated fundamentals: once the "trigger" condition becomes the norm rather than
the exception, the signal degenerates into undifferentiated leverage.**

## 7. Caveats

- Only 34 quarterly capex observations (2018Q1-2026Q2) back this signal; the full-sample result
  is driven by two regime transitions (2020 COVID-recovery reacceleration, 2023-24 AI-capex
  reacceleration) plus one multi-year "stuck on" period (2024-25), not a statistically rich
  sample.
- `REGIME_HI = 1.25` was reused unchanged from an unrelated overlay (grid-congestion Layer-2,
  which itself failed its own strict gate on the same book, per `docs/grid-regime-layer2-results.md`)
  rather than calibrated for this signal specifically — not a searched parameter, by design, but
  also not validated as the right cap for a mostly-on multiplier.
- No transaction-cost or turnover accounting beyond what the existing vol-target/valuation
  overlay machinery already includes; the accelerator changes exposure at most 34
  times over the sample (quarterly, point-in-time), so turnover is not a first-order concern
  here.
- Per this task's explicit instructions, this stays a probe — not wired into
  `electrification_strategy/` or `grid_equipment_basket/` regardless of the gate result.

## 8. Disposition

**Caveated PASS on the letter of the pre-registered gate; NOT recommended for adoption as
specified.** The full-sample two-part AND-gate is literally satisfied (Sharpe improves, MaxDD
does not worsen), which is the headline result this task asked for. But the honest read is that
this is two different findings glued together by the full-sample average: a real, cost-free win
confirming a genuine reacceleration off a trough (2019-22), and a structural-leverage drag that
deepens drawdowns in the very window (2023-26) this signal was built to help with. Before any
future adoption step, the open question is whether a variant that reads the crossover as a
one-time regime *transition* (e.g. only accelerate on the crossing quarter, decay back toward
neutral rather than holding 1.25x for as long as growth stays above-trend) fixes the "stuck at
the ceiling" failure mode — untried here, and out of scope for this probe.
