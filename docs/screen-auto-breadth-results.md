# `screen_auto` Part A — breadth levers, and the price panel that was doing the work

**Date:** 2026-09-23 · **Handoff:** `docs/handoff-2026-09-23-screen-auto-breadth-hedge-costs.md` Part A
· **Window:** 2019-01 → 2026-08 · **Overlay:** shipped (20% vol target + valuation de-lever + 15% GLD/IEF)
· **Convention:** gross of costs, equal-weight, 25% cap, 42-day lag, 252-day listing gate.

**Verdict: the breadth target was already met — the deficit was a measurement artifact, not a
property of the rule. `A1` (lookback 2→4) is the only lever that does not cost Sharpe. `A3` and
`A4` each cost Sharpe alone; `A1` pays for `A4`. `A2` is NOT NEEDED.**

**The material finding is not a lever.** `screen_auto` has been evaluated through a 48-ticker price
panel built from the 33-name hand-recorded seed list it exists to replace. That restriction is
where the "14 effective names" figure comes from, and it is also where most of `screen_auto`'s
measured edge comes from.

---

## 1. Control — the harness reproduces the shipped numbers

Before measuring anything, the published figures were reproduced through the package's own
`run_comparison` path (`research` scratch, 48-ticker panel, listing gate given pre-2019 history):

| book | CAGR | Sharpe | MaxDD | Calmar | effective N | published |
|---|---|---|---|---|---|---|
| `screen` +val+sleeve | 21.0% | **1.000** | −23.17% | 0.905 | 27.0 | 1.000 / −23.2% / 0.905 ✓ |
| `screen_auto` +val+sleeve | 23.5% | **1.132** | −19.77% | 1.188 | **14.0** | 1.132 / −19.8% / 1.188 ✓ |

Exact match. Separately, the membership frame rebuilt from cache is **byte-identical** to the
committed build artifact — 0 of 25,662 date × ticker cells differ. The rule and the artifact are
sound; everything below is about the panel the rule is evaluated on.

---

## 2. The breadth deficit is mostly a price-panel artifact

`run_comparison` fetches prices for `MARQUEE_UNIVERSE ∪ universe.load_seed()` — 33 tickers, plus
benchmarks and sleeve = 48. `screen_auto` picks from the ~100-name mechanical SIC universe, so any
name it selects that is not on the hand-recorded seed list **has no price column and is silently
dropped before it can be traded**.

At the last rebalance `screen_auto` holds **31** names. **14** are on the seed panel. The other
**17** — AZZ, CW, DY, FELE, IEX, KEYS, LII, MIR, MKSI, MOG-A, MWA, PLPC, TDY, TMO, WTS, WWD, XYL —
are discarded.

| stage | mean names/rebalance | at last rebalance |
|---|---|---|
| membership (`screen_auto_members.csv`) | **18.52** | 29 |
| ∩ broad mechanical price panel | 18.52 | 29 |
| ∩ 252-day listing gate | **18.52** | 29 |
| ∩ seed-shaped panel (what `run_comparison` sees) | **11.16** | **14** |

**The listing gate costs nothing** — membership and tradable are identical at every rebalance. The
entire 18.5 → 11.2 loss is the seed-list-shaped price panel.

Against the handoff's breadth target — *"no month below ~10 names, and effective N above ~20 at the
last rebalance (currently 14)"* — on its own universe `screen_auto` already runs **29 effective at
the last rebalance** and is below 10 names at only **3 of 31 rebalances**, all in 2019. On the seed
panel it is below 10 at **9 of 31**, through to 2021-02.

### 2.1 The panel restriction is also where the edge comes from

Same membership rule, same harness, same overlay — only the investable panel differs:

| panel | names/mo | effective N (last) | CAGR | Sharpe | MaxDD | Calmar | Sharpe 2019-22 | Sharpe 2023-26 |
|---|---|---|---|---|---|---|---|---|
| seed-shaped (as published) | 11.16 | 14 | 23.5% | **1.129** | −18.73% | **1.253** | **0.762** | 1.533 |
| broad / genuinely mechanical | 18.52 | 29 | 22.4% | **1.049** | −20.41% | 1.099 | 0.616 | 1.534 |

Running `screen_auto` **without** the seed list in the loop costs **−0.080 Sharpe, −1.7pp MaxDD,
−0.154 Calmar**, and almost all of it lands in **2019-22** (0.762 → 0.616); 2023-26 is unchanged
(1.533 vs 1.534).

This is consistent with the precision problem already logged in
`docs/screen-auto-mechanical-universe-results.md` §2 — the vocabulary filter admits 53 non-`screen`
filers, and the 17 dropped names include life-science and instrument businesses (TMO, KEYS, MKSI,
TDY, IEX) that are off-thesis. The seed list was quietly filtering those out.

**Consequence for the record:** `screen_auto`'s headline result — Sharpe 1.132 vs `screen`'s 1.000,
and the pre-registered gate in §3 of that doc — was obtained with the hand-recorded seed list still
acting as a universe filter. Run genuinely seed-free it is **1.049**: still above `screen`, but by
0.05 rather than 0.13, and the gate has not been re-run on that basis. The module's stated purpose
("no seed list") is not yet what the evaluation measures.

---

## 3. The 2019 thinness is an EDGAR as-filed coverage edge, not a filter setting

The first rebalance (2019-02-11) holds **2** names. That is a fundamentals-panel problem, and no
lever touches it:

| date | eligible (liquidity + lexicon) | with a capex score | S2 +vocab | S3 +profit | S4 −policy |
|---|---|---|---|---|---|
| 2019-01-31 | 100 | **47** | 4 | 3 | **2** |
| 2019-02-28 | 100 | **74** | 15 | 11 | 9 |
| 2019-03-31 | 100 | 77 | 15 | 11 | 9 |

Scored-as-% -of-eligible, by month × year, shows January 2019 is a **one-off**, not a seasonal
trough — every other January runs 78–89%:

| month | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| Jan | 79 | 81 | **47** | 89 | 83 | 78 | 82 | 84 | 81 | 80 |
| Feb | 81 | 83 | 74 | 85 | 83 | 80 | 83 | 83 | 82 | 80 |

Two fixes were tried and **both failed**:

1. **Warming the panel** — rebuilding the TTM ladder from 2016-01 instead of 2019-01 (415s). Jan-2019
   coverage is **still 47%** (Dec-2018 48%, Nov-2018 49%). The facts are not there to warm.
2. **A1 (lookback 4)** — inert. The cached hit table's earliest `file_date` is **2017-01-12**
   (`fetch_vocabulary_hits` queries from `min(years) − AUTO_LOOKBACK_YEARS` = 2017), so universe-year
   2019 reads [2017, 2018] under *both* lookback 2 and lookback 4. All 12 grid variants give
   **2 names** at 2019-02-11.

So 2019-01 sits at the edge of EDGAR's as-filed vintage coverage. The honest options are to treat it
as a warm-up month or to start the evaluation at 2019-02 — not to loosen a filter until it fills.

---

## 4. The lever grid

Broad panel, shipped overlay, full window. **Keep only levers that do not cost Sharpe.**

| id | variant | names/mo | N (last) | Sharpe | ΔSharpe | MaxDD | Calmar | Sh 2019-22 | Sh 2023-26 | keep? |
|---|---|---|---|---|---|---|---|---|---|---|
| — | **baseline** (lb2, pol≥1, q0.50) | 18.52 | 29 | 1.049 | — | −20.41% | 1.099 | 0.616 | 1.534 | — |
| A1 | lookback 3 | 18.74 | 30 | 1.054 | +0.005 | −20.69% | 1.087 | 0.613 | 1.551 | ~ |
| **A1** | **lookback 4** | 18.94 | 30 | **1.063** | **+0.014** | −20.32% | 1.115 | 0.610 | 1.572 | **KEEP** |
| A3 | policy ≥2 hits | 18.77 | 30 | 1.038 | −0.011 | −20.41% | 1.089 | 0.592 | 1.538 | DROP |
| A3′ | policy ≥3 hits | 19.45 | 33 | 1.047 | −0.002 | −20.30% | 1.104 | 0.630 | 1.513 | ~ |
| A4 | quantile 0.60 | 20.42 | 34 | 1.031 | −0.018 | −20.10% | 1.096 | 0.610 | 1.502 | DROP alone |
| A4′ | quantile 0.70 | 20.45 | 34 | 1.032 | −0.017 | −20.10% | 1.097 | 0.610 | 1.504 | DROP |
| | A3+A4 | 20.68 | 35 | 1.019 | −0.030 | −20.10% | 1.086 | 0.586 | 1.505 | DROP |
| | A1+A3 (lb4, pol≥2) | 19.19 | 31 | 1.062 | +0.013 | −20.54% | 1.103 | 0.603 | 1.576 | ok |
| **A1+A4** | **lb4, q0.60** | **20.94** | **35** | **1.050** | **+0.001** | **−20.01%** | **1.118** | 0.603 | 1.552 | **KEEP** |
| | A1+A3+A4 (lb4, pol≥2, q0.60) | 21.19 | 36 | 1.048 | −0.001 | −20.24% | 1.105 | 0.597 | 1.555 | ~ |
| | A1+A3′+A4 (lb4, pol≥3, q0.60) | 21.84 | 37 | 1.037 | −0.012 | −20.35% | 1.086 | 0.615 | 1.515 | DROP |

**Reads**

1. **A1 is the only free lever.** Lookback 4 adds breadth *and* Sharpe (+0.014) and Calmar (+0.016).
   Its measured gain is a **lower bound** — inert for universe-years 2019–20 and only fully effective
   from 2021 (hit cache starts 2017), so a refetch from 2015 is needed to price it honestly.
2. **A3 does not do what the handoff expected.** Requiring ≥2 policy hits recovers +0.26 names/mo
   (not ~2.5) and costs −0.011 Sharpe. The ~2.5-name S3→S4 gap is real but the names it restores are
   bad ones. ≥3 hits is breadth-positive and Sharpe-neutral but weakens 2023-26.
3. **A4 alone costs Sharpe** (−0.018) — the light half is already the right cut; widening to 0.60
   admits weaker names. Note 0.60 and 0.70 are near-identical (34 names both), so the quantile is
   flat past ~0.6.
4. **A1 pays for A4.** Together: +2.42 names/mo, +6 names at the last rebalance, Sharpe +0.001,
   **best Calmar (1.118) and best MaxDD (−20.01%) of any broad variant**. This is the best
   breadth-per-Sharpe trade available.
5. **A2 (10-K → 10-K+10-Q) is not needed.** Its purpose was breadth; breadth already clears the
   target. Do not pay the ~30-minute refetch for it. A refetch **back to 2015** for A1 is the better
   use of the same budget.

---

## 5. What changed in the code

Behaviour-preserving parameterisation only, TDD (failing test before each change), 602 tests pass
(597 + 5 new):

- `auto_screen.vocabulary_years(..., lookback=None, min_hits=1)` — the lookback is now an argument
  rather than a global read, and hits are counted **per universe-year window** (`[Y-lookback, Y-1]`),
  not lifetime, so a name with one hit a year never clears `min_hits=2`.
- `auto_screen.build_membership(..., lookback=None, policy_min_hits=None)` — threads both through.
- `config.AUTO_POLICY_MIN_HITS = 1` — new knob, defaulting to the shipped behaviour.

**No shipped default was changed.** `AUTO_LOOKBACK_YEARS` is still 2 and `AUTO_CAPEX_QUANTILE` still
0.50 — adopting A1 or A1+A4 is a decision for the user, and it should be taken together with the
panel question in §2, which moves the baseline those levers are measured against.

---

## 6. Open decisions

1. **The price panel (§2.1) — the big one.** Should `screen_auto` be evaluated on the mechanical
   universe it selects from? Doing so is correct on the module's own terms but re-bases every
   published `screen_auto` number (Sharpe 1.132 → 1.049) and requires re-running the pre-registered
   gate. Leaving it is defensible only if the seed list is re-declared as a deliberate universe
   filter — which contradicts the point of the build.
2. **A1 refetch.** Re-query EDGAR FTS from 2015 (~30 min) so lookback 4 can be measured over the
   whole window rather than 2021+.
3. **A1 / A1+A4 adoption**, once (1) is settled.
4. Parts **B** (funded hedge) and **C** (net of costs) are untouched — C in particular should wait,
   since membership has not settled.
