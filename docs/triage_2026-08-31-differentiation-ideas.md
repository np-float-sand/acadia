# Triage — three "what would they pay for?" ideas (2026-08-31)

Follows the grid-regime work (`docs/grid-regime-layer2-results.md`). The long
basket is recognised thematic beta; the congestion overlay did not clear its
bar and lost to the plain price gate out-of-sample. This session probed three
ways to find something differentiated. **All three came back weak.** Data /
probe code reused existing modules; no new strategy module was built.

---

## 1. "Sell the hedge" — package the risk overlay as drawdown insurance

**Premise:** the AI-power long is crowded and structurally un-hedgeable; the one
robust thing the overlay does is cut drawdowns, so sell it as portfolio
protection to people who already own the theme.

**Test (2020-01 → 2026-08, basket + each overlay):**

| calendar year | buy&hold | price-gate Δ | congestion Δ |
|---|---|---|---|
| calm years (2021, 2023, 2024) avg | — | **−27.7 pp/yr** | **−16.4 pp/yr** |
| drawdown years (2020, 2022, 2025) avg | — | −10.3 pp/yr | −2.5 pp/yr |

| worst peak-to-trough | |
|---|---|
| buy & hold | −45.4% |
| price gate + vol target | −41.5% |
| congestion overlay | −32.4% |
| **naive static 70% invested / 30% cash** | −33.7% |

Full period: buy&hold CAGR 40.5% / Sharpe 1.00; congestion overlay 32.0% / 1.09;
**static 70% 30.3% / 1.00** — the overlay barely beats a dumb permanent de-lever.

**Verdict: WEAK.**
- The "premium" is 16–28 pp/yr of forgone return in calm years. A rolled 6-month
  10%-OTM put program on a ~36%-vol basket costs **~10%/yr**. The overlay is a
  *more expensive* hedge than options.
- On a calendar-year basis it doesn't reliably help even in the bad years — it
  misses the V-shaped recoveries (2020, 2023).
- It barely beats "just hold 70%." The signal is adding almost nothing over
  static sizing.
- **The only real, sellable sub-component:** the price trend gate as an
  *episode-level* tail cutter — it halved-to-thirded the 2022 (−14% vs −30%),
  DeepSeek (−14% vs −41%) and 2026 (−11% vs −29%) drawdowns. But it's a bad
  standalone strategy (CAGR 40%→24%). Only useful to a client who explicitly
  wants a hard floor and will pay for it in upside.

---

## 2. Interconnection-queue velocity — the forward order book

**Premise:** large-load MW entering ISO queues is a 2–4-yr forward demand book
for this equipment, disclosed by the grid operator, not in anyone's model.

**Data check (unchanged from 15 months ago):**
- `seed/ercot_large_load_monthly.csv`: 11 monthly points, **all back-filled from
  a single March-2026 TAC report** — one vintage, system-wide totals only, no
  TSP/zone split. Not a panel.
- `cache/pjm_large_load_adjustment_2026.parquet`: 399 rows (zone × year ×
  industry, forecast to 2040) but **one vintage** (2025-09-16 LAS deck). No
  history of how the forecast evolved — which is exactly what "velocity" needs.
- `cache/pjm_interconnection_queue.parquet`: 1,294 rows with real history back to
  1997 — but it's **generation** interconnection (supply side), not large-load
  demand.

**Verdict: NOT TESTABLE, same as before.** The forward-demand-book data cannot
be assembled into a backtestable time series today. Doing this properly is a
multi-year data-collection project: start archiving the quarterly PJM LAS decks
and ERCOT TAC reports now, and/or a serious archive dig, then revisit ~2028. It
is the only idea with genuine alpha potential, but it is not a strategy that can
be built or validated now.

---

## 3. Un-crowd the universe — non-marquee, bottleneck-selected names

**Premise:** the crowding is in VRT/GEV/ETN/PWR; a basket of second-derivative
US-listed names picked by physical-bottleneck logic is a different product.

**Test:** equal-weight basket of POWL (switchgear), ATKR (conduit/cable), AZZ
(galvanising + electrical for T&D), SPXC (transformers + HVAC), BWXT (nuclear /
SMR), GNRC (backup power), CMI (power gen), ITRI (grid-edge / meters), BE
(on-site power), WCC (electrical distribution).

| 2020-01 → 2026-08 | CAGR | Sharpe | Max DD | corr to marquee |
|---|---|---|---|---|
| **uncrowded (10)** | 34.9% | 0.92 | −52.0% | **0.84** |
| marquee (9) | 40.5% | 1.00 | −45.4% | — |
| combined (19) | 37.6% | 0.98 | −49.3% | — |

Primary window (2023-26): uncrowded 50.1% / 1.31 / −33% vs marquee 54.2% /
1.26 / −41% — a marginally smoother ride recently, but lower return; reverses in
2020–22 (uncrowded Sharpe 0.53 vs marquee 0.69). Per-name correlation to the
marquee basket 0.52–0.68.

**Verdict: WEAK.** The non-marquee names **underperform** the marquee basket on
return and Sharpe over the full period, carry **deeper drawdowns** (−52%), and
at 0.84 basket correlation they **don't diversify** — same beta, more
idiosyncratic noise. The marquee names have been the better performers *because*
they're the quality names in the theme; "un-crowding" trades quality for
obscurity and gets a worse portfolio. Crowding has not impaired the marquee
basket's performance to date.

---

## Bottom line

Counting this session: ~8 distinct cross-sectional / signal attempts
(backlog-growth, value-chain, backlog-surprise, grid-demand sensitivity,
transmission rate-base, the 6-rung congestion ladder, relative-congestion A,
signal-tilted pair B) plus these three — **no demonstrated edge in this theme
beyond thematic beta and basic risk management.** The out-of-sample 2026 data
made the congestion signal weaker, not stronger.

Realistic paths, honestly:

1. **Accept it as a well-run thematic sleeve** and compete on packaging, risk
   discipline, transparency and access — not on alpha. The basket + a simple
   trend/vol de-lever is a legitimate product; it is not a differentiated one.
2. **Commit to the queue-velocity data-collection project** (idea 2) if the
   thesis is believed — but that is a 2–3-year bet with no interim validation.
3. **Move the search to a less picked-over area.** Eight-plus honest attempts in
   one theme with one macro cycle of data is a strong prior that the edge, if it
   exists, is not reachable with this data.
