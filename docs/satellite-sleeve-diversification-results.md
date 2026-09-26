# Satellite-sleeve diversification test — miners/generators alongside the shipped book

**Date:** 2026-09-18
**Type:** Mechanical portfolio-construction test. **NOT a signal/hypothesis test** — no
significance gate, no p-hacking risk to guard against. The question is purely: does holding a
fixed-weight, equal-weight, non-timed basket of crypto-miner and/or merchant-generator stocks
alongside the shipped `electrification_strategy` book improve combined risk-adjusted returns
through diversification (low correlation), independent of whether those names have any
predictive signal of their own.
**Origin:** `docs/electricity-price-volatility-miner-results.md` (2026-09-16) tested the same
names (CIFR/WULF primary miners; MARA/RIOT/IREN/BTDR widened; VST/NRG/CEG/TLN generators) as an
electricity-price **predictive signal** — FAILED on every arm. That result says nothing about
whether the same names, held with zero timing, diversify the book — this probe asks that
question directly. Tracked as `docs/RESEARCH-LOG.md` §4 item 25.
**Code:** `research/satellite_sleeve_diversification/` (`universe.py`, `data.py`, `sleeve.py`,
`run_probe.py`). Reproduce with `python3 -m research.satellite_sleeve_diversification.run_probe`
(writes `research/satellite_sleeve_diversification/_output/combination_results.csv` and
`sleeve_windows.json`).

**Headline: NOT ADOPTED.** The two requested test windows turn out to be the *same* window (see
§2) — a single ~3.25-3.5 year AI/crypto-bull stretch (2023-04/06 -> 2026-09-17), because the
youngest name in every sleeve (BTDR for miners, TLN for generators/combined) only has genuine
trading history from mid-2023. Within that one window, the generators and combined sleeves raise
combined Sharpe modestly (+0.02 to +0.05) at every weight tested against both base-book variants,
but MaxDD is **worse, with no exception, in all 36 combinations tested** — Sharpe/Calmar improve
because the sleeves' own CAGR is very high (63-87%/yr) in this window, not because their
volatility is low. Sleeve-to-book correlation is moderate (0.49-0.62), not low. This cannot be
distinguished, on this data, from "these stocks also rallied hard in the same bull market as the
book" — it is not a clean demonstration of low-alpha diversification. The miners-only sleeve
fails outright past 5% weight (its own -61.6% standalone MaxDD dominates the blend).

---

## 1. Method

Three sleeves, equal-weight internally, no re-selection:

| sleeve | tickers |
|---|---|
| miners | CIFR, WULF, MARA, RIOT, IREN, BTDR, CLSK (all 7 -- CLSK's exclusion from the signal probe was about hub-proxy defensibility, irrelevant to a plain basket) |
| generators | VST, NRG, CEG, TLN |
| combined | equal-weight union of both (11 names) |

Tested at fixed weights **5%, 10%, 15% of NAV**, funded pro-rata out of the base book, against
two base-book cells: **`screen +val+sleeve`** (primary -- the name used everywhere else in this
project, including the just-finished crypto-miner signal probe) and **`screen_reit +val+sleeve`**
(secondary -- `docs/electrification-reit-breadth-probe-results.md`: strictly dominates `screen
+val+sleeve` in this lineage on both Sharpe and MaxDD).

**Construction matches `electrification_strategy/hedge_overlay.py`'s existing GLD/IEF sleeve
exactly -- no new methodology invented:**
- Within-sleeve return: equal-weight mean of each name's daily simple return (same form as
  `hedge_overlay._ew_returns`) -- implicitly rebalanced daily.
- Blend into the base book: fixed weight applied every day, `(1-w)*book_r + w*sleeve_r` (same
  form as `hedge_overlay.apply_hedge`'s sleeve term) -- a constant daily weight is equivalent to
  rebalancing the book/sleeve split back to target daily, matching the cadence the shipped
  overlay already uses (no monthly-hold layer exists for the GLD/IEF sleeve either).
- `book_r` is the **fully composed** `+val+sleeve` cell return series straight out of
  `electrification_strategy.backtest.run_comparison()` (vol-target + valuation overlay + the
  shipped GLD/IEF sleeve already applied) -- the new satellite sleeve is layered on top of the
  complete shipped book, not on the raw basket.
- Fetcher: `grid_equipment_basket.data.prices.fetch_prices` (same one
  `electrification_strategy/__main__.py` itself uses). Checked side-by-side against
  `grid_resilience.data.equity_prices.fetch_prices` (the module that silently dropped ticker
  columns under a threaded per-month cache warmer in the 2026-09-16 miner probe) before use:
  `grid_equipment_basket`'s version issues one whole-span `yf.download` call per missing-months
  gap and never fans work out across a `ThreadPoolExecutor`, so it does not hit that bug. No new
  fetcher or non-threaded workaround was needed.

Metrics: Sharpe, MaxDD, CAGR from `grid_equipment_basket.backtest.compute_metrics` (same function
the shipped book itself is scored with); Calmar = CAGR / abs(MaxDD) computed on top (not in that
function). Book-alone metrics are **recomputed over the exact same truncated window** as the
combined-portfolio metrics in every row below, for a fair apples-to-apples comparison -- they are
not the book's longer-sample headline numbers from `docs/RESEARCH-LOG.md` §1.

---

## 2. A data-quality correction, and why it collapses both requested windows into one

Raw yfinance history for three of the four "2021-2023 vintage" miner tickers is contaminated by
pre-merger SPAC-shell or unrelated-company trading under the same ticker symbol -- confirmed by
inspecting the fetched price series directly:

| ticker | raw yfinance first date | what that early data actually is | corrected effective start |
|---|---|---|---|
| CIFR | 2020-10-20 | flat ~9.90-9.98 SPAC-trust band; jumps to 11.34 on 2021-08-26 (Cipher Mining business-combination close) | **2021-08-26** |
| WULF | 2005-01-03 (ticker-dependent) | flat/quasi-flat 8-9.6 band for years; jumps to 15.82 on 2021-06-25 (reverse-merger uplisting) | **2021-06-25** |
| BTDR | 2021-07-28 | flat ~10.70-10.85 SPAC-trust band; resets to 7.03 on 2023-04-14 (Bitdeer business-combination close) | **2023-04-14** |
| IREN | 2021-11-17 | real, moving prices from day one -- no correction needed | 2021-11-17 (raw) |

Using the raw dates would silently splice years of near-zero-volatility, business-irrelevant
data onto the front of the sleeve, deflating its measured volatility and corrupting the
correlation-to-book number over that period -- exactly the kind of bias the task asked to avoid
rather than commit. `universe.py`'s `EFFECTIVE_START_OVERRIDES` hard-codes the three corrected
dates with this evidence inline. MARA/RIOT/CLSK also carry older (pre-2017/2018) history as
unrelated businesses before their own bitcoin-mining pivots, but none of the three is the
sleeve's binding constraint, so it does not affect the reported windows.

**Result: the sleeve-wide common start (the latest effective-start date across every name in the
sleeve) is later than both requested window boundaries for every sleeve:**

| sleeve | binding ticker | common start | "full available sample" window used | "2023-01-01 -> present" window used |
|---|---|---|---|---|
| miners | BTDR | 2023-04-14 | 2023-04-14 -> 2026-09-17 | 2023-04-14 -> 2026-09-17 (identical) |
| generators | TLN | 2023-06-02 | 2023-06-02 -> 2026-09-17 | 2023-06-02 -> 2026-09-17 (identical) |
| combined | TLN | 2023-06-02 | 2023-06-02 -> 2026-09-17 | 2023-06-02 -> 2026-09-17 (identical) |

Both requested windows are reported below, but for every sleeve they are numerically identical
(same start date, same n, same return series) -- stated explicitly here rather than silently
merged. **This means no multi-regime test is possible with these tickers as currently listed**:
there is no window in this data covering a bear market, a rate shock, or any regime other than
the 2023-2026 AI/crypto/power-demand bull market. The book's own Sharpe inside this window
(1.52-1.60, see §3) is far above its documented full-sample Sharpe (0.69, 2017-2026,
`docs/RESEARCH-LOG.md` §1) -- this is a favorable-regime slice for the *base book* too, not only
for the sleeve, which further limits how far this result generalizes.

---

## 3. Results -- both requested windows (identical for every sleeve; see §2)

n = 860 trading days (miners, 2023-04-14 -> 2026-09-17); n = 826 (generators/combined,
2023-06-02 -> 2026-09-17).

### Miners sleeve (CIFR/WULF/MARA/RIOT/IREN/BTDR/CLSK)

Sleeve standalone: Sharpe 0.98, CAGR 68.2%/yr, MaxDD -61.6%.

| weight | base book | book Sharpe | combined Sharpe | dSharpe | book MaxDD | combined MaxDD | dMaxDD | book Calmar | combined Calmar | dCalmar | book CAGR | combined CAGR | corr(sleeve,book) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5% | screen | 1.602 | 1.611 | +0.009 | -18.9% | -20.8% | -2.0pp | 1.67 | 1.69 | +0.02 | 31.4% | 35.2% | 0.49 |
| 10% | screen | 1.602 | 1.561 | -0.041 | -18.9% | -22.8% | -4.0pp | 1.67 | 1.70 | +0.04 | 31.4% | 38.8% | 0.49 |
| 15% | screen | 1.602 | 1.491 | -0.111 | -18.9% | -24.7% | -5.9pp | 1.67 | 1.71 | +0.04 | 31.4% | 42.4% | 0.49 |
| 5% | screen_reit | 1.553 | 1.571 | +0.018 | -18.7% | -20.7% | -2.0pp | 1.67 | 1.69 | +0.02 | 31.2% | 34.9% | 0.49 |
| 10% | screen_reit | 1.553 | 1.530 | -0.023 | -18.7% | -22.7% | -4.0pp | 1.67 | 1.70 | +0.04 | 31.2% | 38.6% | 0.49 |
| 15% | screen_reit | 1.553 | 1.467 | -0.086 | -18.7% | -24.6% | -5.9pp | 1.67 | 1.71 | +0.04 | 31.2% | 42.1% | 0.49 |

Only the 5% weight raises Sharpe, and only marginally (+0.009 / +0.018); 10% and 15% both lower
it. MaxDD worsens monotonically with weight in every case.

### Generators sleeve (VST/NRG/CEG/TLN)

Sleeve standalone: Sharpe 1.24, CAGR 62.7%/yr, MaxDD -38.9%.

| weight | base book | book Sharpe | combined Sharpe | dSharpe | book MaxDD | combined MaxDD | dMaxDD | book Calmar | combined Calmar | dCalmar | book CAGR | combined CAGR | corr(sleeve,book) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5% | screen | 1.560 | 1.588 | +0.028 | -18.9% | -19.6% | -0.7pp | 1.63 | 1.66 | +0.03 | 30.7% | 32.6% | 0.62 |
| 10% | screen | 1.560 | 1.597 | +0.037 | -18.9% | -20.5% | -1.7pp | 1.63 | 1.68 | +0.05 | 30.7% | 34.4% | 0.62 |
| 15% | screen | 1.560 | 1.592 | +0.032 | -18.9% | -21.5% | -2.6pp | 1.63 | 1.69 | +0.06 | 30.7% | 36.2% | 0.62 |
| 5% | screen_reit | 1.519 | 1.552 | +0.033 | -18.7% | -19.4% | -0.7pp | 1.64 | 1.67 | +0.03 | 30.6% | 32.5% | 0.61 |
| 10% | screen_reit | 1.519 | 1.566 | +0.047 | -18.7% | -20.3% | -1.6pp | 1.64 | 1.69 | +0.06 | 30.6% | 34.3% | 0.61 |
| 15% | screen_reit | 1.519 | 1.566 | +0.047 | -18.7% | -21.2% | -2.6pp | 1.64 | 1.70 | +0.06 | 30.6% | 36.1% | 0.61 |

Sharpe improves at every weight against both base cells, peaking around 10%. MaxDD still worsens
in every single case (never improves), by 0.7-2.6pp.

### Combined sleeve (all 11 names)

Sleeve standalone: Sharpe 1.23, CAGR 86.7%/yr, MaxDD -49.4%.

| weight | base book | book Sharpe | combined Sharpe | dSharpe | book MaxDD | combined MaxDD | dMaxDD | book Calmar | combined Calmar | dCalmar | book CAGR | combined CAGR | corr(sleeve,book) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5% | screen | 1.560 | 1.599 | +0.039 | -18.9% | -20.4% | -1.5pp | 1.63 | 1.67 | +0.04 | 30.7% | 34.1% | 0.59 |
| 10% | screen | 1.560 | 1.602 | +0.042 | -18.9% | -21.9% | -3.0pp | 1.63 | 1.71 | +0.08 | 30.7% | 37.4% | 0.59 |
| 15% | screen | 1.560 | 1.584 | +0.024 | -18.9% | -23.4% | -4.6pp | 1.63 | 1.74 | +0.11 | 30.7% | 40.7% | 0.59 |
| 5% | screen_reit | 1.519 | 1.564 | +0.045 | -18.7% | -20.2% | -1.5pp | 1.64 | 1.68 | +0.04 | 30.6% | 34.0% | 0.59 |
| 10% | screen_reit | 1.519 | 1.573 | +0.054 | -18.7% | -21.8% | -3.1pp | 1.64 | 1.71 | +0.08 | 30.6% | 37.3% | 0.59 |
| 15% | screen_reit | 1.519 | 1.561 | +0.042 | -18.7% | -23.3% | -4.6pp | 1.64 | 1.74 | +0.11 | 30.6% | 40.6% | 0.59 |

Best of all nine (sleeve x weight) combinations tested: combined sleeve at 10% weight, +0.042 to
+0.054 Sharpe. MaxDD still worsens (-3.0 to -3.1pp) -- the single largest dMaxDD improvement
recorded anywhere in this probe is **zero**; it never happens.

**Calmar improves in all 18 rows above, in every case** -- but mechanically, because CAGR rises
faster (in relative terms) than |MaxDD| deepens, driven by the sleeve's own very high standalone
CAGR in this window, not because the blended series is less prone to deep drawdowns. Read
alongside the dMaxDD column, not as a substitute for it.

---

## 4. Verdict

**NOT ADOPTED**, for three separable reasons:

1. **MaxDD never improves.** Across all 3 sleeves x 3 weights x 2 base cells x 2 (collapsed)
   windows = 36 combinations, MaxDD is worse in literally every one. If the intent is drawdown
   control, this sleeve does the opposite at every tested weight.
2. **The Sharpe/Calmar improvement that does show up is not cleanly attributable to
   diversification.** The task's actual question was whether a *low-Sharpe, low-correlation*
   sleeve could lift combined Sharpe with zero alpha of its own. These sleeves are the opposite of
   that: standalone Sharpe 0.98-1.24 (higher than most factor strategies in this whole project),
   CAGR 63-87%/yr, and sleeve-to-book correlation 0.49-0.62 -- moderate, not low. That combination
   (high standalone Sharpe, positive-but-moderate correlation) is consistent with "the sleeve
   simply performed very well over the same period," which is a return-stacking effect, not the
   diversification effect being tested for.
3. **No multi-regime test is possible with these tickers.** §2's window collapse means both
   requested windows are the identical single AI/crypto-bull stretch (2023-04/06 -> 2026-09-17).
   There is no bear-market, rate-shock, or risk-off period in this data to check whether the
   correlation stays moderate (helpful) or spikes toward 1 (unhelpful, exactly when it would
   matter most) -- which is the standard failure mode of "diversifiers" that are actually just
   long high-beta risk assets. This probe cannot rule that out.

The **miners-only sleeve is the clearest failure**: it only helps at the smallest weight tested
(5%, +0.009/+0.018 Sharpe) and actively hurts at 10-15%, because its own -61.6% standalone MaxDD
is too large to blend at any real size without dragging the combined drawdown down with it.

The **generators and combined sleeves are the closest thing to a positive result** -- a small,
consistent Sharpe uplift (+0.02 to +0.05) at every weight tested, against both base-book variants
-- but per points 2-3 above, this is reported as **not adopted, not a candidate pending more
data**: the finding is confounded with a concurrent bull regime and cannot be stress-tested
against a drawdown scenario with the price history these tickers actually have.

---

## 5. Portfolio role -- how this differs from the signal probe's framing

`docs/handoff-2026-09-16-crypto-miner-power-volatility.md` framed two ways these names could be
useful even after a standalone signal FAIL: (a) a standalone satellite sleeve, or (b) a
risk-overlay input if the volatility-stabilization mechanism were real. This probe tested (a)
directly. The answer is **no, not as currently testable**: not because the numbers are bad in the
one window that exists (Sharpe *does* tick up for two of the three sleeves), but because that one
window cannot distinguish "this diversifies the book" from "this basket of extremely high-beta
AI/crypto/power names happened to rally alongside the book in the same 3-year bull market." A
sleeve that only shows its stripes in a bull market, and whose one closest brush with a stress
test (the miners sleeve's own -61.6% MaxDD) is actively destructive when blended at real size, is
not a diversifier by the standard this project holds risk overlays to elsewhere (e.g. the shipped
GLD/IEF sleeve, which was validated across COVID, the 2022 rate shock, DeepSeek, and the 2026
selloff -- `docs/electrification-short-leg-insurance-probe-results.md`). Revisit only once BTDR
and TLN (or their eventual replacements) have traded through a real down-regime -- there is no
shortcut to that with the price history these specific tickers have today.
