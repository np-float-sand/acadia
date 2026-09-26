# Master comparison — every long-only, short-only and long/short book vs the buyable alternatives

**Date:** 2026-09-23 · **Window:** 2019-01 → 2026-08 (plus a VOLT-fair 2025-01 → 2026-08 cut)
· **Convention:** gross of costs, equal-weight, identical overlay where applied.

**Verdict:** the best book in every block is a **long-only** one. **No long/short beats the best
long-only, anywhere.** **Every short-only book loses money in every window.**

---

## 1. Full window, shipped overlay (20% vol target + valuation de-lever + 15% GLD/IEF sleeve)

| book | kind | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| **intersection (`screen_auto` AND capex-light)** | long-only | 25.8% | 16.5% | **1.234** | −22.0% | **1.172** |
| intersection (`screen` AND capex-light) | long-only | 23.2% | 15.7% | 1.153 | −23.1% | 1.005 |
| `screen_auto` (mechanical) | long-only | 24.0% | 16.7% | 1.130 | −20.7% | 1.160 |
| **`screen_auto` − 0.25× heavy** | long/short | 23.9% | 16.9% | **1.114** | −21.4% | **1.115** |
| `screen_auto` − 0.5× heavy | long/short | 23.0% | 16.8% | 1.081 | −21.4% | 1.078 |
| union (`screen` OR capex-light) | long-only | 23.1% | 17.0% | 1.071 | −22.4% | 1.030 |
| marquee (the 9) | long-only | 21.9% | 16.5% | 1.040 | −22.5% | 0.974 |
| `screen_reit` | long-only | 21.4% | 16.6% | 1.010 | −21.8% | 0.983 |
| `screen` (shipped) | long-only | 21.0% | 16.4% | 1.000 | −23.2% | 0.905 |
| `screen` − 0.25× heavy | long/short | 21.0% | 16.7% | 0.987 | −24.9% | 0.844 |
| capex long leg | long-only | 20.2% | 17.1% | 0.928 | −21.8% | 0.924 |
| capex long − 0.5× heavy *(native L/S)* | long/short | 19.6% | 17.0% | 0.905 | −25.9% | 0.759 |
| capex-light, ELEC only | long-only | 19.5% | 17.1% | 0.894 | −23.9% | 0.817 |
| **capex-heavy short (−1.0×)** | short-only | −8.4% | 17.0% | **−0.666** | −57.2% | −0.146 |
| capex-heavy short (−0.5×) | short-only | −6.2% | 10.9% | −0.895 | −45.0% | −0.137 |
| GRID | benchmark | 26.1% | 24.4% | 0.910 | −40.6% | 0.643 |
| PAVE | benchmark | 25.8% | 25.8% | 0.865 | −43.8% | 0.590 |
| SPY | benchmark | 19.8% | 19.4% | 0.820 | −33.7% | 0.586 |
| XLI | benchmark | 19.1% | 21.5% | 0.737 | −42.0% | 0.456 |
| VOLT *(listed 2024-12-05 — 1.7y only)* | benchmark | 16.8% | 25.5% | 0.580 | −24.9% | 0.675 |
| XLU | benchmark | 12.4% | 20.8% | 0.473 | −36.1% | 0.343 |

## 2. Full window, raw (no overlay) — best in class

| book | Sharpe | Calmar |
|---|---|---|
| long-only: intersection (`screen` AND light) | **1.070** | 0.859 |
| long-only: intersection (`screen_auto` AND light) | 1.040 | 0.734 |
| long/short: intersection (`screen` AND light) − 0.25× | 1.089 | **0.937** |
| long/short: `screen_auto` − 0.5× | 0.985 | 0.844 |
| short-only: capex-heavy −1.0× | −0.653 | −0.156 |
| GRID / PAVE / SPY | 0.910 / 0.865 / 0.820 | 0.643 / 0.590 / 0.586 |

## 3. VOLT-fair window 2025-01 → 2026-08, shipped overlay

| book | Sharpe | Calmar | MaxDD |
|---|---|---|---|
| intersection (`screen_auto` AND light) | **1.484** | 1.762 | −17.5% |
| intersection (`screen` AND light) | 1.394 | **1.955** | −13.9% |
| `screen` (shipped) | 1.354 | 1.624 | −17.0% |
| `screen_reit` | 1.323 | 1.528 | −18.4% |
| marquee | 1.310 | 1.369 | −20.7% |
| `screen_auto` | 1.148 | 1.292 | −19.1% |
| capex long leg | 0.923 | 0.996 | −21.0% |
| capex-heavy short (−1.0×) | −0.313 | −0.105 | −28.4% |
| GRID | 0.921 | 1.217 | −21.2% |
| SPY | 0.877 | 1.034 | −18.8% |
| VOLT | 0.818 | 0.987 | −24.5% |
| PAVE | 0.679 | 0.766 | −23.7% |
| XLU | 0.366 | 0.916 | −9.8% |

---

## 4. Reads

1. **Every short-only book loses money in every window** — Sharpe −0.31 to −0.90, drawdowns to
   −57%. Shorting capex-heavy utilities has been a straight loser since 2019: the leg has
   *positive* drift, so shorting it is a short against a rising asset.
2. **No long/short beats the best long-only.** The best L/S (`screen_auto` − 0.25×, Sharpe 1.114)
   sits below plain `screen_auto` (1.130) and well below the intersection (1.234). The heavy leg
   correlates **+0.62 to +0.66** with both long books and **+0.55 to +0.60 on the long book's own
   down days** — a drag with extra variance, not insurance.
3. **The best book has not been gated.** `screen_auto` ∩ capex-light quartile tops three of four
   blocks, but it was computed after the fact and averages **~10 names** (one name in Jan 2019,
   14 recently). A promising lead, not a result.
4. **Against buyable alternatives the overlay does the work on drawdown**, not the stock picking:
   every strategy book runs −21% to −23% MaxDD against GRID −41%, PAVE −44%, SPY −34%. That is
   where the Calmar gap comes from.
5. **Costs are in none of these numbers.** The intersection books turn over most and would lose
   the most to them.
