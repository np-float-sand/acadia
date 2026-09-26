# Screened Electrification Equipment vs Grid-Stress Relief — head-to-head

**Date:** 2026-09-21
**Basis:** both books run through the **production `compute_metrics`** at `config.RF = 0.04`,
same price data, same session, common period **2017-06-02 → 2026-09-18**. The shipped book's
returns come from the real production path (`backtest.run_comparison`), not a reconstruction.

---

## 1. What each one is

| | **Screened Electrification Equipment** | **Grid-Stress Relief (GSR)** |
|---|---|---|
| Selection rule | capex-cycle-pair classifier: *institutional customer + ≥2 of {profitable, earnings-valued, low-policy-dependence}*, **supplier side only** | grid physics: does the asset **relieve grid stress**? firm capacity + dispatchable generation + delivery buildout |
| What it owns | equipment/EPC **suppliers** to the buildout | suppliers **plus asset owners** (nuclear fuel cycle, merchant generation) |
| **Long names** | **27** equities + 2 sleeve ETFs (GLD/IEF, ~15%) | **23** equities (FIRM 6 · DISP 7 · DELIV 10) |
| **Short names** | **0** (`+val+sleeve`) · **2** in the `+short` variant (DLR/EQIX, real-yield gated) | **0** |
| Overlays | 42-day lag, 25% single-name cap, 20% vol target, valuation de-lever, GLD/IEF sleeve | 25% vol target (cap 1.5x), monthly rebalance |
| Deliberately excludes | asset owners (generators, fuel) | intermittent generation, commodity fuel |

They are **not independent ideas**: GSR keeps the equipment names, which is why they correlate
**0.78–0.85**.

---

## 2. Performance

### 2023–2026
| strategy | long / short | CAGR | Vol | **Sharpe** | MaxDD | **Calmar** |
|---|---|---|---|---|---|---|
| **Screened Electrification Eqpt** | 27 / 0 | 32.2% | 15.8% | **1.60** | −17.0% | 1.89 |
| ...`+short` variant | 27 / 2 | 29.4% | 14.5% | 1.57 | **−12.3%** | **2.39** |
| **Grid-Stress Relief** | 23 / 0 | **48.1%** | 26.1% | 1.48 | −24.5% | 1.96 |
| SPY | — | 22.1% | 15.0% | 1.14 | −18.8% | 1.18 |

### 2025–2026
| strategy | long / short | CAGR | Vol | **Sharpe** | MaxDD | **Calmar** |
|---|---|---|---|---|---|---|
| **Screened Electrification Eqpt** | 27 / 0 | 27.6% | 16.4% | 1.32 | −17.0% | 1.62 |
| ...`+short` variant | 27 / 2 | 28.5% | 14.9% | **1.48** | **−11.9%** | **2.40** |
| **Grid-Stress Relief** | 23 / 0 | **31.2%** | 26.7% | 1.00 | −24.5% | 1.27 |
| SPY | — | 18.0% | 17.2% | 0.82 | −18.8% | 0.96 |

### Full sample (2017-06 → 2026)
| strategy | long / short | CAGR | Vol | **Sharpe** | MaxDD | **Calmar** |
|---|---|---|---|---|---|---|
| Screened Electrification Eqpt | 27 / 0 | 14.8% | 15.7% | 0.70 | −24.0% | 0.62 |
| ...`+short` variant | 27 / 2 | 13.7% | 14.9% | 0.67 | −23.2% | 0.59 |
| **Grid-Stress Relief** | 23 / 0 | **31.9%** | 25.4% | **1.06** | −36.5% | **0.87** |
| SPY | — | 14.9% | 18.5% | 0.63 | −33.7% | 0.44 |

### Read
- **The shipped book wins Sharpe in both target windows** (1.60 vs 1.48; 1.32 vs 1.00). It is the
  better *risk-adjusted* book, at roughly **60% of GSR's volatility**.
- **GSR wins on raw return and on the full sample** (CAGR 48.1% vs 32.2%; full Sharpe 1.06 vs 0.70,
  Calmar 0.87 vs 0.62). The shipped book's screen was built for this theme and is weak pre-2023;
  GSR travels better across regimes.
- **The best single cell for 2025-26 is the shipped `+short` variant** — Sharpe 1.48, Calmar
  **2.40**, MaxDD only **−11.9%**. Nothing here beats it on drawdown.
- GSR's only clean win in the target windows is **2023-26 Calmar (1.96 vs 1.89)**, and it buys that
  with a 7.5pp deeper drawdown and 10pp more vol.

---

## 3. Would combining them help?

Correlation **0.83 (2023-26) / 0.85 (2025-26) / 0.78 (full)** — too high for real diversification.
Blends of `w·GSR + (1−w)·Screened`:

| w_GSR | 2023-26 Sh / Cal / DD | 2025-26 Sh / Cal / DD | Full Sh / Cal / DD |
|---|---|---|---|
| **0.00** (shipped alone) | 1.60 / 1.89 / −17.0 | **1.32 / 1.62 / −17.0** | 0.70 / 0.62 / −24.0 |
| 0.20 | **1.62** / 1.93 / −18.4 | 1.27 / 1.55 / −18.4 | 0.85 / 0.71 / −25.7 |
| 0.25 | **1.62** / 1.93 / −18.8 | 1.25 / 1.53 / −18.8 | 0.87 / 0.73 / −26.3 |
| 0.33 | **1.62** / **1.95** / −19.3 | 1.22 / 1.50 / −19.3 | 0.91 / 0.75 / −27.3 |
| 0.50 | 1.59 / 1.96 / −20.6 | 1.16 / 1.44 / −20.6 | 0.98 / 0.79 / −29.7 |
| 1.00 (GSR alone) | 1.48 / **1.96** / −24.5 | 1.00 / 1.27 / −24.5 | **1.06 / 0.87 / −36.5** |

**Answer: only marginally, and it depends entirely on which window you are underwriting.**

- **2023-26:** a 20–33% GSR sleeve is a small genuine improvement — Sharpe 1.60→1.62, Calmar
  1.89→1.95 — but costs 2.3pp of drawdown.
- **2025-26:** **every** GSR weight hurts. 1.32→1.27 at just 20%. If the recent window is what
  you're underwriting, add none.
- **Full sample:** GSR helps a lot (0.70→0.87 at 25%, →1.06 standalone), because it rescues the
  pre-2023 period where the supplier screen is weak.
- **No blend beats both standalone books on Sharpe *and* Calmar in any window.**

**Recommendation.** Don't blend for diversification — at 0.85 correlation there isn't any to
harvest. Treat this as a single decision about **which regime you are underwriting**:
if you believe the grid-stress regime persists and you want the most robust cross-regime book,
GSR (or a 25–33% sleeve) is defensible; if you are optimising the windows actually under
discussion, **the shipped `screen +val+sleeve+short` is still the best book on the table** —
Sharpe 1.48 and Calmar 2.40 in 2025-26 with an 11.9% max drawdown.

---

## 4. Correction recorded

An earlier statement this session claimed GSR beat the shipped book (1.556 → 1.84). **That was an
apples-to-oranges error**: GSR's 1.84 used a zero risk-free rate against the log's RF=4% figures.
On the identical production metric the shipped book wins Sharpe in both target windows. All GSR
numbers above and in `docs/grid-stress-relief-results.md` are now on the RF=4% basis.
