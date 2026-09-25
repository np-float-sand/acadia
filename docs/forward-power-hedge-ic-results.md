# Forward power × hedge-adjusted exposure → generator returns — results

**2026-09-24. Verdict: FAIL** (pre-registered bar |t| > 1.5, 2023-01 → 2026-08).
Pre-registration: `docs/forward-power-hedge-ic-preregistration.md` (written before returns were loaded).
Code: `research/forward_power_hedge_ic/` (`fetch_releases.py` → `hedges.py` → `probe.py`; panel in `panel.csv`).

## What was tested
Bushnell-Chong-Mansur (AEJ:Policy 2013) show equities price *power-price* news. Probe A (2026-09-07)
failed because hedges sever realised price → earnings. This probe puts the hedge in explicitly:

    signal = Σ_year w · Σ_hub (unhedged generation share) · %Δ STEO forward price (PJM West / ERCOT North)

evaluated at each point-in-time STEO vintage, against the next-21-trading-day return in excess of SPY.
Hedge % from each company's quarterly 8-K earnings exhibit (filing date = availability date).

## Data coverage (the binding constraint)
| name | hedge % disclosure | pairs in window |
|---|---|---|
| VST | every quarter 2022→2026, current + next 1-2 years ("hedged approximately X% ... for YYYY") | 21 |
| TLN | every quarter from relisting 2024-08 (incl. nuclear PTC) | 15 |
| CEG | regional table (expected GWh + % hedged by Midwest/Mid-Atlantic/ERCOT/NY) — **2023 decks only; dropped in 2024** | 5 |
| NRG | **none** — integrated retail nets its own load; excluded | 0 |

41 (name, vintage) pairs over 21 STEO vintages. With ≤3 names per date the per-date cross-sectional IC is
degenerate, so the primary stat was switched (before looking at returns) to a pooled panel rank-IC.

## Result
| test | rank-IC | t | n |
|---|---|---|---|
| **Primary: hedge-adjusted signal** | **+0.088** | **+0.55** | 41 |
| vintage-clustered bootstrap p (sign) | | p = 0.63 | |
| comparator: no hedge adjustment (h ≡ 0) | +0.039 | +0.24 | 41 |
| partial IC, controlling for the same-vintage Henry Hub revision | +0.211 | +1.35 | 41 |
| leave out VST / TLN / CEG | +0.003 / −0.030 / +0.216 | +0.01 / −0.15 / +1.29 | 20 / 26 / 36 |
| VST only / TLN only / CEG only | +0.189 / +0.244 / −0.900 | +0.84 / +0.91 / −3.58 | 21 / 15 / 5 |
| sensitivity: drop same-edition Wayback re-captures (Δ = 0) | +0.081 | +0.47 | 35 |

**FAIL.** Right sign (under-reaction) and hedge adjustment does beat the naive version (t 0.55 vs 0.24),
but both are noise; nothing clears 1.5 even after the gas control (1.35), and the CEG-only −0.90 is 5 points.

## Why it probably can't work in this form
1. **These names are nearly fully hedged for the years STEO covers.** VST is 94-100% hedged for the current
   year and 80-97% for next year in most filings; the unhedged share STEO can price is ~0-20%. The open
   exposure sits in Y+2/Y+3 (VST 2028 72% hedged, TLN 2028 30%), beyond STEO's ~16-month tail.
2. **STEO is a model forecast on irregular Wayback captures** (gaps up to 5 months; 3 of 21 vintages were
   re-captures of the same edition), so by the time a revision is observed the traded curve has long moved.
3. **n = 41 on 2-3 names.** Even a real effect of IC ≈ 0.2 needs n ≈ 60+ to reach t 1.5.

## What would make it testable (not run)
Daily *traded* forwards for Y+2/Y+3 (ICE/CME PJM West and ERCOT North calendar strips — paywalled for
history; CME delayed quotes are free going forward only) combined with the same hedge panel, which is
built and reusable (`hedges.py`). That is the only version that addresses reasons 1 and 2.
