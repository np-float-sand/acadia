# Electricity → Earnings → Equity Mispricing — Probe Results

**Run:** 2026-09-07
**Prompt:** external "dislocation alpha" critique + "Electricity Intelligence for Equity
Alpha" proposal — move one layer closer to the power market, but keep the instrument set
equities/options only. Two pre-registered spikes.
**Verdict:** both **FAIL**. The "physical electricity exposure → company earnings → equity
mispricing" thesis does not clear its own gate 1 (predict fundamentals) or gate 3
(survive factor neutralisation). Logged; not carried forward.

Code: `research/electricity_earnings_exposure/` (throwaway probes, re-run to regenerate).

---

## Probe A — merchant-generation margin retrodiction (VST, NRG, CEG, TLN)

**A1 gate (pre-registered):** a bottom-up model (owned fleet × realised ISO DA price −
fuel cost) must retrodict *reported* quarterly results — pooled Spearman
corr(Δmodel, Δreported) ≥ 0.50, QoQ sign-agreement ≥ 70%, both halves.

Fleet: hand-coded GW by (company, market, fuel) with acquisition effective-dates
(VST+Energy Harbor 2024-Q2, CEG spin 2022, TLN emergence 2023). Prices: ERCOT HB avg +
PJM Western Hub DA from local cache. Gas: Henry Hub daily (EIA). Reported:
EDGAR XBRL `Revenues` and `OperatingIncomeLoss` (energy filers don't tag `GrossProfit`).

| target | pooled Δ-corr | sign-agree | 2018–22 | 2023–25 |
|---|---|---|---|---|
| revenue | **+0.45** | 62% | +0.54 | +0.33 |
| operating income | **−0.05** | 48% | +0.00 | −0.07 |

Per name (operating income): CEG −0.46, NRG +0.10, TLN −0.29, VST −0.01.

**Read:** the fleet×price model tracks the *direction of revenue* (sanity check — the
mapping isn't broken) but has **zero** relationship to *operating income*. The path from
realised wholesale prices to reported earnings is severed by hedge mark-to-market, the
retail/again-non-generation books, D&A and impairments — none of which a public-data
model reconstructs quarter-to-quarter. This is the "informative kill" the plan
pre-registered: **A1 FAIL → A2 (returns test) not run.** Independitently corroborates the
`grid-stress-signal-evidence.md` §5 VST/NRG regime-instability finding, from the
fundamentals side.

---

## Probe B — electricity-cost exposure cross-section (~18 power-intensive industrials)

Universe: aluminium (AA, CENX, KALU), EAF steel (NUE, STLD, CMC, ATI), chlor-alkali
(OLN, WLK), industrial gas (APD, LIN), bitcoin/AI-hosting miners (MARA, RIOT, CLSK, WULF,
CIFR, IREN, BTDR). Hand-coded per-name `{state: weight}` exposure vector × an
electricity-intensity constant (fraction of cash cost).

`signal_i(q) = −intensity_i · %Δ(weighted power price)_i(q)`, cross-sectionally z-scored.
Power price: EIA state industrial retail (primary) and, as a measurement check, ERCOT/PJM
DA hub for the TX / PJM-state weights.

**Pre-registered 6-gate tree** (from the proposal): G1 predict fundamentals (|t|≥2) → G2
predict fwd returns (|t|≥2) → G3 survive FF5+MOM+sector neutralisation (|t|≥2) → G4
breadth ≥ 15 → G5 sign-stable across regimes → G6 costs.

| gate | state-retail price | wholesale-hub price | pass? |
|---|---|---|---|
| G1 Δgross-margin, contemp. | IC −0.04 (t −0.6) | IC −0.01 (t −0.2) | **FAIL** |
| G1 Δgross-margin, next-q | IC +0.03 (t +0.4) | IC −0.07 (t −0.9) | **FAIL** |
| G1 Δop-margin, contemp. | IC +0.13 (t +2.2) | IC +0.17 (t +2.6) | passes, but **contemporaneous only** |
| G1 Δop-margin, next-q | IC −0.12 (t −1.9) | IC −0.12 (t −1.7) | **FAIL** (sign flips, not predictive) |
| G2 fwd-1m return | IC +0.02 (t +0.5) | IC +0.02 (t +0.6) | **FAIL** |
| G2 fwd-3m return | IC +0.01 (t +0.3) | IC +0.03 (t +0.9) | **FAIL** |
| G2 fwd-3m, 2022–26 only | IC +0.12 (t +1.96) | IC +0.04 (t +0.7) | **FAIL** (proxy-specific; not robust) |
| G3 neutralised panel OLS | coef t +1.7, p 0.09 | coef t +1.0, p 0.34 | **FAIL** |
| G4 breadth | 18 names | 18 names | PASS |
| G5 regime sign-stability | fwd-3m: −0.02 → +0.12 | — | **FAIL** |

**Read:** the only thing that survives is a **contemporaneous** operating-margin
association (cheaper power quarter ⇒ higher op-margin that same quarter) — an accounting
identity the market already has when the 10-Q prints, and which does **not** appear in
gross margin (where a real power-cost effect would land first) and does **not** predict
forward returns after factor/sector controls. The 2022–26 fwd-3m return blip
(t≈1.96 on the retail-price proxy) collapses to t≈0.7 on the wholesale proxy — it was
proxy noise, not a regime effect. **Probe B FAIL.**

**Confirmatory test on the t=1.96 cell** (`probe_b_focus.py`, pre-specified: 2022–26 only,
drop the 7 BTC miners, sector-neutralise, Newey–West t): fwd-3m rank-IC NW-t **+0.75**
(retail) / +0.68 (wholesale); **sector-neutral** IC NW-t **−2.12** (retail) / −0.55 —
i.e. once the between-sector variation is removed the signal points the *wrong* way;
FF5+MOM+sector panel OLS 2022–26 signal t **−0.39** / +0.23; tertile L/S Sharpe +0.42 /
−0.17. The raw +0.118 was the signal sorting "crypto miners vs aluminium vs gas" in the
AI window — a sector bet, not stock selection. **Dead.**

---

## Why this settles the direction (against the two external proposals)

- The "Electricity Intelligence for Equity Alpha" proposal's own research tree kills it:
  **gate 1 (predict fundamentals) fails** — a hand-mapped physical power-cost shock leaves
  no fingerprint on gross margin over 15 names × 37 quarters, in either price proxy; and
  **gate 3 (factor neutralisation) fails** (|t| < 2).
- The merchant-generator earnings path (Probe A) has no foundation: hedging + non-gen
  segments break the realised-price → reported-earnings link even though the top line
  tracks.
- "Move one layer closer to the market" does not rescue the equity signal, because the
  binding constraint was never *distance from the market* — it is that the equity payoff
  is a multiple/narrative story (merchant gens) or a multi-input-cost margin story
  (industrials) that a power-market model cannot see.

## Honest limitations (not "try harder" — genuine caveats on the null)

- Exposure vectors / intensities are hand-coded and approximate. A facility-level
  point-in-time map could sharpen the signal — but the **gross-margin null** is hard to
  explain with measurement noise alone.
- No point-in-time consensus feed, so the "vs. what equities imply" leg was proxied by
  naive expectation models rather than I/B/E/S. G1 failing makes the downstream
  mispricing test moot regardless.
- 5 of 18 names are BTC-driven miners; their equity noise is large. The G1 gross-margin
  panel is already dominated by the industrial core (miners mostly lack the gross-margin
  history), and that core is where the null sits.

---

## Probe C — input-cost hedge screen (copper / electrical steel / labour) — 2026-09-07

**Question:** the supplier book's real cost inputs are copper, grain-oriented electrical
steel (GOES) and skilled electrical labour. Which stocks, *any industry*, have high beta
to those costs and could hedge the book's input-cost risk?

**Factors (monthly Δlog):** copper `HG=F`; electrical steel `CLF` (sole US GOES producer,
proxy); labour = FRED construction AHE `CES2000000003` + a tradable staffing basket
RHI/MAN/KFY/ASGN. Universe: 184 names across all 11 sectors. `screen|plain` book return
(no sleeve), 109 months 2017-07 → 2026-07.

### Gate 0 — does the book even have input-cost risk? **NO.**

| book excess return regressed on | beta | t | ΔR² vs market-only |
|---|---|---|---|
| MktRF + copper | copper **+0.02** | +0.2 | 0.00 |
| MktRF + esteel (CLF) | esteel **+0.02** | +0.6 | 0.00 |
| MktRF + staffing basket | staff **−0.03** | −0.5 | 0.00 |
| MktRF + construction AHE | AHE **+1.3** | +0.9 | 0.00 |
| MktRF + all three | all ≈ 0 | all \|t\|<1 | 0.00 |

Market beta alone explains R² = 0.58; copper/steel/labour add **nothing**. These
companies pass input costs through (cable makers index copper, equipment makers reprice
on lag, contractors escalate) — in a shortage they've had *pricing power*.

### Tail check — and it's actually **backwards**

| 12 biggest shock months | mean book excess ret | vs all-month | book down? |
|---|---|---|---|
| copper spike | **+3.6%** | +1.5% | 33% |
| e-steel spike | **+5.0%** | +1.5% | 25% |
| staffing-cost spike | **+5.8%** | +1.5% | 25% |

The book **outperforms** when input costs spike — copper/steel/labour inflate exactly
when the industrial economy is hot and grid-capex demand is strong, the same conditions
that drive the book.

### Screen (for completeness)

Top input-cost-beta names are mechanical: `CLF` (β_esteel 1.0, t 50 — it *is* the proxy),
copper miners HBM/FCX/ERO/TECK/SCCO (β_copper ~1.0–1.7), aluminium CENX/AA, staffing
RHI/MAN (β_staff ~1.0–1.3 — they *are* the basket), steel MT/NUE/STLD. Most also carry
**positive book beta 0.3–0.9** (CENX 0.89, ATI 0.88, SCCO 0.37, STLD 0.44) → going long
them adds correlated cyclical risk, not a hedge. The near-zero-book-beta names
(RIO −0.00, BHP −0.07, VALE −0.07, TECK −0.02, GOLD 0.00, ERO 0.03) are diversified /
gold miners with mediocre standalone Sharpe (0.4–0.7). **"All industries" surfaced nothing
non-obvious** — no consumer / tech / health / financial name has meaningful copper, steel
or labour beta.

### Verdict — **no hedge warranted**

There is no input-cost risk in the book to hedge (Gate 0), and the raw exposure is if
anything favourable (tail check). A copper/steel/labour-beneficiary position against the
book would strip out exposure that helps. Betas CSV: `research/electricity_earnings_exposure/_cache/probe_c_betas.csv`.
