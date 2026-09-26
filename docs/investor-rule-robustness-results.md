# Investor strategy — is the rule stack sensible or data-mined? (2026-09-25)

Research record only — **not** for the investor proposal. Tests 3–5 of the plan in the 2026-09-25
session: parameter plateaus, an ablation ladder, and a deflated Sharpe. All runs use
`research/electrification_strategy_probes/investor_book.py` (the proposal's construction) and charge
10bp on every trade plus borrow on the short. Reproduce with
`python research/electrification_strategy_probes/rule_robustness.py` (writes
`output_html/img/tophalf/rule_robustness.json`).

**The strategy tested (final):** Sharpe **1.28** full 2019-01→2026-08 / **0.55** 2019–22 / **1.85**
2023+, Calmar 1.66, max drawdown −11.9%, 19.8% a year.

## Verdict

**MIXED.** The settings are not fitted to spikes (plateaus PASS), and the strategy's edge over zero
survives any plausible multiple-testing haircut. But the ablation ladder shows that **none of the
added rules improved the 2019–22 Sharpe** — the pre-tight period, which is the closest thing we have to
out-of-sample — and that **the ranking (rule 6) made 2019–22 worse**. The rule stack's Sharpe gain is
concentrated in 2023+, the period the ranking was chosen on, plus the drawdown cut from rule 11 and the
short. The honest claim is "a sound core (screen + overlay) with a tight-grid tilt on top", not "every
rule is independently proven".

## 3. Parameter plateaus — PASS

One setting moved at a time around the chosen value (✱ = chosen).

| setting | values tested → Sharpe full (2019–22 / 2023+) | reading |
|---|---|---|
| No-trade band | 0.10: 1.29 · 0.15: 1.25 · **0.20✱: 1.28** · 0.25: 1.31 · 0.30: 1.22 | flat 1.22–1.31; chosen value is mid-plateau |
| Long hold buffer | none: 1.23 · 0.55: 1.26 · 0.60: 1.34 · **0.65✱: 1.28** · 0.70: 1.32 · 0.75: 1.28 | flat; any buffer beats none |
| Short size | 0.15: 1.24 · 0.20: 1.26 · **0.25✱: 1.28** · 0.30: 1.29 · 0.35: 1.30 | smooth and monotonic — bigger short, smaller drawdown (−14.8% → −9.2%), lower return |
| Short cutoff | 15%: 1.31 · **20%✱: 1.28** · 25%: 1.28 | flat |
| Short re-formation | monthly: 1.23 · bimonthly: 1.23 · **quarterly✱: 1.28** · semi-annual: 1.25 | flat; quarterly marginally best |
| Rule 11 real-yield threshold | −0.5: 1.20 · −0.25: 1.23 · **0✱: 1.28** · +0.25: 1.23 · +0.5: 1.30 | full-period flat, but **2019–22 Sharpe swings 0.37–0.55** — one episode |
| Rule 11 growth threshold | −2%: 1.27 · −1%: 1.27 · **0✱: 1.28** · +1%: 1.26 · +2%: 1.33 | flat |

No setting sits on a spike; in most rows a neighbouring value scores slightly higher than the one
chosen, which is what an un-mined choice looks like. The one fragile parameter is rule 11's
real-yield threshold, because the whole test rests on the 2019–22 episode.

## 4. Ablation ladder — MIXED

| step | Sharpe full | 2019–22 | 2023+ | Calmar | max DD | return |
|---|---|---|---|---|---|---|
| L0 rules 1–5, equal weight, no overlay, no short | 0.90 | **0.61** | 1.28 | 0.64 | −45.6% | 29.3% |
| L1 + risk overlay (rules 7–9, band, monthly sleeve) | 1.04 | 0.55 | 1.59 | 1.10 | −20.9% | 23.0% |
| L2 + ranking, top half (rule 6) | 1.02 | **0.47** | 1.68 | **0.83** | **−27.0%** | 22.4% |
| L3 + hold buffer 65% | 1.07 | 0.54 | 1.70 | 0.89 | −26.3% | 23.4% |
| L4 + short $0.25, quarterly (rule 10) | 1.13 | 0.49 | 1.85 | 1.31 | −15.9% | 20.8% |
| L5 + rule 11, real yield only | 1.27 | 0.55 | 1.85 | 1.67 | −11.9% | 20.0% |
| L6 + rule 11, both conditions (final) | 1.28 | 0.55 | 1.85 | 1.66 | −11.9% | 19.8% |

Rule by rule, against the standard "must help in both periods":

- **Risk overlay (L1): KEEPER.** Halves the drawdown (−46% → −21%), lifts Sharpe 0.90 → 1.04. Set in
  earlier work, before this proposal.
- **Ranking (L2): FAILS the both-periods test.** Helps 2023+ (1.59 → 1.68) but hurts 2019–22 (0.55 →
  0.47), full Sharpe (1.04 → 1.02), Calmar (1.10 → 0.83) and drawdown (−21% → −27%). It was chosen on
  2023+ behaviour; this is the rule most exposed to the mining charge.
- **Hold buffer (L3): PASS.** Helps both periods (and partly repairs the ranking's 2019–22 damage).
- **Short (L4): drawdown PASS, Sharpe mixed.** Cuts drawdown −26% → −16% and lifts 2023+ Sharpe; costs
  2019–22 Sharpe 0.54 → 0.49.
- **Rule 11 (L5/L6): helps where it acts, by construction.** Only 2019–22 changes (0.49 → 0.55,
  drawdown −16% → −12%); both-conditions ≈ real-yield-only.
- **Net:** the stack takes full-period Sharpe from 1.04 (L1) to 1.28 and drawdown from −21% to −12%, but
  **2019–22 Sharpe ends where L1 started (0.55)** and below the raw screen (0.61). Every Sharpe gain
  beyond the overlay is a 2023+ gain.

## 5. Deflated Sharpe — PASS vs zero; ~90% vs GRID; depends on the assumed spread of trials

Probability the strategy's true Sharpe exceeds a threshold, allowing for skew and fat tails
(probabilistic Sharpe ratio):

| threshold (annual Sharpe) | full period | since 2023 |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 0.50 | 0.985 | 0.994 |
| 0.82 (GRID's full-period Sharpe) | **0.90** | — |
| 0.93 (GRID's 2023+ Sharpe) | — | **0.96** |
| 1.00 | 0.78 | 0.95 |
| 1.25 | 0.54 | 0.87 |

The deflated Sharpe replaces the threshold with the best Sharpe luck alone would produce across N
trials. That depends on how much Sharpe varies from trial to trial — which we cannot observe for the
~30 abandoned research lines:

| spread of Sharpe across trials (annual sd) | luck's best of 20 / 100 / 300 | deflated Sharpe (prob. real) for N = 20 / 100 / 300 |
|---|---|---|
| 0.09 (measured across our 41 variants) | 0.17 / 0.22 / 0.25 | 0.999 / 0.998 / 0.998 |
| 0.20 | 0.38 / 0.51 / 0.58 | 0.994 / 0.984 / 0.974 |
| 0.30 | 0.57 / 0.76 / 0.87 | 0.976 / 0.926 / 0.874 |
| 0.50 (genuinely different strategies) | 0.95 / 1.27 / 1.45 | 0.821 / 0.518 / 0.322 |

The measured spread (0.09) understates the true one, because our 41 variants are small nudges of one
design. With ~100 genuinely different attempts in this research line and a spread of 0.3, the
probability the edge is real is about **93%**; at a spread of 0.5 it falls to about a coin flip. The
defensible statement: **positive skill is very likely; beating GRID by the full backtest margin is
not proven.**

## What this implies (not acted on)

1. The ranking is the weakest link. Candidates: drop it (keep L1 + buffer-less equal weight + short +
   rule 11), or keep it but describe it as a tight-grid tilt with a 2019–22 cost.
2. The 2010–18 holdout (plan step 7) is now the most valuable next test — it is the only way to tell
   whether the ranking and the short work outside the period they were chosen on.
3. Freeze the whole rule set with today's date and track it live.

## Addendum — is the ranking a worry, and what can a 2010–18 holdout test? (2026-09-25)

**Ranking (rule 6).** Its 2019–22 damage falls in the regime the thesis says it should not work in, and
rule 11 now keeps the strategy out of that regime, so the live cost is largely avoided. The residual
worry is that "it only works when the grid is tight" is also exactly the explanation an in-sample fit
to 2023+ would produce; the two cannot be told apart on this history. Judged acceptable for now;
revisit with the holdout and the live record.

**What a 2010–18 holdout can and cannot show.** 2010–18 was a *slack* grid — flat US electricity
demand, replacement-only utility capex, a 2015–16 industrial downturn. So it **cannot** test whether
the ranking works when the grid is tight (the only tight period is 2023+). It **can** test:
1. whether the full rule stack survives a slack grid (lags, but no blow-up);
2. whether rule 11 flags the slack spells — real yields were negative 2012–13, and supplier real
   revenue likely fell 2015–16 — and whether the ranking hurts only when rule 11 would be off (supports
   the thesis) or also while invested (a warning);
3. whether the ranking's slack-grid effect is consistently mildly negative (fits the thesis) or noisy.

A cleaner out-of-sample test of the tight-grid idea would be other capacity crunches where suppliers
were paid ahead of owners (US energy/pipeline builds 2005–08 and 2011–14; semiconductor equipment in
chip shortages) — a larger project, not started.

**Data needed for the holdout (checked 2026-09-25).** Cached prices and dollar volume start 2015;
the fundamental signals are computed only from 2018-06 (hard-coded date range); the published
membership starts 2015; the 10-K wording cache starts 2017; EDGAR XBRL facts go back to filings of
2009. So the holdout needs: price/volume history from ~2008, the signal and membership pipelines
re-run from ~2009, EDGAR full-text wording hits back to ~2006, and the power-market signals in the
short score (availability before ~2013 unverified). **Survivorship bias:** the candidate universe is
today's listed companies, so firms that failed or were acquired before 2019 are missing — this
flatters the long side and removes the short side's best (bankrupt) shorts.
