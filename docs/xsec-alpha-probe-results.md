# Cross-sectional alpha probe — can we pick names inside the electrification universe?

**Date:** 2026-09-24 · **Branch:** `research/xsec-alpha-probe` · **Code:** `research/xsec_alpha_probe/`

## Question

The shipped screened book earns the theme, not selection (GRID-hedged selection Sharpe **0.09**).
Does any signal rank next-month / next-quarter returns *inside* the mechanical eligible universe
(SIC ELEC/POWER + fuel-cycle lexicon, $10m/day liquidity, ~146 names/month, 2019-01 → 2026-07)?

## Pre-registered bar (set before results)

Monthly rank-IC, Newey-West t (lag 1 for 1m, lag 2 for 3m). **PASS** = |t| ≥ 1.5 pooled AND in
both 2019-22 and 2023-26, same sign. **CLOSE** = pooled clears, halves agree in sign, one half short.
Every signal winsorized 5/95 cross-sectionally then z-scored. Books: top quintile minus
equal-weight universe, monthly, 10bp one-way, weighted equal / inverse-63d-vol / inverse-63d
downside-semivol. 14 signals × 2 horizons = 28 tests → expect ~1–2 false CLOSEs by chance.

## Verdict: **NO SIGNAL PASSES AS PRE-REGISTERED.** Two leads, one post-hoc composite worth building on.

| signal | 1m t (full / 19-22 / 23-26) | 3m t (full / 19-22 / 23-26) | verdict | top-quintile book SR (full / 19-22 / 23-26) |
|---|---|---|---|---|
| **capex_light** −(TTM capex/revenue) | 1.60 / 0.84 / 1.51 | **2.52 / 1.37 / 2.41** | CLOSE | 0.70 / 0.24 / 1.32 |
| rev_accel (Δ TTM rev YoY growth) | 1.28 / 0.93 / 0.95 | 0.07 / −0.38 / 1.03 | FAIL | **0.69 / 0.64 / 0.75** |
| rev_growth | −0.78 / −1.44 / 0.48 | −1.73 / −1.85 / −0.43 | CLOSE (inverted) | 0.23 / 0.21 / 0.25 |
| rev_sue | 0.61 / 0.42 / 0.51 | −0.14 | FAIL | 0.32 |
| ni_sue (PEAD) | 0.21 / −1.14 / 2.18 | 0.33 / −1.41 / 2.82 | FAIL (regime flip) | 0.09 |
| capex_restraint (asset growth) | −0.36 / 1.06 / −2.52 | −0.39 / 0.95 / −2.92 | FAIL (regime flip) | 0.26 |
| mom_12_1 | 0.64 / −0.31 / 1.24 | 1.05 | FAIL | 0.57 |
| reversal_1m | −1.30 / 0.24 / −2.31 | 0.54 | FAIL | −0.21 |
| beta_energy (PJM+ERCOT hub LMP) | 0.76 / −0.43 / 1.64 | 0.52 | FAIL | 0.52 / −0.15 / 0.89 |
| beta_congestion (PJM zonal \|cong\|) | −0.36 | −0.48 | FAIL | −0.02 |
| beta_scarcity (ERCOT p99 hourly) | 2.01 / 0.02 / 2.97 | 1.66 / −0.13 / 3.10 | CLOSE (2023-26 only) | 0.44 / −0.13 / 0.76 |
| bxs_energy / _congestion / _scarcity (β × 3m shock) | all \|t\| < 1 | all \|t\| < 1.4 | FAIL | ≤ 0.40 |

**Energy and congestion prices do not select names.** Exposure-times-recent-shock (the
"who benefits from the move" lead-lag) is flat for all three factors. Beta levels to energy and
ERCOT scarcity work only in 2023-26 (t ≈ 0 in 2019-22): being long power-exposed names in the
AI era is the theme again, not selection. MISO/CAISO caches are empty before 2023, so factors
use PJM + ERCOT only.

**Vol / semivol weighting did not help.** Inverse-vol and inverse-semivol top-quintile books
lowered full-sample Sharpe vs equal weight for 11 of 14 signals (semivol: 10 of 14) (they tilt to low-vol names,
which lagged in this universe); semivol and invvol differ by ≤ 0.09 Sharpe everywhere.

## Follow-ups on the leads

**capex_light is not just utilities-vs-suppliers.** Group-neutral (z within SIC ELEC / POWER)
3m t = 3.49 / 1.63 / 3.64 → PASS; ELEC group alone 3.95 / 1.78 / 4.31 → PASS. **But the
information is in the bottom** (capex-heavy names underperform), not the top:

| case | top − univ SR | univ − bottom SR | Q5−Q1 SR (full / 19-22 / 23-26) | Q5−Q1 ann | Q5−Q1 maxDD |
|---|---|---|---|---|---|
| capex_light, all | 0.70 | 0.99 | 0.90 / 0.41 / 1.59 | 21% | −28% |
| capex_light, ELEC only | 0.25 | 0.75 | 0.62 / 0.26 / 1.02 | 18% | −27% |
| rev_accel, ELEC only | 0.59 | 0.22 | 0.47 / 0.14 / 1.09 | 14% | −60% |
| **composite, all (post-hoc)** | **1.07** | 0.64 | 0.97 / 0.50 / 1.52 | 19% | −22% |
| composite, ELEC only (post-hoc) | 0.76 | 0.39 | 0.61 / 0.12 / 1.41 | 18% | −56% |

The **composite** = mean z of capex_light and rev_accel, chosen *after* seeing results (2 of 14),
so it cannot earn a PASS. It is the best long-side selector found in this research line:
top-quintile-minus-universe Sharpe **1.07 (0.80 / 1.33)**, 12%/yr, **GRID-hedged 0.42** vs the
shipped screen's 0.09. Its 1m IC is CLOSE (1.92 / 1.58 / 1.14); ELEC-only 1m is PASS (3.07 / 1.65 / 2.74).

## Caveats

- **2023-26 carries everything.** Every lead is roughly twice as strong in 2023-26 as in 2019-22.
- **Survivorship.** Delisted names have no free price history; bottom-bucket losses are understated.
- **capex_light overlaps the shipped screen**, which already selects the capex-light end; the
  incremental value *inside* the 35-name book is untested.
- **Control universe:** price/power signals behave similarly on 77 non-power industrials
  (mom_12_1 t 1.51, 2023-26 only); fundamentals untestable there — the cached broad facts
  have no annual facts, so Q4 cannot be derived.

## Next steps (not run)

1. Out-of-sample discipline for the composite: freeze it now, re-test on live months from 2026-09.
2. Rank the 35 screened names by the composite (tilt, not exclusion) and measure vs equal weight.
3. Refetch annual facts for the control universe to test whether capex_light/rev_accel are power-specific.

Reproduce: `python -m research.xsec_alpha_probe.run`, `... run --control`, `... followup`, `... followup --legs`.

## Web triage — new firm-level variables (2026-09-24, SCOPED, NOT RUN)

Filter: only variables that **differ by company** can produce GRID-hedged alpha. Aggregate ISO /
forward / queue / policy / utility-demand series are market-wide; they already failed as timing
signals and (above) as exposure×shock.

| # | variable | firm-level? | free point-in-time history? | evidence | call |
|---|---|---|---|---|---|
| 1 | Customer momentum (supplier lags customer) | yes, via supplier→customer map | prices free; map is the work | Cohen-Frazzini ~1.55%/mo t 5.4; post-discovery (2005-18) ~0.6%/mo, VW insignificant ([arXiv 2301.11394](https://arxiv.org/html/2301.11394)) | build — but see citation scan |
| 2 | Analyst estimate revisions | yes | paid ([Zacks ZEEH](https://data.nasdaq.com/databases/ZEEH), [FMP](https://site.financialmodelingprep.com/datasets/analyst-estimates-targets)) | strongest standard anomaly not yet tried here | if paid feed approved |
| 3 | Form 4 insider open-market buys | yes | free (SEC quarterly archives) | CMP 2012 ~82bp/mo opportunistic; mixed 2020-25 ([study](https://equibles.com/research/what-happens-after-an-insider-buys-evidence-from-47-458-open-market-purchases)) | cheap test; sparse |
| 4 | EIA-860M planned generators → makers | by technology | free ([EIA](https://www.eia.gov/todayinenergy/detail.php?id=36513)) | **already used** — Thread 6/8, `docs/interconnection-queue-signal-results.md` | not new |
| 5 | Federal awards by recipient | yes | free ([USAspending API](https://api.usaspending.gov/)) | awards mostly to utilities | low |
| 6 | Large-load queue by utility | utility/region | no vintages ([interconnection.fyi](https://www.interconnection.fyi/?status=All&type=Load), [EEI](https://www.eei.org/-/media/Project/EEI/Documents/Issues%20and%20Policy/List%20of%20Large%20Customer%20Projects%20and%20Tariffs)) | queue-velocity already NOT TESTABLE | start snapshotting |
| 7 | Power forwards | no | paid; [EIA](https://www.eia.gov/electricity/wholesalemarkets/data.php?rto=pjm) is spot | market-wide | skip |
| 8 | FERC Form 1 utility capex | utilities only | free ([PUDL](https://docs.catalyst.coop/pudl/en/stable/data_sources/ferc1.html)) | Proposal B failed | skip |

## Citation scan — what the papers that build on Cohen-Frazzini use instead (2026-09-24)

**Headline: customer links are a *dominated* link type.** Ali & Hirshleifer (JFE 2020,
[SSRN](https://www.ssrn.com/abstract=3015582)) show that firms linked by **shared analyst coverage**
give a connected-firm momentum factor of **1.68%/mo (t 9.67)**, and that industry, geographic,
customer, customer/supplier-industry, conglomerate and technology momentum all have
**insignificant or negative alpha** once it is controlled for. Testing customer momentum "as is"
therefore tests the weakest member of a family whose strongest member is known.

| link type | paper | data (free?) | result | fit for this universe |
|---|---|---|---|---|
| Customer → supplier | Cohen-Frazzini 2008 | Compustat segment customers (paid); free proxy = 10-K named-customer text | 1.55%/mo, decayed to ~0.6 | grid suppliers rarely name customers; utilities/hyperscalers are the customers |
| **Shared analyst coverage** | Ali-Hirshleifer 2020 | I/B/E/S (paid) | 1.68%/mo t 9.67; subsumes all others | best signal; needs a free proxy (below) |
| Upstream/downstream **industry** | Menzly-Ozbas 2010 ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=989080)) | BEA input-output tables (free) | significant; **weaker with more analyst coverage / institutional ownership** | our names are heavily covered → expect weak |
| **Text-based peers (TNIC)** | Hoberg-Phillips JFQA 2018 ([paper](https://faculty.tuck.dartmouth.edu/images/uploads/faculty/gordon-phillips/hoberg_phillips_TNICmomJFQA_FinV1.pdf)) | [Hoberg-Phillips library](http://hobergphillips.tuck.dartmouth.edu/) (free, 1989-2023, annual; also **vertical-relatedness** pairs) | less-visible peers → large momentum profits, stronger than own momentum | good — catches non-SIC peers; 2024-26 would need own 10-K embeddings |
| Technology (patents) | Lee-Sun-Wang-Zhang JFE 2019 ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3036241)) | USPTO/PatentsView (free) | 117bp/mo | log already found patents miss EPCs |
| **Geographic (co-HQ)** | Parsons-Sabbatucci-Titman RFS 2020 ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2780139)) | EDGAR business address (free) | 5-6%/yr; **unrelated to analyst coverage** — works for well-covered stocks | cheap; one of few that should survive in large caps |
| **News co-mention** | Scherbina-Schlusche ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5634686)) | news corpora; free proxy = GDELT (already wired in repo) | co-mentioned stocks cross-predict; info flows small→large and across industries | natural free proxy for analyst co-coverage |
| LLM / embedding 10-K networks | arXiv [2606.29290](https://arxiv.org/abs/2606.29290) (S&P 500, 2011-25), [2604.19476](https://arxiv.org/abs/2604.19476) | EDGAR 10-K text (free) + embeddings | Sharpe 0.86, FF5 alpha 7.3%/yr t 2.3; 0.74→0.82 | newest, covers the AI era; heavier build |
| Overnight/intraday split | Wang, JFQA 2025 ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4179413)) | daily open/close (free) | peer *intraday* returns carry the lagged signal; overnight reverses | refinement for any link type |

**Implications for design.** (1) Leaders should come from **outside** the 146-name universe
(hyperscalers, utilities, DC REITs, semis) — the spillover is from linked firms, not from within the
book. (2) Our names are large and heavily covered, which is where Menzly-Ozbas and Hoberg-Phillips
find the effect *weakest*; geographic and less-visible text peers are the exceptions. (3) Use peer
**intraday** returns (Wang 2025). (4) Keep the same bar and GRID hedge.

## Plan step 1 — tilt the `screen_auto` book by the frozen composite (2026-09-24)

Composite frozen as above (mean z of capex_light and rev_accel, z across the eligible universe
plus the book's members). Monthly `screen_auto` membership, 90 months, 9–30 names/month, 10bp.
**Bar fixed before running:** active return (TILT_0.5 − EW) Newey-West t ≥ 1.5 pooled, positive
in both halves, and TILT_0.5's GRID-hedged Sharpe beats EW's in both halves.
Hedge = r − β·GRID, β = full-sample OLS on monthly returns.

| book | SR full / 19-22 / 23-26 | ann | GRID-hedged SR full / 19-22 / 23-26 | active ann | active t full / 19-22 / 23-26 |
|---|---|---|---|---|---|
| EW (baseline) | 1.05 / 0.78 / 1.36 | 29% | 0.44 / −0.23 / 1.01 | — | — |
| **TILT_0.5** (primary) w ∝ 1+0.5z | 1.08 / 0.81 / 1.39 | 30% | 0.51 / −0.16 / 1.06 | +1% | **1.63** / 1.13 / 1.17 |
| TILT_1 w ∝ 1+z | 1.11 / 0.84 / 1.40 | 30% | 0.56 / −0.11 / 1.08 | +1% | 1.58 / 1.08 / 1.16 |
| **TOP_HALF** (EW, top half by composite) | 1.23 / 0.92 / 1.61 | 35% | **0.81 / 0.22 / 1.34** | **+6%** | **2.33** / 1.29 / 2.31 |

**Verdict: PASS (narrow) on the primary tilt** — +1%/yr, t 1.63, positive in both halves, hedged
Sharpe better in both halves. The gentle tilt barely moves weights; the **concentrated TOP_HALF
variant is the real finding**: +6%/yr over equal weight, t 2.33, and the only book whose
GRID-hedged Sharpe is positive in 2019-22. TOP_HALF was a pre-registered secondary variant, not the
primary, and each half's t is below 1.5 on its own (2019-22 t 1.29).

**Caveats.** (1) This EW baseline is plain monthly equal weight with no vol target or valuation
layer, hedged at β≈1.05; it is *not* the 0.09 figure in the book write-up (that was hedged at the
vol-targeted book's matched beta) — compare books within this table only. (2) Book membership uses
point-in-time `screen_auto` rows, but the screen itself was designed in 2026. (3) 2023-26 still
carries most of it.

**New benchmark for later steps:** TOP_HALF of the book by the frozen composite — GRID-hedged
Sharpe **0.81** (0.22 / 1.34), +6%/yr over the equal-weight book. Reproduce:
`python -m research.xsec_alpha_probe.tilt`.

### Step 1 add-on — Calmar and alpha-vs-beta label (2026-09-24)

**Label rule fixed before running:** ALPHA if the GRID-hedged return (Jensen alpha vs GRID) has
NW t ≥ 1.5 pooled and a positive mean in both halves; else BETA (the theme). Calmar = ann / |maxDD|.

| book | SR | Calmar | maxDD | alpha/yr vs GRID | hedged SR | hedged Calmar | alpha t full / 19-22 / 23-26 | label |
|---|---|---|---|---|---|---|---|---|
| EW | 1.05 | 0.89 | −32% | 6% | 0.44 | 0.23 | 1.17 / −0.44 / 1.94 | BETA |
| TILT_0.5 | 1.08 | 0.91 | −33% | 7% | 0.51 | 0.27 | 1.36 / −0.31 / 2.02 | BETA |
| TILT_1 | 1.11 | 0.92 | −33% | 7% | 0.56 | 0.30 | 1.48 / −0.21 / 2.06 | BETA |
| **TOP_HALF** | 1.23 | 0.89 | −39% | **12%** | **0.81** | 0.41 | **2.07** / 0.39 / 2.53 | **ALPHA** |

TOP_HALF is the only book that earns the ALPHA label, but its 2019-22 alpha is only ~3%/yr
(t 0.39): the label rests on 2023-26. Unhedged, concentration raises return but not Calmar (deeper
drawdown, −39%). The tilts are beta: their alpha is negative in 2019-22.

**Live out-of-sample log started:** `python -m research.xsec_alpha_probe.live_log` appends each
month's frozen composite z and TOP_HALF flag for the book to `research/xsec_alpha_probe/live/composite_scores.csv`
(first month 2026-08-31, logged 2026-09-24). Never rewrite past rows.

## Plan step 2 — how to hedge the TOP_HALF book (2026-09-24)

Three hedges on the TOP_HALF book: GRID short at matched beta; short the bottom quintile of the
eligible universe by composite (the capex-heavy avoid list) at the notional matching GRID beta
(1.50×), paying 1%/yr borrow + 10bp trading; 50/50 of the two.

| hedge | SR full / 19-22 / 23-26 | ann | maxDD | Calmar full / 19-22 / 23-26 | alpha t | label |
|---|---|---|---|---|---|---|
| GRID short (matched beta) | 0.81 / 0.22 / 1.34 | 12% | −29% | 0.41 / 0.10 / 1.17 | 2.07 | ALPHA |
| 1.50× capex-heavy quintile short | 0.71 / 0.12 / 1.55 | 18% | **−41%** | 0.44 / 0.08 / 2.07 | 1.93 | ALPHA |
| **50/50** | **0.87** / 0.18 / 1.67 | 15% | −30% | **0.51** / 0.10 / 1.93 | **2.30** | ALPHA |

**Verdict: GRID short stays the recommended hedge; the single-name short is NOT ADOPTED.** The
capex-heavy short earns more in 2023-26 but deepens the drawdown to −41%, is weaker in 2019-22, needs
1.5× gross short in names that were un-borrowable in the earlier short-leg work, and the
50/50 blend's gain (0.81 → 0.87 SR, Calmar 0.41 → 0.51) is within noise and all 2023-26. All three
are labelled ALPHA, but every one is ~zero in 2019-22.

## Plan step 3a — text-similar peer momentum (2026-09-24)

**Build.** Hoberg-Phillips TNIC is keyed by Compustat gvkey (paid mapping) and ends 2023, so the
same measure was rebuilt from free EDGAR text: 10-K Item 1 (Business) for the eligible universe
ever + current S&P 500 (655 firms, 5,375 filings 2017-2026, `research/xsec_alpha_probe/tenk.py`).
Each month-end: TF-IDF (stop words out, words in >25% of docs out, sublinear tf) over every firm's
latest Item 1 filed ≤ t and within 18 months; follower → its 10 highest-cosine leaders. Pool
475-562 firms/month. Two extraction problems found and fixed before running: (1) 640 filings /
109 firms where Item 1 could not be located fell back to the whole 10-K and matched each other on
legal boilerplate (WESCO's "peers" were MCD, O, CAH) → excluded; (2) 708 filings / 159 firms where a
cross-reference inside Risk Factors was taken as Item 1 (MYRG) → rule changed to the first real
section, refetched.

**Pre-registered signal** (`peers.py`): cosine-weighted mean of the leaders' month-t returns,
predicting month t+1. Primary `peer_day_out` = leaders outside the month's eligible universe,
daytime (open→close) returns; secondaries `peer_day_all` (all leaders), `peer_cc_out` (close-to-close).

| signal | 1m t full / 19-22 / 23-26 | 3m t | verdict | top-quintile − universe SR (full / 19-22 / 23-26) | GRID-hedged |
|---|---|---|---|---|---|
| **peer_day_out** (primary) | 1.62 / 0.53 / 1.90 | 0.79 | CLOSE (1m) / FAIL (3m) | 0.31 / 0.12 / 0.46 | 0.18 |
| peer_day_all | 2.17 / **−0.18** / 3.87 | 0.67 | FAIL (sign flip) | 0.64 / −0.18 / 1.35 | 0.42 |
| peer_cc_out | 1.10 / 0.56 / 1.04 | 0.78 | FAIL | 0.51 / 0.66 / 0.40 | 0.22 |

**Against the benchmark (TOP_HALF of the book, GRID-hedged 0.81 / Calmar 0.41, ALPHA):**

| book | SR | Calmar | maxDD | alpha/yr | hedged SR full / 19-22 / 23-26 | hedged Calmar | alpha t | label |
|---|---|---|---|---|---|---|---|---|
| TOP_HALF composite (benchmark) | 1.23 | 0.89 | −39% | 12% | 0.81 / 0.22 / 1.34 | 0.41 | 2.07 | ALPHA |
| TOP_HALF by peer_day_out alone | 1.02 | **1.03** | **−28%** | 6% | 0.40 / −0.18 / 0.95 | 0.21 | 1.13 | BETA |
| TOP_HALF by composite + peer | 1.11 | 0.95 | −35% | 8% | 0.58 / −0.08 / 1.23 | 0.29 | 1.63 | BETA |

**Verdict: FAIL / NOT ADOPTED.** The primary signal is CLOSE at 1m only (2019-22 t 0.53) and fails
at 3m. Blending it into the composite *lowers* the benchmark's hedged Sharpe (0.81 → 0.58) and
turns it from ALPHA to BETA. The all-leaders version is the theme again: t 3.87 in 2023-26, negative
in 2019-22. The one thing it does is lower drawdown unhedged (Calmar 1.03, maxDD −28%) — a beta
property, not selection. Consistent with the literature: spillovers are weakest in heavily covered
large caps. Caveats: S&P 500 pool is today's list; ~12% of filings excluded; residual extraction
noise (~112 filings still read like Risk Factors).
