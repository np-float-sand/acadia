# PJM large-load-adjustment vintages — retrieval + MW-revision test (2026-09-03)

**Ask:** get the historical PJM "Large Load Adjustment Requests" so we can test whether
**MW revisions within the data-center category** would help time the buildout-basket long.

## What was retrieved

The consolidated machine-readable *request-level* spreadsheet is a 2025 innovation (only the
2026-LF cycle exists: `grid_resilience/data/cache/pjm_large_load_adjustment_2026.parquet`, from the
9/16/2025 LAS deck). For prior cycles the accepted numbers live in the annual **Load Forecast
Report, Table B-9** ("Adjustments Above Embedded to Summer Peak Load, MW, by zone & forecast year").
Pulled all six vintages:

| Vintage | Source | Format |
|---|---|---|
| 2021 LF | `https://www.pjm.com/-/media/DotCom/library/reports-notices/load-forecast/2021-load-report.pdf` | PDF Table B-9 |
| 2022 LF | `.../2022-load-report.pdf` | PDF Table B-9 |
| 2023 LF | `.../2023-load-report.pdf` | PDF Table B-9 |
| 2024 LF | `.../2024-load-report.pdf` | PDF Table B-9 |
| 2025 LF | `https://www.pjm.com/-/media/DotCom/planning/res-adeq/load-forecast/2025-load-report-tables.xlsx` | xlsx `Table B9` / `Table B9b` |
| 2026 LF | `.../2026-load-report-tables.xlsx` | xlsx `Table B9` / `Table B9b` |

Parsed to `grid_resilience/data/seed/pjm_large_load_b9_vintages.csv`
(`vintage, zone, target_year, mw_adj`; 2,422 rows; zones incl. `PJM RTO` total).
2026 B-9 carries a PJM note: "updated 2/6/2026 to correct for an error calculating embedded [load]".

## PJM RTO large-load adjustment (data-center-driven), fixed target year 2030, by vintage

| Vintage (pub ~Jan) | RTO adj @2030 (MW) | Δ vs prior vintage | near-term Δ (@2028) |
|---|---|---|---|
| 2021 LF | -34 | — | — |
| 2022 LF | 4,834 | +4,868 | +4,428 |
| 2023 LF | 12,762 | +7,928 | +5,425 |
| 2024 LF | 16,480 | +3,718 | +2,777 |
| 2025 LF | 32,795 | +16,315 | +6,677 |
| 2026 LF | 33,707 | **+912** | **-3,078** |

Zone detail (2030 target, MW): DOM 150→4,600→11,586→11,831→11,228→10,135 (peaked 2024 LF, now easing);
AEP 0→0→248→3,544→9,990→7,918; COMED -222→-218→0→0→2,311→4,838 (turns on late, still rising);
APS ~flat 94→806→1,214→1,213; PS 0→0→0→295→999→1,194.

## Test: revision vs buildout-basket forward 12-month excess return (vs SPY)

| Vintage window | Δrev @2030 | basket − SPY, fwd 12m | basket − DER-sleeve, fwd 12m |
|---|---|---|---|
| 2021→2022 | +4,868 | +17.5% | +2.8% |
| 2022→2023 | +7,928 | +28.0% | +117.9% |
| 2023→2024 | +3,718 | +39.0% | +107.7% |
| 2024→2025 | +16,315 | +28.6% | +27.7% |
| 2025→2026 | +912 | +4.3% (7.5 mo, partial) | +44.6% |

- `corr(Δrev@2030, basket−SPY fwd)` = **0.42** (n=5).
- `corr(Δrev@2030, basket−DER fwd)` = **-0.16** — the DER short has no link to PJM MW.
- ERCOT monthly large-load queue (`ercot_large_load_monthly.csv`, 11 pts May-25→Mar-26,
  +41%): all one up-regime, no usable relationship to basket fwd-1m.

## Timing check on the one real test — the 2026 "stall"

The stall was **public at the LAS preliminary summary, 2025-11-24** ("Load Adjustment Requests
Summary for 2026 Load Forecast - Preliminary"), not at the January report. Basket path *after* that
date (equal-weight US-9 + 4 ADRs):

| signal date | → 2026-08-31 basket | SPY | excess |
|---|---|---|---|
| LAS preliminary 2025-11-24 | **+30.7%** | +15.7% | **+15.1%** |
| report publish 2026-01-15 | +15.7% | +11.4% | +4.3% |

Monthly cum index (base 2025-08-31 = 1.00): 1.10 (Nov) → 1.09 (Dec) → 1.26 (Jan) → 1.32 (Feb) →
1.51 (Apr) → **1.64 (Jun)** → 1.43 (Jul) → 1.36 (Aug). Nearly the entire 2026 rally
(≈ +55% off the late-Nov low to the June peak) came **after** the bearish signal. Only the
Jul-Aug-2026 give-back (-17% from the peak) goes the signal's way — 8 months late, inside the
basket's normal noise.

## Verdict

**Fails its one out-of-sample test.** Acting on the 2026 stall (de-risk the long, ~late Nov 2025)
would have **cost ~+15% of excess return / ~+30% absolute** over the next nine months. The equity
move ran on the live AI-capex narrative; PJM's *accepted* forecast is a lagging, once-a-year,
methodology-constrained number (the 2026 down-revision is partly PJM's new firm/non-firm filter,
not demand) and it de-risked straight into a rally.

- The 2022-2025 "it worked" rows are **in-sample and near-coincident** — every vintage revised up,
  the basket rose every year, both just track "AI capex kept going." n=5 annual points, one cycle.
  Relationship is binary at best (biggest revision, 2025 LF +16.3 GW, gave a middling return), and
  2023 is the only vintage with a clean lead.
- Useless for the short leg regardless: DER is a rates/own-cycle story, corr(Δrev, basket-DER) = -0.16.

Not a usable timing dial or de-risk trigger on this evidence. Revisit only with 2018-2020 vintages
added and each new PJM LAS / ERCOT TAC vintage archived going forward.

## Signal-gated basket — the numbers

The revision signal only changes exposure once in the available sample (STALL at the 2026-LF
preliminary, 2025-11-24; every other vintage = stay fully invested). Basket = EW US-9 + 4 ADRs,
daily rebal; de-risk = move the stalled fraction to cash at RF 4%.

| window | variant | total | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|---|
| **FULL 2021-01→2026-08** | plain | +451% | +35.6% | 1.05 | -39.8% |
| | stall→50% @prelim | +390% | +32.8% | 1.04 | **-39.8%** |
| | stall→0% @prelim | +324% | +29.4% | 0.97 | **-39.8%** |
| **2025-11-24→2026-08** | plain | +34% | — | 1.09 | -24.2% |
| | stall→50% @prelim | +19% | — | 1.09 | -12.5% |
| | stall→0% @prelim | +3% | — | — | 0.0% |

- **MaxDD is identical (-39.8%) in every full-sample variant** — the historical worst case was the
  2024 DeepSeek drawdown, long before the signal fired. The 2026 signal does nothing for it.
- Over the signal window, stall→50% halves the summer drawdown (-24%→-12.5%) but also halves the
  gain (+34%→+19%); Sharpe unchanged at 1.09 — pure de-lever, no information.
- Signal-gated underperforms plain on return/CAGR in every window and on Sharpe in the full sample.
  Same shape as the layer-2 congestion overlay and the capex-decel trigger: a slow annual
  demand-forecast number can't time a basket that trades on the live narrative.

## For reference — long-only basket vs long/short-vs-DER, same periods

Plain **dollar-neutral** long/short (long EW US-9 + 4 ADRs / short EW DER sleeve
ENPH,SEDG,CHPT,RUN,BLNK,STEM). NOT the beta-hedged + 20%-vol-target build the historic 2.23 Sharpe
came from — that figure needed the overlay and only held in 2023-26.

| window | variant | total | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|---|
| 2021-01 -> 2026-08 | plain basket | +451% | 35.6% | 1.05 | -40% |
| | signal stall->50% | +390% | 32.8% | 1.04 | -40% |
| | signal stall->0% | +324% | 29.4% | 0.97 | -40% |
| | long/short vs DER | +998% | 53.4% | **0.98** | **-57%** |
| 2023-01 -> 2026-08 | plain basket | +349% | 51.1% | 1.36 | -40% |
| | long/short vs DER | +716% | 77.9% | **1.23** | **-56%** |
| 2025-11-24 -> 2026-08 | plain basket | +34% | 47.0% | 1.09 | -24% |
| | signal stall->50% | +19% | -- | 1.09 | -12.5% |
| | signal stall->0% | +3% | -- | -- | 0% |
| | long/short vs DER | +54% | 76.0% | **1.35** | -29% |

- Plain DN long/short Sharpe **~1.2-1.35**, not 2.2 (the 2.23 needed beta-hedge + vol-target, 2023-26 only).
- L/S **MaxDD deeper** than long-only (-56% vs -40%) -- high-beta DER shorts fall with you in risk-off.
- **Full-period** L/S Sharpe (0.98) is *below* long-only (1.05) -- the short is pure drag in 2021-22.
- The MW-revision signal never interacts with the short; it only trims the long leg and costs return in every window.

**Status: closed / negative.** MW-revision timing signal does not work; recorded for the "things tried" ledger.
