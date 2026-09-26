# capex_intensity_strategy

A rules-based long/short equity book over the US power / electrification complex.
**Every input is point-in-time.** Nothing is hand-picked: the universe comes from SEC
filer industry codes plus filer-authored language, and the score comes from filings
keyed on the date each was *filed*.

Proposal and evidence: `docs/capex-intensity-proposal.md`.
Discovery run: `docs/capex-intensity-rule-results.md`.

## The idea in one line

Buy the companies that **sell into** the power buildout; short the companies that
**are** the buildout.

Equipment makers, electrical contractors and fuel-cycle firms spend little of their own
money on plant. Utilities, pipelines and miners spend enormous amounts. The first group
captures the capital cycle without funding it. Capital spending divided by revenue
separates them cleanly, and it is the only metric tested that separated them **both** in
the 2019-2022 fit window and afterwards.

## Run

    python -m capex_intensity_strategy                      # cached data
    python -m capex_intensity_strategy --short-weight 0     # long-only
    python -m capex_intensity_strategy --refresh            # re-fetch EDGAR + prices

Writes `summary.csv`, `daily_returns.csv`, `weights.csv`, `diagnostics.csv` to
`--output` (default `./output_capex_intensity`). The first `--refresh` run takes
roughly 20-40 minutes (one EDGAR companyfacts call per filer, rate-limited).

## The rule

| step | what |
|---|---|
| **Universe A** (structural) | every US 10-K filer whose SEC **SIC code** is in `data/sic_codes.csv` -- electrical equipment, switchgear, power-line construction, utilities, generation, gas transmission |
| **Universe B** (language) | any filer whose prior-year 10-K/40-F/20-F matches the **fuel-cycle lexicon** ("uranium concentrates", "yellowcake", "enriched uranium", ...) |
| **Liquidity** | trailing 1-yr median dollar volume > $10M, re-checked monthly |
| **Score** | point-in-time **TTM capital expenditure / TTM revenue** from EDGAR `companyfacts` |
| **Book** | long the lightest 25%, short the heaviest 25% at **0.5x**, monthly, 20% vol target, 10bp costs |

Universe B exists because SIC alone cannot find the nuclear fuel cycle: CCJ and UEC file
under **1090** (Misc Metal Ores) and LEU/UUUU under **1400** (Nonmetallic Minerals),
alongside hundreds of gold miners. What separates them is their own filing language.

`3674` (semiconductors), `1311` (crude oil & gas) and `1040` (metal mining) are
deliberately **excluded** -- they are off-thesis and would drag in AMD, AVGO and several
hundred gold juniors.

## Point-in-time discipline

- **Fundamentals** use each XBRL fact's `filed` date, never its period end. A quarter is
  invisible until the day it was filed. Restatements resolve to the most recently filed
  value as of the observation date, and the year-to-date differencing below is redone at
  every observation date so a quarter is always built from the vintages then available.
- **Quarters are 60-110 day spans.** Cash-flow items are tagged year-to-date (a 10-Q
  reports 3, 6 and 9 months from the fiscal-year start; the 10-K reports 12), so only Q1
  is ever a standalone quarter. `fundamentals.derive_quarters` differences consecutive
  rungs of that ladder -- Q4 = FY - 9M -- and each derived quarter inherits the later of
  the two filing dates.
- **A name needs four filed quarters** before it is scored -- no annualising a partial year --
  and those four must span 330-400 days, so four same-quarter facts from four different
  years are not mistaken for a year.
- **Stale series are dropped.** The newest quarter must be within 270 days of the
  observation date, and the revenue and capex windows must end on the same quarter.
- **The lexicon is queried per universe-year** over filings published strictly *before*
  that year begins.
- **SIC codes** are assigned by the filer at filing time.

## Layout

| file | role |
|---|---|
| `config.py` | every frozen parameter; never revised from results |
| `edgar.py` | thin cached network wrappers (SIC populations, full-text search, companyfacts). `filer_mention_dates` is the CIK-filtered form of `full_text_ciks` -- one request per (filer, phrase) instead of paging an unbounded result set -- used by `electrification_strategy`'s mechanical `screen_auto` construction. |
| `fundamentals.py` | `filed`-date-aware TTM panel (item-agnostic: carries `net_income` too) -> the capex-intensity score |
| `universe.py` | SIC + lexicon + liquidity -> a monthly eligibility mask |
| `backtest.py` | target weights, daily returns, diagnostics, metrics |
| `pipeline.py` | end-to-end build, all intermediates returned for inspection |
| `data/sic_codes.csv` | the SIC table (the only seed data) |

## Fact coverage (fixed 2026-09-23)

Three defects in the fact layer were repaired together; every number above post-dates them.

| defect | effect | fix |
|---|---|---|
| the tag loop stopped at the first tag *present in the JSON*, even when the span filter removed all of its facts | ETN, EMR, PWR had capex but no revenue; HUBB, GEV, CARR, JCI, EME, WCC had revenue but no capex -- 13 of the 28 screened-electrification names were unscoreable for the whole history | every tag is parsed; `ttm_panel` keeps, per filer and per date, the single best-covered non-stale tag |
| `PaymentsToAcquireProductiveAssets` was not in the capex tag list | the only capex tag HUBB, GEV, CARR, JCI, EME and WCC use, and where AEP reports construction expenditure | added |
| a 60-200 day span was called "a quarter", and no guard required the four quarters to be contiguous or current | 6-month YTD facts counted as quarters; AEP's capex facts stop in 2021 but were divided into 2026 revenue, scoring it 2.4% and putting two regulated utilities in the **long** leg | quarters are 60-110 days, derived from the YTD ladder, with span/staleness/alignment guards |

Scoreable names went from 116 to 193.

All logic is pure and network-free; `edgar.py` is the only module that touches the wire,
so the test suite runs offline against synthetic filings.

## Results (RF = 4%, 10bp costs, 20% vol target)

| window | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| 2019-2022 *(fit)* | 4.7% | 21.2% | 0.13 | -34.1% | 0.14 |
| **2023-2026 *(oos)*** | 32.0% | 21.3% | **1.22** | -22.8% | **1.40** |
| **2025-2026 *(oos)*** | 21.1% | 21.9% | **0.80** | -22.8% | **0.92** |
| full 2019-2026 | 17.0% | 21.3% | 0.66 | -34.1% | 0.50 |

Restated 2026-09-23 on the repaired fact panel (see *Fact coverage* below); the earlier
table was computed on a panel that could score only ~116 names and mis-scored several
utilities. The out-of-sample window is essentially unchanged; the fit window is weaker.
~27 long / 27 short, turnover 1.0x/yr. SPY over the same windows: 0.48 / 1.14 / 0.82 / 0.72.
2023 onward is out-of-sample for the discovery, which used only 2019-2022 to choose the metric.

## Caveats

- **Weak before 2023** (fit-window Sharpe 0.41). This is a capex-cycle bet and it pays
  while the cycle runs. Size accordingly and read the kill condition in the proposal.
- **Metric selection used ex-post labels.** Eleven metrics were screened against a
  hand-classified name list; capex intensity won by t = -15 versus -7.2 for the next best.
  Once stated the rule needs no labels -- but *which* metric was chosen carries selection
  risk, and only forward out-of-sample resolves it.
- **Asset-light overlaps known quality factors.** The claim is a correctly specified
  factor for this universe, not a new anomaly.
- **Survivorship.** Prices come from yfinance, which serves no delisted names, so the
  backtest runs on survivors only. Long legs are flattered and short legs understated.
  This affects every backtest in this repo equally. Fixing it needs CRSP or equivalent.
