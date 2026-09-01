# Handoff — "Grid Buildout, GRID-hedged" long/short, and the full record of what was tried

**Date:** 2026-09-01
**Supersedes the search line of:** `docs/handoff_2026-08-31-two-layer-overlay.md` (that handoff asked
to brainstorm a layer-2 congestion signal — it was built in full and failed; see §3).

---

## TL;DR

- **Proposed trade:** long a concentrated basket of US + foreign **grid / data-center power buildout**
  equipment makers and electric-utility contractors, **short the GRID ETF** (beta-hedged ~1.2×,
  vol-targeted). Also documented: a cleaner variant that shorts the residential-solar + EV-charging
  sleeve directly instead of the whole index.
- **What it honestly is:** a *crowded, regime-dependent style bet* — "grid / AI-power equipment beats
  the broad clean-energy / smart-grid complex." Strong 2023–26 backtest (Sharpe ~1.1 vs GRID, ~2.2
  vs the solar/EV sleeve). **Negative Sharpe pre-2023.** It is not a discovered signal and the two
  legs are +0.5 correlated, not anti-correlated.
- **After ~10 signal attempts** across this session and prior ones, **no tradable signal-based edge
  was found in this theme.** The proposal stands on structural logic (who the customer is) plus
  trailing performance, and must be pitched with full disclosure of the crowding and regime risk.
- **One unexplored avenue with real potential:** interconnection-queue *velocity* — a forward
  demand book — but it needs a multi-year data-collection commitment before it can be tested.

---

## 1. The proposed trade — "Grid Buildout, GRID-hedged"

### 1.1 Construction

**Long — grid / data-center power buildout (equal-weight):**

| Group | Names |
|---|---|
| US pure-plays (the "9") | ETN, GEV, HUBB, VRT, NVT (electrical equipment) · PWR, MYRG, PRIM (electric-utility / power-line contractors) · FLNC (grid-scale storage) |
| Foreign grid equipment | ABB (ABBNY), Schneider Electric (SBGSY), Prysmian (PRYMY), Hitachi (HTHIY); add Siemens Energy (SIEGY / SMNEY — could not fetch via yfinance during this work, source it), Legrand, Nexans as liquidity allows |

Inclusion rule (unchanged from the basket spec): latest-10-K business description only — grid T&D
equipment, data-center power & cooling, grid-scale storage, or electric-power-infrastructure EPC.
Never selected on past returns. Equal-weight, quarterly rebalance, 25% single-name cap.

**Short — the GRID ETF** (First Trust NASDAQ Clean Edge Smart Grid Infrastructure Fund). Practically
you short the ETF; economically that nets to *long the concentrated buildout position / short
everything else GRID holds*. What "everything else" is (holdings 2026-08-28):

- **~24% international transmission utilities** — National Grid 4.2%, E.ON 4.2%, Terna 1.8%, Hydro
  One 1.5%, Elia, Red Eléctrica, Iberdrola, plus a Brazilian transmission cluster (Equatorial,
  Copel, Energisa, Alupar, ISA, Transmissora Aliança). **Only US utility: AES at 0.04%.**
- **Residential solar inverters** — Enphase, SolarEdge.
- **EV charging** — ChargePoint (Blink not in GRID but same sleeve).
- **Behind-the-meter storage / DER software** — Stem, etc.
- **Grid metering & grid-edge software** — Itron, Landis+Gyr.
- **Building electrification / controls** — Johnson Controls, Acuity.
- **Foreign equipment makers** (the same names you also hold long, at index weight).

**Sizing:** beta-hedge the long to the short (~1.2× short notional per 1× long over 2023–26), then a
20 %-vol target on the spread with a 1.5× cap. Borrow on GRID is cheap (~0.4 %/yr); you forgo GRID's
dividend (~1 %).

### 1.2 The numbers (beta-hedged + vol-targeted)

| window | CAGR | Sharpe | Max DD | SPY β |
|---|---|---|---|---|
| **2023-01 → 2026-08 (AI/data-center era)** | ~26 % | **~1.1** | ~−22 % | ~0.15–0.20 |
| 2020-01 → 2026-08 (full) | ~23 % | ~0.86 | ~−22 % | ~0.03 |
| **2026 YTD (out-of-sample-ish)** | slightly **negative** (~−3 %) | ~−0.2 | ~−15 % | — |
| 1:1 dollar-neutral, 2023–26 | ~17 % | ~0.92 | ~−22 % | ~0.35 |

Adding the foreign grid names to the long **barely moves the spread** (US-only vs GRID ≈ Sharpe 1.0;
US+foreign vs GRID ≈ 1.1). The geography leg is real but minor.

### 1.3 The argument

The long book's demand comes from **utility and hyperscaler capex** — transmission, substations,
switchgear, data-center power & cooling, grid-scale storage. GRID's non-overlapping half is
dominated by **consumer / distributed electrification** (rooftop solar, home batteries, EV chargers)
and **regulated international transmission utilities**. Going long one and short the index nets out
the shared "electrification theme / decarbonization sentiment / clean-energy fund flows" beta and
leaves a tilt toward **large-load capex demand vs. household electrification demand**, plus a
duration/rates component (consumer clean-tech is long-duration and rate-sensitive).

Framed for a pitch: *"a concentrated, index-hedged position in the physical grid and data-center
power buildout — long the equipment and construction that utilities and hyperscalers are buying,
short the parts of the smart-grid index (rooftop solar, EV charging, foreign transmission utilities)
whose demand drivers are unrelated to that buildout."*

### 1.4 Honest risks — say these first in any pitch

1. **It's the most crowded pairs trade in energy-transition long/short.** "Long AI/grid power, short
   clean-energy/solar" is expressible a dozen near-identical ways (our-9 / TAN, our-9 / ICLN,
   FSLR / ENPH, semis+power / solar+wind+EV) that all printed Sharpe ~2 over 2023–26. Everyone owns
   some version. The *long* leg is separately mainstream — a public Fidelity explainer (and BlackRock /
   T. Rowe / JPM equivalents) recommends essentially this name list; `docs/handoff_2026-08-31-
   strategy-proposals.md` already flagged this before this session.
2. **Regime-dependent.** The identical pair had a **negative Sharpe from 2019–2022**, when solar/EV
   led (+403 % for the solar/EV sleeve vs +97 % for the long book) and the correlations were the
   same ~0.5. The 2023–26 result is one macro regime — a rate shock that punished long-duration
   consumer clean-tech and an AI capex boom that rewarded profitable industrial capex.
3. **Not a hedge — a return-dispersion bet.** The short candidates (GRID, ICLN, TAN, solar/EV) are
   **+0.4 to +0.8 correlated** with the long book, not anti-correlated. Monthly corr 2023–26: GRID
   0.78, ICLN 0.55, TAN 0.51, solar/EV 0.43. The short reduces market beta partway (0.5 corr, not
   1.0) but you are shorting a correlated asset that happened to lag.
4. **The engine has largely run.** Enphase −86 %, SolarEdge −89 %, ChargePoint −97 %. The short leg's
   further downside is capped and its **squeeze risk is elevated** — a rate-cut / risk-on rotation
   would rally the washed-out shorts hard.
5. **Adverse scenario, not yet observed:** a growth scare where rates fall, beaten-down solar
   rallies on the rate relief, and the expensive AI-power longs de-rate — you'd lose on both legs.
   Hasn't happened in the data-center era; more plausible now than in 2023.
6. **Shorting your customers.** ~24 % of GRID is international transmission utilities (Terna, Hydro
   One, National Grid) — the entities *doing* the grid buildout your long thesis depends on. It's a
   conceptual contradiction and a low-beta, rate-sensitive, FX-exposed (GBP/EUR/BRL) drag on the
   short.
7. **Hindsight universe.** The long names were fixed in 2026 knowing which of the theme's companies
   won; the absolute figures are inflated by selection.

### 1.5 Where the short leg *did* work (data-center era only; COVID excluded as pre-theme)

| episode | long book | solar/EV short |
|---|---|---|
| DeepSeek drawdown (Nov 2024 – Apr 2025) | −32 % | −38 % |
| April 2025 tariff shock | −7 % | −12 % |
| 2026 (long book up) | +45 % | +3 % (cheap drag) |

In both data-center-era drawdowns the short fell harder, so it offset — but that is **two episodes,
one regime**, both broad risk-off events where higher-beta clean-energy naturally falls more. It is
weak evidence for "the short pays off when we're hurting."

---

## 2. Cleaner variant — short the DER sleeve directly

Shorting **residential solar + EV charging + behind-meter storage** (ENPH, SEDG, CHPT, RUN, BLNK,
STEM) instead of the whole GRID ETF:

| window | Sharpe | CAGR | Max DD | SPY β |
|---|---|---|---|---|
| 2023-01 → 2026-08 | **2.23** | 68 % | −23 % | 0.46 |
| 2019-06 → 2022-12 (pre-AI) | **−0.12** | −2 % | −42 % | 0.59 |
| 2026 YTD | 1.21 | 31 % | −10 % | 0.41 |

Pros: **all the spread lives in this short** (shorting GRID gives only Sharpe ~1 because GRID is
~60 % the same names you're long — you're shorting yourself). Removes the "shorting the transmission
utilities" contradiction. Cons: higher SPY beta (0.46 — *not* market-neutral), even more crowded,
and the pre-AI Sharpe is squarely negative. This is the purest expression of "long AI power / short
the solar-and-EV bust" — highest backtest, highest concentration, highest regime risk.

Analogs (all Sharpe ~2, 2023–26): our-9 / TAN 2.03 · our-9 / ICLN 2.02 · utility-scale solar
(FSLR, NXT) / residential solar+EV 1.29. Structural cousins in other sectors: oil-services / E&P,
lithium-equipment / lithium-miners.

---

## 3. What else was tried — and the conclusions

### 3.1 Layer-2 grid-congestion regime signal — BUILT, gate FAILED, adopted by PM override

Full module `grid_equipment_basket/grid_regime.py` (merged to `main`): a 6-rung pre-registered
ladder mapping trailing-z-scored transmission-congestion $ in the DC-heavy PJM zones (DOM, AEP,
COMED, PPL) to a monthly basket-exposure multiplier, composed with the vol target in place of
layer-1's price trend gate. Gate `--overlay-l2`, config `REGIME_*`, spec
`docs/superpowers/specs/2026-08-31-grid-regime-layer2-design.md`, results
`docs/grid-regime-layer2-results.md` (§1–9).

- **No rung cleared the pre-registered gate** (beat layer-1-only on Sharpe AND Calmar on both the
  2023–25 and 2020–22 windows, on a plateau). Every rung beat on Sharpe, none on primary-window
  Calmar (2.16 vs 2.32); plateau 0/N.
- **Adopted rung 1 as the "recommended overlay" by PM decision** (`REGIME_ENABLED = True`) despite
  the miss, on a generalisation + physical-differentiator argument (§1a of the results doc).
- **Later evidence walked that back:** see §3.2–§3.7. The overlay's drawdown protection is a generic
  vol-target de-lever; the *congestion-specific* contribution is not there.

### 3.2 Option A — DC-minus-rest-of-PJM relative congestion — FAILED

`relative_regime_composite` — z-score of congestion in the 4 DC zones minus the mean of the other 16
PJM zones, to strip the common-mode (system-wide demand) component. Still missed the primary-window
Calmar bar, and **removing the common-mode component erased the prior-window edge** (prior Sharpe
0.61 → 0.21). This proved the absolute signal's 2020–22 "win" was the **COVID demand-collapse
coincidence**, not a data-center signal.

### 3.3 Option B — signal-tilted long-basket / short-utilities pair — FAILED

Hedge ratio on a rest-of-PJM utility short, tilted by the relative congestion signal. Failed the
pre-registered **spread-informative** check: the basket beat XLU by *more* when the signal said
congestion was easing (+18 bp/day) than tightening (+7 bp/day) — the signal was backwards. Any
Sharpe improvement over a static pair came from carrying a smaller average short, not from signal
content.

### 3.4 Out-of-sample update — Jan–Aug 2026 — congestion signal weakened

Fetched PJM zone LMP for 2026 (prices already cached). In the 8 months of genuine OOS:

| | Sharpe | Max DD |
|---|---|---|
| price gate + vol target | **1.16** | −11 % |
| congestion overlay | 1.03 | −11 % |
| relative (option A) | 0.85 | −11 % |

Drawdown protection held; **the congestion signal did not beat the plain price gate OOS.** Option A
called "step back" in Apr and Aug 2026 and cost ~4 points. On the extended primary window the strict
gate is still not met (gaps narrowed: Sharpe 1.54 vs 1.51, Calmar 1.98 vs 2.18).

### 3.5 Big-four capex-deceleration de-risk trigger — FAILED

`grid_equipment_basket/capex_signal.py` (tested, **not wired live**). Aggregate discrete-quarter
capex of MSFT + Alphabet + Amazon + Meta from SEC XBRL (YTD-ladder derivation for GOOGL/META, which
tag only Q1 as discrete; verified against known annual totals — big-4 CY2024 $228 B, CY2025 $376 B).
One-directional de-risk when yoy growth < 15 % and 2-quarter-decelerating, point-in-time via a
50-day filing lag.

- **Fires ~a year late.** It de-risked Feb 2023 → Feb 2024 — the basket's **best year (+77 %)**.
- **Did NOT fire during the DeepSeek drawdown** — big-4 capex was *accelerating* +59 % → +68 % yoy
  through it.
- "Price gate + capex" == "price gate alone" on both real drawdowns. The one useful fire (Feb 2020,
  pre-COVID) is a coincidence.
- Structural: capex is reported ~6 weeks late and moves slowly; the stocks price the capex *cycle*
  6–12 months ahead. Lagging confirmation, not a leading signal.

### 3.6 Three "differentiation" ideas — all weak (`docs/triage_2026-08-31-differentiation-ideas.md`)

| idea | verdict |
|---|---|
| **Sell the overlay as drawdown insurance** | FAILED — "premium" is 16–28 pp/yr of forgone return in calm years vs ~10 %/yr for a rolled put program; barely beats a naive static 70 %-invested. Only the price gate as an *episode-level* tail cutter has any legs, and it drags CAGR 40 %→24 %. |
| **Interconnection-queue velocity** | NOT TESTABLE — one vintage each for PJM (2025-09 LAS deck) and ERCOT (one back-filled TAC report); no history of how the forecasts moved. **The only idea with genuine alpha potential** — a 2–3-year data-collection project. |
| **Un-crowd the universe** (POWL, ATKR, AZZ, SPXC, BWXT, GNRC, CMI, ITRI, BE, WCC) | FAILED — underperforms the marquee basket on return and Sharpe, drawdown deeper (−52 %), 0.84 correlation so no diversification. The marquee names are the *better* performers because they are the quality names in the theme. |

### 3.7 Sector-neutralisation / correlation diagnostics

- Congestion signal vs monthly basket return: **corr −0.06 raw, +0.05 to +0.12 after hedging out
  the sector**, ~0 at a 1-month lead. No information for it to time.
- After hedging out XLI / GRID, the overlay still improves the residual's Sharpe — **but since
  congestion is uncorrelated with the residual, that is the vol-target de-lever, not congestion.**
- These are **multi-year-backlog companies** (transformer lead times 2–4 yr). A monthly
  weather-driven congestion print cannot transmit to their fundamentals or their stocks on a
  tradable horizon. Whatever relationship exists is sentiment/narrative coupling or coincidence —
  which is why every congestion variant worked in-sample and faded out-of-sample.

### 3.8 Home-bias decomposition — grid infra rewards it ~10× more than the broad market

| 2023-01 → 2026-08 (beta-hedged + vol-tgt) | spread CAGR | Sharpe |
|---|---|---|
| US grid / global GRID | ~28 % | ~1.0–1.4 |
| SPY / EAFE (ex-US developed) | ~3 % | ~−0.03 |
| SPY / all-country ex-US | ~2.5 % | ~−0.10 |

The plain "US vs rest-of-world" equity bet returned ~3 % at ~zero Sharpe over the era. In grid
infrastructure the US-vs-ex-US premium was an order of magnitude larger — the AI/data-center capex
is genuinely US-concentrated and the US pure-plays (VRT, GEV) have no European equivalent that
re-rated as hard. **But** adding the foreign grid names to the long barely changes the spread — the
geography leg is real but a minor contributor next to the sub-sector leg.

### 3.9 Sub-sector decomposition — the spread lives in the short leg

| 2023-01 → 2026-08 (beta-hedged + vol-tgt) | Sharpe |
|---|---|
| our 9 / GRID | 1.03 |
| our 9 / **solar+EV+DER sleeve** | **2.22** |
| our 9 + foreign / solar+EV+DER sleeve | 2.31 |

Shorting GRID gets ~1.0 because GRID is ~60 % the same buildout names — you are shorting yourself.
The full effect requires shorting the **residential-solar + EV-charging** names directly. GRID's
non-buildout parts over 2023–26: ENPH −86 %, SEDG −89 %, CHPT −97 % (solar/EV), plus flat-to-modest
grid software (Itron +87 %) and the international transmission-utility sleeve.

---

## 4. Prior-session record (context — pre-2026-08-31)

- **Five cross-sectional stock-selection attempts, all FAILED:** disclosed-backlog-growth tilt;
  value-chain maker-vs-contractor tilt; backlog-surprise factor (1,778 events); grid-demand
  sensitivity factor (Proposal D, IC t = −0.47); transmission rate-base compounders (FERC Form 1,
  Q5−Q1 Sharpe −0.61). Plus Proposal A (zone-matched congestion pair) killed at triage on breadth.
- **Layer 1 (trend gate + vol target)** built and merged: helps in-sample on the primary window
  (Sharpe 1.36 → 1.58, DD −41 %→−18 %) but **hurts on 2020–22** (Sharpe 0.69 → 0.20) — overfit to
  the single DeepSeek drawdown.
- **No structural short was ever found.** Every candidate (short SMH, QQQ, XLU, TAN, energy-intensive
  industrials, crypto miners, homebuilders, put-spreads) is positively-correlated equity beta that
  also rose over the window — shorting it bleeds carry and cuts Sharpe.

Accumulated conclusion, reinforced this session: **no demonstrable cross-sectional or signal-based
stock-selection edge in this theme at this sample (one macro cycle, ~15 yrs of grid data, ~3.7 yrs
of the actual data-center regime).**

---

## 5. Still on the list / not attempted

Genuinely untried ideas, roughly in order of expected value:

1. **Interconnection-queue velocity** (from the two-layer handoff, and triage §2). A forward demand
   book — Δ large-load MW in the PJM LAS / ERCOT large-load queues, mapped to which equipment names
   get the orders. **Not testable now** (one vintage each). Requires committing to archive the
   quarterly vintages going forward (and/or an archive dig of old PJM LAS decks and ERCOT TAC
   reports) and revisiting in ~2028. This is the one avenue with real, non-consensus alpha
   potential.
2. **RT/DA (real-time minus day-ahead) LMP spread** — the deferred rung 7 of the layer-2 ladder,
   input #3 of `docs/handoff_2026-08-31-two-layer-overlay.md`, and the subject of
   `docs/handoff_2026-08-15-rt-da-spread-signal.md`. The two-layer handoff frames it as a
   *cross-sectional* axis (RT spikes benefit merchant-generation-levered names — GEV, IPP-exposed —
   and hurt T&D / pure-equipment names), not another timing signal. Needs a `fetch_lmp_rt()` +
   `ISO_RT_LOCATION_TYPE` in `grid_resilience/data/grid_data.py` and a live `gridstatus` RT-LMP
   availability probe over 2018–2026 (CLAUDE.md says infra is ~90 % ready, not built). Low expected
   value — day-ahead congestion showed zero equity correlation, and RT/DA is a related quantity —
   but it is the one item literally on the list that was never attempted.
3. **Earnings-estimate-revision / analyst-breadth momentum** (from `docs/handoff_2026-08-29-grid-
   equipment-basket.md`). A standard momentum-of-fundamentals signal, never tested here. Needs an
   estimates data source (I/B/E/S, Visible Alpha, or a scrape).
4. **Backlog *coverage* / sustained book-to-bill > 1 / lead-time & price-escalation language in
   filings** — a *pricing-power* signal distinct from backlog *growth* (which failed). Text-mining
   the 10-Ks/10-Qs for lead-time and escalation language is the untried piece.
5. **Quality / balance-sheet weighting** — weight the basket by leverage / margin stability instead
   of equal-weight. Never run standalone.
6. **Options-structured hedge** — put spreads financed by a call overwrite on the strongest-momentum
   name, or a wide collar. Needs an options-data source; not attempted.

Items considered done/closed: layer-2 congestion (built, failed); option (c) "one-directional
veto / combine price gate + congestion" (tested as the c1/c2/c3 union variants — all worse than
congestion-only); "widen the universe" (the un-crowd test — failed as a diversifier); per-name
6-month momentum (explored in earlier sessions, no drawdown control).

### Cross-check against every other handoff (done 2026-09-01)

| Handoff | Anything left for the grid-equipment trade? |
|---|---|
| `handoff_2026-08-31-two-layer-overlay.md` | Only **RT/DA spread (#3)** — see §5.2. #2 (congestion) built+failed; #1 (queue velocity) = §5.1; design Qs 1–7 all addressed. |
| `handoff_2026-08-15-rt-da-spread-signal.md` | The RT/DA idea itself — §5.2. Never brainstormed in depth. |
| `handoff_2026-08-15-outage-reserve-margin-signal.md` | **Nothing — dead.** A live spike showed EIA-860's `status` field is annual-scale (retirements/mothballing), does not register days-to-weeks forced outages — `OP` ratio was 1.0 through Winter Storm Uri. "Do not pick this back up without a new data source." |
| `handoff_2026-08-15-pairs-basket-construction.md` | Adjacent — it's the **`grid_resilience` utility strategy**, not this basket. Its idea (basket-vs-basket *within beta-matched peer groups* to cancel sector beta) is a more rigorous version of what §1 does crudely (long concentrated / short the index). Worth borrowing the peer-group discipline if §1 is built as a real module. Open question there (merchant group only 2 names) doesn't apply to grid-equipment. |
| `handoff_2026-08-31-strategy-proposals.md` | All three proposals resolved: B (transmission rate-base) built + gate FAILED; D (grid-demand sensitivity) gate FAILED; A (zone congestion pair) killed on breadth. It also records — independently — that **a public Fidelity explainer recommends the same names**, i.e. the long thesis was already flagged as mainstream before this session. |
| `compact_2026-08-*` (dc-load, peer-group, dc-multi-source, original-thesis) | All `grid_resilience` utility-strategy diagnostics; conclusion there mirrors ours — the long book rides sector beta, no short edge. Nothing new to port. |

---

## 6. Code & docs in the repo now (all on `main`)

| Path | What |
|---|---|
| `grid_equipment_basket/grid_regime.py` | congestion signal + pre-registered gate + ladder runner + relative-composite (option A) + signal-tilted pair (option B) + `shipped_config()`/`live_multiplier()`. `REGIME_ENABLED = True` (rung 1). ~55 tests. |
| `grid_equipment_basket/capex_signal.py` | big-4 capex-deceleration signal. Tested, **not wired to any live path**. |
| `grid_equipment_basket/overlay.py` | layer-2 seam: `regime_exposure`, `apply_overlay_l2`, `exposure_series_l2` (layer-1 functions untouched). |
| `grid_equipment_basket/__main__.py` | `--overlay-l2` flag. |
| `docs/superpowers/specs/2026-08-31-grid-regime-layer2-design.md` | layer-2 spec (marked BUILT / gate not passed / rung 1 adopted). |
| `docs/grid-regime-layer2-results.md` | full results §1–9 incl. the OOS update and options A/B. |
| `docs/triage_2026-08-31-differentiation-ideas.md` | the 3 differentiation ideas + capex-decel (§1–4). |
| Artifact | 1-page tear sheet for the basket + congestion overlay (private): claude.ai/code/artifact/0e5f0591-94b0-48b0-a884-24097940f1e4 — **stale** (2023–25 window, pre-OOS; frames congestion as a driver). |

No module was built for the §1 long/short trade — it was analysed with ad-hoc scripts over the
existing `overlay._basket_series` / `data.prices.fetch_prices` helpers.

---

## 7. Recommended next steps

1. **Decide whether to ship the §1 long/short at all.** If yes, pitch it honestly as a
   concentrated, index-hedged grid-buildout position — a *style/relative-value* trade with a strong
   trailing record and explicit crowding + regime + squeeze risk — not as a signal. Build it as a
   real module (long universe incl. foreign ADRs, the GRID or DER-sleeve short, beta-hedge + vol
   target) with a pre-registered out-of-sample rule.
2. **Turn `REGIME_ENABLED` back to `False`.** The OOS data and the correlation work say the
   congestion overlay does not beat a plain price-gate + vol-target and adds no congestion-specific
   information. Keep the code and the negative result; stop running it live.
3. **If a genuinely differentiated edge is required:** commit to the interconnection-queue-velocity
   data-collection project (§5.1) — accept no validation until ~2028 — or move the search to a
   less-arbitraged theme. Ten honest attempts here is a strong prior that the edge, if it exists,
   is not reachable with the data available.
