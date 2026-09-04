# FTR bid-implied forward-congestion signal — results (2026-09-02)

**Status: FAILED.** Same pre-registered gate as the layer-2 congestion regime
(`docs/grid-regime-layer2-results.md`), same outcome. Logged as attempt #12
against this theme's search for a signal-based edge — see
`docs/handoff_2026-09-01-grid-buildout-long-short.md` for the full prior
record.

## 1. What this was

`docs/handoff_2026-09-01-grid-buildout-long-short.md` §8.1 proposed an
RT/DA (real-time minus day-ahead) LMP spread as the next untried signal.
Brainstorming it surfaced a real design gap — the proposed cross-sectional
construct requires mapping specific companies (ETN, GEV, HUBB, VRT, NVT, PWR,
MYRG, PRIM, FLNC) to merchant-generation price-node exposure, and none of
them is a generator sitting behind an identifiable node. The conversation
pivoted to a related, more promising idea: **FTR (Financial Transmission
Rights) prices are a genuinely forward-looking congestion instrument** —
unlike a trailing LMP z-score, an FTR price is itself a multi-year-forward
bet on congestion, a much closer duration match to these companies'
multi-year order books than anything tried in the layer-2 work.

Feasibility check:
- **PJM's actual FTR auction-clearing prices require a PJM membership login**
  (the historical-results page redirects to PJM SSO; confirmed 2026-09-01).
  Not accessible the way this repo's LMP/load data is.
- **PJM's FTR *bid* data (`ftr_bids_mnt`) is public, no login** — but bids are
  not clearing prices; they reflect what participants were willing to pay,
  not what the auction awarded. Treated throughout as a sentiment/demand
  proxy, not a reconstruction of the true market price.
- **ERCOT's CRR (their FTR) monthly auction *results* are genuinely public**
  with real clearing prices, but the live report API only surfaces a rolling
  ~365-day window — full 2018-2026 history would need ERCOT's separate
  archival portal or its own scrape, untested. Not pursued this pass
  (PJM alone was enough to test the idea).

## 2. Construction

- **Data**: PJM DataMiner `ftr_bids_mnt`, buy-side Obligation bids only
  (the standard load-hedging instrument), `OnPeak` class. `sink_pnode_name` /
  `source_pnode_name` are **not filterable server-side** — bids are bus-level
  across all of PJM (150k-600k rows/month even after the trade/hedge/class
  filter) — so each month is downloaded in full and filtered client-side to
  the bare zone-aggregate sink names confirmed present for the four DC-heavy
  zones already used in `grid_regime.py`: `AEP`, `COMED`, `DOM`
  (+ `DOMINION HUB`), `PPL`. Real volume at these sinks: 240-1097 bids and
  700-5000 MW per zone per month (checked against Jan 2019).
- **Signal**: MW-weighted mean `quoted_price` per zone per auction month,
  equal-weighted across the four zones, trailing-z-scored (same window/winsor
  as `grid_regime`: 756-day window, 252-day min periods, ±3 winsor).
- **Point-in-time lag: 4 months**, taken directly from the feed's own
  description ("posted on a four month delay") — a January auction's bids
  aren't usable in the backtest until May.
- **Overlay**: feeds `grid_regime.regime_multiplier` (discrete mode, same
  thresholds) unchanged — same seam as the congestion regime, `overlay.
  apply_overlay_l2`, so the comparison to layer-1 is apples-to-apples.
- Code: `grid_equipment_basket/ftr_signal.py` (`ftr_composite`,
  `ftr_signal_report`) + `grid_resilience/data/grid_data.py`
  (`_fetch_pjm_ftr_bids_raw`, `fetch_ftr_bids_by_month`). ~25 tests.

## 3. Live backfill

96 months (2018-01 through 2025-12) of `ftr_bids_mnt`, Buy+Obligation+OnPeak,
downloaded and cached individually (no date column on this feed — each
auction month is its own cache file). **106 minutes wall-clock** — far
heavier than the RT-LMP feasibility check earlier in this session, and
heavier than a sparse manual probe suggested: PJM's rate limiter throttled
hard under sustained sequential pagination (frequent 429s, exponential
backoff up to 30s+ per retry). A single-page manual probe is not a reliable
predictor of a full-backfill's real cost against this API.

One genuine robustness bug surfaced and was fixed along the way: `_pjm_get`
retried on HTTP 429 but not on a transient read-timeout/connection-error
mid-download of a large (50k-row) page — the first backfill attempt crashed
14 months in on exactly that. Fixed generically (any PJM DataMiner caller
benefits, not just this one) and covered by tests in `tests/data/test_pjm_retry.py`.

## 4. Signal mechanics — it engaged, it just didn't help

Composite coverage: 2551/2922 days non-NaN (~87%) over the signal-scoring
span. The exposure multiplier actually moved, not a degenerate no-op:

| multiplier | days |
|---|---|
| 0.60 (step back) | 1158 |
| 1.00 (neutral) | 1004 |
| 1.25 (lean in) | 760 |

Average exposure 0.91 — real, substantial engagement in both directions.

## 5. Gate result

| window | buy&hold Sharpe/MaxDD | vol-target-only | layer-1-only | **FTR rung** |
|---|---|---|---|---|
| primary (2023-25) | 1.44 / -41% | 1.66 / -22% | 1.58 / -18% | **1.57 / -21%** |
| prior (2020-22) | 0.69 / -45% | 0.42 / -42% | 0.20 / -42% | **0.21 / -42%** |

| gate | result |
|---|---|
| G1 (beats layer-1 on Sharpe AND Calmar, primary) | **FAIL** (Sharpe gap -0.012, Calmar gap -0.46) |
| G2 (beats layer-1 on Sharpe AND Calmar, prior) | PASS, but **marginal** (Sharpe gap +0.006, Calmar gap +0.001 — both inside the 0.05 marginal band) |
| G3 (no worse drag than layer-1, mean annual return, both windows) | **FAIL** |

**Verdict: FAIL.** The FTR rung tracks layer-1-only almost exactly on both
windows despite real multiplier variation — the timing information in the
bid-implied composite doesn't translate into better risk-adjusted outcomes
than the plain price-gate + vol-target already shipped (and currently
`REGIME_ENABLED = False`, per `docs/handoff_2026-09-01-grid-buildout-long-
short.md` §7.2).

## 6. Conclusion and what's still open

Same story as every congestion variant in `docs/grid-regime-layer2-results.md`:
a physically-motivated, forward-looking-by-construction signal still fails to
beat a plain vol-target overlay on this basket. This is now attempt #12
(counting the layer-2 ladder's variants, the capex-deceleration trigger, and
this one) with no demonstrable signal-based edge in this theme.

**Genuinely still open, not touched this pass:**
- The original §8.1 idea (RT/DA spread as a *cross-sectional* company-ranking
  axis) was never built — only its data feasibility was confirmed (PJM clean
  2019-2026, ERCOT clean for completed years). The conceptual gap (no company
  in the universe is a merchant generator behind an identifiable node) is
  still unresolved and would need to be solved before it's worth building.
- ERCOT CRR *clearing prices* (as opposed to PJM bids) were never tested —
  genuinely public, but full historical depth needs either ERCOT's archival
  portal or a scrape of its dated market notices, neither attempted.
- §8.2-§8.6 of the 2026-09-01 handoff remain untried.
