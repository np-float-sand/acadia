# Handoff — Peer-Group (Basket-vs-Basket) Portfolio Construction

**Status:** Brainstorming in progress, NOT yet a finalized spec. One open design question below needs to be resolved before writing the design doc and implementation plan.

**Start a fresh conversation with:** "Let's continue the peer-group portfolio construction brainstorm — read docs/handoff_2026-08-15-pairs-basket-construction.md" and use the `superpowers:brainstorming` skill to pick up where this left off (open question below is the next thing to resolve).

---

## Why this exists — the diagnosis that led here

Session context (full detail in `docs/compact_2026-08-13-dc-load-signal-results.md` and `output_dcqueue/dc_executive_summary.html`): a new "DC load signal" was built to replace ICR in the regulated-utility path of the Grid Resilience Strategy's factor model. It works — beats ICR on Sharpe (-0.213 vs -0.276) — but **neither configuration beats simply buying and holding XLU** (Sharpe 0.162, total return +87.7% over the Apr 2018–Dec 2025 backtest).

Diagnosis of *why*, computed directly from the backtest's long_book/short_book P&L legs and correlation to XLU:

| | Sharpe | Total Return | Correlation to XLU |
|---|---|---|---|
| XLU (buy & hold) | **0.162** | **+87.7%** | — |
| DC-queue long book | 0.102 | +53.4% | +0.76 |
| DC-queue short book | -0.615 | -23.2% | -0.78 |
| Naive equal-weight short of all 18 tickers | -0.557 | -41.9% | — |

Two findings:
1. **The long book is mostly riding sector beta**, not adding real stock-selection alpha — it's highly correlated with XLU (0.76) and its total return (+53%) is *less* than just holding XLU (+88%), despite being the strategy's most concentrated, highest-conviction picks.
2. **The short book has no edge over naive selection** — the factor's picked shorts (Sharpe -0.615) perform about the same as or worse than an equal-weight short of the *entire* 18-name universe (Sharpe -0.557). Most of the short book's loss is uncompensated short-beta drag (utilities had positive drift this whole period), not "wrong stock picks" in a fixable sense.

**Conclusion:** the problem is architectural (portfolio construction exposes the whole book to sector beta), not a signal-quality problem that a better regulated-path signal alone can fix. Re-validating the grid search or tuning existing signals (already tried: per-ticker zone GSI — rejected, made IC worse; XLU hedge on the short side — every config went negative; revenue-mix/dual-track signal blending — both underperformed hard-switch) won't address this.

## The idea

Instead of ranking the **whole 18-name universe** and taking top-3 long / bottom-5 short (today's `grid_resilience/portfolio/construction.py::build_weights()` — pure equal-weight, dollar-neutral, no beta consideration), split the universe into **peer groups with similar business-model/beta exposure**, and build long+short baskets **within each group separately**. This cancels most of the sector-beta exposure (long and short legs within a group have similar rate/regulatory/beta profile) while preserving diversification within each leg (multiple names per side, not a single fragile pair).

### Why not literal 1-vs-1 pairs trading?
User's objection, and it's correct: with only 18 tickers, true 1-vs-1 pairs concentrate all risk on two names — if one has an idiosyncratic surprise (M&A, earnings miss, unrelated-to-thesis event), the whole pair's P&L is dominated by that one mistake with zero diversification. Basket-vs-basket within peer groups keeps the beta-cancellation benefit of pairing while avoiding this fragility, as long as each group has enough names to form a real basket on both sides.

### Group definitions considered
Business-model-only grouping (not ISO) was chosen as the more natural split, since the original PJM-alpha-destruction problem (see `docs/superpowers/specs/2026-06-29-business-model-aware-factor-design.md`) was diagnosed as a business-model mismatch, not an ISO mismatch. Universe breakdown (`grid_resilience/data/utility_node_map.py`, `business_model` field):

| Group | Tickers | Count |
|---|---|---|
| merchant | VST, NRG | 2 |
| mixed | CNP, DTE, ETR, EXC, PEG | 5 |
| regulated | AEE, AEP, CMS, D, EIX, EVRG, FE, PCG, PPL, WEC, XEL | 11 |

**Problem:** the merchant group only has 2 names — too small for a diversified basket on both sides, no matter how it's grouped. This is the open question below.

## OPEN QUESTION — resume here

How should the 2-name merchant group (VST, NRG) be handled, given it's too small for the basket-diversification benefit either way? Three options were on the table when this session ended, none yet decided:

**A.** Leave merchant (VST/NRG) construction exactly as it is today (small directional long/short, effectively a pair regardless) — it's a small part of the book, and stress-beta was already found to work reasonably for generators specifically (that's why hard-switch routes them to a different signal path than regulated names in the first place). Apply the new basket-vs-basket redesign **only** to the regulated+mixed group (16 names), which is where the diagnosed problem actually lives.

**B.** Make the merchant sleeve **long-only** — drop shorting within that 2-name group entirely, to avoid the fragile-pair problem there too. Redesign focuses on regulated+mixed as in A.

**C.** Something else — not yet explored (e.g., merge merchant into the mixed group despite the beta mismatch, or size the merchant sleeve down significantly to limit its risk contribution regardless of direction).

**Recommendation if picking up cold:** option A is the most defensible default — it's the smallest change (merchant path untouched) and correctly scopes the redesign to where the diagnosis says the problem is. But confirm with the user before proceeding; this was left open, not decided.

## After this question is resolved

Per the `superpowers:brainstorming` skill flow: a few more implementation-shape questions remain before writing the design doc (e.g., exact n_long/n_short per group — the current global 3-long/5-short doesn't map cleanly onto sub-groups of size 5-16; whether group-level dollar-neutrality should sum independently per group or globally; whether `build_rolling_weights()`'s monthly rebalance cadence changes). Then: present design → write spec to `docs/superpowers/specs/YYYY-MM-DD-peer-group-construction-design.md` → self-review → user reviews spec → `superpowers:writing-plans` for the implementation plan.

## Related handoffs (independent, can be worked in parallel or afterward)

- `docs/handoff_2026-08-15-rt-da-spread-signal.md` — a new signal (RT/DA LMP spread) that could feed into this basket's ranking once it exists, but is independently scoped.
- `docs/handoff_2026-08-15-outage-reserve-margin-signal.md` — another candidate signal, focused on differentiating merchant generators specifically.

These were explicitly *not* folded into one mega-design; the user asked to keep them as separate fresh-context conversations. This construction redesign should probably land first, since #3 and #4 are signal improvements that plug into whatever ranking/construction mechanism exists — building them before the construction redesign risks wasted rework if the ranking mechanism changes shape.
