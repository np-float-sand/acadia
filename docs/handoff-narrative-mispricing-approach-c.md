# Handoff: narrative mispricing — Approach C (narrative-fundamentals divergence)

**Context:** `docs/untried-tasks-log.md` item 9. Approach A (composite rank-IC): done,
NOT ADOPTED — not significant at any horizon, even after refining both data sources
(`docs/narrative-mispricing-approach-a-results.md`). Approach B (event study): handoff
written, not started (`docs/handoff-narrative-mispricing-approach-b.md`). This is C.

**Reuse, don't re-fetch:** `research/narrative_mispricing/composite_monthly_refined.parquet`
(the fixed EDGAR "data center"+"large load" / GDELT self-baseline composite, 25 tickers,
monthly, 2018-2025) and `research/icr_edgar_retest/icr_edgar_panel.parquet` (full-history
ICR per ticker, quarterly, from item 2 — ICR failed as a *standalone* signal there, but
that doesn't disqualify it as a *conditioning* variable here).

## The idea

Short names where the narrative score is high but a fundamentals measure is weak — a
classic "story stock" divergence, distinct from A (narrative alone) and B (narrative
timing). Candidate fundamentals legs: ICR (already built, item 2), a valuation multiple
(P/E or EV/EBITDA — not yet fetched for this universe), or margin trend (reuse the
value-chain gross-margin machinery if it generalizes). Any of these works as a first cut;
don't over-invest in picking the "best" one before checking whether the divergence itself
carries anything.

## Important: don't judge this on a standalone linear test alone

Every signal in this whole research line — this one included, so far — has been screened
with a linear/rank-based test (Spearman rank-IC, HAC-OLS, quintile sorts). That's the right
first gate, but it can miss an interaction that only shows up as a *conjunction*: narrative
extremity might do nothing on its own and fundamentals weakness might do nothing on its
own, while the two *together* (high narrative AND weak fundamentals, specifically) carry
real information a linear rank-IC on either variable alone can't see.

**So: run the standard rank-IC gate first (cheap, consistent with everything else here),
but if it comes back flat, don't stop there** — try folding the narrative composite and a
fundamentals measure into a small multivariate model (random forest or GBM) alongside
other features already built in this project (stress beta, ICR, momentum, valuation) and
check whether narrative earns any feature importance / improves out-of-sample fit even
where its own rank-IC was near zero. A flat standalone IC does not prove the data is
useless — it proves the *simple linear form* doesn't capture it.

**Guard against the exact trap this project already hit once.** The Henry Hub forward-gas
work tried a "kitchen-sink GBM/RF on 25 pooled signals" and got a strong-looking 2023-26
out-of-fold rank-IC (+0.2 to +0.3) that completely vanished when tested on the full
2018-2026 window — a textbook overfit on a small, autocorrelated sample
(`docs/hhub-forward-gas-signal-results.md`). This universe has ~90 monthly cross-sections
at most, heavily autocorrelated — plenty of features, very few independent time periods.
If a tree-based model is used: pre-register the OOS window *before* looking at results,
use a purged/embargoed split (not random k-fold, given the autocorrelation), and treat any
in-sample feature-importance number as a hypothesis to re-test out-of-sample, not a result.

## Suggested order

1. Build the fundamentals leg (start with ICR, already available — cheapest).
2. Run the plain interaction test first: sort on narrative score, within the top tercile
   sort again on the fundamentals measure, compare the high-narrative/weak-fundamentals
   bucket's forward return to the rest. Cheap, linear-ish, matches this project's usual
   first gate.
3. Only if that's flat: try the RF/GBM broadening described above, with the overfit
   guardrails above treated as non-negotiable, not optional polish.
4. Pre-register a PASS bar before step 2, not after seeing the numbers.
