# Forward power × hedge-adjusted exposure → generator returns — pre-registration

**Written 2026-09-24, before any return data was loaded.** Feasibility work (filings only) done first;
nothing below has been fitted to returns.

## Hypothesis
Changes in the forward power curve move merchant generators' future earnings in proportion to their
**unhedged** volume. If equity investors react to forward-curve moves with a lag, a hedge-adjusted
forward-curve revision predicts next-month excess returns. Motivation: Bushnell-Chong-Mansur (AEJ:Policy
2013) — equity prices power-price news; Probe A (2026-09-07) failed because hedges sever
realised-price → earnings, so this test puts the hedge ratio in explicitly.

## Data (all point-in-time)
- **Forward curve:** EIA STEO `ELWHU_PJ` (PJM West) and `ELWHU_TX` (ERCOT North), Wayback vintages
  already cached in `research/steo_spark_spread/data/steo_panel.parquet`. Event date = Wayback capture
  date (on or after EIA release → conservative).
- **Hedge %:** quarterly earnings 8-K EX-99 exhibits (EDGAR), valid from filing date.
  - VST: "% of expected generation hedged" for Y, Y+1(, Y+2) — every quarter 2022→2026.
  - CEG: regional table (expected GWh + % hedged, Midwest ex-CMC / Mid-Atlantic / ERCOT / NY),
    2023-02 → 2023-11 decks only (CEG stopped publishing it in 2024). Midpoint of ranges.
  - TLN: "% hedged incl. nuclear PTC", 2024-08 → 2026.
  - **NRG excluded** — no hedge % disclosed (integrated retail nets its own load).
  - A hedge disclosure is stale (name dropped for that event) if older than 130 days.
- **Region weights:** VST from 10-K production volumes: ERCOT 0.53 / PJM 0.47 before 2024-03-01,
  0.46 / 0.54 after (Energy Harbor); East + Sunset mapped to PJM, West (CAISO) dropped.
  TLN 100% PJM. CEG from its own regional GWh (Midwest ex-CMC, Mid-Atlantic, NY → PJM; ERCOT → TX;
  CMC plants treated 100% hedged, per CEG).

## Signal
For name i at STEO vintage v (vs previous vintage v−1):

    s_i,v = Σ_Y w_Y · Σ_hub r_i,hub · (1 − h_i,Y) · ΔF_hub,Y / F_hub,Y(v−1)

- F_hub,Y = mean STEO forecast over months of year Y that are ≥ vintage month (forecast months only),
  same months in both vintages.
- Y ∈ {current year, next year}; w_current = remaining months / 12, w_next = 1.
- h = hedge % for that year from the latest non-stale filing before the vintage date (CEG: open GWh share
  computed regionally).

**Comparator (does hedge-adjustment add anything?):** same formula with h ≡ 0.

## Target
Return from the first close after the vintage date to 21 trading days later, **in excess of SPY**.

## Test and bar (window 2023-01 → 2026-08)
- **Primary:** pooled Spearman rank-IC over all (name, vintage) pairs. **Changed from per-date
  cross-sectional IC before looking at returns** because only 2-3 names have hedge data at any date
  (a 2-name cross-section gives IC = ±1). t = ρ·√(n−2)/√(1−ρ²).
- **PASS: |t| > 1.5** (user-set bar, 2026-09-24). Sign reported: + = under-reaction (tradable
  momentum), − = over-reaction.
- **Robustness (reported, not gating):** (a) vintage-clustered bootstrap p (resample vintages,
  2,000 draws); (b) comparator h ≡ 0 — hedge adjustment is "useful" only if its |t| exceeds the
  comparator's; (c) partial rank-IC controlling for the same-vintage Henry Hub (`NGHHUUS`) revision;
  (d) leave-one-name-out.
- **Kill:** |t| ≤ 1.5, or passes only because of one name (leave-one-out |t| < 1.0), or fully explained by
  the gas revision (partial |t| < 1.0).

## Known weaknesses (stated up front)
n is small (~40-45 pairs), STEO is a model forecast not a traded curve, Wayback vintages are
irregular (gaps up to 5 months), and pairs within a vintage share the same ΔF (hence the clustered
bootstrap).
