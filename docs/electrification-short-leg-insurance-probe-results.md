# Short-leg-as-insurance probe — results (2026-09-06)

**Question.** The electrification long-only book is defensible enhanced thematic beta
(≈ VOLT). Given the durable-long/short search is CLOSED (`electrification-ls-strategy-note.md`
§8), is there a *short leg sized purely as insurance* that measurably cuts the book's
downturn vulnerability at an acceptable cost (user bar: 3–6%/yr drag OK if drawdowns
shrink and beta falls)? Menu approved in brainstorming; this note is the record.

**Method.** Long book = 20%-vol-targeted `grid_equipment_basket` proxy (staggered
inception, 25% cap, 42-day lag), 2017-06 → 2026-08. Proxy check: Sharpe 0.84 / MaxDD
−31% / SPY β 0.76 (strategy note's book: 0.71 / −26% / 0.78); corr 0.82 vs PAVE, 0.88
vs VOLT. Each hedge added at fixed short notional 0.25 / 0.50 / 0.75× (masked series for
conditionals). Signals, pre-registered: **S1** = 6-mo change in 10y real yield > 0
(FRED DFII10, the handoff §1b signal; the AND-with-HY-OAS-z version fires 1% of days
over this sample — degenerate, dropped); **S2** = book < 100d MA AND 20d vol > 252d
median (`hedges.conditional_short_mask`). Scoring: full-period CAGR/drag/Sharpe/MaxDD/
β; drawdown over 5 pre-specified episodes; calm-window drag (pre-COVID 17-19, AI-bull
23-24); **feared-scenario check** = cumulative hedge P&L across the 17 months where the
book fell *and* TAN rose (the "solar squeezes while AI-power de-rates" scenario).
Decision rule: max **insurance efficiency = avg of 3 largest episode-DD reductions (pp)
÷ full-period CAGR drag (pp)**, subject to gates — (g1) no episode drawdown deepened,
(g2) feared-scenario P&L ≥ −2%, (g3) pre-COVID and full-period drag ≤ 6%/yr.
Throwaway code: `scratchpad/hedge_probe.py`, `scratchpad/hedge_feasibility.py`.

## What passes all three gates (ranked by insurance efficiency)

| structure | IE | drag %/yr | MaxDD | β | feared P&L | note |
|---|---|---|---|---|---|---|
| **+GLD/IEF 15% sleeve** | **1.9** | 1.9 | −27.3 | 0.65 | **+12.9%** | *raises* Sharpe 0.84→0.87; not a short; already the docs' pick |
| −DC-REIT (DLR EQIX) short, S1, 0.50× | 1.6 | 3.1 | −30.8 | 0.59 | −0.1% | rates hedge in a data-center costume — see caveats |
| de-lever to 0.85× | 1.5 | 2.5 | −26.6 | 0.65 | +10.8% | humble floor; does most of the job |
| **−SPY short, S1, 0.50×** | 1.3 | 2.5 | −26.4 | 0.55 | −0.6% | clean index short, no thematic baggage, no feared-scenario tail |
| −SPY static, 0.25× | 1.1 | 4.0 | −28.4 | 0.51 | +3.3% | static index short bleeds carry (−15.7%/yr standalone) |

Baseline (unhedged): CAGR 22.2% / MaxDD −31.2% / β 0.76. Episode DDs COVID −25.6 /
Rate-22 −28.2 / DeepSeek −24.3 / Tariff-25 −8.8 / Selloff-26 −12.9.

## What fails — and why it's structural, not a sizing artifact

| thematic short | standalone carry | why it fails as a hedge |
|---|---|---|
| **DER / consumer-clean-energy sleeve** (ENPH SEDG CHPT RUN BLNK STEM — the handoff §1b short) | −42%/yr | **Anti-hedge in the feared scenario:** feared-scenario P&L −24% to −110% across sizes. When the book fell and solar rose (rate-relief squeeze), the short bled catastrophically. Definitive kill; confirms strategy note §6–7. |
| **firm / merchant generation** (VST NRG CEG TLN) | −33%/yr | **Deepens the 2022 rate-shock drawdown** (−36% vs −28% baseline at 0.5×) — firm-gen co-moved *down* with the book in the rate-driven selloff — and feared-scenario P&L −4% to −18% (firm bounces with solar on rate relief). Helps only in AI-capex-specific selloffs (COVID/DeepSeek/tariff). My brainstorm hope that "short firm = hedge" (from §7's +0.60 corr / −5.7% in worst months) was wrong: the co-movement is regime-dependent and reverses in the rate episode. |
| **crowding basket** (SMCI DELL ANET GEV) | −45%/yr | Passes only at 0.25× S1 (IE 0.67 — worst of the passers); deepens episodes at S2; ruinous carry. Not worth it. |
| ratepayer / consumer-advocacy basket (handoff §2 B) | — | No such equity exists. Nearest proxy = rate-capped utilities = XLU = +0.3–0.5 corr bond proxy, already rejected. |
| retail / suburban-office REITs (handoff §2 C) | — | Previously tested: +0.58 corr, regime-dependent, −5.2% in the book's worst months. Not re-run; prior negative stands. |

## Conclusion

1. **No data-center-backlash thematic short works as a hedge.** They are either not
   investable (ratepayer baskets) or correlated rate-sensitive equity that *deepens* the
   drawdown in the exact scenario the strategy fears (firm-gen, the DER sleeve). This is
   the same wall the ~25-candidate search hit — the absence of a listed anti-asset, not a
   failure of search.
2. **The best "less vulnerable to downturns" structure is not a short leg.** The
   **GLD + short-duration Treasury 15% sleeve** has the highest insurance efficiency of
   anything passing (1.9), *raises* Sharpe, costs 1.9%/yr, cuts β 0.76→0.65, helps in
   every episode and is strongly positive (+12.9%) in the feared divergence. A mild
   **de-lever to 0.85×** is the humble floor (IE 1.5, 2.5%/yr) and captures two-thirds
   of the benefit. This *confirms* the strategy note's existing recommendation with a
   proper test rather than overturning it.
3. **If a short leg is mandated,** the only defensible one is a **small (~0.5×)
   conditional index short — short SPY (or QQQ) only while the 6-month change in the
   10-year real yield is positive.** ≈2.5%/yr through-cycle drag, MaxDD −31→−26, β
   0.76→0.55, feared-scenario-neutral, no borrow/concept issues. Understand it as
   duration insurance, not an AI-thesis hedge — it earns its keep only in rate-driven
   drawdowns (2022, DeepSeek). Pair it *with* the GLD/duration sleeve, not instead of it.
4. The **conditional DC-REIT short** is a marginally more efficient version of the same
   rates trade (DLR/EQIX carry more duration than SPY) but adds borrow cost, the
   conceptual wart of shorting the buildout's landlords, and a large in-window drag
   during rate-rising equity bull phases (−18%/yr annualised across 2023–24, masked by
   the 3%/yr full-period figure). Not worth it over the plain index short.

**Net:** ship the long-only book with the **GLD/short-duration sleeve + optional
de-lever** as the downturn defense. Add the **conditional real-yield-gated SPY/QQQ
short at ≤0.5×** only if the mandate specifically requires a short leg; disclose it as
duration insurance with a lumpy carry, not a differentiated hedge.

## Follow-up (2026-09-06): composite short legs

Tested combining the DC-REIT short with the DER/solar sleeve and a "rate-sensitive"
sleeve (XLU / ARKK), various weights, gated on S1.

- **The DER sleeve poisons every composite.** Feared-scenario P&L: DER-alone S1 0.25×
  −24% → 50/50 with DC-REITs −12% → DER only 20% of the notional −5% — still worse than
  the −2% gate. Any DER-containing mix also has a *worse* full-period max drawdown than
  unhedged (−33% to −63% vs −31%): shorting solar into its bounces enlarges the
  drawdown. DER is structurally wrong-signed in the squeeze scenario, not noise that
  blends out — same lesson as strategy-note §7/§8 ("diluting the one thing that works
  kills it").
- **Adding XLU to the DC-REIT short lowers efficiency** (2.1 → 1.0). XLU is defensive —
  barely falls in rate shocks — so it adds carry drag without drawdown protection, and
  DC-REITs are already a rate-sensitive bond-proxy short, so XLU doubles that factor
  rather than diversifying it.
- Composites collapse to their one working leg: DC-REITs alone, S1-gated. No composite
  beats the GLD/short-duration sleeve.

## Follow-up (2026-09-06): GLD/duration sleeve + conditional DC-REIT short — BEST STRUCTURE

Pairing the winning long diversifier with the best-scoring short:
**0.85x long book + 0.15 GLD/IEF sleeve − 0.25x (DLR+EQIX) gated on S1 (6-mo real yield rising).**

| | CAGR | drag/yr | Sharpe | MaxDD | beta | feared P&L | eff |
|---|---|---|---|---|---|---|---|
| unhedged | 22.2% | – | 0.84 | −31.2% | 0.76 | – | – |
| GLD/IEF 15% alone | 20.3% | 1.9% | 0.87 | −27.3% | 0.65 | +13% | 1.91 |
| DC-REIT short S1 0.25x alone | 20.7% | 1.5% | 0.81 | −29.7% | 0.68 | ~0% | 2.13 |
| **combined** | 18.8% | 3.3% | 0.83 | **−25.8%** | **0.57** | **+13%** | 1.89 |

Complementary — closes each leg's blind spot. Episode drawdowns: COVID −25.6→−22.5
(sleeve; short is off, real yields were falling), Rate-22 −28.2→−20.5, DeepSeek
−24.3→−16.1 (short does the work), Tariff −8.8→−6.1, Selloff-26 −12.9→−11.3. **Every
episode reduced** — no single structure did that. Feared "solar squeeze" scenario stays
+13% (sleeve dominates, short ~neutral). Sizing: 0.25x DC-REIT is the knee; 0.15x barely
moves the rate episodes, 0.40x pushes bull-window drag to 20%/yr for little extra. GLD/SHY
≈ GLD/IEF (SHY slightly less feared-scenario juice, slightly better calm drag).

**Cost to disclose:** ~15%/yr drag *within* a rate-rising equity melt-up (2023-24; ~25%
cumulative over that window), mostly the DC-REIT short fighting the AI re-rating of
DLR/EQIX. Full-period 3.3% is low only because the short pays in 2022/DeepSeek and
switches off in 2025-26. DC-REIT warts (2 names, ~2.5%/yr dividend paid while short, n≈2
rate episodes) remain but are tolerable at 0.25x paired with the sleeve.

**Revised recommendation:** if a short leg is wanted, ship
**GLD/short-duration 15% sleeve + real-yield-gated DLR+EQIX short at 0.25x** — biggest
drawdown reduction of anything tested (−31→−26), beta 0.76→0.57, Sharpe held, positive in
the feared scenario, 3.3%/yr through-cycle drag. Disclose the rate-rising-bull drag.

## Follow-up (2026-09-06): crowding fixes — quality broadening + valuation overlay

Throwaway: `scratchpad/crowding_probe.py`.

### Quality broadening — PASSES
Marquee 9 (ETN HUBB GEV VRT PWR MYRG NVT FLNC PRIM) vs broadened 20 (drop FLNC for
chronic losses; add established profitable EMR AME RRX POWL ATKR AEIS GNRC EME FIX MTZ
TT AYI). 2017-06→2026-08, 20% vol-targeted:

| | marquee 9 | broadened 20 |
|---|---|---|
| CAGR | 22.1% | 22.4% |
| Sharpe | 0.84 | 0.86 |
| MaxDD | −31.2% | −27.1% |
| top name weight | 11% | 5% |
| effective #names | 9 | 20 |
| 2023-26 Sharpe | 1.54 | 1.47 |

Keeps ~all the return, 4pp shallower MaxDD, every episode drawdown equal-or-better, top
concentration halved. Costs a sliver of the 2023-26 hot streak (diluting VRT/GEV).
**Contrast with the failed 2026-09-01 "un-crowd" test** — that one added speculative
names (BE, ITRI, BWXT) and underperformed with a −52% DD; keeping a strict profitability
screen is what makes this version work. Corr vs VOLT rises 0.83→0.89 (a wider basket
tracks the theme factor more tightly) — the anti-crowding value is concentration/
robustness, not de-correlation. The 20 names are illustrative; a live version needs a
codified screen (sub-industry map + positive trailing net income & FCF + ETF-membership).

### Valuation / extension de-risk overlay — mild PASS
Monthly multiplier (never levered): `m = 1.0` if book <15% above its 200d MA and has
outrun SPY <25% over 12m; `m = 0.6` if >35% above MA or >50% vs SPY; else `m = 0.8`.
Freed capital at rf; known at prior month-end. Plateau-stable over ±20% thresholds.

- Marquee 9: MaxDD −31.2→−28.5 (+2.7pp, just under the 3pp bar), CAGR drag 2.5%/yr,
  Sharpe −0.03, avg multiplier 0.89, no sub-window Sharpe hit.
- Broadened 20: MaxDD unchanged, but Sharpe +0.01 and 2023-26 sub-Sharpe +0.11 (trims
  the 2024 top). CAGR drag 1.3%/yr.
- It's a graduated "sell-when-euphoric" trim, not a crash shield — much better-behaved
  than the binary layer-1 trend gate (`overlay.py`, "underperforms every year except
  2025"). Keep it, don't oversell it.

### Full stack (broadened + valuation overlay + hedge overlay)
`0.85×(broadened, vol-tgt, valuation-trimmed) + 0.15 GLD/IEF − 0.25× (DLR+EQIX | S1)`:
CAGR 17.9% / Sharpe 0.86 (held) / **MaxDD −31→−22** / SPY β 0.76→0.57 / feared-scenario
P&L +17% / Rate-22 episode −28→−17, DeepSeek −24→−14. ~4%/yr total CAGR cost to cut the
worst-case drawdown by a third and nearly halve market beta, at unchanged Sharpe.
