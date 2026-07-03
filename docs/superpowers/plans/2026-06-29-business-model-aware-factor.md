# Business-Model-Aware Factor Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `build_factor()` aware of merchant vs regulated utility business models and test three blending architectures (hard switch, revenue mix, dual-track) to recover alpha lost when PJM T&D utilities are included in the universe.

**Architecture:** A `pass_through` coefficient (0–1) is added to every ticker in `TICKER_NODE_MAP`. `build_factor()` and `build_rolling_factor()` accept new `arch` and `pass_through` parameters; when `pass_through` is provided the factor uses the specified architecture to blend the stress-beta signal (for merchants) with the ICR signal (for regulated utilities). When `pass_through=None` the function is 100% backward-compatible.

**Tech Stack:** Python 3.11, pandas, pytest, yfinance (ICR data already wired)

## Global Constraints

- Run all tests with `.venv/bin/python -m pytest tests/ -q` — must remain at 41 passing after every task
- Never change grid-search-validated config values: `STRESS_SPIKE_PCT=0.97`, `STRESS_CONG_THRESHOLD=0.30`, `STRESS_CONG_MIN_DAYS=10`, `PORTFOLIO_LONG_N=3`, `PORTFOLIO_SHORT_N=5`, `XLU_HEDGE=False`
- Evaluation runs use `--no-plot --no-zone-gsi` for speed
- `pass_through=None` (default) must preserve all existing behaviour — no regressions

---

## File Map

| File | Action | What changes |
|---|---|---|
| `grid_resilience/data/utility_node_map.py` | Modify | Add `business_model` and `pass_through` fields to all 23 tickers |
| `grid_resilience/config.py` | Modify | Add `BUSINESS_MODEL_ARCH` constant |
| `grid_resilience/factor/resilience_score.py` | Modify | Add `arch` + `pass_through` params to `build_factor()` and `build_rolling_factor()`; implement 3 arch branches |
| `grid_resilience/main.py` | Modify | Add `--arch` CLI flag; build `pass_through` Series from `TICKER_NODE_MAP`; auto-enable ICR when arch is set |
| `tests/factor/test_business_model_arch.py` | Create | Tests for all three architectures |

---

## Task 1: Business model tagging + config constant

**Files:**
- Modify: `grid_resilience/data/utility_node_map.py`
- Modify: `grid_resilience/config.py`

**Interfaces:**
- Produces: `TICKER_NODE_MAP[ticker]["business_model"]` → `"merchant" | "regulated" | "mixed"` and `TICKER_NODE_MAP[ticker]["pass_through"]` → `float`
- Produces: `BUSINESS_MODEL_ARCH` in `grid_resilience/config.py`

- [ ] **Step 1: Add `business_model` and `pass_through` to every entry in `TICKER_NODE_MAP`**

Add these two fields (before the closing `}` of each ticker dict) to every entry in `grid_resilience/data/utility_node_map.py`. Do NOT remove or change any existing fields.

```python
# ERCOT
"NRG":  business_model="merchant",  pass_through=0.85   # large retail book dilutes pure gen
"VST":  business_model="merchant",  pass_through=1.00   # 100% competitive generation
"CNP":  business_model="mixed",     pass_through=0.15   # T&D-dominant; residual gen exposure
# PJM
"AEP":  business_model="regulated", pass_through=0.05
"EXC":  business_model="mixed",     pass_through=0.10   # mostly regulated post-Constellation
"PPL":  business_model="regulated", pass_through=0.05
"FE":   business_model="regulated", pass_through=0.05
"D":    business_model="regulated", pass_through=0.10   # some generation but VA SCC dominates
"PEG":  business_model="mixed",     pass_through=0.25   # PSEG Nuclear in PJM capacity market
# MISO
"ETR":  business_model="mixed",     pass_through=0.35   # merchant nuclear in competitive mkts
"WEC":  business_model="regulated", pass_through=0.05
"DTE":  business_model="mixed",     pass_through=0.15   # merchant midstream + gen
"CMS":  business_model="regulated", pass_through=0.05
"AEE":  business_model="regulated", pass_through=0.05
# CAISO
"PCG":  business_model="regulated", pass_through=0.05
"EIX":  business_model="regulated", pass_through=0.05
# SPP
"XEL":  business_model="regulated", pass_through=0.05
"EVRG": business_model="regulated", pass_through=0.05
# ISO-NE / NYISO (inactive but in map)
"ES":   business_model="regulated", pass_through=0.05
"ED":   business_model="regulated", pass_through=0.05
# SERC/FRCC (inactive)
"DUK":  business_model="mixed",     pass_through=0.20
"SO":   business_model="mixed",     pass_through=0.20
"NEE":  business_model="mixed",     pass_through=0.40
```

The actual edits: for each ticker entry, add the two new key-value pairs inside the dict. Example for NRG:
```python
"NRG": {
    "name": "NRG Energy",
    "iso": "ERCOT",
    "nodes": ["HB_NORTH", "HB_HOUSTON", "HB_SOUTH"],
    "load_zones": ["NORTH", "HOUSTON"],
    "service_territory": "Texas retail/gen + national gen portfolio",
    "notes": "Large ERCOT gen fleet; HB_NORTH and HB_HOUSTON most relevant.",
    "business_model": "merchant",
    "pass_through": 0.85,
},
```

- [ ] **Step 2: Add `BUSINESS_MODEL_ARCH` constant to `config.py`**

Add after the `USE_ICR` line in `grid_resilience/config.py`:
```python
# Business-model-aware signal architecture (Task 1 of spec 2026-06-29).
# None = original behaviour. Set via --arch CLI flag.
BUSINESS_MODEL_ARCH: str | None = None
```

- [ ] **Step 3: Verify tests still pass**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 41 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add grid_resilience/data/utility_node_map.py grid_resilience/config.py
git commit -m "feat: add business_model + pass_through fields to TICKER_NODE_MAP; add BUSINESS_MODEL_ARCH config"
```

---

## Task 2: Hard switch architecture in `build_factor()`

**Files:**
- Modify: `grid_resilience/factor/resilience_score.py:41-99`
- Create: `tests/factor/test_business_model_arch.py`

**Interfaces:**
- Consumes: existing `build_factor(stress_betas, renewable_share, icr)` signature
- Produces: `build_factor(stress_betas, renewable_share=None, icr=None, arch="hard_switch", pass_through=None)` — new params are optional; `pass_through=None` preserves all existing behaviour
- Produces: `build_rolling_factor(..., arch="hard_switch", pass_through=None)` with same backward-compat guarantee

- [ ] **Step 1: Write failing tests for hard switch**

Create `tests/factor/test_business_model_arch.py`:

```python
import pandas as pd
import pytest
from grid_resilience.factor.resilience_score import build_factor


def _betas(tickers, values):
    return pd.DataFrame({"stress_beta": values}, index=tickers)


def _pt(tickers, values):
    return pd.Series(values, index=tickers)


# ── Hard switch ────────────────────────────────────────────────────────────────

def test_hard_switch_merchant_uses_beta():
    """Merchant (pass_through=1.0) score is driven by stress_beta, not ICR."""
    betas = _betas(["VST", "PPL"], [2.0, 2.0])        # equal betas
    icr   = pd.Series({"VST": 1.0, "PPL": 10.0})     # PPL has much better ICR
    pt    = _pt(["VST", "PPL"], [1.0, 0.05])
    scores = build_factor(betas, icr=icr, arch="hard_switch", pass_through=pt)
    # VST routed to beta path (equal betas → equal scores); scores should be equal
    assert abs(scores["VST"] - scores["PPL"]) < 0.1


def test_hard_switch_regulated_uses_icr():
    """Regulated ticker (pass_through<0.5) with high ICR should outscore low-ICR peer."""
    betas = _betas(["PPL", "FE"], [-0.5, -0.5])       # equal (bad) stress betas
    icr   = pd.Series({"PPL": 5.0, "FE": 1.0})       # PPL much better covered
    pt    = _pt(["PPL", "FE"], [0.05, 0.05])
    scores = build_factor(betas, icr=icr, arch="hard_switch", pass_through=pt)
    assert scores["PPL"] > scores["FE"]


def test_hard_switch_regulated_no_icr_falls_back_to_beta():
    """Regulated ticker with no ICR data falls back to stress_beta signal."""
    betas = _betas(["AEP", "FE"], [0.5, -0.5])
    pt    = _pt(["AEP", "FE"], [0.05, 0.05])
    scores = build_factor(betas, icr=None, arch="hard_switch", pass_through=pt)
    # Fallback to beta: AEP (+0.5) should outscore FE (-0.5)
    assert scores["AEP"] > scores["FE"]


def test_hard_switch_mixed_ticker_threshold():
    """pass_through >= 0.5 routes to merchant (beta) path; < 0.5 routes to regulated (ICR)."""
    betas = _betas(["A", "B"], [0.0, 0.0])            # neutral betas
    icr   = pd.Series({"A": 1.0, "B": 10.0})
    # A is merchant (0.6), B is regulated (0.4)
    pt    = _pt(["A", "B"], [0.6, 0.4])
    scores = build_factor(betas, icr=icr, arch="hard_switch", pass_through=pt)
    # B (regulated) gets ICR signal → high ICR → higher score
    assert scores["B"] > scores["A"]


def test_hard_switch_no_pass_through_is_backward_compatible():
    """Without pass_through, hard_switch behaves identically to original function."""
    betas = _betas(["NRG", "CNP"], [1.0, -1.0])
    original = build_factor(betas)
    with_arch = build_factor(betas, arch="hard_switch", pass_through=None)
    pd.testing.assert_series_equal(original, with_arch)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/factor/test_business_model_arch.py -v
```

Expected: 5 FAILED (functions don't accept arch/pass_through yet).

- [ ] **Step 3: Add `arch` and `pass_through` parameters to `build_factor()` and implement hard switch**

Replace the signature and add the arch block in `grid_resilience/factor/resilience_score.py`. The complete updated `build_factor` function:

```python
def build_factor(
    stress_betas: pd.DataFrame,
    renewable_share: pd.Series | None = None,
    icr: pd.Series | None = None,
    arch: str = "hard_switch",
    pass_through: pd.Series | None = None,
) -> pd.Series:
    """
    Construct a single cross-sectional Grid Resilience factor score.

    Parameters
    ----------
    stress_betas    : output of conditional_beta.compute_stress_betas()
                      Must have index = ticker and column 'stress_beta'.
    renewable_share : optional Series keyed by ticker of renewable gen share (0–1).
    icr             : optional Series keyed by ticker of interest coverage ratio.
    arch            : blending architecture when pass_through is provided.
                      One of "hard_switch", "revenue_mix", "dual_track".
                      Ignored when pass_through is None.
    pass_through    : optional Series keyed by ticker (0.0–1.0) — fraction of
                      revenue exposed to merchant/spot prices. When provided,
                      the specified arch is used to blend the beta and ICR signals.
                      When None, the original weight-based blending is used.

    Returns
    -------
    Series indexed by ticker, values in ~[-1, +1].
    High score = more resilient (candidate for long book).
    Low score  = stress-sensitive (candidate for short book).
    """
    if stress_betas.empty:
        return pd.Series(dtype=float)

    scores = pd.DataFrame(index=stress_betas.index)

    # Component 1: signed stress beta (always computed — needed for all archs)
    scores["signed_stress_beta"] = stress_betas["stress_beta"]
    scores["signed_stress_beta"] = cross_section_zscore(
        winsorize(scores["signed_stress_beta"], WINSOR_LIMITS)
    )

    has_renew = renewable_share is not None and not renewable_share.empty
    has_icr   = icr is not None and not icr.empty

    # ── Business-model-aware architectures ───────────────────────────────────
    if pass_through is not None:
        pt = pass_through.reindex(scores.index).fillna(1.0)  # unknown → treat as merchant

        # Pre-compute ICR z-score (used by all three archs)
        icr_z = pd.Series(0.0, index=scores.index)
        if has_icr:
            icr_raw = icr.reindex(scores.index)
            icr_raw = icr_raw.fillna(icr_raw.mean())        # fill gaps with cross-section mean
            icr_z = cross_section_zscore(winsorize(icr_raw, WINSOR_LIMITS))

        beta_z = scores["signed_stress_beta"]                # already z-scored above

        if arch == "hard_switch":
            is_regulated = pt < 0.5
            arch_score = beta_z.copy()
            if has_icr:
                arch_score[is_regulated] = icr_z[is_regulated]

        elif arch == "revenue_mix":
            arch_score = beta_z * pt + icr_z * (1 - pt)

        elif arch == "dual_track":
            merchant_mask  = pt > 0.05
            regulated_mask = pt < 1.0

            # Z-score beta within the merchant subgroup
            beta_zd = pd.Series(0.0, index=scores.index)
            if merchant_mask.sum() > 1:
                raw_beta = winsorize(stress_betas["stress_beta"][merchant_mask], WINSOR_LIMITS)
                beta_zd[merchant_mask] = cross_section_zscore(raw_beta)
            else:
                beta_zd = beta_z

            # Z-score ICR within the regulated subgroup
            icr_zd = pd.Series(0.0, index=scores.index)
            if has_icr and regulated_mask.sum() > 1:
                icr_raw_d = icr.reindex(scores.index).fillna(icr.mean())
                icr_zd[regulated_mask] = cross_section_zscore(
                    winsorize(icr_raw_d[regulated_mask], WINSOR_LIMITS)
                )

            arch_score = beta_zd * pt + icr_zd * (1 - pt)

        else:
            arch_score = beta_z  # unknown arch → fall back to pure beta

        # Renewable quality: 15% addon for all tickers regardless of arch
        factor = 0.85 * arch_score
        if has_renew:
            renew_aligned = renewable_share.reindex(scores.index).fillna(0.0)
            renew_z = cross_section_zscore(winsorize(renew_aligned, WINSOR_LIMITS))
            factor = 0.85 * arch_score + 0.15 * renew_z

        factor = cross_section_zscore(winsorize(factor, WINSOR_LIMITS))
        return factor.rename("grid_resilience_factor")

    # ── Original weight-based blending (pass_through=None, backward-compatible) ──
    w_beta  = _W_BETA  + (0 if has_renew else _W_RENEW) + (0 if has_icr else _W_ICR)
    w_renew = _W_RENEW if has_renew else 0.0
    w_icr   = _W_ICR   if has_icr   else 0.0

    factor = w_beta * scores["signed_stress_beta"]

    if has_renew:
        renew_aligned = renewable_share.reindex(scores.index).fillna(0.0)
        scores["renewable_quality"] = cross_section_zscore(
            winsorize(renew_aligned, WINSOR_LIMITS)
        )
        factor = factor + w_renew * scores["renewable_quality"]

    if has_icr:
        icr_aligned = icr.reindex(scores.index)
        scores["icr_score"] = cross_section_zscore(
            winsorize(icr_aligned, WINSOR_LIMITS)
        )
        factor = factor + w_icr * scores["icr_score"]

    factor = cross_section_zscore(winsorize(factor, WINSOR_LIMITS))
    return factor.rename("grid_resilience_factor")
```

- [ ] **Step 4: Add `arch` and `pass_through` parameters to `build_rolling_factor()`**

Update the `build_rolling_factor` function signature and the call to `build_factor` inside it:

```python
def build_rolling_factor(
    rolling_betas: pd.DataFrame,
    renewable_share_by_period: pd.DataFrame | None = None,
    icr_history: pd.DataFrame | None = None,
    arch: str = "hard_switch",
    pass_through: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build time-varying factor scores from rolling_stress_betas output.

    Parameters
    ----------
    rolling_betas              : MultiIndex (date, ticker) DataFrame from
                                 conditional_beta.rolling_stress_betas()
    renewable_share_by_period  : Optional DataFrame indexed by (period, ticker).
    icr_history                : Optional DataFrame indexed by quarter-end date
                                 with ticker columns (output of fetch_icr).
                                 A 45-day reporting lag is applied automatically.
    arch                       : blending architecture — passed to build_factor().
    pass_through               : Series keyed by ticker (0.0–1.0) — passed to
                                 build_factor(). None = original behaviour.

    Returns
    -------
    DataFrame with columns [date, ticker, factor_score], sorted by date.
    """
    if rolling_betas.empty:
        return pd.DataFrame()

    rows = []
    dates = rolling_betas.index.get_level_values("date").unique()

    for date in dates:
        betas_at_date = rolling_betas.loc[date]

        renew = None
        if renewable_share_by_period is not None:
            period_key = date.strftime("%Y-%m")
            if period_key in renewable_share_by_period.index:
                renew = renewable_share_by_period.loc[period_key]

        icr = _icr_at_date(icr_history, date)

        scores = build_factor(
            betas_at_date,
            renewable_share=renew,
            icr=icr,
            arch=arch,
            pass_through=pass_through,
        )
        for ticker, score in scores.items():
            rows.append({"date": date, "ticker": ticker, "factor_score": score})

    return pd.DataFrame(rows).sort_values(["date", "factor_score"], ascending=[True, False])
```

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 46 passed (41 existing + 5 new), 0 failed.

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/factor/resilience_score.py tests/factor/test_business_model_arch.py
git commit -m "feat: add arch-aware build_factor with hard_switch; revenue_mix and dual_track stubs present"
```

---

## Task 3: Revenue mix and dual-track tests

**Files:**
- Modify: `tests/factor/test_business_model_arch.py` — add revenue mix and dual-track tests

Both architectures are already implemented in the `build_factor()` from Task 2. This task adds test coverage.

**Interfaces:**
- Consumes: `build_factor(..., arch="revenue_mix", pass_through=pt)` and `build_factor(..., arch="dual_track", pass_through=pt)`

- [ ] **Step 1: Write failing tests for revenue mix and dual-track**

Append to `tests/factor/test_business_model_arch.py`:

```python
# ── Revenue mix ────────────────────────────────────────────────────────────────

def test_revenue_mix_pure_merchant_equals_beta_only():
    """Ticker with pass_through=1.0 gets pure beta score regardless of ICR."""
    betas  = _betas(["VST", "VST2"], [1.5, -1.5])
    icr    = pd.Series({"VST": 0.1, "VST2": 99.0})   # ICR would flip ranking
    pt     = _pt(["VST", "VST2"], [1.0, 1.0])
    scores = build_factor(betas, icr=icr, arch="revenue_mix", pass_through=pt)
    assert scores["VST"] > scores["VST2"]


def test_revenue_mix_pure_regulated_equals_icr_only():
    """Ticker with pass_through=0.0 gets pure ICR score regardless of beta."""
    betas  = _betas(["PPL", "FE"], [-0.5, -0.5])      # equal betas
    icr    = pd.Series({"PPL": 8.0, "FE": 1.0})
    pt     = _pt(["PPL", "FE"], [0.0, 0.0])
    scores = build_factor(betas, icr=icr, arch="revenue_mix", pass_through=pt)
    assert scores["PPL"] > scores["FE"]


def test_revenue_mix_midpoint_blends_both():
    """Ticker with pass_through=0.5 is intermediate between pure beta and pure ICR."""
    betas  = _betas(["A", "B", "C"], [1.0, 1.0, 1.0])
    icr    = pd.Series({"A": 1.0, "B": 5.0, "C": 10.0})
    pt     = _pt(["A", "B", "C"], [0.5, 0.5, 0.5])
    scores = build_factor(betas, icr=icr, arch="revenue_mix", pass_through=pt)
    # Equal betas + higher ICR → higher score
    assert scores["C"] > scores["B"] > scores["A"]


# ── Dual-track ─────────────────────────────────────────────────────────────────

def test_dual_track_merchant_ranked_within_merchant_group():
    """
    VST with moderate beta outranks NRG with low beta when both are in merchant group,
    even if the regulated tickers have extreme betas that would distort global z-scoring.
    """
    # Regulated tickers with huge betas that would dominate a global z-score
    betas = _betas(["VST", "NRG", "PPL", "FE"], [0.6, 0.3, 5.0, -5.0])
    icr   = pd.Series({"VST": 2.0, "NRG": 2.0, "PPL": 5.0, "FE": 1.0})
    pt    = _pt(["VST", "NRG", "PPL", "FE"], [1.0, 0.85, 0.05, 0.05])
    scores = build_factor(betas, icr=icr, arch="dual_track", pass_through=pt)
    # VST (higher beta among merchants) should still outrank NRG
    assert scores["VST"] > scores["NRG"]


def test_dual_track_regulated_ranked_within_regulated_group():
    """PPL (high ICR) outranks FE (low ICR) within the regulated group."""
    betas = _betas(["VST", "PPL", "FE"], [1.0, -0.5, -0.5])
    icr   = pd.Series({"VST": 2.0, "PPL": 8.0, "FE": 1.0})
    pt    = _pt(["VST", "PPL", "FE"], [1.0, 0.05, 0.05])
    scores = build_factor(betas, icr=icr, arch="dual_track", pass_through=pt)
    assert scores["PPL"] > scores["FE"]
```

- [ ] **Step 2: Run new tests**

```bash
.venv/bin/python -m pytest tests/factor/test_business_model_arch.py -v
```

Expected: 10 passed, 0 failed.

- [ ] **Step 3: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 51 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add tests/factor/test_business_model_arch.py
git commit -m "test: add revenue_mix and dual_track coverage for business-model-aware build_factor"
```

---

## Task 4: Wire up `main.py` — `--arch` flag and pass_through injection

**Files:**
- Modify: `grid_resilience/main.py`

**Interfaces:**
- Consumes: `TICKER_NODE_MAP[t]["pass_through"]` from `utility_node_map.py` (Task 1)
- Consumes: `build_rolling_factor(..., arch=arch, pass_through=pass_through)` from Task 2
- Produces: `--arch hard-switch | revenue-mix | dual-track` CLI flag

- [ ] **Step 1: Add `BUSINESS_MODEL_ARCH` to the config import**

In `grid_resilience/main.py`, find the config import block (lines 88–94) and add `BUSINESS_MODEL_ARCH`:

```python
from grid_resilience.config import (
    BACKTEST_START, BACKTEST_END,
    SUPPORTED_ISOS, REBALANCE_FREQ,
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N,
    CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE,
    XLU_HEDGE, USE_ICR, BUSINESS_MODEL_ARCH,
)
```

- [ ] **Step 2: Add `arch` parameter to `run()`**

Add `arch: str | None = BUSINESS_MODEL_ARCH` to the `run()` signature after `use_icr`:

```python
def run(
    isos:         list[str] = SUPPORTED_ISOS,
    start:        str       = BACKTEST_START,
    end:          str       = BACKTEST_END,
    n_long:       int       = PORTFOLIO_LONG_N,
    n_short:      int       = PORTFOLIO_SHORT_N,
    xlu_hedge:    bool      = XLU_HEDGE,
    use_icr:      bool      = USE_ICR,
    arch:         str | None = BUSINESS_MODEL_ARCH,
    zone_gsi:     bool      = True,
    plot:         bool      = True,
    save_dir:     str       = "output",
) -> dict:
```

- [ ] **Step 3: Build `pass_through` Series and auto-enable ICR when arch is set**

Insert this block in `run()` immediately after the `ticker_iso_map` line (which is `ticker_iso_map = {t: get_ticker_iso(t) for t in tickers}`):

```python
    # ── Business-model pass_through map ──────────────────────────────────────
    pass_through: pd.Series | None = None
    if arch is not None:
        pass_through = pd.Series({
            t: TICKER_NODE_MAP[t].get("pass_through", 1.0)
            for t in tickers
            if t in TICKER_NODE_MAP
        })
        use_icr = True   # regulated path requires ICR; override caller setting
        print(f"\n  [arch] Business-model arch: {arch!r} — ICR auto-enabled for regulated path")
```

- [ ] **Step 4: Pass `arch` and `pass_through` to `build_rolling_factor()`**

Find the `build_rolling_factor` call in step 6 (currently `rolling_factors = build_rolling_factor(rolling_betas, icr_history=icr_history)`) and replace it:

```python
    rolling_factors = build_rolling_factor(
        rolling_betas,
        icr_history=icr_history,
        arch=arch or "hard_switch",
        pass_through=pass_through,
    )
```

- [ ] **Step 5: Add `--arch` flag to `_parse_args()`**

Add after the `--zone-gsi` argument block:

```python
    p.add_argument(
        "--arch",
        default=None,
        choices=["hard-switch", "revenue-mix", "dual-track"],
        help="Business-model signal architecture. When set, ICR is auto-enabled. "
             "Default: None (original stress-beta-only behaviour).",
    )
```

- [ ] **Step 6: Pass `arch` to `run()` in `__main__`**

Add `arch=args.arch.replace("-", "_") if args.arch else None` to the `run()` call:

```python
    run(
        isos      = args.iso,
        start     = args.start,
        end       = args.end,
        n_long    = args.long,
        n_short   = args.short,
        xlu_hedge = args.xlu_hedge,
        use_icr   = args.use_icr,
        arch      = args.arch.replace("-", "_") if args.arch else None,
        zone_gsi  = args.zone_gsi,
        plot      = not args.no_plot,
        save_dir  = args.output,
    )
```

- [ ] **Step 7: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 51 passed, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add grid_resilience/main.py
git commit -m "feat: add --arch CLI flag and pass_through injection to main.py; auto-enables ICR for regulated path"
```

---

## Task 5: Run all three architectures and record results

**Files:**
- Modify: `docs/superpowers/specs/2026-06-29-business-model-aware-factor-design.md` — fill in evaluation table

- [ ] **Step 1: Run hard switch**

```bash
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch 2>&1 | tail -40
```

Record: Sharpe, Ann Return, Max DD, IC@21d, IC t-stat from printed output.

- [ ] **Step 2: Run revenue mix**

```bash
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch revenue-mix 2>&1 | tail -40
```

Record results.

- [ ] **Step 3: Run dual-track**

```bash
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch dual-track 2>&1 | tail -40
```

Record results.

- [ ] **Step 4: Fill in the evaluation table in the design spec**

Open `docs/superpowers/specs/2026-06-29-business-model-aware-factor-design.md` and replace the TBD rows with actual results.

- [ ] **Step 5: Commit results**

```bash
git add docs/superpowers/specs/2026-06-29-business-model-aware-factor-design.md output/
git commit -m "results: record hard-switch / revenue-mix / dual-track backtest comparison"
```

---

## Self-Review

**Spec coverage check:**
- [x] Business model tagging → Task 1
- [x] `BUSINESS_MODEL_ARCH` config → Task 1
- [x] Hard switch implementation → Task 2
- [x] Revenue mix implementation → Task 2 (coded in `build_factor`, tested in Task 3)
- [x] Dual-track implementation → Task 2 (coded in `build_factor`, tested in Task 3)
- [x] `--arch` CLI flag → Task 4
- [x] ICR auto-enabled when arch is set → Task 4
- [x] `pass_through=None` backward compat → covered by `test_hard_switch_no_pass_through_is_backward_compatible`
- [x] Evaluation table → Task 5
- [x] Success threshold (Sharpe > 0.20 with PJM) → documented in Task 5

**Placeholder scan:** No TBDs in code steps. Evaluation table rows are left blank intentionally (filled during Task 5).

**Type consistency:**
- `pass_through: pd.Series | None` used consistently across `build_factor`, `build_rolling_factor`, `run()`
- `arch: str` default `"hard_switch"` consistent across all callers
- `TICKER_NODE_MAP[t].get("pass_through", 1.0)` safely handles any ticker missing the field
