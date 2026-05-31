# Signal Fixes & Universe Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the generator beta sign so generators go long, add an XLU hedge flag (default on) to replace the short book, and fix the MISO/CAISO data fetch bug that collapses the universe to 3 stocks.

**Architecture:** Three independent changes — factor sign logic (resilience_score.py), portfolio construction flag (construction.py + main.py + config.py), and grid data fetch (grid_data.py + config.py). All three can be executed in parallel. MISO fails because `get_lmp()` is called with `location_type=` but those ISOs only accept `market=`; the TypeError is silently caught and the ISO is skipped.

**Tech Stack:** Python 3.11, pandas, numpy, yfinance, gridstatus, pytest

---

## File Map

| File | What changes |
|---|---|
| `grid_resilience/factor/resilience_score.py` | Apply `+stress_beta` for generators, `-stress_beta` for everyone else |
| `grid_resilience/data/universe.py` | Already has `GENERATORS` list — no change needed |
| `grid_resilience/data/grid_data.py` | Add `_MARKET_PARAM_ISOS` constant; branch `_fetch_lmp_raw` to skip `location_type=` for MISO/CAISO |
| `grid_resilience/config.py` | Lower `MIN_STRESS_OBS` from 20 → 10 to capture SPP tickers (XEL, EVRG) |
| `grid_resilience/portfolio/construction.py` | Add `xlu_hedge: bool` to `build_weights` and `build_rolling_weights` |
| `grid_resilience/main.py` | Add `--xlu-hedge / --no-xlu-hedge` flag; fetch XLU returns; thread flag through pipeline |
| `tests/factor/test_resilience_score.py` | New — tests for generator sign flip |
| `tests/portfolio/test_construction.py` | New — tests for XLU hedge weights |
| `tests/data/test_grid_data_fetch.py` | New — tests for MISO/CAISO fetch branch |

---

## Task 1: Fix Generator Beta Sign

**Files:**
- Modify: `grid_resilience/factor/resilience_score.py`
- Create: `tests/factor/test_resilience_score.py`

### Background

The factor currently computes `neg_stress_beta = -stress_beta` for every ticker.
For generators (NRG, VST, ETR, NEE), a *positive* stress_beta means their stock
rallies when LMPs spike — that is the desired behaviour (they profit from grid stress).
The negation inverts this and puts generators in the short book.
Fix: use `+stress_beta` for tickers in `GENERATORS`, `-stress_beta` for everything else.

- [ ] **Step 1: Create the test file with a failing test**

Create `tests/factor/__init__.py` (empty) and `tests/factor/test_resilience_score.py`:

```python
import pandas as pd
import pytest
from grid_resilience.factor.resilience_score import build_factor


def _betas(tickers, values):
    return pd.DataFrame({"stress_beta": values}, index=tickers)


def test_generator_positive_beta_ranks_above_td_negative_beta():
    """NRG (generator) with +beta should outrank CNP (T&D) with -beta."""
    betas = _betas(["NRG", "CNP"], [1.0, -1.0])
    scores = build_factor(betas)
    assert scores["NRG"] > scores["CNP"]


def test_generator_positive_beta_is_long_candidate():
    """Generator with positive beta (profits from LMP spikes) must score positive."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    assert scores["NRG"] > 0
    assert scores["VST"] > 0


def test_td_negative_beta_is_short_candidate():
    """T&D hurt by grid stress (negative beta) must score negative."""
    betas = _betas(["NRG", "VST", "CNP"], [0.8, 0.4, -0.6])
    scores = build_factor(betas)
    assert scores["CNP"] < 0


def test_generator_ranks_highest_among_three():
    """With one high-beta generator and two T&D names, generator tops the ranking."""
    betas = _betas(["NRG", "EXC", "CNP"], [1.5, -0.2, -1.0])
    scores = build_factor(betas)
    assert scores.idxmax() == "NRG"


def test_non_generator_ranking_unchanged():
    """Two non-generators with identical beta magnitudes: more negative = lower score."""
    betas = _betas(["EXC", "CNP"], [-0.5, -1.5])
    scores = build_factor(betas)
    assert scores["EXC"] > scores["CNP"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/sandhyapersad/acadia
.venv/bin/pytest tests/factor/test_resilience_score.py -v
```

Expected: all 5 FAIL (generator test gets wrong sign with current code).

- [ ] **Step 3: Implement the fix in resilience_score.py**

Open `grid_resilience/factor/resilience_score.py`. Add the import at the top of the import block:

```python
from grid_resilience.data.universe import GENERATORS
```

Replace the `build_factor` function's component-1 block (the `neg_stress_beta` lines) with:

```python
    # Component 1: signed stress beta
    # Generators (NRG, VST, ETR, NEE) profit from LMP spikes so high stress_beta
    # is *good* for them → use +stress_beta.
    # T&D and integrated utilities are exposed to stress costs → use -stress_beta.
    beta_sign = pd.Series(
        {t: 1.0 if t in GENERATORS else -1.0 for t in stress_betas.index}
    )
    scores["signed_stress_beta"] = beta_sign * stress_betas["stress_beta"]
    scores["signed_stress_beta"] = cross_section_zscore(
        winsorize(scores["signed_stress_beta"], WINSOR_LIMITS)
    )
```

Remove the old lines:
```python
    scores["neg_stress_beta"] = -stress_betas["stress_beta"]
    scores["neg_stress_beta"] = cross_section_zscore(
        winsorize(scores["neg_stress_beta"], WINSOR_LIMITS)
    )
```

Also update the component-2 conditional to use `scores["signed_stress_beta"]`:

```python
    if renewable_share is not None and not renewable_share.empty:
        renew_aligned = renewable_share.reindex(scores.index).fillna(0.0)
        scores["renewable_quality"] = cross_section_zscore(
            winsorize(renew_aligned, WINSOR_LIMITS)
        )
        factor = _W_BETA * scores["signed_stress_beta"] + _W_RENEW * scores["renewable_quality"]
    else:
        factor = scores["signed_stress_beta"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/factor/test_resilience_score.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/factor/resilience_score.py tests/factor/__init__.py tests/factor/test_resilience_score.py
git commit -m "fix: apply +stress_beta for generators, -stress_beta for T&D and integrated

Generators (NRG, VST, ETR, NEE) profit from LMP spikes so a positive
stress beta means the stock rallies during grid stress events.  The prior
sign inversion was placing these names in the short book, directly causing
the -4.3%/yr short-book drag identified in attribution analysis."
```

---

## Task 2: XLU Hedge Flag

**Files:**
- Modify: `grid_resilience/portfolio/construction.py`
- Modify: `grid_resilience/main.py`
- Modify: `grid_resilience/config.py`
- Create: `tests/portfolio/__init__.py` (empty)
- Create: `tests/portfolio/test_construction.py`

### Background

Replace the individual short book with a fixed -0.5 weight on XLU (SPDR Utilities ETF)
when `xlu_hedge=True` (the default). XLU's daily returns are fetched by yfinance
alongside other tickers; it is included in the weights matrix but excluded from factor
scoring. The flag can be turned off to restore the original L/S construction.

- [ ] **Step 1: Create test file with failing tests**

Create `tests/portfolio/__init__.py` (empty) and `tests/portfolio/test_construction.py`:

```python
import pandas as pd
import pytest
from grid_resilience.portfolio.construction import build_weights


def _scores(tickers=("VST", "NRG", "CNP"), values=(1.0, 0.0, -1.0)):
    return pd.Series(dict(zip(tickers, values)))


def test_xlu_hedge_true_no_individual_shorts():
    """When xlu_hedge=True, no ticker from factor_scores should have a negative weight."""
    scores = _scores()
    weights = build_weights(scores, xlu_hedge=True)
    for ticker in ["VST", "NRG", "CNP"]:
        assert weights.get(ticker, 0.0) >= 0.0


def test_xlu_hedge_true_xlu_weight_is_minus_half():
    """XLU hedge position must be exactly -0.5."""
    weights = build_weights(_scores(), xlu_hedge=True)
    assert weights["XLU"] == pytest.approx(-0.5)


def test_xlu_hedge_true_long_weights_sum_to_half():
    """Long book must still sum to +0.5."""
    weights = build_weights(_scores(), n_long=1, xlu_hedge=True)
    assert weights[weights > 0].sum() == pytest.approx(0.5)


def test_xlu_hedge_false_no_xlu_in_weights():
    """When xlu_hedge=False, XLU must not appear in the output."""
    weights = build_weights(_scores(), xlu_hedge=False)
    assert "XLU" not in weights.index


def test_xlu_hedge_false_short_weights_sum_to_minus_half():
    """Original L/S: short side sums to -0.5."""
    weights = build_weights(_scores(), n_short=2, xlu_hedge=False)
    assert weights[weights < 0].sum() == pytest.approx(-0.5)


def test_xlu_is_excluded_from_ranking():
    """XLU in factor_scores input must not be ranked into long positions."""
    scores = _scores(("VST", "NRG", "CNP", "XLU"), (1.0, 0.5, -0.5, 99.0))
    weights = build_weights(scores, n_long=1, xlu_hedge=True)
    assert weights.get("XLU", 0.0) == pytest.approx(-0.5)
    # VST should still be long (highest score among non-XLU names)
    assert weights.get("VST", 0.0) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/portfolio/test_construction.py -v
```

Expected: all FAIL (build_weights doesn't accept xlu_hedge yet).

- [ ] **Step 3: Add XLU_HEDGE default to config.py**

In `grid_resilience/config.py`, add after the PORTFOLIO block:

```python
# Replace short book with XLU sector-ETF hedge (default on)
# Set to False to restore original long/short individual-name construction
XLU_HEDGE: bool = True
```

- [ ] **Step 4: Update build_weights in construction.py**

In `grid_resilience/portfolio/construction.py`, update the `build_weights` function signature and body:

```python
from grid_resilience.config import PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N, REBALANCE_FREQ, XLU_HEDGE


def build_weights(
    factor_scores: pd.Series,
    n_long:    int  = PORTFOLIO_LONG_N,
    n_short:   int  = PORTFOLIO_SHORT_N,
    xlu_hedge: bool = XLU_HEDGE,
) -> pd.Series:
    """
    Construct dollar-neutral long/short weights from a factor score vector.

    When xlu_hedge=True (default), the short book is replaced by a single
    -0.5 position in XLU (SPDR Utilities ETF) to avoid idiosyncratic short risk.
    When xlu_hedge=False, the bottom n_short names are shorted as before.

    XLU is always excluded from ranking even if it appears in factor_scores.
    """
    scores = factor_scores.drop("XLU", errors="ignore").dropna().sort_values(ascending=False)

    if not xlu_hedge and len(scores) < n_long + n_short:
        n_long  = max(1, len(scores) // 2)
        n_short = max(1, len(scores) - n_long)
    elif xlu_hedge and len(scores) < n_long:
        n_long = max(1, len(scores))

    long_tickers = scores.iloc[:n_long].index
    weights = pd.Series(0.0, index=scores.index)
    weights[long_tickers] = 0.5 / n_long

    if xlu_hedge:
        weights["XLU"] = -0.5
    else:
        short_tickers = scores.iloc[-n_short:].index
        weights[short_tickers] = -0.5 / n_short

    return weights
```

Also update `build_rolling_weights` signature to pass `xlu_hedge` through:

```python
def build_rolling_weights(
    rolling_factors: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex | None = None,
    n_long:    int  = PORTFOLIO_LONG_N,
    n_short:   int  = PORTFOLIO_SHORT_N,
    xlu_hedge: bool = XLU_HEDGE,
) -> pd.DataFrame:
    if rebalance_dates is None:
        rebalance_dates = pd.DatetimeIndex(rolling_factors["date"].unique())

    rows = []
    for date in rebalance_dates:
        day_factors = rolling_factors[rolling_factors["date"] == date]
        if day_factors.empty:
            continue
        scores  = day_factors.set_index("ticker")["factor_score"]
        weights = build_weights(scores, n_long=n_long, n_short=n_short, xlu_hedge=xlu_hedge)
        for ticker, w in weights.items():
            if w != 0.0:
                rows.append({"date": date, "ticker": ticker, "weight": w})

    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/portfolio/test_construction.py -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Wire XLU into main.py**

In `grid_resilience/main.py`:

1. Add `XLU_HEDGE` to the config import at the top:

```python
from grid_resilience.config import (
    BACKTEST_START, BACKTEST_END,
    SUPPORTED_ISOS, REBALANCE_FREQ,
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N,
    CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE,
    XLU_HEDGE,
)
```

2. Add `xlu_hedge: bool = XLU_HEDGE` to `run()` signature.

3. After the `tickers` list is finalized (after the `returns = returns[tickers]` line), add XLU fetch:

```python
    # ── XLU hedge returns (fetched separately, not factor-scored) ────────────
    xlu_returns = pd.Series(dtype=float)
    if xlu_hedge:
        xlu_prices = fetch_prices(["XLU"], start, end)
        if not xlu_prices.empty and "XLU" in xlu_prices.columns:
            xlu_returns = np.log(xlu_prices["XLU"] / xlu_prices["XLU"].shift(1)).dropna()
            xlu_returns.name = "XLU"
```

4. Before the `weights_df = build_rolling_weights(...)` call, merge XLU into returns:

```python
    if xlu_hedge and not xlu_returns.empty:
        returns = returns.join(xlu_returns, how="left")
        tickers = list(returns.columns)
```

5. Pass `xlu_hedge` to `build_rolling_weights`:

```python
    weights_df = build_rolling_weights(
        rolling_factors,
        rebalance_dates=pd.DatetimeIndex(rolling_factors["date"].unique()),
        n_long=n_long,
        n_short=n_short,
        xlu_hedge=xlu_hedge,
    )
```

6. Pass updated `tickers` (now including XLU) to `weights_to_matrix`:

```python
    weights_matrix = weights_to_matrix(weights_df, returns.index, list(returns.columns))
```

- [ ] **Step 7: Wire XLU flag into CLI**

In `_parse_args()`, add after the `--short` argument:

```python
    p.add_argument(
        "--xlu-hedge", dest="xlu_hedge",
        action=argparse.BooleanOptionalAction,
        default=XLU_HEDGE,
        help="Replace short book with -0.5 XLU hedge (default: on)",
    )
```

And pass it to `run()` in `__main__`:

```python
    run(
        isos      = args.iso,
        start     = args.start,
        end       = args.end,
        n_long    = args.long,
        n_short   = args.short,
        xlu_hedge = args.xlu_hedge,
        plot      = not args.no_plot,
        save_dir  = args.output,
    )
```

- [ ] **Step 8: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add grid_resilience/config.py grid_resilience/portfolio/construction.py grid_resilience/main.py tests/portfolio/__init__.py tests/portfolio/test_construction.py
git commit -m "feat: add XLU hedge flag (default on) to replace short book

Replaces individual short positions with a fixed -0.5 XLU position,
removing idiosyncratic short risk.  Use --no-xlu-hedge to restore the
original long/short construction.  XLU is fetched via yfinance and
excluded from factor scoring."
```

---

## Task 3: Fix MISO/CAISO Data + Lower SPP Min Obs

**Files:**
- Modify: `grid_resilience/data/grid_data.py`
- Modify: `grid_resilience/config.py`
- Create: `tests/data/test_grid_data_fetch.py`

### Background

**Root cause (MISO and CAISO):** `_fetch_lmp_raw` calls
`iso_obj.get_lmp(date=start, end=end, location_type=loc_type, verbose=False)`.
MISO's signature is `(date, end, market, locations, verbose)` and CAISO's is
`(date, market, locations, sleep, end, verbose)`. Neither accepts `location_type=`,
so a `TypeError` is raised, silently caught, and the ISO returns empty data.
Fix: detect the parameter name via `inspect` and branch accordingly.

**SPP:** Only 83 high-stress days over 7 years (~12/year). With a 252-day
rolling window and `MIN_STRESS_OBS=20`, SPP tickers (XEL, EVRG) never get enough
observations. Lower to 10.

**PJM:** Requires a paid API key — out of scope here. Document as a note.

- [ ] **Step 1: Create failing test for the fetch branch**

Create `tests/data/test_grid_data_fetch.py`:

```python
"""
Tests that _fetch_lmp_raw uses the correct parameter for each ISO.
Uses a mock iso_obj to intercept the call without hitting the network.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from grid_resilience.data.grid_data import _fetch_lmp_raw


def _make_iso_obj(param_names):
    """Create a mock whose get_lmp() accepts only the given parameter names."""
    mock = MagicMock()
    mock.get_lmp.__name__ = "get_lmp"

    import inspect
    params = {n: inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
              for n in param_names}
    mock.get_lmp.__signature__ = inspect.Signature(list(params.values()))
    mock.get_lmp.return_value = pd.DataFrame()
    return mock


def test_ercot_passes_location_type():
    iso_obj = _make_iso_obj(["date", "end", "location_type", "verbose"])
    _fetch_lmp_raw(iso_obj, "ERCOT", "2024-01-01", "2024-01-31", "settlement point")
    call_kwargs = iso_obj.get_lmp.call_args.kwargs
    assert "location_type" in call_kwargs
    assert "market" not in call_kwargs


def test_miso_omits_location_type():
    iso_obj = _make_iso_obj(["date", "end", "market", "locations", "verbose"])
    _fetch_lmp_raw(iso_obj, "MISO", "2024-01-01", "2024-01-31", "LMP")
    call_kwargs = iso_obj.get_lmp.call_args.kwargs
    assert "location_type" not in call_kwargs


def test_caiso_omits_location_type():
    iso_obj = _make_iso_obj(["date", "market", "locations", "sleep", "end", "verbose"])
    _fetch_lmp_raw(iso_obj, "CAISO", "2024-01-01", "2024-01-31", "trading_hub")
    call_kwargs = iso_obj.get_lmp.call_args.kwargs
    assert "location_type" not in call_kwargs


def test_empty_return_on_exception():
    """If get_lmp raises, _fetch_lmp_raw returns an empty DataFrame (no re-raise)."""
    iso_obj = MagicMock()
    import inspect
    iso_obj.get_lmp.__signature__ = inspect.Signature([
        inspect.Parameter("date", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("end", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("location_type", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("verbose", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    iso_obj.get_lmp.side_effect = RuntimeError("network error")
    result = _fetch_lmp_raw(iso_obj, "ERCOT", "2024-01-01", "2024-01-31", "hub")
    assert result.empty
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/data/test_grid_data_fetch.py -v
```

Expected: `test_miso_omits_location_type` and `test_caiso_omits_location_type` FAIL
(current code passes `location_type=` for all ISOs).

- [ ] **Step 3: Fix _fetch_lmp_raw in grid_data.py**

Add `import inspect` to the imports at the top of `grid_resilience/data/grid_data.py`
(add after the existing `import os` line).

Replace the `_fetch_lmp_raw` function with:

```python
def _fetch_lmp_raw(iso_obj, iso: str, start: str, end: str, loc_type: str) -> pd.DataFrame:
    print(f"[grid] Fetching {iso} LMP {start} → {end}…")
    if iso == "SPP":
        return _fetch_spp_lmp_raw(iso_obj, start, end)
    try:
        params = inspect.signature(iso_obj.get_lmp).parameters
        if "location_type" in params:
            df = iso_obj.get_lmp(date=start, end=end, location_type=loc_type, verbose=False)
        else:
            df = iso_obj.get_lmp(date=start, end=end, verbose=False)
    except Exception as exc:
        print(f"  [grid] {iso} LMP fetch failed: {exc}")
        return pd.DataFrame()
    return _normalise_lmp_columns(df)
```

- [ ] **Step 4: Lower MIN_STRESS_OBS in config.py**

In `grid_resilience/config.py`, change:

```python
MIN_STRESS_OBS = 20
```

to:

```python
MIN_STRESS_OBS = 10
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/data/test_grid_data_fetch.py tests/data/test_grid_data_spread.py -v
```

Expected: all tests PASS including the original spread tests.

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add grid_resilience/data/grid_data.py grid_resilience/config.py tests/data/test_grid_data_fetch.py
git commit -m "fix: use inspect to pick location_type vs market for MISO/CAISO; lower MIN_STRESS_OBS to 10

MISO and CAISO get_lmp() accept 'market=' not 'location_type='.
The TypeError was silently caught, collapsing the universe to the 3
ERCOT tickers.  Now detected via inspect.signature so each ISO gets
the correct parameter.  MIN_STRESS_OBS lowered from 20 to 10 so SPP
tickers (XEL, EVRG) qualify in the 252-day rolling window (~12 stress
days/year exceeds the new threshold).

Note: PJM requires a PJM_API_KEY env var (free at pjm.com/api) and
is not covered by this fix."
```

---

## Notes

- **PJM** is excluded from this fix — it requires `export PJM_API_KEY=<key>` (free registration at pjm.com/api). Once set, PJM will work automatically (the `_get_iso("PJM")` branch already handles it). This adds AEP, EXC, PPL, FE, D to the universe.
- **After fixing MISO/CAISO**, verify that the benchmark hub node names in `ISO_BENCHMARK_NODES` match what gridstatus returns. MISO hub names (`ILLINOIS HUB`, `MICHIGAN HUB`, etc.) should be present in the returned data. If not, update the names in `config.py` to match gridstatus's naming convention.
- **Universe will grow** from 3 tickers to potentially 10–12 once MISO (5 tickers) and CAISO (2 tickers) are active, plus XEL/EVRG from SPP (2 tickers). This makes the cross-sectional z-scoring meaningful.

