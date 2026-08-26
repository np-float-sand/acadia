from __future__ import annotations

import pandas as pd


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across this layer's own tickers (its subpopulation),
    never mixed with another layer's raw scale before scoring.

    NaN in, NaN out (changed 2026-08-26, whole-branch review finding #3):
    a NaN input cell means "this layer has no information about this ticker
    on this date" and must stay NaN so a lower-precedence layer can supply
    the cell in combine_dc_demand_layers(). Only cells that DID have a real
    input value but produced a NaN z-score are filled with 0.0 — that
    happens on a zero-spread row (e.g. exactly one non-NaN ticker in this
    layer on a given date), where "no cross-sectional dispersion to score
    against" legitimately means a neutral 0.

    Using float("nan") here rather than pandas' pd.NA keeps the division
    and the zero-spread fill entirely within plain float64 arithmetic —
    pd.NA is a nullable-dtype sentinel, and dividing a float64 array by a
    Series containing it silently upcasts the result to dtype=object, which
    a subsequent fillna does NOT downcast back. That object dtype then
    survives unnoticed all the way to a much later, much more confusing
    failure at a boolean-mask assignment in resilience_score.build_factor
    (see 2026-08-26 dc-multi wiring investigation).
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, float("nan"))
    z = df.sub(mean, axis=0).div(std, axis=0)
    return z.mask(df.notna() & z.isna(), 0.0)


def _union_mask(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Boolean OR of two boolean frames over the union of their axes."""
    index = a.index.union(b.index)
    columns = a.columns.union(b.columns)
    return (
        a.reindex(index=index, columns=columns, fill_value=False)
        | b.reindex(index=index, columns=columns, fill_value=False)
    )


def apply_layer_precedence(layers: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Trim each DataFrame in `layers` (given HIGHEST precedence first) down to
    a non-overlapping, real-coverage-only set of (date, ticker) CELLS, ready
    to hand to combine_dc_demand_layers().

    Each of compute_dc_load_signal() / compute_ercot_signal() /
    compute_hyperscaler_signal() returns a column for every universe
    ticker (NaN where that layer doesn't cover it), not just the tickers
    it actually covers. combine_dc_demand_layers() merges the layers
    cell-by-cell in precedence order, which is only well-defined if no two
    layers claim the same (date, ticker) — e.g. TICKER_NODE_MAP["CEG"] and
    ["TLN"] are both iso="PJM", so the PJM generation-queue layer has real
    coverage for them that collides with the hyperscaler disclosed-deal
    layer's coverage of the same two tickers.

    Precedence is applied PER DATE, not per whole column (changed
    2026-08-26, whole-branch review finding #3). The earlier column-level
    rule claimed a ticker for a layer if that layer had a value ANYWHERE in
    the column, so the hyperscaler layer — which used to return 0.0 rather
    than NaN before a ticker's first disclosure — claimed CEG/TLN for every
    rebalance date back to 2018 and silently displaced the PJM
    generation-queue layer's real, dispersed data for the whole
    pre-disclosure period. Per-date precedence means the hyperscaler layer
    only wins the dates on which it actually has post-disclosure
    information, and the PJM proxy keeps the rest.

    Parameters
    ----------
    layers : list of DataFrames, ordered highest-precedence first.

    Returns
    -------
    A list the same length and order as `layers`; each entry is that
    layer's input DataFrame with every cell an earlier/higher-precedence
    layer already claimed set to NaN, and any column left entirely NaN
    dropped (so a layer may come back with zero columns). Because a ticker
    can be won by different layers on different dates, the SAME ticker may
    legitimately appear as a column in more than one returned layer — the
    per-cell coverage is what is guaranteed disjoint.
    """
    claimed: pd.DataFrame | None = None
    trimmed: list[pd.DataFrame] = []

    for layer in layers:
        if layer.empty or layer.columns.empty:
            trimmed.append(layer)
            continue

        real = layer.notna()
        if claimed is None:
            win = real
        else:
            already = claimed.reindex(index=layer.index, columns=layer.columns, fill_value=False)
            win = real & ~already

        kept = layer.where(win)
        kept = kept.loc[:, kept.notna().any()]
        trimmed.append(kept)

        claimed = real if claimed is None else _union_mask(claimed, real)

    return trimmed


def _check_disjoint_coverage(layers: list[pd.DataFrame]) -> None:
    """Enforce combine_dc_demand_layers()'s documented input contract.

    Raises (rather than asserts) so the contract survives `python -O`; this
    replaces the bare `assert not combined.columns.duplicated().any()` that
    used to sit in main.py's dc_multi branch (whole-branch review finding
    #7). It is a strictly stronger check: a duplicate output column was
    only the symptom of two layers covering the same ticker, and this
    catches the cause — two layers covering the same (date, ticker) cell —
    including partial-overlap cases the column check could not see.
    """
    if len(layers) < 2:
        return
    index = layers[0].index
    columns = layers[0].columns
    for layer in layers[1:]:
        index = index.union(layer.index)
        columns = columns.union(layer.columns)

    counts = None
    for layer in layers:
        present = layer.notna().reindex(index=index, columns=columns, fill_value=False).astype(int)
        counts = present if counts is None else counts + present

    overlapping = counts.columns[(counts > 1).any()].tolist()
    if overlapping:
        raise ValueError(
            "combine_dc_demand_layers() received layers that cover the same "
            f"(date, ticker) cell more than once: {overlapping}. Run "
            "apply_layer_precedence() on the layers first — this function "
            "merges cell-by-cell and cannot decide which layer should win."
        )


def combine_dc_demand_layers(
    pjm_history: pd.DataFrame,
    ercot_history: pd.DataFrame,
    hyperscaler_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the three DC-demand layers into one dates x tickers score.
    Each layer is z-scored within its own covered tickers before merging —
    concatenating raw values across layers before a single z-score is the
    bug already found and fixed once in fill_with_icr() (see spec).

    Merging is cell-by-cell in precedence order (hyperscaler disclosed
    deals > PJM generation-queue proxy > ERCOT TSP level), so one ticker
    may be sourced from different layers on different dates. Output columns
    are unique by construction.

    Callers whose layers may have overlapping real coverage for the same
    (date, ticker) — as pjm_history/hyperscaler_history do for CEG/TLN —
    MUST run apply_layer_precedence() first. This function does not decide
    precedence for unresolved overlaps; it raises ValueError instead.
    """
    ordered = [hyperscaler_history, pjm_history, ercot_history]   # highest precedence first
    layers = [df for df in ordered if not df.empty and not df.columns.empty]
    if not layers:
        return pd.DataFrame()

    _check_disjoint_coverage(layers)

    zscored = [_zscore_columns(df) for df in layers]
    combined = zscored[0]
    for z in zscored[1:]:
        combined = combined.combine_first(z)
    return combined.astype(float)
