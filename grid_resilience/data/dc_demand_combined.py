from __future__ import annotations

import pandas as pd


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across this layer's own tickers (its subpopulation),
    never mixed with another layer's raw scale before scoring.

    A zero-spread row (e.g. exactly one non-NaN ticker in this layer on a
    given date) divides by std=0. Using float("nan") here rather than
    pandas' pd.NA keeps the division and the final fillna(0.0) entirely
    within plain float64 arithmetic — pd.NA is a nullable-dtype sentinel,
    and dividing a float64 array by a Series containing it silently
    upcasts the result to dtype=object, which the subsequent .fillna(0.0)
    does NOT downcast back. That object dtype then survives unnoticed all
    the way to a much later, much more confusing failure at a boolean-mask
    assignment in resilience_score.build_factor (see 2026-08-26 dc-multi
    wiring investigation).
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, float("nan"))
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def apply_layer_precedence(layers: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Trim each DataFrame in `layers` (given HIGHEST precedence first) down to
    a non-overlapping, real-coverage-only column set, ready to hand to
    combine_dc_demand_layers().

    Each of compute_dc_load_signal() / compute_ercot_signal() /
    compute_hyperscaler_signal() returns a column for every universe
    ticker (NaN where that layer doesn't cover it), not just the tickers
    it actually covers. combine_dc_demand_layers()'s plain
    pd.concat(axis=1) does not merge same-named columns, so feeding it
    the raw, full-universe layer outputs directly would triplicate every
    ticker's column — not just the tickers genuinely double-covered (e.g.
    TICKER_NODE_MAP["CEG"]/["TLN"] are both iso="PJM", so the PJM
    generation-queue layer has real coverage for them that collides with
    the hyperscaler disclosed-deal layer's coverage of the same two
    tickers).

    For each ticker, only the highest-precedence layer with real
    (non-all-NaN) coverage keeps that column; lower-precedence layers
    have it dropped, even if they also have real data for it.

    Parameters
    ----------
    layers : list of DataFrames, ordered highest-precedence first.

    Returns
    -------
    A list the same length and order as `layers`; each entry is that
    layer's input DataFrame restricted to the tickers it "won" (may have
    zero columns if every ticker it covers was claimed by an
    earlier/higher-precedence layer).
    """
    claimed: set[str] = set()
    trimmed = []
    for layer in layers:
        covered = [t for t in layer.columns if layer[t].notna().any()]
        winners = [t for t in covered if t not in claimed]
        claimed.update(winners)
        trimmed.append(layer[winners])
    return trimmed


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

    Callers whose layers may have overlapping real coverage for the same
    ticker (as pjm_history/hyperscaler_history do for CEG/TLN) MUST run
    apply_layer_precedence() first — this function does not deduplicate;
    pd.concat(axis=1) does not merge same-named columns, so an unresolved
    overlap reaches the output as two same-named columns.
    """
    layers = [df for df in (pjm_history, ercot_history, hyperscaler_history) if not df.empty and not df.columns.empty]
    if not layers:
        return pd.DataFrame()
    zscored = [_zscore_columns(df) for df in layers]
    return pd.concat(zscored, axis=1)
