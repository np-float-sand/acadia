import pandas as pd
import numpy as np


def test_combine_dc_demand_layers_zscores_each_layer_separately():
    from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers

    dates = [pd.Timestamp("2025-12-31")]
    pjm = pd.DataFrame({"AEP": [10.0], "FE": [20.0]}, index=dates)       # raw scale ~0-100s
    ercot = pd.DataFrame({"CNP": [34000.0]}, index=dates)                # raw scale ~10,000s
    hyperscaler = pd.DataFrame({"CEG": [835.0], "TLN": [1920.0]}, index=dates)

    result = combine_dc_demand_layers(pjm, ercot, hyperscaler)

    assert set(result.columns) == {"AEP", "FE", "CNP", "CEG", "TLN"}
    # single-column-per-layer or single-value populations z-score to 0 (no within-layer spread)
    assert result.loc[dates[0], "CNP"] == 0.0
    # two-ticker PJM population: AEP (lower) should be negative, FE positive
    assert result.loc[dates[0], "AEP"] < 0 < result.loc[dates[0], "FE"]


def test_combine_dc_demand_layers_ticker_with_no_layer_is_absent():
    from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers
    dates = [pd.Timestamp("2025-12-31")]
    pjm = pd.DataFrame({"AEP": [10.0]}, index=dates)
    ercot = pd.DataFrame(index=dates)
    hyperscaler = pd.DataFrame(index=dates)
    result = combine_dc_demand_layers(pjm, ercot, hyperscaler)
    assert "PCG" not in result.columns


def test_zscore_columns_zero_spread_row_stays_float64():
    """A layer with exactly one non-NaN ticker on a given date has std=0 for
    that row. _zscore_columns must not let this upcast the column to
    dtype=object (see 2026-08-26 dc-multi wiring investigation: an
    object-dtype column here silently corrupted a much later, unrelated
    boolean-mask assignment in resilience_score.build_factor)."""
    from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers

    dates = [pd.Timestamp("2025-12-31")]
    # Single non-NaN ticker in this layer -> zero cross-sectional spread on this row.
    hyperscaler = pd.DataFrame({"CEG": [500.0]}, index=dates)
    pjm = pd.DataFrame({"AEP": [10.0], "FE": [20.0]}, index=dates)
    ercot = pd.DataFrame(index=dates)

    result = combine_dc_demand_layers(pjm, ercot, hyperscaler)

    assert result["CEG"].dtype == np.float64
    assert result.loc[dates[0], "CEG"] == 0.0
    assert result.dtypes.eq(np.float64).all()


class TestApplyLayerPrecedence:
    """Covers grid_resilience.data.dc_demand_combined.apply_layer_precedence,
    the trimming/precedence logic previously an inline nested function in
    main.py's dc_multi branch. Precedence order in production usage is
    hyperscaler > PJM generation-queue proxy > ERCOT TSP level."""

    def _covered(self, df: pd.DataFrame) -> set[str]:
        return {t for t in df.columns if df[t].notna().any()}

    def test_no_overlap_all_columns_kept(self):
        from grid_resilience.data.dc_demand_combined import apply_layer_precedence

        dates = [pd.Timestamp("2025-12-31")]
        hyperscaler = pd.DataFrame({"CEG": [500.0], "VST": [300.0]}, index=dates)
        pjm = pd.DataFrame({"AEP": [10.0], "FE": [20.0]}, index=dates)
        ercot = pd.DataFrame({"CNP": [34000.0]}, index=dates)

        trimmed = apply_layer_precedence([hyperscaler, pjm, ercot])

        assert self._covered(trimmed[0]) == {"CEG", "VST"}
        assert self._covered(trimmed[1]) == {"AEP", "FE"}
        assert self._covered(trimmed[2]) == {"CNP"}

    def test_hyperscaler_pjm_overlap_hyperscaler_wins(self):
        """The real current case: CEG/TLN are covered with real data by both
        the PJM generation-queue layer and the hyperscaler layer."""
        from grid_resilience.data.dc_demand_combined import apply_layer_precedence

        dates = [pd.Timestamp("2025-12-31")]
        hyperscaler = pd.DataFrame({"CEG": [500.0], "TLN": [900.0]}, index=dates)
        pjm = pd.DataFrame(
            {"AEP": [10.0], "FE": [20.0], "CEG": [0.6], "TLN": [0.4]}, index=dates
        )
        ercot = pd.DataFrame(index=dates)

        hyperscaler_out, pjm_out, ercot_out = apply_layer_precedence([hyperscaler, pjm, ercot])

        assert self._covered(hyperscaler_out) == {"CEG", "TLN"}
        assert self._covered(pjm_out) == {"AEP", "FE"}
        assert self._covered(ercot_out) == set()
        # winner's own value is preserved untouched (pre-z-score)
        assert hyperscaler_out.loc[dates[0], "CEG"] == 500.0

    def test_pjm_ercot_overlap_pjm_wins(self):
        """Hypothetical: unexercised in the current live seed data (ERCOT's
        seed currently has zero real rows), but AEP is mapped in both
        TICKER_NODE_MAP (PJM) and ERCOT_TSP_MAP, so this combination is
        reachable in principle. PJM should win over ERCOT per the
        documented precedence: hyperscaler > PJM > ERCOT."""
        from grid_resilience.data.dc_demand_combined import apply_layer_precedence

        dates = [pd.Timestamp("2025-12-31")]
        hyperscaler = pd.DataFrame(index=dates)
        pjm = pd.DataFrame({"AEP": [10.0], "FE": [20.0]}, index=dates)
        ercot = pd.DataFrame({"AEP": [9500.0], "CNP": [34000.0]}, index=dates)

        hyperscaler_out, pjm_out, ercot_out = apply_layer_precedence([hyperscaler, pjm, ercot])

        assert self._covered(hyperscaler_out) == set()
        assert self._covered(pjm_out) == {"AEP", "FE"}
        assert self._covered(ercot_out) == {"CNP"}
        assert pjm_out.loc[dates[0], "AEP"] == 10.0

    def test_all_empty_layers(self):
        from grid_resilience.data.dc_demand_combined import apply_layer_precedence

        dates = [pd.Timestamp("2025-12-31")]
        empty1 = pd.DataFrame(index=dates)
        empty2 = pd.DataFrame()
        empty3 = pd.DataFrame(index=dates)

        trimmed = apply_layer_precedence([empty1, empty2, empty3])

        assert len(trimmed) == 3
        assert all(t.empty for t in trimmed)

    def test_output_feeds_combine_dc_demand_layers_without_duplicate_columns(self):
        """End-to-end sanity check tying apply_layer_precedence's contract to
        combine_dc_demand_layers's documented requirement that overlaps be
        pre-resolved."""
        from grid_resilience.data.dc_demand_combined import (
            apply_layer_precedence, combine_dc_demand_layers,
        )

        dates = [pd.Timestamp("2025-12-31")]
        hyperscaler = pd.DataFrame({"CEG": [500.0], "TLN": [900.0]}, index=dates)
        pjm = pd.DataFrame(
            {"AEP": [10.0], "FE": [20.0], "CEG": [0.6], "TLN": [0.4]}, index=dates
        )
        ercot = pd.DataFrame(index=dates)

        hyperscaler_t, pjm_t, ercot_t = apply_layer_precedence([hyperscaler, pjm, ercot])
        combined = combine_dc_demand_layers(pjm_t, ercot_t, hyperscaler_t)

        assert not combined.columns.duplicated().any()
        assert set(combined.columns) == {"AEP", "FE", "CEG", "TLN"}
