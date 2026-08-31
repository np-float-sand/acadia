"""
grid_demand_factor — triage probe for Proposal D.

Question under test (pre-registered gate in ``probe.py``): does ranking a broad
equity universe by its return-*sensitivity* to a grid-demand nowcast produce any
usable cross-sectional signal?

This package is a *triage* artifact, not a shipped strategy. It reuses the
already-cached multi-ISO Grid Stress Index (GSI) series written by
``grid_resilience`` and a broad price panel pulled through
``grid_resilience.data.equity_prices``.

Modules
-------
nowcast      : assemble a monthly grid-demand nowcast from cached daily GSI.
sensitivity  : rolling sensitivity betas, quintile spread, rank-IC, gate.
probe        : end-to-end run + pre-registered pass/fail verdict.
"""
