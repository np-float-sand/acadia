from __future__ import annotations

"""
Defines the investable utility universe.

Tickers are drawn from TICKER_NODE_MAP but can be overridden here
(e.g., to exclude non-grid names or add new names mid-backtest).

Each entry records the primary ISO, whether the name has an actionable
LMP node mapping, and a broad category used for sector-neutral construction.
"""

from .utility_node_map import TICKER_NODE_MAP, TICKERS_BY_ISO

# Full universe: every ticker in the node map
UNIVERSE: list[str] = list(TICKER_NODE_MAP.keys())

# Subset with LMP-mapped nodes — these get a grid signal; others get
# sector-average imputation in the factor layer
LMP_MAPPED: list[str] = [t for t, v in TICKER_NODE_MAP.items() if v["nodes"]]

# Tickers where we rely on EIA-417 outage data instead of LMP
OUTAGE_MAPPED: list[str] = [t for t, v in TICKER_NODE_MAP.items() if not v["nodes"]]

# Groupings used by the portfolio layer for sector-neutral construction
# (all are electric utilities; sub-group by generation vs. wires)
GENERATORS = ["NRG", "VST", "ETR", "NEE"]          # primary generation exposure
WIRES_ONLY  = ["CNP", "EXC", "PPL", "FE", "ES", "ED"]  # T&D focused
INTEGRATED  = ["AEP", "WEC", "DTE", "CMS", "XEL", "PCG", "EIX", "SO"]  # gen + wires


def get_iso_tickers(iso: str) -> list[str]:
    """Return tickers whose primary ISO matches `iso`."""
    return TICKERS_BY_ISO.get(iso, [])


def get_ticker_nodes(ticker: str) -> list[str]:
    """Return the list of ISO nodes for a ticker, or [] if not mapped."""
    return TICKER_NODE_MAP.get(ticker, {}).get("nodes", [])


def get_ticker_iso(ticker: str) -> str | None:
    """Return the primary ISO string for a ticker."""
    return TICKER_NODE_MAP.get(ticker, {}).get("iso")
