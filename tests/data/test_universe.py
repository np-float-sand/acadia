from grid_resilience.data.universe import UNIVERSE, get_ticker_iso
from grid_resilience.data.utility_node_map import TICKER_NODE_MAP
from grid_resilience.config import SUPPORTED_ISOS


def test_ceg_and_tln_in_universe():
    assert "CEG" in UNIVERSE
    assert "TLN" in UNIVERSE


def test_ceg_and_tln_resolve_to_pjm():
    assert get_ticker_iso("CEG") == "PJM"
    assert get_ticker_iso("TLN") == "PJM"


def test_ceg_and_tln_are_tagged_merchant():
    assert TICKER_NODE_MAP["CEG"]["business_model"] == "merchant"
    assert TICKER_NODE_MAP["TLN"]["business_model"] == "merchant"


def test_active_universe_has_twenty_tickers():
    """Active universe = UNIVERSE filtered to the 5 supported ISOs.
    Was 18 (merchant=2, mixed=5, regulated=11); adding CEG+TLN (both PJM,
    merchant) should bring it to 20 without disturbing the other groups."""
    active = [t for t in UNIVERSE if get_ticker_iso(t) in SUPPORTED_ISOS]
    assert len(active) == 20

    from collections import Counter
    counts = Counter(TICKER_NODE_MAP[t]["business_model"] for t in active)
    assert counts == {"merchant": 4, "mixed": 5, "regulated": 11}
