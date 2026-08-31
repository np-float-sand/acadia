import pandas as pd
import pytest

from transmission_rate_base.data import utility_map as um


def _xwalk():
    return pd.DataFrame([
        dict(utility_id_ferc1=11, utility_name_ferc1="Appalachian Power Co", utility_id_pudl=100),
        dict(utility_id_ferc1=12, utility_name_ferc1="Appalachian Power Co (XBRL)", utility_id_pudl=100),
        dict(utility_id_ferc1=13, utility_name_ferc1="Ohio Power Co", utility_id_pudl=101),
        dict(utility_id_ferc1=20, utility_name_ferc1="Florida Power & Light Co", utility_id_pudl=200),
    ])


def test_resolve_unions_pudl_id_siblings():
    parents = {"AEP": ["Appalachian Power Co", "Ohio Power Co"], "NEE": ["Florida Power & Light Co"]}
    got = um.resolve_filers(_xwalk(), parents)
    assert sorted(got["AEP"]) == [11, 12, 13]   # 12 pulled in via shared pudl id 100
    assert got["NEE"] == [20]


def test_resolve_unknown_name_raises():
    with pytest.raises(KeyError):
        um.resolve_filers(_xwalk(), {"X": ["No Such Utility Co"]})


def test_validate_rejects_double_mapped_pudl_id():
    parents = {"AEP": ["Appalachian Power Co"], "OTHER": ["Appalachian Power Co (XBRL)"]}
    with pytest.raises(ValueError):
        um.validate(_xwalk(), parents)


def test_validate_passes_for_disjoint_parents():
    parents = {"AEP": ["Appalachian Power Co", "Ohio Power Co"], "NEE": ["Florida Power & Light Co"]}
    um.validate(_xwalk(), parents)  # must not raise


def test_shipped_map_keys_are_within_universe_seed():
    from transmission_rate_base import config
    extra = set(um.PARENT_FILERS) - set(config.UNIVERSE_SEED)
    assert not extra, f"PARENT_FILERS has tickers not in UNIVERSE_SEED: {extra}"
