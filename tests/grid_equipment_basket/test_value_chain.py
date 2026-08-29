import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import config, value_chain as vc


def test_buckets_are_the_frozen_split():
    assert sorted(config.BUCKET_MAKERS) == ["ETN", "GEV", "HUBB", "NVT", "VRT"]
    assert sorted(config.BUCKET_CONTRACTORS) == ["FLNC", "MYRG", "PRIM", "PWR"]
    assert set(config.BUCKET_MAKERS) & set(config.BUCKET_CONTRACTORS) == set()


def test_bucket_of_classifies_and_rejects_unknown():
    assert vc.bucket_of("ETN") == "maker"
    assert vc.bucket_of("PWR") == "contractor"
    with pytest.raises(KeyError):
        vc.bucket_of("AAPL")


def test_composite_rank_is_mean_of_ascending_ranks():
    a = pd.Series({"X": 0.05, "Y": -0.02, "Z": 0.10})   # ranks 2, 1, 3
    b = pd.Series({"X": 1.0, "Y": 3.0, "Z": 2.0})        # ranks 1, 3, 2
    out = vc.composite_rank(a, b, ["X", "Y", "Z"])
    assert out["X"] == pytest.approx(1.5)
    assert out["Y"] == pytest.approx(2.0)
    assert out["Z"] == pytest.approx(2.5)


def test_composite_rank_one_missing_component_uses_the_other():
    a = pd.Series({"X": 0.05, "Y": -0.02})               # Z missing from a
    b = pd.Series({"X": 1.0, "Y": 3.0, "Z": 2.0})
    out = vc.composite_rank(a, b, ["X", "Y", "Z"])
    assert not np.isnan(out["Z"])                        # ranked on b alone
    assert np.isnan(vc.composite_rank(pd.Series(dtype=float),
                                     pd.Series(dtype=float), ["Z"])["Z"])
