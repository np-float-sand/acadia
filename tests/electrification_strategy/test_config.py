from pathlib import Path

from electrification_strategy import config as c


def test_frozen_scalar_params():
    assert c.VOL_TARGET == 0.20
    assert c.VOL_LOOKBACK == 21
    assert c.VOL_MAX_LEVERAGE == 1.5
    assert c.REBALANCE_LAG_DAYS == 42
    assert c.MAX_SINGLE_NAME_WEIGHT == 0.25
    assert c.LISTING_MIN_DAYS == 252
    assert (c.VAL_EXT_LO, c.VAL_EXT_HI) == (0.15, 0.35)
    assert (c.VAL_RS_LO, c.VAL_RS_HI) == (0.25, 0.50)
    assert (c.VAL_MULT_MID, c.VAL_MULT_LOW) == (0.8, 0.6)
    assert c.VAL_MA_DAYS == 200 and c.VAL_RS_DAYS == 252
    assert c.HEDGE_SLEEVE_TICKERS == ("GLD", "IEF") and c.HEDGE_SLEEVE_WEIGHT == 0.15
    assert c.HEDGE_SHORT_TICKERS == ("DLR", "EQIX") and c.HEDGE_SHORT_WEIGHT == 0.25
    assert c.HEDGE_SHORT_SIGNAL == "DFII10"
    assert c.HEDGE_SHORT_LOOKBACK_DAYS == 126 and c.HEDGE_SHORT_LAG_DAYS == 5
    assert c.RF == 0.04 and c.ANN == 252


def test_frozen_collections():
    assert c.MARQUEE_UNIVERSE == ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]
    assert c.BENCHMARKS == ["VOLT", "PAVE", "GRID", "SPY", "XLI"]
    assert c.FEARED_PROXY == "TAN"
    assert set(c.EPISODES) == {"COVID 20", "Rate shock 22", "DeepSeek 25", "Tariff Apr-25", "Selloff 26"}
    for (pk, trough_end) in c.EPISODES.values():
        assert isinstance(pk, tuple) and len(pk) == 2 and isinstance(trough_end, str)
    assert set(c.SUBWINDOWS) == {"2019-22", "2023-26"}
    assert c.VAL_PLATEAU_SCALES == (0.8, 1.0, 1.2)
    assert c.SUB_INDUSTRIES == {
        "electrical_equipment", "heavy_electrical", "electrical_e&c",
        "dc_thermal", "grid_scale_storage",
    }


def test_winner_rule_constants():
    assert c.WINNER_MAXDD_MAX == -0.27
    assert c.WINNER_SUBWINDOW_SHARPE_MIN == 0.4
    assert c.WINNER_FEARED_PNL_MIN == -0.02
    assert c.WINNER_DRAG_MAX == 0.06


def test_paths_exist():
    assert isinstance(c.DATA_DIR, Path) and c.DATA_DIR.is_dir()
    assert isinstance(c.CACHE_DIR, Path) and c.CACHE_DIR.is_dir()
