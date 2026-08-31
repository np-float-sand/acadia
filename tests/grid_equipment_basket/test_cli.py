import sys

import pandas as pd
import pytest

from grid_equipment_basket import __main__ as cli


def test_construction_flag_dispatches_to_value_chain_report(monkeypatch, tmp_path, capsys):
    called = {}

    def fake_report(start, end, drop_winners=False, **kw):
        called["start"] = start
        called["drop_winners"] = drop_winners
        # Structurally-valid stub so main() can write value_chain outputs.
        return {"_fake": True, "equal_weight": {}, "value_chain_tilt": {},
                "pair_standalone": {}, "benchmarks": {}, "gate1": {}, "gate2": {},
                "episode": {}}

    monkeypatch.setattr(cli.backtest, "value_chain_report", fake_report)
    monkeypatch.setattr(cli.backtest, "value_chain_table", lambda r: "TABLE-OK")
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--construction", "value-chain-tilt", "--drop-winners",
                         "--start", "2023-01-01", "--end", "2024-01-01",
                         "--output", str(tmp_path), "--no-plot"])
    cli.main()
    out = capsys.readouterr().out
    assert "TABLE-OK" in out
    assert called["drop_winners"] is True
    assert called["start"] == "2023-01-01"


def test_tilt_backlog_still_works_as_alias(monkeypatch, tmp_path):
    seen = {}

    def _fake_run_capture(s, e, target_fn=None, **k):
        seen["target_fn"] = target_fn
        return _fake_run()

    monkeypatch.setattr(cli.backtest, "run", _fake_run_capture)
    monkeypatch.setattr(cli.backtest, "results_table", lambda r: "")
    monkeypatch.setattr(cli.backtest, "plot", lambda r, p: None)
    monkeypatch.setattr(cli, "_backlog_target_fn", lambda: "BL_FN")
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--tilt", "backlog", "--output", str(tmp_path), "--no-plot"])
    cli.main()
    assert seen["target_fn"] == "BL_FN"


def _fake_run():
    idx = pd.bdate_range("2023-01-02", periods=5)
    return {"basket": {"n_obs": 5, "cagr": 0.1, "ann_vol": 0.2, "sharpe": 0.5,
                       "sortino": 0.6, "max_dd": -0.1, "hit_rate": 0.5},
            "basket_returns": pd.Series([0.0] * 5, index=idx),
            "benchmarks": {}}


def test_overlay_l2_flag_prints_ladder_and_writes_outputs(monkeypatch, tmp_path, capsys):
    from grid_equipment_basket import grid_regime

    idx = pd.bdate_range("2023-01-02", periods=6)
    stub_report = {
        "signal_start": "2018-01-01",
        "windows": {"primary": ("2023-01-01", "2025-12-31"), "prior": ("2020-01-01", "2022-12-31")},
        "baselines": {"layer1_only": {"primary": {"metrics": {"sharpe": 1.4}}}},
        "rungs": [
            {"name": "1: discrete, PJM-4, congestion-only",
             "verdict": "FAIL", "gate": {"G1": False, "G2": True, "G3": True, "marginal": False},
             "mechanics": {"avg_mult": 0.98, "flips": 4},
             "block": {"primary": {"metrics": {"cagr": 0.3, "sharpe": 1.3, "max_dd": -0.2}},
                       "prior": {"metrics": {"cagr": 0.1, "sharpe": 0.5, "max_dd": -0.4}}},
             "multiplier": pd.Series([1.0, 1.0, 0.6, 0.6, 1.25, 1.25], index=idx)},
            {"name": "7: + RT/DA spread sub-signal", "status": "deferred", "reason": "x"},
        ],
        "stopped_at": None,
    }
    monkeypatch.setattr(grid_regime, "regime_report", lambda **kw: stub_report)
    monkeypatch.setattr(grid_regime, "regime_table", lambda r: "L2-LADDER-TABLE")
    monkeypatch.setattr(cli.backtest, "run", lambda s, e, **k: _fake_run())
    monkeypatch.setattr(cli.backtest, "results_table", lambda r: "")
    monkeypatch.setattr(cli.backtest, "plot", lambda r, p: None)
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--overlay-l2", "--output", str(tmp_path), "--no-plot"])
    cli.main()
    out = capsys.readouterr().out
    assert "L2-LADDER-TABLE" in out
    assert (tmp_path / "regime_metrics.csv").exists()
    assert (tmp_path / "regime_timeline.csv").exists()
