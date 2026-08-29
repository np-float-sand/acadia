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
