import sys

import pytest

from backlog_factor import __main__ as cli


def test_event_study_flag_dispatches(monkeypatch, tmp_path):
    seen = {}

    def fake_run(start, end, out):
        seen["args"] = (start, end, str(out))
        return {"passed": False}

    monkeypatch.setattr(cli, "_run_event_study", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "--event-study", "--start", "2019-01-01",
                                      "--end", "2024-01-01", "--output", str(tmp_path)])
    cli.main()
    assert seen["args"][0] == "2019-01-01"
    assert seen["args"][1] == "2024-01-01"


def test_no_flag_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        cli.main()
