import pandas as pd

from electrification_strategy import __main__ as cli


def _fake_short_interest_snapshot(tickers, **kwargs):
    return pd.DataFrame([
        {"ticker": t, "shares_short": 1_000_000, "short_ratio": 2.0, "short_pct_float": 0.03,
         "date_short_interest": pd.Timestamp("2024-12-01"), "fetched_at": pd.Timestamp("2024-12-15")}
        for t in tickers
    ]).set_index("ticker")


def test_cli_smoke(monkeypatch, tmp_path, synthetic_prices, synthetic_dfii10):
    monkeypatch.setattr(cli, "fetch_prices", lambda tickers, start, end: synthetic_prices)
    monkeypatch.setattr(cli, "fetch_series", lambda sid, start, end: synthetic_dfii10)
    monkeypatch.setattr(cli, "fetch_short_interest", _fake_short_interest_snapshot)

    cli.main(["--start", "2018-01-01", "--end", "2024-12-31",
              "--no-plot", "--output", str(tmp_path)])

    for f in ("metrics.csv", "episode_drawdowns.csv", "plateau.json", "returns.csv"):
        assert (tmp_path / f).exists()
    assert not (tmp_path / "performance.png").exists()
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert (m["name"] == "frozen +val+sleeve+short").any()


def test_cli_single_universe(monkeypatch, tmp_path, synthetic_prices, synthetic_dfii10):
    monkeypatch.setattr(cli, "fetch_prices", lambda tickers, start, end: synthetic_prices)
    monkeypatch.setattr(cli, "fetch_series", lambda sid, start, end: synthetic_dfii10)
    monkeypatch.setattr(cli, "fetch_short_interest", _fake_short_interest_snapshot)
    cli.main(["--start", "2018-01-01", "--end", "2024-12-31", "--no-plot",
              "--universe", "marquee", "--output", str(tmp_path)])
    m = pd.read_csv(tmp_path / "metrics.csv")
    cells = [n for n in m["name"] if not n.startswith("BENCH")]
    assert all(n.startswith("marquee") for n in cells)


# -- Task 8: risk-exposure / short-interest diagnostics wired into the CLI --
def test_cli_smoke_diagnostics_absent_by_default(monkeypatch, tmp_path, synthetic_prices, synthetic_dfii10):
    """No real electrification_strategy/data/risk_exposures.csv or cached short-interest
    history exists yet (spec 3.1/3.3 placeholders), so the CLI should run end-to-end
    without crashing and simply not emit the diagnostic CSVs -- not fail. Points CACHE_DIR
    at an empty tmp dir so this doesn't depend on (or pollute) the repo's real cache dir."""
    monkeypatch.setattr(cli, "fetch_prices", lambda tickers, start, end: synthetic_prices)
    monkeypatch.setattr(cli, "fetch_series", lambda sid, start, end: synthetic_dfii10)
    monkeypatch.setattr(cli, "fetch_short_interest", _fake_short_interest_snapshot)
    monkeypatch.setattr(cli.config, "CACHE_DIR", tmp_path / "empty_cache")

    out_dir = tmp_path / "out"
    cli.main(["--start", "2018-01-01", "--end", "2024-12-31",
              "--no-plot", "--output", str(out_dir)])

    for f in ("metrics.csv", "episode_drawdowns.csv", "plateau.json", "returns.csv"):
        assert (out_dir / f).exists()
    assert not (out_dir / "risk_exposure_report.csv").exists()
    assert not (out_dir / "short_interest_diagnostic.csv").exists()


def test_cli_writes_short_interest_diagnostic_when_history_cached(
    monkeypatch, tmp_path, synthetic_prices, synthetic_dfii10
):
    """When a cached short-interest history parquet exists (e.g. from a prior run),
    the CLI should thread it through to write_outputs and produce the diagnostic CSV."""
    monkeypatch.setattr(cli, "fetch_prices", lambda tickers, start, end: synthetic_prices)
    monkeypatch.setattr(cli, "fetch_series", lambda sid, start, end: synthetic_dfii10)
    monkeypatch.setattr(cli, "fetch_short_interest", _fake_short_interest_snapshot)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    history = pd.DataFrame([
        {"ticker": "DLR", "shares_short": 1, "short_ratio": 2.0, "short_pct_float": 0.03,
         "date_short_interest": pd.Timestamp("2024-12-01"), "fetched_at": pd.Timestamp("2024-12-15")},
        {"ticker": "EQIX", "shares_short": 1, "short_ratio": 1.0, "short_pct_float": 0.01,
         "date_short_interest": pd.Timestamp("2024-12-01"), "fetched_at": pd.Timestamp("2024-12-15")},
    ])
    history_path = cache_dir / cli.config.SHORT_INTEREST_CACHE_FILE
    history.to_parquet(history_path, index=False)
    monkeypatch.setattr(cli.config, "CACHE_DIR", cache_dir)

    out_dir = tmp_path / "out"
    cli.main(["--start", "2018-01-01", "--end", "2024-12-31",
              "--no-plot", "--output", str(out_dir)])

    assert (out_dir / "short_interest_diagnostic.csv").exists()
