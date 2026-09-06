import pandas as pd

from electrification_strategy import __main__ as cli


def test_cli_smoke(monkeypatch, tmp_path, synthetic_prices, synthetic_dfii10):
    monkeypatch.setattr(cli, "fetch_prices", lambda tickers, start, end: synthetic_prices)
    monkeypatch.setattr(cli, "fetch_series", lambda sid, start, end: synthetic_dfii10)

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
    cli.main(["--start", "2018-01-01", "--end", "2024-12-31", "--no-plot",
              "--universe", "marquee", "--output", str(tmp_path)])
    m = pd.read_csv(tmp_path / "metrics.csv")
    cells = [n for n in m["name"] if not n.startswith("BENCH")]
    assert all(n.startswith("marquee") for n in cells)
