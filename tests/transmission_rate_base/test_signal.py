import numpy as np
import pandas as pd
import pytest

from transmission_rate_base import signal as sig


# ── Task 5: parent panel ─────────────────────────────────────────────────────

def test_build_parent_panel_sums_filers_and_tracks_count():
    net_tx = pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2018, net_tx=100.0, net_tx_prorated=False),
        dict(utility_id_ferc1=2, report_year=2018, net_tx=50.0, net_tx_prorated=False),
        dict(utility_id_ferc1=1, report_year=2019, net_tx=110.0, net_tx_prorated=False),
        # 2019 has only filer 1 reporting -> summed anyway, n_filers=1
    ])
    totals = pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2018, net_total=400.0, gross_total=500.0),
        dict(utility_id_ferc1=2, report_year=2018, net_total=100.0, gross_total=120.0),
        dict(utility_id_ferc1=1, report_year=2019, net_total=420.0, gross_total=520.0),
    ])
    panel = sig.build_parent_panel(net_tx, totals, {"AEP": [1, 2]})
    p18 = panel[panel.year == 2018].iloc[0]
    assert p18["net_tx"] == 150.0 and p18["net_total"] == 500.0
    assert p18["tx_share"] == pytest.approx(0.3) and p18["n_filers"] == 2
    p19 = panel[panel.year == 2019].iloc[0]
    assert p19["net_tx"] == 110.0 and p19["n_filers"] == 1


def test_build_parent_panel_ignores_unmapped_filers():
    net_tx = pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2020, net_tx=100.0, net_tx_prorated=False),
        dict(utility_id_ferc1=99, report_year=2020, net_tx=999.0, net_tx_prorated=False),
    ])
    totals = pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2020, net_total=400.0, gross_total=500.0),
        dict(utility_id_ferc1=99, report_year=2020, net_total=9999.0, gross_total=9999.0),
    ])
    panel = sig.build_parent_panel(net_tx, totals, {"AEP": [1]})
    assert list(panel["ticker"]) == ["AEP"]
    assert panel.iloc[0]["net_tx"] == 100.0


# ── Task 6: primary signal ───────────────────────────────────────────────────

def _panel_two_names():
    rows = []
    for y, tx, tot in [(2015, 100, 500), (2016, 120, 520), (2017, 150, 550), (2018, 200, 600)]:
        rows.append(dict(ticker="FAST", year=y, net_tx=float(tx), net_total=float(tot),
                         tx_share=tx / tot, n_filers=1))
    for y, tx, tot in [(2015, 100, 500), (2016, 101, 505), (2017, 102, 510), (2018, 103, 515)]:
        rows.append(dict(ticker="SLOW", year=y, net_tx=float(tx), net_total=float(tot),
                         tx_share=tx / tot, n_filers=1))
    return pd.DataFrame(rows)


def test_primary_signal_ranks_fast_above_slow():
    out = sig.primary_signal(_panel_two_names())
    y2018 = out[out["year"] == 2018].set_index("ticker")
    assert y2018.loc["FAST", "g3_net_tx"] == pytest.approx(2 ** (1 / 3) - 1)
    assert y2018.loc["FAST", "raw_signal"] > y2018.loc["SLOW", "raw_signal"]
    assert y2018.loc["FAST", "raw_signal"] == pytest.approx(1.0)
    assert y2018.loc["SLOW", "raw_signal"] == pytest.approx(0.5)


def test_primary_signal_structural_break_nans_the_year():
    p = _panel_two_names()
    p.loc[(p.ticker == "FAST") & (p.year == 2017), ["net_tx", "tx_share"]] = [400.0, 400 / 550]
    out = sig.primary_signal(p).set_index(["ticker", "year"])
    assert np.isnan(out.loc[("FAST", 2018), "raw_signal"])


def test_primary_signal_requires_four_consecutive_years():
    p = _panel_two_names()
    p = p[~((p.ticker == "FAST") & (p.year == 2015))]
    out = sig.primary_signal(p).set_index(["ticker", "year"])
    assert np.isnan(out.loc[("FAST", 2018), "raw_signal"])


def test_primary_signal_filer_count_change_nans_the_year():
    p = _panel_two_names()
    p.loc[(p.ticker == "FAST") & (p.year == 2018), "n_filers"] = 2
    out = sig.primary_signal(p).set_index(["ticker", "year"])
    assert np.isnan(out.loc[("FAST", 2018), "raw_signal"])
