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
