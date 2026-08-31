import numpy as np
import pandas as pd

from transmission_rate_base import report


def _synthetic_ferc():
    rows_204, rows_219, rows_200 = [], [], []
    profiles = {1: ("FAST", 1.18), 2: ("MID", 1.06), 3: ("SLOW", 1.005)}
    for fid, (_, g) in profiles.items():
        tx = 100.0
        for yr in range(2005, 2023):
            rows_204.append(dict(utility_id_ferc1=fid, report_year=yr,
                                 ferc_account_label="transmission_plant",
                                 row_type_xbrl="calculated_value", ending_balance=tx))
            rows_219.append(dict(utility_id_ferc1=fid, report_year=yr, plant_function="transmission",
                                 depreciation_type="accumulated_depreciation",
                                 row_type_xbrl="reported_value", ending_balance=-tx * 0.3))
            rows_200.append(dict(utility_id_ferc1=fid, report_year=yr, utility_type="electric",
                                 utility_plant_asset_type="utility_plant_in_service_classified",
                                 row_type_xbrl="reported_value", ending_balance=tx * 4))
            rows_200.append(dict(utility_id_ferc1=fid, report_year=yr, utility_type="electric",
                                 utility_plant_asset_type="utility_plant_net",
                                 row_type_xbrl="calculated_value", ending_balance=tx * 2.8))
            tx *= g
    xwalk = pd.DataFrame([
        dict(utility_id_ferc1=1, utility_name_ferc1="Fast Power Co", utility_id_pudl=1),
        dict(utility_id_ferc1=2, utility_name_ferc1="Mid Power Co", utility_id_pudl=2),
        dict(utility_id_ferc1=3, utility_name_ferc1="Slow Power Co", utility_id_pudl=3),
    ])
    return {"plant_in_service": pd.DataFrame(rows_204), "dep_by_function": pd.DataFrame(rows_219),
            "plant_summary": pd.DataFrame(rows_200), "utility_xwalk": xwalk}


class _StubPrices:
    def __init__(self, idx):
        self._i = idx
        self._t = ["FAST", "MID", "SLOW"]

    def daily_returns(self, tickers, start, end, *, offline=False):
        rng = np.random.default_rng(7)
        base = pd.DataFrame(rng.normal(0, 0.01, (len(self._i), len(self._t))),
                            index=self._i, columns=self._t)
        base["FAST"] += 0.0004
        base["SLOW"] -= 0.0004
        return base[[t for t in tickers if t in base.columns]].loc[start:end]

    def monthly_returns(self, tickers, start, end, *, offline=False):
        d = self.daily_returns(tickers, start, end)
        return (1 + d).resample("ME").prod() - 1


def test_pipeline_runs_end_to_end_offline(tmp_path, monkeypatch):
    from transmission_rate_base.data import utility_map
    monkeypatch.setattr(utility_map, "PARENT_FILERS",
                        {"FAST": ["Fast Power Co"], "MID": ["Mid Power Co"], "SLOW": ["Slow Power Co"]})
    idx = pd.bdate_range("2005-01-01", "2022-12-31")
    out = report.run_pipeline(start="2011-01-01", end="2022-12-31", offline=True,
                              out_dir=str(tmp_path), _ferc=_synthetic_ferc(),
                              _prices=_StubPrices(idx))
    assert set(out) >= {"verdict", "metrics", "ic", "additivity", "annual_returns"}
    assert (tmp_path / "verdict.txt").exists()
    assert (tmp_path / "signal_panel.csv").exists()
    assert isinstance(out["verdict"]["passed"], bool)
    assert len(out["verdict"]["reasons"]) == 5
