import numpy as np
import pandas as pd

from grid_equipment_basket import backlog_forward_check as bfc


def _prices():
    idx = pd.bdate_range("2022-01-03", periods=700)
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx))))
         for t in ["AAA", "BBB"]},
        index=idx,
    )


def _backlog():
    rows = []
    for t, base in [("AAA", 100), ("BBB", 200)]:
        for k, qe in enumerate(pd.date_range("2022-03-31", periods=12, freq="QE")):
            rows.append({
                "ticker": t, "quarter_end": qe,
                "availability_date": qe + pd.Timedelta(days=35),
                "metric_value": base * (1 + 0.03 * k),
                "metric_unit": "USD_million", "disclosure_type": "xbrl_rpo",
                "segment_scope": "total", "source_url": "http://x", "notes": "",
            })
    return pd.DataFrame(rows)


def test_forward_return_table_has_row_per_name_and_horizons():
    tbl = bfc.forward_return_table(_backlog(), _prices(), horizons=(63, 126))
    assert set(tbl.index) == {"AAA", "BBB"}
    assert {"corr_63", "corr_126", "n"}.issubset(tbl.columns)
    assert (tbl["n"] > 0).all()
