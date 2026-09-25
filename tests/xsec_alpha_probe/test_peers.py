import pandas as pd

from research.xsec_alpha_probe.peers import peer_signal
from research.xsec_alpha_probe.tenk import item1


def test_item1_skips_toc_and_cross_references():
    html = (b"<p>Item 1. Business 3</p><p>Item 1A. Risk Factors 12</p>"
            b"<p>ITEM 1. B<span>USINESS</span></p><p>We make transformers. " + b"grid " * 500 +
            b"See Item 1A. Risk Factors of this report.</p><p>ITEM 1A. RIS K FACTORS</p><p>Risks here. "
            b"Item 1. Business \xe2\x80\x94 Backlog explains more. " + b"risk " * 2000 +
            b"</p><p>Item 2. Properties</p>")
    text = item1(html)
    assert "We make transformers" in text
    assert "Risks here" not in text          # a later cross-reference must not win on length


def test_peer_signal_uses_top_outside_leaders_weighted_by_cosine():
    dt = pd.Timestamp("2024-01-31")
    sim = pd.DataFrame([[0, .1, .5, .3], [.1, 0, .2, .4], [.5, .2, 0, 0], [.3, .4, 0, 0]],
                       index=list("ABXY"), columns=list("ABXY"), dtype=float)
    ret = pd.DataFrame({"A": [0.0], "B": [0.9], "X": [0.10], "Y": [-0.20]}, index=[dt])
    univ = pd.DataFrame({"A": [True], "B": [True], "X": [False], "Y": [False]}, index=[dt])
    out = peer_signal({dt: sim}, ret, univ, outside=True)
    assert abs(out.loc[dt, "A"] - (.5 * .10 + .3 * -.20) / .8) < 1e-12   # B excluded: inside
    allp = peer_signal({dt: sim}, ret, univ, outside=False)
    assert abs(allp.loc[dt, "A"] - (.5 * .10 + .3 * -.20 + .1 * .9) / .9) < 1e-12


def test_geo_signal_excludes_same_industry_and_inside_peers():
    from research.xsec_alpha_probe.geo import geo_signal
    dt = pd.Timestamp("2024-01-31")
    addr = pd.DataFrame({"ticker": list("ABCDE"), "city": ["HOUSTON"] * 5, "state": ["TX"] * 5,
                         "sic": ["3600", "3600", "4911", "2800", "7370"]})
    ret = pd.DataFrame({t: [v] for t, v in zip("ABCDE", [0.0, 0.5, 0.1, 0.3, 0.2])}, index=[dt])
    univ = pd.DataFrame({t: [t in "AC"] for t in "ABCDE"}, index=[dt])
    out = geo_signal(addr, ret, univ, outside=True)
    assert abs(out.loc[dt, "A"] - (0.3 + 0.2) / 2) < 1e-12     # B same SIC2, C inside universe
    allp = geo_signal(addr, ret, univ, outside=False)
    assert abs(allp.loc[dt, "A"] - (0.1 + 0.3 + 0.2) / 3) < 1e-12
