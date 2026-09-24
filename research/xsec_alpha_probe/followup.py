"""Follow-up on the two leads from run.py (capex_light CLOSE, rev_accel stable book).

    python -m research.xsec_alpha_probe.followup

1. Group-neutral: is capex_light just utilities-vs-suppliers? Z-score within SIC
   group (ELEC vs POWER), and separately evaluate inside the ELEC group alone.
2. Composite (POST-HOC, chosen after seeing run.py): mean z of capex_light and
   rev_accel. Reported for information; it cannot earn a pre-registered PASS.
"""
from __future__ import annotations

import pandas as pd

from research.xsec_alpha_probe import data, evaluate as ev, run, signals as sg


def group_map() -> pd.Series:
    cand = pd.read_parquet(data._cis() / "candidates.parquet")
    return cand.drop_duplicates("ticker").set_index("ticker")["group"]


def group_neutral(raw: pd.DataFrame, univ: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    cols = raw.columns.intersection(univ.columns)
    x = raw.reindex(index=univ.index, columns=cols).where(univ[cols])
    parts = []
    for g in groups.reindex(cols).dropna().unique():
        members = [c for c in cols if groups.get(c) == g]
        parts.append(ev.prep(x[members]))
    return pd.concat(parts, axis=1)


def main() -> pd.DataFrame:
    px_d = data.prices()
    univ = data.universe()
    px_m = data.monthly_prices(px_d)
    ret_m = px_m.pct_change(fill_method=None)
    fwd1, fwd3 = ret_m.shift(-1), px_m.shift(-3) / px_m - 1
    grid_fwd = data.monthly_prices(data.benchmarks())["GRID"].pct_change().shift(-1)
    risks = run.risk_panels(px_d)
    fund = sg.fundamental_signals()
    groups = group_map()
    elec = [c for c in univ.columns if groups.get(c) == "ELEC"]
    univ_elec = univ.copy()
    univ_elec[[c for c in univ.columns if c not in elec]] = False

    cases = {}
    for name in ("capex_light", "rev_accel"):
        cases[f"{name} | group-neutral"] = (group_neutral(fund[name], univ, groups), univ)
        cases[f"{name} | ELEC only"] = (fund[name], univ_elec)
    comp = (ev.prep(fund["capex_light"].reindex(index=univ.index).where(univ))
            .add(ev.prep(fund["rev_accel"].reindex(index=univ.index).where(univ)), fill_value=0) / 2)
    cases["composite (post-hoc)"] = (comp, univ)
    cases["composite (post-hoc) | ELEC only"] = (comp, univ_elec)

    rows = []
    for name, (raw, u) in cases.items():
        rows.append(run.evaluate_signal(name, raw, u, fwd1, fwd3, risks, grid_fwd))
    res = pd.DataFrame(rows).set_index("signal")
    res.to_csv(data.CACHE / "results_followup.csv")
    return res


def legs() -> pd.DataFrame:
    """Where does the IC live? top-minus-universe vs universe-minus-bottom vs Q5-Q1."""
    px_d = data.prices()
    univ = data.universe()
    fwd1 = data.monthly_prices(px_d).pct_change(fill_method=None).shift(-1)
    fund = sg.fundamental_signals()
    groups = group_map()
    elec = [c for c in univ.columns if groups.get(c) == "ELEC"]
    ue = univ.copy()
    ue[[c for c in univ.columns if c not in elec]] = False

    def z(x, u):
        return ev.prep(x.reindex(index=u.index, columns=u.columns).where(u))

    comp = z(fund["capex_light"], univ).add(z(fund["rev_accel"], univ), fill_value=0) / 2
    cases = {"capex_light ALL": (z(fund["capex_light"], univ), univ),
             "capex_light ELEC": (z(fund["capex_light"], ue), ue),
             "rev_accel ELEC": (z(fund["rev_accel"], ue), ue),
             "composite ALL (post-hoc)": (comp, univ),
             "composite ELEC (post-hoc)": (comp.where(ue), ue)}
    rows = []
    for name, (s, u) in cases.items():
        top = run.book_returns(s, fwd1, u, None)
        bot = run.book_returns(-s, fwd1, u, None)
        ls = (top["long"] - bot["long"]).dropna()
        for w, (a, b) in run.WINDOWS.items():
            p = ev.perf(ls.loc[a:b])
            rows.append({"case": name, "window": w,
                         "top-univ SR": ev.perf(top["spread"].loc[a:b])["sharpe"],
                         "univ-bottom SR": ev.perf(-bot["spread"].loc[a:b])["sharpe"],
                         "Q5-Q1 SR": p["sharpe"], "Q5-Q1 ann": p["ann"], "Q5-Q1 maxDD": p["maxdd"]})
    return pd.DataFrame(rows).set_index(["case", "window"])


if __name__ == "__main__":
    import sys
    if "--legs" in sys.argv:
        print(legs().round(2).to_string())
    else:
        run._print(main())
