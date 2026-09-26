"""
PROBE C — input-cost hedge screen for the electrification supplier book.

The book's real cost risks are copper, grain-oriented electrical steel (GOES), and
skilled electrical labour. A hedge is something that RISES when those input costs rise
(so it pays off in the state that squeezes supplier margins) while NOT just re-buying
the theme.

Steps:
  Gate 0 : does the book itself have a measurable negative beta to copper / steel /
           labour after controlling for the market?  (if ~0, the premise is weak)
  Screen : regress a wide all-sector universe on [MktRF, book, copper, steel, labour];
           rank by input-cost beta, filter to low book-beta and non-loser names.
  Check  : neutralise the book's copper beta with each top candidate, look at behaviour
           in the worst copper-spike months.

Factors (monthly dlog): copper HG=F ; electrical steel CLF (sole US GOES producer,
proxy) ; labour = FRED construction AHE (CES2000000003) and a tradable staffing
basket RHI/MAN/KFY/ASGN.

Throwaway. `poetry run`, `.env` sourced.
"""
from __future__ import annotations
import io, os, sys, zipfile
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import probe_b as pb   # ff_factors()

# ---------------- wide universe: {ticker: industry} , all sectors ----------------
UNIV = {
    # copper / diversified miners
    "FCX": "copper", "SCCO": "copper", "TECK": "copper", "ERO": "copper", "HBM": "copper",
    "RIO": "diversified_mining", "BHP": "diversified_mining", "VALE": "diversified_mining",
    # steel (CLF = electrical steel proxy)
    "CLF": "steel", "NUE": "steel", "STLD": "steel", "RS": "steel", "CMC": "steel",
    "ATI": "steel", "MT": "steel", "X": "steel", "WOR": "steel",
    # aluminium / other metals & mining
    "AA": "aluminium", "CENX": "aluminium", "KALU": "aluminium",
    "NEM": "gold", "GOLD": "gold", "AEM": "gold", "MP": "rare_earth", "ALB": "lithium",
    # chemicals
    "DOW": "chemicals", "LYB": "chemicals", "DD": "chemicals", "CE": "chemicals",
    "EMN": "chemicals", "OLN": "chemicals", "WLK": "chemicals", "APD": "ind_gas",
    "LIN": "ind_gas", "SHW": "coatings", "PPG": "coatings", "ECL": "specialty_chem",
    "CF": "fertilizer", "MOS": "fertilizer", "NTR": "fertilizer", "FMC": "ag_chem",
    # electrical equipment / grid (the book & neighbours)
    "ETN": "elec_equip", "HUBB": "elec_equip", "NVT": "elec_equip", "GEV": "elec_equip",
    "VRT": "elec_equip", "PNR": "elec_equip", "AYI": "elec_equip", "POWL": "elec_equip",
    "ABBNY": "elec_equip", "SIEGY": "elec_equip", "PRY.MI": "cable",
    # machinery / industrials
    "EMR": "machinery", "ROK": "machinery", "PH": "machinery", "DOV": "machinery",
    "ITW": "machinery", "IR": "machinery", "GGG": "machinery", "NDSN": "machinery",
    "ROP": "machinery", "FTV": "machinery", "XYL": "machinery", "FLS": "machinery",
    "AOS": "machinery", "CAT": "machinery", "DE": "machinery", "CMI": "machinery",
    "PCAR": "machinery", "GE": "aerospace", "HON": "multi_industry", "MMM": "multi_industry",
    "AME": "machinery",
    # E&C / contractors (labour-heavy)
    "PWR": "e_and_c", "PRIM": "e_and_c", "MYRG": "e_and_c", "EME": "e_and_c",
    "FIX": "e_and_c", "ACM": "e_and_c", "J": "e_and_c", "FLR": "e_and_c",
    "STRL": "e_and_c", "DY": "e_and_c", "MTZ": "e_and_c", "GVA": "e_and_c",
    "TPC": "e_and_c", "APG": "e_and_c",
    # staffing / labour proxies
    "RHI": "staffing", "MAN": "staffing", "KFY": "staffing", "ASGN": "staffing",
    "HSII": "staffing", "NSP": "peo", "TNET": "peo",
    # energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "EOG": "energy", "SLB": "oil_svc",
    "HAL": "oil_svc", "BKR": "oil_svc", "PSX": "refiner", "VLO": "refiner", "MPC": "refiner",
    "WMB": "midstream", "OKE": "midstream", "KMI": "midstream", "LNG": "lng",
    # utilities / IPP
    "NEE": "utility", "DUK": "utility", "SO": "utility", "D": "utility", "AEP": "utility",
    "EXC": "utility", "XEL": "utility", "ED": "utility", "WEC": "utility", "ES": "utility",
    "VST": "ipp", "NRG": "ipp", "CEG": "ipp", "TLN": "ipp",
    # tech / semis
    "AAPL": "tech", "MSFT": "tech", "NVDA": "semis", "AVGO": "semis", "AMD": "semis",
    "TXN": "semis", "ADI": "semis", "MU": "semis", "AMAT": "semicap", "LRCX": "semicap",
    "KLAC": "semicap", "TER": "semicap", "ENPH": "solar", "FSLR": "solar", "SEDG": "solar",
    "GOOGL": "internet", "META": "internet", "AMZN": "ecommerce",
    # consumer / staples / health / fin (controls, "all industries")
    "HD": "retail", "LOW": "retail", "TSLA": "autos", "GM": "autos", "F": "autos",
    "NKE": "apparel", "SBUX": "restaurants", "MCD": "restaurants",
    "PG": "staples", "KO": "staples", "PEP": "staples", "COST": "staples", "WMT": "staples",
    "UNH": "health", "JNJ": "health", "LLY": "health", "PFE": "health", "TMO": "life_sci",
    "DHR": "life_sci", "JPM": "banks", "BAC": "banks", "GS": "banks", "BLK": "asset_mgmt",
    "SPGI": "exchanges", "ICE": "exchanges", "CME": "exchanges",
    "DIS": "media", "T": "telecom", "VZ": "telecom",
    "PLD": "reit", "AMT": "reit", "EQIX": "reit_dc", "DLR": "reit_dc",
    # building products / aggregates (steel + labour + copper adjacency)
    "VMC": "aggregates", "MLM": "aggregates", "BLD": "building_prod", "BLDR": "building_prod",
    "MAS": "building_prod", "CSL": "building_prod", "AWI": "building_prod",
    "CARR": "hvac", "TT": "hvac", "JCI": "hvac", "LII": "hvac",
}
BOOK_COL = "screen|plain"
_PXCACHE = os.path.join(os.path.dirname(__file__), "_cache", "probe_c_px.parquet")
_EXTRA = ["HG=F", "CPER", "CLF", "SLX", "FCX", "RHI", "MAN", "KFY", "ASGN", "SPY"]


def _all_daily_px() -> pd.DataFrame:
    """One cached daily-close pull for the whole ticker set (yfinance monthly-interval
    calls kept 401-ing on the combined load; daily batch + resample is stable)."""
    if os.path.exists(_PXCACHE):
        return pd.read_parquet(_PXCACHE)
    import yfinance as yf
    tks = sorted(set(list(UNIV) + _EXTRA))
    frame = None
    for attempt in range(3):
        raw = yf.download(tks, start="2015-06-01", end="2026-09-01", auto_adjust=True,
                          group_by="column", progress=False, threads=True)
        px = raw.xs("Close", axis=1, level=0) if isinstance(raw.columns, pd.MultiIndex) else raw
        px = px.ffill(limit=5).dropna(how="all")
        got = [t for t in tks if t in px.columns and px[t].notna().sum() > 200]
        if frame is None or len(got) > frame.shape[1]:
            frame = px[got]
        if len(got) >= len(tks) - 6:
            break
    frame.to_parquet(_PXCACHE)
    return frame


def _mret(px: pd.DataFrame) -> pd.DataFrame:
    m = px.resample("ME").last()
    return m.pct_change()


def _mdlog(px: pd.Series) -> pd.Series:
    return np.log(px.resample("ME").last()).diff()


def _dlog(s):
    return np.log(s).diff()


def fred_csv(series_id, start="2015-01-01"):
    u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    d = pd.read_csv(io.StringIO(requests.get(u, timeout=30).text))
    d.columns = ["date", "v"]
    d["date"] = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="M")
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.set_index("date")["v"]


def nw_t(y, X, lag=4):
    import statsmodels.api as sm
    m = sm.OLS(y, sm.add_constant(X), missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": lag})
    return m


def main():
    ff = pb.ff_factors()                                   # monthly, /100 already
    px = _all_daily_px()
    print(f"px: {px.shape[1]} tickers cached\n")
    # ---- factor proxies ----
    copper = _mdlog(px["HG=F"]).rename("copper")
    esteel = _mdlog(px["CLF"]).rename("esteel")            # GOES proxy (sole US producer)
    staff_cols = [c for c in ["RHI", "MAN", "KFY", "ASGN"] if c in px.columns]
    staff = _mdlog(px[staff_cols].mean(axis=1)).rename("staff")
    ahe = fred_csv("CES2000000003").pipe(lambda s: np.log(s).diff()).rename("ahe_constr")
    for s in (copper, esteel, staff):
        s.index = pd.PeriodIndex(s.index, freq="M")

    F = pd.concat([ff[["MktRF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]],
                   copper, esteel, staff, ahe], axis=1).dropna(subset=["MktRF"])

    # ---- book ----
    bk = pd.read_csv("output_electrification/returns.csv", parse_dates=["Date"]).set_index("Date")
    bk = bk[BOOK_COL]
    bk = bk[bk.ne(0).cumsum() > 0]                         # drop warmup zeros
    bkm = (1 + bk).resample("ME").prod() - 1
    bkm.index = pd.PeriodIndex(bkm.index, freq="M")
    d = F.join(bkm.rename("book"), how="inner").dropna(subset=["book", "copper", "esteel", "staff"])
    d["book_ex"] = d["book"] - d["RF"]

    print(f"=== PROBE C — input-cost hedge screen ===\nmonths: {len(d)}  "
          f"{d.index.min()}..{d.index.max()}\n")
    print("--- GATE 0: does the BOOK have negative beta to copper / e-steel / labour? ---")
    for cols in (["MktRF", "copper"], ["MktRF", "esteel"], ["MktRF", "staff"],
                 ["MktRF", "ahe_constr"], ["MktRF", "copper", "esteel", "staff"]):
        sub = d.dropna(subset=cols)
        m = nw_t(sub["book_ex"], sub[cols])
        terms = "  ".join(f"{c} {m.params[c]:+.2f}(t{m.tvalues[c]:+.1f})" for c in cols)
        print(f"  {terms}   R2={m.rsquared:.2f}")

    # ---- universe monthly excess returns ----
    rp = _mret(px[[c for c in UNIV if c in px.columns]])
    rp.index = pd.PeriodIndex(rp.index, freq="M")
    rows = []
    for tk in UNIV:
        if tk not in rp.columns:
            continue
        s = rp[tk].rename("r")
        j = d.join(s, how="inner").dropna(subset=["r", "copper", "esteel", "staff", "book_ex"])
        if len(j) < 48:
            continue
        j["r_ex"] = j["r"] - j["RF"]
        m = nw_t(j["r_ex"], j[["MktRF", "book_ex", "copper", "esteel", "staff"]])
        ann = (1 + j["r"]).prod() ** (12 / len(j)) - 1
        vol = j["r"].std() * np.sqrt(12)
        rows.append(dict(ticker=tk, industry=UNIV[tk], n=len(j),
                         b_mkt=m.params["MktRF"], b_book=m.params["book_ex"],
                         b_copper=m.params["copper"], t_copper=m.tvalues["copper"],
                         b_esteel=m.params["esteel"], t_esteel=m.tvalues["esteel"],
                         b_staff=m.params["staff"], t_staff=m.tvalues["staff"],
                         ann=ann, sharpe=ann / vol if vol else np.nan, r2=m.rsquared))
    R = pd.DataFrame(rows)
    for c in ["b_copper", "b_esteel", "b_staff"]:
        R[c + "_z"] = (R[c] - R[c].mean()) / R[c].std()
    R["input_score"] = R[["b_copper_z", "b_esteel_z", "b_staff_z"]].mean(axis=1)

    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("\n--- TOP 25 by input-cost beta (any industry) ---")
    top = R.sort_values("input_score", ascending=False).head(25)
    print(top[["ticker", "industry", "b_book", "b_copper", "t_copper", "b_esteel",
               "t_esteel", "b_staff", "t_staff", "ann", "sharpe", "input_score"]]
          .round(2).to_string(index=False))

    print("\n--- CLEAN HEDGE candidates: high input beta, |b_book|<=0.4, ann>-5% ---")
    clean = R[(R["b_book"].abs() <= 0.4) & (R["ann"] > -0.05)] \
        .sort_values("input_score", ascending=False).head(20)
    print(clean[["ticker", "industry", "b_book", "b_copper", "b_esteel", "b_staff",
                 "ann", "sharpe", "input_score"]].round(2).to_string(index=False))

    print("\n--- for context: BOTTOM 10 (negative input beta — these ADD to the risk) ---")
    bot = R.sort_values("input_score").head(10)
    print(bot[["ticker", "industry", "b_book", "b_copper", "b_esteel", "b_staff",
               "input_score"]].round(2).to_string(index=False))

    R.sort_values("input_score", ascending=False).to_csv(
        os.path.join(os.path.dirname(__file__), "_cache", "probe_c_betas.csv"), index=False)

    # ---- tail check: does the book actually get hit when copper/steel/labour spike? ----
    print("\n--- TAIL CHECK: book return in the 12 biggest cost-shock months ---")
    for fac, lab in [("copper", "copper"), ("esteel", "e-steel(CLF)"), ("staff", "staffing")]:
        big = d[fac].nlargest(12).index
        print(f"  {lab:14s} shock months: mean book_ex {d.loc[big,'book_ex'].mean():+.2%}  "
              f"vs all-month {d['book_ex'].mean():+.2%}   "
              f"(hit rate book<0: {(d.loc[big,'book_ex']<0).mean():.0%})")


if __name__ == "__main__":
    main()
