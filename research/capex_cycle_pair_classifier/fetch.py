"""Regenerate the price parquets the backtests in this dir depend on.

Run from the repo root:  python research/capex_cycle_pair_classifier/fetch.py
Writes _cache/{der_extra,gen,hist,live,policy}_prices.parquet (all gitignored).
Also needs, for parse_b9.py only, the PJM Load Forecast PDFs/xlsx in _cache/pjm_lf
and _cache/pjm_xls -- see README (hand-download, URLs in parse_b9 / RESEARCH-LOG).
The grid-equipment price panel (grid_equipment_basket/data/cache/prices_*.parquet)
is reused as-is from that package's cache.
"""
from __future__ import annotations
import pathlib, warnings
import pandas as pd, yfinance as yf
warnings.simplefilter("ignore")

C = pathlib.Path(__file__).parent / "_cache"; C.mkdir(exist_ok=True)

BATCHES = {
    "der_extra_prices.parquet": [
        "NOVA","SPWR","EVGO","MAXN","WBX","SUNW","FTCI","SEDG","ENPH","RUN","CHPT","BLNK","STEM","GNRC"],
    "gen_prices.parquet": [
        "BWA","APTV","VC","LEA","ST","ALSN","MGA","DAN",
        "RIVN","LCID","NKLA","WKHS","PSNY","NIO","XPEV","FSR","RIDE","GOEV","ARVL","FFIE","LI",
        "BWXT","CCJ","LEU","UUUU","UEC","SMR","OKLO","NNE","LTBR","ASPI",
        "IRDM","RKLB","ASTS","PL","BKSY","RDW","SPCE","SATS","GSAT"],
    "hist_prices.parquet": [
        "PLUG","FCEL","BLDP","BE","LIN","APD","CMI",
        "TLRY","CGC","ACB","CRON","SNDL","OGI","SMG","IIPR","GRWG",
        "CRSP","NTLA","BEAM","EDIT","ILMN","PACB","TXG","TMO","DHR"],
    "live_prices.parquet": [
        "QS","MVST","SLDP","AMPX","ALB","SQM","LAC","ENVX","PLL","SES",
        "JOBY","ACHR","EVTL","BLDE","LILM","HON","TDG","HEI","TXT","GRMN",
        "IONQ","RGTI","QBTS","QUBT","SOUN","BBAI","AI","RCAT","ONDS","UMAC","AVAV","KTOS"],
    "policy_prices.parquet": sorted(set([
        "VRT","ANET","CIEN","APH","FIX","EME","IESC","CARR",
        "AMAT","LRCX","KLAC","ONTO","ACLS","KLIC","AEIS","ENTG",
        "HEI","TDG","HWM","CW","WWD","TXT","HXL",
        "J","STRL","PWR","ROK","EMR","NDSN","PH","DOV","AME",
        "XYL","BMI","MWA","FELE","PNR","AOS","ROP",
        "SIEGY","ETN","HUBB","POWL","NVT","WCC","PRYMY",
        "CR","SPXC","FLS","ITT","GGG",
        "MMM","ITW","GWW","FAST","ODFL","JBHT","EXPD","CHRW","PCAR","GPC","SNA","SWK","WSO",
        "XLI","SPY","PAVE"])),
}
for fn, tk in BATCHES.items():
    raw = yf.download(tk, start="1999-01-01", end="2026-09-01",
                      auto_adjust=True, progress=False, threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px.to_parquet(C / fn)
    got = [c for c in px.columns if px[c].notna().sum() > 30]
    print(f"{fn}: {len(got)}/{len(tk)} tickers  ({sorted(set(tk) - set(got))} missing/delisted)")
