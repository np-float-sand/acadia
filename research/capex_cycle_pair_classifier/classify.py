"""Capex-cycle-pair classifier -- REFERENCE IMPLEMENTATION (throwaway spike), grid/DER theme.
Rule (loose): theme pure-play  AND  customer axis matches  AND  >=2 of {profit, duration, policy}.
Static classification (characteristic held through most of 2019-2026). Time-varying = stricter follow-up.
Each criterion scored from the LONG perspective: True = looks LONG, False = looks SHORT.
"""
import pandas as pd

# ticker: (theme_pureplay, customer_LONG?, profit_LONG?, duration_LONG?, policy_LONG?, one-line rationale)
C = {
 # ---- institutional-capex supply ----
 "ETN":  (1,1,1,1,1,"electrical eq -> utilities/industrial/DC; profitable; P/E; no subsidy gate"),
 "HUBB": (1,1,1,1,1,"utility & electrical eq; profitable; P/E"),
 "NVT":  (1,1,1,1,1,"electrical connection/protection; profitable"),
 "VRT":  (1,1,1,1,1,"data-center power&cooling; B2B hyperscalers; profitable since ~2022"),
 "GEV":  (1,1,1,1,1,"grid equipment + power; B2B utilities (px from 2024-03)"),
 "PWR":  (1,1,1,1,1,"utility T&D construction/EPC; profitable; P/E"),
 "MYRG": (1,1,1,1,1,"electrical construction; profitable"),
 "PRIM": (1,1,1,1,1,"infrastructure EPC; profitable"),
 "POWL": (1,1,1,1,1,"electrical apparatus for utilities/oil&gas; profitable"),
 "ATKR": (1,1,1,1,1,"electrical conduit/cable; profitable"),
 "WCC":  (1,1,1,1,1,"electrical distribution (WESCO); profitable"),
 "ABBNY":(1,1,1,1,1,"foreign grid equipment; profitable"),
 "SBGSY":(1,1,1,1,1,"Schneider - grid/DC electrical; profitable"),
 "PRYMY":(1,1,1,1,1,"Prysmian - cable; profitable"),
 "HTHIY":(1,1,1,1,1,"Hitachi - grid equipment; profitable"),
 "NXT":  (1,1,1,1,0,"utility-scale solar trackers; B2B developers; profitable; some ITC exposure"),
 "FSLR": (1,1,1,1,0,"utility-scale solar mfr; B2B developers; profitable; heavy 45X credit reliance"),
 # ---- distributed / retail demand ----
 "ENPH": (1,0,0,0,0,"residential solar microinverters; homeowners; long-duration; ITC/NEM-gated (profitable -> profit=LONG)"),
 "SEDG": (1,0,0,0,0,"residential/C&I solar inverters; homeowners; lossmaking since 2023; EV/S; NEM-gated"),
 "RUN":  (1,0,0,0,0,"residential solar lease/finance; homeowners; lossmaking; lease-duration; ITC/NEM-gated"),
 "CHPT": (1,0,0,0,0,"EV charging networks/hw; retail EV demand; lossmaking; EV/S; NEVI exposure"),
 "BLNK": (1,0,0,0,0,"EV charging hw/network; retail; lossmaking; EV/S; grant-dependent"),
 "EVGO": (1,0,0,0,0,"consumer EV charging network; lossmaking; EV/S; NEVI-dependent"),
 "WBX":  (1,0,0,0,0,"home/SMB EV chargers (Wallbox); lossmaking; EV/S; subsidy-exposed"),
 "STEM": (1,0,0,0,0,"behind-the-meter storage+software; C&I SMB; lossmaking; EV/S; ITC-gated"),
 # ---- deliberately ambiguous: customer + <2 of 3 -> EXCLUDED from both ----
 "FLNC": (1,1,0,0,0,"grid-scale storage; B2B utilities/developers (customer=LONG) BUT lossmaking+EV/S+ITC -> only 1/3"),
 "FTCI": (1,1,0,0,0,"utility-scale solar trackers; B2B developers (customer=LONG) BUT lossmaking+EV/S -> 1/3"),
 "ARRY": (1,1,0,1,0,"utility-scale solar BOS; B2B developers; thin/again-lossmaking; 45X exposure -> 2/3? borderline"),
 "SHLS": (1,1,1,0,0,"utility-scale solar BOS (Shoals); B2B developers; profitable but EV/S + tariff-exposed -> 2/3 borderline"),
 "GNRC": (1,0,1,1,0,"home backup gen + residential solar/storage; homeowners (customer=SHORT) BUT profitable+P/E -> only 1/3 SHORT"),
}

rows=[]
for t,(pp,cust,prof,dur,pol,why) in C.items():
    long_conf = prof+dur+pol            # # of confirmors pointing LONG
    short_conf = (1-prof)+(1-dur)+(1-pol)
    if not pp:
        sleeve="excluded (not pure-play)"
    elif cust==1 and long_conf>=2:
        sleeve="LONG"
    elif cust==0 and short_conf>=2:
        sleeve="SHORT"
    else:
        sleeve="excluded (customer + <2/3)"
    rows.append(dict(ticker=t,sleeve=sleeve,cust=("inst" if cust else "retail"),
                     long_conf=long_conf,why=why))
df=pd.DataFrame(rows).sort_values(["sleeve","ticker"])
df.to_csv('research/capex_cycle_pair_classifier/_cache/capex_pair_classification.csv',index=False)
for s in ["LONG","SHORT","excluded (customer + <2/3)","excluded (not pure-play)"]:
    sub=df[df.sleeve==s]
    print(f"\n### {s}  (n={len(sub)}) ###")
    for _,r in sub.iterrows(): print(f"  {r.ticker:6s} [{r.cust:5s}] {r.why}")
