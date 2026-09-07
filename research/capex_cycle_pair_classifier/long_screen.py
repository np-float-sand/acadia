import yfinance as yf, pandas as pd, warnings
warnings.simplefilter("ignore")

CAND={
 "AI / data-center physical buildout":['VRT','MOD','ANET','CIEN','COHR','APH','FIX','EME','APG','IESC','AAON','CARR'],
 "Semicap (fab capex cycle)":['AMAT','LRCX','KLAC','ONTO','ACLS','KLIC','AEIS','UCTT','ICHR','CCMP','ENTG'],
 "Aerospace & defense supply":['HEI','TDG','HWM','CW','AXON','MRCY','WWD','TXT','HXL'],
 "Reshoring / E&C / automation":['ACM','J','STRL','PWR','MTZ','ROK','EMR','NDSN','PH','DOV','AME'],
 "Water infrastructure (utility capex)":['XYL','BMI','MWA','FELE','PNR','AOS','ROP'],
 "Rail / freight infra":['WAB','GATX','TRN'],
 "Grid equipment (intl / add'l)":['LGRDY','SIEGY','HPS-A.TO','NVT','ETN','HUBB','POWL','ATKR','WCC','PRYMY'],
 "Electrification / thermal":['CR','SPXC','GTLS','FLS','ITT','GGG'],
}
allt=sorted({t for v in CAND.values() for t in v})
rows=[]
for t in allt:
    try:
        i=yf.Ticker(t).info
        pe=i.get('trailingPE'); om=i.get('operatingMargins'); pm=i.get('profitMargins')
        fcf=i.get('freeCashflow'); mc=i.get('marketCap')
        passes = (pe is not None and pe>0) and (om is not None and om>0) and (fcf is not None and fcf>0)
        rows.append(dict(t=t,mktcap_B=(round(mc/1e9,1) if mc else None),trailingPE=(round(pe,1) if pe else None),
                         op_margin=(round(om*100,1) if om is not None else None),
                         fcf_B=(round(fcf/1e9,2) if fcf else None),PASS=passes))
    except Exception as e:
        rows.append(dict(t=t,PASS=None,note=str(e)[:40]))
df=pd.DataFrame(rows).set_index('t')
for th,lst in CAND.items():
    sub=df.loc[[x for x in lst if x in df.index]]
    p=sub[sub.PASS==True]
    print(f"\n### {th}")
    print(sub[['mktcap_B','trailingPE','op_margin','fcf_B','PASS']].to_string())
