"""Parse PJM Table B-9 (Adjustments Above Embedded to Summer Peak, MW, by zone & forecast year)
from the 2021-2024 Load Forecast Report PDFs and the 2025-2026 supplementary xlsx.
Emit one tidy long CSV: vintage, zone, target_year, mw_adj."""
import re, pandas as pd, pdfplumber, glob

PDF_DIR="research/capex_cycle_pair_classifier/_cache/pjm_lf"
XLS_DIR="research/capex_cycle_pair_classifier/_cache/pjm_xls"
ZONES={"AE","BGE","DPL","JCPL","METED","PECO","PENLC","PEPCO","PL","PS","RECO","UGI",
       "AEP","APS","ATSI","COMED","DAYTON","DEOK","DLCO","EKPC","OVEC","DOM"}

rows=[]

# ---- 2021-2024 from PDF text (Table B-9 block) ----
starts={"2021":6181,"2022":6186,"2023":6347,"2024":6412}
for y,s in starts.items():
    txt=open(f"{PDF_DIR}/{y}.txt").read().split("\n")   # NOT splitlines(): pdftotext emits \f page breaks
    block=txt[s-1:s+80]
    # stop before the Notes / next table so Table B-10 rows don't leak in
    for j,ln in enumerate(block):
        if j>3 and ("Notes:" in ln or "Table B-10" in ln):
            block=block[:j]; break
    # find the year-header line
    hdr=None
    for ln in block:
        toks=ln.split()
        if len(toks)>=10 and all(re.fullmatch(r"20\d\d",t) for t in toks[:6]):
            hdr=[int(t) for t in toks if re.fullmatch(r"20\d\d",t)]
            break
    for ln in block:
        toks=ln.split()
        if not toks: continue
        z=toks[0]
        if z=="PJM" and len(toks)>1 and toks[1]=="RTO":
            z="PJM RTO"; vals=toks[2:]
        elif z in ZONES:
            vals=toks[1:]
        else:
            continue
        nums=[]
        for v in vals:
            v=v.replace(",","")
            if re.fullmatch(r"-?\d+",v): nums.append(int(v))
        for yr,mw in zip(hdr,nums):
            rows.append((f"{y} LF",z,yr,mw))

# ---- 2025-2026 from xlsx Table B9 ----
for y in ("2025","2026"):
    d=pd.read_excel(f"{XLS_DIR}/{y}-load-report-tables.xlsx",sheet_name="Table B9",header=None)
    # locate header row with years
    hr=None
    for i in range(len(d)):
        vals=[str(x) for x in d.iloc[i].tolist()]
        yrs=[v for v in vals if re.fullmatch(r"20\d\d(\.0)?",v)]
        if len(yrs)>=10: hr=i; break
    years=[int(float(x)) for x in d.iloc[hr].tolist() if re.fullmatch(r"20\d\d(\.0)?",str(x))]
    for i in range(hr+1,len(d)):
        z=str(d.iloc[i,0]).strip()
        if z=="PJM RTO": pass
        elif z not in ZONES: continue
        vals=d.iloc[i,1:1+len(years)].tolist()
        for yr,mw in zip(years,vals):
            if pd.notna(mw): rows.append((f"{y} LF",z,yr,int(round(float(mw)))))

df=pd.DataFrame(rows,columns=["vintage","zone","target_year","mw_adj"])
out="grid_resilience/data/seed/pjm_large_load_b9_vintages.csv"
df.to_csv(out,index=False)
print("rows:",len(df),"-> ",out)

piv=df[df.zone=="PJM RTO"].pivot(index="vintage",columns="target_year",values="mw_adj")
print("\n=== PJM RTO large-load adjustment (MW) by vintage x target year ===")
print(piv[[c for c in (2026,2027,2028,2029,2030,2031,2032) if c in piv.columns]].to_string())

for z in ("DOM","AEP","APS","COMED","PS"):
    p=df[df.zone==z].pivot(index="vintage",columns="target_year",values="mw_adj")
    cols=[c for c in (2027,2028,2029,2030,2031) if c in p.columns]
    print(f"\n=== {z} adjustment (MW) ===")
    print(p[cols].to_string())
