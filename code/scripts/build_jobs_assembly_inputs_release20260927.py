"""Inputs for bea.assemble_employment from the 27 Sep 2026 BEA-jobs release (BEA_EMP/release_20260927).
  county proprietors  <- county_sector_nonfarm_proprietors (nonfarm lines); farm line 81 rows kept from the FINAL file
  state proprietors   <- sum of the county file by state
  state uncovered W&S <- state_sector series noncovered - state proprietors (floored at 0; where the
                         county proprietor detail exceeds noncovered, proprietors are cut to noncovered)
Same column layout as the files they replace."""
import os, pandas as pd, numpy as np
H=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(H,'..'); R=os.path.join(H,'..','..','..','BEA_EMP','release_20260927')
cs=pd.read_csv(os.path.join(R,'county_sector_nonfarm_proprietors_2022_2024_2026-09-27_1248Z.csv'),dtype={'GeoFIPS':str})
ss=pd.read_csv(os.path.join(R,'state_sector_private_nonfarm_jobs_2021_2024_2026-09-27_1535Z.csv'),dtype={'st':str})
old_c=pd.read_csv(os.path.join(P,'FINAL_prop_jobs_county_sector_2022_2024.csv'),dtype={'GeoFIPS':str,'lc':str})
old_s=pd.read_csv(os.path.join(P,'FINAL_prop_jobs_income_state_sector_2022_2024.csv'),dtype={'state':str,'lc':str})
old_u=pd.read_csv(os.path.join(P,'uncovered_ws_state_sector_2022_2024.csv'),dtype={'st':str,'lc':str})
nonfarm=[str(l) for l in sorted(ss.lc.unique())]
cs['lc']=cs.lc.astype(str); ss['lc']=ss.lc.astype(str)
cs=cs[cs.lc.isin(nonfarm)]
# state proprietors from counties
cs['state']=cs.GeoFIPS.str[:2]+'000'
ps=cs.groupby(['state','lc','year'],as_index=False)[['prop_jobs','proprietors_income_k']].sum()
ss=ss[ss.year.isin([2022,2023,2024])].copy(); ss['state']=ss.st.str.zfill(2)+'000'
m=ss.merge(ps,on=['state','lc','year'],how='left').fillna({'prop_jobs':0.0})
neg=m.prop_jobs>m.noncovered
print('cells where county proprietors exceed noncovered:', m.groupby('year').apply(lambda d:(d.prop_jobs>d.noncovered).sum()).to_dict(),
      'jobs', round(float((m.prop_jobs-m.noncovered).clip(lower=0).sum()),0))
cut=np.where(m.prop_jobs>0, np.minimum(1.0, m.noncovered/m.prop_jobs.replace(0,np.nan)), 1.0)
m['cut']=np.nan_to_num(cut, nan=1.0)
m['prop_adj']=m.prop_jobs*m.cut
m['u']=(m.noncovered-m.prop_adj).clip(lower=0.0)
# county file scaled by the same cut
cs=cs.merge(m[['state','lc','year','cut']],on=['state','lc','year'],how='left').fillna({'cut':1.0})
cs['prop_jobs']=cs.prop_jobs*cs.cut
farm_c=old_c[~old_c.lc.isin(nonfarm)]; farm_s=old_s[~old_s.lc.isin(nonfarm)]; farm_u=old_u[~old_u.lc.isin(nonfarm)]
outc=pd.concat([cs[['GeoFIPS','lc','prop_jobs','year']],farm_c],ignore_index=True)
outs=pd.concat([m.rename(columns={'prop_adj':'x'})[['state','lc','year']].assign(prop_jobs=m.prop_adj.values,proprietors_income_k=m.proprietors_income_k.values),farm_s],ignore_index=True)
outu=pd.concat([m.assign(st=m.st.str.zfill(2),basis='BEA-jobs release 2026-09-27: state noncovered less county-sum proprietors')[['st','lc','u','year','basis']],farm_u],ignore_index=True)
outc.to_csv(os.path.join(H,'prop_jobs_county_sector_2022_2024_release20260927.csv'),index=False)
outs.to_csv(os.path.join(H,'prop_jobs_income_state_sector_2022_2024_release20260927.csv'),index=False)
outu.to_csv(os.path.join(H,'uncovered_ws_state_sector_2022_2024_release20260927.csv'),index=False)
for y in (2022,2024):
    a=m[m.year==y]; ou=old_u[(old_u.year==y)&old_u.lc.isin(nonfarm)].u.sum(); op=old_s[(old_s.year==y)&old_s.lc.isin(nonfarm)].prop_jobs.sum()
    print(y,'US uncovered new/old',round(a.u.sum()),round(ou),' proprietors new/old',round(a.prop_adj.sum()),round(op))
    nj=a[a.state=='34000']; print('  NJ uncovered',round(nj.u.sum()),' NJ proprietors',round(nj.prop_adj.sum()))
