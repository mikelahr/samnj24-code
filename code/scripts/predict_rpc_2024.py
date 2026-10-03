"""2024 RPC predictions for the US, states and counties on the 421-industry axis.
Replicates rpc.operations.calculate_rpc with the RECON2024 settings, except where noted:
  * model: process/RPC/rpc_model.pickle (fractional probit fitted on the culled FAF6 2022 RPC)
  * inputs: process/SUT/supply_demand_2024.csv (supply = output - international exports,
    demand_domestic = use net of international imports, QCEW 2024) -- all 2024;
    us_s_wv_ratio = FAF6 2022 (process/RPC/us_s_wv_ratio.csv); landshare from process/RPC/area.csv
  * demand = demand_domestic (pipeline.yaml calculate_rpc.demand: domestic), for both l_demand and
    the supply/demand ratio  [calculate_rpc still computes supply + imports -- legacy]
  * esc/pac regional dummies assigned to counties through their state  [calculate_rpc matches the
    region code itself, so counties never received them]
  * post-processing per pipeline.yaml: structural_zero, services_supply_demand (sctg 0/42/72),
    NaN -> clipped supply/demand, lodging_override (721000), upper-bound squeeze on
    supply/demand cells only, final clip to [0,1]."""
import numpy as np, pandas as pd, statsmodels.api as sm, yaml
P='../'
v=pd.read_csv(P+'SUT/supply_demand_2024.csv',dtype={'region':str,'industry':str,'sctg':str})
m=sm.load('rpc_model.pickle'); exo=yaml.safe_load(open('rpc_names.yaml'))['exo']
wv=pd.read_csv('us_s_wv_ratio.csv',dtype={'sctg':str}).set_index('sctg').us_s_wv_ratio
area=pd.read_csv('area.csv',dtype={'code':str}).set_index('code').km2
v['demand']=v.demand_domestic
v['supplydemand']=np.where(v.demand>0,v.supply/v.demand.where(v.demand>0),0.0)
k=v.region+'_'+v.industry; v.index=k
US=v[v.region=='00000'].set_index('industry')
v['emp_estab']=np.where(v.qcew_establishments>0,v.qcew_employment/v.qcew_establishments.where(v.qcew_establishments>0),0.0)
use=US.qcew_employment/US.qcew_establishments
v['emp_estab_ratio']=v.emp_estab/v.industry.map(use)
v['emp_estab_ratio']=v.emp_estab_ratio.replace([np.inf,-np.inf],0).fillna(0)
for c in ['supply','demand']:
    t=v[c].copy(); t[t==0]=1.0; v['l_'+c]=np.log(t)
tot=v.groupby('region').qcew_employment.transform('sum'); v['empshare']=np.where(tot>0,v.qcew_employment/tot,0.0)
v['empLQ']=v.empshare/v.industry.map(v[v.region=='00000'].set_index('industry').empshare)
v['empLQ']=v.empLQ.replace([np.inf,-np.inf],0).fillna(0)
wt=v.groupby('region').qcew_wages.transform('sum'); v['wageshare']=np.where(wt>0,v.qcew_wages/wt,0.0)
L=v[v.industry=='721000'].set_index('region')
v['lodging_empLQ']=v.region.map(L.empLQ); v['lodging_wageshare']=v.region.map(L.wageshare)
v['landshare']=v.region.map(area)/area.loc['00000']
st=v.region.str[:2]+'000'
v['esc']=st.isin(['47000','01000','21000','28000']).astype(int); v['pac']=st.isin(['53000','06000','15000','02000','41000']).astype(int)
v.loc[v.region=='00000',['esc','pac']]=0
D={'animals':['1'],'aggood':['5','6','3','7','2','22','8','4'],'woodpaper':['28','29','27','26'],'metalequip':['34','33'],
   'sctg10':['10'],'sctg17':['17'],'sctg30':['30'],'sctg31':['31'],'sctg37':['37'],'sctg38':['38'],'sctg39':['39']}
for kk,cc in D.items(): v[kk]=v.sctg.isin(cc).astype(int)
v['us_s_wv_ratio']=v.sctg.map(wv); v['const']=1.0
X=v[exo]; ok=X.notna().all(1)&np.isfinite(X).all(1)
v['rpc_model']=np.nan; v.loc[ok,'rpc_model']=m.predict(X[ok])
v['rpc']=v.rpc_model; v['rpc_source']=np.where(ok,'model','')
z=v.supply<=0; v.loc[z,'rpc']=0.0; v.loc[z,'rpc_source']='structural_zero'
sd=v.sctg.isin(['0','42','72']); v.loc[sd,'rpc']=v.loc[sd,'supplydemand']; v.loc[sd,'rpc_source']='supply_demand'
na=v.rpc.isna(); v.loc[na,'rpc']=v.loc[na,'supplydemand'].clip(0,1); v.loc[na,'rpc_source']='supply_demand_fallback'
lo=v.industry=='721000'
a=lo&(v.lodging_empLQ<=1.42); v.loc[a,'rpc']=0.5*v.loc[a,'landshare']/0.015
b=lo&(v.lodging_empLQ>1.42); v.loc[b,'rpc']=v.loc[b,'lodging_empLQ']/1.42; v.loc[lo,'rpc_source']='lodging_override'
sq=v.rpc_source.isin(['supply_demand','supply_demand_fallback'])&(v.rpc>0.95); v.loc[sq,'rpc']=1-np.exp(-3.154*v.loc[sq,'rpc'])
v['rpc']=v.rpc.clip(0,1)
out=v[['region','level','industry','sctg','supply','demand','supplydemand','qcew_employment','emp_estab','emp_estab_ratio','empLQ',
       'lodging_empLQ','lodging_wageshare','landshare','esc','pac','rpc_model','rpc_source','rpc']]
out.to_csv('predict_2024/rpc_predictions_2024.csv',index=False)
g=out[out.sctg!='0']; lvl={0:'US',1:'states',2:'counties'}
for L_,nm in lvl.items():
    x=out[out.level==L_]; gx=x[~x.sctg.isin(['0','42','72'])]
    print(f"{nm}: cells {len(x)} | goods mean rpc {gx.rpc.mean():.3f} (supply>0: {gx[gx.supply>0].rpc.mean():.3f}) | services mean {x[x.sctg.isin(['0','42','72'])].rpc.mean():.3f} | sources {x.rpc_source.value_counts().to_dict()}")
# 2026-09-21: Connecticut and Alaska counties bypassed for now (county geography changed);
# their state rows are kept. See rpc_predictions_2024_noCTAKcounties.csv.
