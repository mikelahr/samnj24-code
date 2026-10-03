"""Rake 2024 goods RPC predictions to the FAF6-based 2024 anchors (culled, value share by mode).
States: for each state x SCTG, shift the model cells' probit index by one delta so that the
  demand-weighted mean RPC of all industry cells of that SCTG equals the target
  A* = w*A + (1-w)*R_pred  (A = anchor, domestic definition intra_c/(intra_c+in_dom); w = anchor
  credibility demand/(demand+$50M); R_pred = current demand-weighted mean). Same geography -> exact.
Counties: a zone's RPC is not an average of its counties' RPCs (cross-county flows inside a zone are
  local at zone scale), so counties take the zone's residual instead: delta_z = w*(Phi^-1(A_z) -
  Phi^-1(P_z)), where P_z is the equation's prediction for the zone x SCTG aggregate (2024 regressors
  built from the zone's counties). Applied to the model cells of every county in the zone.
Structural zeros, services (supply/demand), lodging, and cells without an anchor are unchanged.
Anchor cells with status structural_zero/no_demand (FAF sees no supply or demand) are not used.
CT and AK counties are bypassed."""
import numpy as np, pandas as pd, statsmodels.api as sm, yaml
from scipy.stats import norm
from scipy.optimize import brentq
R='../'; AN='../../../../anchors_2024_FAF6/'
p=pd.read_csv('rpc_predictions_2024.csv',dtype={'region':str,'industry':str,'sctg':str})
v=pd.read_csv(R+'../SUT/supply_demand_2024.csv',dtype={'region':str,'industry':str,'sctg':str})
p=p.merge(v[['region','industry','qcew_establishments','qcew_wages']],on=['region','industry'],how='left')
m=sm.load(R+'rpc_model.pickle'); exo=yaml.safe_load(open(R+'rpc_names.yaml'))['exo']
wv=pd.read_csv(R+'us_s_wv_ratio.csv',dtype={'sctg':str}).set_index('sctg').us_s_wv_ratio
area=pd.read_csv(R+'area.csv',dtype={'code':str}).set_index('code').km2
xw=pd.read_csv('county_to_faf6_zone.csv',dtype={'region':str}).set_index('region').faf6_zone
GOODS=~p.sctg.isin(['0','42','72'])
def anchors(fn,key):
    A=pd.read_csv(AN+fn); den=A.intra_c_2024+A.in_dom_2024
    A['A']=np.where(den>0,A.intra_c_2024/den.where(den>0),np.nan); A['sctg']=A.sctg.astype(str)
    A=A[A.status.isin(['observed','thin'])]; return A.set_index([key,'sctg'])[['A','cred_weight','status']]
AS=anchors('rpc_anchor_state_sctg_2024_FAF6.csv','state_fips'); AZ=anchors('rpc_anchor_zone_sctg_2024_FAF6.csv','faf6_zone')
Phi,iPhi=norm.cdf,lambda x: norm.ppf(np.clip(x,1e-6,1-1e-6))
p['rpc_raked']=p.rpc; p['rake_delta']=0.0
# ---------------- states: exact rake ----------------
S=p[(p.level==1)&GOODS]; rows=[]
for (reg,sc),g in S.groupby(['region','sctg']):
    key=(int(reg[:2]),sc)
    if key not in AS.index: continue
    A,w,_=AS.loc[key]; d=g.demand.clip(lower=0); D=d.sum()
    if D<=0: continue
    mod=g.rpc_source.eq('model')&g.rpc_model.notna()
    if not mod.any(): continue
    Rp=(g.rpc*d).sum()/D; T=w*A+(1-w)*Rp
    fixed=(g.rpc[~mod]*d[~mod]).sum(); z0=iPhi(g.rpc_model[mod].values); dm=d[mod].values
    f=lambda dl: (fixed+(Phi(z0+dl)*dm).sum())/D-T
    lo,hi=f(-8),f(8)
    dl=brentq(f,-8,8) if lo<0<hi else (-8 if lo>=0 else 8)
    idx=g.index[mod]; p.loc[idx,'rpc_raked']=Phi(z0+dl); p.loc[idx,'rake_delta']=dl
    rows.append(dict(region=reg,sctg=sc,anchor=A,w=w,pred_before=Rp,target=T,after=(fixed+(Phi(z0+dl)*dm).sum())/D,delta=dl))
SR=pd.DataFrame(rows); SR.to_csv('rake_state_sctg_2024.csv',index=False)
print('STATES: cells raked',len(SR),'| before: corr %.3f MAE %.3f | after: corr %.3f MAE %.3f | median |delta| %.2f'%(
   SR[['pred_before','anchor']].corr().iloc[0,1],(SR.pred_before-SR.anchor).abs().mean(),SR[['after','anchor']].corr().iloc[0,1],(SR.after-SR.anchor).abs().mean(),SR.delta.abs().median()))
# ---------------- counties: zone residual ----------------
C=p[(p.level==2)&~p.region.str[:2].isin(['09','02'])].copy(); C['zone']=C.region.map(xw)
US=p[p.level==0]
def agg_X(df,key):
    """equation regressors for key x SCTG aggregates built from df (industry rows)."""
    tot=df.groupby(key)[['qcew_employment','qcew_wages']].sum()
    lod=df[df.industry=='721000'].groupby(key)[['qcew_employment','qcew_wages']].sum()
    G=df[~df.sctg.isin(['0','42','72'])].groupby([key,'sctg'])[['supply','demand','qcew_employment','qcew_establishments']].sum().reset_index()
    ut=US.qcew_employment.sum(); ul=US[US.industry=='721000'].qcew_employment.sum()
    uG=US.groupby('sctg')[['qcew_employment','qcew_establishments']].sum()
    G['emp_estab']=np.where(G.qcew_establishments>0,G.qcew_employment/G.qcew_establishments.where(G.qcew_establishments>0),0)
    G['emp_estab_ratio']=G.emp_estab/G.sctg.map(uG.qcew_employment/uG.qcew_establishments)
    G['empLQ']=(G.qcew_employment/G[key].map(tot.qcew_employment))/G.sctg.map(uG.qcew_employment/ut)
    G['lodging_empLQ']=(G[key].map(lod.qcew_employment)/G[key].map(tot.qcew_employment))/(ul/ut)
    G['lodging_wageshare']=G[key].map(lod.qcew_wages)/G[key].map(tot.qcew_wages)
    for c in ['supply','demand']:
        t=G[c].where(G[c]!=0,1.0); G['l_'+c]=np.log(t)
    return G
G=agg_X(C,'zone')
za=C.groupby('zone').region.apply(lambda r: area.reindex(r.unique()).sum()); G['landshare']=G.zone.map(za)/area.loc['00000']
zst=C.groupby('zone').region.first().str[:2]+'000'
G['esc']=G.zone.map(zst).isin(['47000','01000','21000','28000']).astype(int); G['pac']=G.zone.map(zst).isin(['53000','06000','15000','02000','41000']).astype(int)
Dm={'animals':['1'],'aggood':['5','6','3','7','2','22','8','4'],'woodpaper':['28','29','27','26'],'metalequip':['34','33'],
    'sctg10':['10'],'sctg17':['17'],'sctg30':['30'],'sctg31':['31'],'sctg37':['37'],'sctg38':['38'],'sctg39':['39']}
for k,c in Dm.items(): G[k]=G.sctg.isin(c).astype(int)
G['us_s_wv_ratio']=G.sctg.map(wv); G['const']=1.0
ok=G[exo].notna().all(1)&np.isfinite(G[exo]).all(1)&(G.supply>0)
G['P_zone']=np.nan; G.loc[ok,'P_zone']=m.predict(G.loc[ok,exo])
G=G.join(AZ,on=['zone','sctg'])
G['delta']=np.where(G.A.notna()&G.P_zone.notna(),G.cred_weight*(iPhi(G.A)-iPhi(G.P_zone)),0.0)
G[['zone','sctg','supply','demand','P_zone','A','cred_weight','status','delta']].to_csv('rake_zone_sctg_2024.csv',index=False)
hv=G.dropna(subset=['A','P_zone'])
print('ZONES: zone x SCTG with anchor and prediction',len(hv),'| P_zone vs anchor corr %.3f MAE %.3f bias %+.3f | median |delta| %.2f'%(
   hv[['P_zone','A']].corr().iloc[0,1],(hv.P_zone-hv.A).abs().mean(),(hv.P_zone-hv.A).mean(),hv.delta.abs().median()))
dz=G.set_index(['zone','sctg']).delta
C['delta']=[dz.get((z,s),0.0) if pd.notna(z) else 0.0 for z,s in zip(C.zone,C.sctg)]
mod=C.rpc_source.eq('model')&C.rpc_model.notna()
C.loc[mod,'rpc_raked']=Phi(iPhi(C.loc[mod,'rpc_model'])+C.loc[mod,'delta']); C.loc[mod,'rake_delta']=C.loc[mod,'delta']
p.loc[C.index,['rpc_raked','rake_delta']]=C[['rpc_raked','rake_delta']]
p['faf6_zone']=p.region.map(xw).where(p.level==2)
p['bypassed']=(p.level==2)&p.region.str[:2].isin(['09','02'])
out=p.drop(columns=['qcew_establishments','qcew_wages'])
out.to_csv('rpc_predictions_2024_raked.csv',index=False)
for lv,nm in [(1,'states'),(2,'counties')]:
    x=out[(out.level==lv)&~out.bypassed&GOODS&(out.supply>0)]
    print(f'{nm} goods (supply>0): mean rpc before {x.rpc.mean():.3f} after {x.rpc_raked.mean():.3f}; demand-weighted before {(x.rpc*x.demand).sum()/x.demand.sum():.3f} after {(x.rpc_raked*x.demand).sum()/x.demand.sum():.3f}')
