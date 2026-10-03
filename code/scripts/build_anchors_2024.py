"""FAF6-based 2024 RPC anchors, state x SCTG and FAF6 zone x SCTG.
1. FAF6.0 2022 flows (value, 2022$; tons) by component and domestic mode.
2. Carried to 2024 by FAF5.7.1 current-dollar growth per cell & component (FAF5 190 -> FAF6 191/199; 399 -> 395/399);
   fallback: national SCTG x component growth.
3. Cull of nonproducer short-haul intra shipments: CFS 2022 value share removed, by origin state x SCTG x mode
   (national SCTG x mode where the state cell is thin); first-handler exemption SCTG 1,2,3,41. 2022 shares held for 2024.
4. rpc = intra_c/(intra_c+in_dom+imp_for); structural zero where supply = 0; thin-cell flag + credibility weight.
"""
import numpy as np, pandas as pd
FAF6='/home/claude/faf/FAF6.0.csv'
OD5='/mnt/user-data/uploads/RPC_explorations/_stage_tmp/faf5_od_2022_2024.csv.gz'
EXEMPT={1,2,3,41}; SCTG=list(range(1,42)); THIN_CFS=5.0; K=50.0    # $M
CMODE={1:1,2:2,3:3,4:4,5:5,6:6,7:7}
f=pd.read_csv(FAF6,dtype={'dms_orig':int,'dms_dest':int,'sctg2':int})
f=f[f.sctg2.isin(SCTG)].rename(columns={'sctg2':'sctg','value_2022':'v22','tons_2022':'t22'})
# ---- FAF5.7.1 growth, by FAF6 geography ----
o=pd.read_csv(OD5); o=o[o.sctg.isin(SCTG)]
def z5to6(z): return {190:[191,199],399:[395,399]}.get(z,[z])
def comp(d,a,b):
    return np.select([(d.trade_type==1)&(d[a]==d[b]),d.trade_type==1,d.trade_type==2,d.trade_type==3],
                     ['intra','dom','imp','exp'],'x')
o['c']=comp(o,'dms_orig','dms_dest')
# component x origin-side / dest-side keys: intra (z), out_dom (orig), in_dom (dest), exp (orig), imp (dest)
def growth(o,geo):
    g=o.copy()
    if geo=='state': g['zo']=g.dms_orig//10; g['zd']=g.dms_dest//10
    else: g['zo']=g.dms_orig; g['zd']=g.dms_dest
    parts=[]
    for c,key,side in [('intra','intra','zo'),('dom','out_dom','zo'),('dom','in_dom','zd'),('exp','exp_for','zo'),('imp','imp_for','zd')]:
        s=g[g.c==c] if not (geo=='state' and c in('intra','dom')) else None
        if geo=='state':   # at state level intra = same state, dom = different state
            same=(g.trade_type==1)&(g.zo==g.zd)
            s={'intra':g[same],'dom':g[(g.trade_type==1)&~same],'exp':g[g.trade_type==3],'imp':g[g.trade_type==2]}[c]
        a=s.groupby([side,'sctg'])[['current_value_2022','current_value_2024']].sum().reset_index().rename(columns={side:'geo'})
        a['comp']=key; parts.append(a)
    G=pd.concat(parts)
    if geo=='zone':
        G=pd.concat([G.assign(geo=z) for zz,G1 in [(None,G)] for z in [None]] ) if False else G
        rows=[]
        for z,sub in G.groupby('geo'):
            for z6 in z5to6(z): rows.append(sub.assign(geo=z6))
        G=pd.concat(rows)
    nat=G.groupby(['comp','sctg'])[['current_value_2022','current_value_2024']].sum()
    nat['g_nat']=nat.current_value_2024/nat.current_value_2022
    G['g']=np.where(G.current_value_2022>0.01,G.current_value_2024/G.current_value_2022.where(G.current_value_2022>0.01),np.nan)
    G=G.join(nat['g_nat'],on=['comp','sctg']); G['g']=G.g.fillna(G.g_nat)
    return G[['geo','sctg','comp','g']], nat['g_nat']
# ---- cull shares by state x sctg x mode ----
cm=pd.read_csv('cfs2022_cull_state_sctg_mode.csv',dtype={'SCTG':str}); cm=cm[cm.SCTG.str.match(r'^\d+$')]
cm=cm.assign(st=cm.ORIG_STATE.astype(int),sctg=cm.SCTG.astype(int),mode=cm.fmode.astype(int))
cm=cm[cm.sctg.isin(SCTG)]
nat=cm.groupby(['sctg','mode'])[['v','vc','w','wc']].sum(); nat['s_nat']=nat.vc/nat.v; nat['sw_nat']=nat.wc/nat.w
cm=cm.join(nat[['s_nat','sw_nat']],on=['sctg','mode'])
cm['s']=np.where(cm.v>=THIN_CFS,cm.vc/cm.v,cm.s_nat); cm['sw']=np.where(cm.w>0,np.where(cm.v>=THIN_CFS,cm.wc/cm.w,cm.sw_nat),cm.sw_nat)
cm['s_src']=np.where(cm.v>=THIN_CFS,'state','national')
CM=cm.set_index(['st','sctg','mode'])[['s','sw','s_src']]
NAT=nat[['s_nat','sw_nat']]
def build(geo):
    d=f.copy()
    if geo=='state': d['go']=d.dms_orig//10; d['gd']=d.dms_dest//10
    else: d['go']=d.dms_orig; d['gd']=d.dms_dest
    same=(d.trade_type==1)&(d.go==d.gd)
    # intra by mode with cull
    I=d[same].groupby(['go','sctg','dms_mode'])[['v22','t22']].sum().reset_index().rename(columns={'go':'geo','dms_mode':'mode'})
    I['st']=I.geo if geo=='state' else I.geo//10
    I=I.join(CM,on=['st','sctg','mode']).join(NAT,on=['sctg','mode'])
    I['s']=I.s.fillna(I.s_nat).fillna(0.0); I['sw']=I.sw.fillna(I.sw_nat).fillna(0.0)
    I.loc[I.sctg.isin(EXEMPT),['s','sw']]=0.0
    I['v22_c']=I.v22*(1-I.s); I['t22_c']=I.t22*(1-I.sw)
    I['v_nat_fallback']=np.where(I.s_src.eq('state'),0.0,I.v22)
    A=I.groupby(['geo','sctg'])[['v22','t22','v22_c','t22_c','v_nat_fallback']].sum().rename(columns={'v22':'intra','t22':'intra_t','v22_c':'intra_c','t22_c':'intra_t_c'})
    comps={'out_dom':d[(d.trade_type==1)&~same].groupby(['go','sctg'])[['v22','t22']].sum(),
           'in_dom':d[(d.trade_type==1)&~same].groupby(['gd','sctg'])[['v22','t22']].sum(),
           'exp_for':d[d.trade_type==3].groupby(['go','sctg'])[['v22','t22']].sum(),
           'imp_for':d[d.trade_type==2].groupby(['gd','sctg'])[['v22','t22']].sum()}
    geos=sorted(set(d.go)|set(d.gd))
    X=pd.MultiIndex.from_product([geos,SCTG],names=['geo','sctg']).to_frame(index=False).set_index(['geo','sctg'])
    X=X.join(A)
    for k,v in comps.items():
        v.index.names=['geo','sctg']; X=X.join(v.rename(columns={'v22':k,'t22':k+'_t'}))
    X=X.fillna(0.0).reset_index()
    G,gn=growth(o,geo); G=G.pivot_table(index=['geo','sctg'],columns='comp',values='g')
    X=X.join(G.add_prefix('g_'),on=['geo','sctg'])
    for k in ['intra','out_dom','in_dom','exp_for','imp_for']:
        X['g_'+k]=X['g_'+k].fillna(X.sctg.map(gn.xs(k).to_dict()))
    out=X[['geo','sctg']].copy()
    for yr in (2022,2024):
        m=(lambda k: X['g_'+k]) if yr==2024 else (lambda k: 1.0)
        ic=X.intra_c*m('intra'); ir=X.intra*m('intra'); idm=X.in_dom*m('in_dom'); imp=X.imp_for*m('imp_for')
        sup=ic+X.out_dom*m('out_dom')+X.exp_for*m('exp_for'); dem=ic+idm+imp
        out[f'intra_raw_{yr}']=ir; out[f'intra_c_{yr}']=ic; out[f'in_dom_{yr}']=idm; out[f'imp_for_{yr}']=imp
        out[f'out_dom_{yr}']=X.out_dom*m('out_dom'); out[f'exp_for_{yr}']=X.exp_for*m('exp_for')
        out[f'supply_{yr}']=sup; out[f'demand_{yr}']=dem
        out[f'rpc_raw_{yr}']=np.where(ir+idm+imp>0,ir/(ir+idm+imp).where(ir+idm+imp>0),np.nan)
        out[f'rpc_{yr}']=np.where(dem>0,ic/dem.where(dem>0),np.nan)
    # tonnage (2022 only; FAF5.7.1 OD extract carries value only) for W/V checks
    dem_t=X.intra_t_c+X.in_dom_t+X.imp_for_t
    out['rpc_tons_2022']=np.where(dem_t>0,X.intra_t_c/dem_t.where(dem_t>0),np.nan)
    out['wv_intra_raw']=np.where(X.intra>0,X.intra_t/X.intra.where(X.intra>0),np.nan)
    out['wv_intra_c']=np.where(X.intra_c>0,X.intra_t_c/X.intra_c.where(X.intra_c>0),np.nan)
    out['cull_share_value']=np.where(X.intra>0,1-X.intra_c/X.intra.where(X.intra>0),np.nan)
    out['cull_share_natfallback']=np.where(X.intra>0,X.v_nat_fallback/X.intra.where(X.intra>0),np.nan)
    # three-way rule
    s=out.supply_2024; dm=out.demand_2024
    out['status']=np.select([s<=0,dm<=0,dm<K/5],['structural_zero','no_demand','thin'],'observed')
    out.loc[s<=0,'rpc_2024']=0.0
    out['cred_weight']=np.where(s<=0,1.0,dm/(dm+K))      # shrink: rpc* = w*rpc_obs + (1-w)*rpc_model
    return out.rename(columns={'geo':'state_fips' if geo=='state' else 'faf6_zone'})
S=build('state'); Z=build('zone')
S.to_csv('rpc_anchor_state_sctg_2024_FAF6.csv',index=False); Z.to_csv('rpc_anchor_zone_sctg_2024_FAF6.csv',index=False)
for n,d in [('STATE',S),('ZONE',Z)]:
    ok=d.status.isin(['observed','thin'])
    print(f"\n{n}: cells {len(d)}  status {d.status.value_counts().to_dict()}")
    print(f"  mean rpc (supply>0, demand>0): raw22 {d.loc[ok,'rpc_raw_2022'].mean():.3f}  culled22 {d.loc[ok,'rpc_2022'].mean():.3f}  culled24 {d.loc[ok,'rpc_2024'].mean():.3f}  tons22 {d.loc[ok,'rpc_tons_2022'].mean():.3f}")
    print(f"  22->24 corr {d.loc[ok,['rpc_2022','rpc_2024']].corr().iloc[0,1]:.4f}  mean|d| {(d.loc[ok,'rpc_2024']-d.loc[ok,'rpc_2022']).abs().mean():.4f}   value-vs-tons corr {d.loc[ok,['rpc_2022','rpc_tons_2022']].corr().iloc[0,1]:.3f}")
    print(f"  intra value culled: {1-d.intra_c_2022.sum()/d.intra_raw_2022.sum():.3f};  W/V intra raw {d.intra_raw_2022.sum() and (S if n=='STATE' else Z).shape[0]}")
    print(f"  share of culled intra value relying on national-fallback cull share: {(d.cull_share_natfallback*d.intra_raw_2022).sum()/d.intra_raw_2022.sum():.3f}")
