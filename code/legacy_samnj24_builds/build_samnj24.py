"""Assemble the 65-account SAMNJ24 (Table B2) from RECON2024 New Jersey results.
Inputs ($ thousands): results/states/recon2024_34000.csv, results/Amatrices_use/recon2024_Amatrix_34000.csv,
results/Usematrices/recon2024_Use_{total,domestic}_34000.csv.gz, process/SUT/aggregation_bridge.xlsx.
Institutional flows ($ millions) are the 2024 figures of Tables 2-5 of the paper."""
import pandas as pd, numpy as np
import os
R=os.environ.get('RECON_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) + '/'
v=pd.read_csv(R+'results/states/recon2024_34000.csv').set_index('industry')
v=v.drop(index='Households')
ind=list(v.index); x=v.output
A=pd.read_csv(R+'results/Amatrices_use/recon2024_Amatrix_34000.csv',index_col=0).loc[ind,ind]
UT=pd.read_csv(R+'results/Usematrices/recon2024_Use_total_34000.csv.gz',index_col=0)
UD=pd.read_csv(R+'results/Usematrices/recon2024_Use_domestic_34000.csv.gz',index_col=0)
rpc=v.rpc
# --- intermediate
Zdom=pd.DataFrame(A.values*x.values,index=ind,columns=ind)          # US-produced inputs, net of int'l imports
Zloc=Zdom.mul(rpc,axis=0)                                             # NJ-produced
inflow_int=(Zdom-Zloc).sum(0)                                         # from rest of US, by buying industry
int_total=UT.sum(0).reindex(ind).fillna(0)
imp_int_col=(int_total-Zdom.sum(0)).clip(lower=0)                     # int'l imports by buying industry
imp_int_row=(UT-UD).sum(1)                                            # int'l intermediate imports by commodity
# --- final demand by commodity(=industry code)
fd=v[['pce','government','investment','inventory']].copy()
fd['kap']=fd.investment+fd.inventory
fd=fd[['pce','government','kap']]
imp_fin=(v.imports-imp_int_row.reindex(ind).fillna(0)).clip(lower=0)
fdpos=fd.clip(lower=0); tot=fdpos.sum(1)
share=fdpos.div(tot.replace(0,np.nan),axis=0).fillna(0)
imp_fin=np.minimum(imp_fin,tot)                                       # cannot import more than final use
imp_fd=share.mul(imp_fin,axis=0)
dom_fd=fd-imp_fd
loc_fd=dom_fd.mul(rpc,axis=0); inflow_fd=dom_fd-loc_fd
# --- value added and taxes by industry
comp=v.compensation; surp=v.surplus; ntp=v.nettax_production; ntc=v.nettax_commodity
# 2026-09-28: net taxes controlled to BEA (SAGDP6 taxes on production and imports 73,887.881 less subsidies 2,698.740 = $71,189.141M).
# The gap to the state IO accounts goes to taxes on production (commodity taxes stay in purchasers' prices); GOS absorbs it through colres.
BEA_NT=71189141.0   # $ thousands
dnt=BEA_NT-(ntp+ntc).sum(); ntp=ntp+dnt*ntp/ntp.sum()
print('net taxes raised to BEA by $M',round(dnt/1e3,3))
coltot=Zloc.sum(0)+inflow_int+imp_int_col+comp+surp+ntp+ntc
colres=x+ntc-coltot                                                    # rounding residual -> capital
surp=surp+colres
rowtot=x+ntc
outflow=rowtot-Zloc.sum(1)-loc_fd.sum(1)-v.exports
# where local sales + exports exceed output, scale local sales down and treat the excess as inflows
locsales=Zloc.sum(1)+loc_fd.sum(1)
f=((rowtot-v.exports).clip(lower=0)/locsales.replace(0,np.nan)).clip(upper=1).fillna(1)
fix=f<1
print('industries with local sales scaled back:',int(fix.sum()),' $bn moved to inflows:',round(((1-f)*locsales).sum()/1e6,2))
Zfix=Zloc.mul(f,axis=0); inflow_int=inflow_int+(Zloc-Zfix).sum(0); Zloc=Zfix
lf=loc_fd.mul(f,axis=0); inflow_fd=inflow_fd+(loc_fd-lf); loc_fd=lf
outflow=rowtot-Zloc.sum(1)-loc_fd.sum(1)-v.exports
print('min outflow $M after fix',round(outflow.min()/1e3,3))
# --- aggregate to 57 sectors
b=pd.read_excel(R+'process/SUT/aggregation_bridge.xlsx',sheet_name='industries').set_index('offspring_code').parent_code.astype(str)
m57={'111CA':'AG11','113FF':'AG11','211':'MIN21','212':'MIN21','213':'MIN21','313TT':'TEX','315AL':'TEX',
 '324':'PPN','326':'PPN','327':'PPN','3361MV':'TEQ336','3364OT':'TEQ336','481':'TRANS_x484','482':'TRANS_x484',
 '483':'TRANS_x484','485':'TRANS_x484','486':'TRANS_x484','487OS':'TRANS_x484','HS':'RE','ORE':'RE',
 '561':'ADMIN56','562':'ADMIN56','623':'HSA','624':'HSA','711AS':'ARTS71','713':'ARTS71'}
sec=b.reindex(ind).map(lambda c:m57.get(c,c))
sec[sec.index=='424400']='424GR'; sec[sec.index=='624400']='624CDC'
S57=['AG11','MIN21','22','23','311FT','TEX','321','322','323','PPN','325','331','332','333','334','335','TEQ336','337','339',
 '42','424GR','441','4A0','445','452','TRANS_x484','484','GFE','493','511','512','513','514','521CI','523','524','525','RE',
 '532RL','5411','5412OP','5415','55','ADMIN56','61','621','622','HSA','624CDC','ARTS71','721','722','81','GSLG','GSLE','GFGD','GFGN']
assert set(sec.unique())<=set(S57), set(sec.unique())-set(S57)
acc=S57+['LAB','CAP','HH','CORP','GOV','KAP','ROUS','ROW']
M=pd.DataFrame(0.0,index=acc,columns=acc)
g=lambda s: s.groupby(sec).sum().reindex(S57).fillna(0)
Zs=Zloc.groupby(sec).sum().T.groupby(sec).sum().T.reindex(index=S57,columns=S57).fillna(0)
M.loc[S57,S57]=Zs.values
M.loc['ROUS',S57]=g(inflow_int).values; M.loc['ROW',S57]=g(imp_int_col).values
M.loc['LAB',S57]=g(comp).values; M.loc['CAP',S57]=g(surp).values; M.loc['GOV',S57]=g(ntp+ntc).values
M.loc[S57,'HH']=g(loc_fd.pce).values; M.loc[S57,'GOV']=g(loc_fd.government).values; M.loc[S57,'KAP']=g(loc_fd.kap).values
M.loc['ROUS','HH']=inflow_fd.pce.sum(); M.loc['ROUS','GOV']=inflow_fd.government.sum(); M.loc['ROUS','KAP']=inflow_fd.kap.sum()
M.loc['ROW','HH']=imp_fd.pce.sum(); M.loc['ROW','GOV']=imp_fd.government.sum(); M.loc['ROW','KAP']=imp_fd.kap.sum()
M.loc[S57,'ROW']=g(v.exports).values; M.loc[S57,'ROUS']=g(outflow).values
M=M/1e3   # -> $ millions
# --- institutions ($ millions; Tables 2-5, 2024)
ssc_er=27952.277; ssc_ee=32881.536; resid=85380.1
M.loc['GOV','LAB']=ssc_er; M.loc['HH','LAB']=M.loc['LAB',S57].sum()-ssc_er
M.loc['HH','ROUS']=resid
M.loc['CORP','CAP']=M.loc['CAP',S57].sum()
corp_hh=61948.484+151300.671+1789.004+1017.058; corp_gov=28514.672+8210.973+27068.380
M.loc['HH','CORP']=corp_hh; M.loc['GOV','CORP']=corp_gov
M.loc['KAP','CORP']=M.loc['CORP','CAP']-corp_hh-corp_gov
hh_tax=88434.983+18880.484+ssc_ee; gov_tr=119288.772+725.174+394.105
M.loc['GOV','HH']=hh_tax; M.loc['HH','GOV']=gov_tr
M.loc['KAP','HH']=M.loc['HH'].sum()-M[['HH']].sum().iloc[0]
M.loc['KAP','GOV']=M.loc['GOV'].sum()-M[['GOV']].sum().iloc[0]
# external balances through the capital account
rowgap=M.loc['ROW'].sum()-M['ROW'].sum()          # NJ payments to ROW minus receipts
M.loc['KAP','ROW']=max(rowgap,0); M.loc['ROW','KAP']+=max(-rowgap,0)
kapgap=M.loc['KAP'].sum()-M['KAP'].sum()
if kapgap>=0: M.loc['ROUS','KAP']+=kapgap
else: M.loc['KAP','ROUS']+=-kapgap
M.to_csv('samnj24_65_millions.csv')
d=(M.sum(1)-M.sum(0)); print('max imbalance $M',d.abs().max().round(3)); print(d[d.abs()>0.5])
print('neg cells',(M<-0.05).sum().sum()); print(M[M<-50].stack().round(0).head(20))
print('total flows $bn',M.values.sum()/1e3)
diag={'intermediate local':Zs.values.sum()/1e3,'inflows int':inflow_int.sum()/1e6,'imports int':imp_int_col.sum()/1e6,'imports final':imp_fd.values.sum()/1e6,
 'inflows final':inflow_fd.values.sum()/1e6,'outflows':outflow.sum()/1e6,'exports':v.exports.sum()/1e6,'neg outflow cells':int((outflow<0).sum()),'neg outflow $bn':outflow[outflow<0].sum()/1e6,
 'colres max $M':colres.abs().max()/1e3,'imp_fin capped $bn':((v.imports-imp_int_row.reindex(ind).fillna(0)).clip(lower=0)-imp_fin).sum()/1e6}
for k,val in diag.items(): print(f'{k:22s} {val:,.2f}')
for a in ['HH','CORP','GOV','KAP','ROUS','ROW','LAB','CAP']: print(a,'rowsum $bn',round(M.loc[a].sum()/1e3,2))
print(M.loc[['HH','CORP','GOV','KAP','ROUS','ROW','LAB','CAP'],['LAB','CAP','HH','CORP','GOV','KAP','ROUS','ROW']].div(1e3).round(1))
