"""Table 6 (19-account macro SAM, FED/STA/LOC) and Table 7 control totals for SAMNJ24.
Runs build_samnj24.py for the 65-account core, then splits GOV by level with the Census 2024
State & Local Government Finances (NJ) and BEA SAINC35, and aggregates. $ millions."""
import runpy, json, numpy as np, pandas as pd
g=runpy.run_path('build_samnj24.py')
M,S57,v,loc_fd,inflow_fd,imp_fd,sec=[g[k] for k in ['M','S57','v','loc_fd','inflow_fd','imp_fd','sec']]
C={k.split('|')[0]:{kk:float(vv)/1e3 for kk,vv in d.items()} for k,d in json.load(open('census24_nj.json')).items()}  # $M
cs=lambda code,lev: C[code].get(lev,0.0)
A=['HH','COR','FED','STA','LOC','TRs','TRd','PIT','CIT','SSC','PRD','COM','GCF','SAV','LAB','CAP','IND','RUS','ROW']
T=pd.DataFrame(0.0,index=A,columns=A)
IND=S57
# ---- industries, factors, external: straight from the 65-account core
T.loc['IND','IND']=M.loc[IND,IND].values.sum()
T.loc['RUS','IND']=M.loc['ROUS',IND].sum(); T.loc['ROW','IND']=M.loc['ROW',IND].sum()
T.loc['LAB','IND']=M.loc['LAB',IND].sum(); T.loc['CAP','IND']=M.loc['CAP',IND].sum()
ntp=g['ntp'].sum()/1e3; ntc=g['ntc'].sum()/1e3   # 2026-09-28: BEA-controlled (build_samnj24.py)
T.loc['PRD','IND']=ntp; T.loc['COM','IND']=ntc
T.loc['IND','RUS']=M.loc[IND,'ROUS'].sum(); T.loc['IND','ROW']=M.loc[IND,'ROW'].sum()
T.loc['LAB','RUS']=M.loc['HH','ROUS']                       # residence adjustment (commuting)
ssc_er=M.loc['GOV','LAB']; T.loc['SSC','LAB']=ssc_er; T.loc['HH','LAB']=T.loc['LAB'].sum()-ssc_er
# ---- capital income: proprietors to HH, rest to corporations
prop=61948.484; T.loc['HH','CAP']=prop; T.loc['COR','CAP']=T.loc['CAP','IND']-prop
div=151300.671; T.loc['TRd','COR']=div; T.loc['HH','TRd']=div
cit_f,cit_s=28514.672,cs('LF0024','State'); T.loc['CIT','COR']=cit_f+cit_s; T.loc['FED','CIT']=cit_f; T.loc['STA','CIT']=cit_s
T.loc['HH','COR']=1789.004+1017.058                           # SAINC35 lines 4000 and 3300
T.loc['STA','COR']=cs('LF0040','State'); T.loc['LOC','COR']=cs('LF0040','Local')   # current charges + misc. general revenue
T.loc['SAV','COR']=T['COR'].sum()*0+T.loc['COR'].sum()-T[['COR']].sum().iloc[0]
# ---- household taxes
pit_f,pit_s=88434.983,cs('LF0023','State'); ssc_ee=32881.536
T.loc['PIT','HH']=pit_f+pit_s; T.loc['FED','PIT']=pit_f; T.loc['STA','PIT']=pit_s
T.loc['SSC','HH']=ssc_ee; T.loc['FED','SSC']=ssc_er+ssc_ee
# ---- taxes on production (Census lines 9, 20, 21; no federal share) and on products (federal = total less S&L sales taxes)
ps=cs('LF0009','State')+cs('LF0029','State')+cs('LF0034','State')+ (cs('LF0025','State')+cs('LF0026','State')+cs('LF0027','State')+cs('LF0028','State')+cs('LF0030','State')+cs('LF0031','State')+cs('LF0032','State')+cs('LF0033','State'))*0
# Census "Other taxes" (line 21) = total taxes less property, sales, income and motor vehicle license
oth=lambda lev: cs('LF0008',lev)-cs('LF0009',lev)-cs('LF0010',lev)-cs('LF0022',lev)-cs('LF0029',lev)
prd_s=cs('LF0009','State')+cs('LF0029','State')+oth('State'); prd_l=cs('LF0009','Local')+cs('LF0029','Local')+oth('Local')
T.loc['STA','PRD']=prd_s; T.loc['LOC','PRD']=prd_l; T.loc['FED','PRD']=ntp-prd_s-prd_l   # 2026-09-28: S&L at Census; federal = BEA total less S&L
sal_s=cs('LF0010','State'); sal_l=cs('LF0010','Local')
T.loc['STA','COM']=sal_s; T.loc['LOC','COM']=sal_l; T.loc['FED','COM']=ntc-sal_s-sal_l
# ---- intergovernmental (Census, gross)
T.loc['STA','FED']=cs('LF0004','State'); T.loc['LOC','FED']=cs('LF0004','Local')
T.loc['LOC','STA']=cs('LF0005','Local'); T.loc['STA','LOC']=cs('LF0006','State')
# ---- social transfers (SAINC35): state pays UI (2410), public assistance medical care (2220) and its NPISH transfers (3200)
sainc={'2000':119288.772,'2410':2836.850,'2220':24344.334,'3100':725.174,'3200':394.105}
T.loc['TRs','STA']=sainc['2410']+sainc['2220']+sainc['3200']
T.loc['TRs','FED']=sainc['2000']-sainc['2410']-sainc['2220']+sainc['3100']
T.loc['HH','TRs']=T.loc['TRs'].sum()
# ---- government consumption by level: federal rows vs state-and-local rows, S&L split by Census current operations
govc=pd.DataFrame({'loc':loc_fd.government,'inf':inflow_fd.government,'imp':imp_fd.government})/1e3
fedrows=[i for i in govc.index if i in ('S00500','S00600')]
slrows=[i for i in govc.index if i.startswith('GSLG')]
assert abs(govc.drop(index=fedrows+slrows).values.sum())<1, govc.drop(index=fedrows+slrows).sum()
co=lambda code,lev: cs(code,lev)
edu=(co('LF0108','State'),co('LF0108','Local'))
hlt=(co('LF0130','State')+co('LF0133','State'),co('LF0130','Local')+co('LF0133','Local'))
tot=(co('LF0104','State'),co('LF0104','Local')); wel=(co('LF0124','State'),co('LF0124','Local'))
oth2=tuple(tot[k]-wel[k]-edu[k]-hlt[k] for k in (0,1))
shs={'GSLGE':edu[0]/sum(edu),'GSLGH':hlt[0]/sum(hlt),'GSLGO':oth2[0]/sum(oth2)}
for col,dest in [('loc','IND'),('inf','RUS'),('imp','ROW')]:
    T.loc[dest,'FED']=govc.loc[fedrows,col].sum()
    T.loc[dest,'STA']=sum(govc.loc[r,col]*shs[r] for r in slrows)
    T.loc[dest,'LOC']=sum(govc.loc[r,col]*(1-shs[r]) for r in slrows)
# ---- household consumption
T.loc['IND','HH']=M.loc[IND,'HH'].sum(); T.loc['RUS','HH']=M.loc['ROUS','HH']; T.loc['ROW','HH']=M.loc['ROW','HH']
# ---- gross capital formation; government investment: S&L capital outlay (Census), federal at the S&L capital-outlay ratio
T.loc['IND','GCF']=M.loc[IND,'KAP'].sum(); T.loc['RUS','GCF']=inflow_fd.kap.sum()/1e3; T.loc['ROW','GCF']=M.loc['ROW','KAP']
co_ratio=cs('LF0095','State and Local')/(cs('LF0095','State and Local')+cs('LF0094','State and Local'))
fed_direct=29600.0   # USAspending FY2024 direct spending net of transfers (text: $3.5bn defense + $26.1bn nondefense)
T.loc['GCF','FED']=co_ratio*fed_direct; T.loc['GCF','STA']=cs('LF0095','State'); T.loc['GCF','LOC']=cs('LF0095','Local')
# ---- savings residuals
for a in ['HH','FED','STA','LOC']: T.loc['SAV',a]=T.loc[a].sum()-T[a].sum()
T.loc['SAV','COR']=T.loc['COR'].sum()-(T['COR'].sum()-T.loc['SAV','COR'])
T.loc['SAV','ROW']=M.loc['KAP','ROW']
T.loc['GCF','SAV']=T['GCF'].sum()-T.loc['GCF'].sum()
T.loc['RUS','SAV']=T.loc['SAV'].sum()-T['SAV'].sum()
d=T.sum(1)-T.sum(0); print('max imbalance $M',d.abs().max().round(3)); print(d[d.abs()>0.01].round(2))
print('neg',T[T<-0.01].stack())
T.to_csv('samnj24_table6_millions.csv')
print((T/1e3).round(1).replace(0.0,'').to_string())
# ---- consistency with the 65-account core
chk={'GOV total receipts vs core':(T.loc[['FED','STA','LOC']].sum().sum()-T.loc[['FED','STA','LOC'],['FED','STA','LOC']].values.sum()+T.loc[['FED','STA','LOC'],['PIT','CIT','SSC','PRD','COM']].values.sum()*0, M.loc['GOV'].sum())}
imp_total=M.loc['ROW'].sum()-M.loc['ROW','KAP']*0
print('core total flows',M.values.sum(),'T6 total flows',T.values.sum())
json.dump({'ntp':ntp,'ntc':ntc,'prd_s':prd_s,'prd_l':prd_l,'shares':shs,'co_ratio':co_ratio,
 'imports_int_final':float(M.loc['ROW',IND].sum()+M.loc['ROW',['HH','GOV','KAP']].sum()),
 'output':float(v.output.sum()/1e3),'comp':float(v.compensation.sum()/1e3),'gos':float(M.loc['CAP',IND].sum()),
 'gdp':float(M.loc[['LAB','CAP','GOV'],IND].values.sum()),'pce':float(v.pce.sum()/1e3),'exports':float(v.exports.sum()/1e3),
 'imports_recon':float(v.imports.sum()/1e3),'hh_income_less_sscee':float(T.loc['HH'].sum()-ssc_ee),'total_flows':float(M.values.sum()),
 'max_imb_core':float((M.sum(1)-M.sum(0)).abs().max()),'max_imb_t6':float(d.abs().max())},open('table7_values.json','w'),indent=1)
