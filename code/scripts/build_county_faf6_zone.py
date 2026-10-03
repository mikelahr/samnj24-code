"""County (BEA region code) -> FAF6 zone crosswalk, from the 2022 CFS area polygons
(Paper1 data/2022_CFS_Areas.zip) by county internal point (TIGER tl_us_county), then
cfs_area_to_faf_zone.csv (faf6_zone). BEA Virginia combination areas take the zone of their
county component; 15901 (Maui+Kalawao) -> 15005's zone. CT planning regions and AK not mapped (bypassed)."""
import pandas as pd
c=pd.read_csv('county_cfs2022_area.csv',dtype={'geoid':str})
def code(g):
    # E330000US + state(2) + CSA(3) + MSA(5); MSA part non-zero -> MSA-defined area (99999 = remainder)
    g=g[g.find('US')+2:]; s,csa,msa=g[:2],g[2:5],g[5:10]
    return s+'-'+(msa if msa!='00000' else csa)
c['cfs_area']=c.cfs22_geoid.map(code)
a=pd.read_csv('../../../../Paper1_RPC_SpaceTime/rebuild_2022_20260920/cfs_area_to_faf_zone.csv',dtype={'cfs_area':str})
c=c.drop(columns=[x for x in ['faf6_zone','faf5_zone'] if x in c]).merge(a[['cfs_area','faf6_zone']],on='cfs_area',how='left')
print('counties without FAF6 zone:',c.faf6_zone.isna().sum(),c[c.faf6_zone.isna()].cfs_area.unique())
x=c[['geoid','cname','cfs_area','cfs22_name','faf6_zone']].rename(columns={'geoid':'region'})
VA={'51901':'51003','51903':'51005','51907':'51015','51911':'51031','51913':'51035','51918':'51053','51919':'51059','51921':'51069',
    '51923':'51081','51929':'51089','51931':'51095','51933':'51121','51939':'51143','51941':'51149','51942':'51153','51944':'51161',
    '51945':'51163','51947':'51165','51949':'51175','51951':'51177','51953':'51191','51955':'51195','51958':'51199','15901':'15005'}
z=x.set_index('region').faf6_zone
add=pd.DataFrame({'region':list(VA),'cname':'BEA combination','cfs_area':'','cfs22_name':'','faf6_zone':[z[v] for v in VA.values()]})
x=pd.concat([x,add]); x['faf6_zone']=x.faf6_zone.astype('Int64')
x.to_csv('county_to_faf6_zone.csv',index=False)
print(len(x), x.faf6_zone.nunique(), x[x.region.str[:2]=='19'].faf6_zone.value_counts().to_dict(), x[x.region.str[:2]=='39'].faf6_zone.value_counts().to_dict())
