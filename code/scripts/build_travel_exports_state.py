"""RECON2024 item 7: state travel and education exports (first cut, 2026-09-21).

National control: SUT exports of S00900, $223.6bn in 2024. These are BEA's
expenditures in the US by nonresidents. In the SUT they are offset in PCE
(domestic-concept PCE of hotels, food services, etc. includes them), and
S00900 is not an industry, so in the industry-based RECON vectors this
spending is part of state PCE, placed by SAPCE4 line shares.

This script estimates where it is spent and on which commodities:
  education E  = NAFSA 2024-25 by state (tuition share -> 611A00, the rest
                 spent like general travel)
  overseas O   = NTTO 2024 overseas visitor spending by state, less the
                 state's NAFSA amount (NTTO includes education), >= 0
  Canada/Mexico C = S00900 exports - E - O, by state share of traveler
                 accommodations (721000) output
General travel spending is split over commodities with the fixed group
weights below (ASSUMPTIONS, to replace with the BEA TTSA inbound mix);
within a group, by national PCE.
Writes RECON2024/process/SUT/travel_exports_state_commodity_2024.csv ($k,
commodity x state), travel_exports_state_summary_2024.csv, and prints the
change against the current vectors: exports up by T_s(c), PCE down by the
same national amounts, spread by the state's current PCE share.
Nothing in vectors.csv is changed."""
import sys, numpy as np, pandas as pd
from pathlib import Path
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/claude/RPC_explorations')
sut = root / 'RECON2024' / 'process' / 'SUT'
TUITION_SHARE = 0.55
GROUPS = {   # group: (weight, commodities)
    'lodging': (0.28, ['721000']),
    'food_services': (0.20, ['722110', '722211', '722A00']),
    'recreation': (0.12, ['711100', '711200', '711500', '711A00', '712000', '713100', '713200', '713900']),
    'local_transport': (0.10, ['485000', '532100', '48A000']),
    'shopping': (0.22, 'goods'),
    'other': (0.08, ['812100', '812900', '621100', '622000']),
}
u = pd.read_excel(sut / 'sut_det_2024_margins.xlsx', sheet_name='use_fin', index_col=0)
u.index = u.index.astype(str)
X = float(u.loc['S00900', 'exports']) * 1000        # $k
pce = u.pce.clip(lower=0) * 1000
goods = [c for c in u.index if c[0] in '123' and not c.startswith(('22', '23')) and pce[c] > 0]
mix = pd.Series(0.0, index=u.index)
for g, (w, cs) in GROUPS.items():
    cs = goods if cs == 'goods' else cs
    mix[cs] += w * pce[cs] / pce[cs].sum()
mix = mix / mix.sum()

codes = pd.read_excel(root / 'RECON2024' / 'code' / 'src' / 'recon2024' / 'sut' / 'government_config.xlsx',
                      sheet_name='state_codes', dtype=str)
name2fips = {}
for _, r in codes.iterrows():
    for col in r.index:
        if 'name' in col.lower() or 'title' in col.lower() or col.lower() == 'state':
            name2fips[str(r[col]).strip()] = str(r['code']).strip().zfill(5)
ntto = pd.read_csv(root / 'raw' / 'Travel' / 'ntto_overseas_visitors_by_state_2024.csv')
naf = pd.read_csv(root / 'raw' / 'Travel' / 'nafsa_international_students_by_state_2024_25.csv')
ntto['fips'] = ntto.state.map(name2fips); naf['fips'] = naf.state.map(name2fips)
ntto = ntto.dropna(subset=['fips']).set_index('fips'); naf = naf.dropna(subset=['fips']).set_index('fips')
states = sorted(ntto.index)
E_s = naf.dollars.reindex(states).fillna(0) / 1000.0                     # $k
O_s = (ntto.spending_millions.reindex(states) * 1000 - E_s).clip(lower=0)
v = pd.read_csv(sut / 'vectors.csv', dtype={'region': str, 'industry': str},
                usecols=['region', 'industry', 'output', 'pce', 'exports'])
st = v[v.region.isin(states)]
hot = st[st.industry == '721000'].set_index('region').output.reindex(states).fillna(0)
C = X - E_s.sum() - O_s.sum()
C_s = C * hot / hot.sum()
print(f"S00900 exports ${X/1e6:,.1f}bn = education ${E_s.sum()/1e6:,.1f}bn + overseas ${O_s.sum()/1e6:,.1f}bn "
      f"+ Canada/Mexico (residual) ${C/1e6:,.1f}bn")
gen_s = O_s + C_s + (1 - TUITION_SHARE) * E_s
T = pd.DataFrame(np.outer(mix.values, gen_s.values), index=u.index, columns=states)
T.loc['611A00'] += TUITION_SHARE * E_s.values
T = T.loc[T.sum(axis=1) > 0]
T.round(1).to_csv(sut / 'travel_exports_state_commodity_2024.csv')
summ = pd.DataFrame({'education_k': E_s, 'overseas_noned_k': O_s, 'canada_mexico_k': C_s, 'total_k': T.sum()})

# impact vs current vectors: PCE of the same commodities is reduced by the
# national amounts, spread by the state's current PCE share of the commodity
cur_pce = st.pivot(index='industry', columns='region', values='pce').reindex(index=T.index, columns=states).fillna(0)
share = cur_pce.div(cur_pce.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
dP = share.mul(T.sum(axis=1), axis=0)
net = T - dP                                           # change in the state's final demand for its output
summ['pce_reduction_k'] = dP.sum()
summ['net_final_demand_change_k'] = net.sum()
tot_fd = st.groupby('region')[['pce', 'exports']].sum().sum(axis=1).reindex(states)
summ['net_change_pct_of_pce_plus_exports'] = 100 * summ.net_final_demand_change_k / tot_fd
summ.round(1).to_csv(sut / 'travel_exports_state_summary_2024.csv')
print(summ.sort_values('net_change_pct_of_pce_plus_exports')[['total_k', 'net_final_demand_change_k',
      'net_change_pct_of_pce_plus_exports']].round(1).iloc[list(range(5)) + list(range(-6, 0))].to_string())
print('top commodities ($bn):', (T.sum(axis=1) / 1e6).sort_values().tail(8).round(1).to_dict())
