"""Write the supply / domestic-demand handoff files for RPC estimation.
Usage: python export_supply_demand.py <RPC_explorations root> [year]"""
import sys, numpy as np, pandas as pd
from pathlib import Path
root = Path(sys.argv[1]); year = sys.argv[2] if len(sys.argv) > 2 else '2024'
sut = root / 'RECON2024' / 'process' / 'SUT'
v = pd.read_csv(sut / 'vectors.csv', dtype={'region': str, 'industry': str},
                usecols=['region', 'industry', 'output', 'exports', 'imports', 'pce',
                         'qcew_employment', 'qcew_establishments', 'qcew_wages'])
d = pd.read_csv(sut / f'domestic_demand_{year}.csv', dtype={'region': str, 'industry': str})
c = pd.read_csv(root / 'raw' / 'RPC' / 'industry_sctg_conversion_421.csv', dtype=str)[['industry', 'sctg']]
m = v.merge(d, on=['region', 'industry'], how='left').merge(c, on='industry', how='left')
m['level'] = np.where(m.region == '00000', 0, np.where(m.region.str.endswith('000'), 1, 2))
m['supply'] = m.output - m.exports
m['supplydemand'] = np.where(m.demand_domestic > 0, m.supply / m.demand_domestic, np.nan)
m = m[['region', 'level', 'industry', 'sctg', 'output', 'exports', 'imports', 'pce', 'supply',
       'demand_domestic', 'supplydemand', 'qcew_employment', 'qcew_establishments', 'qcew_wages']]
m.to_csv(sut / f'supply_demand_{year}.csv', index=False, float_format='%.5g')
st = m[m.level == 1].groupby(['region', 'sctg'])[['output', 'exports', 'imports', 'supply', 'demand_domestic']].sum().reset_index()
st.to_csv(sut / f'supply_demand_state_sctg_{year}.csv', index=False)
print(m.shape, st.shape)
