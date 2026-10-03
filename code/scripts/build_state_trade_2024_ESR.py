"""State netted exports and retained imports, 2024, on the BEA commodity frame -- ESR vectors.

Replaces the re-export estimation of build_state_trade_2024.py with the vectors of
Lahr, "What passes through" (ESR package 2026-09-27 1541Z, account_state_hs6_2024.csv.gz):
  netted (domestic) exports  d = x - r      scenario: common-rate (d_cr)
  retained imports           m               benchmark gateway margin mu = .2529 (m_cr)
  gateway service margin     v               benchmark (v_cr): the wholesale part (78.6%) is added to the state's
                                             exports of the re-exported goods (exports_with_gateway_state_bea_2024.csv);
                                             the SUT stage places the transport part (21.4%) on transport services exports.
Everything downstream is unchanged from build_state_trade_2024.py:
  XX (not assigned by Census) spread within commodity to the 50 states + DC by their values
  (codes no state carries spread by state totals); PR and VI dropped;
  HS 27 refined products, coal, LPG reallocated within PADD by SEDS consumption, with the
  tank-farm gateway margin M27/(1+M27), M27 = 0.10246 as before, kept out of the moved goods;
  HS6 -> BEA bridge with the same hold-outs; retained imports scaled to the 2024 SUT by 71-group.
Output: raw/Trade/out_2024_ESR/ (same file names as out_2024/ plus gateway_margin_state_2024.csv).
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, '..'); OUT = os.path.join(T, 'out_2024_ESR')
ROOT = os.path.join(T, '..', '..')
# the run used ESR_package_20260927_1541Z (now in _superseded/); its series file is byte-identical to 1626Z's
SERIES = os.path.join(ROOT, 'Statewise International Trade', 'ESR_package_20260927_1626Z', 'ESR_3_Supplementary', 'series',
                      'account_state_hs6_2024.csv.gz')
sys.path.insert(0, HERE); import bridge
SCEN = 'cr'; MU = 0.2529
M27 = 0.060 + 0.060 * 0.041 + 0.030 + 0.010          # 0.10246, HS 27 tank-farm margin (unchanged)
PADD = {1: 'CT DE DC FL GA ME MD MA NH NJ NY NC PA RI SC VT VA WV', 2: 'IL IN IA KS KY MI MN MO NE ND OH OK SD TN WI',
        3: 'AL AR LA MS NM TX', 4: 'CO ID MT UT WY', 5: 'AK AZ CA HI NV OR WA'}
PADD = {s: p for p, ss in PADD.items() for s in ss.split()}
STATES = sorted(PADD)
os.makedirs(OUT, exist_ok=True)

A = pd.read_csv(SERIES, dtype={'hs6': str, 'state': str})
A = A[A.state.isin(STATES + ['XX'])]
cols = {'x': 'exports_gross_fas', 'g': 'imports_general', 'r': f'r_{SCEN}', 'd': f'd_{SCEN}', 'm': f'm_{SCEN}', 'v': f'v_{SCEN}'}

def spread_unknown(df, col):
    d = df[df.state.isin(STATES)][['state', 'hs6', col]].copy(); xx = df[df.state == 'XX'].groupby('hs6')[col].sum()
    tot = d.groupby('hs6')[col].transform('sum')
    add = d.hs6.map(xx).fillna(0.0) * np.where(tot > 0, d[col] / tot.replace(0, np.nan), 0.0)
    d[col] = d[col] + add.fillna(0.0)
    orph = xx[~xx.index.isin(d.loc[d[col] > 0, 'hs6'])]
    orph = orph[orph != 0]
    if len(orph):
        sh = d.groupby('state')[col].sum(); sh = sh / sh.sum()
        d = pd.concat([d, pd.DataFrame([dict(state=s_, hs6=h, **{col: v * sh[s_]}) for h, v in orph.items() for s_ in sh.index])], ignore_index=True)
    return d.pivot_table(index='hs6', columns='state', values=col, aggfunc='sum').reindex(columns=STATES).fillna(0.0), float(xx.sum()), float(orph.sum())

mats = {}
for k, c in cols.items():
    mats[k], xx, orph = spread_unknown(A, c)
    print(f'{k} ({c}): 51 states {mats[k].values.sum()/1e9:,.1f}bn after spreading XX {xx/1e9:,.2f}bn ({orph/1e9:,.2f}bn orphan codes)')
Tm, Gm, R, NX, ret, V = (mats[k] for k in ('x', 'g', 'r', 'd', 'm', 'v'))

# ---------------- HS 27 reallocation (unchanged rules) ----------------
seds = pd.read_csv(os.path.join(T, 'bea_rec_2022', 'seds_2022_ch27_allocator.csv')).set_index('state').reindex(STATES).fillna(0.0)
def key_for(h):
    if h[:4] in ('2709', '2716'): return None
    if h[:4] in ('2701', '2702', '2703', '2704'): return seds.CLTCP
    if h == '271012': return seds.MGTCP + seds.AVTCP
    if h in ('271019', '271020'): return seds.DFTCP + seds.JFTCP + seds.RFTCP + seds.KSTCP
    if h[:4] == '2711': return seds.HLTCP
    if h[:2] == '27': return seds.PATCP
    return None
margin_kept = pd.DataFrame(0.0, index=[], columns=STATES); moves = []
for c in [h for h in ret.index if h.startswith('27')]:
    k = key_for(c)
    if k is None: continue
    v = ret.loc[c].copy(); marg = v * M27 / (1 + M27); goods = v - marg
    new = pd.Series(0.0, index=STATES)
    for p in sorted(set(PADD.values())):
        ss = [s for s in STATES if PADD[s] == p]; pool = goods[ss].sum(); w = k[ss]
        new[ss] = pool * (w / w.sum() if w.sum() > 0 else goods[ss] / goods[ss].sum() if pool > 0 else 0)
    margin_kept.loc[c] = marg; moves.append(dict(hs6=c, moved=float((new - goods).clip(lower=0).sum()), margin=float(marg.sum())))
    ret.loc[c] = new
mv = pd.DataFrame(moves); print(f'HS27: {len(mv)} codes reallocated within PADD; value moved {mv.moved.sum()/1e9:.1f}bn, tank-farm margin kept {mv.margin.sum()/1e9:.2f}bn')

# ---------------- bridge to BEA ----------------
B = bridge.load()
def to_bea(mat):
    df = mat.stack().rename('v').reset_index(); df.columns = ['hs6', 'state', 'v']
    hold = df.hs6.str[:2].isin(['98', '99']); gold = df.hs6.str[:4].eq('7108')
    m = df[~hold & ~gold].merge(B, on='hs6', how='left'); m['bea'] = m.bea.fillna('UNMATCHED'); m['w'] = m.w.fillna(1.0)
    out = (m.v * m.w).groupby([m.state, m.bea]).sum()
    extra = pd.concat([df[hold].assign(bea='S00300'), df[gold].assign(bea='V00VAL')]).groupby(['state', 'bea']).v.sum()
    return pd.concat([out, extra]).groupby(level=[0, 1]).sum().unstack(0).fillna(0.0).reindex(columns=STATES)
imp_bea = to_bea(ret); exp_bea = to_bea(NX); gross_exp_bea = to_bea(Tm)
# Goods exports are valued at purchasers' prices in the SUT, with wholesale margins carried inside the
# goods rows. The wholesale part (78.6%, 2017 benchmark) of the gateway margin on re-exports therefore
# stays with the re-exported goods in the state they passed through; the transport part (21.4%) is
# placed on transport services exports in the SUT stage (gateway_margin_state_2024.csv).
WS = 0.786
exp_gw_bea = to_bea(NX + WS * V)
print('bridged: retained imports', round(imp_bea.values.sum()/1e9, 1), 'bn; netted exports', round(exp_bea.values.sum()/1e9, 1), 'bn')

# ---------------- scale retained imports to the 2024 SUT (71-group controls) ----------------
n = np.load(os.path.join(ROOT, 'Construction_IO_2024', 'outputs', 'SUT_detail_2024.npz'), allow_pickle=True)
com = list(n['com']); cols_ = list(n['sup_cols']); sut_imp = pd.Series(n['SC'][:, cols_.index('MCIF')] + n['SC'][:, cols_.index('MADJ')], index=com) * 1e6
grp = pd.Series(n['cg'], index=com)
nat = imp_bea.sum(axis=1); cmpb = pd.DataFrame({'census_retained': nat.reindex(com).fillna(0.0), 'sut_import': sut_imp, 'group': grp})
gsum = cmpb.groupby('group')[['census_retained', 'sut_import']].sum(); gsum['factor'] = gsum.sut_import / gsum.census_retained.replace(0, np.nan)
goods_groups = gsum[(gsum.census_retained > 0)].index
fac = grp.map(gsum.factor).reindex(imp_bea.index)
imp_sut = imp_bea.mul(fac.fillna(0.0), axis=0)
print(f'SUT scaling: goods groups {len(goods_groups)}, factor median {gsum.loc[goods_groups].factor.median():.3f}, range {gsum.loc[goods_groups].factor.min():.2f}-{gsum.loc[goods_groups].factor.max():.2f}; scaled total {imp_sut.values.sum()/1e9:.1f}bn')

# ---------------- write ----------------
imp_sut.round(0).to_csv(os.path.join(OUT, 'retained_imports_state_bea_2024_SUTscaled.csv'))
gsum.round(3).to_csv(os.path.join(OUT, 'sut_group_scaling_factors_2024.csv'))
ret.round(0).to_csv(os.path.join(OUT, 'retained_imports_state_hs6_2024.csv.gz'), compression='gzip')
NX.round(0).to_csv(os.path.join(OUT, 'netted_exports_state_hs6_2024.csv.gz'), compression='gzip')
R.round(0).to_csv(os.path.join(OUT, 'reexports_state_hs6_2024.csv.gz'), compression='gzip')
imp_bea.round(0).to_csv(os.path.join(OUT, 'retained_imports_state_bea_2024.csv'))
exp_bea.round(0).to_csv(os.path.join(OUT, 'netted_exports_state_bea_2024.csv'))
gross_exp_bea.round(0).to_csv(os.path.join(OUT, 'gross_exports_state_bea_2024.csv'))
exp_gw_bea.round(0).to_csv(os.path.join(OUT, 'exports_with_gateway_state_bea_2024.csv'))
gm = V.sum().rename('gateway_margin'); gm.index.name = 'state'
gm.round(0).to_csv(os.path.join(OUT, 'gateway_margin_state_2024.csv'))
tot = pd.DataFrame({'gross_exports': Tm.sum(), 'reexports': R.sum(), 'netted_exports': NX.sum(), 'general_imports': Gm.sum(),
                    'retained_imports': ret.sum(), 'gateway_margin': V.sum(),
                    'hs27_tankfarm_margin': margin_kept.sum() if len(margin_kept) else 0.0})
tot['export_cut_pct'] = tot.reexports / tot.gross_exports * 100; tot['import_cut_pct'] = (tot.general_imports - tot.retained_imports) / tot.general_imports * 100
tot.round(0).to_csv(os.path.join(OUT, 'state_totals_2024.csv'))
open(os.path.join(OUT, 'RUN_NOTES.txt'), 'w').write(f'source: {os.path.basename(SERIES)} (ESR package 20260927_1541Z)\nscenario={SCEN}\nmu={MU}\nM27={M27:.5f}\n')
print(tot.sort_values('general_imports', ascending=False).head(8).div(1e9).round(2).to_string())
