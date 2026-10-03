"""extract_nj_inputs.py -- build the data/ folder of the SAMNJ24 replication package.

Run ONCE, on the full RPC_explorations tree (the machine where RECON2024 was run),
from the package root:

    python code/extract_nj_inputs.py  <path to RPC_explorations>

What it does
------------
It copies or extracts, and never modifies, every input the SAMNJ24 accounts use,
restricted to New Jersey wherever the file is organised by region.  Three kinds of
files result (all under data/):

  data/national_sut_421/   the 421-sector national 2024 supply-use tables (SUTs):
                           the construction-revised package as received, the balanced
                           and margin-adjusted version RECON2024 actually used, and
                           plain-CSV dumps of its blocks (use, supply, final use,
                           primary inputs, market-share matrix D, margin totals).
                           National by nature: every state's accounts start here.
  data/bridges/            every concordance ("bridge") used between classifications:
                           QCEW NAICS -> SUT industries, BEA SAEMP/SAGDP lines -> SUT,
                           HS6 -> NAICS -> BEA commodity (trade), 421 SUT -> 71 summary,
                           421 -> SAMNJ24's 57 sectors, 12 BEA construction structure
                           types -> 31 NAICS construction industries, PCE (NIPA) lines
                           -> SUT commodities, SUT industries -> SCTG (freight), counties
                           -> FAF6 zones.
  data/nj/                 New Jersey's rows of every region-organised input and
                           intermediate product, grouped by source (see
                           data/DATA_DICTIONARY.md, written by this script).

Anything national that NJ's numbers depend on only through a national control or
share (US totals, the sum over states used in a share) is written as a one-line or
one-column extract next to the NJ rows, never as the full 50-state file.

Units: files keep their source units.  RECON2024 vectors and state matrices are in
$ thousands; SUT blocks in $ millions; BEA SAINC in $ thousands (SAINC5N, SAINC35) or
$ millions (SAINC4, SAGDP1); Census finance in $ thousands.
"""
import sys, os, shutil, json, hashlib, zipfile, io, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path('~/mnt/RPC_explorations').expanduser()
PKG = Path(__file__).resolve().parents[1]
R24 = ROOT / 'RECON2024'
NJ, US = '34000', '00000'
LOG = []


def out(rel):
    """Destination path under data/, creating folders."""
    p = PKG / 'data' / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def copy(src, rel, note):
    """Copy a file unchanged and record it in the data dictionary."""
    src = Path(src)
    dst = out(rel)
    shutil.copy2(src, dst)
    LOG.append((rel, str(src.relative_to(ROOT)), 'copied unchanged', note))


def rows(src, rel, note, key, keep, **kw):
    """Write the rows of a CSV whose column `key` satisfies keep(value)."""
    src = Path(src)
    d = pd.read_csv(src, dtype=str, **kw)
    d = d[d[key].astype(str).map(keep)]
    comp = 'gzip' if rel.endswith('.gz') else None
    d.to_csv(out(rel), index=False, compression=comp)
    LOG.append((rel, str(src.relative_to(ROOT)), f'rows where {key} is New Jersey ({len(d)} rows)', note))


def frame(df, rel, src, how, note, index=True):
    comp = 'gzip' if rel.endswith('.gz') else None
    df.to_csv(out(rel), index=index, compression=comp)
    LOG.append((rel, src, how, note))


# ---------------------------------------------------------------------------
# 1. National 421-sector SUTs
# ---------------------------------------------------------------------------
CIO = ROOT / 'Construction_IO_2024' / 'outputs'
copy(CIO / 'Use_revised_NAICS421_2024.csv', 'national_sut_421/Use_revised_NAICS421_2024.csv',
     'Construction-revised 2024 Use table, 421 commodities x 421 industries + final demand ($M); '
     'BEA 2017 benchmark detail projected to the 2024 summary SUT, 12 BEA construction structure '
     'types replaced by 31 NAICS construction industries.')
copy(CIO / 'Supply_revised_NAICS421_2024.csv', 'national_sut_421/Supply_revised_NAICS421_2024.csv',
     'Matching 2024 Supply table ($M).')
copy(CIO / 'A_total_NAICS421_2024.csv', 'national_sut_421/A_total_NAICS421_2024.csv',
     'National total-requirements A (industry x industry) from the package, for reference.')
copy(CIO / 'A_domestic_NAICS421_2024.csv', 'national_sut_421/A_domestic_NAICS421_2024.csv',
     'National domestic A from the package, for reference.')
S = R24 / 'process' / 'SUT'
copy(S / 'sut_det_2024_balanced.xlsx', 'national_sut_421/sut_det_2024_balanced.xlsx',
     'RECON2024 process_suts/create_target: the package balanced on industries as well as commodities '
     '(max residual $8M).')
copy(S / 'sut_det_2024_margins.xlsx', 'national_sut_421/sut_det_2024_margins.xlsx',
     'RECON2024 apply_margins: balanced SUT with trade and transport margins endogenised (purchasers\' '
     'prices); sheets margin_trade / margin_transport hold the margin matrices. THIS is the SUT the state '
     'accounts use.')
copy(S / 'iot.xlsx', 'national_sut_421/iot.xlsx',
     'RECON2024 create_symmetric: market-share matrix D, national A, and use blocks on the 421 axis.')
copy(S / 'Use_2024.xlsx', 'national_sut_421/Use_2024_summary71.xlsx', 'BEA 2024 summary Use table (71 industries), control totals.')
copy(S / 'Supply_2024.xlsx', 'national_sut_421/Supply_2024_summary71.xlsx', 'BEA 2024 summary Supply table (71 industries).')

# plain-CSV dumps of the blocks actually read by the NJ Stage A in run_samnj24.py
sys.path.insert(0, str(R24 / 'code' / 'src'))
os.chdir(R24 / 'code')
from recon2024 import utilities                      # noqa: E402
from recon2024.sut import operations as sop          # noqa: E402
from recon2024.bea import operations as bop          # noqa: E402
cfg = utilities.get_config('sut'); T = cfg.dirs.target
sut = sop.Sut.read_excel(T / cfg.apply_margins.target.file, 'sut')
iot = sop.Sut.read_excel(T / cfg.create_symmetric.target.files.iot, 'iot', is_iot=True)
src_m = 'RECON2024/process/SUT/sut_det_2024_margins.xlsx'
for blk in ['use_int', 'use_fin', 'use_pri', 'sup_int']:
    frame(sut.data.__dict__[blk], f'national_sut_421/csv/{blk}.csv', src_m, f'sheet block {blk}',
          {'use_int': 'intermediate use, commodity x industry ($M, purchasers\' prices)',
           'use_fin': 'final use, commodity x final-demand category ($M)',
           'use_pri': 'primary inputs (value added, noncomparable imports), row x industry ($M)',
           'sup_int': 'supply (make), commodity x industry ($M)'}[blk])
frame(sut.codes.commodities, 'national_sut_421/csv/commodities.csv', src_m, 'codes sheet',
      'commodity codes, titles, total use/supply ($M)')
frame(sut.codes.industries, 'national_sut_421/csv/industries.csv', src_m, 'codes sheet',
      'industry codes, titles, total output ($M)')
frame(iot.data.D, 'national_sut_421/csv/D_market_shares.csv', 'RECON2024/process/SUT/iot.xlsx', 'block D',
      'market-share matrix D (commodity x industry): share of commodity c supplied by industry j')
mt = pd.read_excel(T / cfg.apply_margins.target.file, sheet_name='margin_trade', index_col=0).sum(axis=1)
mx = pd.read_excel(T / cfg.apply_margins.target.file, sheet_name='margin_transport', index_col=0).sum(axis=1)
frame(pd.DataFrame({'margin_trade': mt, 'margin_transport': mx}), 'national_sut_421/csv/margin_totals_by_commodity.csv',
      src_m, 'row sums of sheets margin_trade and margin_transport', 'trade and transport margins on each commodity ($M)')

# ---------------------------------------------------------------------------
# 2. Bridges
# ---------------------------------------------------------------------------
copy(S / 'aggregation_bridge.xlsx', 'bridges/aggregation_bridge_421_to_71.xlsx',
     '421 SUT industries/commodities -> 71 BEA summary codes (sheet industries: offspring_code -> parent_code).')
Q = R24 / 'process' / 'QCEW'
copy(Q / 'qcew_bridge.xlsx', 'bridges/qcew_bridge_NAICS_to_SUT421.xlsx',
     'QCEW six-digit NAICS -> 421 SUT industries (process_qcew_bridge + process_naics_bridge + retarget_sut_axis).')
copy(Q / 'codes.xlsx', 'bridges/qcew_codes.xlsx', 'QCEW area, ownership and industry code lists.')
B = R24 / 'process' / 'BEA'
copy(B / 'bea_bridge.xlsx', 'bridges/bea_bridge_lines_to_SUT421.xlsx',
     'BEA regional line codes (SAGDP, SAINC/CAINC earnings, SAEMP employment) -> 421 SUT industries.')
copy(B / 'bea_codes.xlsx', 'bridges/bea_codes.xlsx', 'BEA regional line and area code lists.')
copy(B / 'bea_structure.xlsx', 'bridges/bea_structure.xlsx', 'BEA regional line hierarchy (parent/child).')
copy(ROOT / 'raw' / 'Trade' / 'hs6_naics_imports_2024.csv', 'bridges/hs6_to_naics_census_2024.csv',
     'Census HS6 -> NAICS concordance (imports), 2024.')
copy(ROOT / 'NAICS_BEA_conversion_17.csv', 'bridges/naics_to_bea_commodity_2017.csv',
     'BEA NAICS -> IO commodity correspondence (2017 benchmark).')
copy(ROOT / 'raw' / 'Trade' / 'code' / 'bridge.py', 'bridges/hs6_to_bea_bridge_builder.py',
     'Code that combines the two trade concordances into HS6 -> BEA commodity (with hold-outs).')
copy(CIO / 'Translator_BEA12_to_NAICS31_2024.csv', 'bridges/construction_BEA12_to_NAICS31_2024.csv',
     'BEA 12 construction structure types -> 31 NAICS construction industries.')
copy(ROOT / 'raw' / 'PCE' / 'bridge_PCE_USE_gauss.txt', 'bridges/pce_nipa_to_sut_bridge.txt',
     'BEA PCE bridge: NIPA PCE categories -> IO commodities (as used for the 413-commodity state PCE).')
copy(ROOT / 'raw' / 'PCE' / 'select_bottom_NIPA113_gauss.txt', 'bridges/pce_nipa_bottom_lines.txt',
     'Bottom-level NIPA table 2.4.5 lines used by the PCE bridge.')
copy(ROOT / 'raw' / 'PCE' / 'PCE_state_gauss_div.txt', 'bridges/pce_sapce_line_map.txt',
     'SAPCE state lines -> NIPA lines.')
copy(ROOT / 'raw' / 'RPC' / 'industry_sctg_conversion_421.csv', 'bridges/sut421_to_sctg.csv',
     '421 SUT industries -> SCTG freight commodity groups (RPC regression and FAF raking).')
rows(R24 / 'process' / 'RPC' / 'predict_2024' / 'county_to_faf6_zone.csv', 'bridges/nj_county_to_faf6_zone.csv',
     'New Jersey counties -> FAF6 zones (county raking).', key=list(pd.read_csv(
         R24 / 'process' / 'RPC' / 'predict_2024' / 'county_to_faf6_zone.csv', nrows=1).columns)[0],
     keep=lambda v: str(v).zfill(5).startswith('34'))
# 421 -> 57 SAMNJ24 sectors: generated from the aggregation bridge and the 57-sector
# rules below (same rules as run_samnj24.py, written out so the map is inspectable)
b = pd.read_excel(S / 'aggregation_bridge.xlsx', sheet_name='industries').set_index('offspring_code').parent_code.astype(str)
M57 = {'111CA': 'AG11', '113FF': 'AG11', '211': 'MIN21', '212': 'MIN21', '213': 'MIN21', '313TT': 'TEX', '315AL': 'TEX',
       '324': 'PPN', '326': 'PPN', '327': 'PPN', '3361MV': 'TEQ336', '3364OT': 'TEQ336', '481': 'TRANS_x484',
       '482': 'TRANS_x484', '483': 'TRANS_x484', '485': 'TRANS_x484', '486': 'TRANS_x484', '487OS': 'TRANS_x484',
       'HS': 'RE', 'ORE': 'RE', '561': 'ADMIN56', '562': 'ADMIN56', '623': 'HSA', '624': 'HSA', '711AS': 'ARTS71',
       '713': 'ARTS71'}
m = pd.DataFrame({'summary71': b})
m['sam57'] = m.summary71.map(lambda c: M57.get(c, c))
m.loc['424400', 'sam57'] = '424GR'; m.loc['624400', 'sam57'] = '624CDC'
m.index.name = 'sut421'
frame(m, 'bridges/sut421_to_samnj24_57.csv', 'generated', 'aggregation_bridge + 57-sector rules',
      '421 SUT industries -> 71 summary -> SAMNJ24 57 sectors (Table B1).')

# ---------------------------------------------------------------------------
# 3. New Jersey: RECON2024 state IO accounts (products of the upstream run)
# ---------------------------------------------------------------------------
RS = R24 / 'results'
copy(RS / 'states' / 'recon2024_34000.csv', 'nj/state_io/recon2024_34000.csv',
     'NJ industry vector, 421 industries: output, value added by component, final demand, exports, '
     'imports, RPC, etc. ($ thousands). Upstream product; input to the SAM.')
copy(RS / 'Amatrices_use' / 'recon2024_Amatrix_34000.csv', 'nj/state_io/recon2024_Amatrix_34000.csv',
     'NJ direct-requirements matrix (use-side domestication), 421 + households. Reproduced by Stage A.')
copy(RS / 'Usematrices' / 'recon2024_Use_total_34000.csv.gz', 'nj/state_io/recon2024_Use_total_34000.csv.gz',
     'NJ intermediate Use in levels, total ($k). Reproduced by Stage A.')
copy(RS / 'Usematrices' / 'recon2024_Use_domestic_34000.csv.gz', 'nj/state_io/recon2024_Use_domestic_34000.csv.gz',
     'NJ intermediate Use net of international imports ($k). Reproduced by Stage A.')
copy(RS / 'Bmatrices' / 'recon2024_Pimport_34000.csv', 'nj/state_io/recon2024_Pimport_34000.csv',
     'NJ international import shares of use by commodity. Reproduced by Stage A.')

vec = pd.read_csv(S / 'vectors.csv', dtype={'region': str})
frame(vec[vec.region == NJ], 'nj/state_io/vectors_34000.csv', 'RECON2024/process/SUT/vectors.csv',
      'rows region == 34000', 'NJ vectors from create_vectors ($k): the inputs of Stage A.', index=False)
nat = vec[vec.region == US].copy()
st = vec[(vec.region.str.len() == 5) & vec.region.str.endswith('000') & (vec.region != US)]
nat['output_sum_of_states'] = nat.industry.map(st.groupby('industry').output.sum())
frame(nat, 'national_sut_421/vectors_00000.csv', 'RECON2024/process/SUT/vectors.csv',
      'rows region == 00000, plus the sum of the 51 state outputs',
      'US vectors ($k); Stage A needs US intermediate-use fractions and NJ\'s share of the state output sum.', index=False)
for f, note in [('imports_state_commodity_2024.csv', 'international imports by commodity ($M)'),
                ('exports_state_commodity_2024.csv', 'international exports by commodity ($M)'),
                ('pce_state_commodity_2024.csv', 'household consumption by commodity ($k)'),
                ('services_imports_state_commodity_2024.csv', 'services imports by commodity ($M)'),
                ('travel_exports_state_commodity_2024.csv', 'travel/education exports by commodity ($M)')]:
    d = pd.read_csv(S / f, index_col=0)
    frame(d[[NJ]] if NJ in d.columns else d[d.index.astype(str) == NJ], f'nj/state_io/{f.replace("state", "nj")}',
          f'RECON2024/process/SUT/{f}', 'New Jersey column', 'NJ ' + note)
copy(S / 'services_imports_method_2024.csv', 'nj/state_io/services_imports_method_2024.csv',
     'allocator used for each services import commodity (applies to every state)')
reg = bop.load_bridged().regions
frame(reg.loc[[US, NJ]], 'nj/state_io/regions_income_2024.csv', 'RECON2024/process/BEA (load_bridged)',
      'US and NJ rows', 'personal income ($k) and population used to split government final demand.')
d = pd.read_csv(S / 'supply_demand_2024.csv', dtype={'region': str})
frame(d[d.region == NJ], 'nj/rpc/supply_demand_34000.csv', 'RECON2024/process/SUT/supply_demand_2024.csv',
      'rows region == 34000', 'NJ supply, domestic demand and regressors for the RPC prediction.', index=False)
d = pd.read_csv(R24 / 'process' / 'RPC' / 'predict_2024' / 'rpc_predictions_2024_raked.csv', dtype={'region': str})
frame(d[d.region == NJ], 'nj/rpc/rpc_predictions_2024_raked_34000.csv',
      'RECON2024/process/RPC/predict_2024/rpc_predictions_2024_raked.csv', 'rows region == 34000',
      'NJ predicted and FAF6-raked RPCs.', index=False)
copy(R24 / 'process' / 'RPC' / 'rpc_model.pickle', 'nj/rpc/rpc_model.pickle',
     'Fitted fractional-probit RPC model (estimated on the all-state FAF6 2022 panel; national by construction).')
copy(R24 / 'process' / 'RPC' / 'rpc_summary.txt', 'nj/rpc/rpc_summary.txt', 'Estimation summary of the RPC model.')
a = pd.read_csv(ROOT / 'anchors_2024_FAF6' / 'rpc_anchor_state_sctg_2024_FAF6.csv', dtype=str)
kc = 'state_fips'
frame(a[a[kc].astype(str).str.zfill(2) == '34'], 'nj/rpc/rpc_anchor_state_sctg_2024_FAF6_NJ.csv',
      'anchors_2024_FAF6/rpc_anchor_state_sctg_2024_FAF6.csv', f'rows where {kc} is NJ', 'FAF6 2024 anchors for NJ.', index=False)

# ---------------------------------------------------------------------------
# 4. New Jersey: public source data (rows of the files as downloaded)
# ---------------------------------------------------------------------------
RAW = ROOT / 'raw'
isnj = lambda v: '34000' in str(v)
for f in ['SAINC4__ALL_AREAS_1929_2025.csv', 'SAINC5N__ALL_AREAS_1998_2025.csv', 'SAINC35__ALL_AREAS_1929_2024.csv']:
    rows(RAW / 'SAINC' / f, f'nj/bea/{f.replace("ALL_AREAS", "NJ")}', 'BEA state personal income tables, NJ rows.',
         key='GeoFIPS', keep=isnj, encoding='latin1')
rows(RAW / 'SAGDP' / 'SAGDP1__ALL_AREAS_1997_2025.csv', 'nj/bea/SAGDP1__NJ_1997_2025.csv',
     'BEA state GDP summary (TOPI, subsidies, GOS), NJ rows.', key='GeoFIPS', keep=isnj, encoding='latin1')
for f in sorted(glob.glob(str(RAW / 'SAGDP' / 'SAGDP*_NJ_*.csv'))):
    copy(f, f'nj/bea/{Path(f).name}', 'BEA state GDP by industry table, NJ file.')
d = pd.read_csv(RAW / 'SAGDP' / 'filled' / 'sagdp_comp_gos_filled_2022_2024.csv', dtype=str)
kc = [c for c in d.columns if 'fips' in c.lower() or c.lower() in ('region', 'geo', 'state')][0]
frame(d[d[kc].astype(str).str.startswith('34')], 'nj/bea/sagdp_comp_gos_filled_2022_2024_NJ.csv',
      'raw/SAGDP/filled/sagdp_comp_gos_filled_2022_2024.csv', f'rows where {kc} is NJ',
      'SAGDP compensation and GOS by line with suppressed cells filled (MAP fill), NJ.', index=False)
rows(RAW / 'PCE' / 'SAPCE4__ALL_AREAS_1997_2024.csv', 'nj/bea/SAPCE4__NJ_1997_2024.csv',
     'BEA state PCE by function, NJ rows.', key='GeoFIPS', keep=isnj, encoding='latin1')
G = RAW / 'Government'
with zipfile.ZipFile(G / 'GOVSLOCALFINTIMESERIES.GS00LOCALFIN_2026-09-21T112416.zip') as z:
    name = [n for n in z.namelist() if n.endswith('-Data.csv')][0]
    g = pd.read_csv(io.BytesIO(z.read(name)), skiprows=[1], dtype=str, encoding='utf-8-sig')
frame(g[g.GEO_ID.str.endswith('US34')], 'nj/government/census_GS00LOCALFIN_2017_2024_NJ.csv',
      'raw/Government/GOVSLOCALFINTIMESERIES.GS00LOCALFIN_2026-09-21T112416.zip', 'rows for New Jersey',
      'Census Annual Survey of State & Local Government Finances, NJ (state, local, state and local).', index=False)
copy(R24 / 'SAMNJ24' / 'census24_nj.json', 'nj/government/census24_nj.json',
     'The 2024 NJ Census finance items read by the SAM (LF item -> level -> $k); extracted from the file above.')
copy(G / 'unprocessed' / '24dbnewjersey.xlsx', 'nj/government/irs_databook_FY2024_newjersey.xlsx',
     'IRS Data Book FY2024, New Jersey collections (federal PIT and CIT).')
rows(G / 'irs_state_collections_FY2024.csv', 'nj/government/irs_state_collections_FY2024_NJ.csv',
     'IRS Data Book FY2024 collections, NJ row.', key='area', keep=lambda v: v == 'New Jersey')
rows(G / 'usaspending_contracts_pop_state_FY2024.csv', 'nj/government/usaspending_contracts_pop_FY2024_NJ.csv',
     'USAspending FY2024 federal contracts by place of performance, NJ row.', key='two_letter', keep=lambda v: v == 'NJ')
gv = pd.read_csv(S / 'government_2024.csv', dtype={'code': str})
gnj = gv[gv.code == NJ].copy(); tot = gv[gv.code != US].select_dtypes('number').sum(); tot['code'] = 'sum_of_states'
frame(pd.concat([gnj, tot.to_frame().T]), 'nj/government/government_proxy_2024_NJ.csv',
      'RECON2024/process/SUT/government_2024.csv', 'NJ row + sum over states',
      'government final-demand proxy (Census S&L $k, USAspending federal $), NJ and its denominator.', index=False)
rows(G / 'irs_soi_historic_table2_TY2022_TY2023_state_totals.csv', 'nj/government/irs_soi_table2_TY2023_NJ.csv',
     'IRS SOI Historic Table 2 state totals, NJ row.', key='STATE', keep=lambda v: v == 'NJ')
# jobs (BEA employment continuation, release 2026-09-27)
E = ROOT / 'BEA_EMP' / 'release_20260927'
for f in sorted(E.glob('*.csv')):
    k = 'st' if 'state' in f.name else list(pd.read_csv(f, nrows=1).columns)[0]
    rows(f, f'nj/jobs/{f.name.replace(".csv", "_NJ.csv")}', 'BEA jobs continuation release 2026-09-27, NJ rows.',
         key=k, keep=lambda v: str(v).zfill(2).startswith('34') if k == 'st' else str(v).zfill(5).startswith('34'))
copy(E / 'README_2026-09-27_1545Z.md', 'nj/jobs/README_release_2026-09-27.md', 'Release notes of the jobs series.')
for f in sorted((RAW / 'Proprietors' / 'release_20260927').glob('*.csv')):
    k = list(pd.read_csv(f, nrows=1).columns)[0]
    rows(f, f'nj/jobs/{f.name.replace(".csv", "_NJ.csv")}', 'Assembly inputs derived from the release, NJ rows.',
         key=k, keep=lambda v: str(v).zfill(2).startswith('34') or str(v).zfill(5).startswith('34'))
d = pd.read_csv(B / 'bea_emp_s_balanced.csv', dtype=str)
frame(d[d.code.str.startswith(NJ)], 'nj/jobs/bea_emp_s_balanced_NJ.csv', 'RECON2024/process/BEA/bea_emp_s_balanced.csv',
      'rows for NJ', 'NJ jobs by BEA line after assemble_employment (release-controlled).', index=False)
d = pd.read_csv(Q / 'qcew_final.csv', dtype=str)
kc = [c for c in d.columns if c.lower() in ('region', 'area_fips', 'area', 'code')][0]
frame(d[d[kc].astype(str).str.startswith('34')], 'nj/qcew/qcew_final_2024_NJ.csv.gz', 'RECON2024/process/QCEW/qcew_final.csv',
      f'rows where {kc} is NJ (state and counties)', 'QCEW 2024 jobs, wages, establishments bridged to the 421 SUT industries, '
      'suppressed cells filled.', index=False)
# international trade (ESR, common-rate scenario)
TR = RAW / 'Trade' / 'out_2024_ESR'
for f in ['retained_imports_state_bea_2024_SUTscaled.csv', 'netted_exports_state_bea_2024.csv',
          'exports_with_gateway_state_bea_2024.csv', 'gross_exports_state_bea_2024.csv',
          'retained_imports_state_bea_2024.csv']:
    d = pd.read_csv(TR / f, index_col=0)
    c = [x for x in d.columns if x in ('NJ', '34', '34000', 'New Jersey')]
    frame(d[c] if c else d.loc[[i for i in d.index if str(i) in ('NJ', '34', '34000')]],
          f'nj/trade/{f.replace("state", "NJ")}', f'raw/Trade/out_2024_ESR/{f}', 'New Jersey column',
          'NJ international trade by BEA commodity ($), ESR common-rate scenario.')
for f in ['gateway_margin_state_2024.csv', 'state_totals_2024.csv', 'sut_group_scaling_factors_2024.csv', 'RUN_NOTES.txt']:
    copy(TR / f, f'nj/trade/{f}', 'Trade build summary (one row per state; small) / run notes.')
# the trade build read ESR_package_20260927_1541Z; that package has since moved to _superseded/ and its
# series file is byte-identical (md5 edae59c6cdf03307879e0f637738c450) to the current 1626Z package's
ESR = ROOT / 'Statewise International Trade' / 'ESR_package_20260927_1626Z' / 'ESR_3_Supplementary' / 'series' / 'account_state_hs6_2024.csv.gz'
rows(ESR, 'nj/trade/account_NJ_hs6_2024.csv.gz', 'ESR HS6 trade account for NJ (gross, re-exports, retained, gateway).',
     key='state', keep=lambda v: v in ('NJ', '34'))

# ---------------------------------------------------------------------------
# 5. Data dictionary
# ---------------------------------------------------------------------------
with open(PKG / 'data' / 'DATA_DICTIONARY.md', 'w', encoding='utf-8') as fh:
    fh.write('# Data dictionary (written by code/extract_nj_inputs.py)\n\n'
             '| file (under data/) | source (under RPC_explorations/) | extraction | content |\n|---|---|---|---|\n')
    for r in LOG:
        fh.write('| ' + ' | '.join(str(x).replace('|', '/') for x in r) + ' |\n')
print(len(LOG), 'files written')
