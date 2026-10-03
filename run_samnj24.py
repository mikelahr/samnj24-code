"""run_samnj24.py -- build the Social Accounting Matrix for New Jersey 2024 (SAMNJ24).

Usage (from the package root, Python >= 3.10 with numpy, pandas, openpyxl):

    python run_samnj24.py                 # writes output/ and output/run_log.txt
    python run_samnj24.py --check         # ... and compares every product with product/

This one script turns the New Jersey inputs in data/ into the accounts reported in
the paper.  It runs four stages, each documented where it is defined:

  Stage A  New Jersey direct-requirements matrix and state Use tables.
           National 421-sector 2024 SUT (data/national_sut_421) + New Jersey's
           industry vectors and international import vector (data/nj/state_io)
           -> A_NJ (industry x industry), Use_NJ total and net of international
           imports.  This re-implements, for New Jersey alone, RECON2024's
           `sut.export_state_use_A`, and checks itself against the RECON2024 files.
  Stage B  The 65-account core SAM (Table B2): 57 productive sectors, labour,
           capital, households, corporations, government, capital account, rest of
           the United States, rest of the world.  $ millions, purchasers' prices.
  Stage C  The 19-account institutional SAM (Table 7): government split into
           federal, state and local; tax and transfer accounts.
  Stage D  Control totals (Table 8) and accounting checks.
  Stage E  The 75-account core SAM: the same sector, factor and external block as
           Stage B, with government split into federal, state and local, taxes
           routed through five tax accounts, and subsidies paid through a subsidy
           account so that no cell is negative.

What is NOT recomputed here
---------------------------
New Jersey's industry vectors (output, value added by component, final demand,
international trade, regional purchase coefficients) are products of the national
RECON2024 run, because each state's figures are constrained to national totals and
the RPC model is estimated on all states.  They enter as data/nj/state_io/
recon2024_34000.csv and vectors_34000.csv.  The code that made them is in
code/recon2024 (the package) and code/scripts; the order in which it was run is in
code/RUN_ORDER_upstream.md; New Jersey's rows of every input to that run are in
data/nj.

Conventions
-----------
* Cell (r, c) of a SAM is a payment from column account c to row account r.
* RECON2024 vectors and state matrices are in $ thousands; the SAM in $ millions.
* SAINC5N and SAINC35 are in $ thousands; SAINC4 and SAGDP1 in $ millions;
  SAGDP5/6 and the Census finance items in $ thousands.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, OUT, PROD = HERE / 'data', HERE / 'output', HERE / 'product'
NJ, US = '34000', '00000'
LOG = []


def log(msg):
    LOG.append(msg)
    print(msg)


def safe_division(num, den):
    """num / den, 0 where den == 0 (RECON2024 utilities.safe_division)."""
    den = np.asarray(den, dtype=float)
    return np.asarray(num, dtype=float) * ((den != 0) / (den + (den == 0)))


# =============================================================================
# Inputs that are read from the source extracts in data/nj
# =============================================================================
def bea_line(table, line, year='2024'):
    """One New Jersey value from a BEA regional table extract (data/nj/bea)."""
    f = {'SAINC4': 'SAINC4__NJ_1929_2025.csv', 'SAINC5N': 'SAINC5N__NJ_1998_2025.csv',
         'SAINC35': 'SAINC35__NJ_1929_2024.csv', 'SAGDP1': 'SAGDP1__NJ_1997_2025.csv',
         'SAGDP5': 'SAGDP5_NJ_1997_2024.csv', 'SAGDP6': 'SAGDP6_NJ_1997_2024.csv'}[table]
    d = pd.read_csv(DATA / 'nj' / 'bea' / f, dtype=str, encoding='latin1')
    v = d.loc[d.LineCode.str.strip() == str(line), year]
    assert len(v) == 1, (table, line)
    return float(v.iloc[0])


def institutional_inputs():
    """Every institutional flow of the SAM ($ millions), with its source.

    Returned as a DataFrame (key, value, source) that is also written to
    output/institutional_inputs_2024.csv.  All values are read from the NJ source
    extracts except the two marked TRANSCRIBED, whose source tables are not in the
    package (see README, 'Open items')."""
    I = []
    def add(key, value, source):
        I.append((key, float(value), source))
    k = 1000.0
    # --- BEA personal income (SAINC5N $k; SAINC4 $M)
    add('ssc_employer', bea_line('SAINC5N', 38) / k, 'BEA SAINC5N line 38: employer contributions for government social insurance')
    add('ssc_employee', bea_line('SAINC5N', 37) / k, 'BEA SAINC5N line 37: employee and self-employed contributions for government social insurance')
    add('residence_adjustment', bea_line('SAINC4', 42), 'BEA SAINC4 line 42: adjustment for residence ($M, as published)')
    add('dividends_interest_rent', bea_line('SAINC5N', 46) / k, 'BEA SAINC5N line 46: dividends, interest, and rent (paid to households through the corporate account)')
    add('proprietors_income', bea_line('SAINC5N', 70) / k, 'BEA SAINC5N line 70: proprietors\' income')
    # --- BEA transfer receipts (SAINC35 $k)
    for line, key, txt in [(2000, 'tr_gov_to_individuals', 'current transfer receipts of individuals from governments'),
                           (2410, 'tr_state_ui', 'state unemployment insurance compensation'),
                           (2220, 'tr_public_assistance_medical', 'public assistance medical care benefits (Medicaid, CHIP)'),
                           (3100, 'tr_federal_to_npish', 'receipts of nonprofits from the federal government'),
                           (3200, 'tr_sl_to_npish', 'receipts of nonprofits from state and local governments'),
                           (3300, 'tr_business_to_npish', 'receipts of nonprofits from businesses'),
                           (4000, 'tr_business_to_individuals', 'current transfer receipts of individuals from businesses')]:
        add(key, bea_line('SAINC35', line) / k, f'BEA SAINC35 line {line}: {txt}')
    # --- BEA GDP: taxes on production and imports less subsidies (SAGDP6/5 $k)
    add('topi', bea_line('SAGDP6', 1) / k, 'BEA SAGDP6 line 1: taxes on production and imports, all industries')
    add('subsidies', -bea_line('SAGDP5', 1) / k, 'BEA SAGDP5 line 1: subsidies, all industries (sign reversed)')
    # --- IRS Data Book FY2024 (federal income taxes, $k)
    irs = pd.read_csv(DATA / 'nj' / 'government' / 'irs_state_collections_FY2024_NJ.csv')
    add('cit_federal', irs.business_income.iloc[0] / k, 'IRS Data Book FY2024, NJ: corporation income tax gross collections')
    add('irs_withheld_fica', 117945.149, 'TRANSCRIBED: IRS Data Book FY2024 Table 5, NJ, individual income tax withheld and FICA ($M)')
    add('irs_payments_seca', 31323.647, 'TRANSCRIBED: IRS Data Book FY2024 Table 5, NJ, individual income tax payments and SECA ($M)')
    # --- USAspending: federal direct spending in NJ net of transfers
    add('federal_direct_spending', 29600.0, 'TRANSCRIBED: USAspending FY2024 federal direct spending in NJ net of transfers ($3.5bn defense + $26.1bn nondefense), paper Section 3')
    df = pd.DataFrame(I, columns=['key', 'value_musd', 'source']).set_index('key')
    # derived: federal personal income tax = IRS individual collections less the two
    # social contributions, which BEA records separately (paper, Table 5 note)
    v = df.value_musd
    df.loc['pit_federal'] = [v.irs_withheld_fica + v.irs_payments_seca - v.ssc_employee - v.ssc_employer,
                             'derived: irs_withheld_fica + irs_payments_seca - ssc_employee - ssc_employer']
    df.loc['net_taxes_bea'] = [v.topi - v.subsidies, 'derived: topi - subsidies (BEA SAGDP1 line 6)']
    return df


def census():
    """Census 2024 State & Local Government Finances, NJ: item code -> level -> $ millions."""
    C = json.load(open(DATA / 'nj' / 'government' / 'census24_nj.json'))
    return {k.split('|')[0]: {lev: float(v) / 1e3 for lev, v in d.items()} for k, d in C.items()}


# =============================================================================
# Stage A -- New Jersey A and Use matrices (use-side domestication)
# =============================================================================
def stage_a():
    """New Jersey's direct-requirements matrix on the 421-industry axis.

    Steps (all on the commodity x industry axis of the balanced, margin-adjusted
    national SUT, data/national_sut_421/sut_det_2024_margins.xlsx):

    1. B = national intermediate-use coefficients, use_int(c, j) / output(j).
    2. Productivity adjustment: column j of B is scaled by r_j = (NJ intermediate
       use / NJ output in j) / (US intermediate use / US output in j), bounded to
       [0.25, 4]; r_j = 1 where the US fraction is 0.  -> B_s.
    3. NJ total use of each commodity, U_s(c) = intermediate use (B_s x_s) + NJ
       household consumption + government use (US government final use x NJ's share
       of US personal income) + investment, inventories and exports (US final use x
       NJ's share of the producing industries' output, through D).
    4. International import share of use at basic prices: imports are valued at
       customs/CIF value and use at purchasers' prices, so the share is taken
       against use net of trade and transport margins:
       p_basic(c) = m_NJ(c) / (U_s(c) x basic_share(c)), capped at 0.95; zero for
       commodities whose national imports are <= 0 (BEA's CIF/FOB freight
       adjustment).  The share removed from the purchasers'-price coefficient is
       p(c) = p_basic(c) x basic_share(c): the margin part is domestic service.
    5. B_dom = (I - diag p) B_s ;  A = D' B_dom (industry x industry), D the
       market-share matrix.  Households row: earnings / output (earnings capped at
       GDP); households column: NJ PCE by industry / NJ personal income.
    Interstate (RPC) domestication is NOT in A; it is applied in Stage B.
    """
    S = DATA / 'national_sut_421' / 'csv'
    D = pd.read_csv(S / 'D_market_shares.csv', index_col=0)
    com, ind = list(D.index.astype(str)), list(D.columns.astype(str))
    D.index, D.columns = com, ind
    U = pd.read_csv(S / 'use_int.csv', index_col=0); U.index = U.index.astype(str); U.columns = U.columns.astype(str)
    U = U.reindex(index=com, columns=ind).fillna(0.0).astype(float)
    ic = pd.read_csv(S / 'industries.csv', index_col=0); ic.index = ic.index.astype(str)
    cc = pd.read_csv(S / 'commodities.csv', index_col=0); cc.index = cc.index.astype(str)
    x_us = ic.loc[ind, 'tot'].astype(float)
    B = U.div(x_us.replace(0, np.nan), axis=1).fillna(0.0)                       # step 1
    F = pd.read_csv(S / 'use_fin.csv', index_col=0); F.index = F.index.astype(str)
    F = F.reindex(index=com).fillna(0.0).astype(float)
    nat = pd.read_csv(DATA / 'national_sut_421' / 'vectors_00000.csv', dtype={'region': str}).set_index('industry')
    nat.index = nat.index.astype(str)
    st = pd.read_csv(DATA / 'nj' / 'state_io' / 'vectors_34000.csv', dtype={'region': str}).set_index('industry')
    st.index = st.index.astype(str); st = st.reindex(ind)
    reg = pd.read_csv(DATA / 'nj' / 'state_io' / 'regions_income_2024.csv', dtype={'code': str}).set_index('code')
    inc_nj, inc_us = float(reg.loc[NJ, 'income']), float(reg.loc[US, 'income'])
    M = pd.read_csv(DATA / 'nj' / 'state_io' / 'imports_nj_commodity_2024.csv', index_col=0)
    M.index = M.index.astype(str); m_s = M[NJ].reindex(com).fillna(0.0).values * 1000.0         # $M -> $k
    pce = pd.read_csv(DATA / 'nj' / 'state_io' / 'pce_nj_commodity_2024.csv', index_col=0)
    pce.index = pce.index.astype(str); pce_s = pce[NJ].reindex(com).fillna(0.0).values          # $k
    mg = pd.read_csv(S / 'margin_totals_by_commodity.csv', index_col=0); mg.index = mg.index.astype(str)
    margin = mg.margin_trade.reindex(com).fillna(0.0) + mg.margin_transport.reindex(com).fillna(0.0)
    use_pp = cc.loc[com, 'use'].astype(float)
    basic_share = (1.0 - margin / use_pp.replace(0, np.nan)).clip(lower=0.05, upper=1.0).fillna(1.0)
    nonpos_import = (nat.imports.reindex(com).astype(float).fillna(1.0) <= 0)
    # step 2
    frac_us = nat.intermediate_use.reindex(ind).astype(float)
    r = safe_division(st.intermediate_use.astype(float).values, frac_us.values)
    r = np.where(frac_us.values > 0, r, 1.0)
    r = np.clip(r, 0.25, 4.0)
    B_s = B.multiply(r, axis=1)
    # step 3
    x_s = st.output.astype(float).fillna(0.0).values
    inter_s = B_s.values.dot(x_s)
    gov_s = 1000.0 * F['government'].values * (inc_nj / inc_us)
    x_share = (st.output.astype(float) / nat.output_sum_of_states.reindex(ind).astype(float).replace(0, np.nan)).fillna(0.0)
    prod_share = (D.values * x_share.values[None, :]).sum(axis=1)
    oth_s = 1000.0 * (F['investment'] + F['inventory'] + F['exports']).values * prod_share
    U_s = inter_s + pce_s + gov_s + oth_s
    # step 4
    U_basic = U_s * basic_share.values
    p_basic = np.where(U_basic > 0, m_s / np.where(U_basic > 0, U_basic, 1.0), 0.0)
    p_basic = np.clip(p_basic, 0.0, 0.95)
    p_basic = np.where(nonpos_import.values, 0.0, p_basic)
    p_s = p_basic * basic_share.values
    # step 5
    B_dom = B_s.multiply(1.0 - p_s, axis=0)
    A = pd.DataFrame(D.values.T.dot(B_dom.values), index=ind, columns=ind)
    egdp = pd.Series(safe_division(st.earnings.astype(float), st.gdp.astype(float)), index=ind).clip(0, 1)
    A.loc['households'] = pd.Series(safe_division(egdp * st.gdp.astype(float), st.output.astype(float)), index=ind).fillna(0.0).values
    A['households'] = 0.0
    A.loc[ind, 'households'] = pd.Series(safe_division(st.pce.astype(float), inc_nj), index=ind).fillna(0.0).values
    A = A.round(6); A.index.name = 'industry'
    UT = B_s.multiply(x_s, axis=1).round(0)
    UD = B_dom.multiply(x_s, axis=1).round(0)
    P = pd.DataFrame({'p_import_basic': p_basic, 'basic_share_of_use': basic_share.values, 'p_applied': p_s},
                     index=pd.Index(com, name='commodity')).round(5)
    o = OUT / 'stageA'; o.mkdir(parents=True, exist_ok=True)
    A.to_csv(o / 'A_NJ_2024.csv'); UT.to_csv(o / 'Use_total_NJ_2024.csv.gz', compression='gzip')
    UD.to_csv(o / 'Use_domestic_NJ_2024.csv.gz', compression='gzip'); P.to_csv(o / 'import_shares_NJ_2024.csv')
    # self-check against the files RECON2024 wrote
    ref = DATA / 'nj' / 'state_io'
    Ar = pd.read_csv(ref / 'recon2024_Amatrix_34000.csv', index_col=0); Ar.index = Ar.index.astype(str); Ar.columns = Ar.columns.astype(str)
    UTr = pd.read_csv(ref / 'recon2024_Use_total_34000.csv.gz', index_col=0); UTr.index = UTr.index.astype(str); UTr.columns = UTr.columns.astype(str)
    UDr = pd.read_csv(ref / 'recon2024_Use_domestic_34000.csv.gz', index_col=0); UDr.index = UDr.index.astype(str); UDr.columns = UDr.columns.astype(str)
    dA = (A - Ar.loc[A.index, A.columns]).abs().values.max()
    dUT = (UT - UTr.loc[UT.index, UT.columns]).abs().values.max()
    dUD = (UD - UDr.loc[UD.index, UD.columns]).abs().values.max()
    log(f'Stage A: A_NJ {A.shape}, column sums (industries) mean {A.loc[ind, ind].sum().mean():.4f}; '
        f'use-weighted import share {np.average(p_s, weights=np.maximum(U_s, 0) + 1e-9):.4f}')
    log(f'Stage A check vs RECON2024: max |dA| = {dA:.2e}; max |dUse_total| = {dUT:.0f} $k; max |dUse_domestic| = {dUD:.0f} $k')
    return A, UT, UD


# =============================================================================
# Stage B -- the 65-account core SAM
# =============================================================================
S57 = ['AG11', 'MIN21', '22', '23', '311FT', 'TEX', '321', '322', '323', 'PPN', '325', '331', '332', '333', '334', '335',
       'TEQ336', '337', '339', '42', '424GR', '441', '4A0', '445', '452', 'TRANS_x484', '484', 'GFE', '493', '511', '512',
       '513', '514', '521CI', '523', '524', '525', 'RE', '532RL', '5411', '5412OP', '5415', '55', 'ADMIN56', '61', '621',
       '622', 'HSA', '624CDC', 'ARTS71', '721', '722', '81', 'GSLG', 'GSLE', 'GFGD', 'GFGN']
INST = ['LAB', 'CAP', 'HH', 'CORP', 'GOV', 'KAP', 'ROUS', 'ROW']


def stage_b(A, UT, UD, I):
    """The 65 x 65 SAM in $ millions (Table B2).

    Industry block (all 421 industries, then aggregated to 57 sectors):
      * NJ inputs from the US: Zdom = A x output (A from Stage A, net of
        international imports).  Of these, the share rpc(i) is produced in NJ
        (Zloc, the local intermediate block); the rest comes from other states
        (row ROUS).  rpc = the industry's regional purchase coefficient.
      * International intermediate imports by buying industry = total
        intermediate use (Use_total column sums) - Zdom column sums, floored at 0
        (row ROW).
      * Final demand (household consumption, government, investment+inventories)
        from the NJ vector.  Imports for final use = the industry's imports less
        its intermediate imports, capped at the commodity's total final use (a
        commodity cannot import more than its final use), spread over the final-
        demand columns by their shares.  Of the rest, rpc is local and 1 - rpc
        comes from other states.
      * Value added: compensation (LAB), gross operating surplus (CAP), net taxes
        on production and on commodities (GOV).  Net taxes are controlled to
        BEA's total (SAGDP6 less SAGDP5), the gap to the state IO accounts being
        added to taxes on production in proportion to them.  The residual that
        closes each industry column goes to GOS.
      * Outflows (sales to other states, column ROUS) = row total (output +
        commodity taxes) - local intermediate sales - local final sales -
        international exports.  Where local sales plus exports exceed the row
        total, local sales are scaled back and the excess is treated as inflows.
    Institutional block: labour income, social contributions, residence
    adjustment, capital income, corporate and household taxes, transfers, from
    institutional_inputs() and the Census.  Saving is the residual of each
    institution; the external balances close through the capital account.
    """
    v = pd.read_csv(DATA / 'nj' / 'state_io' / 'recon2024_34000.csv', dtype={'industry': str}).set_index('industry')
    v = v.drop(index='Households')
    ind = list(v.index); x = v.output
    A = A.loc[ind, ind]
    rpc = v.rpc
    # --- intermediate
    Zdom = pd.DataFrame(A.values * x.values, index=ind, columns=ind)          # US-produced inputs ($k)
    Zloc = Zdom.mul(rpc, axis=0)                                             # NJ-produced
    inflow_int = (Zdom - Zloc).sum(axis=0)                                        # from other states, by buying industry
    int_total = UT.sum(axis=0).reindex(ind).fillna(0)
    imp_int_col = (int_total - Zdom.sum(axis=0)).clip(lower=0)                    # international, by buying industry
    imp_int_row = (UT - UD).sum(axis=1)                                           # international, by commodity
    # --- final demand
    fd = v[['pce', 'government', 'investment', 'inventory']].copy()
    fd['kap'] = fd.investment + fd.inventory
    fd = fd[['pce', 'government', 'kap']]
    imp_fin = (v.imports - imp_int_row.reindex(ind).fillna(0)).clip(lower=0)
    fdpos = fd.clip(lower=0); tot = fdpos.sum(axis=1)
    share = fdpos.div(tot.replace(0, np.nan), axis=0).fillna(0)
    imp_fin_uncapped = imp_fin.copy()
    imp_fin = np.minimum(imp_fin, tot)                                       # cap: no more than final use
    imp_fd = share.mul(imp_fin, axis=0)
    dom_fd = fd - imp_fd
    loc_fd = dom_fd.mul(rpc, axis=0); inflow_fd = dom_fd - loc_fd
    # --- value added and taxes
    comp = v.compensation; surp = v.surplus; ntp = v.nettax_production; ntc = v.nettax_commodity
    BEA_NT = I.value_musd.net_taxes_bea * 1000.0                             # $k
    dnt = BEA_NT - (ntp + ntc).sum(); ntp = ntp + dnt * ntp / ntp.sum()
    log(f'Stage B: net taxes raised to BEA by ${dnt / 1e3:,.3f}M (to production taxes; GOS absorbs it)')
    coltot = Zloc.sum(axis=0) + inflow_int + imp_int_col + comp + surp + ntp + ntc
    colres = x + ntc - coltot                                                # closes the column -> capital
    surp = surp + colres
    rowtot = x + ntc
    locsales = Zloc.sum(axis=1) + loc_fd.sum(axis=1)
    f = ((rowtot - v.exports).clip(lower=0) / locsales.replace(0, np.nan)).clip(upper=1).fillna(1)
    log(f'Stage B: industries with local sales scaled back: {int((f < 1).sum())}; '
        f'${((1 - f) * locsales).sum() / 1e6:,.2f}bn moved to inflows')
    Zfix = Zloc.mul(f, axis=0); inflow_int = inflow_int + (Zloc - Zfix).sum(axis=0); Zloc = Zfix
    lf = loc_fd.mul(f, axis=0); inflow_fd = inflow_fd + (loc_fd - lf); loc_fd = lf
    outflow = rowtot - Zloc.sum(axis=1) - loc_fd.sum(axis=1) - v.exports
    # --- 421 -> 57 sectors
    m = pd.read_csv(DATA / 'bridges' / 'sut421_to_samnj24_57.csv', dtype=str).set_index('sut421').sam57
    sec = m.reindex(ind)
    assert sec.notna().all() and set(sec) <= set(S57)
    acc = S57 + INST
    M = pd.DataFrame(0.0, index=acc, columns=acc)
    g = lambda s: s.groupby(sec).sum().reindex(S57).fillna(0)
    M.loc[S57, S57] = Zloc.groupby(sec).sum().T.groupby(sec).sum().T.reindex(index=S57, columns=S57).fillna(0).values
    M.loc['ROUS', S57] = g(inflow_int).values; M.loc['ROW', S57] = g(imp_int_col).values
    M.loc['LAB', S57] = g(comp).values; M.loc['CAP', S57] = g(surp).values; M.loc['GOV', S57] = g(ntp + ntc).values
    M.loc[S57, 'HH'] = g(loc_fd.pce).values; M.loc[S57, 'GOV'] = g(loc_fd.government).values; M.loc[S57, 'KAP'] = g(loc_fd.kap).values
    M.loc['ROUS', 'HH'] = inflow_fd.pce.sum(); M.loc['ROUS', 'GOV'] = inflow_fd.government.sum(); M.loc['ROUS', 'KAP'] = inflow_fd.kap.sum()
    M.loc['ROW', 'HH'] = imp_fd.pce.sum(); M.loc['ROW', 'GOV'] = imp_fd.government.sum(); M.loc['ROW', 'KAP'] = imp_fd.kap.sum()
    M.loc[S57, 'ROW'] = g(v.exports).values; M.loc[S57, 'ROUS'] = g(outflow).values
    M = M / 1e3                                                              # $ millions
    # --- institutions ($ millions)
    q = I.value_musd; C = census(); cs = lambda code, lev: C[code].get(lev, 0.0)
    M.loc['GOV', 'LAB'] = q.ssc_employer                                     # employer SSC out of compensation
    M.loc['HH', 'LAB'] = M.loc['LAB', S57].sum() - q.ssc_employer
    M.loc['HH', 'ROUS'] = q.residence_adjustment                             # net earnings of commuters
    M.loc['CORP', 'CAP'] = M.loc['CAP', S57].sum()                           # all GOS to corporations
    corp_hh = q.proprietors_income + q.dividends_interest_rent + q.tr_business_to_individuals + q.tr_business_to_npish
    corp_gov = q.cit_federal + cs('LF0024', 'State') + cs('LF0040', 'State and Local')
    M.loc['HH', 'CORP'] = corp_hh; M.loc['GOV', 'CORP'] = corp_gov
    M.loc['KAP', 'CORP'] = M.loc['CORP', 'CAP'] - corp_hh - corp_gov          # corporate saving
    M.loc['GOV', 'HH'] = q.pit_federal + cs('LF0023', 'State') + q.ssc_employee
    M.loc['HH', 'GOV'] = q.tr_gov_to_individuals + q.tr_federal_to_npish + q.tr_sl_to_npish
    M.loc['KAP', 'HH'] = M.loc['HH'].sum() - M[['HH']].sum().iloc[0]         # household saving
    M.loc['KAP', 'GOV'] = M.loc['GOV'].sum() - M[['GOV']].sum().iloc[0]      # government saving
    rowgap = M.loc['ROW'].sum() - M['ROW'].sum()                             # NJ payments to ROW less receipts
    M.loc['KAP', 'ROW'] = max(rowgap, 0); M.loc['ROW', 'KAP'] += max(-rowgap, 0)
    kapgap = M.loc['KAP'].sum() - M['KAP'].sum()                             # closes through the rest of the US
    if kapgap >= 0: M.loc['ROUS', 'KAP'] += kapgap
    else: M.loc['KAP', 'ROUS'] += -kapgap
    d = (M.sum(axis=1) - M.sum(axis=0))
    log(f'Stage B: 65-account SAM, total flows ${M.values.sum() / 1e3:,.1f}bn; max row-column imbalance ${d.abs().max():.3f}M; '
        f'negative cells: {[(r, c, round(M.loc[r, c], 1)) for r, c in M[M < -0.05].stack().index]}')
    log(f'Stage B: imports for final use capped by ${(imp_fin_uncapped - imp_fin).sum() / 1e6:,.2f}bn; '
        f'column residual to GOS ${colres.sum() / 1e3:,.1f}M')
    parts = dict(v=v, loc_fd=loc_fd, inflow_fd=inflow_fd, imp_fd=imp_fd, ntp=ntp, ntc=ntc, colres=colres, sec=sec,
                 imp_int_col=imp_int_col, imp_fin=imp_fin, imp_fin_uncapped=imp_fin_uncapped)
    return M, parts


# =============================================================================
# Stage C -- the 19-account institutional SAM (Table 7)
# =============================================================================
T7 = ['HH', 'COR', 'FED', 'STA', 'LOC', 'TRs', 'TRd', 'PIT', 'CIT', 'SSC', 'PRD', 'COM', 'GCF', 'SAV', 'LAB', 'CAP', 'IND', 'RUS', 'ROW']


def stage_c(M, parts, I):
    """Table 7: government split into federal (FED), state (STA) and local (LOC).

    Industry, factor and external entries are carried over from the core
    unchanged.  Institutional entries are rebuilt:
      * taxes pass through tax accounts (PIT, CIT, SSC, PRD production, COM
        commodities) to the level that collects them: state and local get their
        Census revenues (property, license and other taxes on production; general
        and selective sales taxes; income taxes); the federal government gets the
        IRS income taxes, all social contributions, and the rest of BEA's net taxes
        on production and imports;
      * intergovernmental revenue is recorded gross, as each level reports it;
      * transfers pass through TRs (social, SAINC35 by paying level) and TRd
        (dividends, interest and rent from corporations);
      * government consumption by level: federal = the federal general-government
        industries' final demand; state/local split by Census current operations
        by function (education, health and hospitals, other);
      * government investment: state and local capital outlay (Census); federal =
        federal direct spending x the state-and-local capital-outlay ratio;
      * each institution's saving is the residual of its account; saving
        transferred to gross capital formation closes GCF, and the savings
        account's payment to the rest of the US closes the matrix.
    """
    v, loc_fd, inflow_fd, imp_fd = parts['v'], parts['loc_fd'], parts['inflow_fd'], parts['imp_fd']
    q = I.value_musd; C = census(); cs = lambda code, lev: C[code].get(lev, 0.0)
    T = pd.DataFrame(0.0, index=T7, columns=T7)
    IND = S57
    T.loc['IND', 'IND'] = M.loc[IND, IND].values.sum()
    T.loc['RUS', 'IND'] = M.loc['ROUS', IND].sum(); T.loc['ROW', 'IND'] = M.loc['ROW', IND].sum()
    T.loc['LAB', 'IND'] = M.loc['LAB', IND].sum(); T.loc['CAP', 'IND'] = M.loc['CAP', IND].sum()
    ntp = parts['ntp'].sum() / 1e3; ntc = parts['ntc'].sum() / 1e3
    T.loc['PRD', 'IND'] = ntp; T.loc['COM', 'IND'] = ntc
    T.loc['IND', 'RUS'] = M.loc[IND, 'ROUS'].sum(); T.loc['IND', 'ROW'] = M.loc[IND, 'ROW'].sum()
    T.loc['LAB', 'RUS'] = M.loc['HH', 'ROUS']                                  # residence adjustment
    ssc_er = M.loc['GOV', 'LAB']; T.loc['SSC', 'LAB'] = ssc_er; T.loc['HH', 'LAB'] = T.loc['LAB'].sum() - ssc_er
    # capital income: proprietors' income to households, the rest to corporations
    T.loc['HH', 'CAP'] = q.proprietors_income; T.loc['COR', 'CAP'] = T.loc['CAP', 'IND'] - q.proprietors_income
    T.loc['TRd', 'COR'] = q.dividends_interest_rent; T.loc['HH', 'TRd'] = q.dividends_interest_rent
    cit_s = cs('LF0024', 'State')
    T.loc['CIT', 'COR'] = q.cit_federal + cit_s; T.loc['FED', 'CIT'] = q.cit_federal; T.loc['STA', 'CIT'] = cit_s
    T.loc['HH', 'COR'] = q.tr_business_to_individuals + q.tr_business_to_npish
    T.loc['STA', 'COR'] = cs('LF0040', 'State'); T.loc['LOC', 'COR'] = cs('LF0040', 'Local')   # charges + misc. revenue
    # household taxes
    pit_s = cs('LF0023', 'State')
    T.loc['PIT', 'HH'] = q.pit_federal + pit_s; T.loc['FED', 'PIT'] = q.pit_federal; T.loc['STA', 'PIT'] = pit_s
    T.loc['SSC', 'HH'] = q.ssc_employee; T.loc['FED', 'SSC'] = ssc_er + q.ssc_employee
    # taxes on production: Census property (9), motor-vehicle license (20) and other taxes (21);
    # "other" = total taxes - property - sales - income - motor-vehicle license
    oth = lambda lev: cs('LF0008', lev) - cs('LF0009', lev) - cs('LF0010', lev) - cs('LF0022', lev) - cs('LF0029', lev)
    prd_s = cs('LF0009', 'State') + cs('LF0029', 'State') + oth('State')
    prd_l = cs('LF0009', 'Local') + cs('LF0029', 'Local') + oth('Local')
    T.loc['STA', 'PRD'] = prd_s; T.loc['LOC', 'PRD'] = prd_l; T.loc['FED', 'PRD'] = ntp - prd_s - prd_l
    sal_s = cs('LF0010', 'State'); sal_l = cs('LF0010', 'Local')
    T.loc['STA', 'COM'] = sal_s; T.loc['LOC', 'COM'] = sal_l; T.loc['FED', 'COM'] = ntc - sal_s - sal_l
    # intergovernmental revenue, gross
    T.loc['STA', 'FED'] = cs('LF0004', 'State'); T.loc['LOC', 'FED'] = cs('LF0004', 'Local')
    T.loc['LOC', 'STA'] = cs('LF0005', 'Local'); T.loc['STA', 'LOC'] = cs('LF0006', 'State')
    # social transfers: the state pays UI, public-assistance medical care and its NPISH transfers
    T.loc['TRs', 'STA'] = q.tr_state_ui + q.tr_public_assistance_medical + q.tr_sl_to_npish
    T.loc['TRs', 'FED'] = q.tr_gov_to_individuals - q.tr_state_ui - q.tr_public_assistance_medical + q.tr_federal_to_npish
    T.loc['HH', 'TRs'] = T.loc['TRs'].sum()
    # government consumption by level
    govc = pd.DataFrame({'loc': loc_fd.government, 'inf': inflow_fd.government, 'imp': imp_fd.government}) / 1e3
    fedrows = [i for i in govc.index if i in ('S00500', 'S00600')]
    slrows = [i for i in govc.index if i.startswith('GSLG')]
    assert abs(govc.drop(index=fedrows + slrows).values.sum()) < 1
    edu = (cs('LF0108', 'State'), cs('LF0108', 'Local'))
    hlt = (cs('LF0130', 'State') + cs('LF0133', 'State'), cs('LF0130', 'Local') + cs('LF0133', 'Local'))
    tot = (cs('LF0104', 'State'), cs('LF0104', 'Local')); wel = (cs('LF0124', 'State'), cs('LF0124', 'Local'))
    oth2 = tuple(tot[k] - wel[k] - edu[k] - hlt[k] for k in (0, 1))
    shs = {'GSLGE': edu[0] / sum(edu), 'GSLGH': hlt[0] / sum(hlt), 'GSLGO': oth2[0] / sum(oth2)}   # state shares
    for col, dest in [('loc', 'IND'), ('inf', 'RUS'), ('imp', 'ROW')]:
        T.loc[dest, 'FED'] = govc.loc[fedrows, col].sum()
        T.loc[dest, 'STA'] = sum(govc.loc[r, col] * shs[r] for r in slrows)
        T.loc[dest, 'LOC'] = sum(govc.loc[r, col] * (1 - shs[r]) for r in slrows)
    # household consumption
    T.loc['IND', 'HH'] = M.loc[IND, 'HH'].sum(); T.loc['RUS', 'HH'] = M.loc['ROUS', 'HH']; T.loc['ROW', 'HH'] = M.loc['ROW', 'HH']
    # gross capital formation and government investment
    T.loc['IND', 'GCF'] = M.loc[IND, 'KAP'].sum(); T.loc['RUS', 'GCF'] = parts['inflow_fd'].kap.sum() / 1e3; T.loc['ROW', 'GCF'] = M.loc['ROW', 'KAP']
    co_ratio = cs('LF0095', 'State and Local') / (cs('LF0095', 'State and Local') + cs('LF0094', 'State and Local'))
    T.loc['GCF', 'FED'] = co_ratio * q.federal_direct_spending
    T.loc['GCF', 'STA'] = cs('LF0095', 'State'); T.loc['GCF', 'LOC'] = cs('LF0095', 'Local')
    # saving residuals and closure
    for a in ['HH', 'FED', 'STA', 'LOC']:
        T.loc['SAV', a] = T.loc[a].sum() - T[a].sum()
    T.loc['SAV', 'COR'] = T.loc['COR'].sum() - (T['COR'].sum() - T.loc['SAV', 'COR'])
    T.loc['SAV', 'ROW'] = M.loc['KAP', 'ROW']
    T.loc['GCF', 'SAV'] = T['GCF'].sum() - T.loc['GCF'].sum()
    T.loc['RUS', 'SAV'] = T.loc['SAV'].sum() - T['SAV'].sum()
    d = T.sum(axis=1) - T.sum(axis=0)
    log(f'Stage C: 19-account SAM, total flows ${T.values.sum() / 1e3:,.1f}bn; max imbalance ${d.abs().max():.3f}M; '
        f'negative cells {int((T < -0.01).sum().sum())}; saving FED/STA/LOC {T.loc["SAV", "FED"]:,.1f}/'
        f'{T.loc["SAV", "STA"]:,.1f}/{T.loc["SAV", "LOC"]:,.1f} $M; closing entry RUS<-SAV {T.loc["RUS", "SAV"]:,.1f} $M')
    return T, dict(prd_s=prd_s, prd_l=prd_l, shares=shs, co_ratio=co_ratio)


# =============================================================================
# Stage D -- control totals (Table 8) and checks
# =============================================================================
def stage_d(M, T, parts, I):
    """Table 8: SAMNJ24 against the published figures it is built from."""
    v = parts['v']; q = I.value_musd
    IND = S57
    gos = M.loc['CAP', IND].sum(); gdp = M.loc[['LAB', 'CAP', 'GOV'], IND].values.sum()
    rows = [
        ('Industry output', 'Controlled to BEA 2024 SUTs', v.output.sum() / 1e3, v.output.sum() / 1e3),
        ('Gross domestic product', 'BEA SAGDP1 line 3', bea_line('SAGDP1', 3), gdp),
        ('Compensation of employees', 'BEA SAGDP1 line 4', bea_line('SAGDP1', 4), M.loc['LAB', IND].sum()),
        ('Gross operating surplus', 'BEA SAGDP1 line 5', bea_line('SAGDP1', 5), gos),
        ('Personal income', 'BEA SAINC4 line 10', bea_line('SAINC4', 10), T.loc['HH'].sum() - q.ssc_employee),
        ('Personal consumption', 'BEA state PCE (state IO accounts)', v.pce.sum() / 1e3,
         M.loc[IND, 'HH'].sum() + M.loc['ROUS', 'HH'] + M.loc['ROW', 'HH']),
        ('International exports', 'Census goods; services allocated', v.exports.sum() / 1e3, M.loc[IND, 'ROW'].sum()),
        ('International imports', 'Census goods; services allocated', v.imports.sum() / 1e3, M.loc['ROW'].sum()),
        ('Intergovernmental, Fed->State+Local', 'Census S&L Govt. Finances 2024', census()['LF0004']['State and Local'],
         T.loc['STA', 'FED'] + T.loc['LOC', 'FED']),
        ('Net production + commodity tax', 'BEA SAGDP6 less SAGDP5', q.net_taxes_bea, M.loc['GOV', IND].sum()),
    ]
    tab = pd.DataFrame(rows, columns=['aggregate', 'source', 'published_musd', 'samnj24_musd'])
    tab['diff_pct'] = 100 * (tab.samnj24_musd / tab.published_musd - 1)
    tab.loc[len(tab)] = ['Maximum account imbalance', 'Accounting identity', 0.0,
                         float((M.sum(axis=1) - M.sum(axis=0)).abs().max()), np.nan]
    log(f'Stage D: GOS {gos:,.1f} = state IO operating surplus {v.surplus.sum() / 1e3:,.1f} + column residual '
        f'{parts["colres"].sum() / 1e3:,.1f}; BEA GOS {bea_line("SAGDP1", 5):,.1f}')
    return tab



# =============================================================================
# Stage E -- the 75-account core SAM with federal, state and local government
# =============================================================================
ACC75_INST = ['LAB', 'CAP', 'HH', 'CORP', 'FED', 'STA', 'LOC', 'TRs', 'TRd',
              'PIT', 'CIT', 'SSC', 'PRD', 'COM', 'SUB', 'KAP', 'ROUS', 'ROW']


def stage_e(M, T, parts, I):
    """The core SAM with the government detail of Table 7 kept at sector level.

    75 accounts: the 57 sectors, labour, capital, households, corporations, the
    three governments (FED, STA, LOC), two transfer accounts (TRs social, TRd
    dividends), five tax accounts (PIT, CIT, SSC, PRD taxes on production, COM
    taxes on products), a subsidy account (SUB), the capital account, and the two
    external accounts.  Differences from the 65-account core:

      * government is three institutions instead of one, and taxes reach them
        through the tax accounts, with the same rules and the same Census and IRS
        figures as Stage C, so the two agree account by account;
      * taxes are GROSS and subsidies are a payment from the paying government
        (NIPA 2024: federal $93.5bn of $94.2bn) through SUB to the sectors.  The
        65-account core nets them inside GOV, which leaves two negative cells
        (GSLE -1,442 and GFE -4 $M).  Subsidies by sector are the national gross
        subsidies of the 2024 summary Use table (rows T00SUB and T00OSUB) spread
        to New Jersey by its share of national output within the summary sector;
      * saving stays in the single capital account of the core (Table 7 instead
        splits GCF and SAV).

    Industry, factor and external entries are carried over from the core
    unchanged, so the sector block is identical to Table B2.
    """
    v, loc_fd, inflow_fd, imp_fd, sec = (parts['v'], parts['loc_fd'], parts['inflow_fd'],
                                        parts['imp_fd'], parts['sec'])
    q = I.value_musd; C = census(); cs = lambda code, lev: C[code].get(lev, 0.0)
    acc = S57 + ACC75_INST
    E = pd.DataFrame(0.0, index=acc, columns=acc)
    keep = S57 + ['LAB', 'CAP', 'HH', 'CORP', 'KAP', 'ROUS', 'ROW']
    core = M.rename(index={'CORP': 'CORP'}, columns={'CORP': 'CORP'})
    E.loc[keep, keep] = core.loc[keep, keep].values
    gg = lambda s: s.groupby(sec).sum().reindex(S57).fillna(0.0)

    # --- gross taxes and subsidies by sector
    u = pd.read_excel(DATA / 'national_sut_421' / 'Use_2024_summary71.xlsx', sheet_name=0, header=None)
    hdr = u.iloc[5].astype(str).tolist()
    nat = {}
    for _, r in u.iterrows():
        if str(r[0]) in ('T00SUB', 'T00OSUB'):
            nat[str(r[0])] = pd.Series({h: pd.to_numeric(r[i], errors='coerce')
                                        for i, h in enumerate(hdr)}).dropna()
    bridge = pd.read_excel(DATA / 'bridges' / 'aggregation_bridge_421_to_71.xlsx', sheet_name='industries',
                           dtype={'offspring_code': str, 'parent_code': str}).set_index('offspring_code').parent_code
    usv = pd.read_csv(DATA / 'national_sut_421' / 'vectors_00000.csv', dtype={'industry': str}).set_index('industry')
    ntp, ntc = parts['ntp'] / 1e3, parts['ntc'] / 1e3          # $M by 421-industry, BEA-controlled
    w = pd.DataFrame({'summary': bridge.reindex(ntp.index), 'nj': v.output / 1e3,
                      'us': usv.output.reindex(ntp.index) / 1e3})
    w['share'] = safe_division(w.nj, w.groupby('summary').us.transform('sum'))
    sub = pd.DataFrame({t: pd.Series(nat[code].reindex(w.summary.values).fillna(0.0).values * w.share.values,
                                     index=w.index)
                        for t, code in (('production', 'T00OSUB'), ('commodity', 'T00SUB'))})
    net = pd.DataFrame({'production': ntp, 'commodity': ntc})
    sub = sub.clip(lower=0) + (-(net + sub.clip(lower=0))).clip(lower=0)   # keep gross non-negative
    gross = net + sub
    E.loc['PRD', S57] = gg(gross.production).values
    E.loc['COM', S57] = gg(gross.commodity).values
    E.loc[S57, 'SUB'] = gg(sub.sum(axis=1)).values
    sub_tot = float(sub.values.sum())
    sub_fed = sub_tot * 93.5 / 94.2                                        # NIPA 2024Q4 subsidies by payer
    E.loc['SUB', 'FED'] = sub_fed; E.loc['SUB', 'STA'] = sub_tot - sub_fed

    # --- taxes to the three levels: Stage C's net amounts plus what each pays to SUB
    for ta, col in (('PRD', 'production'), ('COM', 'commodity')):
        share_fed = sub_fed / sub_tot if sub_tot else 1.0
        s_sub = float(sub[col].sum())
        E.loc['STA', ta] = T.loc['STA', ta] + s_sub * (1 - share_fed)
        E.loc['LOC', ta] = T.loc['LOC', ta]
        E.loc['FED', ta] = E.loc[ta].sum() - E.loc['STA', ta] - E.loc['LOC', ta]

    # --- the other institutional entries, as Stage C
    ssc_er = M.loc['GOV', 'LAB']
    E.loc['SSC', 'LAB'] = ssc_er; E.loc['SSC', 'HH'] = q.ssc_employee
    E.loc['FED', 'SSC'] = ssc_er + q.ssc_employee
    E.loc['PIT', 'HH'] = q.pit_federal + cs('LF0023', 'State')
    E.loc['FED', 'PIT'] = q.pit_federal; E.loc['STA', 'PIT'] = cs('LF0023', 'State')
    E.loc['CIT', 'CORP'] = q.cit_federal + cs('LF0024', 'State')
    E.loc['FED', 'CIT'] = q.cit_federal; E.loc['STA', 'CIT'] = cs('LF0024', 'State')
    E.loc['STA', 'CORP'] = cs('LF0040', 'State'); E.loc['LOC', 'CORP'] = cs('LF0040', 'Local')
    E.loc['TRd', 'CORP'] = q.dividends_interest_rent; E.loc['HH', 'TRd'] = q.dividends_interest_rent
    E.loc['HH', 'CORP'] = q.tr_business_to_individuals + q.tr_business_to_npish
    E.loc['TRs', 'STA'] = q.tr_state_ui + q.tr_public_assistance_medical + q.tr_sl_to_npish
    E.loc['TRs', 'FED'] = (q.tr_gov_to_individuals - q.tr_state_ui
                           - q.tr_public_assistance_medical + q.tr_federal_to_npish)
    E.loc['HH', 'TRs'] = E.loc['TRs'].sum()
    E.loc['STA', 'FED'] = cs('LF0004', 'State'); E.loc['LOC', 'FED'] = cs('LF0004', 'Local')
    E.loc['LOC', 'STA'] = cs('LF0005', 'Local'); E.loc['STA', 'LOC'] = cs('LF0006', 'State')

    # --- government purchases by level, at sector level
    govc = pd.DataFrame({'loc': loc_fd.government, 'inf': inflow_fd.government,
                         'imp': imp_fd.government}) / 1e3
    fedrows = [i for i in govc.index if i in ('S00500', 'S00600')]
    slrows = [i for i in govc.index if i.startswith('GSLG')]
    edu = (cs('LF0108', 'State'), cs('LF0108', 'Local'))
    hlt = (cs('LF0130', 'State') + cs('LF0133', 'State'), cs('LF0130', 'Local') + cs('LF0133', 'Local'))
    tot = (cs('LF0104', 'State'), cs('LF0104', 'Local')); wel = (cs('LF0124', 'State'), cs('LF0124', 'Local'))
    oth2 = tuple(tot[k] - wel[k] - edu[k] - hlt[k] for k in (0, 1))
    shs = {'GSLGE': edu[0] / sum(edu), 'GSLGH': hlt[0] / sum(hlt), 'GSLGO': oth2[0] / sum(oth2)}
    for col, dest in [('loc', None), ('inf', 'ROUS'), ('imp', 'ROW')]:
        for r in fedrows + slrows:
            amt = govc.at[r, col]
            if abs(amt) < 1e-12:
                continue
            pay = {'FED': amt} if r in fedrows else {'STA': amt * shs[r], 'LOC': amt * (1 - shs[r])}
            row = dest if dest is not None else sec.get(r)
            for lv, a in pay.items():
                E.loc[row, lv] += a

    # --- government investment, then saving through the capital account
    co_ratio = cs('LF0095', 'State and Local') / (cs('LF0095', 'State and Local') + cs('LF0094', 'State and Local'))
    E.loc['KAP', 'FED'] += co_ratio * q.federal_direct_spending
    E.loc['KAP', 'STA'] += cs('LF0095', 'State'); E.loc['KAP', 'LOC'] += cs('LF0095', 'Local')
    saving = {}
    for a in ['HH', 'CORP', 'FED', 'STA', 'LOC']:
        saving[a] = E.loc[a].sum() - E[a].sum()
        E.loc['KAP', a] = saving[a] + E.loc['KAP', a]
    rowgap = E.loc['ROW'].sum() - E['ROW'].sum()
    if rowgap >= 0:
        E.loc['KAP', 'ROW'] += rowgap
    else:
        E.loc['ROW', 'KAP'] += -rowgap
    kapgap = E.loc['KAP'].sum() - E['KAP'].sum()
    if kapgap >= 0:
        E.loc['ROUS', 'KAP'] += kapgap
    else:
        E.loc['KAP', 'ROUS'] += -kapgap

    d = E.sum(axis=1) - E.sum(axis=0)
    log(f'Stage E: 75-account SAM, total flows ${E.values.sum() / 1e3:,.1f}bn; max imbalance ${d.abs().max():.3f}M; '
        f'negative cells {int((E < -0.01).sum().sum())}; gross taxes PRD ${E.loc["PRD"].sum():,.1f}M + COM '
        f'${E.loc["COM"].sum():,.1f}M less subsidies ${sub_tot:,.1f}M = ${E.loc["PRD"].sum() + E.loc["COM"].sum() - sub_tot:,.1f}M '
        f'(core GOV from industries ${M.loc["GOV", S57].sum():,.1f}M)')
    for a in ('FED', 'STA', 'LOC'):
        log(f'Stage E: {a} receipts ${E.loc[a].sum() / 1e3:,.1f}bn, outlays ${E[a].sum() / 1e3:,.1f}bn, '
            f'saving ${saving[a]:,.1f}M (Table 7 ${T.loc["SAV", a]:,.1f}M), investment '
            f'${E.loc["KAP", a] - saving[a]:,.1f}M')
    return E


# =============================================================================
def compare(name, new, ref_path, tol):
    """Compare a product of this run with the published product in product/."""
    if not ref_path.exists():
        log(f'check {name}: no reference file {ref_path.name}')
        return
    ref = pd.read_csv(ref_path, index_col=0).fillna(0)
    new = new.fillna(0)
    d = (new - ref.reindex(index=new.index, columns=new.columns).fillna(0)).abs().values.max()
    log(f'check {name}: max |difference| vs product/{ref_path.name} = {d:.6f} $M -> {"OK" if d <= tol else "DIFFERS"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='compare the outputs with product/')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    log(f'SAMNJ24 pipeline, run {time.strftime("%Y-%m-%d %H:%M:%S")}')
    I = institutional_inputs()
    I.to_csv(OUT / 'institutional_inputs_2024.csv')
    A, UT, UD = stage_a()
    M, parts = stage_b(A, UT, UD, I)
    M.to_csv(OUT / 'samnj24_65_millions.csv')
    T, extra = stage_c(M, parts, I)
    T.to_csv(OUT / 'samnj24_table7_19acc_millions.csv')
    E = stage_e(M, T, parts, I)
    E.to_csv(OUT / 'samnj24_75_millions.csv')
    tab = stage_d(M, T, parts, I)
    tab.to_csv(OUT / 'samnj24_table8_control_totals.csv', index=False)
    json.dump({'net_taxes_production': float(parts['ntp'].sum() / 1e3), 'net_taxes_commodity': float(parts['ntc'].sum() / 1e3),
               'census_production_taxes_state': extra['prd_s'], 'census_production_taxes_local': extra['prd_l'],
               'state_share_of_SL_consumption': extra['shares'], 'capital_outlay_ratio': extra['co_ratio'],
               'core_total_flows_musd': float(M.values.sum()), 'table7_total_flows_musd': float(T.values.sum())},
              open(OUT / 'samnj24_parameters.json', 'w'), indent=1)
    if a.check:
        compare('65-account SAM', M, PROD / 'samnj24_65_millions.csv', 0.001)
        compare('Table 7', T, PROD / 'samnj24_table7_19acc_millions.csv', 0.001)
        compare('75-account SAM', E, PROD / 'samnj24_75_millions.csv', 0.001)
    log(f'done in {time.time() - t0:.0f} s')
    (OUT / 'run_log.txt').write_text('\n'.join(LOG) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
