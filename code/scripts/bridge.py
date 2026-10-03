"""HS6 -> BEA detail commodity bridge for Census merchandise trade.

HS10 -> NAICS from the Census import concordance (CONCCOMM, Dec 2024), collapsed to
HS6 with HS10-line counts as weights; NAICS -> BEA IO code via NAICS_BEA_conversion_17
(prefix match), with the 2017->2022 NAICS reclassifications used in RECON2022's QCEW
stage.  Special-classification NAICS (91/93/98/99) have no BEA commodity: kept as
'UNALLOC' so totals can be reconciled explicitly.
"""
import os, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
RECLASS = {'333998': '33399A', '333310': '333318', '333248': '33329A', '335910': '335911',
           '335139': '335110', '335131': '335120', '335132': '335120', '337126': '33712N',
           '336110': None, '1121XX': '1121A0'}   # 336110 split 336111/336112 by BEA output below
SPECIAL = {'910000': 'S00401', '930000': 'S00402', '980000': 'S00300', '990000': 'S00300'}   # scrap, used/secondhand+art, special classification -> noncomparable

def load():
    hs = pd.read_csv(os.path.join(ROOT, 'raw', 'Trade', 'hs6_naics_imports_2024.csv'), header=None,
                     names=['hs6', 'naics', 'n'], dtype={'hs6': str, 'naics': str})
    hs['hs6'] = hs.hs6.str.zfill(6)
    xw = pd.read_csv(os.path.join(ROOT, 'NAICS_BEA_conversion_17.csv'), header=None,
                     names=['bea', 'naics', 'a', 'b'], dtype=str)
    xmap = dict(zip(xw.naics, xw.bea))
    def bea(code):
        if code in SPECIAL: return SPECIAL[code]
        if code in RECLASS:
            r = RECLASS[code]
            if r is None: return '336111|336112'
            if r in xmap.values(): return r
            code = r
        c = code.replace('X', '')
        for k in range(len(c), 2, -1):
            if c[:k] in xmap: return xmap[c[:k]]
        return 'UNALLOC'
    hs['bea'] = hs.naics.map(bea)
    rows = []
    for _, r in hs.iterrows():
        parts = r.bea.split('|')
        for p in parts: rows.append((r.hs6, p, r.n / len(parts)))
    b = pd.DataFrame(rows, columns=['hs6', 'bea', 'w'])
    b['w'] = b.w / b.groupby('hs6').w.transform('sum')
    return b   # hs6 x bea share weights (sum to 1 per hs6)

def bridge(series_by_hs6, b=None):
    """series indexed by hs6 (values $) -> Series by BEA commodity."""
    b = load() if b is None else b
    s = series_by_hs6.rename('v').reset_index().rename(columns={series_by_hs6.index.name or 'index': 'hs6'})
    s['hs6'] = s.hs6.astype(str).str.zfill(6)
    m = s.merge(b, on='hs6', how='left')
    m['bea'] = m.bea.fillna('UNMATCHED'); m['w'] = m.w.fillna(1.0)
    return (m.v * m.w).groupby(m.bea).sum()

if __name__ == '__main__':
    b = load()
    print(len(b), 'hs6-bea rows;', b.hs6.nunique(), 'hs6;', b.bea.nunique(), 'bea codes; UNALLOC hs6:', b[b.bea == 'UNALLOC'].hs6.nunique())
