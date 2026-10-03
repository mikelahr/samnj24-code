"""RECON2024: domestic demand in the rpc stage (Mike, 2026-09-21).

Terms: imports/exports are international; inflows/outflows are between
states (or counties) within the nation.

The RPC regression is fitted to FAF flows as rpc = m / (m + inflows): the share
of demand met from U.S. sources (local supply + inflows) that is met locally, and the RPCs are applied to state A
matrices from which international imports have already been removed. The rpc
stage therefore uses DOMESTIC demand (demand net of international imports,
from sut step export_domestic_demand) instead of supply + international
imports:

  supply        = output - international exports        (unchanged)
  demand        = domestic demand D_r(c)
  supplydemand  = supply / D_r(c)                       (services rpc)
  l_demand      = log D_r(c)                            (regressor)

rpc/operations.calculate_rpc (other session's file) is not edited: it forms
demand = supply + imports, so this wrapper passes imports := D - supply for
the call and restores the international imports afterwards. Switched on by
`calculate_rpc: demand: domestic` in rpc/pipeline.yaml; installed by main.py.
"""
import logging
import pandas as pd
from . import utilities

logger = logging.getLogger(__name__)
_installed = False


def load_domestic_demand():
  cfg = utilities.get_config('sut')
  c = cfg.export_domestic_demand
  path = cfg.dirs.__dict__[c.target.folder] / c.target.file
  d = pd.read_csv(path, dtype={'region': str, 'industry': str})
  d.index = d.region + '_' + d.industry
  return d['demand_domestic'].astype(float)


def install():
  global _installed
  if _installed:
    return
  from .rpc import operations as rpc_ops
  cfg = utilities.get_config('rpc').calculate_rpc
  if str(getattr(cfg, 'demand', 'total')) != 'domestic':
    logger.info("rpc demand = supply + international imports (legacy)")
    return
  original = rpc_ops.calculate_rpc

  def calculate_rpc_domestic(vectors, drop=True):
    dd = load_domestic_demand().reindex(vectors.index)
    n_miss = int(dd.isna().sum())
    supply = (vectors.output - vectors.exports).astype(float)
    dd = dd.fillna(supply + vectors.imports.astype(float))   # legacy where missing
    imports_int = vectors['imports'].copy()
    vectors['imports'] = dd - supply
    logger.info(f"rpc stage on DOMESTIC demand ({len(dd) - n_miss} cells; "
                f"{n_miss} fall back to supply + imports)")
    out = original(vectors, drop=drop)
    out['imports'] = imports_int.reindex(out.index).values
    ov = getattr(cfg, 'rpc_override', None)
    if ov is not None:
      # final RPCs estimated outside the stage (other chat, 2026-09-21:
      # process/RPC/predict_2024 - culled FAF6 refit, raked to FAF6 2024
      # state and zone anchors); the stage's own prediction is kept as rpc_stage
      path = utilities.get_config('rpc').dirs.__dict__[ov.folder] / ov.file
      r = pd.read_csv(path, dtype={'region': str, 'industry': str}, usecols=['region', 'industry', ov.column])
      r.index = r.region + '_' + r.industry
      new = r[ov.column].reindex(out.index)
      n_miss = int(new.isna().sum())
      out['rpc'] = new.fillna(out['rpc']).clip(lower=0.0, upper=1.0).values
      logger.info(f"rpc replaced from {ov.file} ({ov.column}); {n_miss} cells kept the stage prediction")
    return out

  rpc_ops.calculate_rpc = calculate_rpc_domestic
  _installed = True
  logger.info("Installed domestic-demand wrapper around rpc.operations.calculate_rpc")


# ---------------------------------------------------------------------------
# RECON2024 item 4 (Mike, 2026-09-21): government split of taxes re-derived
# from 2024 Census State & Local Government Finances
# ---------------------------------------------------------------------------
_taxes_installed = False


def _state_of(region):
  r = str(region).zfill(5)
  return r if r == '00000' else r[:2] + '000'


def install_taxes():
  """Wrap rpc.operations.calculate_taxes (other session's file, not edited).

  Production taxes (other taxes on production less subsidies, BEA T00OTOP -
  T00OSUB): the legacy split gave the federal government the region's share of
  federal business income tax (US: ~$400bn of $850bn), but federal other taxes
  on production are negligible - these are property taxes, licenses and other
  S&L taxes. Now: positive rates -> `production.federal_share` federal, the
  rest state vs local by the state's Census shares of `production.lines`
  (property, motor vehicle licenses, other taxes); negative rates (subsidies)
  stay federal.
  Product taxes (taxes on products and imports less subsidies): the legacy
  industry rules are kept for the pattern, then calibrated to 2024 national
  totals - federal = total - Census S&L sales and gross receipts (lines of
  `products.sl_lines`), state vs local in the Census proportion - by scaling
  the federal part and the state odds of each cell."""
  global _taxes_installed
  if _taxes_installed:
    return
  from .rpc import operations as rpc_ops
  full = utilities.get_config('rpc')
  cfg = getattr(full.calculate_taxes, 'recalibrate', None)
  if cfg is None:
    logger.info("tax split: legacy rules (no calculate_taxes.recalibrate)")
    return
  original = rpc_ops.calculate_taxes

  def calculate_taxes_2024(vectors, drop=True):
    import numpy as np
    from scipy.optimize import brentq
    out = original(vectors, drop=False)
    rev = rpc_ops.load_state_local_revenue()
    rev = rev.assign(value=rev['value'].astype(float), line_code=rev.line_code.astype(str),
                     region_code=rev.region_code.astype(str).str.zfill(5))
    piv = rev.pivot_table(index=['region_code', 'line_code'], columns='government', values='value', aggfunc='sum')
    st_of = out.region.map(_state_of)
    x = out.output.astype(float)
    gov = ['federal', 'state', 'local']

    # production taxes
    pl = [str(l) for l in cfg.production.lines]
    ps = piv.loc[piv.index.get_level_values(1).isin(pl)].groupby(level=0).sum()
    th_p = (ps['state'] / (ps['state'] + ps['local'])).reindex(st_of.values).fillna(0.5).values
    p = out['nettax_production_output'].astype(float).values
    f = float(cfg.production.federal_share)
    pos = p > 0
    out['nettax_production_output_federal'] = np.where(pos, f * p, p)
    out['nettax_production_output_state'] = np.where(pos, (1 - f) * p * th_p, 0.0)
    out['nettax_production_output_local'] = np.where(pos, (1 - f) * p * (1 - th_p), 0.0)

    # product taxes
    c = out['nettax_commodity_output'].astype(float).values
    fed = out['nettax_commodity_output_federal'].astype(float).values
    st = out['nettax_commodity_output_state'].astype(float).values
    lo = out['nettax_commodity_output_local'].astype(float).values
    sl = [str(l) for l in cfg.products.sl_lines]
    ss = piv.loc[piv.index.get_level_values(1).isin(sl)].groupby(level=0).sum()
    us_sl = ss.loc['00000']
    usm = (out.region.astype(str).str.zfill(5) == '00000').values
    tot = (c * x)[usm].sum()
    fed_t = tot - (us_sl['state'] + us_sl['local'])
    st_t = (tot - fed_t) * us_sl['state'] / (us_sl['state'] + us_sl['local'])
    k = fed_t / (fed * x)[usm].sum()
    fed2 = fed * k
    rest = c - fed2
    gen = [str(l) for l in cfg.products.default_line]
    g = piv.loc[piv.index.get_level_values(1).isin(gen)].groupby(level=0).sum()
    th_g = (g['state'] / (g['state'] + g['local'])).reindex(st_of.values).fillna(0.5).values
    sl0 = st + lo
    th = np.where(np.abs(sl0) > 1e-12, st / np.where(np.abs(sl0) > 1e-12, sl0, 1.0), th_g)
    th = np.clip(th, 0.0, 1.0)

    def st_total(a):
      odds = a * th / np.maximum(1 - th, 1e-9)
      t2 = np.where(th >= 1.0, 1.0, odds / (1 + odds))
      return (rest * t2 * x)[usm].sum() - st_t
    a = brentq(st_total, 1e-3, 1e3)
    odds = a * th / np.maximum(1 - th, 1e-9)
    t2 = np.where(th >= 1.0, 1.0, odds / (1 + odds))
    out['nettax_commodity_output_federal'] = fed2
    out['nettax_commodity_output_state'] = rest * t2
    out['nettax_commodity_output_local'] = rest * (1 - t2)
    logger.info(f"tax split recalibrated to 2024 Census: products federal x{k:.3f} "
                f"(${fed_t / 1e6:,.1f}bn), state odds x{a:.3f} (state ${st_t / 1e6:,.1f}bn, "
                f"local ${(tot - fed_t - st_t) / 1e6:,.1f}bn); production federal share {f:.2f}, "
                f"state/local by Census lines {pl}")
    for gv in gov:
      out[f'nettax_output_{gv}'] = (out[f'nettax_production_output_{gv}']
                                    + out[f'nettax_commodity_output_{gv}'])
      us_t = (out[f'nettax_output_{gv}'].astype(float) * x)[usm].sum()
      logger.info(f"  US {gv}: ${us_t / 1e6:,.1f}bn")
    if drop:
      for gv in ['_federal', '_state', '_local', '']:
        for type_ in ['production', 'commodity']:
          out.drop(f"nettax_{type_}_output{gv}", axis=1, inplace=True)
    return out

  rpc_ops.calculate_taxes = calculate_taxes_2024
  _taxes_installed = True
  logger.info("Installed 2024 Census tax-split wrapper around rpc.operations.calculate_taxes")
