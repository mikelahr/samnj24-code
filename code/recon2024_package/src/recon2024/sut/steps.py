"""Steps of SUT pipeline."""
from .. import utilities
from . import operations
from recon2024.bea import operations as bea_operations
from recon2024 import __version__
import pandas as pd
import numpy as np
from logging import getLogger
from copy import deepcopy
import time
import pdb
import os
import requests
logger = getLogger('root')


def create_symmetric(task_str, full_config):
  logger.info("Creating industry-by-industry symmetric system")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  mar_conf = full_config.apply_margins.target

  logger.info("Loading data")
  pce_stat, gdp_stat = operations.load_bea_statistics()

  dir_ = dirs.__dict__[mar_conf.folder]
  sut = operations.Sut.read_excel(
    dir_ / mar_conf.file, 'target_det')

  logger.info("Reallocating SUT")

  logger.info("Moving imports and net taxes on products to negative "
              "final demand")
  sut.data.use_fin[sut.codes.trade.index] = - sut.data.sup_tra
  sut.data.sup_tra = 0 * sut.data.sup_tra
  sut.codes.final = pd.concat(
    [sut.codes.final, sut.codes.trade], axis=0)

  logger.info("Moving noncomparable imports to primary inputs")
  sut.codes.primary.loc['noncomparable'] = sut.codes.commodities.loc[
    config.noncomparable, sut.codes.primary.columns]
  sut.codes.commodities.drop(config.noncomparable,
                             axis=0, inplace=True)

  setattr(sut.data, 'pri_fin', pd.DataFrame(
    index=sut.codes.primary.index, columns=sut.codes.final.index,
    data=float(0)))
  sut.data.pri_fin.loc['noncomparable'] = sut.data.use_fin.loc[
    config.noncomparable]
  sut.data.use_pri.loc['noncomparable'] = sut.data.use_int.loc[
    config.noncomparable]
  sut.data.use_int.drop(config.noncomparable, axis=0, inplace=True)
  sut.data.sup_int.drop(config.noncomparable, axis=0, inplace=True)
  sut.data.use_fin.drop(config.noncomparable, axis=0, inplace=True)

  setattr(sut.config.data, 'pri_fin', sut.config.data.use_fin)
  sut.config.data.pri_fin.index = 'primary'
  sut.config.data.pri_fin.index = 'primary_final'
  sut.calculate_sums(endogenous=True)

  logger.info("Adding nettax_extra to industries")
  sut.codes.industries[
    'extra_bea'] = gdp_stat.nettax.det.loc[
      sut.codes.industries.index, config.nettax_extra]

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  sut.write_excel(dir_ / config.target.files.sut)

  logger.info("Create symmetric IOT")

  iot = deepcopy(sut)
  setattr(iot.data, 'D', iot.data.sup_int.multiply(
    utilities.safe_inverse(iot.codes.commodities['sup']), axis=0))
  setattr(iot.config.data, 'D', iot.config.data.sup_int)
  iot.config.data.D.label = 'technical_supply'

  logger.info("Sum of commodity-to-industry matrix rows and columns")

  tmp = iot.data.D.sum(axis=1)
  tmp.sort_values(inplace=True)
  logger.info(f"Commodities (should be  == 1 if not '4200ID'):\n{tmp}")

  tmp = iot.data.D.sum(axis=0)
  tmp.sort_values(inplace=True)
  logger.info(f"Industries (should be != 0 if not '4200ID'):\n{tmp}")

  # transforming uses and removing supply
  for table in ['use_int', 'use_fin']:
    iot.data.__dict__[table] = iot.data.D.transpose().dot(
      iot.data.__dict__[table])
    iot.config.data.__dict__[table].index = 'industries'
  for table in ['sup_int', 'sup_tra']:
    delattr(iot.data, table)
    delattr(iot.config.data, table)
  delattr(iot.config.codes, 'commodities')
  delattr(iot.config.codes, 'trade')

  # moving nettax_extra around
  iot.codes.industries[
    'extra_sut'] = - iot.data.use_fin['nettax_extra']
  iot.codes.industries['tot'] = (
    iot.data.use_int.sum(1) + iot.data.use_fin.sum(1))

  iot.codes.primary.loc[
    'nettax_extra'] = iot.codes.final.loc[
      'nettax_extra']
  iot.codes.final.drop('nettax_extra', axis=0, inplace=True)

  iot.data.use_pri.loc['nettax_extra'] = (
    iot.codes.industries.tot
    - (iot.data.use_int.sum(0) + iot.data.use_pri.sum(0)))
  iot.data.pri_fin.loc['nettax_extra'] = 0

  iot.data.use_fin.drop('nettax_extra', axis=1, inplace=True)
  iot.data.pri_fin.drop('nettax_extra', axis=1, inplace=True)

  # creating domestication objects
  iot.codes.industries['R'] = utilities.safe_division(
    iot.codes.industries.tot - iot.data.use_fin.exports,
    iot.codes.industries.tot - iot.data.use_fin[[
      'exports', 'imports']].sum(1))
  iot.codes.industries = iot.codes.industries[
    ['tot', 'extra_bea', 'extra_sut', 'R', 'title']]

  setattr(iot.data, 'A', iot.data.use_int.multiply(
    utilities.safe_inverse(iot.codes.industries['tot']), axis=1))
  setattr(iot.config.data, 'A', iot.config.data.use_int)
  iot.config.data.A.label = 'technical_use'

  # setattr(iot.data, 'A', iot.data.B.multiply(
  #   iot.codes.industries['R'], axis=0))
  # setattr(iot.config.data, 'A', iot.config.data.B)
  # iot.config.data.A.label = 'technical_domestic'

  for key in iot.data.__dict__.keys():
    if key in ['D', 'A']:
      decimals = config.decimals
      type_ = float
    else:
      decimals = 0
      type_ = int
    iot.data.__dict__[key] = iot.data.__dict__[key].round(
      decimals).astype(type_)

  for key in iot.codes.industries.columns:
    if key == 'title':
      continue

    if key == 'R':
      decimals = config.decimals
      type_ = float
    else:
      decimals = 0
      type_ = int
    iot.codes.industries[key] = iot.codes.industries[key].round(
      decimals).astype(type_)

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  iot.write_excel(dir_ / config.target.files.iot)

  logger.info("Finished step")
  return None


def apply_margins(task_str, full_config):
  logger.info("Applying trade and transport margins to SUT")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  target_config = full_config.create_target.target

  logger.info("Loading data")
  margins = operations.load_margins()

  dir_ = dirs.__dict__[target_config.folder]
  sut = operations.Sut.read_excel(
    dir_ / target_config.bal_det, 'target_det')

  logger.info("Applying margins")
  sut = operations.apply_margins(sut, margins)

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  sut.write_excel(dir_ / config.target.file)

  @utilities.open_excel(dir_ / config.target.file)
  def save_bridge(writer, table, key):
    table.to_excel(writer, sheet_name='margin_' + key, index=True)

  for key, table in margins.__dict__.items():
    save_bridge('_', table, key)

  logger.info("Finished step")
  return None


def process_margins(task_str, full_config):
  logger.info("Loading trade and transport margins")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  struct_config = full_config.sut_structure
  sut_config = full_config.process_suts

  # RECON2024: the archived source called a read_sut() that no longer exists;
  # the margin template is extracted from the detailed source SUT container.
  sut = operations.Sut.read_excel(
    dirs.__dict__[sut_config.folder] / sut_config.suts.source_det.file,
    'source_det')
  data, codes = sut.data, sut.codes

  margins = operations.extract_margins(data, codes, config)

  margins.raw_transport = operations.load_raw_transport(config, dirs)

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  with pd.ExcelWriter(dir_ / config.target.file,
                      engine='xlsxwriter') as writer:
    for key0, val in config.target.sheets.__dict__.items():
      for key1, sheet_name in val.__dict__.items():
        tmp = margins.__dict__[key0].__dict__[key1]
        tmp.to_excel(writer, sheet_name=sheet_name, index=True)

  logger.info("Finished step")
  return None


def create_target(task_str, full_config):
  logger.info("Creating 2022 detailed SUT")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  suts = operations.Suts()

  logger.info("Creating initial target with BEA va and pce")
  suts.create_initial_target(keep_intermediates=False)

  """
  logger.info("Performing adhoc fix of OTHER = S00300 + S00900")
  for code in config.adhoc_fix_other:
    diff = (
      suts.target_det.codes.commodities.loc[code, 'use']
      - suts.target_det.codes.commodities.loc[code, 'sup'])
    suts.target_det.data.use_fin.loc[code, 'pce'] = (
      suts.target_det.data.use_fin.loc[code, 'pce']
      - diff).astype(int)
  suts.target_det.calculate_sums(endogenous=True)
  """

  logger.info("Exporting initial target")
  suts.aggregate_sut('target_det', 'target_agg_alt')
  sut = suts.target_agg._subtract(suts.target_agg_alt,
                                  name='target_agg_diff')
  suts.add_sut(sut, sut.__name__)

  dir_ = dirs.__dict__[config.target.folder]
  suts.target_det.write_excel(dir_ / config.target.init_det)
  suts.target_agg_alt.write_excel(dir_ / config.target.init_agg)
  suts.target_agg_diff.write_excel(dir_ / config.target.init_diff)

  logger.info("Converting from SUT to balancing structure")
  bal = operations.Balance(suts,
                           aggregate='target_agg',
                           detailed='target_det')

  logger.info("Perform balancing")
  bal.execute()

  logger.info("Converting result to SUT")
  suts = bal.export(suts,
                    source='target_det',
                    target='target_det')

  logger.info("Exporting balanced results")
  suts.aggregate_sut('target_det', 'target_agg_alt')
  sut = suts.target_agg._subtract(suts.target_agg_alt,
                                  name='target_agg_diff')
  suts.add_sut(sut, sut.__name__)

  dir_ = dirs.__dict__[config.target.folder]
  suts.target_det.write_excel(dir_ / config.target.bal_det)
  suts.target_agg_alt.write_excel(dir_ / config.target.bal_agg)
  suts.target_agg_diff.write_excel(dir_ / config.target.bal_diff)

  logger.info("Finished step")

  return None


def create_bridge(task_str, full_config):
  logger.info("Create bridge from ~400 to ~70 sectors")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  suts = operations.Suts(load_bridge=False)

  for code_key in config.codes:
    logger.info(f"Classification {code_key}")

    results = operations.create_aggregation_bridge_codes(
      suts.__dict__[config.levels.parent].codes.__dict__[code_key],
      suts.__dict__[config.levels.offspring].codes.__dict__[code_key],
      config.detailed_aggregate_pairs)

    dir_ = dirs.__dict__[config.target.folder]

    @utilities.open_excel(dir_ / config.target.file)
    def save_bridge(writer, results, code_key):
      results.to_excel(writer, sheet_name=code_key, index=False)

    save_bridge('_', results, code_key)

  # check that all sectors are mapped
  # compare reference year detailed vs aggregate values
  # operations.check_aggregation(
  #  config, dirs, source_config, struct_config)

  logger.info("Finished step")

  return None



def extract_summary_year(task_str, full_config):
  """RECON2024: write Use_{year}.xlsx / Supply_{year}.xlsx (single sheet
  'Table', the layout load_aggregate expects) from the multi-year
  Use_Summary.xlsx / Supply_Summary.xlsx in raw/SUT_71sectors. The summary
  workbooks mark empty cells '...' where the single-year files used '---'."""
  import openpyxl
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year = str(full_config.year)
  for key, val in config.files.__dict__.items():
    src = dirs.__dict__[config.source_folder] / val.source
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ws = wb[year]
    out = openpyxl.Workbook(); wo = out.active; wo.title = config.sheet
    n = 0
    for row in ws.iter_rows(values_only=True):
      wo.append([('---' if (isinstance(v, str) and v.strip() == '...') else v) for v in row])
      n += 1
    target = dirs.__dict__[config.target_folder] / val.target
    out.save(target)
    logger.info(f"{key}: sheet {year} of {val.source} -> {target} ({n} rows)")
  logger.info("Finished step")
  return None


def process_suts(task_str, full_config):
  logger.info("Loading multiple SUTs")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  struct_config = full_config.sut_structure

  for key, val in config.suts.__dict__.items():
    logger.info(f"Loading SUT {key}: {val.config}, {val.info}")
    aux_config = full_config.__dict__[val.config]
    if key in config.replace.__dict__.keys():
      logger.info("Replacing configuration info")
      for replace_tmp in config.replace.__dict__[key]:
        tmp = [deepcopy(aux_config)]
        for str_ in replace_tmp.path:
          tmp.append(getattr(tmp[-1], str_))
        tmp[-1] = replace_tmp.value
        for pos, str_ in enumerate(reversed(replace_tmp.path)):
          setattr(tmp[-pos-2], str_, tmp[-pos-1])
        aux_config = deepcopy(tmp[0])

    if getattr(aux_config, 'reader', 'excel') == 'construction':   # RECON2024
      data, codes = operations.extract_construction(
        aux_config, struct_config, dirs)
    else:
      data, codes = operations.extract_excel(
        aux_config, struct_config, dirs)

    setattr(config, 'bridge',
            operations.load_external_codes_bridge(
              config.codes_bridges, val.config, full_config.stage,
              codes))

    data, codes = operations.process_sut(
      data, codes, config, struct_config)

    sut = operations.Sut(data, codes, struct_config,
                         endogenous_sums=True)

    path = dirs.__dict__[config.folder] / val.file
    sut.write_excel(path)

  logger.info("Finished step")

  return None


def process_bea(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Processing national-level BEA PCE and GDP")

  logger.info("Loading data")
  suts = operations.Suts()
  bea = bea_operations.load_bridged()

  extra_tax = {}
  for key, val in config.source.sheets.__dict__.items():
    tmp_ = pd.read_excel(
      utilities.INTERNAL_PATH / full_config.stage / config.source.file,
      sheet_name=val,
      index_col='code',
      dtype={'code': str})
    tmp_.fillna(0, inplace=True)
    for col in config.source.columns.__dict__.values():
      tmp_[col] = tmp_[col].astype(int)
    extra_tax[key] = (tmp_[config.source.columns.tax]
                      - tmp_[config.source.columns.subsidy])
  extra_tax = utilities.DictToObject(extra_tax)

  # PCE
  logger.info("Processing personal consumption expenditure")
  pce = {'agg': pd.DataFrame(
    index=suts.target_agg.codes.commodities.index),
    'det': pd.DataFrame(index=suts.source_det.codes.commodities.index)}
  pce['det']['source'] = deepcopy(suts.source_det.data.use_fin.pce)
  pce['agg']['target'] = deepcopy(suts.target_agg.data.use_fin.pce)
  pce = utilities.DictToObject(pce)

  logger.info("Converting from thousand to million dollars")
  tmp0_ = bea.pce.loc[bea.pce.region == '00000']
  tmp0_.set_index('commodity', inplace=True, drop=True)
  # RECON2024: the PCE file is on the 2017 commodity axis (12 BEA construction
  # types); the SUT is on the 421 axis (31 NAICS). Construction PCE is ~0.
  off_axis = tmp0_.loc[~tmp0_.index.isin(pce.det.index)]
  if len(off_axis) > 0:
    logger.info(f"PCE commodities not in the SUT axis (dropped, ${off_axis['pce'].sum() / 1000:,.0f}M): "
                f"{off_axis.index.tolist()}")
    tmp0_ = tmp0_.loc[tmp0_.index.isin(pce.det.index)]
  pce.det['raw'] = 0
  pce.det.loc[tmp0_.index, 'raw'] = (
    tmp0_['pce'] / 1000).round(0).astype(int)
  del tmp_, tmp0_, key, val, col

  pce.agg['raw'] = suts.bridge.commodities.matrix.dot(pce.det.raw)

  logger.info("Resetting PCE religious, grant and civic services")
  reference_sum = 0
  edit_sum = 0
  for index in config.pce_other_service.control:
    reference_sum = reference_sum + pce.det.at[index, 'source']
    edit_sum = edit_sum + pce.det.at[index, 'raw']

  pce.det['initial'] = deepcopy(pce.det.raw)
  for index in config.pce_other_service.reset:
    pce.det.at[index, 'initial'] = int(pce.det.at[
      index, 'source'] * edit_sum / reference_sum)

  logger.info("Hardcoding OTHER sectors constrained by aggregate SUT")
  for key, val in config.pce_adhoc.__dict__.items():
    pce.det.at[key, 'initial'] = val

  pce.agg = operations.calc_agg_errors(
    agg=pce.agg,
    det=pce.det,
    bridge=suts.bridge.commodities.matrix,
    source='target',
    target='initial')

  logger.info("Balancing personal consumption expenditure")

  ratio = utilities.safe_division(pce.agg.target, pce.agg.initial)
  ratio = ratio + 1 * (ratio == 0)
  ratio = suts.bridge.commodities.matrix.T.dot(ratio)
  pce.det['final'] = (pce.det.initial * ratio).astype(int)

  pce.agg = operations.calc_agg_errors(
    agg=pce.agg,
    det=pce.det,
    bridge=suts.bridge.commodities.matrix,
    source='target',
    target='final')

  # GDP
  logger.info("Processing value added components")
  gdp = {'wages': {}, 'nettax': {}, 'surplus': {}}
  for key in gdp.keys():
    gdp[key] = {'agg': pd.DataFrame(
      index=suts.target_agg.codes.industries.index),
      'det': pd.DataFrame(
        index=suts.source_det.codes.industries.index)}
    gdp[key]['det']['source_vasut'] = deepcopy(
      suts.source_det.data.use_pri.loc[key])
    gdp[key]['agg']['target_vasut'] = deepcopy(
      suts.target_agg.data.use_pri.loc[key])
  gdp = utilities.DictToObject(gdp)

  logger.info("Adding extra taxes to SUT versions")
  for scale_key, scale_val in {'agg': 'target',
                               'det': 'source'}.items():
    for gdp_key, gdp_val in gdp.__dict__.items():
      gdp.__dict__[gdp_key].__dict__[scale_key][
        scale_val + '_extra'] = 0
      gdp.__dict__[gdp_key].__dict__[scale_key][
        scale_val + '_total'] = (
        gdp.__dict__[gdp_key].__dict__[scale_key][scale_val + '_vasut'])

    gdp.nettax.__dict__[scale_key][scale_val + '_extra'] = deepcopy(
      extra_tax.__dict__[scale_key])
    gdp.nettax.__dict__[scale_key][scale_val + '_extra'] = (
      gdp.nettax.__dict__[scale_key][scale_val + '_extra'].fillna(0))

    gdp.nettax.__dict__[scale_key][scale_val + '_total'] = deepcopy(
      gdp.nettax.__dict__[scale_key][scale_val + '_vasut']
      + gdp.nettax.__dict__[scale_key][scale_val + '_extra'])
    gdp.nettax.__dict__[scale_key][scale_val + '_total'] = (
      gdp.nettax.__dict__[scale_key][scale_val + '_total'].fillna(0))

  logger.info("Converting from thousand to million dollars")
  tmp0_ = bea.bea.loc[bea.bea.region == '00000']
  tmp0_.set_index('industry', inplace=True, drop=True)
  for gdp_key in gdp.__dict__.keys():
    gdp.__dict__[gdp_key].det['raw'] = 0
    gdp.__dict__[gdp_key].det.loc[tmp0_.index, 'raw'] = (
      tmp0_[gdp_key] / 1000).round(0).astype(int)

    gdp.__dict__[gdp_key].agg['raw'] = (
      suts.bridge.industries.matrix.dot(
        gdp.__dict__[gdp_key].det.raw))

  logger.info("Adding overseas expenditure to BEA versions")
  for gdp_key in gdp.__dict__.keys():
    gdp.__dict__[gdp_key].det['overseas'] = (
      gdp.__dict__[gdp_key].det['raw'])

  for line in config.overseas:
    gdp.__dict__[line.primary].det.loc[line.industry, 'overseas'] = (
      gdp.__dict__[line.primary].det.loc[line.industry, 'overseas']
      + line.value)

  for gdp_key in gdp.__dict__.keys():
    gdp.__dict__[gdp_key].agg['overseas'] = (
      suts.bridge.industries.matrix.dot(
        gdp.__dict__[gdp_key].det.overseas))

  logger.info("Reassigning housing")
  for gdp_key in gdp.__dict__.keys():
    gdp.__dict__[gdp_key].det['alloc'] = deepcopy(
      gdp.__dict__[gdp_key].det['overseas'])

  for line in config.housing:
    diff = (gdp.__dict__[line.va].agg.loc[line.agg, 'overseas']
            - gdp.__dict__[line.va].agg.loc[line.agg, 'target_total'])
    gdp.__dict__[line.va].det.loc[line.source, 'alloc'] = (
      gdp.__dict__[line.va].det.loc[line.source, 'alloc'] - diff)
    gdp.__dict__[line.va].det.loc[line.target, 'alloc'] = (
      gdp.__dict__[line.va].det.loc[line.target, 'alloc'] + diff)

  logger.info("Reassigning other services")
  for va in config.gdp_other_service.va:
    diff = gdp.__dict__[va].det.loc[config.gdp_other_service.remove,
                                    'alloc']
    gdp.__dict__[va].det.loc[config.gdp_other_service.remove,
                             'alloc'] = 0

    tot = 0
    for index in config.gdp_other_service.distribute:
      tot = tot + gdp.__dict__[va].det.loc[index, 'alloc']

    for index in config.gdp_other_service.distribute:
      gdp.__dict__[va].det.loc[index, 'alloc'] = ((
        1 + diff / tot) * gdp.__dict__[va].det.loc[index, 'alloc']
      ).astype(int)

  logger.info("Reassigning zero value added entries")
  for va in config.va_zeros:
    for gdp_key in gdp.__dict__.keys():
      diff = (
        gdp.__dict__[gdp_key].det.loc[va.source, 'overseas']
        * va.fraction)  # overseas and NOT alloc to stay constant

      gdp.__dict__[gdp_key].det.loc[va.target, 'alloc'] = (
        gdp.__dict__[gdp_key].det.loc[va.target, 'alloc']
        + diff).astype(int)

      gdp.__dict__[gdp_key].det.loc[va.source, 'alloc'] = (
        gdp.__dict__[gdp_key].det.loc[va.source, 'alloc']
        - diff).astype(int)

  logger.info("Reassigning trade")
  for gdp_key in gdp.__dict__.keys():
    bulk_diff = 0
    for individual in config.trade.individual:
      individual_diff = (
        gdp.__dict__[gdp_key].det.loc[individual.detailed, 'alloc']
        - gdp.__dict__[gdp_key].agg.loc[individual.aggregate,
                                        'target_total'])
      gdp.__dict__[gdp_key].det.loc[individual.detailed, 'alloc'] = (
        gdp.__dict__[gdp_key].det.loc[individual.detailed, 'alloc']
        - individual_diff)
      bulk_diff = bulk_diff + individual_diff

    bulk_sum = 0
    for key in config.trade.bulk:
      bulk_sum = bulk_sum + gdp.__dict__[gdp_key].det.loc[key, 'alloc']

    for key in config.trade.bulk:
      gdp.__dict__[gdp_key].det.loc[key, 'alloc'] = (
        gdp.__dict__[gdp_key].det.loc[key, 'alloc']
        * (1 + bulk_diff / bulk_sum)).astype(int)

  logger.info("Reassigning government")
  for line in config.government_aggregate:
    diff = (gdp.__dict__[line.va].agg.loc[line.agg, 'overseas']
            - gdp.__dict__[line.va].agg.loc[line.agg, 'target_total'])
    gdp.__dict__[line.va].det.loc[line.source, 'alloc'] = (
      gdp.__dict__[line.va].det.loc[line.source, 'alloc'] - diff)
    gdp.__dict__[line.va].det.loc[line.target, 'alloc'] = (
      gdp.__dict__[line.va].det.loc[line.target, 'alloc'] + diff)

  for line in config.government_detailed:
    diff = (gdp.__dict__[line.va].det.loc[line.source, 'alloc'])
    gdp.__dict__[line.va].det.loc[line.source, 'alloc'] = (
      gdp.__dict__[line.va].det.loc[line.source, 'alloc'] - diff)
    gdp.__dict__[line.va].det.loc[line.target, 'alloc'] = (
      gdp.__dict__[line.va].det.loc[line.target, 'alloc'] + diff)

  for gdp_key in gdp.__dict__.keys():
    gdp.__dict__[gdp_key].agg = operations.calc_agg_errors(
      agg=gdp.__dict__[gdp_key].agg,
      det=gdp.__dict__[gdp_key].det,
      bridge=suts.bridge.industries.matrix,
      source='target_total',
      target='alloc')

  logger.info("Balancing value added components")
  for gdp_key in gdp.__dict__.keys():
    ratio = utilities.safe_division(
      gdp.__dict__[gdp_key].agg['target_total'],
      gdp.__dict__[gdp_key].agg['alloc'])
    ratio = ratio + 1 * (ratio == 0)
    ratio = suts.bridge.industries.matrix.T.dot(ratio)
    gdp.__dict__[gdp_key].det['result_total'] = (
      gdp.__dict__[gdp_key].det.alloc * ratio).round(0).astype(int)

    gdp.__dict__[gdp_key].agg = operations.calc_agg_errors(
      agg=gdp.__dict__[gdp_key].agg,
      det=gdp.__dict__[gdp_key].det,
      bridge=suts.bridge.industries.matrix,
      source='target_total',
      target='result_total')

  logger.info("Reallocating other federal government")
  for va in config.government_scaled.va:
    ratio = utilities.safe_division(
      gdp.__dict__[va].det.loc[
        config.government_scaled.target, 'source_total'],
      gdp.__dict__[va].det.loc[
        config.government_scaled.ratio, 'source_total'])

    diff = ratio * gdp.__dict__[va].det.loc[
      config.government_scaled.ratio, 'result_total']

    gdp.__dict__[va].det.at[
      config.government_scaled.target, 'result_total'] = int(
      gdp.__dict__[va].det.at[
        config.government_scaled.target, 'result_total']
      + diff)
    gdp.__dict__[va].det.at[
      config.government_scaled.source, 'result_total'] = int(
      gdp.__dict__[va].det.at[
        config.government_scaled.source, 'result_total']
      - diff)

  for gdp_key in gdp.__dict__.keys():
    gdp.__dict__[gdp_key].det['result_vasut'] = (
      gdp.__dict__[gdp_key].det['result_total'])

    gdp.__dict__[gdp_key].agg = operations.calc_agg_errors(
      agg=gdp.__dict__[gdp_key].agg,
      det=gdp.__dict__[gdp_key].det,
      bridge=suts.bridge.industries.matrix,
      source='target_vasut',
      target='result_vasut')

  logger.info("Removing extra tax")

  gdp.nettax.det['result_extra'] = gdp.nettax.det['source_extra']
  gdp.nettax.det['result_vasut'] = gdp.nettax.det['source_vasut']

  # fixing
  total_non_extra_null_vasut_null = (
    (gdp.nettax.det.result_total != 0)
    & (gdp.nettax.det.result_extra == 0)
    & (gdp.nettax.det.result_vasut == 0))

  total_pos_extra_neg_vasut_null = (
    (gdp.nettax.det.result_total > 0)
    & (gdp.nettax.det.result_extra < 0)
    & (gdp.nettax.det.result_vasut == 0))

  total_pos_extra_null_vasut_neg = (
    (gdp.nettax.det.result_total > 0)
    & (gdp.nettax.det.result_extra == 0)
    & (gdp.nettax.det.result_vasut < 0))

  total_neg_extra_pos_vasut_null = (
    (gdp.nettax.det.result_total < 0)
    & (gdp.nettax.det.result_extra > 0)
    & (gdp.nettax.det.result_vasut == 0))

  total_neg_extra_null_vasut_pos = (
    (gdp.nettax.det.result_total < 0)
    & (gdp.nettax.det.result_extra == 0)
    & (gdp.nettax.det.result_vasut > 0))

  total_pos_extra_neg_vasut_neg = (
    (gdp.nettax.det.result_total > 0)
    & (gdp.nettax.det.result_extra < 0)
    & (gdp.nettax.det.result_vasut < 0))

  total_neg_extra_pos_vasut_pos = (
    (gdp.nettax.det.result_total < 0)
    & (gdp.nettax.det.result_extra > 0)
    & (gdp.nettax.det.result_vasut > 0))

  total_pos_extra_pos_vasut_null = (
    (gdp.nettax.det.result_total > 0)
    & (gdp.nettax.det.result_extra > 0)
    & (gdp.nettax.det.result_vasut == 0))

  total_pos_extra_null_vasut_pos = (
    (gdp.nettax.det.result_total > 0)
    & (gdp.nettax.det.result_extra == 0)
    & (gdp.nettax.det.result_vasut > 0))

  total_neg_extra_neg_vasut_null = (
    (gdp.nettax.det.result_total < 0)
    & (gdp.nettax.det.result_extra < 0)
    & (gdp.nettax.det.result_vasut == 0))

  total_neg_extra_null_vasut_neg = (
    (gdp.nettax.det.result_total < 0)
    & (gdp.nettax.det.result_extra == 0)
    & (gdp.nettax.det.result_vasut < 0))

  gdp.nettax.det['result_extra'] = (
    gdp.nettax.det.result_extra
    + 0.5 * gdp.nettax.det.result_total
    * total_non_extra_null_vasut_null
    + (total_pos_extra_null_vasut_neg + total_neg_extra_null_vasut_pos)
    * (gdp.nettax.det.result_total - gdp.nettax.det.result_vasut)
    + 1 * total_pos_extra_null_vasut_pos
    - 1 * total_neg_extra_null_vasut_neg
  )

  gdp.nettax.det['result_extra'] = (
    gdp.nettax.det.result_extra * (
      (~ total_pos_extra_neg_vasut_neg)
      + (~ total_neg_extra_pos_vasut_pos))
    + 0.5 * gdp.nettax.det.result_total * (
      total_pos_extra_neg_vasut_neg + total_neg_extra_pos_vasut_pos)
  )

  gdp.nettax.det['result_vasut'] = (
    gdp.nettax.det.result_vasut
    + 0.5 * gdp.nettax.det.result_total
    * total_non_extra_null_vasut_null
    + (total_pos_extra_neg_vasut_null + total_neg_extra_pos_vasut_null)
    * (gdp.nettax.det.result_total - gdp.nettax.det.result_extra)
    + 1 * total_pos_extra_pos_vasut_null
    - 1 * total_neg_extra_neg_vasut_null
  )

  gdp.nettax.det['result_vasut'] = (
    gdp.nettax.det.result_vasut * (
      (~ total_pos_extra_neg_vasut_neg)
      + (~ total_neg_extra_pos_vasut_pos))
    + 0.5 * gdp.nettax.det.result_total * (
      total_pos_extra_neg_vasut_neg + total_neg_extra_pos_vasut_pos)
  )

  gdp.nettax.det['comparison'] = (
    gdp.nettax.det.result_extra + gdp.nettax.det.result_vasut)

  for key in ['extra', 'vasut']:
    gdp.nettax.det['result_' + key] = gdp.nettax.det[
      'result_' + key] * (
        utilities.safe_division(gdp.nettax.det.result_total,
                                gdp.nettax.det.comparison))

    gdp.nettax.agg = operations.calc_agg_errors(
      agg=gdp.nettax.agg,
      det=gdp.nettax.det,
      bridge=suts.bridge.industries.matrix,
      source='target_' + key,
      target='result_' + key)

  gdp.nettax.det['comparison'] = (
    gdp.nettax.det.result_extra + gdp.nettax.det.result_vasut)

  comparison_absolute = (
    abs(gdp.nettax.det.result_extra) + abs(gdp.nettax.det.result_vasut))

  aggregate_absolute = {}
  for key in ['extra', 'vasut']:
    aggregate_absolute[key] = suts.bridge.industries.matrix.dot(
      abs(gdp.nettax.det['result_' + key]))

  alpha = config.balancing.alpha
  n = config.balancing.n
  logger.info(f"Balancing with alpha {alpha} and n {n}")
  for k in range(n):
    # logger.info(f"Iteration {k}/{n}")
    for key in ['extra', 'vasut']:
      gdp.nettax.det['result_' + key] = gdp.nettax.det[
        'result_' + key] + abs(gdp.nettax.det[
          'result_' + key]) * alpha * (
            utilities.safe_division(
              gdp.nettax.det.result_total - gdp.nettax.det.comparison,
              comparison_absolute)
          + suts.bridge.industries.matrix.T.dot(
            utilities.safe_division(
              gdp.nettax.agg['target_' + key] - gdp.nettax.agg[
                'result_' + key],
              aggregate_absolute[key])))

      gdp.nettax.agg = operations.calc_agg_errors(
        agg=gdp.nettax.agg,
        det=gdp.nettax.det,
        bridge=suts.bridge.industries.matrix,
        source='target_' + key,
        target='result_' + key,
        show=False)

      aggregate_absolute[key] = suts.bridge.industries.matrix.dot(
        abs(gdp.nettax.det['result_' + key]))

    comparison_absolute = (
      abs(gdp.nettax.det.result_extra)
      + abs(gdp.nettax.det.result_vasut))

    gdp.nettax.det['comparison'] = (
      gdp.nettax.det.result_extra + gdp.nettax.det.result_vasut)

    gdp.nettax.det['comparison_diff'] = (
      gdp.nettax.det.comparison - gdp.nettax.det.result_total)

    gdp.nettax.det['comparison_perc'] = (utilities.safe_division(
      gdp.nettax.det.comparison_diff, gdp.nettax.det.result_total)
      * 100).round(2)

  for level in ['det', 'agg']:
    for key in ['extra', 'vasut']:
      gdp.nettax.__dict__[level]['result_' + key] = (
        gdp.nettax.__dict__[level][
          'result_' + key]).round(0).astype(int)

  gdp.nettax.det['comparison'] = (
    gdp.nettax.det.result_extra + gdp.nettax.det.result_vasut)

  gdp.nettax.det['comparison_diff'] = (
    gdp.nettax.det.comparison - gdp.nettax.det.result_total)

  gdp.nettax.det['comparison_perc'] = (utilities.safe_division(
    gdp.nettax.det.comparison_diff, gdp.nettax.det.result_total)
    * 100).round(2)

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]

  @ utilities.open_excel(dir_ / config.target.file)
  def save_table(writer, results, sheet_name):
    results.to_excel(writer, sheet_name=sheet_name, index=True)

  for key, val in pce.__dict__.items():
    save_table('_', val, 'pce_' + key)

  for key, val in gdp.__dict__.items():
    for key0, val0 in val.__dict__.items():
      save_table('_', val0, key + '_' + key0)

  logger.info("Finished step")
  return None


def process_government(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Processing government expenditure proxies")

  logger.info("Loading data")

  url = config.url
  states = operations.load_government_config().state_codes
  states['defense'] = 0.0
  states['nondefense'] = 0.0

  state_local = operations.load_government_config().state_local
  for column in state_local.columns:
    if 'Unnamed' in column:
      state_local.drop(column, axis=1, inplace=True)
  state_local.columns = [x.strip() for x in state_local.columns]
  index = state_local.index.dropna()
  state_local = state_local.loc[index]
  del column, index

  logger.info("Extracting state and local government")
  for index, row in states.iterrows():
    if row.title not in state_local.index:
      logger.error(f"'{row.title}' missing from 'state_local' sheet index"
                   f"in government_config.xlsx")
      raise ValueError

  for target_col, col_list in config.state_local.__dict__.items():
    states[target_col] = 0.0
    for source_col in col_list:
      if source_col not in state_local.columns:
        logger.error(f"'{source_col}' missing from 'state_local' sheet "
                     f"columns in government_config.xlsx")
        raise ValueError
      for index, row in states.iterrows():
        states.loc[index, target_col] = states.loc[index, target_col] + (
          state_local.loc[row.title, source_col])

  states.other = states.other - (states.education + states.health)

  logger.info("Extracting federal government")
  for index, row in states.iterrows():
    tmp = row.two_letter
    config.request.filters.place_of_performance_locations[0].state = tmp
    request_body = config.request.reverse()
    response = requests.post(url, json=request_body)

    if response.status_code == 200:
      response_body = response.json()["results"]
      logger.info(f"{row.title}, {row.two_letter}: "
                  f" Found {len(response_body)} results")
    else:
      logger.info(f"Request failed with status: {response.status_code}")
      continue

    df = pd.DataFrame(response_body)

    states.loc[index, 'defense'] = df.loc[
      df.code == 'DOD', 'amount'].values[0].round(0)
    states.loc[index, 'nondefense'] = (
      df['amount'].sum() - states.loc[index, 'defense']).round(0)

  for column in states.columns:
    if column not in ['title', 'two_letter']:
      states[column] = states[column].astype(int)

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  states.to_csv(
    dir_ / config.target.file, index=True)

  logger.info("Finished step")

  return None



def process_government_census(task_str, full_config):
  """RECON2024: state & local columns of the government proxy from the Census
  Annual Survey of State and Local Government Finances time series
  (GOVSLOCALFINTIMESERIES.GS00LOCALFIN, 2017-2024, State and Local, $k).
  Items matched to the 2022 config columns (median |log ratio| 2022 < 0.6%,
  health 5%): Current operations LF0094, Education total LF0107, Hospitals
  total LF0129, Health total LF0132. Same arithmetic as process_government:
  other = current operations - (education + health), health = hospitals +
  health. Federal columns (defense, nondefense; USAspending) are kept from
  the previous government.csv, since the API is not reachable here."""
  import zipfile, io
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year = str(full_config.year)
  with zipfile.ZipFile(dirs.__dict__[config.source.folder] / config.source.file) as z:
    name = [n for n in z.namelist() if n.endswith('-Data.csv')][0]
    d = pd.read_csv(io.BytesIO(z.read(name)), skiprows=[1], dtype=str, encoding='utf-8-sig')
  d = d.loc[(d.time == year) & (d.GOVTYPE == config.source.govtype)]
  d['AMOUNT'] = pd.to_numeric(d.AMOUNT, errors='coerce')
  d['code'] = d.GEO_ID.str[-2:] + '000'
  piv = d.pivot_table(index='code', columns='AGG_DESC', values='AMOUNT', aggfunc='first')
  item = config.items.__dict__
  gov_file = dirs.__dict__[config.target.folder] / config.target.file
  gov = pd.read_csv(gov_file, dtype={'code': str})
  gov['code'] = gov.code.str.zfill(5); gov = gov.set_index('code')
  cur = piv[item['current_operations']]; edu = piv[item['education']]
  hea = piv[item['hospitals']] + piv[item['health']]
  missing = [c for c in gov.index if c not in piv.index]
  if missing:
    raise ValueError(f"states missing from the Census file: {missing}")
  old = gov[['other', 'education', 'health']].sum()
  gov['education'] = edu.reindex(gov.index).round(0).astype('int64')
  gov['health'] = hea.reindex(gov.index).round(0).astype('int64')
  gov['other'] = (cur - edu - hea).reindex(gov.index).round(0).astype('int64')
  new = gov[['other', 'education', 'health']].sum()
  logger.info(f"state & local {year} ($bn): other {old['other'] / 1e6:,.0f} -> {new['other'] / 1e6:,.0f}, "
              f"education {old['education'] / 1e6:,.0f} -> {new['education'] / 1e6:,.0f}, "
              f"health {old['health'] / 1e6:,.0f} -> {new['health'] / 1e6:,.0f}; federal columns unchanged")
  gov.to_csv(gov_file, index=True)
  logger.info("Finished step")
  return None



def process_government_usaspending(task_str, full_config):
  """RECON2024: federal columns of the government proxy = FY{year} federal
  CONTRACT obligations (award types A-D) by place of performance, DOD
  (defense) and all other agencies (nondefense), from USAspending
  spending_by_geography (pulled 2026-09-21 through the browser, saved as
  raw/Government/usaspending_contracts_pop_state_FY{year}.csv). Replaces
  João's all-award totals by recipient location, which carried Medicare
  Advantage, loans and insurance to insurers' headquarters states."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  src = pd.read_csv(dirs.__dict__[config.source.folder] / config.source.file)
  src = src.set_index('two_letter')
  gov_file = dirs.__dict__[config.target.folder] / config.target.file
  gov = pd.read_csv(gov_file, dtype={'code': str})
  gov['code'] = gov.code.str.zfill(5)
  miss = [t for t in gov.two_letter if t not in src.index]
  if miss:
    raise ValueError(f"states missing from the USAspending file: {miss}")
  old_d, old_n = gov.defense.sum(), gov.nondefense.sum()
  gov['defense'] = src.loc[gov.two_letter, 'contracts_dod'].round(0).astype('int64').values
  gov['nondefense'] = src.loc[gov.two_letter, 'contracts_nondod'].round(0).astype('int64').values
  logger.info(f"federal ($bn): defense {old_d / 1e9:,.1f} -> {gov.defense.sum() / 1e9:,.1f}, "
              f"nondefense {old_n / 1e9:,.1f} -> {gov.nondefense.sum() / 1e9:,.1f} "
              f"(contracts by place of performance)")
  gov.set_index('code').to_csv(gov_file, index=True)
  logger.info("Finished step")
  return None


def process_rpc_tax_inputs(task_str, full_config):
  """RECON2024: {year} replacements for the rpc stage's three 2022 tax inputs,
  written in the same layout into the rpc process folder (the rpc steps
  process_state_local_revenue / process_business_income_tax /
  process_total_income_tax are then not run):
    state_local_revenue.csv - Census Annual Survey of State & Local Government
        Finances {year} (GS00LOCALFIN), the fifteen 22slsstab1 lines mapped to
        item codes (2022 values within 1.5% of the table, revisions);
        state & local / state / local
    business_income_tax.csv - IRS Data Book FY{year} business income tax
        collections by state (same concept as 22dbs01t05co)
    total_income_tax.csv    - IRS SOI individual income tax by state: kept at
        the latest tax year on disk ({config.total_income_tax.note})."""
  import zipfile, io
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year = str(full_config.year)
  out = dirs.__dict__[config.target_folder]
  out.mkdir(parents=True, exist_ok=True)

  # ---- state & local revenue
  with zipfile.ZipFile(dirs.__dict__[config.census.folder] / config.census.file) as z:
    name = [n for n in z.namelist() if n.endswith('-Data.csv')][0]
    d = pd.read_csv(io.BytesIO(z.read(name)), skiprows=[1], dtype=str, encoding='utf-8-sig')
  d = d.loc[d.time == year]
  d['AMOUNT'] = pd.to_numeric(d.AMOUNT, errors='coerce').fillna(0.0)
  d['code'] = d.GEO_ID.str[-2:] + '000'
  govmap = {'001': 'state_local', '002': 'state', '003': 'local'}
  d = d.loc[d.GOVTYPE.isin(govmap)]
  piv = d.pivot_table(index=['GOVTYPE', 'code', 'NAME'], columns='AGG_DESC', values='AMOUNT', aggfunc='sum').fillna(0.0)
  rows = []
  for line, spec in config.census.lines.__dict__.items():
    plus = spec.plus; minus = getattr(spec, 'minus', [])
    val = piv[plus].sum(axis=1) - (piv[minus].sum(axis=1) if minus else 0.0)
    for (gt, code, nm), v in val.items():
      rows.append({'line_code': int(line[1:]), 'region_code': code, 'government': govmap[gt],
                   'value': int(round(v)), 'line_description': spec.label, 'region_description': nm})
  rev = pd.DataFrame(rows)
  # complete every line x region x government (DC has no separate state
  # government in the time series; the 2022 table carried it as zeros)
  full = pd.MultiIndex.from_product([sorted(rev.line_code.unique()), sorted(rev.region_code.unique()),
                                     list(govmap.values())], names=['line_code', 'region_code', 'government'])
  rev = rev.set_index(['line_code', 'region_code', 'government'])
  rev = rev[~rev.index.duplicated()].reindex(full).reset_index()
  rev['value'] = rev['value'].fillna(0).astype('int64')
  rev['line_description'] = rev.groupby('line_code').line_description.transform('first')
  rev['region_description'] = rev.groupby('region_code').region_description.transform('first')
  us = rev.groupby(['line_code', 'government', 'line_description'], as_index=False).value.sum()
  us['region_code'] = '00000'; us['region_description'] = 'United States Total'
  rev = pd.concat([us[rev.columns], rev], ignore_index=True).sort_values(['region_code', 'government', 'line_code'])
  rev.to_csv(out / 'state_local_revenue.csv', index=False)
  tot = rev.loc[(rev.region_code == '00000') & (rev.government == 'state_local')].value.sum()
  logger.info(f"state_local_revenue.csv: {len(rev)} rows, {year}; US state & local, 15 lines, ${tot / 1e6:,.0f}bn")

  # ---- business income tax (IRS Data Book collections)
  irs = pd.read_csv(dirs.__dict__[config.irs.folder] / config.irs.file)
  codes = pd.read_excel(utilities.INTERNAL_PATH / 'sut' / 'government_config.xlsx', sheet_name='state_codes', dtype=str)
  name2code = dict(zip(codes.title.str.strip(), codes.code.str.strip().str.zfill(5)))
  name2code['United States, total'] = '00000'
  irs = irs.loc[irs.area.isin(name2code)]
  bit = pd.DataFrame({'code': irs.area.map(name2code), 'value': irs[config.irs.column].round(0).astype('int64'),
                      'description': irs.area})
  bit = bit.sort_values('code')
  bit.to_csv(out / 'business_income_tax.csv', index=False)
  logger.info(f"business_income_tax.csv: {len(bit)} rows, FY{year}; US ${bit.loc[bit.code == '00000', 'value'].sum() / 1e6:,.0f}bn")

  # ---- individual income tax: IRS SOI Historic Table 2, latest tax year
  # (A06500 income tax after credits, $k; same concept as 22in01stateshares
  # col W, which it reproduces within 1.2% for 2022), carried to {year} by
  # each state's personal income growth (BEA SAINC4 line 10), i.e. the
  # latest observed state tax/income rate applied to {year} income
  cfg = config.total_income_tax
  soi = pd.read_csv(dirs.__dict__[cfg.folder] / cfg.file)
  ty = int(soi.TY.max())
  soi = soi.loc[soi.TY == ty].set_index('STATE')
  sainc = pd.read_csv(dirs.__dict__[cfg.income_folder] / cfg.income_file, dtype=str, encoding='latin-1')
  sainc['GeoFIPS'] = sainc.GeoFIPS.str.strip().str.replace('"', '')
  pi = sainc.loc[sainc.LineCode == '10'].set_index('GeoFIPS')
  growth = (pd.to_numeric(pi[year], errors='coerce') / pd.to_numeric(pi[str(ty)], errors='coerce'))
  ab2 = dict(zip(codes.two_letter.str.strip(), codes.code.str.strip().str.zfill(5)))
  ab2['US'] = '00000'
  rows = []
  for ab, code in ab2.items():
    if ab not in soi.index:
      raise ValueError(f"{ab} missing from the SOI file")
    g = float(growth.get(code, np.nan))
    rows.append({'region_description': 'United States' if code == '00000' else
                 codes.set_index('two_letter').title.get(ab, ab),
                 'value': int(round(float(soi.loc[ab, cfg.column]) * g)), 'region_code': code})
  tit = pd.DataFrame(rows).sort_values('region_code')
  tit.to_csv(out / 'total_income_tax.csv', index=False)
  us = tit.loc[tit.region_code == '00000', 'value'].iloc[0]
  logger.info(f"total_income_tax.csv: SOI TY{ty} {cfg.column} x state personal income "
              f"growth {ty}-{year}; US ${us / 1e6:,.0f}bn (states sum "
              f"${tit.loc[tit.region_code != '00000', 'value'].sum() / 1e6:,.0f}bn)")
  logger.info("Finished step")
  return None


def create_vectors(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  bea_config = full_config.process_bea
  mar_config = full_config.apply_margins
  sym_config = full_config.create_symmetric
  gov_config = full_config.process_government
  logger.info("Processing regional vectors")

  logger.info("Loading data")
  bea = bea_operations.load_bridged()
  qcew = bea_operations.load_qcew()
  qcew['wages'] = (qcew.wages / 1000).round(0).astype(int)

  pce_stat, gdp_stat = operations.load_bea_statistics()

  dir_ = dirs.__dict__[mar_config.target.folder]
  sut = operations.Sut.read_excel(
    dir_ / mar_config.target.file, 'sut')

  dir_ = dirs.__dict__[sym_config.target.folder]
  iot = operations.Sut.read_excel(
    dir_ / sym_config.target.files.iot, 'iot', is_iot=True)
  del dir_

  logger.info("Convert BEA statistics PCE and GDP to thousand dollars")
  for agg_key in pce_stat.__dict__.keys():
    pce_stat.__dict__[agg_key] = 1000 * (
      pce_stat.__dict__[agg_key]).astype(float)
    for gdp_key in sut.codes.primary.index:
      gdp_stat.__dict__[gdp_key].__dict__[agg_key] = 1000 * (
        gdp_stat.__dict__[gdp_key].__dict__[agg_key]).astype(float)

  logger.info("Remove margins from BEA statistics")
  for gdp_key, gdp_val in gdp_stat.__dict__.items():
    for agg_key in gdp_val.__dict__.keys():
      gdp_stat.__dict__[gdp_key].__dict__[agg_key].drop(
        ['TRADE', 'TRANS'], axis=0, inplace=True)

  logger.info("Resetting reference PCE and GDP with SUT values")
  pce_stat.det['final'] = 1000 * sut.data.use_fin['pce']
  for gdp_key in sut.codes.primary.index:
    gdp_stat.__dict__[gdp_key].det['result_vasut'] = 1000 * (
      iot.data.use_pri.loc[gdp_key])
    if gdp_key == 'nettax':
      gdp_stat.__dict__[gdp_key].det['result_total'] = (
        gdp_stat.__dict__[gdp_key].det['result_vasut']
        + 1000 * gdp_stat.__dict__[gdp_key].det['result_extra'])
    else:
      gdp_stat.__dict__[gdp_key].det['result_total'] = (
        gdp_stat.__dict__[gdp_key].det['result_vasut'])

  logger.info("Adjust PCE data (national and state-level)")
  vec_com = operations.process_bea_pce(
    bea, pce_stat, bea_config)

  tmp = pd.DataFrame(
    index=sut.codes.commodities.index,
    columns=bea.regions.loc[bea.regions.level == 2].index,
    data=int(0))
  tmp['commodity'] = tmp.index
  tmp = pd.melt(
    tmp,
    id_vars=['commodity'],
    value_vars=tmp.columns,
    var_name='region', value_name='pce')
  tmp.index = tmp.region + '_' + tmp.commodity

  vec_com = pd.concat([vec_com, tmp], axis=0)
  del tmp

  logger.info("Adding other final demand at national level")
  for final_demand in sut.codes.final.index:
    if final_demand != 'pce':
      vec_com[final_demand] = 0.0
      tmp = deepcopy(
        sut.data.use_fin[final_demand].astype(float) * 1000)
      tmp.index = '00000_' + tmp.index
      vec_com.loc[tmp.index, final_demand] = tmp

  # logger.info("Saving commodity data")
  # dir_ = dirs.__dict__[config.target.folder]
  # vec_com = vec_com.round(0)
  # vec_com.to_csv(
  #  dir_ / config.target.files.commodity, index=False)

  logger.info("Performing adhoc adjustments to QCEW variables")
  bea_employment = deepcopy(bea.bea.employment)

  bea.bea['employment'] = deepcopy(qcew.establishments)
  vec_ind = operations.process_bea_gdp(
    bea, gdp_stat, bea_config)
  qcew_establishments = deepcopy(vec_ind['employment'])

  bea.bea['employment'] = deepcopy(qcew.employment)
  vec_ind = operations.process_bea_gdp(
    bea, gdp_stat, bea_config)
  qcew_employment = deepcopy(vec_ind['employment'])

  bea.bea['employment'] = deepcopy(qcew.wages)
  vec_ind = operations.process_bea_gdp(
    bea, gdp_stat, bea_config)
  qcew_wages = deepcopy(vec_ind['employment'])

  logger.info("Performing adhoc adjustments to BEA variables")
  bea.bea['employment'] = deepcopy(bea_employment)
  del bea_employment
  vec_ind = operations.process_bea_gdp(
    bea, gdp_stat, bea_config)

  vec_ind.insert(2, 'qcew_establishments', qcew_establishments)
  vec_ind.insert(3, 'qcew_employment', qcew_employment)
  vec_ind.insert(4, 'qcew_wages', qcew_wages)
  del qcew_establishments, qcew_employment, qcew_wages

  logger.info("Adjust GDP data (national, state- and county-level)")
  # nettax_extra will later be revised to be scaled from gdp
  # imports, noncomparable and imports will be added, scaled from gdp
  # gdp will later be revised to include nettax_extra
  vec_ind['gdp'] = (vec_ind.wages + vec_ind.nettax
                    + vec_ind.surplus + vec_ind.nettax_extra)
  # RECON2024: taxes on production and imports less subsidies (SAGDP/CAINC
  # total, before it is split) - regional nettax is re-derived from it below
  va_basic = bool(getattr(config, 'va_basic', False))
  tls_all = (vec_ind.nettax + vec_ind.nettax_extra).astype(float)
  vec_ind['parent'] = None
  reg_offspring = vec_ind.loc[vec_ind.region != '00000']
  reg_parent = bea.regions.loc[reg_offspring.region.values, 'parent']
  vec_ind.loc[reg_offspring.index, 'parent'] = (
    reg_parent.values + '_'
    + vec_ind.loc[reg_offspring.index, 'industry'])

  national = pd.DataFrame(index=sut.codes.industries.index)
  national['fraction'] = iot.data.A.sum(axis=0)
  national['output'] = iot.codes.industries['tot'] * 1000
  national['imports'] = - iot.data.use_fin['imports'] * 1000
  national['noncomparable'] = iot.data.use_pri.loc[
    'noncomparable'] * 1000
  national['nettax_extra'] = iot.data.use_pri.loc[
    'nettax_extra'] * 1000  # override bea, not yet in GDP
  if va_basic:
    # RECON2024: the 2024 SUT balances industry columns at basic prices
    # (purchasers'-price intermediate + VA at basic prices = output), so the
    # residual above is ~0. Taxes on products less subsidies by industry
    # (VAPRO - VABAS, $1.01T in 2024) come from the BEA statistics instead;
    # they are carried for GDP at producer prices and for the tax split in
    # the rpc stage, and are NOT part of the column identity.
    national['nettax_extra'] = iot.codes.industries['extra_bea'].reindex(
      national.index).fillna(0.0).astype(float) * 1000
    logger.info(f"nettax_extra (taxes on products less subsidies) from BEA: "
                f"{national['nettax_extra'].sum() / 1e6:.1f} $bn")

  for column in ['fraction', 'output', 'imports', 'noncomparable',
                 'nettax_extra']:
    vec_ind[column] = 0.0
    vec_ind.loc['00000_' + national.index, column] = national[
      column].values
  del national

  logger.info(
    "Generating industry output and intermediate use fraction")
  va_list = (['wages', 'surplus', 'nettax', 'noncomparable'] if va_basic else
             ['wages', 'surplus', 'nettax', 'noncomparable', 'nettax_extra'])
  for key_reg, val_reg in {'state': 1, 'county': 2}.items():
    vec_trunc = deepcopy(vec_ind.loc[(bea.regions.loc[
      vec_ind.region.values, 'level'] == val_reg).values])

    # offspring fraction of parent wages
    proxy = utilities.safe_inverse(vec_ind.loc[
      vec_trunc.parent.values, 'wages'].values)
    proxy = proxy + 1 * (proxy == 0)
    proxy = proxy * vec_trunc['wages'].values

    # RECON2024: industries whose value added is mostly not compensation
    # (owner-occupied housing, lessors of IP ...) cannot be placed by wages;
    # their output is split by the regions' GDP share instead
    if hasattr(config, 'va_output'):
      par_vals = vec_ind.loc[vec_trunc.parent.values]
      par_va = par_vals[['wages', 'surplus', 'nettax']].sum(axis=1).values
      comp_share = utilities.safe_division(par_vals['wages'].values, par_va)
      use_gdp = (comp_share < config.va_output.min_comp_share) & (par_va > 0)
      if use_gdp.any():
        gproxy = utilities.safe_division(np.clip(vec_trunc['gdp'].values, 0, None),
                                         np.clip(par_vals['gdp'].values, 0, None))
        proxy = np.where(use_gdp, gproxy, proxy)
        logger.info(f"{key_reg}: {int(use_gdp.sum())} cells in industries with compensation "
                    f"< {config.va_output.min_comp_share:.0%} of value added split by GDP share: "
                    f"{sorted(set(vec_trunc.industry.values[use_gdp]))}")

    # using wages as proxy to split variables
    for column in ['output', 'imports',
                   'noncomparable', 'nettax_extra']:
      vec_trunc[column] = (
        proxy * vec_ind.loc[vec_trunc.parent.values, column].values
      ).round(0)
    if va_basic and key_reg == 'state':
      # other taxes on production less subsidies = the state's SAGDP total
      # less its share of product taxes; state totals are kept and the
      # national vasut/extra ratios (which explode where the two parts have
      # opposite signs, e.g. farm subsidies in 111900) are not used
      tls_s = tls_all.loc[vec_trunc.index].astype(float)
      # where the US product-tax share of TLS is in [0, 1] (both parts the
      # same sign), states get that share of their own SAGDP TLS: product
      # taxes follow where TLS is actually collected (states without a sales
      # tax get little), no negative 'other production taxes' residuals.
      # Elsewhere (sign conflicts: farm subsidies ...) the output proxy stays.
      par_idx = vec_trunc.parent.values
      us_ex = vec_ind.loc[par_idx, 'nettax_extra'].values.astype(float)
      us_tl = tls_all.loc[par_idx].values.astype(float)
      ratio = np.where(us_tl != 0, us_ex / np.where(us_tl != 0, us_tl, 1.0), np.nan)
      ok = (ratio >= 0) & (ratio <= 1)
      vec_trunc['nettax_extra'] = np.where(ok, (ratio * tls_s.values).round(0), vec_trunc['nettax_extra'].values)
      logger.info(f"state product taxes by the state's TLS in {int(ok.sum())} cells, "
                  f"output proxy in {int((~ok).sum())}")
      vec_trunc['nettax'] = (tls_s - vec_trunc['nettax_extra']).round(0)
    elif va_basic:
      # counties: county TLS is itself an allocation and is noisy for small
      # cells, so the counties keep their split and the gap to the revised
      # state nettax is spread by the same proxy as output
      par_tax = vec_ind.loc[vec_trunc.parent.values, 'nettax'].values.astype(float)
      kid_sum = vec_trunc.groupby('parent')['nettax'].transform('sum').values.astype(float)
      vec_trunc['nettax'] = (vec_trunc['nettax'].astype(float)
                             + proxy * (par_tax - kid_sum)).round(0)

    # calculating intermediate use fraction
    vec_trunc['fraction'] = 1 - utilities.safe_division(
      vec_trunc[va_list].sum(1).values,
      vec_trunc.output.values)
    vec_trunc.loc[vec_trunc.output == 0, 'fraction'] = 0

    # RECON2024: VA > output handled by operations.reconcile_va_output
    # (plan 2: raise output, keep regional GDP; plan 1: cut GOS), conserving
    # parent totals. João's two fallbacks below then find nothing to do.
    if hasattr(config, 'va_output'):
      vec_trunc = operations.reconcile_va_output(
        vec_trunc, vec_ind, config.va_output, key_reg, va_cols=va_list)
      vec_trunc['fraction'] = 1 - utilities.safe_division(
        vec_trunc[va_list].sum(1).values,
        vec_trunc.output.values)
      vec_trunc.loc[vec_trunc.output == 0, 'fraction'] = 0

    # if intermediate use fraction is negative, use gdp as proxy
    tmp = (vec_trunc.fraction < 0)
    tmp = tmp.loc[tmp]
    neg_index = tmp.index
    del tmp
    if len(neg_index) > 0:
      tmp_show = vec_trunc.loc[neg_index,
                               ['wages', 'gdp', 'output', 'fraction']]
      logger.info(
        f"Using wages as proxy found {len(neg_index)} "
        "negative intermediate "
        f"use factions at {key_reg} level:´\n"
        f"{tmp_show}")
      del tmp_show

      # offspring fraction of parent wages
      proxy = utilities.safe_inverse(vec_ind.loc[
        vec_trunc.loc[neg_index, 'parent'].values, 'gdp'].values)
      proxy = proxy + 1 * (proxy == 0)
      proxy = proxy * vec_trunc.loc[neg_index, 'gdp'].values

      # using wages as proxy to split variables
      for column in ['output', 'imports', 'noncomparable',
                     'nettax_extra']:
        vec_trunc.loc[neg_index, column] = (
          proxy * vec_ind.loc[
            vec_trunc.loc[neg_index, 'parent'].values, column].values
        ).round(0)
      if va_basic and key_reg == 'state':
        vec_trunc.loc[neg_index, 'nettax'] = (
          tls_all.loc[neg_index] - vec_trunc.loc[neg_index, 'nettax_extra']).round(0)

      vec_trunc.loc[neg_index, 'fraction'] = (
        1 - utilities.safe_division(
          vec_trunc.loc[neg_index, va_list].sum(1).values,
          vec_trunc.loc[neg_index, 'output'].values))
      vec_trunc.loc[vec_trunc.output == 0, 'fraction'] = 0

      # if intermediate use fraction is negative, assign difference
      tmp = (vec_trunc.fraction < 0)
      tmp = tmp.loc[tmp]
      neg_index = tmp.index
      del tmp
      if len(neg_index) > 0:
        tmp_show = vec_trunc.loc[neg_index,
                                 ['wages', 'gdp', 'output', 'fraction']]
        logger.info(
          f"Using wages as proxy found {len(neg_index)} "
          "negative intermediate "
          f"use factions at {key_reg} level:´\n"
          f"{tmp_show}")
        del tmp_show

        if not va_basic:
         vec_trunc.loc[neg_index, 'nettax_extra'] = (
          vec_trunc.loc[neg_index, 'output']
          - vec_trunc.loc[neg_index, [
            'wages', 'surplus', 'nettax',
            'noncomparable']].sum(1).values)
        vec_trunc.loc[neg_index, 'fraction'] = 0

    # vec_trunc['output'] = vec_trunc[[
    #   'wages', 'surplus', 'nettax', 'nettax_extra',
    #   'noncomparable']].sum(1).values
    vec_trunc['fraction'] = 1 - utilities.safe_division(
      vec_trunc[va_list].sum(1).values,
      vec_trunc['output'].values)
    vec_trunc.loc[vec_trunc.output == 0, 'fraction'] = 0
    if va_basic:
      n_small = int((vec_trunc.fraction < 0).sum())
      if n_small:
        logger.info(f"{key_reg}: {n_small} slightly negative intermediate fractions "
                    f"(min {vec_trunc.fraction.min():.4f}) set to 0")
      vec_trunc['fraction'] = vec_trunc['fraction'].clip(lower=0)

    vec_ind.loc[vec_trunc.index] = vec_trunc

  # RECON2024: state PCE by commodity ($k) is needed by the services-import
  # allocation in apply_state_trade; keep it before vec_com is dropped
  pce_com_state = vec_com.pivot(index='commodity', columns='region', values='pce').astype(float)
  pce_com_state.to_csv(dirs.__dict__[config.target.folder] / f"pce_state_commodity_{full_config.year}.csv")

  logger.info("Converting final demand to industries")
  for final_demand in sut.codes.final.index:
    tmp = vec_com.pivot(index='commodity',
                        columns='region',
                        values=final_demand).astype(float)
    tmp = tmp.loc[iot.data.D.index]
    tmp = iot.data.D.transpose().dot(tmp)
    tmp['industry'] = tmp.index
    tmp = pd.melt(
      tmp,
      id_vars=['industry'],
      value_vars=tmp.columns,
      var_name='region', value_name='total')
    tmp.index = tmp.region + '_' + tmp.industry
    vec_ind[final_demand] = tmp['total']
  del vec_com

  logger.info(
    "Use output as proxy of final demand except pce and government")
  for final_demand in sut.codes.final.index:
    if final_demand not in ['pce', 'government']:
      national = vec_ind.loc[vec_ind.region == "00000",
                             ["output", final_demand]]
      national["ratio"] = utilities.safe_division(
        national[final_demand].values,
        national.output.values)
      vec_ind[final_demand] = vec_ind.output.values * national.loc[
        '00000_' + vec_ind.industry, "ratio"].values

  if hasattr(config, 'state_trade'):   # RECON2024
    logger.info("Applying state-specific international imports and exports")
    vec_ind = operations.apply_state_trade(
      vec_ind, sut, iot, bea, config.state_trade, dirs,
      pce_com_state=pce_com_state, year=full_config.year,
      out_dir=dirs.__dict__[config.target.folder])
  if getattr(config, 'travel_exports', None) is not None:
    vec_ind = operations.apply_travel_exports(vec_ind, bea, config.travel_exports, dirs)

  logger.info("Split government at state level using proxies")
  government = operations.load_government()

  # state and country split of final demand
  government_sum = government.sum()
  for column, industry in gov_config.industries.__dict__.items():
    for state in government.index:
      vec_ind.loc[state + '_' + industry, 'government'] = (
        vec_ind.loc['00000_' + industry, 'government']
        * utilities.safe_division(
          government.loc[state, column],
          government_sum[column]))

  logger.info("Split pce and government at county level using income")
  for final_demand in ['pce', 'government']:
    tmp = deepcopy(vec_ind.loc[vec_ind.region.isin(
      bea.regions.loc[bea.regions.level == 2].index)])
    tmp['parent'] = bea.regions.loc[
      tmp.region.values, 'parent'].values + '_' + tmp['industry']
    tmp[final_demand] = vec_ind.loc[
      tmp.parent.values, final_demand].values

    tmp[final_demand] = tmp[final_demand] * bea.regions.loc[
      tmp.region.values, 'income'].values

    tmp[final_demand] = utilities.safe_division(
      tmp[final_demand],
      bea.regions.loc[vec_ind.loc[
        tmp.parent.values, 'region'].values, 'income'].values
    ).astype(float)
    vec_ind.loc[tmp.index, final_demand] = tmp[final_demand]
    del tmp

  for column in ['output', 'imports', 'noncomparable',
                 'nettax_extra'] + list(sut.codes.final.index):
    vec_ind[column] = vec_ind[column].round(0)

  vec_ind['fraction'] = vec_ind['fraction'].round(config.decimals)
  vec_ind.drop('parent', axis=1, inplace=True)

  logger.info("Calculating GDP, supply/demand and earnings/gdp ratios")
  vec_ind['gdp'] = (
    vec_ind.wages + vec_ind.surplus + vec_ind.nettax
    + vec_ind.nettax_extra)

  vec_ind = vec_ind.rename(columns={
    'fraction': 'intermediate_use',
    'nettax': 'nettax_production',
    'nettax_extra': 'nettax_commodity',
    'wages': 'compensation',
  })

  logger.info("Saving vectors data")
  for column in vec_ind:
    if column not in ['region', 'industry', 'intermediate_use']:
      vec_ind[column] = vec_ind[column].round(0).astype(int)
  dir_ = dirs.__dict__[config.target.folder]
  vec_ind.to_csv(
    dir_ / config.target.file, index=False)

  logger.info("Finished step")
  return None



def export_state_use_A(task_str, full_config):
  """RECON2024: state Use-side domestication (Mike, 2026-09-20).

  For each state s, on the commodity x industry axis of the balanced,
  margin-adjusted national SUT:
    B      = national total-use coefficients  use_int / industry output
    B_s    = B with column j scaled by the state's intermediate-use fraction
             relative to the nation (the productivity adjustment João applied
             to A, now applied to Use)
    P_s    = diag( m_s(c) / U_s(c) )  international import share of the
             state's total use of c, m_s from the state import vectors
             (imports_state_commodity), U_s = sum_j B_s(c,j) x_j(s) + final
             demand of c in s
    B_s^d  = (I - P_s) B_s           domestic (of international trade) Use
    A_s    = D B_s^d                  industry x industry, national market
             shares D, plus João's households row/column
  Writes results/Bmatrices/recon{year}_Bdom_SS000.csv.gz and
  results/Amatrices_use/recon{year}_Amatrix_SS000.csv. Interregional (RPC)
  domestication is applied afterwards in the rpc stage as before; the
  supply/demand ratio João multiplied into A is NOT applied here."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year = full_config.year
  mar_config = full_config.apply_margins
  sym_config = full_config.create_symmetric
  T = dirs.__dict__[mar_config.target.folder]

  sut = operations.Sut.read_excel(T / mar_config.target.file, 'sut')
  iot = operations.Sut.read_excel(T / sym_config.target.files.iot, 'iot', is_iot=True)
  D = iot.data.D                                  # commodities x industries
  com = list(D.index); ind = list(D.columns)
  U = sut.data.use_int.reindex(index=com, columns=ind).fillna(0.0).astype(float)
  x_us = sut.codes.industries.loc[ind, 'tot'].astype(float)
  B = U.div(x_us.replace(0, np.nan), axis=1).fillna(0.0)
  F = sut.data.use_fin.reindex(index=com).fillna(0.0).astype(float)

  vec = operations.load_vectors()
  vec.index = vec.region + '_' + vec.industry
  bea = bea_operations.load_bridged()
  regions = bea.regions
  states = regions.loc[regions.level == 1].index.tolist()
  nat = vec.loc[vec.region == '00000'].set_index('industry')
  frac_us = nat.intermediate_use.reindex(ind).astype(float)

  M = pd.read_csv(T / f"imports_state_commodity_{year}.csv", index_col=0).reindex(index=com).fillna(0.0) * 1000.0  # $k
  nonpos_import = (nat.imports.reindex(com).astype(float).fillna(1.0) <= 0)
  logger.info(f"national imports <= 0 (import share set to 0): {list(nonpos_import[nonpos_import].index)}")

  # basic-price share of each commodity's use: imports are valued at
  # customs/CIF value, use at purchasers' prices, so the import share is
  # taken against use net of trade and transport margins (margin totals per
  # commodity from the margin template written by apply_margins)
  mt = pd.read_excel(T / mar_config.target.file, sheet_name='margin_trade', index_col=0).sum(axis=1)
  mx = pd.read_excel(T / mar_config.target.file, sheet_name='margin_transport', index_col=0).sum(axis=1)
  margin = (mt.reindex(com).fillna(0.0) + mx.reindex(com).fillna(0.0))
  use_pp = sut.codes.commodities.loc[com, 'use'].astype(float)
  basic_share = (1.0 - margin / use_pp.replace(0, np.nan)).clip(lower=0.05, upper=1.0).fillna(1.0)
  logger.info(f"basic-price share of use: {int((basic_share < 1).sum())} margined commodities, "
              f"min {basic_share.min():.2f}, median of those {basic_share[basic_share < 1].median():.2f}")
  pce = pd.read_csv(T / f"pce_state_commodity_{year}.csv", index_col=0).reindex(index=com).fillna(0.0)          # $k
  inc = regions['income'].astype(float)
  x_st = vec.pivot(index='industry', columns='region', values='output').reindex(ind).astype(float)
  x_share = x_st[states].div(x_st[states].sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

  out_B = dirs.__dict__[config.folder] / config.subfolder_B
  out_A = dirs.__dict__[config.folder] / config.subfolder_A
  out_B.mkdir(parents=True, exist_ok=True); out_A.mkdir(parents=True, exist_ok=True)
  diag_rows = []
  for s in states:
    st = vec.loc[vec.region == s].set_index('industry').reindex(ind)
    r = utilities.safe_division(st.intermediate_use.astype(float).values, frac_us.values)
    r = np.where(frac_us.values > 0, r, 1.0)
    r = np.clip(r, config.ratio_bounds[0], config.ratio_bounds[1])
    B_s = B.multiply(r, axis=1)
    x_s = st.output.astype(float).fillna(0.0).values
    inter_s = B_s.values.dot(x_s)                                  # total intermediate use of c in s ($k)
    # other final demand of c in s: government by income, the rest by the
    # state's share of the commodity's producing industries' output
    gov_s = 1000.0 * F['government'].values * (inc.get(s, 0.0) / inc.get('00000', 1.0))
    prod_share = (D.values * x_share[s].reindex(ind).fillna(0.0).values[None, :]).sum(axis=1)
    oth_s = 1000.0 * (F['investment'] + F['inventory'] + F['exports']).values * prod_share
    U_s = inter_s + (pce[s].values if s in pce.columns else 0.0) + gov_s + oth_s
    U_basic = U_s * basic_share.values                      # purchasers' -> basic-price content
    m_s = M[s].values if s in M.columns else np.zeros(len(com))
    p_basic = np.where(U_basic > 0, m_s / np.where(U_basic > 0, U_basic, 1.0), 0.0)
    p_basic = np.clip(p_basic, 0.0, config.import_share_cap)
    # RECON2024: commodities whose national SUT imports are negative
    # (483000 water and 492000 couriers: BEA's CIF/FOB freight adjustment;
    # also 532400, S00201, S00203, S00600) carry no import leakage of their
    # own - the freight on imported goods is already in the goods' import
    # values - so the Use-side import share is zero for them
    if getattr(config, 'zero_if_national_imports_nonpositive', True):
      p_basic = np.where(nonpos_import.values, 0.0, p_basic)
    # the margin part of a purchasers'-price cell is domestic trade/transport
    # service, so the share removed from the purchasers'-price coefficient is
    # p_basic x basic_share
    p_s = p_basic * basic_share.values
    B_dom = B_s.multiply(1.0 - p_s, axis=0)
    A = pd.DataFrame(D.values.T.dot(B_dom.values), index=ind, columns=ind)   # D' B: industry x industry
    # households (as in export_A_matrices)
    egdp = utilities.safe_division(st.earnings.astype(float), st.gdp.astype(float)).clip(0, 1)
    A.loc['households'] = utilities.safe_division(egdp * st.gdp.astype(float), st.output.astype(float)).fillna(0.0).values
    A['households'] = 0.0
    A.loc[ind, 'households'] = utilities.safe_division(st.pce.astype(float), inc.get(s, np.nan)).fillna(0.0).values
    A = A.round(config.decimals); A.index.name = 'industry'
    A.to_csv(out_A / f"{config.prefix}Amatrix_{s}{config.suffix}")
    B_dom.round(config.decimals).to_csv(out_B / f"{config.prefix}Bdom_{s}{config.suffix}.gz", compression='gzip')
    # state Use in levels ($ thousand): total (productivity-adjusted) and
    # domestic of international imports, commodity x industry
    out_U = dirs.__dict__[config.folder] / config.subfolder_U
    out_U.mkdir(parents=True, exist_ok=True)
    B_s.multiply(x_s, axis=1).round(0).to_csv(out_U / f"{config.prefix}Use_total_{s}{config.suffix}.gz", compression='gzip')
    B_dom.multiply(x_s, axis=1).round(0).to_csv(out_U / f"{config.prefix}Use_domestic_{s}{config.suffix}.gz", compression='gzip')
    diag_rows.append({'region': s, 'int_use_ratio_median': float(np.median(r)), 'ratio_min': float(r.min()),
                      'ratio_max': float(r.max()), 'import_share_mean': float(np.average(p_s, weights=np.maximum(U_s, 0) + 1e-9)),
                      'cells_capped': int((m_s / np.where(U_basic > 0, U_basic, 1.0) > config.import_share_cap).sum()),
                      'A_colsum_mean': float(A.loc[ind, ind].sum().mean())})
    pd.DataFrame({'p_import_basic': p_basic, 'basic_share_of_use': basic_share.values, 'p_applied': p_s},
                 index=pd.Index(com, name='commodity')).round(5).to_csv(out_B / f"{config.prefix}Pimport_{s}.csv")
  diag = pd.DataFrame(diag_rows)
  diag.to_csv(out_A / f"{config.prefix}state_use_diagnostics.csv", index=False)
  logger.info(f"state Use domestication: {len(states)} states; intermediate-use ratio median "
              f"{diag.int_use_ratio_median.median():.3f}; use-weighted import share mean "
              f"{diag.import_share_mean.mean():.3f}; A column sums mean {diag.A_colsum_mean.mean():.3f}")
  logger.info("Finished step")
  return None


def export_A_matrices(task_str, full_config):
  dirs = full_config.dirs
  sym_config = full_config.create_symmetric
  config = full_config.__dict__[task_str]
  logger.info("Exporting A matrices")

  logger.info("Loading data")
  bea = bea_operations.load_bridged()

  dir_ = dirs.__dict__[sym_config.target.folder]
  iot = operations.Sut.read_excel(
    dir_ / sym_config.target.files.iot, 'iot', is_iot=True)
  del dir_

  logger.info("Loading vectors")
  vec_ind = operations.load_vectors()
  vec_ind['earningsgdp'] = utilities.safe_division(
    vec_ind.earnings, vec_ind.gdp).round(
    config.decimals)

  vec_ind['supply'] = vec_ind.output - vec_ind.exports
  vec_ind['demand'] = vec_ind.supply + vec_ind.imports
  vec_ind['supplydemand'] = utilities.safe_division(
    vec_ind.supply, vec_ind.demand)

  dir_ = dirs.__dict__[config.folder]
  tmp = vec_ind.loc[vec_ind.region == '00000']
  tmp.index = tmp.industry
  interuse_ref = tmp.intermediate_use
  del tmp

  logger.info("Generating A matrix of:")
  regions_selection = bea.regions.loc[bea.regions.level < 2]
  for region_key, region_val in regions_selection.iterrows():
    logger.info(f"{region_key}")

    industries = deepcopy(vec_ind.loc[vec_ind.region == region_key])
    industries.set_index('industry', drop=True, inplace=True)
    industries.drop(['region'], inplace=True, axis=1)

    # RECON2024: supply/demand > 1 arises where national imports are negative
    # (BEA CIF/FOB freight adjustment on 483000, 492000, ...); a regional
    # purchase coefficient above one has no meaning, so it is bounded to [0, 1]
    sd = industries.supplydemand.astype(float)
    if getattr(config, 'clip_supplydemand', True):
      sd = sd.clip(lower=0.0, upper=1.0)
    A = iot.data.A.multiply(sd, axis=0)
    A = A.loc[industries.index]
    A = A[industries.index]

    interuse_ratio = utilities.safe_division(
      industries.intermediate_use, interuse_ref)
    # RECON2024: bounded as in export_state_use_A; unbounded ratios reach ~7
    # where state farm subsidies make GDP negative (111900 in LA, AK, OK, NM)
    rb = getattr(config, 'ratio_bounds', [0.25, 4.0])
    interuse_ratio = interuse_ratio.clip(lower=rb[0], upper=rb[1])
    A = A.multiply(interuse_ratio, axis=1)

    # limiting earnings to be between 0 and gdp
    industries['one'] = 1
    industries['zero'] = 0
    industries['tmp'] = deepcopy(industries['earningsgdp'])
    industries['tmp'] = industries[['tmp', 'one']].min(axis=1)
    industries['tmp'] = industries[['tmp', 'zero']].max(axis=1)
    earningsgdp = industries.tmp
    industries.drop(['one', 'zero', 'tmp'], axis=1, inplace=True)

    A.loc['households'] = utilities.safe_division(
      earningsgdp * industries.gdp, industries.output)

    Ahh = pd.DataFrame(columns=['households'],
                       index=iot.codes.industries.index,
                       data=0.0)
    A = pd.concat([A, Ahh], axis=1)
    del Ahh
    A.loc['households', 'households'] = 0

    A.loc[industries.index, 'households'] = (
      utilities.safe_division(industries.pce, region_val.income))
    A = A.round(config.decimals)
    A.index.name = 'industry'
    filename = config.prefix + region_key + config.suffix
    A.to_csv(dir_ / config.subfolder / filename)

  logger.info("Finished step")
  return None


def export_domestic_demand(task_str, full_config):
  """RECON2024: domestic demand by region and commodity, for the rpc stage.

  The RPC regression is estimated on FAF flows as rpc = m / (m + im_dom): the
  share of demand met from DOMESTIC sources that is met locally. The rpc
  stage therefore needs demand net of international imports:

    V_r(c)   = sum_j B(c,j) r_s(j) x_r(j) + pce + investment + inventory
               + government                (purchasers' prices, $k)
               + margin demand: sum_k V_r(k) mu(k,c) for trade/transport c
    D_r(c)   = phi(c) V_r(c) - M_r(c)       (domestic demand, basic prices)

  B = national Use coefficients, r_s = bounded state/US intermediate-use
  ratio (as in export_state_use_A; counties take their state's r_s, i.e.
  state input recipes), mu = national margin per $ of use,
  M_r = the region's international imports. phi(c) calibrates to the US
  commodity balance so that D_US(c) = X_US(c) - E_US(c): nationally,
  (commodity demand is moved to industries with the market shares D',
  since the vectors are on the industry basis)
  supply / domestic demand = 1 for every commodity."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  sym_config = full_config.create_symmetric
  mar_config = full_config.apply_margins
  logger.info("Domestic demand by region (demand net of international imports)")
  T = dirs.__dict__[sym_config.target.folder]

  sut = operations.Sut.read_excel(T / mar_config.target.file, 'sut')
  iot = operations.Sut.read_excel(T / sym_config.target.files.iot, 'iot', is_iot=True)
  D = iot.data.D
  com = list(D.index); ind = list(D.columns)
  U = sut.data.use_int.reindex(index=com, columns=ind).fillna(0.0).astype(float)
  x_us = sut.codes.industries.loc[ind, 'tot'].astype(float)
  B = U.div(x_us.replace(0, np.nan), axis=1).fillna(0.0)
  use_us = sut.codes.commodities.loc[com, 'use'].astype(float)          # $M, purchasers'
  mt = pd.read_excel(T / mar_config.target.file, sheet_name='margin_trade', index_col=0)
  mx = pd.read_excel(T / mar_config.target.file, sheet_name='margin_transport', index_col=0)
  mar = pd.concat([mt, mx], axis=1).reindex(index=com).fillna(0.0)
  mar = mar.T.groupby(level=0).sum().T                                  # k x margin commodity
  mu = mar.div(use_us.replace(0, np.nan), axis=0).fillna(0.0)          # margin per $ of use

  vec = operations.load_vectors()
  cols = ['output', 'intermediate_use', 'imports', 'exports',
          'pce', 'investment', 'inventory', 'government']
  wide = {c: vec.pivot(index='industry', columns='region', values=c)
          .reindex(ind).fillna(0.0).astype(float) for c in cols}
  regions = list(wide['output'].columns)
  frac_us = wide['intermediate_use']['00000']
  ratio = wide['intermediate_use'].div(frac_us.replace(0, np.nan), axis=0)
  ratio = ratio.where(pd.DataFrame(np.repeat((frac_us.values > 0)[:, None], ratio.shape[1], axis=1),
                                   index=ratio.index, columns=ratio.columns), 1.0).fillna(1.0)
  ratio = ratio.clip(lower=config.ratio_bounds[0], upper=config.ratio_bounds[1])

  # counties use their STATE's input recipes (Mike 2026-09-21): the state's
  # productivity ratio, not the county's own (county VA is itself allocated)
  regs = bea_operations.load_bridged().regions
  lvl = regs.reindex(regions)['level']
  par = regs.reindex(regions)['parent']
  counties = [r for r in regions if lvl.get(r) == 2]
  if getattr(config, 'county_state_recipes', True) and counties:
    ratio = ratio.copy()
    ratio[counties] = ratio.reindex(columns=par.loc[counties].values).values
    logger.info(f"{len(counties)} counties use their state's input recipes")
  inter = pd.DataFrame(B.values @ (ratio.values * wide['output'].values),
                       index=com, columns=regions)
  fin = sum(wide[c].reindex(com).fillna(0.0) for c in
            ['pce', 'investment', 'inventory', 'government'])
  V = inter + fin
  md = pd.DataFrame(mu.values.T @ V.values, index=mu.columns, columns=regions)
  # vectors (output, trade, final demand) are on the INDUSTRY basis of the
  # symmetric table: commodity-based intermediate and margin demand is moved
  # to the producing industries with the market shares D (rows sum to 1)
  V_com = inter.add(md.reindex(com).fillna(0.0), fill_value=0.0)
  V = pd.DataFrame(D.values.T @ V_com.values, index=ind, columns=regions) + fin.reindex(ind).fillna(0.0).values

  X = wide['output'].reindex(ind).fillna(0.0)
  E = wide['exports'].reindex(ind).fillna(0.0)
  M = wide['imports'].reindex(ind).fillna(0.0)
  target = X['00000'] - E['00000'] + M['00000']
  phi = utilities.safe_division(target, V['00000'])
  phi = phi.where(V['00000'] > 0, 0.0)
  lo, hi = config.phi_bounds
  logger.info(f"phi: median {phi[phi > 0].median():.3f}; "
              f"{int(((phi > 0) & ((phi < lo) | (phi > hi))).sum())} outside [{lo}, {hi}]")
  # counties: the vectors place a state's imports in its counties by output
  # (wage) share; for domestic demand the state's imports are placed by the
  # counties' share of the state's use instead (config county_imports_by_use)
  if getattr(config, 'county_imports_by_use', True):
    Vc = V[counties]
    par_c = par.loc[counties].values
    Vsum = Vc.T.groupby(par_c).sum().T                      # industry x state
    share = Vc / Vsum.reindex(columns=par_c).values
    share = share.where(Vsum.reindex(columns=par_c).values > 0, 0.0).fillna(0.0)
    M = M.copy()
    M[counties] = share.values * M.reindex(columns=par_c).values
    logger.info(f"county imports re-placed by use share for {len(counties)} counties")
  Vp = V.mul(phi, axis=0)
  # same cap as export_state_use_A: imports cover at most import_share_cap
  # of a region's use of a commodity (heavy trucks, instruments, turbines...)
  cap = float(getattr(config, 'import_share_cap', 0.95))
  M_eff = np.minimum(M.values, cap * np.clip(Vp.values, 0, None))
  n_cap = int((M.values > M_eff + 0.5).sum())
  logger.info(f"import-share cap {cap}: {n_cap} region-commodity cells capped")
  Dd = Vp - M_eff
  n_neg = int((Dd < 0).values.sum())
  Dd = Dd.clip(lower=0.0)
  logger.info(f"{n_neg} region-commodity cells with negative domestic demand set to 0")

  sd_us = utilities.safe_division(X['00000'] - E['00000'], Dd['00000'])
  logger.info(f"US check supply/domestic demand: min {sd_us[Dd['00000'] > 0].min():.4f}, "
              f"max {sd_us[Dd['00000'] > 0].max():.4f}")

  # RECON2024 item 5 (2026-09-21): write the use-based county imports back to
  # the vectors, so the county `imports` column (results files) is the same
  # split as the one domestic demand uses; states and US are unchanged
  if getattr(config, 'county_imports_by_use', True) and getattr(config, 'write_county_imports', False) and counties:
    vfile = dirs.__dict__[full_config.create_vectors.target.folder] / full_config.create_vectors.target.file
    vv = pd.read_csv(vfile, index_col=None, dtype={'region': str, 'industry': str})
    key = vv.region + '_' + vv.industry
    Mc = M[counties].stack()
    Mc.index = Mc.index.get_level_values(1) + '_' + Mc.index.get_level_values(0)
    new = key.map(Mc)
    sel = new.notna()
    before = vv.loc[sel, 'imports'].astype(float).sum()
    vv.loc[sel, 'imports'] = new[sel].round(0).astype(int)
    vv.to_csv(vfile, index=False)
    logger.info(f"county imports in {vfile.name} replaced by the use-based split: {int(sel.sum())} cells, "
                f"total {before / 1e6:,.1f} -> {vv.loc[sel, 'imports'].sum() / 1e6:,.1f} $bn")
  out = Dd.stack().rename('demand_domestic').reset_index()
  out.columns = ['industry', 'region', 'demand_domestic']
  out['demand_domestic'] = out.demand_domestic.round(config.decimals)
  dir_ = dirs.__dict__[config.target.folder]
  out[['region', 'industry', 'demand_domestic']].to_csv(dir_ / config.target.file, index=False)
  pd.DataFrame({'phi': phi, 'use_us_purch_k': V['00000'], 'target_k': target}).to_csv(
    dir_ / config.target.diagnostics)
  logger.info(f"Saved {len(out)} rows to {config.target.file}")
  logger.info("Finished step")
  return None
