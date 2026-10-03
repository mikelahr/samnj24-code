"""Steps of BEA pipeline."""
from .. import utilities
from pathlib import Path
from recon2024.qcew import operations as qcew_operations
from . import operations
import pdb
import os
import pandas as pd
import numpy as np
from logging import getLogger
from copy import deepcopy
logger = getLogger('root')


def process_structure(task_str, full_config):
  logger.info("Loading BEA data structure (sector, region, info)")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  bea = operations.import_data()

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  with pd.ExcelWriter(dir_ / config.target.file,
                      engine='xlsxwriter',
                      mode='w') as writer:
    for table in ['sector', 'region', 'info']:
      for key, val in bea.__dict__.items():
        bea.__dict__[key].__dict__[table].to_excel(
          writer, sheet_name=key + '_' + table, index=True)

  logger.info("Finished step")
  return None


def process_codes(task_str, full_config):
  logger.info(
    "Loading and editing BEA codes and region/sector bridges")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  dir_ = utilities.INTERNAL_PATH / full_config.stage
  tmp_codes = {}
  for key in config.source.codes.__dict__.keys():
    tmp_codes[key] = pd.read_excel(
      dir_ / config.source.file,
      sheet_name=key,
      index_col='code',
      dtype={'code': str, 'line': int, 'parent': str})
    tmp_codes[key].index = tmp_codes[key].index.astype(str)
  tmp_codes = utilities.DictToObject(tmp_codes)

  bridge = {}
  for key, val in config.source.bridge.__dict__.items():
    bridge[val] = pd.read_excel(
      dir_ / config.source.file,
      sheet_name=key,
      index_col=None,
      dtype=str)
  bridge = utilities.DictToObject(bridge)

  qcew_sut = qcew_operations.QcewToSut()

  logger.info("Rearranging industry codes")

  codes = {}
  for key, val in config.source.codes.__dict__.items():
    tmp_ = deepcopy(tmp_codes.__dict__[key])
    if val not in codes.keys():
      if set(['value', 'parent', 'level']).issubset(set(
          tmp_.columns)):
        tmp_.fillna({'level': -1}, inplace=True)
        tmp_ = tmp_.loc[tmp_.level >= 0]
        tmp_.rename({'line': f'line_{key}', 'value': f'value_{key}'},
                    axis=1, inplace=True)
        tmp_['is_leaf'] = (~ tmp_.index.isin(tmp_.parent.values))
        tmp_['level'] = tmp_['level'].astype(int)

      codes[val] = tmp_
    else:
      if set(['value', 'parent', 'level']).issubset(set(
         tmp_.columns)):
        codes[val][[f'line_{key}', f'value_{key}']] = None
        codes[val][[f'line_{key}', f'value_{key}']] = tmp_.loc[
          codes[val].index, ['line', 'value']]
  codes = utilities.DictToObject(codes)

  logger.info("Editing regions")
  # removing Alaska empty counties
  codes.region.drop(config.remove_region_codes, axis=0, inplace=True)

  # Creating region tree
  codes.region['level'] = int(2)
  codes.region['parent'] = codes.region.index.str[:2] + '000'
  codes.region.loc[
    codes.region.index.str[-3:] == '000', 'level'] = int(1)
  codes.region.loc[
    codes.region.index.str[-3:] == '000', 'parent'] = '00000'
  codes.region.loc[
    codes.region.index == '00000', 'level'] = int(0)
  codes.region.loc[
    codes.region.index == '00000', 'parent'] = None
  codes.region = codes.region.reindex(
    ['parent', 'level', 'title'], axis=1)
  codes.region['level'] = codes.region['level'].astype(int)

  logger.info("Creating region bridge")
  codes.region['qcew'] = [str(int(x)) for x in codes.region.index]
  codes.region.loc['00000', 'qcew'] = 'US000'
  codes.region['no_match'] = (~ codes.region.qcew.isin(
    qcew_sut.codes.area.index))
  codes.region.loc[bridge.region_qcew.region_code.values,
                   'no_match'] = False

  tmp = codes.region.loc[codes.region.no_match]
  logger.info(f"{len(tmp)} BEA regions are not mapped to QCEW")
  if len(tmp) > 0:
    logger.info(f"{tmp}")
  del codes.region['no_match']

  bridge_region_tmp = pd.DataFrame(columns=bridge.region_qcew.columns)
  tmp = codes.region.loc[codes.region.qcew.isin(
    qcew_sut.codes.area.index)]
  bridge_region_tmp['region_code'] = tmp.index
  bridge_region_tmp['region_title'] = tmp.title.values
  bridge_region_tmp['qcew_code'] = tmp.qcew.values
  bridge_region_tmp['qcew_title'] = qcew_sut.codes.area.loc[
    tmp.qcew.values, 'title'].values
  bridge.region_qcew = pd.concat(
    [bridge.region_qcew, bridge_region_tmp], axis=0)
  bridge.region_qcew.index = range(len(bridge.region_qcew))
  del codes.region['qcew']

  tmp = qcew_sut.codes.area.loc[
    (~ qcew_sut.codes.area.index.isin(
      bridge.region_qcew.qcew_code.values))]
  tmp = tmp.loc[~ tmp.index.str[-3:]
                .isin(['996', '997', '998', '999'])]

  logger.info(f"{len(tmp)} QCEW regions are neither mapped to BEA "
              "nor have code ending in 996-999 (special cases)")
  if len(tmp) > 0:
    logger.info(f"{tmp}")

  logger.info("Checking BEA to SUT industry bridge")
  tmp = qcew_sut.sut.loc[
    (~ qcew_sut.sut.index.isin(
      bridge.base_sut.sut_code.values))]
  logger.info(f"{len(tmp)} SUT industries are not mapped to BEA")
  if len(tmp) > 0:
    logger.info(f"{tmp}")

  tmp = codes.base.loc[
    (~ codes.base.index.isin(
      bridge.base_sut.base_code.values))]
  logger.info(f"{len(tmp)} BEA base industries not mapped to SUT")
  if len(tmp) > 0:
    logger.info(f"{tmp}")

  logger.info("Creating county employment (cemp) to other bridge")
  tmp = codes.cemp.loc[codes.cemp.is_leaf]
  tmp0 = tmp.loc[~ tmp.index.isin(codes.other.index)]
  logger.info(f"{len(tmp0)} cemp industries are not mapped to other")
  if len(tmp0) > 0:
    logger.info(f"{tmp0}")
  bridge.cemp_other = pd.DataFrame()
  bridge.cemp_other['cemp_code'] = list(tmp.index)
  bridge.cemp_other['cemp_title'] = tmp.title.values
  bridge.cemp_other['other_code'] = list(tmp.index)
  bridge.cemp_other['other_title'] = codes.other.loc[list(tmp.index),
                                                     'title'].values

  def inspect_bridge(codes, bridge, from_, to_):
    logger.info(f"Inspecting {from_} to {to_} bridge")

    for code_ in [from_, to_]:
      tmp = bridge.__dict__[f"{from_}_{to_}"].loc[
        (~ bridge.__dict__[f"{from_}_{to_}"][
          f"{code_}_code"].isin(
          codes.__dict__[f"{code_}"].index))]
      logger.info(f"{len(tmp)} {code_} from config missing from codes")
      if len(tmp) > 0:
        logger.error(f"{tmp}")

    for code_, alt_ in [[from_, to_], [to_, from_]]:
      tmp0 = codes.__dict__[f"{code_}"].loc[
        codes.__dict__[f"{code_}"].is_leaf]
      tmp1 = tmp0.loc[
        (~ tmp0.index.isin(
          bridge.__dict__[f"{from_}_{to_}"][
            f"{code_}_code"]))]
      tmp2 = tmp1.loc[
        (~ tmp1.index.isin(
          codes.__dict__[f"{alt_}"].index))]

      logger.info(f"{len(tmp2)} {code_} not present in config missing "
                  f"from {alt_}")
      if len(tmp2) > 0:
        logger.error(f"{tmp2}")

    tmp0 = codes.__dict__[f"{from_}"].loc[
      codes.__dict__[f"{from_}"].is_leaf]
    tmp1 = tmp0.loc[
      (~ tmp0.index.isin(
        bridge.__dict__[f"{from_}_{to_}"][
          f"{from_}_code"]))]

    tmp3 = pd.DataFrame()
    tmp3[f'{from_}_code'] = list(tmp1.index)
    tmp3[f'{from_}_title'] = tmp1.title.values
    tmp3[f'{to_}_code'] = list(tmp1.index)
    tmp3[f'{to_}_title'] = codes.__dict__[f"{to_}"].loc[
      list(tmp1.index), 'title'].values

    bridge.__dict__[f"{from_}_{to_}"] = pd.concat([
      bridge.__dict__[f"{from_}_{to_}"], tmp3], axis=0)
    bridge.__dict__[f"{from_}_{to_}"].index = range(len(
      bridge.__dict__[f"{from_}_{to_}"]))

    return bridge

  codes.base['is_leaf'] = True
  bridge = inspect_bridge(codes, bridge, 'other', 'base')
  bridge = inspect_bridge(codes, bridge, 'gdp', 'base')

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.codes.folder]

  @utilities.open_excel(dir_ / config.target.codes.file)
  def save_codes(writer, codes, config):
    for key in config.target.codes.sheets:
      logger.debug(f"Saving codes '{key}'")
      codes.__dict__[key].to_excel(writer, sheet_name=key, index=True)

  save_codes('_', codes, config)

  @utilities.open_excel(dir_ / config.target.bridge.file)
  def save_bridge(writer, bridge, config):
    for key in config.target.bridge.sheets:
      logger.debug(f"Saving bridge '{key}'")
      bridge.__dict__[key].to_excel(
        writer, sheet_name=key, index=False)

  save_bridge('_', bridge, config)

  logger.info("Finished step")
  return None


def process_data(task_str, full_config):
  logger.info(
    "Loading BEA data: GDP, state/county employment "
    "and earnings/compensation")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  bea = operations.import_data()

  codes, bridge = operations.load_codes_bridge()

  logger.info("Subtracting overseas GDP")
  for table in config.overseas.tables:
    logger.info(f"{table}")
    for overseas, list_ in config.overseas.substitute.__dict__.items():
      if overseas in bea.__dict__[table].data.index:
        for inland in list_:
          if inland in bea.__dict__[table].data.index:
            main_val = bea.__dict__[table].data.loc[
              inland, config.overseas.column]
            sub_val = bea.__dict__[table].data.loc[
              overseas, config.overseas.column]
            bea.__dict__[table].data.loc[
              inland, config.overseas.column] = (
                main_val - sub_val)
            logger.info(
              f"{overseas} = {sub_val} from {inland} = {main_val} ")

  logger.info("Aligning industry codes across datasets")
  for bea_key, bea_val in bea.__dict__.items():
    tmp_data = bea_val.data
    tmp_data = tmp_data.loc[
      tmp_data.region_code.isin(codes.region.index)]
    ind_bridge = deepcopy(codes.__dict__[
      config.sector_bridge.__dict__[bea_key]])
    line_col = f'line_{bea_key}'
    if line_col not in ind_bridge.columns:
      line_col = 'line_' + bea_key.split('_')[0]
    logger.info(
      f'{config.sector_bridge.__dict__[bea_key]}: '
      f'column {line_col} -> {bea_key}')
    tmp_data = tmp_data.loc[
      tmp_data.sector_line.isin(ind_bridge[line_col].values)]

    ind_bridge['code'] = ind_bridge.index
    ind_bridge.index = ind_bridge[line_col]
    tmp_data['sector_code'] = ind_bridge.loc[
      tmp_data.sector_line.values, 'code'].values
    tmp_data.drop('sector_line', axis=1, inplace=True)
    tmp_data.index = (tmp_data['region_code'] + '_' +
                      tmp_data['sector_code'])
    bea.__dict__[bea_key].data = tmp_data

  logger.info("Merging data with simitar classifications")
  bea_combined = {}
  for bea_key, bea_dict in config.combine_tables.__dict__.items():
    source_table = bea.__dict__[
      list(bea_dict.__dict__.values())[0]].data
    bea_tmp = pd.DataFrame(index=source_table.index)
    bea_tmp.index.name = 'code'
    bea_combined[bea_key] = {
      'tree': deepcopy(bea_tmp),
      'value': deepcopy(bea_tmp),
      'flag': deepcopy(bea_tmp)}
    bea_combined[bea_key]['tree']['region'] = source_table[
      'region_code'].astype(str)
    bea_combined[bea_key]['tree']['sector'] = source_table[
      'sector_code']

    for col_name, table_name in bea_dict.__dict__.items():
      source_table = bea.__dict__[table_name].data
      bea_combined[bea_key]['value'][col_name] = source_table['value']
      bea_combined[bea_key]['flag'][col_name] = source_table['flag']

    bea_combined[
      bea_key]['value'] = bea_combined[bea_key]['value'].fillna(0)
    bea_combined[
      bea_key]['flag'] = bea_combined[bea_key]['flag'].fillna(False)
  bea_combined = utilities.DictToObject(bea_combined)

  logger.info("Adding data parent-offspring relations and levels")
  for bea_key, bea_val in bea_combined.__dict__.items():

    bea_dict = config.combine_tables.__dict__[bea_key]
    original_dataset = list(bea_dict.__dict__.values())[0]
    industry_code = config.sector_bridge.__dict__[original_dataset]

    region_parent = codes.region.loc[
      list(bea_val.tree.region), 'parent']
    region_parent.index = bea_val.tree.index
    sector_parent = codes.__dict__[industry_code].loc[
      list(bea_val.tree.sector), 'parent']
    sector_parent.index = bea_val.tree.index
    bea_val.tree['region_parent'] = (
      region_parent + '_' + bea_val.tree.sector)
    bea_val.tree['sector_parent'] = (
      bea_val.tree.region + '_' + sector_parent)
    bea_val.tree.loc[region_parent.isna(), 'region_parent'] = None
    bea_val.tree.loc[sector_parent.isna(), 'sector_parent'] = None

    bea_val.tree['region_level'] = codes.region.loc[
      bea_val.tree.region.values, 'level'].values
    bea_val.tree['sector_level'] = codes.__dict__[industry_code].loc[
      bea_val.tree.sector.values, 'level'].values

    bea_combined.__dict__[bea_key].tree = bea_val.tree

  logger.info("Rescaling total GDP from million to thousand dollars")
  bea_combined.gdp.value[
    'total'] = bea_combined.gdp.value['total'] * 1000

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  for dataset, file in config.target.files.__dict__.items():
    df = bea_combined.__dict__[dataset].tree
    for table in ['value', 'flag']:
      source_table = bea_combined.__dict__[dataset].__dict__[table]
      for column in source_table.columns:
        df[f'{table}_{column}'] = source_table[column]
    df.to_csv(dir_ / file, index=True)

  logger.info("Finished step")
  return None


def balance_gdp(task_str, full_config):
  logger.info(
    "Estimating missing records in BEA GDP and balancing")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  codes, bridge = operations.load_codes_bridge()
  bea = operations.load_data()

  data = deepcopy(bea.gdp)

  def bottom_up_correction_gdp(data):
    logger.info(
      "Bottom-up correction to avoid negative initial estimates, "
      "including sum of components addition")
    for column in data.value.columns:
      data.value[column] = qcew_operations.bottom_up_correction(
        source=data.value[column],
        tree=data.tree,
        dim0='region',
        dim1='sector',
        keep_max=True)

    data.value['alt'] = data.value[[
      'nettax', 'wages', 'surplus']].sum(axis=1)
    data.value['total'] = data.value[['total', 'alt']].max(axis=1)
    del data.value['alt']
    column = 'total'

    data.value[column] = qcew_operations.bottom_up_correction(
      source=data.value[column],
      tree=data.tree,
      dim0='region',
      dim1='sector',
      keep_max=True)

    return data

  logger.info("Initial bottom-up correction")
  data = bottom_up_correction_gdp(data)

  logger.info("Filling in disclosed zeros as differences")
  #  Note: this part assumes there are 3 components of GDP
  tmp = ((data.value == 0) * (data.flag == False)).sum(axis=1)
  index = tmp.loc[tmp > 0].index
  value = deepcopy(data.value.loc[index]).astype(int)
  flag = deepcopy(data.flag.loc[index])
  tmp = ((value > 0).sum(1) == 3) * ((flag == False).sum(1) == 4)
  index = tmp.loc[tmp].index
  value = value.loc[value.index.isin(index)]

  logger.info(f"Found {len(value)} instances to edit difference.")
  if len(value) > 0:
    for index, row in value.iterrows():
      tmp = (row == 0)
      column = tmp.loc[tmp].index[0]
      if column != 'total':
        data.value.loc[index, column] = 2*row.total - row.sum()
      else:
        data.value.loc[index, column] = row.sum()

    logger.info("Follow-up bottom-up correction")
    data = bottom_up_correction_gdp(data)

  logger.info("Setting initial estimates")
  for column in data.value.columns:
    if (column == 'total'):  # there were two instances, handled above
      continue
    data.value[column] = qcew_operations.set_initial(
      quant=data.value[column],
      proxy=data.value.total,
      nondisclosed=data.flag[column],
      tree=data.tree,
      dim0='region',
      dim1='sector',)

  logger.info("Creating variable vector and aggregation matrix")
  t_combined = pd.DataFrame()
  g_combined = pd.DataFrame()
  k_index = []
  for var_pos, column in enumerate(data.value.columns):
    logger.info(f"component: {column}")
    t = pd.DataFrame(index=data.value.index,)
    t['pos'] = range(len(t))
    t['val'] = data.value[column]
    t.loc[t.val == 0, 'val'] = int(1)
    t['flag'] = int(1)
    t.loc[data.flag[column], 'flag'] = int(2)

    g = pd.DataFrame(columns=['row', 'col', 'val'])
    for pos, dim in enumerate(['region', 'sector']):
      logger.info(f"dimension: {dim}")

      # disaggregate values
      g0 = pd.DataFrame()
      g0 = deepcopy(
        data.tree.loc[data.tree[dim + '_parent'].notna()])
      g0['current_pos'] = t.loc[g0.index, 'pos']
      g0['parent_pos'] = t.loc[g0[
        dim + '_parent'].values, 'pos'].values
      g1 = pd.DataFrame()
      g1['row'] = g0.parent_pos + pos * len(t)
      g1['col'] = g0.current_pos
      g1['val'] = int(1)
      g = pd.concat([g, g1], axis=0)

      # aggregate values
      g2 = pd.Series(index=g0[dim + '_parent'].unique())
      g2 = t.loc[g2.index, 'pos'].astype(int)
      g3 = pd.DataFrame(index=g2.index)
      g3['row'] = g2 + pos * len(t)
      g3['col'] = g2
      g3['val'] = int(-1)
      g = pd.concat([g, g3], axis=0)
      k_index = k_index + list(column + '_' + dim + '_' + t.index)

    g.row = g.row + 2 * var_pos * len(t)  # two dims
    g.col = g.col + var_pos * len(t)

    # cross-component constraint
    g4 = t['pos'].astype(int)
    g5 = pd.DataFrame(index=g4.index)
    g5['row'] = g4 + 2 * len(data.value.columns) * len(t)
    g5['col'] = g4 + var_pos * len(t)

    if column == 'total':
      g5['val'] = int(-1)
    else:
      g5['val'] = int(1)
    g = pd.concat([g, g5], axis=0)
    del g0, g1, g2, g3, g4, g5, pos, dim

    t.index = f"{column}_" + t.index
    g_combined = pd.concat([g_combined, g], axis=0)
    t_combined = pd.concat([t_combined, t], axis=0)

  k_index = k_index + list('cross_' + t.index)
  g = g_combined
  t = t_combined
  del g_combined, t_combined
  g = g.astype({'row': int, 'col': int, 'val': float})
  g.index = range(len(g))
  #  grand total label hardcoded
  t.loc['total_00000_T', 'flag'] = int(0)

  logger.info("Balance")
  bal = utilities.Balance.from_array(
    t, g, config.params, k_index)
  del g, t

  err, val = bal.get_error_value(error='source', value='source')
  bal.run()
  err_tmp, val_tmp = bal.get_error_value()
  err['target'] = err_tmp.error
  val['target'] = val_tmp.value
  err.sort_values('target', inplace=True)
  val['dif'] = val.target - val.source
  val['rel'] = 100 * (val.dif / val.source)
  val.sort_values('dif', inplace=True)

  logger.info("Statistics on balancing displacement:")
  for label in ['nondisclosed', 'disclosed']:
    if label == 'nondisclosed':
      val_condition = (val.flag == 2)
    else:
      val_condition = (val.flag != 2)
    val_tmp = deepcopy(val.loc[val_condition])
    val_max = abs(val_tmp.target.max())
    logger.info(f"{label}: {len(val_tmp)} records")
    logger.info(
      "Min abs, max abs, n records, max abs rel displacement")
    minval = 0.0
    maxval = config.params.eps
    while minval < val_max:
      val_curr = val_tmp.loc[
        (val_tmp.target > minval)
        & (val_tmp.target <= maxval)]
      logger.info(
        f"{minval:5.1e}, {maxval:5.1e}, {len(val_curr):5}, "
        f"{abs(val_curr.rel).max():5.2f}%")
      minval = maxval
      maxval = 10 * maxval
  del label, val_condition, val_tmp, val_max, minval, maxval, val_curr

  logger.info("Reshaping results and rescaling integers to match")
  val['column'] = [x.split('_')[0] for x in val.index]
  val['row'] = ['_'.join(x.split('_')[1:]) for x in val.index]

  df = pd.DataFrame(index=data.value.index,
                    columns=data.value.columns,
                    data=int(0))
  for column in df.columns:
    tmp = val.loc[val.column == column]
    tmp.set_index('row', inplace=True)
    df[column] = tmp.loc[df.index, 'target'].astype(int)
    df[column] = qcew_operations.bottom_up_correction(
      source=df[column],
      tree=data.tree,
      dim0='region',
      dim1='sector')
  df['total'] = df[['nettax', 'wages', 'surplus']].sum(axis=1)

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  df.to_csv(dir_ / config.target.file, index=True)

  logger.info("Finished step")
  return None


def process_proxies(task_str, full_config):
  logger.info(
    'Aggregating SUT-industry QCEW employment/wages to BEA format')
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info('Loading data')
  bea = operations.BeaToSut(load_proxy=True)

  logger.info('Creating region and industry bridges')
  base_sut = bea.bridge.base_sut.matrix
  other_base = bea.bridge.other_base.matrix
  other_sut = other_base.dot(base_sut)
  other_sut_norm = bea.normalize_columns(other_sut)
  logger.info(
    "Complete industry bridge; "
    f"{other_sut.shape} = {other_base.shape} x {base_sut.shape}")
  bea.show_coverage(other_sut_norm)

  region_qcew = bea.bridge.region_qcew.matrix
  region_qcew_norm = bea.normalize_columns(region_qcew)
  logger.info(
    f"Complete region bridge: {region_qcew.shape}")
  bea.show_coverage(region_qcew_norm)
  del base_sut, other_base, other_sut, region_qcew

  logger.info("Identify one-to-many proxies that need resetting")
  tmp = ((other_sut_norm != 0) * (other_sut_norm < 1)).sum(1)
  reset_other = tmp.loc[tmp != 0].index
  logger.info(f"sector {len(reset_other)}:\n{reset_other}")
  tmp = ((region_qcew_norm != 0) * (region_qcew_norm < 1)).sum(1)
  reset_region = tmp.loc[tmp != 0].index
  logger.info(f"sector {len(reset_region)}:\n{reset_region}")

  logger.info('Pivot proxy data, apply bridges and melt again')
  values_list = config.columns
  proxies = pd.DataFrame(index=bea.data.county_other.value.index,
                         columns=values_list,
                         data=int(0))
  for value in values_list:
    logger.info(f'Proxy for {value}')
    df_tmp = deepcopy(bea.data.proxy)
    df_tmp = df_tmp.pivot(
      index='area', columns='industry', values=value)
    proxy = pd.DataFrame(index=bea.codes.qcew.index,
                         columns=bea.codes.sut.index,
                         data=float(0))
    proxy.loc[df_tmp.index, df_tmp.columns] = df_tmp
    del df_tmp
    logger.info(f'Shape in QCEW/SUT categories: {proxy.shape}')

    logger.info('Multiplying')
    proxy = region_qcew_norm.dot(proxy)
    proxy = proxy.dot(other_sut_norm.T)
    logger.info(f'Shape in BEA categories (reg/ind): {proxy.shape}')

    proxy['region'] = proxy.index
    proxy = proxy.melt(id_vars='region', value_vars=proxy.columns,
                       var_name='sector')
    proxy.index = proxy.region + '_' + proxy.sector
    # removing proxies one-to-many proxies
    if len(reset_other) > 0:
      proxy.loc[proxy.sector.isin(reset_other), 'value'] = 0
    if len(reset_region) > 0:
      proxy.loc[proxy.sector.isin(reset_region), 'value'] = 0

    proxy['value'] = qcew_operations.bottom_up_correction(
      source=proxy['value'],
      tree=bea.data.county_other.tree,
      dim0='region',
      dim1='sector')

    proxies.loc[proxy.index, value] = proxy['value'].astype(int)
  proxies = proxies.astype(int)

  logger.info(
    "In QCEW there were 'unclassified' counties which are ignored."
    " So state and national totals of proxies in BEA classification "
    "are lower than in QCEW classification.")

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  proxies.index.name = 'code'
  proxies.to_csv(dir_ / config.target.file, index=True)

  logger.info("Finished step")
  return None


def process_qcew(task_str, full_config):
  logger.info(
    'Converting QCEW data region classification only')
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info('Loading data')
  bea = operations.BeaToSut(load_proxy=True)

  logger.info('Creating region bridge')
  region_qcew = bea.bridge.region_qcew.matrix
  region_qcew_norm = bea.normalize_columns(region_qcew)
  logger.info(
    f"Complete region bridge: {region_qcew.shape}")
  bea.show_coverage(region_qcew_norm)
  del region_qcew

  logger.info("Identify one-to-many proxies that need resetting")
  tmp = ((region_qcew_norm != 0) * (region_qcew_norm < 1)).sum(1)
  reset_region = tmp.loc[tmp != 0].index
  logger.info(f"{len(reset_region)} sectors:\n{reset_region}")

  logger.info('Pivot QCEW table, apply bridge and melt again')
  qcew = pd.DataFrame(index=[],
                      columns=config.columns,
                      data=int(0))
  for value in config.columns:
    logger.info(f'Applying region bridge variable {value}')
    df_tmp = deepcopy(bea.data.proxy)
    df_tmp = df_tmp.pivot(
      index='area', columns='industry', values=value)
    proxy = pd.DataFrame(index=bea.codes.qcew.index,
                         columns=bea.codes.sut.index,
                         data=float(0))
    proxy.loc[df_tmp.index, df_tmp.columns] = df_tmp
    del df_tmp
    logger.info(f'Shape in QCEW/SUT categories: {proxy.shape}')

    logger.info('Multiplying')
    proxy = region_qcew_norm.dot(proxy)
    logger.info(f'Shape in BEA categories (reg/ind): {proxy.shape}')

    proxy['region'] = proxy.index
    proxy = proxy.melt(id_vars='region', value_vars=proxy.columns,
                       var_name='sector')
    proxy.index = proxy.region + '_' + proxy.sector
    # removing proxies one-to-many proxies
    if len(reset_region) > 0:
      proxy.loc[proxy.sector.isin(reset_region), 'value'] = 0

    if len(qcew) == 0:
      qcew[value] = proxy['value'].astype(int)
    qcew.loc[proxy.index, value] = proxy['value'].astype(int)
  qcew = qcew.astype(int)

  logger.info(
    "In QCEW there were 'unclassified' counties which are ignored."
    " So state and national totals of proxies in BEA classification "
    "are lower than in QCEW classification.")

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  qcew.index.name = 'code'   # RECON2024: load_qcew reads index_col='code'
  qcew.to_csv(dir_ / config.target.file, index=True)

  logger.info("Finished step")
  return None


def balance_other(task_str, full_config):
  logger.info(
    "Estimating and balancing BEA earnings and compensation")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  codes, bridge = operations.load_codes_bridge()
  data = operations.load_data().county_other
  data.value['proxy'] = operations.load_proxies().proxy

  logger.info("Setting sector '516' as disclosed")
  data.flag.loc[
    data.tree.sector == config.set_nondisclosed] = False

  for column, proxy in [
      ['compensation', 'proxy'],
      ['earnings', 'compensation']]:
    logger.info(f"Handling variable {column} with proxy {proxy}")

    n_sector = max(data.tree.sector_level) + 1
    n_region = max(data.tree.region_level) + 1
    for k_sector in range(n_sector):
      for k_region in range(n_region):

        trunc = ((data.tree.sector_level <= k_sector)
                 & (data.tree.region_level <= k_region))

        logger.info(
          f"sector level {k_sector}; "
          f"region level {k_region}; "
          f"{trunc.sum()} elements")

        if trunc.sum() < 2:
          logger.info("Skipping\n")
          continue

        quant = operations.balance_county_other(
          quant=deepcopy(data.value[column].loc[trunc]),
          proxy=deepcopy(data.value[proxy].loc[trunc]),
          nondisclosed=deepcopy(data.flag[column].loc[trunc]),
          tree=deepcopy(data.tree.loc[trunc]),
          params=config.params,
          canopy_only=True)

        data.value.loc[quant.index, column] = quant
        logger.info("\n")

  data.value = data.value.round(0).astype(int)
  data.value.drop('proxy', axis=1, inplace=True)

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  data.value.to_csv(dir_ / config.target.file, index=True)

  logger.info("Finished step")
  return None



def import_gdp_filled(task_str, full_config):
  """RECON2024: replaces balance_gdp. The state GDP components with (D)/(L)
  cells filled by the SAGDP MAP solver (raw/SAGDP/filled) are written in the
  bea_gdp_balanced.csv layout (code index; nettax, wages, surplus, total;
  integer thousands of dollars), with the same bottom-up rounding correction
  balance_gdp applied."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year = full_config.year

  codes, _ = operations.load_codes_bridge()
  data = operations.load_data().gdp
  line_of = codes.gdp['line_gdp'].astype(int)

  logger.info("Loading filled SAGDP components")
  filled = pd.read_csv(dirs.__dict__[config.source.folder] / config.source.file,
                       dtype={'region': str, 'line': str})
  filled = filled.loc[filled.year == year]
  filled['line'] = filled.line.astype(int)
  filled.index = filled.region + '_' + filled.line.astype(str)

  key = data.tree.region + '_' + line_of.loc[data.tree.sector].astype(str).values
  hit = key.isin(filled.index)
  logger.info(f"{hit.sum()} of {len(key)} GDP cells matched in the filled file")
  if (~hit).sum() > 0:
    logger.info(f"Unmatched (kept as published):\n{data.tree.loc[~hit, ['region', 'sector']]}")

  df = pd.DataFrame(index=data.value.index, columns=['nettax', 'wages', 'surplus'],
                    data=0.0)
  df['nettax'] = data.value['nettax'].astype(float)
  df['wages'] = data.value['wages'].astype(float)
  df['surplus'] = data.value['surplus'].astype(float)
  src = filled.loc[key[hit].values]
  df.loc[hit.values, 'wages'] = src[config.source.columns.wages].values
  df.loc[hit.values, 'surplus'] = src[config.source.columns.surplus].values
  tax = src[config.source.columns.nettax]
  ok = tax.notna().values
  idx = df.index[hit.values]
  df.loc[idx[ok], 'nettax'] = tax.values[ok]

  # published cells must be reproduced
  for column in ['wages', 'surplus']:
    pub = ~data.flag[column].values
    dev = (df.loc[pub, column] - data.value.loc[pub, column]).abs().max()
    logger.info(f"{column}: max |filled - published| on published cells = {dev:.0f}")

  df = df.round(0).astype(int)
  for column in df.columns:
    df[column] = qcew_operations.bottom_up_correction(
      source=df[column], tree=data.tree, dim0='region', dim1='sector')
  df['total'] = df[['nettax', 'wages', 'surplus']].sum(axis=1)
  gap = (df['total'] - data.value['total']).abs()
  logger.info(f"|components - published total|: max {gap.max()}, "
              f"US {df.loc['00000_T', 'total'] - data.value.loc['00000_T', 'total']}")

  dir_ = dirs.__dict__[config.target.folder]
  df.to_csv(dir_ / config.target.file, index=True)
  logger.info("Finished step")
  return None


def import_other_filled(task_str, full_config):
  """RECON2024: replaces balance_other. County earnings and compensation with
  (D) cells filled by the CAINC MAP solver (raw/CAINC/filled), keyed exactly
  as bea_other.csv (code = region_sector), written as bea_other_balanced.csv
  (earnings, compensation; integer thousands of dollars)."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year = full_config.year

  data = operations.load_data().county_other
  filled = pd.read_parquet(dirs.__dict__[config.source.folder] / config.source.file)
  filled = filled.loc[filled.year == year].set_index('code')
  hit = data.tree.index.isin(filled.index)
  logger.info(f"{hit.sum()} of {len(data.tree)} county cells matched in the filled file")
  if (~hit).sum() > 0:
    logger.info(f"Unmatched (kept as published):\n{data.tree.loc[~hit].head(20)}")

  df = pd.DataFrame(index=data.value.index)
  for column, src_col in config.source.columns.__dict__.items():
    df[column] = data.value[column].astype(float)
    df.loc[hit, column] = filled.loc[data.tree.index[hit], src_col].values
    pub = ~data.flag[column].values
    dev = (df.loc[pub, column] - data.value.loc[pub, column]).abs().max()
    logger.info(f"{column}: max |filled - published| on published cells = {dev:.0f}")
  df = df.round(0).astype(int)
  for column in df.columns:
    df[column] = qcew_operations.bottom_up_correction(
      source=df[column], tree=data.tree, dim0='region', dim1='sector')

  dir_ = dirs.__dict__[config.target.folder]
  df.to_csv(dir_ / config.target.file, index=True)
  logger.info("Finished step")
  return None


def balance_employment(task_str, full_config):
  logger.info(
    "Estimating and balancing BEA state and county employment")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  codes, bridge = operations.load_codes_bridge()
  full_data = operations.load_data()
  state_data = full_data.state_emp
  county_data = full_data.county_emp
  proxy = full_data.county_other.balanced.compensation

  tree = utilities.DictToObject(
    {'state': state_data.tree, 'county': county_data.tree})
  data = utilities.DictToObject(
    {'state': pd.DataFrame(), 'county': pd.DataFrame()})
  data.state['nondisclosed'] = state_data.flag.total
  data.county['nondisclosed'] = county_data.flag.total
  data.state['quant'] = state_data.value.total
  data.county['quant'] = county_data.value.total
  data.state['proxy'] = proxy.loc[data.state.index]
  data.county['proxy'] = proxy.loc[data.county.index]
  del state_data, county_data, full_data, proxy

  for table in ['state', 'county']:
    data.__dict__[table].loc[
      tree.__dict__[table].sector == config.set_nondisclosed,
      'nondisclosed'] = False

  n_sector = max(tree.state.sector_level) + 1
  n_region = max(tree.county.region_level) + 1
  for k_sector in range(n_sector):
    for k_region in range(n_region):
      trunc_sum = 0
      trunc = {}
      for table in ['state', 'county']:
        trunc[table] = (
          (tree.__dict__[table].sector_level <= k_sector)
          & (tree.__dict__[table].region_level <= k_region))
        trunc_sum = trunc_sum + data.__dict__[table].loc[
          trunc[table], 'nondisclosed'].sum()
      logger.info(
        f"sector level {k_sector}; "
        f"region level {k_region}; "
        f"{trunc_sum} nondisclosed elements")

      if trunc_sum < 1:
        logger.info("Skipping\n")
        continue

      tree_tmp = deepcopy(tree)
      data_tmp = deepcopy(data)
      for table in ['state', 'county']:
        data_tmp.__dict__[table] = data_tmp.__dict__[table].loc[
          trunc[table]]
        tree_tmp.__dict__[table] = tree_tmp.__dict__[table].loc[
          trunc[table]]

      data_tmp = operations.balance_employment(
        data_tmp,
        tree_tmp,
        config.params,
        k_region,
        k_sector)

      for table in ['state', 'county']:
        data.__dict__[table].loc[
          data_tmp.__dict__[table].index,
          'quant'] = data_tmp.__dict__[table].quant
      logger.info("\n")

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  for table in ['state', 'county']:
    df = pd.DataFrame()
    df['total'] = (
      data.__dict__[table].quant.round(0).astype(int))
    df.to_csv(
      dir_ / config.target.files.__dict__[table], index=True)

  logger.info("Finished step")
  return None



def assemble_employment(task_str, full_config):
  """RECON2024: replaces balance_employment. BEA's SAEMP25N / CAEMP25N end in
  2022, so {year} total employment by industry is assembled:

    total = covered (QCEW, all ownerships, disclosure-filled, bridged to BEA
            sectors = bea_proxies employment)
          + uncovered wage & salary  (raw/Proprietors/uncovered_ws_state_sector,
            state x 20 private lines; to counties by covered share)
          + proprietors               (raw/Proprietors/FINAL_prop_jobs_county_sector)
    farm  = farm panel (raw/Proprietors/farm_panel), NASS-carried from 2022
    GC/GSL = covered + the 2022 published-minus-covered residual x covered growth
    GM    = 2022 CAEMP25N military x county military compensation growth (CAINC6N)

  State detail below the 20 lines (the 'other' tree, level 4-5) gets the
  non-covered part in the 2022 proportions of (published - covered).
  Output: bea_emp_s_balanced.csv / bea_emp_c_balanced.csv (code,total)."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year = full_config.year
  prev = config.previous_year

  codes, bridge = operations.load_codes_bridge()
  data = operations.load_data()
  other = codes.other
  cemp = codes.cemp

  # ---------------- covered employment, this year and previous, by ownership
  # straight from the filled QCEW panel: own 5 for the private NAICS lines,
  # own 1 for GC, own 2+3 for GSL. State rows carry the unknown-county jobs.
  dir_t = dirs.__dict__[config.target.folder]
  leaves_all = other.loc[other.is_leaf].index.tolist()
  cleaves_all = cemp.loc[cemp.is_leaf].index.tolist()

  def naics_list(code):
    out = []
    code = config.code_aliases.__dict__.get(str(code), str(code)) \
      if hasattr(config, 'code_aliases') else str(code)
    for part in str(code).split(','):
      part = part.strip()
      if '-' in part and part not in config.qcew_range_codes:
        lo, hi = part.split('-')
        out += [str(x) for x in range(int(lo), int(hi) + 1)]
      else:
        out.append(part)
    return out

  def covered_from_panel(yr, leaves):
    qdir = dirs.__dict__[config.qcew_panel.folder]
    src = pd.read_parquet(qdir / (config.qcew_panel.source + str(yr) + config.qcew_panel.suffix))
    bal = pd.read_parquet(qdir / (config.qcew_panel.balanced + str(yr) + config.qcew_panel.suffix))
    src['emp'] = bal['employment'].values
    src['area'] = src.area.astype(str)
    src['region'] = ['00000' if a == 'US000' else a.zfill(5) for a in src.area]
    src = src.loc[~src.region.str[-3:].isin(['996', '997', '998', '999'])]
    piv = src.pivot_table(index='region', columns='oind', values='emp', aggfunc='sum').fillna(0.0)
    # county-sum basis: the uncovered W&S series was pinned in 2022 to a
    # covered universe without the unknown-county rows, so states are the sum
    # of their counties and the nation the sum of the states (the unknown-
    # county jobs are thereby part of "uncovered" and reach counties with it)
    cty = piv.loc[piv.index.str[-3:] != '000']
    st_sum = cty.groupby(cty.index.str[:2] + '000').sum()
    piv.loc[st_sum.index] = st_sum.values
    piv.loc['00000'] = st_sum.sum(axis=0).values
    out = pd.DataFrame(index=piv.index, columns=leaves, data=0.0)
    for k in leaves:
      if k == 'GC':
        cols = ['1_10']
      elif k == 'GSL':
        cols = ['2_10', '3_10']
      elif k == 'GM':
        cols = []
      else:
        cols = ['5_' + n for n in naics_list(k)]
      cols = [c for c in cols if c in piv.columns]
      if cols:
        out[k] = piv[cols].sum(axis=1)
    ser = out.stack(); ser.index = [f"{r}_{k}" for r, k in ser.index]
    return ser

  all_codes = leaves_all + [c for c in cleaves_all if c not in leaves_all]
  cov24 = covered_from_panel(year, all_codes)
  cov22 = covered_from_panel(prev, all_codes)

  # NAICS 2022 recoded retail (44-45) and information (51) below the line
  # level, so the 2017-NAICS leaves of the SAEMP tree cannot be summed from
  # the panel there. Where the leaves recover less than `leaf_cover_min` of
  # the line's own covered jobs, the line total is split over its leaves by
  # the state's 2022 published employment shares.
  def lc_anc(code):
    c = code
    while other.loc[c, 'level'] > 3:
      c = other.loc[c, 'parent']
    return c
  pub22 = data.state_emp.value['total'].where(~data.state_emp.flag['total'])
  for ser in (cov24, cov22):
    regions_in = sorted(set(i.split('_', 1)[0] for i in ser.index))
    for L in [c for c in other.loc[other.level == 3].index if c not in ('GC', 'GM', 'GSL')]:
      kids = [k for k in leaves_all if lc_anc(k) == L and k != L]
      if not kids:
        continue
      us_leaf = sum(ser.get(f"00000_{k}", 0.0) for k in kids); us_line = ser.get(f"00000_{L}", 0.0)
      if us_line <= 0 or us_leaf >= config.leaf_cover_min * us_line:
        continue
      logger.info(f"line {L}: leaves recover {us_leaf / us_line:.1%} of covered jobs; "
                  f"splitting the line by 2022 published shares")
      for r in regions_in:
        st = r[:2] + '000' if r != '00000' else '00000'
        w = np.array([pub22.get(f"{st}_{k}", np.nan) for k in kids], dtype=float)
        if np.isnan(w).all() or np.nansum(w) <= 0:
          w = np.array([ser.get(f"{r}_{k}", 0.0) for k in kids], dtype=float)
        w = np.nan_to_num(w); w = w / w.sum() if w.sum() > 0 else np.ones(len(kids)) / len(kids)
        tot = ser.get(f"{r}_{L}", 0.0)
        for k, wk in zip(kids, w):
          ser.loc[f"{r}_{k}"] = tot * wk
  for nm, ser in [(year, cov24), (prev, cov22)]:
    us = ser.loc[ser.index.str.startswith('00000_')]
    logger.info(f"covered jobs {nm}: US private leaves {us.loc[[i for i in us.index if i.split('_', 1)[1] in leaves_all and i.split('_', 1)[1] not in ('GC', 'GM', 'GSL')]].sum():,.0f}; "
                f"GC {us.get('00000_GC', 0):,.0f}; GSL {us.get('00000_GSL', 0):,.0f}")

  def lc_ancestor(code):
    """the line-code sector (level 3, or 111-112) above a leaf of 'other'."""
    c = code
    while other.loc[c, 'level'] > 3:
      c = other.loc[c, 'parent']
    return c

  leaves = other.loc[other.is_leaf].index.tolist()
  line_of = other['line_earnings'].astype(int)
  lc_of = {k: lc_ancestor(k) for k in leaves}
  lc_sectors = sorted(set(lc_of.values()), key=lambda x: line_of[x])
  private_lc = [k for k in lc_sectors if k not in ('GC', 'GM', 'GSL', '111-112')]
  logger.info(f"{len(leaves)} leaf sectors under {len(lc_sectors)} line-code sectors")

  # ---------------- external series
  P = dirs.__dict__[config.proprietors.folder]
  unc = pd.read_csv(P / config.proprietors.uncovered, dtype={'st': str, 'lc': str})
  unc = unc.loc[unc.year == year]
  unc['region'] = unc.st.str.zfill(2) + '000'
  unc['sector'] = [other.index[line_of == int(x)][0] for x in unc.lc]
  U = unc.pivot_table(index='region', columns='sector', values='u', aggfunc='sum').fillna(0.0)

  prop_c = pd.read_csv(P / config.proprietors.county, dtype={'GeoFIPS': str, 'lc': str})
  prop_c = prop_c.loc[prop_c.year == year]
  prop_c['sector'] = [other.index[line_of == int(x)][0] for x in prop_c.lc]
  PC = prop_c.pivot_table(index='GeoFIPS', columns='sector', values='prop_jobs', aggfunc='sum').fillna(0.0)
  prop_all = pd.read_csv(P / config.proprietors.state, dtype={'state': str, 'lc': str})
  prop_all['sector'] = [other.index[line_of == int(x)][0] for x in prop_all.lc]
  PS = prop_all.loc[prop_all.year == year].pivot_table(
    index='state', columns='sector', values='prop_jobs', aggfunc='sum').fillna(0.0)
  # a state whose {year} proprietors collapsed against the year before (Connecticut
  # in the FINAL file: the county-to-planning-region switch) takes the previous
  # year's pattern; the SAINC4 control below sets its level
  PS_prev = prop_all.loc[prop_all.year == year - 1].pivot_table(
    index='state', columns='sector', values='prop_jobs', aggfunc='sum').fillna(0.0)
  for st in PS.index:
    if st in PS_prev.index and PS.loc[st].sum() < config.prop_collapse_ratio * PS_prev.loc[st].sum():
      logger.warning(f"{st}: state proprietors {year} = {PS.loc[st].sum():,.0f} vs "
                     f"{year - 1} = {PS_prev.loc[st].sum():,.0f}; using the {year - 1} pattern")
      PS.loc[st] = PS_prev.loc[st].reindex(PS.columns).fillna(0.0).values

  farm = pd.read_csv(P / config.farm.county, dtype={'fips': str})
  farm = farm.loc[farm.year == year].set_index('fips')['farm_employment']
  farm_ct = pd.read_csv(P / config.farm.county_ct, dtype={'fips': str})
  farm_ct = farm_ct.loc[farm_ct.year == year].set_index('fips')['farm_employment']
  farm = pd.concat([farm.loc[~farm.index.str.startswith('09')], farm_ct])
  farm_s = pd.read_csv(P / config.farm.state, dtype={'st': str})
  farm_s['fips'] = farm_s.st.str.zfill(2) + '000'
  farm_s = farm_s.loc[farm_s.year == year].set_index('fips')['farm_employment']

  # military compensation growth (county), CAINC6N line 2002
  mil = pd.read_csv(dirs.__dict__[config.military.folder] / config.military.file,
                    dtype=str, encoding='latin-1')
  mil['GeoFIPS'] = mil.GeoFIPS.str.strip().str.replace('"', '')
  mil = mil.loc[mil.LineCode == str(config.military.line)].set_index('GeoFIPS')
  mil_g = (pd.to_numeric(mil[str(year)], errors='coerce')
           / pd.to_numeric(mil[str(prev)], errors='coerce')).replace([np.inf], np.nan)

  regions = codes.region
  states = regions.loc[regions.level == 1].index.tolist()
  counties = regions.loc[regions.level == 2].index.tolist()

  # ---------------- helpers
  def get(series, region, sector, default=0.0):
    return float(series.get(f"{region}_{sector}", default))

  def published(dataset, region, sector):
    val = dataset.value['total']; flag = dataset.flag['total']
    key = f"{region}_{sector}"
    if key in val.index and not flag.loc[key]:
      return float(val.loc[key])
    return np.nan

  # ---------------- states, all leaves of the 'other' tree
  # three parts are kept apart so the SAINC4 controls can be applied:
  #   S_c covered W&S (QCEW), S_u uncovered W&S, S_p proprietors
  S_c = pd.DataFrame(index=states, columns=leaves, data=0.0)
  S_u = S_c.copy(); S_p = S_c.copy()
  farm_p = pd.read_csv(P / config.farm.state, dtype={'st': str})
  farm_p['fips'] = farm_p.st.str.zfill(2) + '000'
  farm_p = farm_p.loc[farm_p.year == year].set_index('fips')['farm_proprietors_emp']
  # RECON2024 (2026-09-27): the BEA-jobs release gives private nonfarm jobs by state
  # and line (covered + noncovered, held to SAINC4). Where present it sets each
  # state-line total: proprietors from the release's county file, uncovered W&S as
  # the remainder over this panel's covered jobs.
  REL = {}
  if hasattr(config, 'state_sector_release'):
    rel = pd.read_csv(dirs.__dict__[config.state_sector_release.folder] / config.state_sector_release.file,
                      dtype={'st': str})
    rel = rel.loc[rel.year == year]
    for st_, lc_, tot_ in zip(rel.st.str.zfill(2) + '000', rel.lc.astype(int), rel.total):
      sec_ = other.index[line_of == lc_]
      if len(sec_):
        REL[(st_, sec_[0])] = float(tot_)
    logger.info(f"state-sector release: {len(REL)} state x line totals for {year}, "
                f"US {sum(REL.values()):,.0f} private nonfarm jobs")
  U_eff = {}
  for s in states:
    for L in private_lc:
      kids = [k for k in leaves if lc_of[k] == L]
      Nu = float(U.loc[s, L]) if (s in U.index and L in U.columns) else 0.0
      Np = float(PS.loc[s, L]) if (s in PS.index and L in PS.columns) else 0.0
      c24 = np.array([get(cov24, s, k) for k in kids])
      if (s, L) in REL:
        Nu = max(REL[(s, L)] - c24.sum() - Np, 0.0)
        if REL[(s, L)] - c24.sum() < Np:
          Np = max(REL[(s, L)] - c24.sum(), 0.0)
      U_eff[(s, L)] = Nu
      c22 = np.array([get(cov22, s, k) for k in kids])
      t22 = np.array([published(data.state_emp, s, k) for k in kids])
      w = np.where(np.isnan(t22), 0.0, np.maximum(t22 - c22, 0.0))
      if w.sum() <= 0:
        w = c24.copy()
      if w.sum() <= 0:
        w = np.ones(len(kids))
      w = w / w.sum()
      S_c.loc[s, kids] = c24
      S_u.loc[s, kids] = Nu * w
      S_p.loc[s, kids] = Np * w
    # government: covered + carried residual (both W&S)
    for k in ['GC', 'GSL']:
      c24 = get(cov24, s, k); c22 = get(cov22, s, k); t22 = published(data.state_emp, s, k)
      resid = max(t22 - c22, 0.0) if not np.isnan(t22) else 0.0
      S_c.loc[s, k] = c24
      S_u.loc[s, k] = resid * (c24 / c22 if c22 > 0 else 1.0)
    t22 = published(data.state_emp, s, 'GM')
    g = mil_g.get(s, np.nan)
    S_u.loc[s, 'GM'] = (t22 if not np.isnan(t22) else 0.0) * (g if not np.isnan(g) else 1.0)
    # farm: proprietors from the panel, the rest is farm wage & salary
    ftot = float(farm_s.get(s, np.nan)) if s in farm_s.index else \
      (published(data.state_emp, s, '111-112') if not np.isnan(published(data.state_emp, s, '111-112')) else 0.0)
    fprop = float(farm_p.get(s, np.nan)) if s in farm_p.index else 0.0
    fprop = min(max(fprop, 0.0), ftot)
    S_p.loc[s, '111-112'] = fprop
    S_c.loc[s, '111-112'] = get(cov24, s, '111-112')
    S_u.loc[s, '111-112'] = max(ftot - fprop - S_c.loc[s, '111-112'], 0.0)

  # ---------------- SAINC4 controls (BEA state employment survived there,
  # through 2024: 7010 total, 7020 wage & salary, 7040 proprietors)
  if hasattr(config, 'state_control') and not REL:
    sc = config.state_control
    sainc = pd.read_csv(dirs.__dict__[sc.folder] / sc.file, dtype=str, encoding='latin-1')
    sainc['GeoFIPS'] = sainc.GeoFIPS.str.strip().str.replace('"', '')
    ctl = sainc.loc[sainc.LineCode.isin([str(sc.lines.wage_salary), str(sc.lines.proprietors)])]
    ctl = ctl.pivot(index='GeoFIPS', columns='LineCode', values=str(year)).apply(pd.to_numeric, errors='coerce')
    ws_ctl = ctl[str(sc.lines.wage_salary)]; pr_ctl = ctl[str(sc.lines.proprietors)]
    lam_u, lam_p = {}, {}
    for s in states:
      if s not in ctl.index or np.isnan(ws_ctl[s]) or np.isnan(pr_ctl[s]):
        lam_u[s] = lam_p[s] = 1.0
        continue
      # farm (from the NASS-carried panel) is held; the factors act on nonfarm
      nf = [k for k in leaves if k != '111-112']
      cov = S_c.loc[s].sum(); u = S_u.loc[s, nf].sum(); pr = S_p.loc[s, nf].sum()
      target_p = pr_ctl[s] - S_p.loc[s, '111-112']
      lam_p[s] = target_p / pr if (pr > 0 and target_p > 0) else 1.0
      target_u = ws_ctl[s] - cov - S_u.loc[s, '111-112']
      lam_u[s] = target_u / u if (u > 0 and target_u > 0) else (0.0 if target_u <= 0 else 1.0)
      if target_u <= 0:
        logger.warning(f"{s}: covered W&S {cov:,.0f} exceeds SAINC4 W&S {ws_ctl[s]:,.0f}; uncovered set to 0")
      S_u.loc[s, nf] *= lam_u[s]; S_p.loc[s, nf] *= lam_p[s]
    lu = pd.Series(lam_u); lp = pd.Series(lam_p)
    logger.info(f"SAINC4 control: uncovered W&S factor median {lu.median():.3f} "
                f"[{lu.min():.3f}, {lu.max():.3f}]; proprietors factor median {lp.median():.3f} "
                f"[{lp.min():.3f}, {lp.max():.3f}] (extremes: "
                f"{lp.idxmin()} {lp.min():.2f}, {lp.idxmax()} {lp.max():.2f})")
    tot_ctl = sainc.loc[sainc.LineCode == str(sc.lines.total)].set_index('GeoFIPS')[str(year)]
    us_ctl = pd.to_numeric(tot_ctl.get('00000', np.nan), errors='coerce')
    logger.info(f"SAINC4 US total {year}: {us_ctl:,.0f}; assembled states sum "
                f"{(S_c.values.sum() + S_u.values.sum() + S_p.values.sum()):,.0f}")
  if REL:
    nf_ = [k for k in leaves if k != '111-112' and k not in ('GC', 'GSL', 'GM')]
    logger.info(f"release control: US private nonfarm assembled {(S_c[nf_].values.sum() + S_u[nf_].values.sum() + S_p[nf_].values.sum()):,.0f}; "
                f"uncovered W&S {S_u[nf_].values.sum():,.0f}; proprietors {S_p[nf_].values.sum():,.0f} (SAINC4 rescaling not applied)")
  S = S_c + S_u + S_p

  # ---------------- counties, leaves of the cemp tree (20 lines + gov + farm)
  cleaves = cemp.loc[cemp.is_leaf].index.tolist()
  C = pd.DataFrame(index=counties, columns=cleaves, data=0.0)
  st_of = regions.loc[counties, 'parent']
  for L in private_lc:
    cov_c = pd.Series({c: get(cov24, c, L) for c in counties})
    cov_s = cov_c.groupby(st_of).sum()
    share = cov_c / st_of.map(cov_s).replace(0, np.nan)
    # fallback: compensation share where a state has no covered jobs in L
    comp = pd.Series({c: get(data.county_other.balanced['compensation'], c, L) for c in counties})
    comp_s = comp.groupby(st_of).sum()
    share = share.fillna(comp / st_of.map(comp_s).replace(0, np.nan)).fillna(0.0)
    u_s = st_of.map(lambda s: U_eff.get((s, L), float(U.loc[s, L]) if (s in U.index and L in U.columns) else 0.0))
    pc = PC[L].reindex(counties).fillna(0.0) if L in PC.columns else pd.Series(0.0, index=counties)
    C[L] = cov_c.values + (u_s * share).values + pc.values
  for k in ['GC', 'GSL']:
    for c in counties:
      c24 = get(cov24, c, k); c22 = get(cov22, c, k); t22 = published(data.county_emp, c, k)
      resid = max(t22 - c22, 0.0) if not np.isnan(t22) else 0.0
      C.loc[c, k] = c24 + resid * (c24 / c22 if c22 > 0 else 1.0)
  for c in counties:
    t22 = published(data.county_emp, c, 'GM'); g = mil_g.get(c, np.nan)
    C.loc[c, 'GM'] = (t22 if not np.isnan(t22) else 0.0) * (g if not np.isnan(g) else 1.0)
  C['111-112'] = pd.Series(farm).reindex(counties).fillna(0.0).values

  # Connecticut planning regions have no 2022 county history: GC/GSL/GM there
  # get the state figure split by covered (GC/GSL) or by compensation (GM)
  ct = [c for c in counties if c.startswith('09')]
  if ct:
    for k in ['GC', 'GSL', 'GM']:
      base = pd.Series({c: get(cov24, c, k) for c in ct}) if k != 'GM' else \
        pd.Series({c: get(data.county_other.balanced['compensation'], c, k) for c in ct})
      base = base / base.sum() if base.sum() > 0 else pd.Series(1.0 / len(ct), index=ct)
      C.loc[ct, k] = S.loc['09000', k] * base.values

  # ---------------- reconcile counties to states at the line-code level
  S_lc = pd.DataFrame(index=states, columns=cleaves, data=0.0)
  for k in cleaves:
    kids = [x for x in leaves if lc_of[x] == k] if k in lc_of.values() else [k]
    S_lc[k] = S[kids].sum(axis=1) if all(x in S.columns for x in kids) else S[k]
  for k in cleaves:
    csum = C[k].groupby(st_of).sum()
    for s in states:
      target = float(S_lc.loc[s, k]); have = float(csum.get(s, 0.0))
      idx = [c for c in counties if st_of[c] == s]
      if have > 0 and abs(target - have) > 0.5:
        C.loc[idx, k] = C.loc[idx, k] * (target / have)
      elif have == 0 and target > 0:
        comp = pd.Series({c: get(data.county_other.balanced['compensation'], c, k) for c in idx})
        w = comp / comp.sum() if comp.sum() > 0 else pd.Series(1.0 / len(idx), index=idx)
        C.loc[idx, k] = target * w.values

  # ---------------- national and aggregates, write in the balanced layout
  def rollup(leaf_frame, tree_codes, dataset):
    """leaf values -> every node of region x sector tree in dataset.value."""
    frame = leaf_frame.copy()
    # add national row
    frame.loc['00000'] = frame.loc[frame.index.str[-3:] == '000'].sum(axis=0) \
      if (frame.index.str[-3:] == '000').any() else frame.sum(axis=0)
    # internal sectors
    for lvl in sorted(tree_codes.level.unique(), reverse=True):
      for sector in tree_codes.loc[(tree_codes.level == lvl) & (~tree_codes.is_leaf)].index:
        kids = tree_codes.loc[tree_codes.parent == sector].index
        frame[sector] = frame[[k for k in kids if k in frame.columns]].sum(axis=1)
    out = pd.Series(0.0, index=dataset.value.index)
    keys = pd.Series(out.index).str.split('_', n=1, expand=True)
    reg, sec = keys[0].values, keys[1].values
    ok = pd.Series(reg).isin(frame.index).values & pd.Series(sec).isin(frame.columns).values
    vals = [frame.loc[r, s] for r, s in zip(reg[ok], sec[ok])]
    out.loc[ok] = vals
    return out

  s_out = rollup(S, other, data.state_emp)
  c_frame = pd.concat([C, S_lc])           # counties + states on cemp leaves
  c_out = rollup(c_frame, cemp, data.county_emp)

  for name, out, dataset in [('state', s_out, data.state_emp), ('county', c_out, data.county_emp)]:
    df = pd.DataFrame({'total': out.round(0).astype(int)})
    df['total'] = qcew_operations.bottom_up_correction(
      source=df['total'], tree=dataset.tree, dim0='region', dim1='sector')
    us = df.loc['00000_T', 'total']; us22 = dataset.value.loc['00000_T', 'total']
    logger.info(f"{name}: US total {year} = {us:,} ({prev} published {us22:,.0f}); "
                f"farm {df.loc['00000_111-112', 'total']:,}; "
                f"government {df.loc['00000_92', 'total']:,}")
    df.to_csv(dir_t / config.target.files.__dict__[name], index=True)

  # diagnostics: state totals by line vs 2022 published
  diag = pd.DataFrame({'2024': s_out, f'{prev}_published': data.state_emp.value['total']})
  diag = diag.loc[[f"00000_{k}" for k in lc_sectors if f"00000_{k}" in diag.index]]
  diag['ratio'] = diag['2024'] / diag[f'{prev}_published'].replace(0, np.nan)
  logger.info(f"National by line code:\n{diag.round(3)}")
  logger.info("Finished step")
  return None



def project_pce_413(task_str, full_config):
  """RECON2024 (v1): {year} state PCE by 413 SUT commodities.

  The 2022 file (raw/PCE/final_results_413_commodities_by_state.xlsx) was
  built in GAUSS from the 113-line SAPCE4 vintage x a fixed 77x412 bridge.
  The current SAPCE4 vintage has 134 lines, so the 2022 commodity values are
  carried forward per state by the growth of the SAPCE4 line each commodity
  is mostly bridged from (old line -> current line by numerical matching of
  the 2022 state vectors; matches worse than `max_reldiff` fall back to the
  state's total PCE growth). Writes the same xlsx layout for {year}.
  Replace with a rebuild on BEA's PCE bridge when that table is on disk."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year, prev = full_config.year, config.previous_year
  P = dirs.__dict__[config.source.folder]

  x = pd.read_excel(P / config.source.pce_previous, header=None)
  hdr = [int(v) for v in x.iloc[1, 3:].tolist()]
  fips = [f"{h:02d}000" for h in hdr]
  body = x.iloc[2:].copy()
  codes413 = body.iloc[:, 0].astype(str).tolist()
  vals22 = body.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0.0)
  vals22.columns = fips; vals22.index = codes413

  old = pd.read_csv(P / config.source.old_lines, sep=r'\s+', header=None, engine='python')
  old = old.set_index(0); old.columns = fips; old = old.apply(pd.to_numeric, errors='coerce').fillna(0.0)
  btm = pd.read_csv(P / config.source.bottom, sep=r'\s+', header=None, engine='python')
  bottom = btm.loc[btm[2] == 1, 0].tolist()
  bridge = pd.read_csv(P / config.source.bridge, sep=r'\s+', header=None, engine='python')
  bridge_lines = bridge[0].astype(int).tolist()
  B = bridge.iloc[:, 1:].values          # 77 x 412, rows sum to 1
  if len(codes413) == B.shape[1] + 1:    # the xlsx carries a total row
    codes413 = codes413[:B.shape[1]]; vals22 = vals22.iloc[:B.shape[1]]

  s4 = pd.read_csv(P / config.source.sapce4, dtype=str, encoding='latin-1')
  s4['GeoFIPS'] = s4.GeoFIPS.str.strip().str.replace('"', '')
  s4 = s4.loc[s4.LineCode.notna()]
  new22 = s4.pivot(index='LineCode', columns='GeoFIPS', values=str(prev)).apply(pd.to_numeric, errors='coerce')
  new24 = s4.pivot(index='LineCode', columns='GeoFIPS', values=str(year)).apply(pd.to_numeric, errors='coerce')
  new22.index = new22.index.astype(int); new24.index = new24.index.astype(int)
  cols = [f for f in fips if f in new22.columns]

  # old bottom line -> current line, by 2022 state-vector distance
  match = {}
  for i in bridge_lines:
    v = old.loc[i, cols].values.astype(float); best = None
    for j in new22.index:
      w = np.nan_to_num(new22.loc[j, cols].values.astype(float))
      if w.sum() == 0:
        continue
      rel = np.abs(v - w).sum() / max(v.sum(), 1e-9)
      if best is None or rel < best[1]:
        best = (j, rel)
    match[i] = best
  good = {i: m[0] for i, m in match.items() if m[1] <= config.max_reldiff}
  logger.info(f"{len(good)} of {len(bridge_lines)} bottom lines matched within "
              f"{config.max_reldiff:.0%}; unmatched use total-PCE growth: "
              f"{sorted(set(bridge_lines) - set(good))}")

  # growth per (old line, state); total PCE growth as fallback
  tot_g = (new24.loc[1, cols] / new22.loc[1, cols]).astype(float)
  G = pd.DataFrame(index=bridge_lines, columns=cols, data=np.nan)
  for i in bridge_lines:
    if i in good:
      j = good[i]
      G.loc[i] = (new24.loc[j, cols] / new22.loc[j, cols].replace(0, np.nan)).values
    G.loc[i] = G.loc[i].fillna(tot_g)
  G = G.astype(float)

  # commodity growth = bridge-weighted average of line growth (weights: 2022 line value x bridge share)
  W = old.loc[bridge_lines, cols].values[:, :, None] * B[:, None, :]      # line x state x commodity
  num = (W * G.values[:, :, None]).sum(axis=0)
  den = W.sum(axis=0)
  g_c = np.where(den > 0, num / np.where(den > 0, den, 1.0), tot_g.values[:, None])   # state x commodity
  vals24 = vals22[cols].values.T * g_c                                     # state x commodity
  out = pd.DataFrame(vals24.T, index=codes413, columns=cols)
  us22, us24 = vals22[cols].values.sum(), out.values.sum()
  logger.info(f"PCE 413 x {len(cols)} states: {prev} {us22:,.0f} -> {year} {us24:,.0f} "
              f"(x{us24 / us22:.4f}); SAPCE4 total PCE x{(new24.loc[1, cols].sum() / new22.loc[1, cols].sum()):.4f}")

  # same xlsx layout as the 2022 file: two header rows, three index columns
  sheet = pd.DataFrame(index=range(len(codes413) + 2), columns=range(3 + len(cols)))
  sheet.iloc[0, 3:] = [x.iloc[0, 3 + fips.index(f)] for f in cols]
  sheet.iloc[1, :3] = x.iloc[1, :3].values; sheet.iloc[1, 3:] = [int(f[:2]) for f in cols]
  sheet.iloc[2:, 0] = codes413
  sheet.iloc[2:, 1] = body.iloc[:len(codes413), 1].values
  sheet.iloc[2:, 2] = body.iloc[:len(codes413), 2].values
  sheet.iloc[2:, 3:] = out.values
  target = dirs.__dict__[config.target.folder] / config.target.file
  with pd.ExcelWriter(target) as writer:
    sheet.to_excel(writer, sheet_name=config.target.sheet, header=False, index=False)
  logger.info(f"Saved {target}")
  logger.info("Finished step")
  return None


def process_pce_income(task_str, full_config):
  logger.info("Loading personal consumption expenditure and income")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  codes, _ = operations.load_codes_bridge()
  regions = codes.region
  del codes

  # loading SUT commodity codes
  dir_ = dirs.__dict__[config.source.commodities.folder]
  col0, row0, col1, row1 = config.source.commodities.range
  commodities = pd.read_excel(
    dir_ / config.source.commodities.file,
    sheet_name=config.source.commodities.sheet,
    index_col=None,
    header=None,
    skiprows=row0 - 1, nrows=(1 + row1 - row0),
    usecols=f'{col0}:{col1}')
  commodities.columns = config.source.commodities.columns
  commodities.code = commodities.code.astype(str)
  commodities.set_index('code', drop=True, inplace=True)

  # loading income and population
  dir_ = dirs.__dict__[config.source.income.folder]
  df = pd.read_csv(dir_ / config.source.income.file, encoding='latin1',
                   dtype=str)

  # loading pce data
  dir_ = dirs.__dict__[config.source.pce.folder]
  pce = pd.read_excel(dir_ / config.source.pce.file,
                      header=config.source.pce.header,
                      index_col=config.source.pce.index,
                      sheet_name=config.source.pce.sheet)

  logger.info('Formatting income and population')
  df = df.iloc[:-4]

  df = df[config.source.income.select.__dict__.keys()]
  df.rename(columns=config.source.income.select.__dict__, inplace=True)
  df = deepcopy(df.replace('"', '', regex=True))
  df = deepcopy(df.replace(' ', '', regex=True))

  logger.info("Extracting fields")
  selection_key = config.source.income.fields.__dict__
  df_res = pd.DataFrame(index=regions.index,
                        columns=list(selection_key.keys()))
  for key, val in selection_key.items():
    logger.info(f"Target column {key}, source field {val}")
    df_tmp = deepcopy(df.loc[df.sector_code == val])
    df_tmp.index = df_tmp.region_code
    df_res[key] = deepcopy(
      df_tmp.loc[list(df_res.index), 'value'].astype(int))
  df_res[regions.columns] = regions
  regions = df_res
  del df, df_res, selection_key, key, val

  # checking if expected codes exist
  logger.info("Formatting personal consumption expenditure")
  pce_commodities = pd.DataFrame(
    columns=['title'],
    index=[str(x[0]) for x in pce.index],
    data=[str(x[2]) for x in pce.index],
  )

  region_index = [str(x[1]) + '000' for x in pce.columns]
  region_index = ['00000'[:-len(x)] + x for x in region_index]
  pce_regions = pd.DataFrame(
    columns=['title'],
    index=region_index,
    data=[str(x[0]) for x in pce.columns],
  )

  logger.info("Commodities")
  n_error = 0
  for commodity in commodities.index:
    if commodity in pce_commodities.index:
      ref_val = commodities.loc[commodity, 'title']
      cmp_val = pce_commodities.loc[commodity, 'title']
      if ref_val != cmp_val:
        logger.warning(f"{commodity}: {ref_val} vs {cmp_val}")
    else:
      logger.info(f"missing commodity: {commodity}")
      n_error += 1

  logger.info("Regions")
  for region in regions.loc[regions.level == 1].index:
    if region in pce_regions.index:
      ref_val = regions.loc[region, 'title']
      cmp_val = pce_regions.loc[region, 'title']
      if ref_val != cmp_val:  # ' '.join(ref_val.split(' ')[:-2])
        logger.warning(f"{region}: {ref_val} vs {cmp_val}")
    else:
      logger.info(f"missing region: {region}")
      n_error += 1

  if n_error > 0:
    logger.error("Found missing commodity or region categories")
    raise KeyError

  # replacing indices and columns
  logger.info("Converting pce from million to thousand")
  pce = (pce * 1000).round(0).astype(int)

  logger.info("Reshaping")

  pce.columns = pce_regions.index
  pce.index = pce_commodities.index

  pce_tmp = pce.loc[commodities.index,
                    regions.loc[regions.level == 1].index]
  pce_tmp['00000'] = pce_tmp.sum(axis=1)
  pce_tmp['commodity'] = commodities.index
  pce = pce_tmp.melt(
    id_vars='commodity',
    value_vars=regions.loc[regions.level <= 1].index,
    var_name='region', value_name='pce')

  data = {'regions': regions, 'commodities': commodities, 'pce': pce}

  dir_ = dirs.__dict__[config.target.folder]

  @utilities.open_excel(dir_ / config.target.file)
  def save_df(writer, df, sheet, index):
    logger.debug(f"Saving table '{sheet}'")
    df.to_excel(writer, sheet_name=sheet, index=index)

  logger.info("And saving")

  for key, val in config.target.sheets.__dict__.items():
    save_df('_', data[key], key, val)

  logger.info("Finished step")
  return None


def apply_bridge(task_str, full_config):
  logger.info("Disaggregating BEA data to county SUT industry level")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  def report_agg_diff(dataframe, codes, string):
    tmp_ori = deepcopy(dataframe.loc[
      (codes.loc[dataframe.index, 'level'] < 2).values])
    tmp_sum = deepcopy(dataframe.loc[
      (codes.loc[dataframe.index, 'level'] > 0).values])
    tmp_sum.index = codes.loc[tmp_sum.index, 'parent'].values
    tmp_sum = tmp_sum.groupby(tmp_sum.index).sum()
    tmp_sum = tmp_sum.loc[tmp_ori.index]
    tmp_diff = tmp_sum - tmp_ori
    tmp_max = tmp_diff.max().max()
    tmp_min = tmp_diff.min().min()
    logger.info(
      f"Difference between {tmp_min} and {tmp_max} {string}")
    return None

  logger.info('Loading data')
  bea = operations.BeaToSut(load_proxy=True)

  logger.info('Converting data across scales')
  data = utilities.DictToObject({
    'proxy': {},
    'employment': {},
    'earnings': {},
    'compensation': {},
    'gdp_wages': {},
    'gdp_surplus': {},
    'gdp_nettax': {},
    'gdp_total': {},
  })

  logger.info('Creating proxy (QCEW wages) in full classification: '
              '3166 BEA regions and 402 SUT industries')
  logger.info('Relations in region (bea) to qcew (area)')
  logger.info('Expected all qcew to be single, region single or multi')
  bea.show_coverage('region_qcew')

  df0 = pd.DataFrame(index=bea.codes.qcew.index,
                     columns=bea.codes.sut.index,
                     data=float(0))
  df = bea.data.proxy.pivot(
    index='area', columns='industry', values='wages')
  df0.loc[df.index, df.columns] = df
  df = bea.bridge.region_qcew.matrix.dot(df0)

  logger.info('A minor discrepancy was found in the wage sums of QCEW '
              'wages for the following industries: '
              '"233210", "233262", "230301", "230302", "2332A0", '
              '"233412", "2334A0", "233230", "233411", "336111", '
              '"336112". It is probably related to rounding error, but '
              'it is systematic, so maybe clarify if there is another '
              'revision')

  report_agg_diff(df0, bea.codes.qcew, 'in QCEW wages in QCEW regions')

  # there are missing counties, so necessary to aggregate

  def reset_bottom_up_region(df, bea):
    df0 = bea.bridge.state_county.matrix.dot(df)
    index = bea.codes.region.loc[bea.codes.region.level == 1].index
    df.loc[index] = df0.loc[index]

    index = bea.codes.region.loc[bea.codes.region.level == 0].index
    df.loc[index[0]] = df0.sum(0)
    return df

  df = reset_bottom_up_region(df, bea)
  setattr(data.proxy, 'full', df)

  report_agg_diff(data.proxy.full, bea.codes.region,
                  'in QCEW wages after converting to BEA regions '
                  'and aggregating up, in full industry clossification')

  logger.info('Relations in base (bea) to sut (industry)')
  logger.info('Expected all sut to be single, base single or multi')
  bea.show_coverage('base_sut')
  setattr(data.proxy, 'base',
          data.proxy.full.dot(bea.bridge.base_sut.matrix.T))

  report_agg_diff(data.proxy.full, bea.codes.region,
                  'in QCEW wages after converting to BEA regions '
                  'and aggregating up, in base industry clossification')

  logger.info('Relations in gdp (industry) to base (industry)')
  logger.info('Expected all base to be single, gdp single or multi')
  bea.show_coverage('base_sut')
  setattr(data.proxy, 'gdp',
          data.proxy.full.dot(bea.bridge.base_sut.matrix.T))

  report_agg_diff(data.proxy.full, bea.codes.region,
                  'in QCEW wages after converting to BEA regions '
                  'and aggregating up, in GDP industry clossification')

  logger.info('Disaggregating earnings and BEA compensation')
  # earnings and compensation
  for column in bea.data.county_other.balanced.columns:
    df0 = pd.DataFrame(index=bea.codes.region.index,
                       columns=bea.codes.other.index,
                       data=float(0))
    df = deepcopy(bea.data.county_other.tree)
    df['value'] = bea.data.county_other.balanced[column]
    df = df.pivot(index='region', columns='sector', values='value')
    df0.loc[df.index, df.columns] = df
    setattr(data.__dict__[column], 'other', df0)
    del df, df0
    data.__dict__[column].other = reset_bottom_up_region(
      data.__dict__[column].other, bea)

    setattr(data.__dict__[column], 'base',
            operations.disaggregate_with_proxy(
              data.__dict__[column].other,
              data.proxy.base,
              bea.bridge.other_base.matrix,
              transpose=True))
    data.__dict__[column].base = reset_bottom_up_region(
      data.__dict__[column].base, bea)

    setattr(data.__dict__[column], 'full',
            operations.disaggregate_with_proxy(
              data.__dict__[column].base,
              data.proxy.full,
              bea.bridge.base_sut.matrix,
              transpose=True))
    data.__dict__[column].full = reset_bottom_up_region(
      data.__dict__[column].full, bea)

    setattr(data.__dict__[column], 'gdp',
            data.__dict__[column].base.dot(
              bea.bridge.gdp_base.matrix.T))
    data.__dict__[column].gdp = reset_bottom_up_region(
      data.__dict__[column].gdp, bea)

    string = ('Difference between original and disaggregated grand '
              f'total of {column}\n')
    grand_total = bea.data.county_other.balanced.loc['00000_T', column]
    for table in ['other', 'base', 'full', 'gdp']:
      dataframe = deepcopy(data.__dict__[column].__dict__[table])
      if table == 'full':
        pass
      else:
        dataframe = dataframe[
          bea.codes.__dict__[table].loc[
            bea.codes.__dict__[table].is_leaf].index]

      part_total = dataframe.loc['00000'].sum()
      diff_total = grand_total - part_total
      string = string + (
        f"{table}: {grand_total} - {part_total} = {diff_total}\n")
    logger.info(string)

  logger.info('Disaggregating GDP (total and components)')
  for column_source in bea.data.gdp.balanced.columns:
    column_target = 'gdp_' + column_source
    df0 = pd.DataFrame(index=bea.codes.region.index,
                       columns=bea.codes.gdp.index,
                       data=float(0))
    df = deepcopy(bea.data.gdp.tree)
    df['value'] = bea.data.gdp.balanced[column_source]
    df = df.pivot(index='region', columns='sector', values='value')

    # df0 now has gdp sector but is missing counties
    df0.loc[df.index, df.columns] = df
    # df has only counties
    df = operations.disaggregate_with_proxy(
      df0,
      data.compensation.gdp,
      bea.bridge.state_county.matrix)
    # df has both
    index = bea.codes.region.loc[bea.codes.region.level < 2].index
    df.loc[index] = df0.loc[index]
    setattr(data.__dict__[column_target], 'gdp', df)
    del df, df0
    data.__dict__[column_target].gdp = reset_bottom_up_region(
      data.__dict__[column_target].gdp, bea)

    setattr(data.__dict__[column_target], 'base',
            operations.disaggregate_with_proxy(
              data.__dict__[column_target].gdp,
              data.compensation.base,
              bea.bridge.gdp_base.matrix,
              transpose=True))
    data.__dict__[column_target].base = reset_bottom_up_region(
      data.__dict__[column_target].base, bea)

    setattr(data.__dict__[column_target], 'full',
            operations.disaggregate_with_proxy(
              data.__dict__[column_target].base,
              data.compensation.full,
              bea.bridge.base_sut.matrix,
              transpose=True))
    data.__dict__[column_target].full = reset_bottom_up_region(
      data.__dict__[column_target].full, bea)

    string = ('Difference between original and disaggregated grand '
              f'total of {column_source}\n')
    grand_total = bea.data.gdp.balanced.loc['00000_T', column_source]
    for table in ['gdp', 'base', 'full']:
      dataframe = deepcopy(
        data.__dict__[column_target].__dict__[table])
      if table == 'full':
        pass
      else:
        dataframe = dataframe[
          bea.codes.__dict__[table].loc[
            bea.codes.__dict__[table].is_leaf].index]

      part_total = dataframe.loc['00000'].sum()
      diff_total = grand_total - part_total
      string = string + (
        f"{table}: {grand_total} - {part_total} = {diff_total}\n")
    logger.info(string)

  logger.info('Disaggregating employment')
  # employment
  # first disaggregate county_emp to other

  # county_emp in cemp classification, with counties
  df0 = pd.DataFrame(index=bea.codes.region.index,
                     columns=bea.codes.cemp.index,
                     data=float(0))
  df = deepcopy(bea.data.county_emp.tree)
  df['value'] = bea.data.county_emp.balanced.total
  df = df.pivot(index='region', columns='sector', values='value')
  df0.loc[df.index, df.columns] = df

  # county_emp in other classification, with counties
  proxy = operations.disaggregate_with_proxy(
    df0,
    data.compensation.other,
    bea.bridge.cemp_other_leaf.matrix,
    transpose=True)

  # disaggregate state_emp to counties

  # state_emp in other classification, without counties
  df0 = pd.DataFrame(index=bea.codes.region.index,
                     columns=bea.codes.other.index,
                     data=float(0))
  df = deepcopy(bea.data.state_emp.tree)
  df['value'] = bea.data.state_emp.balanced.total
  df = df.pivot(index='region', columns='sector', values='value')
  df0.loc[df.index, df.columns] = df

  # state_emp in other classification, with counties
  df = operations.disaggregate_with_proxy(
    df0,
    proxy,
    bea.bridge.state_county.matrix)

  # df has both
  index = bea.codes.region.loc[bea.codes.region.level < 2].index
  df.loc[index] = df0.loc[index]
  setattr(data.employment, 'other', df)
  data.employment.other = reset_bottom_up_region(
    data.employment.other, bea)

  setattr(data.employment, 'base',
          operations.disaggregate_with_proxy(
            data.employment.other,
            data.compensation.base,
            bea.bridge.other_base.matrix,
            transpose=True))
  data.employment.base = reset_bottom_up_region(
    data.employment.base, bea)

  setattr(data.employment, 'full',
          operations.disaggregate_with_proxy(
            data.employment.base,
            data.compensation.full,
            bea.bridge.base_sut.matrix,
            transpose=True))
  data.employment.full = reset_bottom_up_region(
    data.employment.full, bea)

  string = ('Difference between original and disaggregated grand '
            'total of employment\n')
  grand_total = bea.data.county_emp.balanced.loc['00000_T', 'total']
  for table in ['other', 'base', 'full']:
    dataframe = deepcopy(data.employment.__dict__[table])
    if table == 'full':
      pass
    else:
      dataframe = dataframe[
        bea.codes.__dict__[table].loc[
          bea.codes.__dict__[table].is_leaf].index]

    part_total = dataframe.loc['00000'].sum()
    diff_total = grand_total - part_total
    string = string + (f"{table}: "
                       f"{grand_total} - {part_total} = {diff_total}\n")
  logger.info(string)

  logger.info('Converting datasets to single table in long format')
  for key, val in data.__dict__.items():
    val.full['region'] = val.full.index
    setattr(data.__dict__[key], 'table',
            val.full.melt(id_vars='region',
                          value_vars=val.full.columns,
                          var_name='industry'))
    data.__dict__[key].table.index = (
      data.__dict__[key].table.region
      + '_' + data.__dict__[key].table.industry)

  results = pd.DataFrame(
    columns=data.__dict__.keys(),
    index=data.__dict__[list(data.__dict__.keys())[0]].table.index)

  for key, val in data.__dict__.items():
    results[key] = val.table.loc[results.index, 'value']

  logger.info('Clean-up operations, for clarity')
  logger.info('Removing QCEW and BEA non-GDP compensation')
  results.drop(['proxy', 'compensation', 'gdp_total'], axis=1,
               inplace=True)
  for column in results.columns:
    if column[:4] == 'gdp_':
      results[column[4:]] = results[column]
      del results[column]

  logger.info('Setting precision at 1 employee or 1000 dollars')
  results = results.round(0)
  results = results.astype(int)

  logger.info('Splitting region/industry codes')
  results_formatted = pd.DataFrame(index=results.index,
                                   columns=['region', 'industry'])
  results['tmp'] = results.index
  results_formatted[[
    'region', 'industry']] = results['tmp'].str.split('_', expand=True)
  del results['tmp']
  results_formatted = results_formatted[
    ['industry', 'region']]
  results_formatted.sort_values(['region', 'industry'], inplace=True)

  results_formatted[results.columns] = results

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  results_formatted.to_csv(dir_ / config.target.file, index=False)

  logger.info("Finished step")


def rebuild_pce_413(task_str, full_config):
  """RECON2024 (v2, 2026-09-21): {year} state PCE by the 413 SUT commodities.

  GAUSS logic of final_work_PCE.g kept (state PCE by bottom line x the fixed
  77 x 412 PCE-line -> Use-commodity bridge), rebuilt on {year} data:
  1. The 77 bottom lines of the old (product-based, 113-line) SAPCE4 vintage
     are carried to {year} per state by the growth of the current
     (function-based, 134-line) SAPCE4 line or line group that contains them
     (`line_map`; one-to-one where the two vintages agree).
  2. National line x commodity flows T = diag(US line {year}) . bridge are
     RAS-balanced to (rows) the US {year} line values scaled to the SUT PCE
     total and (columns) national {year} PCE by commodity from the SUT.
  3. State commodity PCE = sum over lines of state line value x the line's
     balanced commodity mix. States sum to the national column totals x the
     SAPCE/SUT scale.
  Writes the same xlsx layout as the 2022 file to `target.file`."""
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  year, prev = full_config.year, config.previous_year
  P = dirs.__dict__[config.source.folder]

  x = pd.read_excel(P / config.source.pce_previous, header=None)
  hdr = [int(v) for v in x.iloc[1, 3:].tolist()]
  fips_all = [f"{h:02d}000" for h in hdr]
  states = [f for f in fips_all if int(f[:2]) < 60]
  body = x.iloc[2:].copy()
  codes = body.iloc[:, 0].astype(str).tolist()

  old = pd.read_csv(P / config.source.old_lines, sep=r'\s+', header=None, engine='python')
  old = old.set_index(0).apply(pd.to_numeric, errors='coerce').fillna(0.0)
  old.columns = fips_all
  old = old[states]
  bridge = pd.read_csv(P / config.source.bridge, sep=r'\s+', header=None, engine='python')
  lines = bridge[0].astype(int).tolist()
  Bm = bridge.iloc[:, 1:].values.astype(float)            # 77 x 412
  codes = codes[:Bm.shape[1]]

  s4 = pd.read_csv(P / config.source.sapce4, dtype=str, encoding='latin-1')
  s4 = s4.loc[s4.LineCode.notna()].copy()
  s4['GeoFIPS'] = s4.GeoFIPS.str.strip().str.replace('"', '').str.strip()
  n0 = s4.pivot(index='LineCode', columns='GeoFIPS', values=str(prev)).apply(pd.to_numeric, errors='coerce')
  n1 = s4.pivot(index='LineCode', columns='GeoFIPS', values=str(year)).apply(pd.to_numeric, errors='coerce')
  n0.index = n0.index.astype(int); n1.index = n1.index.astype(int)
  n0 = n0.reindex(columns=states).fillna(0.0); n1 = n1.reindex(columns=states).fillna(0.0)

  # 1. old bottom lines to {year}
  lmap = {int(k): [int(j) for j in v] for k, v in config.line_map.__dict__.items()}
  tot_g = n1.loc[1] / n0.loc[1]
  L1 = pd.DataFrame(0.0, index=lines, columns=states)
  unmapped = []
  for i in lines:
    J = lmap.get(i)
    if not J:
      unmapped.append(i); L1.loc[i] = old.loc[i] * tot_g; continue
    g0, g1 = n0.loc[J].sum(), n1.loc[J].sum()
    g = (g1 / g0.where(g0 != 0)).fillna(tot_g)
    L1.loc[i] = old.loc[i] * g
  if unmapped:
    logger.info(f"bottom lines without a line_map entry use total PCE growth: {unmapped}")
  us_lines = L1.sum(axis=1)

  # 2. national RAS to SUT PCE by commodity
  sut_pce = None
  for f in config.national.files:
    path = dirs.__dict__[f.folder] / f.file
    if path.exists():
      if str(path).endswith('.xlsx'):
        sut_pce = pd.read_excel(path, sheet_name=f.sheet, index_col=0)[f.column]
      else:
        sut_pce = pd.read_csv(path, index_col=0)[f.column]
      sut_pce.index = sut_pce.index.astype(str)
      logger.info(f"national PCE by commodity from {path.name}: {sut_pce.sum():,.0f}")
      break
  col_t = sut_pce.reindex(codes).fillna(0.0).clip(lower=0).values * config.national.unit_factor
  T = us_lines.values[:, None] * Bm
  scale = col_t.sum() / us_lines.sum()
  row_t = us_lines.values * scale
  pos = row_t > 0; neg = ~pos
  ok_c = (T[pos].sum(axis=0) > 0)
  miss = col_t[~ok_c].sum()
  logger.info(f"SUT/SAPCE scale {scale:.4f}; commodities with SUT PCE but no bridge path: "
              f"{int(((col_t > 0) & ~ok_c).sum())} (${miss / 1e6:,.1f}bn, dropped)")
  col_t = np.where(ok_c, col_t, 0.0)
  # commodities with non-positive SUT PCE (S00900 net foreign travel
  # adjustment, S00401 scrap) keep the bridge values; the rest are scaled so
  # that columns and rows add to the same total
  fixed = sut_pce.reindex(codes).fillna(0.0).values <= 0
  col_t = np.where(fixed, T.sum(axis=0) * scale, col_t)
  free = ~fixed
  col_t[free] = col_t[free] * (row_t.sum() - col_t[fixed].sum()) / col_t[free].sum()
  Tn = T.copy()
  for it in range(config.ras.nmax):
    # rows: only the free columns are scaled, the fixed ones stay as bridged
    fix_r = Tn[:, fixed].sum(axis=1)
    rs = Tn[:, free].sum(axis=1)
    fac = np.where(rs != 0, (row_t - fix_r) / np.where(rs != 0, rs, 1.0), 1.0)
    fac = np.clip(fac, 0.0, None)
    Tn[:, free] = Tn[:, free] * fac[:, None]
    # columns: positive lines scaled; negative lines (nonresident spending,
    # NPISH receipts) keep their bridged values
    cs = Tn[pos].sum(axis=0)
    need = col_t - Tn[neg].sum(axis=0)
    fc = np.where(cs > 0, need / np.where(cs > 0, cs, 1.0), 1.0)
    fc = np.clip(fc, 0.0, None)
    fc[fixed] = 1.0
    Tn[pos] = Tn[pos] * fc[None, :]
    err = np.abs(Tn[:, free].sum(axis=0) - col_t[free]).sum() / np.abs(col_t[free]).sum()
    rerr = np.abs(Tn.sum(axis=1) - row_t).sum() / np.abs(row_t).sum()
    if max(err, rerr) < config.ras.eps:
      break
  logger.info(f"RAS: {it + 1} iterations, column error {err:.2e}, row error {rerr:.2e}")
  rerr = np.abs(Tn.sum(axis=1) - row_t).sum() / np.abs(row_t).sum()
  cdiff = pd.Series(Tn.sum(axis=0) - col_t, index=codes)
  logger.info(f"largest column gaps after RAS ($M): {cdiff.abs().sort_values().tail(6).round(0).to_dict()}")
  rdiff = pd.Series(Tn.sum(axis=1) - row_t, index=lines)
  logger.info(f"largest row gaps after RAS ($M): {rdiff.abs().sort_values().tail(6).round(0).to_dict()}")
  mix = np.where(Tn.sum(axis=1, keepdims=True) != 0,
                 Tn / np.where(Tn.sum(axis=1, keepdims=True) != 0, Tn.sum(axis=1, keepdims=True), 1.0), 0.0)

  # 3. states
  # NPISH: gross output (+) and sales receipts (-) are separate lines, and
  # their difference by state is noise (NJ took 10% of US religious
  # organizations). The two lines are netted nationally into one commodity
  # mix and states get it in proportion to SAPCE4 NPISH final consumption.
  np_lines = [int(v) for v in getattr(config, 'npish_lines', [])]
  keep = [k for k, i in enumerate(lines) if i not in np_lines]
  out = L1.values[keep].T @ mix[keep]
  if np_lines:
    idx = [lines.index(i) for i in np_lines]
    net = Tn[idx].sum(axis=0)
    n_neg = int((net < 0).sum())
    net = np.clip(net, 0.0, None)
    np_mix = net / net.sum()
    np_state = n1.loc[int(config.npish_line_new)]
    np_state = np_state / np_state.sum() * L1.loc[np_lines].values.sum()
    out = out + np.outer(np_state.values, np_mix)
    logger.info(f"NPISH lines {np_lines} netted into one mix ({n_neg} negative commodity "
                f"cells clipped), states by SAPCE4 line {config.npish_line_new}")
  out = pd.DataFrame(out, index=states, columns=codes).T * scale
  logger.info(f"state PCE by commodity {year}: {out.values.sum():,.0f} "
              f"(SAPCE4 line 1 {n1.loc[1].sum():,.0f}, x SUT scale {scale:.4f})")

  # comparison with the current file (v1)
  cmp_path = dirs.__dict__[config.target.folder] / config.compare_with
  if cmp_path.exists():
    y = pd.read_excel(cmp_path, header=None)
    v1 = y.iloc[2:2 + len(codes), 3:].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    v1 = pd.DataFrame(v1, index=codes, columns=[f"{int(h):02d}000" for h in y.iloc[1, 3:].tolist()])[states]
    sh_new = out.div(out.sum(axis=1).replace(0, np.nan), axis=0)
    sh_old = v1.div(v1.sum(axis=1).replace(0, np.nan), axis=0)
    w = out.sum(axis=1).abs()
    d = (sh_new - sh_old).abs().sum(axis=1) / 2
    logger.info(f"state-share difference v2 vs v1 (half L1, PCE-weighted): {np.nansum(d * w) / w.sum():.4f}; "
                f"largest: {d.sort_values().tail(5).round(3).to_dict()}")

  sheet = pd.DataFrame(index=range(len(codes) + 2), columns=range(3 + len(states)))
  names = dict(zip(fips_all, x.iloc[0, 3:].tolist()))
  sheet.iloc[0, 3:] = [names[f] for f in states]
  sheet.iloc[1, :3] = x.iloc[1, :3].values; sheet.iloc[1, 3:] = [int(f[:2]) for f in states]
  sheet.iloc[2:, 0] = codes
  sheet.iloc[2:, 1] = body.iloc[:len(codes), 1].values
  sheet.iloc[2:, 2] = body.iloc[:len(codes), 2].values
  sheet.iloc[2:, 3:] = out.values
  target = dirs.__dict__[config.target.folder] / config.target.file
  with pd.ExcelWriter(target) as writer:
    sheet.to_excel(writer, sheet_name=config.target.sheet, header=False, index=False)
  logger.info(f"Saved {target.name}")
  logger.info("Finished step")
  return None
