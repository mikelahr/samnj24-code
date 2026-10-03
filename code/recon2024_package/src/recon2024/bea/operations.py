"""Operations of BEA stage."""
import recon2024.qcew.operations as qcew_operations
import os
from pathlib import Path
import numpy as np
import pdb
import scipy.sparse as sp
from .. import utilities
import re
import pandas as pd
from logging import getLogger
from copy import deepcopy
from bs4 import BeautifulSoup
import xmltodict
from time import time
logger = getLogger('root')


def load_codes_bridge():
  logger.debug("Loading BEA codes and bridge")
  full_config = utilities.get_config('bea')
  dirs = full_config.dirs
  config = full_config.process_codes

  dir_ = dirs.__dict__[config.target.codes.folder]
  codes = {}
  with pd.ExcelFile(dir_ / config.target.codes.file) as xls:
    for sheet in config.target.codes.sheets:
      codes[sheet] = pd.read_excel(
        xls,
        sheet_name=sheet,
        index_col='code',
        dtype={'code': str, 'parent': str, 'level': int})
    codes[sheet].index = codes[sheet].index.astype(str)
  codes = utilities.DictToObject(codes)

  dir_ = dirs.__dict__[config.target.bridge.folder]
  bridge = {}
  with pd.ExcelFile(dir_ / config.target.bridge.file) as xls:
    for sheet in config.target.bridge.sheets:
      bridge[sheet] = pd.read_excel(
        xls,
        sheet_name=sheet,
        index_col=None,
        dtype=str)
  bridge = utilities.DictToObject(bridge)

  return codes, bridge


def load_qcew():
  logger.debug("Loading QCEW data in BEA regions")
  full_config = utilities.get_config('bea')
  dirs = full_config.dirs
  config = full_config.process_qcew

  dir_ = dirs.__dict__[config.target.folder]
  df = pd.read_csv(
    dir_ / config.target.file,
    index_col='code')

  return df


def load_proxies():
  logger.debug("Loading BEA proxies")
  full_config = utilities.get_config('bea')
  dirs = full_config.dirs
  config = full_config.process_proxies

  dir_ = dirs.__dict__[config.target.folder]
  df = pd.read_csv(
    dir_ / config.target.file,
    index_col='code')

  ref_data = load_data().county_other

  df['proxy'] = deepcopy(df['wages']) * config.select.factor

  df_tmp = deepcopy(ref_data.value[['compensation']])
  df_tmp['ratio'] = df_tmp['compensation']/df['proxy']
  df_tmp = df_tmp.loc[(ref_data.tree.region_level == 0)]

  df_tmp.loc[
    df_tmp.ratio > config.select.upper_bound, 'proxy'] = 0
  df_tmp.loc[
    df_tmp.ratio < config.select.lower_bound, 'proxy'] = 0

  sector_no_proxy = ref_data.tree.loc[
    df_tmp.loc[df.proxy == 0].index, 'sector'].values
  df.loc[
    ref_data.tree.sector.isin(sector_no_proxy), 'proxy'] = 0

  return df


def load_data():
  logger.debug("Loading BEA data")
  full_config = utilities.get_config('bea')
  dirs = full_config.dirs
  config = full_config.process_data

  dir_ = dirs.__dict__[config.target.folder]
  bea = {}
  for dataset, file in config.target.files.__dict__.items():
    bea[dataset] = {}
    bea[dataset]['tree'] = pd.read_csv(
      dir_ / file,
      index_col='code',
      dtype={'code': str, 'region': str, 'sector': str,
             'region_parent': str, 'sector_parent': str})
    for table in ['value', 'flag']:
      bea[dataset][table] = pd.DataFrame(
        index=bea[dataset]['tree'].index)
      for key in config.combine_tables.__dict__[
          dataset].__dict__.keys():
        bea[dataset][table][key] = bea[
          dataset]['tree'][f'{table}_{key}']
        bea[dataset]['tree'].drop(
          f'{table}_{key}', axis=1, inplace=True)

    for column in bea[dataset]['value'].columns:
      bea[dataset]['value'][column] = (
        bea[dataset]['value'][column].astype(int))

    file_balanced = Path(file).stem + '_balanced.csv'
    if os.path.exists(dir_ / file_balanced):
      bea[dataset]['balanced'] = pd.read_csv(
        dir_ / file_balanced,
        index_col='code')
      for column in bea[dataset]['balanced'].columns:
        bea[dataset]['balanced'][column] = (
          bea[dataset]['balanced'][column].astype(int))

  return utilities.DictToObject(bea)


def import_data():
  full_config = utilities.get_config('bea')
  dirs = full_config.dirs
  config = full_config.process_structure

  bea = {}
  for table_key, table_val in config.source.__dict__.items():
    dir_ = dirs.__dict__[table_val.folder]
    prefix = table_val.file.split('_')[0]

    # load sector line definition and table info
    suffix = config.suffixes.definition
    with open(dir_ / (prefix + suffix), 'r', encoding='utf-8',
              errors='replace') as file:
      xml_content = file.read()
    tmp_dict = xmltodict.parse(xml_content)['TABLE_DEFINITIONS']
    tmp_df = pd.DataFrame(tmp_dict['LINE'])
    sector_definition = pd.DataFrame()
    for col_key, col_val in config.definition_columns.__dict__.items():
      sector_definition[col_key] = tmp_df[col_val]
    del tmp_df
    sector_definition['line'] = sector_definition.line.astype(int)
    # sector.loc[sector.index, 'code'] = sector.code.astype(int)
    sector_definition.set_index("line", inplace=True)

    info = pd.DataFrame(columns=['text'])
    for key, val in tmp_dict.items():
      if '@' in key:
        info.loc['def_' + key[1:]] = {'text': val}

    # load footnotes, append to info
    suffix = config.suffixes.footnotes
    with open(dir_ / (prefix + suffix), 'r', encoding='utf-8',
              errors='replace') as file:   # RECON2024: cp1252 footnotes
      html_str = file.read()
    soup = BeautifulSoup(html_str, 'html.parser')
    tmp_ = [li.text for li in soup.find_all('li')]
    sep = '\xa0'
    footnotes = {x.split(sep)[0]: x.split(sep)[-1] for x in tmp_}
    for key_foot, val_foot in footnotes.items():
      info.loc['foot_' + key_foot] = {'text': val_foot}

    # load data
    try:
      data_tmp = pd.read_csv(dir_ / table_val.file, dtype=str)
    except UnicodeDecodeError as e:
      logger.error(f"UnicodeDecodeError occurred: {e}")
      data_tmp = pd.read_csv(dir_ / table_val.file,
                             encoding='us-ascii',
                             encoding_errors='ignore',
                             dtype=str)

    # extract columns
    data = pd.DataFrame()
    for col_key, col_val in config.columns.__dict__.items():
      # RECON2024: a table may carry its own value year (SAEMP25N/CAEMP25N
      # end in 2022 and are read for structure only)
      if col_key == 'value' and hasattr(table_val, 'value_year'):
        col_val = str(table_val.value_year)
        logger.info(f"{table_key}: value column overridden to {col_val}")
      data[col_key] = data_tmp[col_val]
    del data_tmp

    # move footnotes to info
    for k, val in enumerate(data.iloc[-4:, 0]):
      info.loc[f'table_{k}'] = {'text': val}
    data = data.iloc[:-4]
    info.loc['table_year'] = {'text': config.columns.value}
    info.loc['table_unit'] = {'text': str(set(data.unit))}

    # clean geographic codes and extract region titles
    data['region_code'] = [x.strip().replace('"', '')
                           for x in data.region_code.values]
    region = pd.DataFrame()
    region['code'] = data['region_code']
    region['title'] = data['region_title']
    region.drop_duplicates(subset=None, keep='first',
                           inplace=True, ignore_index=True)
    region.set_index('code', inplace=True)
    # # remove macro-regions, harcoded as having first digit > 5
    # for index in reversed(region.index):
    #   if int(index[0]) > 5:
    #     region.drop(index, axis=0, inplace=True)

    # extract sector titles
    data['sector_line'] = data.sector_line.astype(int)
    data.loc[data.index, 'sector_title'] = [
      x.strip() for x in data.sector_title.values]

    sector = pd.DataFrame()
    sector['line'] = data['sector_line']
    sector['code'] = data['sector_code']
    sector['title'] = data['sector_title']
    sector.drop_duplicates(subset=None, keep='first',
                           inplace=True, ignore_index=True)
    sector.set_index('line', inplace=True)
    sector['comment'] = sector_definition.loc[
      sector_definition.index, 'comment']

    # split flag from value
    data_tmp = pd.to_numeric(data.value, errors='coerce')
    flag_tmp = data.value.loc[data_tmp.isna()]
    data['flag'] = False
    data.loc[flag_tmp.index, 'flag'] = True
    data['value'] = data_tmp.fillna(0)

    # drop redundant columns
    data.drop(['region_title', 'sector_code', 'sector_title',
               'unit'],
              axis=1,
              inplace=True)
    # data = data.loc[data.region.isin(region.index)]
    data.index = (data['region_code'] + '_' +
                  data['sector_line'].astype(str))

    logger.info(
      f'{table_key}: {len(data)}, {len(sector)}, {len(region)}')
    bea[table_key] = {'data': data, 'sector': sector,
                      'region': region, 'info': info}

  return utilities.DictToObject(bea)


def save_data(gdp, config, dirs):
  target_dir = dirs.__dict__[config.target.folder]
  for key_gdp, val_gdp in gdp.__dict__.items():
    filename = config.target.files.__dict__[key_gdp]
    with pd.ExcelWriter(target_dir / filename,
                        engine='xlsxwriter',
                        mode='w') as writer:
      val_gdp.data.to_excel(writer, sheet_name='data', index=True)
      val_gdp.sector.to_excel(
        writer, sheet_name='sector', index=True)
      val_gdp.region.to_excel(writer, sheet_name='region', index=True)
      val_gdp.info.to_excel(writer, sheet_name='info', index=True)


class BeaToSut():
  def __init__(self, load_proxy=False, load_data_bool=True,
               create_additional_bridges=True):
    # full_config = utilities.get_config('bea')
    # dirs = full_config.dirs
    # config = full_config.process_codes

    self.codes, bridge_table = load_codes_bridge()
    if create_additional_bridges:
      self.codes.sut = qcew_operations.QcewToSut().sut
    self.codes.qcew = qcew_operations.load_codes().area
    if load_data_bool:
      self.data = load_data()

    if load_proxy:
      setattr(self.data, 'proxy', qcew_operations.load_bridged_data())

    # adding bridge tables from config
    bridge = {}
    for key, val in bridge_table.__dict__.items():
      if not create_additional_bridges:
        if key != 'region_qcew':
          continue
      bridge[key] = {}
      bridge[key]['left'] = key.split('_')[0]
      bridge[key]['right'] = key.split('_')[1]
      bridge[key]['table'] = val
    setattr(self, 'bridge', utilities.DictToObject(bridge))

    # creating bridge matrices
    for key, bridge in self.bridge.__dict__.items():
      if not create_additional_bridges:
        if key != 'region_qcew':
          continue
      tmp = deepcopy(bridge.table)
      tmp['value'] = float(1)
      df_tmp = tmp.pivot(
        index=f'{bridge.left}_code',
        columns=f'{bridge.right}_code', values='value')
      df_tmp.fillna(float(0), inplace=True)
      df_tmp = df_tmp.astype(float)
      df = pd.DataFrame(
        index=self.codes.__dict__[bridge.left].index,
        columns=self.codes.__dict__[bridge.right].index,
        data=float(0))
      df.loc[df_tmp.index, df_tmp.columns] = df_tmp
      setattr(self.bridge.__dict__[key], 'matrix', df)

    if create_additional_bridges:
      # adding state-county bridge
      bridge = {}
      bridge['left'] = 'region'
      bridge['right'] = 'region'
      df_tmp = self.codes.region.loc[self.codes.region.level == 2]
      df = pd.DataFrame()
      df['state_code'] = df_tmp.parent.values
      df['state_title'] = self.codes.region.loc[
        df_tmp.parent.values, 'title'].values
      df['county_code'] = list(df_tmp.index)
      df['county_title'] = df_tmp.title.values
      bridge['table'] = df
      setattr(self.bridge, 'state_county',
              utilities.DictToObject(bridge))

      bridge = self.bridge.state_county
      tmp = deepcopy(bridge.table)
      tmp['value'] = float(1)
      df_tmp = tmp.pivot(
        index='state_code',
        columns='county_code', values='value')
      df_tmp.fillna(float(0), inplace=True)
      df_tmp = df_tmp.astype(float)
      df = pd.DataFrame(index=self.codes.__dict__[bridge.left].index,
                        columns=self.codes.__dict__[
                          bridge.right].index,
                        data=float(0))
      df.loc[df_tmp.index, df_tmp.columns] = df_tmp
      setattr(self.bridge.state_county, 'matrix', df)

      # adding cemp_other leaf-to-leaf: this should be changed in the
      # creation of the bridge, but maybe the current format is used
      # already
      bridge = {}
      bridge['left'] = 'cemp'
      bridge['right'] = 'other'
      df = pd.DataFrame(columns=['other_code', 'other_title',
                                 'cemp_code', 'cemp_title',
                                 'parent', 'no_match'])
      df['other_code'] = self.codes.other.loc[
        self.codes.other.is_leaf].index
      df['other_title'] = self.codes.other.loc[
        self.codes.other.is_leaf, 'title'].values
      df['parent'] = self.codes.other.loc[
        self.codes.other.is_leaf, 'parent'].values
      df['no_match'] = True
      cemp_leaf = self.codes.cemp.loc[self.codes.cemp.is_leaf]

      for index, row in df.iterrows():
        if row.other_code in cemp_leaf.index:
          df.loc[index, 'cemp_code'] = row.other_code
          df.loc[index, 'cemp_title'] = row.other_title
          df.loc[index, 'no_match'] = False

      n_no = df.no_match.sum()
      # logger.info("cemp to other leaf bridge")
      # logger.info(f"{len(df)}: {n_no}")
      while n_no > 0:
        for index, row in df.iterrows():
          if row.no_match:
            if row.parent in cemp_leaf.index:
              df.loc[index, 'cemp_code'] = row.parent
              df.loc[index, 'cemp_title'] = cemp_leaf.loc[
                row.parent, 'title']
              df.loc[index, 'no_match'] = False
            else:
              df.loc[index, 'parent'] = self.codes.other.loc[
                row.parent, 'parent']
        n_no = df.no_match.sum()
        # logger.info(f"{len(df)}: {n_no}")

      bridge['table'] = df
      setattr(self.bridge, 'cemp_other_leaf',
              utilities.DictToObject(bridge))

      bridge = self.bridge.cemp_other_leaf
      tmp = deepcopy(bridge.table)
      tmp['value'] = float(1)
      df_tmp = tmp.pivot(
        index='cemp_code',
        columns='other_code', values='value')
      df_tmp.fillna(float(0), inplace=True)
      df_tmp = df_tmp.astype(float)
      df = pd.DataFrame(index=self.codes.__dict__[bridge.left].index,
                        columns=self.codes.__dict__[
                          bridge.right].index,
                        data=float(0))
      df.loc[df_tmp.index, df_tmp.columns] = df_tmp
      setattr(self.bridge.cemp_other_leaf, 'matrix', df)

    return None

  def show_coverage(self, bridge, left='left', right='right'):
    if isinstance(bridge, str):
      right = self.bridge.__dict__[bridge].right
      left = self.bridge.__dict__[bridge].left
      bridge = self.bridge.__dict__[bridge].matrix
    elif not isinstance(bridge, pd.DataFrame):
      logger.error("bridge argument should be string or dataframe")
      raise ValueError

    df = {'top': (bridge != 0), 'val': bridge}
    for typ_ in ['top', 'val']:
      logger.info(f"Type: {typ_}")
      for key, val in {right: 0, left: 1}.items():
        logger.info(f"{key} neg   : {(df[typ_].sum(val) < 0).sum()}")
        logger.info(f"{key} single: {(df[typ_].sum(val) == 1).sum()}")
        logger.info(f"{key} zero  : {(df[typ_].sum(val) == 0).sum()}")
        logger.info(f"{key} multi : {(df[typ_].sum(val) > 1).sum()}\n")
    return None

  def normalize_columns(self, bridge):
    if isinstance(bridge, str):
      bridge = self.bridge.__dict__[bridge].matrix
    elif not isinstance(bridge, pd.DataFrame):
      logger.error("bridge argument should be string or dataframe")
      raise ValueError

    logger.debug("Make column sums = 1")
    right_code = list(bridge.columns)
    df = pd.DataFrame(
      index=right_code, columns=right_code, data=float(0))
    tmp = bridge.sum(0)
    values = (tmp != 0) / (tmp + (tmp == 0))
    np.fill_diagonal(df.values, values)
    return bridge.dot(df)


def set_initial(
    quant: pd.Series,
    proxy: pd.Series,
    nondisclosed: pd.Series,
    tree: pd.DataFrame,
    dim0: str,
    dim1: str,
    level: str = 'level',
    parent: str = 'parent'):
  """Creates initial estimate, given 2-dim tree constraint.

  All parameters must have the same index.
  Parent indices must be valid.
  If proxy available arithmetic average from both dimensions.
  If not proportion from both parents.
  If proxy is not zero its parents must be not zero too.
  Base level in each dimension must be disclosed.

  Parameters
  ----------
  quant: pd.Series(dtype=float)
    Vector with partial missing data to be estimated
  proxy: pd.Series(dtype=float)
    Vector without missing data used as proxy
  nondisclosed: pd.Series(dtype=bool)
    True if entry is nondisclosed
  tree: pandas.DataFrame
    Index: str
    Columns:
      Name: dim0 + '_' + level, dtype: int
      Name: dim1 + '_' + level, dtype: int
      Name: dim0 + '_' + parent, dtype: str
      Name: dim1 + '_' + parent, dtype: str
  level: str = 'level'
    String representing 'level' in tree column names
  level: str = 'parent'
    String representing 'parent' in tree column names
  dim0: str = 'area',
    String representing 'first dimension' in tree column names
  dim1: str = 'oind'
    String representing 'second dimension' in tree column names

  Returns
  -------
  pd.Series(dtype=float)
  """
  logger.info(f'Setting initial estimate, using {dim1} for proxy')

  df = pd.DataFrame()
  old_columns = [
    f'{dim0}_{level}', f'{dim1}_{level}', f'{dim0}_{parent}',
    f'{dim1}_{parent}']
  new_columns = [
    'dim0_level', 'dim1_level', 'dim0_parent', 'dim1_parent'
  ]
  df[new_columns] = deepcopy(tree[old_columns])
  df['nondisclosed'] = deepcopy(nondisclosed)
  df['quant'] = deepcopy(quant.astype(float))
  df['proxy'] = deepcopy(proxy.astype(float))
  del tree, nondisclosed, quant, proxy

  df.loc[df.nondisclosed, 'quant'] = int(0)

  max_dim0 = max(df.dim0_level)
  max_dim1 = max(df.dim1_level)

  #  checking nondisclosed with null proxy
  tmp = deepcopy(df.loc[df.nondisclosed & (df.proxy == 0)])
  if len(tmp) > 0:
    tmp0 = deepcopy(tmp.loc[(tmp.dim0_level == 0)
                            | (tmp.dim1_level == 0)])
    if len(tmp0) > 0:
      logger.error(
        f"There are {len(tmp0)} nondisclosed record(s) with "
        f"null proxy in some dimension base level")
      raise ValueError

    logger.info(
      f"{df.nondisclosed.sum()} nondisclosed, "
      f"{len(tmp)} with null proxy")

    df['cross_parent'] = None
    tmp = deepcopy(df.loc[(df.dim0_level > 0) & (df.dim1_level > 0)])
    tmp['cross_parent'] = deepcopy(df.loc[
      deepcopy(tmp.dim0_parent.values), 'dim1_parent'].values)
    df.loc[tmp.index, 'cross_parent'] = tmp.cross_parent

    del tmp

  #  checking non-null proxy with null parent
  tmp = deepcopy(df.loc[df.nondisclosed & (df.proxy != 0)])
  if len(tmp) > 0:
    for prefix in ['dim0', 'dim1']:
      tmp0 = deepcopy(tmp.loc[tmp[prefix + '_level'] > 0])
      if len(tmp0) > 0:
        tmp0['proxy_parent'] = df.loc[
          tmp0[prefix + '_parent'].values, 'proxy'].values
        null_parent = (tmp0['proxy_parent'] == 0).sum()
        if null_parent > 0:
          logger.error(
            f"Found {null_parent} proxies with null parent "
            f"for {eval(prefix)}")
          raise ValueError

  for k_dim0 in range(max_dim0 + 1):
    for k_dim1 in range(max_dim1 + 1):
      if (k_dim0 == 0) & (k_dim1 == 0):
        continue

      # when proxy is available
      select = deepcopy(df.loc[
        (df.dim0_level == k_dim0)
        & (df.dim1_level == k_dim1)
        & df.nondisclosed
        & (df.proxy != 0)])

      if len(select) > 0:
        factor = 1.0
        if (k_dim0 > 0) & (k_dim1 > 0):
          factor = 0.5
        for dim in ['dim0', 'dim1']:
          k_dim = eval('k_' + dim)
          if k_dim > 0:
            select['code_parent'] = df.loc[
              select.index, dim + '_parent'].values
            for name in ['quant', 'proxy']:
              select[name + '_parent'] = df.loc[
                select.code_parent.values, name].values

            select.quant = select.quant + factor * (
              select.proxy
              * select.quant_parent
              / select.proxy_parent)
      df.loc[select.index, 'quant'] = select.quant

      # when proxy is not available: k_dim0, k_dim1 > 0
      select = deepcopy(df.loc[
        (df.dim0_level == k_dim0)
        & (df.dim1_level == k_dim1)
        & df.nondisclosed
        & (df.proxy == 0)])

      if len(select) > 0:
        if (k_dim0 == 0) | (k_dim1 == 0):
          logger.error(
            'No proxy and nondisclosed value in top dimension')
          raise ValueError

        for prefix in ['dim0', 'dim1', 'cross']:
          select[prefix + '_quant'] = deepcopy(df.loc[
            select[prefix + '_parent'].values, 'quant'].values)

        select.quant = (
          select.dim0_quant * select.dim1_quant
          / select.cross_quant)
        df.loc[select.index, 'quant'] = select.quant

  logger.debug('Finished setting initial estimate')
  return df.quant


def balance_county_other(
    quant: pd.Series,
    proxy: pd.Series,
    nondisclosed: pd.Series,
    tree: pd.DataFrame,
    params: utilities.DictToObject,
    canopy_only=True):
  """Initializes and balances BEA compensation/earnings.

  It is meant to be applied sequentially: for initial estimation
    all levels below canopy are are set as disclosed.
  In balancing all nondisclosed entries are treated the same way.
  All parameters must have the same index.

  Parameters
  ----------
  quant: pd.Series(dtype=float)
    Vector with partial missing data to be estimated
  proxy: pd.Series(dtype=float)
    Vector without missing data used as proxy
  nondisclosed: pd.Series(dtype=bool)
    True if entry is nondisclosed
  tree: pandas.DataFrame
    Index: str
    Columns:
      Name: region_level, dtype: int
      Name: sector_level, dtype: int
      Name: region_parent, dtype: str
      Name: sector_parent, dtype: str
  params: DictToObj
    object with balancing parameters

  Returns
  -------
  pd.Series(dtype=float)
  """

  nondisclosed_initial = deepcopy(nondisclosed)
  if canopy_only:
    nondisclosed_initial.loc[
      ~ ((tree.region_level == max(tree.region_level))
         & (tree.sector_level ==
            max(tree.sector_level)))] = False

  logger.info("Setting initial estimates")
  quant = set_initial(
    quant=quant,
    proxy=proxy,
    nondisclosed=nondisclosed_initial,
    tree=tree,
    dim0='region',
    dim1='sector',)
  quant.loc[quant == 0] = 1

  logger.info("Creating variable vector and aggregation matrix")
  g, t, k_index = get_hierarchical_constraints(
    quant=quant,
    proxy=proxy,
    nondisclosed=nondisclosed,
    tree=tree)

  logger.info("Balance")
  bal = utilities.Balance.from_array(
    t, g, params, k_index)
  del g, t

  err, val = bal.get_error_value(error='source', value='source')
  logger.info('Started')

  err.sort_values('source', inplace=True)
  logger.info(f"Discrepancies before balancing:\n{err}")

  bal.run()
  err_tmp, val_tmp = bal.get_error_value()
  err['target'] = err_tmp.error
  val['target'] = val_tmp.value
  show_error(err, val, params)

  quant.loc[val.index] = val.target
  return quant


def show_error(err, val, params):
  err.sort_values('target', inplace=True)
  val['dif'] = val.target - val.source
  val['rel'] = 100 * (val.dif / val.source)
  val.sort_values('rel', inplace=True)

  logger.info("Statistics on balancing displacement:")
  for label in ['nondisclosed without proxy',
                'nondisclosed with proxy',
                'disclosed']:
    if label == 'nondisclosed without proxy':
      val_condition = (val.flag == 3)
    elif label == 'nondisclosed with proxy':
      val_condition = (val.flag == 2)
    else:
      val_condition = (val.flag < 2)
    val_tmp = deepcopy(val.loc[val_condition])
    val_max = abs(val_tmp.target.max())
    logger.info(f"{label}: {len(val_tmp)} records")
    logger.info(
      "Min abs, max abs, n records, max abs rel displacement")
    minval = 0.0
    maxval = params.eps
    while minval < val_max:
      val_curr = val_tmp.loc[
        (val_tmp.target > minval)
        & (val_tmp.target <= maxval)]
      logger.info(
        f"{minval:5.1e}-{maxval:5.1e}: {len(val_curr):5}, "
        f"{abs(val_curr.rel).max():5.2f}%")
      minval = maxval
      maxval = 10 * maxval

  return None


def get_hierarchical_constraints(
    quant: pd.Series,
    proxy: pd.Series,
    nondisclosed: pd.Series,
    tree: pd.DataFrame,):
  """Creates variable vector and constraint matrix for balancing.

  Data are 2-dim hierarchical.

  Parameters
  ----------
  quant: pd.Series(dtype=float)
    Vector with partial missing data to be estimated
  proxy: pd.Series(dtype=float)
    Vector without missing data used as proxy
  nondisclosed: pd.Series(dtype=bool)
    True if entry is nondisclosed
  tree: pandas.DataFrame
    Index: str
    Columns:
      Name: region_level, dtype: int
      Name: sector_level, dtype: int
      Name: region_parent, dtype: str
      Name: sector_parent, dtype: str

  Returns
  -------
  g: pd.DataFrame
  t: pd.DataFrame
  k_index: list
  """

  t = pd.DataFrame(index=quant.index,)
  t['pos'] = range(len(t))
  t['val'] = quant
  # t.loc[t.val == 0, 'val'] = int(1)
  t['flag'] = int(1)
  t.loc[nondisclosed & (proxy != 0), 'flag'] = int(2)
  t.loc[nondisclosed & (proxy == 0), 'flag'] = int(3)
  t.loc['00000_T', 'flag'] = int(0)

  g = pd.DataFrame(columns=['row', 'col', 'val'])
  k_index = []
  for pos, dim in enumerate(['region', 'sector']):
    logger.info(f"dimension: {dim}")

    # disaggregate values
    g0 = pd.DataFrame()
    g0 = deepcopy(
      tree.loc[tree[dim + '_parent'].notna()])
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
    k_index = k_index + list(dim + '_' + t.index)

  g = g.astype({'row': int, 'col': int, 'val': float})
  g.index = range(len(g))

  return g, t, k_index


def balance_employment(
    data,
    tree,
    params,
    k_region,
    k_sector):

  var_tmp = {}
  for table in ['state', 'county']:
    var_tmp[table] = {}

    quant = data.__dict__[table].quant
    proxy = data.__dict__[table].proxy
    nondisclosed = data.__dict__[table].nondisclosed

    logger.info(f"Setting initial estimate of {table}")
    quant = set_initial(
      quant=quant,
      proxy=proxy,
      nondisclosed=nondisclosed,
      tree=tree.__dict__[table],
      dim0='region',
      dim1='sector',)
    quant.loc[quant == 0] = 1

    (var_tmp[table]['g'],
     var_tmp[table]['t'],
     var_tmp[table]['k_index']) = (
      get_hierarchical_constraints(
        quant=quant,
        proxy=proxy,
        nondisclosed=nondisclosed,
        tree=tree.__dict__[table],))
  # del table, quant, proxy, nondisclosed
  var_tmp = utilities.DictToObject(var_tmp)

  # constraints between tables
  g = pd.DataFrame(columns=['row', 'col', 'val'])
  k_index = []
  county_pos = var_tmp.county.t.loc[
    tree.county.index.isin(var_tmp.state.t.index), 'pos']
  state_pos = var_tmp.state.t.loc[county_pos.index, 'pos']

  # county table
  g1 = pd.DataFrame(index=county_pos.index)
  g1['row'] = state_pos + 2 * (
    len(var_tmp.state.t) + len(var_tmp.county.t))
  g1['col'] = county_pos + len(var_tmp.state.t)
  g1['val'] = int(1)
  g = pd.concat([g, g1], axis=0)

  # state table
  g3 = pd.DataFrame(index=state_pos.index)
  g3['row'] = state_pos + 2 * (
    len(var_tmp.state.t) + len(var_tmp.county.t))
  g3['col'] = state_pos
  g3['val'] = int(-1)
  g = pd.concat([g, g3], axis=0)

  var_tmp.county.g.row = (
    var_tmp.county.g.row + 2 * len(var_tmp.state.t))
  var_tmp.county.g.col = (
    var_tmp.county.g.col + 1 * len(var_tmp.state.t))
  g = pd.concat([var_tmp.state.g, var_tmp.county.g, g], axis=0)
  g.index = range(len(g))
  g = g.astype({'row': int, 'col': int, 'val': float})

  # adding labels for state and county
  k_index = []
  k_index_cross = ['cross_' + x for x in var_tmp.state.t.index]
  for table in ['state', 'county']:
    k_index = k_index + [table + '_' + x for x in
                         var_tmp.__dict__[table].k_index]
    var_tmp.__dict__[table].t.index = (
      table + '_' + var_tmp.__dict__[table].t.index)
  k_index = k_index + k_index_cross

  t = pd.concat([var_tmp.state.t, var_tmp.county.t], axis=0)
  t.loc['county_00000_T', 'flag'] = int(1)

  logger.info("Balance")
  bal = utilities.Balance.from_array(
    t, g, params, k_index)
  del g, t

  err, val = bal.get_error_value(error='source', value='source')
  err.sort_values('source', inplace=True)
  logger.info(f"Discrepancies before balancing:\n{err}")
  logger.info(f"{err.source.describe()}")

  bal.run()
  err_tmp, val_tmp = bal.get_error_value()
  err['target'] = err_tmp.error
  val['target'] = val_tmp.value
  show_error(err, val, params)

  val['table'] = [x.split('_')[0] for x in val.index]
  val['code'] = ['_'.join(x.split('_')[1:]) for x in val.index]
  for table in ['state', 'county']:
    select = val.loc[val.table == table]
    data.__dict__[table].loc[select.code.values,
                             'quant'] = select.target.values
  return data


def disaggregate_with_proxy(aggregate, proxy, bridge, transpose=False):
  """aggregate (left) x proxy (right)"""

  if transpose:
    aggregate = aggregate.T
    proxy = proxy.T

  proxy = proxy + 1 * (proxy == 0)
  bridge_square = 1 * (bridge.T.dot(bridge) != 0)
  denominator = bridge_square.dot(proxy)
  numerator = bridge.T.dot(aggregate) * proxy
  results = utilities.safe_division(numerator, denominator)

  if transpose:
    results = results.T

  return results


def load_bridged():
  """Loads final (balanced) BEA industry- and commodity-based data"""
  full_config = utilities.get_config('bea')
  dirs = full_config.dirs
  bea_config = full_config.apply_bridge
  pce_config = full_config.process_pce_income

  bridged = {}

  dir_ = dirs.__dict__[bea_config.target.folder]
  bridged['bea'] = pd.read_csv(
    dir_ / bea_config.target.file,
    index_col=None,
    dtype={'region': str, 'industry': str})
  bridged['bea'].index = (
    bridged['bea'].region + '_' + bridged['bea'].industry)

  bridged['industries'] = qcew_operations.QcewToSut().sut

  dir_ = dirs.__dict__[pce_config.target.folder]
  with pd.ExcelFile(dir_ / pce_config.target.file) as xls:
    for sheet, set_index in pce_config.target.sheets.__dict__.items():
      bridged[sheet] = pd.read_excel(
        xls,
        sheet_name=sheet,
        index_col=None,
        dtype={'code': str, 'commodity': str, 'region': str,
               'parent': str, 'level': int})
      if set_index:
        bridged[sheet].set_index('code', inplace=True)
      else:
        bridged[sheet].index = (
          bridged[sheet].region + '_' + bridged[sheet].commodity)

  return utilities.DictToObject(bridged)
