"""Transform imported data."""
from .. import utilities
import pdb
import re
import pandas as pd
import numpy as np
from logging import getLogger
from copy import deepcopy
from scipy import sparse as sp
logger = getLogger('root')


def rescale_integer(
    source: pd.Series,
    tree: pd.DataFrame,
    eps: float = 1.0,
    nexit: int = 10,
    level: str = 'level',
    dim0: str = 'area',
    dim1: str = 'oind'):
  """Given 2-dim tree of floats, rescales canopy to int keeping total.

  source and tree are assumed to have the same index.

  Parameters
  ----------
  source: pd.Series(dtype=float)
    Vector with data to be rescaled
  tree: pandas.DataFrame
    Index: str
    Columns:
      Name: dim0 + '_' + level, dtype: int
      Name: dim1 + '_' + level, dtype: int
  eps: float
    Lowest difference accepted between initial and final canopy sum
  nexit: int
    Maximum number of iterations
  level: str = 'level'
    String representing 'level' in tree column names
  dim0: str = 'area',
    String representing 'first dimension' in tree column names
  dim1: str = 'oind'
    String representing 'second dimension' in tree column names

  Returns
  -------
  pd.Series(dtype=int)
  """
  logger.info('Rescaling so that integers match grand total')

  tree_tmp = deepcopy(tree)
  tree = pd.DataFrame(index=tree_tmp.index)
  tree['dim0_level'] = tree_tmp[f"{dim0}_{level}"]
  tree['dim1_level'] = tree_tmp[f"{dim1}_{level}"]
  del tree_tmp
  max_dim0 = max(tree.dim0_level)
  max_dim1 = max(tree.dim1_level)

  df = pd.DataFrame(index=tree.index)
  df['source'] = source
  df['scaled'] = df.source
  df['target'] = df.scaled.round(0).astype(int)
  select = df.loc[
    (tree.dim0_level == max_dim0)
    & (tree.dim1_level == max_dim1)].sum()
  err = abs(select.source - select.target)
  logger.info(f'Maximum number of iterations: {nexit}')
  logger.info(f'Maximum allowed difference: {eps}')
  logger.info('Iteration: scaling factor, difference')
  logger.info(f'0: 0, {err:.2f}')
  k = 0
  while err > eps:
    k += 1
    a = select.source / select.target
    df['scaled'] = (df.scaled * a)
    df['target'] = df.scaled.round(0).astype(int)
    select = df.loc[
      (tree.dim0_level == max_dim0)
      & (tree.dim1_level == max_dim1)].sum()
    err = abs(select.source - select.target)
    logger.info(f'{k}: {a:.5f}, {err:.2f}')
    if k > nexit:
      break

  logger.info('Finished rescaling integers')
  return df.target


def set_initial(
    quant: pd.Series,
    proxy: pd.Series,
    nondisclosed: pd.Series,
    tree: pd.DataFrame,
    level: str = 'level',
    parent: str = 'parent',
    dim0: str = 'area',
    dim1: str = 'oind'):
  """Creates initial estimate, given 2-dim tree constraint.

  All parameters are assumed to have the same index.
  Parent columns should be valid indices.

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
  logger.debug('Setting initial estimate')

  df = pd.DataFrame()
  old_columns = [
    f'{dim0}_{level}', f'{dim1}_{level}', f'{dim0}_{parent}',
    f'{dim1}_{parent}']
  new_columns = [
    'dim0_level', 'dim1_level', 'dim0_parent', 'dim1_parent'
  ]
  df[new_columns] = deepcopy(tree[old_columns])
  df['nondisclosed'] = deepcopy(nondisclosed)
  df['quant'] = deepcopy(quant)
  df['proxy'] = deepcopy(proxy)

  for dim in ['quant', 'proxy']:
    df.loc[df[dim] == 0, dim] = 1
  df.loc[df.nondisclosed, 'quant'] = 0

  max_dim0 = max(df.dim0_level)
  max_dim1 = max(df.dim1_level)

  for k_dim0 in range(max_dim0 + 1):
    for k_dim1 in range(max_dim1 + 1):
      if (k_dim0 == 0) & (k_dim1 == 0):
        continue

      select = deepcopy(df.loc[
        (df.dim0_level == k_dim0)
        & (df.dim1_level == k_dim1)
        & df.nondisclosed, ['quant', 'proxy']].astype(float))

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

  logger.debug('Finished setting initial estimate')
  return df.quant


def bottom_up_correction(source,
                         tree,
                         level: str = 'level',
                         parent: str = 'parent',
                         dim0: str = 'area',
                         dim1: str = 'oind',
                         keep_max: bool = False):
  """Sums up values across all levels, given 2-dim tree constraint.

  'source' and 'tree' are assumed to have the same index.
  Parent columns should be valid indices.

  Parameters
  ----------
  source: pd.Series(dtype=float)
  tree: pandas.DataFrame
    Index: str
    Columns:
      Name: dim0 + '_' + level, dtype: int
      Name: dim1 + '_' + level, dtype: int
      Name: dim0 + '_' + parent, dtype: str
      Name: dim1 + '_' + parent, dtype: str
  level: str
    String representing 'level' in tree column names
  level: str
    String representing 'parent' in tree column names
  dim0: str
    String representing 'first dimension' in tree column names
  dim1: str
    String representing 'second dimension' in tree column names
  keep_max: bool
    Whether to keep the maximum value

  Returns
  -------
  pd.Series(dtype=float)
  """
  logger.debug("Performing bottom-up correction")

  tree_tmp = deepcopy(tree)
  tree = pd.DataFrame()
  old_columns = [
    f'{dim0}_{level}', f'{dim1}_{level}', f'{dim0}_{parent}',
    f'{dim1}_{parent}']
  new_columns = [
    'area_level', 'oind_level', 'area_parent', 'oind_parent'
  ]
  tree[new_columns] = deepcopy(tree_tmp[old_columns])
  del tree_tmp
  max_area = max(tree.area_level)
  max_oind = max(tree.oind_level)
  target = deepcopy(source)

  # moving up
  for k_area in reversed(range(max_area + 1)):
    for k_oind in reversed(range(max_oind + 1)):
      logger.debug(f"area: {k_area}, oind: {k_oind}")

      if (k_area == max_area) and (k_oind == max_oind):
        continue
      elif (k_area == max_area):
        alt = 'area'
        dim = 'oind'
      else:
        dim = 'area'
        alt = 'oind'

      k_dim = eval(f"k_{dim}")
      k_alt = eval(f"k_{alt}")

      offspring_index = list(tree.loc[
        (tree[f"{dim}_level"] == k_dim + 1) &
        (tree[f"{alt}_level"] == k_alt)].index)
      parent_index = list(tree.loc[
        offspring_index, f'{dim}_parent'])

      values = target.loc[offspring_index]
      values.index = parent_index
      totals = values.groupby(values.index).sum()

      if keep_max:
        totals = totals.combine(target.loc[totals.index], max)

      target.loc[totals.index] = totals

  logger.debug("Finished bottom-up correction")

  return target


def load_codes():
  full_config = utilities.get_config('qcew')
  dirs = full_config.dirs
  config = full_config.process_codes

  codes = {}
  with pd.ExcelFile(dirs.target / config.target.file) as xls:
    for key, val in config.target.sheets.__dict__.items():
      codes[key] = pd.read_excel(
        xls,
        sheet_name=val,
        index_col='code',
        dtype={'code': str, 'parent': str, 'level': int,
               'ownership': str, 'industry': str})
      codes[key].replace({np.nan: None}, inplace=True)

  return utilities.DictToObject(codes)


def load_data(load_results=True):
  full_config = utilities.get_config('qcew')
  dirs = full_config.dirs
  source_config = full_config.process_data
  results_config = full_config.balance

  qcew = {}
  qcew_tree = pd.read_csv(
    dirs.target / source_config.target.tree,)
  qcew_tree.index = qcew_tree.current
  del qcew_tree['current']
  for dim in ['area', 'oind']:
    qcew_tree.loc[
      qcew_tree[dim + '_parent'].isna(), dim + '_parent'] = None
  qcew['tree'] = qcew_tree

  qcew_source = pd.read_csv(
    dirs.target / source_config.target.data,
    dtype={'area': str, 'oind': str})
  qcew_source.index = qcew_tree.index
  qcew['source'] = qcew_source

  qcew_previous = pd.read_csv(
    dirs.target / source_config.target.data_previous,
    dtype={'area': str, 'oind': str})
  qcew['previous'] = qcew_previous

  if load_results:
    qcew_results = pd.read_csv(
      dirs.target / results_config.target.data)
    qcew_results.index = qcew_tree.index
    qcew['results'] = qcew_results

  return utilities.DictToObject(qcew)


def load_bridge(load_comparison=True):
  full_config = utilities.get_config('qcew')
  dirs = full_config.dirs
  qcew_config = full_config.process_qcew_bridge.target

  sheet_list = ['qcew', 'sut', 'bridge']
  if load_comparison:
    sheet_list.append('comparison')

  bridge = {}
  for sheet in sheet_list:
    bridge[sheet] = pd.read_excel(
      dirs.target / qcew_config.file,
      sheet_name=sheet,
      dtype={'level': str, 'ownership': str, 'sut': str, 'qcew': str,
             'version': str},
      index_col='code')

  return utilities.DictToObject(bridge)


def load_construction():
  stage = 'qcew'
  full_config = utilities.get_config(stage)
  config = full_config.apply_bridge

  dir_ = utilities.INTERNAL_PATH / stage
  element = {}
  for key, val in config.source.blocks.__dict__.items():
    col0, row0, col1, row1 = val
    element[key] = pd.read_excel(
      dir_ / config.source.file,
      sheet_name=config.source.sheet,
      index_col=None,
      header=None,
      skiprows=row0 - 1, nrows=(1 + row1 - row0),
      usecols=f'{col0}:{col1}')
  element = utilities.DictToObject(element)
  element.columns = element.columns.transpose()
  element.data.index = list(
    element.index[element.index.columns[0]].astype(str))
  element.data.columns = list(
    element.columns[element.columns.columns[0]].astype(str))

  return element.data.fillna(0)


class QcewToSut():
  def __init__(self):
    stage = 'qcew'
    full_config = utilities.get_config(stage)
    config = full_config.apply_bridge

    self.auto_truck = config.auto_truck
    self.government = config.government

    self.codes = load_codes()
    self.construction = load_construction()

    bridge_all = load_bridge(load_comparison=False)

    self.bridge = bridge_all.bridge
    self.sut = bridge_all.sut
    self.qcew = bridge_all.qcew

    self.qcew.drop(self.codes.oind.columns, axis=1,
                   inplace=True)

    return None

  # @profile
  def apply(self, qcew):
    """"QCEW: pd.Series with index oind (ownership_industry)"""

    results = pd.DataFrame(index=self.sut.index)

    qcew_input = qcew
    qcew = pd.DataFrame()
    qcew['data'] = qcew_input
    # del qcew_input
    qcew['ownership'] = [oind.split('_')[0] for oind in qcew.index]
    qcew['industry'] = [
      '_'.join(oind.split('_')[1:]) for oind in qcew.index]

    for own in self.codes.ownership.index:
      tmp = qcew.loc[(qcew.ownership == own)]
      tmp.set_index('industry', inplace=True)
      vector = pd.Series(index=self.codes.industry.index,
                         data=int(0))
      vector.loc[tmp.index] = tmp.data
      vector = vector.astype(int)
      results[own] = self.combine(vector)

    # apply government reallocation
    logger.debug("Applying government reallocation")
    for new_key, list_ in self.government.__dict__.items():
      for val in list_:
        old_key = val.industry
        for own in val.ownership:
          results.at[new_key, own] = (
            results.at[new_key, own]
            + results.at[old_key, own])
          results.at[old_key, own] = 0
    results['0'] = (results.sum(axis=1)).astype(int)

    return results['0'].fillna(0)

  # @profile
  def combine(self, vector):
    """Receives pd.Series with QCEW industry index"""

    logger.debug(
      "Applying many-to-one QCEW to benchmark transformations")
    tmp = vector.loc[list(self.bridge.qcew)]
    tmp.index = list(self.bridge.sut)
    tmp = tmp.groupby(tmp.index).sum()
    results = pd.Series(index=self.sut.index)
    results.loc[tmp.index] = tmp
    results.fillna(0, inplace=True)
    results = results.astype(int)

    logger.debug("Applying auto and light-truck split")
    for key, val in self.auto_truck.benchmark.__dict__.items():
      logger.debug(f"({val}) {self.auto_truck.qcew_id} -> {key}")
      results.loc[key] = results.loc[key] + int(
        val * vector.loc[self.auto_truck.qcew_id])

    logger.debug("Applying construction transformations")
    results = results.astype(float)
    vector = vector.astype(float)
    results.loc[self.construction.columns] = (
      self.construction.transpose().dot(
        vector.loc[self.construction.index]))

    return results


def load_qcew_sut_sparse_bridge(qcew):
  logger.info('Loading QCEW to SUT sparse bridge')

  full_config = utilities.get_config('qcew')
  dirs = full_config.dirs
  config = full_config.test_bridge

  dir_ = dirs.__dict__[config.target.folder]
  bridge_nnz = pd.read_excel(
    dir_ / config.target.file,
    sheet_name=config.target.sheet,
    index_col='code',
    dtype=int)

  bridge_tmp = pd.DataFrame(
    index=qcew.codes.oind.index,
    columns=qcew.sut.index,
    data=int(0))
  bridge_tmp.loc[bridge_nnz.index, bridge_nnz.columns] = bridge_nnz

  index = pd.Series(data=qcew.codes.oind.index)
  columns = pd.Series(data=qcew.sut.index)

  bridge_tmp = sp.coo_array(np.array(bridge_tmp))
  bridge = {
    'trunc': bridge_nnz,
    'sparse': sp.csr_array(bridge_tmp),
    'index': index,
    'columns': columns}

  return utilities.DictToObject(bridge)


def load_qcew_multilevel_sparse_bridge(qcew):
  logger.info('Loading multilevel QCEW sparse bridge')
  positions = deepcopy(qcew.codes.oind[['level']])
  positions['offspring'] = range(len(positions))
  tmp = positions.loc[
    qcew.codes.oind.parent.iloc[1:].values,
    'offspring']
  tmp.index = positions.index[1:]
  positions['parent'] = None
  positions.loc[tmp.index, 'parent'] = tmp

  bridge = sp.coo_array(
    (np.ones(len(positions) - 1),
     (positions.iloc[1:]['offspring'].values,
      positions.iloc[1:]['parent'].values)),
    shape=(len(positions), len(positions)))

  bridge = sp.csr_array(bridge)
  bridge = bridge + sp.diags(np.ones(len(positions)), 0)

  logger.info('Number of nonzero elements after n products')
  logger.info(f'-1: {bridge.nnz}, {max(bridge.data)}')
  n = range(len(set(positions.level)))
  previous_nnz = bridge.nnz
  for k in n:
    bridge = bridge.dot(bridge)
    bridge.data = 1 + 0 * bridge.data
    logger.info(f'{k}: {bridge.nnz}, {max(bridge.data)}')
    if bridge.nnz == previous_nnz:
      break
    previous_nnz = bridge.nnz

  maxlevel = max(set(positions.level))
  maxcodes = positions.loc[positions.level == maxlevel].index
  maxrows = positions.loc[maxcodes, 'offspring'].values

  bridge = bridge[maxrows, :]
  logger.info(f'Truncated bridge shape: {bridge.shape}')

  index = pd.Series(data=qcew.codes.oind.loc[
    qcew.codes.oind.level == maxlevel].index)
  columns = pd.Series(data=qcew.codes.oind.index)

  bridge_combined = {
    'sparse': bridge,
    'index': index,
    'columns': columns}

  return utilities.DictToObject(bridge_combined)


def load_bridged_data():
  """Load QCEW employment/wages for all regions in SUT industries"""
  full_config = utilities.get_config('qcew')
  dirs = full_config.dirs
  config = full_config.apply_bridge

  dir_ = dirs.__dict__[config.target.folder]
  qcew_bridged = pd.read_csv(
    dir_ / config.target.file,
    dtype={'area': str, 'industry': str,
           'employment': int, 'wages': int})

  return qcew_bridged
