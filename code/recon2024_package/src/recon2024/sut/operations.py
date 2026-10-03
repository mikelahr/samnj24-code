"""Transform imported data."""
from recon2024.bea import operations as bea_operations
import numpy as np
import scipy.sparse as sp
from .. import utilities
import re
import pandas as pd
from logging import getLogger
from copy import deepcopy
import pdb
logger = getLogger('root')


def read_qcew_proxy():
  logger.debug("Loading comparison between 2017 and 2022 QCEW wages")
  full_config = utilities.get_config('qcew')
  dirs = full_config.dirs
  config = full_config.apply_bridge

  dir_ = dirs.__dict__[config.statistics.folder]
  df = pd.read_excel(
    dir_ / config.statistics.file,
    sheet_name=config.statistics.sheet,
    index_col='code')

  return df


class Sut():
  def __init__(self, data, codes, config, name=None,
               endogenous_sums=False):

    self.config = config
    setattr(self, 'data', data)
    setattr(self, 'codes', codes)
    self.__name__ = name

    if endogenous_sums:
      self.calculate_sums(endogenous=True)
      self.sums_info()

    return None

  def calculate_sums(self, endogenous=False):
    """If endogenous==False, sum should already exist"""

    tables = self.config.internal_codes
    if endogenous:
      self.codes.industries['tot'] = 0.5 * ((
        self.data.use_int.sum(0) + self.data.use_pri.sum(0)).T
        + self.data.sup_int.sum(0).T)
      self.codes.commodities['tot'] = 0.5 * (
        self.data.use_int.sum(1) + self.data.use_fin.sum(1)
        + self.data.sup_int.sum(1) + self.data.sup_tra.sum(1))
    else:
      for table in tables:
        if 'tot' not in self.codes.__dict__[table].columns:
          logger.error(f'Column tot missing from {table} table')
          raise ValueError

    self.codes.industries['use'] = (
      self.data.use_int.sum(0) + self.data.use_pri.sum(0)).T
    self.codes.industries['sup'] = self.data.sup_int.sum(0).T

    self.codes.commodities['use'] = (
      self.data.use_int.sum(1) + self.data.use_fin.sum(1))
    self.codes.commodities['sup'] = (
      self.data.sup_int.sum(1) + self.data.sup_tra.sum(1))

    for table in tables:
      xtmp = self.codes.__dict__[table]['tot']
      for typ_ in ['use', 'sup']:
        self.codes.__dict__[table][typ_ + '_dif'] = (
          self.codes.__dict__[table][typ_] - xtmp)
        self.codes.__dict__[table][typ_ + '_rel'] = (
          100 * utilities.safe_division(
            self.codes.__dict__[table][typ_ + '_dif'],
            xtmp))

    self.codes.industries['use_nnz'] = (
      (~(self.data.use_int == 0)).sum(0)
      + (~(self.data.use_pri == 0)).sum(0)).T
    self.codes.industries['sup_nnz'] = (
      ~(self.data.sup_int == 0)).sum(0).T

    self.codes.commodities['use_nnz'] = (
      (~(self.data.use_int == 0)).sum(1)
      + (~(self.data.use_fin == 0)).sum(1))
    self.codes.commodities['sup_nnz'] = (
      (~(self.data.sup_int == 0)).sum(1)
      + (~(self.data.sup_tra == 0)).sum(1))

    for table in tables:
      self.codes.__dict__[table] = self.codes.__dict__[table][
        ['tot', 'use', 'sup', 'use_dif', 'sup_dif',
         'use_rel', 'sup_rel', 'use_nnz', 'sup_nnz', 'title']].round(1)

    return None

  def sums_info(self):
    logger.info(
      "Discrepancies in SUT sums:\n"
      f"Type, max abs err, max rel err (%), min abs val, min nnz\n"
      "ind use, "
      f"{abs(self.codes.industries.use_dif).max():.0f}, "
      f"{abs(self.codes.industries.use_rel).max():.2f}, "
      f"{abs(self.codes.industries.use).min():.1f}, "
      f"{abs(self.codes.industries.use_nnz).min()}\n"
      "ind sup, "
      f"{abs(self.codes.industries.sup_dif).max():.0f}, "
      f"{abs(self.codes.industries.sup_rel).max():.2f}, "
      f"{abs(self.codes.industries.sup).min():.1f}, "
      f"{abs(self.codes.industries.sup_nnz).min()}\n"
      "com use, "
      f"{abs(self.codes.commodities.use_dif).max():.0f}, "
      f"{abs(self.codes.commodities.use_rel).max():.2f}, "
      f"{abs(self.codes.commodities.use).min():.1f}, "
      f"{abs(self.codes.commodities.use_nnz).min()}\n"
      "com sup, "
      f"{abs(self.codes.commodities.sup_dif).max():.0f}, "
      f"{abs(self.codes.commodities.sup_rel).max():.2f}, "
      f"{abs(self.codes.commodities.sup).min():.1f}, "
      f"{abs(self.codes.commodities.sup_nnz).min()}")

    return None

  def write_excel(self, path):
    """Write a SUT to Excel."""

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:

      for key, val in self.config.data.__dict__.items():
        df = self.data.__dict__[key]
        df.to_excel(writer, sheet_name=key, index=True)

      for key in self.config.codes.__dict__.keys():
        df = self.codes.__dict__[key]
        df.to_excel(writer, sheet_name=key, index=True)

    return None

  @staticmethod
  def read_excel(path, name, is_iot=False):
    """Reads existing SUT."""

    full_config = utilities.get_config('sut')
    if is_iot:
      config = full_config.iot_structure
    else:
      config = full_config.sut_structure

    logger.debug('Read data and codes')
    # load data
    data = {}
    for key, val in config.data.__dict__.items():
      data[key] = pd.read_excel(
        path,
        sheet_name=key,
        index_col=0)
    data = utilities.DictToObject(data)

    # load codes
    codes = {}
    for key, val in config.codes.__dict__.items():
      codes[key] = pd.read_excel(
        path,
        sheet_name=key,
        index_col=0)
    codes = utilities.DictToObject(codes)

    return Sut(data, codes, config, name=name)

  def set_to_zero(self):
    for table in self.config.data.__dict__.keys():
      self.data.__dict__[table] = 0 * self.data.__dict__[table]
    return None

  def _subtract(self, other, name=None):
    logger.debug(f"Subtracting {other.__name__} from {self.__name__}")
    return self._add(other._opposite(name=name), name=name)

  def _divide(self, other, name=None):
    logger.debug(f"Dividing {self.__name__} by {other.__name__}")
    return self._multiply(other._inverse(name=name), name=name)

  def _operator(operation='unknown'):
    """Decorator to perform arithmetic operation"""
    def decorator(func):
      def wrapper(
          self, other=None, first=None, second=None, name=None):
        if name is None:
          name = operation + '_' + self.__name__
          if other is not None:
            name = name + '_' + other.__name__
        sut = self.copy(name=name)
        for key in self.config.data.__dict__.keys():
          if other is None:
            other_arg = None
          else:
            other_arg = other.data.__dict__[key]
          sut.data.__dict__[key] = func(self, other,
                                        self.data.__dict__[key],
                                        other_arg)

        for key in ['industries', 'commodities']:
          if other is None:
            other_arg = None
          else:
            other_arg = other.codes.__dict__[key]['tot']
          sut.codes.__dict__[key]['tot'] = func(
            self, other,
            self.codes.__dict__[key]['tot'],
            other_arg)
        sut.calculate_sums()
        return sut
      return wrapper
    return decorator

  @_operator('add')
  def _add(self, other, first=None, second=None, name=None):
    return first + second

  @_operator('multiply')
  def _multiply(self, other, first=None, second=None, name=None):
    return first * second

  @_operator('opposite')
  def _opposite(self, other=None, first=None, second=None, name=None):
    return - first

  @_operator('inverse')
  def _inverse(self, other=None, first=None, second=None, name=None):
    return utilities.safe_inverse(first)

  def summarize(self):

    info_str = ''
    for key in self.config.data.__dict__.keys():
      info_str = info_str + (
        f"{key}, "
        f"{self.data.__dict__[key].min(axis=None)}, "
        f"{self.data.__dict__[key].max(axis=None)}, "
        f"{(self.data.__dict__[key] != 0).sum().sum()}\n")

    for key in self.config.internal_codes:
      info_str = info_str + (
        f"{key}, "
        f"{self.codes.__dict__[key]['tot'].min(axis=None)}, "
        f"{self.codes.__dict__[key]['tot'].max(axis=None)}, "
        f"{(self.codes.__dict__[key]['tot'] != 0).sum().sum()}\n")

    logger.info(f"Main features of {self.__name__}\n"
                f"table, min, max, nonzeros\n{info_str}")
    return None

  def copy(self, name):
    sut = deepcopy(self)
    sut.__name__ = name
    return sut

  def calculate_requirements(self, decimals=6):
    use = self.data.use_int.multiply(
      utilities.safe_inverse(self.codes.industries['tot']), axis=1)

    sup = self.data.sup_int.multiply(
      utilities.safe_inverse(self.codes.commodities['tot']), axis=0)

    full = sup.T.dot(use)

    reqs = {'use': use, 'sup': sup, 'full': full}
    setattr(self, 'reqs', utilities.DictToObject(reqs))

    logger.info(f"Applying cutoff to {decimals} decimals")
    for key, val in self.reqs.__dict__.items():
      nnz0 = (val != 0).sum().sum()
      self.reqs.__dict__[key] = val.round(decimals=decimals)
      nnz1 = (self.reqs.__dict__[key] != 0).sum().sum()
      logger.info(f"{key}: {nnz0} -> {nnz1}")

    return None

  def compare_requirements(self, other,
                           self_name='target',
                           other_name='source',
                           cutoff=0.0):
    positions = np.where(np.array(
      (self.req != 0) + (other.req != 0)))
    comparison = pd.DataFrame(columns=[
      'row', 'col', other_name, self_name, 'dif', 'rel'],
      index=range(len(positions[0])))

    comparison['row'] = self.req.index[positions[0]]
    comparison['col'] = self.req.columns[positions[1]]
    comparison[other_name] = np.array(other.req)[positions]
    comparison[self_name] = np.array(self.req)[positions]
    comparison['dif'] = comparison[self_name] - comparison[other_name]
    comparison['rel'] = comparison.dif / comparison.target * 100
    comparison.sort_values('target', inplace=True, ascending=False)

    comparison.index = comparison[[
      'row', 'col']].agg('_'.join, axis=1)

    comparison = comparison.loc[abs(comparison[self_name]) > cutoff]
    comparison = comparison.loc[abs(comparison[other_name]) > cutoff]

    setattr(self, 'req_compare', comparison)

    return None


class Balance():
  def __init__(self, suts,
               aggregate='target_agg',
               detailed='target_det'):
    logger.info("Initializing SUT balancing object")
    full_config = utilities.get_config('sut')
    # dirs = full_config.dirs
    self.config = full_config.create_target

    logger.info("Create sparse aggregate and detailed target SUT")
    vars_df = pd.DataFrame(
      columns=['level', 'table', 'row', 'col', 'source'])
    vars_df = vars_df.astype({
      'level': int, 'table': str, 'row': str, 'col': str,
      'source': float})

    # grand totals
    for table_key in suts.bridge.__dict__.keys():
      vars_tmp = pd.DataFrame([
        {'level': int(0), 'table': 'T_' + table_key, 'row': 'T',
         'col': 'T', 'source':
           suts.__dict__[aggregate].codes.__dict__[
             table_key]['tot'].sum()}])
      vars_df = pd.concat([vars_df, deepcopy(vars_tmp)], axis=0)

    # aggregate and detailed SUTs
    for level_key, level_val in {aggregate: int(1),
                                 detailed: int(2)}.items():

      # sums
      for table_key in suts.bridge.__dict__.keys():
        values = suts.__dict__[level_key].codes.__dict__[
          table_key]['tot']
        positions = np.where(values)
        vars_tmp = pd.DataFrame(columns=vars_df.columns)
        vars_tmp['source'] = np.array(values)[positions]
        vars_tmp['row'] = values.index[positions[0]]
        vars_tmp['col'] = values.index[positions[0]]
        vars_tmp['table'] = table_key
        vars_tmp['level'] = level_val
        vars_df = pd.concat([vars_df, deepcopy(vars_tmp)], axis=0)

      # other tables
      for table_key in suts.config.data.__dict__.keys():
        values = suts.__dict__[level_key].data.__dict__[table_key]
        positions = np.where(values)
        vars_tmp = pd.DataFrame(columns=vars_df.columns)
        vars_tmp['source'] = np.array(values)[positions]
        vars_tmp['row'] = values.index[positions[0]]
        vars_tmp['col'] = values.columns[positions[1]]
        vars_tmp['table'] = table_key
        vars_tmp['level'] = level_val
        vars_df = pd.concat([vars_df, deepcopy(vars_tmp)], axis=0)

    vars_df.reset_index(inplace=True, drop=True)

    logger.info(
      "Aggregates and detailed va/pce are semi-flexible (flag==1)")
    vars_df['flag'] = deepcopy(vars_df['level'])

    vars_df.loc[
      (vars_df.level == 1), 'flag'] = int(0)

    vars_df.loc[
      (vars_df.table == 'use_pri')
      & (vars_df.level == 2), 'flag'] = int(1)

    vars_df.loc[
      (vars_df.table == 'use_fin')
      & (vars_df.col == 'pce')
      & (vars_df.level == 2), 'flag'] = int(1)

    # to not adjust negative values
    # vars_df.loc[vars_df.source < 0, 'flag'] = int(1)

    setattr(self, 'variables', vars_df)

    logger.info(
      "Create constraints of aggregate and detailed target SUT")

    eqs_df = pd.DataFrame(
      columns=[
        'const_table', 'const_row', 'const_col',
        'var_level', 'var_table', 'var_row', 'var_col',
        'val'])

    logger.info("Creating grand totals to sums components")
    for table_key, bridge in suts.bridge.__dict__.items():
      # agg sums to grand totals: grand totals
      eqs_tmp = {}
      eqs_tmp['const_table'] = f'agg_tot_{table_key}'
      eqs_tmp['const_row'] = 'T'
      eqs_tmp['const_col'] = 'T'
      eqs_tmp['var_level'] = int(0)
      eqs_tmp['var_table'] = 'T_' + table_key
      eqs_tmp['var_row'] = 'T'
      eqs_tmp['var_col'] = 'T'
      eqs_tmp['val'] = -1
      eqs_tmp = pd.DataFrame([eqs_tmp])
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      # aggregate sums, used 4 times
      agg_sums = vars_df.loc[
        (vars_df.level == 1) & (vars_df.table == table_key)]
      eqs_tmp = pd.DataFrame(columns=eqs_df.columns)
      eqs_tmp['var_row'] = list(agg_sums.row)
      eqs_tmp['var_col'] = list(agg_sums.col)
      eqs_tmp['var_level'] = int(1)
      eqs_tmp['var_table'] = table_key

      # agg sums to grand totals: agg sums
      eqs_tmp['const_table'] = f'agg_tot_{table_key}'
      eqs_tmp['const_row'] = 'T'
      eqs_tmp['const_col'] = 'T'
      eqs_tmp['val'] = 1
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      # det sums to agg sums: agg sums
      eqs_tmp['const_table'] = f'det_agg_{table_key}'
      eqs_tmp['const_row'] = list(agg_sums.row)
      eqs_tmp['const_col'] = list(agg_sums.col)
      eqs_tmp['val'] = -1
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      # agg use to agg sums: agg sums
      eqs_tmp['const_table'] = f'agg_use_{table_key}'
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      # agg sup to agg sums: agg sums
      eqs_tmp['const_table'] = f'agg_sup_{table_key}'
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      # detailed sums, used 3 times
      det_sums = vars_df.loc[
        (vars_df.level == 2) & (vars_df.table == table_key)]
      eqs_tmp = pd.DataFrame(columns=eqs_df.columns)
      eqs_tmp['var_row'] = list(det_sums.row)
      eqs_tmp['var_col'] = list(det_sums.col)
      eqs_tmp['var_level'] = int(2)
      eqs_tmp['var_table'] = table_key

      # det sums to agg sums: det sums
      eqs_tmp['const_table'] = f'det_agg_{table_key}'
      eqs_tmp['const_row'] = bridge.table.loc[
        det_sums.row, 'parent_code'].values
      eqs_tmp['const_col'] = bridge.table.loc[
        det_sums.col, 'parent_code'].values
      eqs_tmp['val'] = 1
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      # det use to det sums: det sums
      eqs_tmp['const_table'] = f'det_use_{table_key}'
      eqs_tmp['const_row'] = det_sums.row.values
      eqs_tmp['const_col'] = det_sums.col.values
      eqs_tmp['val'] = -1
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      # det sup to det sums: det sums
      eqs_tmp['const_table'] = f'det_sup_{table_key}'
      eqs_tmp['val'] = -1
      eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

      del table_key, bridge

    logger.info("Creating SUT to sums components")
    # det/agg use/sup to det/agg sums: det/agg use/sup
    for level_key, level_val in {'agg': 1, 'det': 2}.items():
      for table_key, table_val in suts.config.data.__dict__.items():
        prefix = table_key.split('_')[0]
        values = vars_df.loc[
          (vars_df.level == level_val) & (vars_df.table == table_key)]
        eqs_tmp = pd.DataFrame(columns=eqs_df.columns)
        eqs_tmp['var_row'] = values.row.values
        eqs_tmp['var_col'] = values.col.values
        eqs_tmp['var_level'] = level_val
        eqs_tmp['var_table'] = table_key
        eqs_tmp['val'] = 1

        if table_val.index in suts.bridge.__dict__.keys():
          eqs_tmp['const_table'] = (
            f'{level_key}_{prefix}_{table_val.index}')
          eqs_tmp['const_row'] = values.row.values
          eqs_tmp['const_col'] = values.row.values
          eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)
        if table_val.columns in suts.bridge.__dict__.keys():
          eqs_tmp['const_table'] = (
            f'{level_key}_{prefix}_{table_val.columns}')
          eqs_tmp['const_row'] = values.col.values
          eqs_tmp['const_col'] = values.col.values
          eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

    logger.info("Creating SUT aggregation components")
    for table_key, table_val in suts.config.data.__dict__.items():
      for level_val in [1, 2]:
        values = vars_df.loc[
          (vars_df.level == level_val) & (vars_df.table == table_key)]
        eqs_tmp = pd.DataFrame(columns=eqs_df.columns)
        eqs_tmp['var_row'] = values.row.values
        eqs_tmp['var_col'] = values.col.values
        eqs_tmp['var_level'] = level_val
        eqs_tmp['var_table'] = table_key
        eqs_tmp['const_table'] = f'det_agg_{table_key}'
        eqs_tmp['const_row'] = values.row.values
        eqs_tmp['const_col'] = values.col.values

        if level_val == 1:
          eqs_tmp['val'] = -1
          eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)
        else:
          eqs_tmp['val'] = 1
          for dim, pos in [['index', 'row'], ['columns', 'col']]:
            if table_val.__dict__[dim] in suts.bridge.__dict__.keys():
              eqs_tmp['const_' + pos] = suts.bridge.__dict__[
                table_val.__dict__[dim]].table.loc[
                  values[pos], 'parent_code'].values
          eqs_df = pd.concat([eqs_df, deepcopy(eqs_tmp)], axis=0)

    logger.info("Extracting constraint indices")
    eqs_df.reset_index(inplace=True, drop=True)
    eqs_df = eqs_df.astype({
      'const_table': str, 'const_row': str, 'const_col': str,
      'var_level': int, 'var_table': str, 'var_row': str,
      'var_col': str, 'val': int})

    errors_df = pd.DataFrame()

    tmp = eqs_df[['const_table', 'const_row', 'const_col']].agg(
      ':'.join, axis=1)
    tmp.drop_duplicates(inplace=True)
    tmp.reset_index(inplace=True, drop=True)

    errors_df[['table', 'row', 'col']] = tmp.str.split(
      ':', expand=True)

    setattr(self, 'equations', eqs_df)
    setattr(self, 'errors', errors_df)

    logger.info("Creating balance object")

    self.variables['level'] = self.variables['level'].astype(str)
    t = pd.DataFrame(
      index=self.variables[['level', 'table', 'row', 'col']].agg(
        '_'.join, axis=1),)
    self.variables['level'] = self.variables['level'].astype(int)
    t['pos'] = range(len(t))
    t['val'] = self.variables.source.values
    t['flag'] = self.variables.flag.values

    k = pd.DataFrame(
      index=self.errors[['table', 'row', 'col']].agg(
        '_'.join, axis=1))
    k['pos'] = range(len(k))

    g = pd.DataFrame()
    g['val'] = self.equations.val
    g['row_code'] = self.equations[[
      'const_table', 'const_row', 'const_col']].agg('_'.join, axis=1)
    self.equations['var_level'] = self.equations[
      'var_level'].astype(str)
    g['col_code'] = self.equations[[
      'var_level', 'var_table', 'var_row', 'var_col']].agg(
        '_'.join, axis=1)
    self.equations['var_level'] = self.equations[
      'var_level'].astype(int)
    g['row'] = k.loc[g.row_code.values, 'pos'].values
    g['col'] = t.loc[g.col_code.values, 'pos'].values

    bal = utilities.Balance.from_array(t, g, self.config.params)

    self.errors['source'] = bal.err.toarray()
    self.balance = bal

    # self.variables.index = t.index
    # self.equations.index = g[[
    #  'row_code', 'col_code']].agg('_'.join, axis=1)
    # self.errors.index = k.index

    logger.info("Constraint checks, should all be zero")
    logger.info("Constraints with parameters < -1: "
                f"{((bal.g < -1).sum(1) != 0).sum()}")
    logger.info("Constraints without parameter -1: "
                f"{((bal.g == - 1).sum(1) == 0).sum()}")
    logger.info("Constraints with parameters > 1: "
                f"{((bal.g > 1).sum(1) != 0).sum()}")
    logger.info("Constraints without parameter 1: "
                f"{((bal.g == 1).sum(1) == 0).sum()}")
    logger.info("Variables without parameters: "
                f"{((abs(bal.g)).sum(0) == 0).sum()}")

    return None

  def execute(self):
    logger.info("Balancing detailed target data")

    self.balance.run()

    self.variables['target'] = self.balance.tval
    self.errors['target'] = self.balance.err.toarray()

    tmp = self.balance.g.dot(sp.spdiags(
      np.sign(self.balance.tval),
      0,
      len(self.balance.tval),
      len(self.balance.tval)))
    self.errors['agg_pos'] = ((tmp < 0) * (self.balance.g < 0)).sum(1)
    self.errors['agg_neg'] = ((tmp > 0) * (self.balance.g < 0)).sum(1)
    self.errors['det_pos'] = ((tmp > 0) * (self.balance.g > 0)).sum(1)
    self.errors['det_neg'] = ((tmp < 0) * (self.balance.g > 0)).sum(1)

    return None

  def export(self, suts, source, target):
    logger.info("Converting balanced results to SUT format")

    sut = suts.__dict__[source].copy(name=target)

    for key_table in sut.config.codes.__dict__.keys():
      sut.codes.__dict__[key_table]['position'] = range(len(
        sut.codes.__dict__[key_table]))

    for key_table, val_table in sut.config.data.__dict__.items():
      row_codes = sut.codes.__dict__[val_table.index]
      col_codes = sut.codes.__dict__[val_table.columns]

      select = self.variables.loc[
        (self.variables.level == 2)
        & (self.variables.table == key_table)]
      rows = list(row_codes.loc[select.row, 'position'])
      cols = list(col_codes.loc[select.col, 'position'])

      tmp = np.zeros((
        len(row_codes),
        len(col_codes)))
      tmp[(rows, cols)] = select.target.values

      sut.data.__dict__[key_table] = pd.DataFrame(
        index=row_codes.index,
        columns=col_codes.index,
        data=tmp)

    for key_table in sut.config.internal_codes:
      row_codes = sut.codes.__dict__[key_table]

      select = self.variables.loc[
        (self.variables.level == 2)
        & (self.variables.table == key_table)]
      rows = list(row_codes.loc[select.row, 'position'])

      tmp = np.zeros((len(row_codes)),)
      tmp[(rows)] = select.target.values

      sut.codes.__dict__[key_table]['tot'] = tmp

    for key_table in sut.config.codes.__dict__.keys():
      sut.codes.__dict__[key_table].drop(
        'position', axis=1, inplace=True)

    sut.calculate_sums()
    sut.sums_info()
    setattr(suts, sut.__name__, sut)
    return suts


class Suts():
  def __init__(self,
               load_bridge=True):
    logger.info("Loading SUT data and codes")
    full_config = utilities.get_config('sut')
    dirs = full_config.dirs
    load_config = full_config.process_suts
    struct_config = full_config.sut_structure
    target_config = full_config.create_target.target

    self.config = struct_config
    sut_info = {}
    for key, val in load_config.suts.__dict__.items():
      sut_info[key] = val.info

    setattr(self.config, 'suts', utilities.DictToObject(sut_info))

    for key, val in load_config.suts.__dict__.items():
      logger.info(f"{key}: {val.info}")
      path = dirs.__dict__[load_config.folder] / val.file
      sut = Sut.read_excel(path, key)
      setattr(self, key, sut)

    if load_bridge:
      self.read_bridge()

    return None

  def read_bridge(self):
    logger.debug(
      "Loading bridges and creating aggregation matrices")
    full_config = utilities.get_config('sut')
    dirs = full_config.dirs
    bridge_config = full_config.create_bridge

    def create_aggregation_matrix(bridge, offspring, parent):
      """Converts bridge from dense to sparse format"""
      index_df = pd.Series(
        index=parent.index, data=range(len(parent)))
      columns_df = pd.Series(index=offspring.index,
                             data=range(len(offspring)))

      index_pos = list(index_df.loc[list(bridge.parent_code)])
      columns_pos = list(columns_df.loc[list(bridge.index)])

      data = np.zeros((len(parent), len(offspring)))
      data[index_pos,
           columns_pos] = 1
      matrix = pd.DataFrame(index=parent.index,
                            columns=offspring.index,
                            data=data).astype(int)

      return matrix

    bridge = {}
    dir_ = dirs.__dict__[bridge_config.target.folder]
    for key in bridge_config.codes:
      bridge[key] = pd.read_excel(
        dir_ / bridge_config.target.file,
        sheet_name=key,
        index_col='offspring_code')

    bridge = utilities.DictToObject(bridge)

    bridge_results = {}
    for key, val in bridge.__dict__.items():
      bridge_results[key] = {}
      offspring = self.source_det.codes.__dict__[key]
      parent = self.source_agg.codes.__dict__[key]

      bridge_results[key]['matrix'] = create_aggregation_matrix(
        val,
        offspring,
        parent)
      bridge_results[key]['offspring'] = offspring[['title']]
      bridge_results[key]['parent'] = parent[['title']]
      bridge_results[key]['table'] = val

      logger.debug(
        f"{key}: {(bridge_results[key]['matrix'].sum(0) == 0).sum()}; "
        f"{(bridge_results[key]['matrix'].sum(1) == 0).sum()}")

    setattr(self, 'bridge', utilities.DictToObject(bridge_results))

    return None

  def aggregate_sut(self, source_offspring, target_parent):
    logger.debug(f"Aggregating {source_offspring} to {target_parent}")

    sut = self.__dict__[source_offspring].copy(name=target_parent)

    config = self.config.data
    for table_key, table_val in config.__dict__.items():
      index_code = table_val.index
      if index_code in self.bridge.__dict__.keys():
        sut.data.__dict__[table_key] = (
          self.bridge.__dict__[index_code].matrix).dot(
            sut.data.__dict__[table_key])

      column_code = table_val.columns
      if column_code in self.bridge.__dict__.keys():
        sut.data.__dict__[table_key] = sut.data.__dict__[
          table_key].dot(self.bridge.__dict__[column_code].matrix.T)

    for code_key, code_val in self.bridge.__dict__.items():
      df = deepcopy(code_val.parent)
      df['tot'] = code_val.matrix.dot(
        sut.codes.__dict__[code_key][['tot']])
      sut.codes.__dict__[code_key] = df
    sut.calculate_sums()

    self.add_sut(sut, sut.__name__)

    return None

  def disaggregate_sut(self, source_parent, target_offspring):
    logger.debug(
      f"Disggregating {source_parent} to {target_offspring}")

    sut = self.__dict__[source_parent].copy(name=target_offspring)

    config = self.config.data
    for table_key, table_val in config.__dict__.items():
      index_code = table_val.index
      if index_code in self.bridge.__dict__.keys():
        sut.data.__dict__[table_key] = (
          self.bridge.__dict__[index_code].matrix).T.dot(
            sut.data.__dict__[table_key])

      column_code = table_val.columns
      if column_code in self.bridge.__dict__.keys():
        sut.data.__dict__[table_key] = sut.data.__dict__[
          table_key].dot(self.bridge.__dict__[column_code].matrix)

    for code_key, code_val in self.bridge.__dict__.items():
      df = deepcopy(code_val.offspring)
      df['tot'] = code_val.matrix.T.dot(
        sut.codes.__dict__[code_key][['tot']])
      sut.codes.__dict__[code_key] = df
    sut.calculate_sums()

    self.add_sut(sut, sut.__name__)

    return None

  def create_initial_target(self, keep_intermediates=False):
    logger.info("Create initial estimate of target SUT")

    logger.info("Replacing source aggregate from the bottom up")
    self.aggregate_sut(
      source_offspring='source_det',
      target_parent='clean_agg')

    logger.info("Maximum absolute differences source and bottom up")
    sut_diff = self.source_agg._subtract(self.clean_agg)
    sut_diff.summarize()
    del sut_diff

    logger.info("Initial disaggregate discrepancies")
    self.source_det.calculate_sums(endogenous=True)
    self.source_det.sums_info()
    sut = self.source_det.copy(name='clean_det')

    logger.info(
      "Checking and correcting source-target aggregate problems")
    for table, table_info in self.config.data.__dict__.items():
      source_table = self.clean_agg.data.__dict__[table]
      target_table = self.target_agg.data.__dict__[table]
      neg_to_pos = (source_table <= 0) * (target_table > 0)
      pos_to_neg = (source_table >= 0) * (target_table < 0)
      nonz_to_zero = (~(source_table == 0)) * (target_table == 0)
      pos_match = np.where(neg_to_pos + pos_to_neg + nonz_to_zero)
      if len(pos_match[0]) == 0:
        continue

      logger.info(f"Found entries to reset in table {table}")
      logger.info("Row, column, source, target -> disaggregate")
      for k_index, k_column in zip(*pos_match):
        parent_index = target_table.index[k_index]
        parent_column = target_table.columns[k_column]
        parent_value = target_table.at[parent_index, parent_column]
        source_value = source_table.at[parent_index, parent_column]
        logger.info(
          f"{parent_index}, {parent_column}, {source_value}, "
          f"{parent_value} -> {np.sign(parent_value)}")

        if table_info.index in self.bridge.__dict__.keys():
          bridge_df = self.bridge.__dict__[table_info.index].table
          child_indices = list(bridge_df.loc[
            bridge_df.parent_code == parent_index].index)
        else:
          child_indices = [parent_index]

        if table_info.columns in self.bridge.__dict__.keys():
          bridge_df = self.bridge.__dict__[table_info.columns].table
          child_columns = list(bridge_df.loc[
            bridge_df.parent_code == parent_column].index)
        else:
          child_columns = [parent_column]

        for child_index in child_indices:
          for child_column in child_columns:
            sut.data.__dict__[table].at[
              child_index, child_column] = np.sign(parent_value)

    setattr(self, sut.__name__, sut)

    logger.info("Replacing source aggregate from the bottom up")
    self.aggregate_sut(
      source_offspring='clean_det',
      target_parent='clean_agg')

    logger.info("Final disaggregate discrepancies")
    self.clean_det.calculate_sums()  # endogenous=True)
    self.clean_det.sums_info()

    logger.info(
      "Create initial estimate using BEA value added and "
      "personal consumption expenditure")

    pce_stat, gdp_stat = load_bea_statistics()
    sut = self.clean_det.copy(name='initial_det')

    sut.data.use_fin['pce'] = pce_stat.det.final
    for va_key, va_val in gdp_stat.__dict__.items():
      sut.data.use_pri.loc[va_key] = va_val.det.result_vasut

    logger.info("Scaling other tables with wages ratio")
    industry_ratio = utilities.safe_division(
      sut.data.use_pri.loc['wages'],
      self.clean_det.data.use_pri.loc['wages'])
    average_ratio = (
      sut.data.use_pri.loc['wages'].sum()
      / self.clean_det.data.use_pri.loc['wages'].sum())

    industry_ratio = industry_ratio * (industry_ratio > 0)
    industry_ratio = (
      industry_ratio +
      (industry_ratio == 0) * average_ratio)

    logger.info("Min, average, max:\n\t"
                f"{industry_ratio.min():0.2f}, "
                f"{average_ratio:0.2f}, "
                f"{industry_ratio.max():0.2f}")

    sut.data.use_int = sut.data.use_int.multiply(
      industry_ratio, axis=1)
    sut.data.sup_int = sut.data.sup_int.multiply(
      industry_ratio, axis=1)
    for column in sut.data.use_fin.columns:
      if column != 'pce':
        sut.data.use_fin[column] = sut.data.use_fin[column] * (
          average_ratio)
    for column in sut.data.sup_tra.columns:
      sut.data.sup_tra[
        column] = sut.data.sup_tra[column] * average_ratio

    sut.codes.industries['tot'] = (
      sut.codes.industries['tot'] * industry_ratio)
    sut.codes.commodities['tot'] = (
      sut.codes.commodities['tot'] * average_ratio)

    sut.calculate_sums()
    sut.sums_info()
    self.add_sut(sut, sut.__name__)

    self.add_sut(
      source_sut=self.initial_det,
      target_name='target_det')

    """
    logger.info("Scaling with aggregate data")
    self.aggregate_sut(
      source_offspring='initial_det',
      target_parent='initial_agg')

    sut = self.target_agg._divide(self.initial_agg, name='ratio_agg')
    self.add_sut(sut, sut.__name__)
    self.ratio_agg.summarize()

    self.disaggregate_sut(
      source_parent='ratio_agg',
      target_offspring='ratio_det')

    sut = self.initial_det._multiply(self.ratio_det, name='scaled_det')
    self.add_sut(sut, sut.__name__)
    self.scaled_det.sums_info()

    self.aggregate_sut(
      source_offspring='scaled_det',
      target_parent='scaled_agg')

    self.add_sut(
      source_sut=self.scaled_det,
      target_name='target_det')
    self.target_det.calculate_sums()
    self.target_det.sums_info()
    """

    """
    logger.info("Resetting intermediate use")
    self.target_det.data.use_int = deepcopy(
      self.clean_det.data.use_int)
    new_sum = (self.target_det.industries['tot']
               - self.target_det.use_pri.sum(axis=0))
    old_sum = (self.target_det.use_int.sum(axis=0))
    factor = utilities.safe_division(new_sum, old_sum)
    self.target_det.use_int = (
      self.target_det.use_int._multiply(factor, axis=1))

    self.target_det.calculate_sums()
    self.target_det.sums_info()

    logger.info("Resetting intermediate supply")
    self.target_det.sup_int = deepcopy(self.clean_det.sup_int)
    new_sum = self.target_det.industries['tot']
    old_sum = (self.target_det.sup_int.sum(axis=0))
    factor = utilities.safe_division(new_sum, old_sum)
    self.target_det.sup_int = (
      self.target_det.sup_int._multiply(factor, axis=1))

    self.target_det.calculate_sums(endogenous=True)
    self.target_det.sums_info()
    """

    if not keep_intermediates:
      for prefix in ['clean', 'initial', 'scaled', 'ratio']:
        for suffix in ['det', 'agg']:
          del_var = prefix + '_' + suffix
          if del_var in self.__dict__.keys():
            delattr(self, del_var)

    return None

  def add_sut(self, source_sut: Sut, target_name: str):
    sut = source_sut.copy(name=target_name)
    setattr(self, sut.__name__, sut)
    setattr(self.config.suts, target_name, None)
    return None


def extract_excel(config, struct_config, dirs):
  "Extract data and codes from excel files"

  logger.debug('Loading codes')
  codes = {}
  for key, val in config.codes.__dict__.items():
    dir_ = dirs.__dict__[config.source.folder]
    col0, row0, col1, row1 = val.data
    logger.debug(f'\t{key}')
    element = pd.read_excel(
      dir_ / config.source.files.__dict__[val.file],
      sheet_name=config.source.sheet,
      index_col=None,
      header=None,
      skiprows=row0 - 1,
      nrows=(1 + row1 - row0),
      usecols=f'{col0}:{col1}')
    if 'transpose' in val.__dict__.keys():
      element = element.transpose()
    element.columns = val.columns
    element['code'] = element['code'].astype(str)
    element['code'] = [str_.strip()  # upper().
                       for str_ in list(element['code'])]
    element.set_index('code', inplace=True)
    if 'skip' in val.__dict__.keys():
      for skip in val.skip:
        element.drop(skip, inplace=True)
    codes[key] = element

  codes = utilities.DictToObject(codes)

  logger.debug('Loading data')
  data = {}
  for key, val in config.data.__dict__.items():
    logger.debug(f'\t{key}')
    element = {'data': None, 'index': None, 'columns': None}
    for elem_key in element.keys():
      logger.debug(f'\t\t{elem_key}')
      dir_ = dirs.__dict__[config.source.folder]
      col0, row0, col1, row1 = val.__dict__[elem_key]
      element[elem_key] = pd.read_excel(
        dir_ / config.source.files.__dict__[val.file],
        sheet_name=config.source.sheet,
        index_col=None,
        header=None,
        skiprows=row0 - 1,
        nrows=(1 + row1 - row0),
        usecols=f'{col0}:{col1}')
    logger.debug('\t\tcombining elements')
    element = utilities.DictToObject(element)
    element.columns = element.columns.transpose()

    element.index[element.index.columns[0]] = (
      element.index[element.index.columns[0]].astype(str))
    element.index[element.index.columns[0]] = [
      str_.strip()  # upper().
      for str_ in list(
        element.index[element.index.columns[0]])]

    element.columns[element.columns.columns[0]] = (
      element.columns[element.columns.columns[0]].astype(str))
    element.columns[element.columns.columns[0]] = [
      str_.strip()  # .upper()
      for str_ in list(
        element.columns[element.columns.columns[0]])]

    element.data.index = list(
      element.index[element.index.columns[0]])
    element.data.columns = list(
      element.columns[element.columns.columns[0]])

    logger.debug('\t\tremoving non-numerical')
    for column in element.data.columns:
      element.data[column] = pd.to_numeric(
        element.data[column], errors='coerce')
    element.data.fillna(0, inplace=True)

    logger.debug('\t\tlinking using codes')
    mapping = struct_config.data.__dict__[key]
    ind_key = codes.__dict__[mapping.index].index
    col_key = codes.__dict__[mapping.columns].index
    element.data = element.data.loc[element.data.index.isin(ind_key)]
    element.data = element.data.transpose()
    element.data = element.data.loc[element.data.index.isin(col_key)]
    element.data = element.data.transpose()
    data[key] = pd.DataFrame(
      data=0,
      index=ind_key,
      columns=col_key,)
    data[key].loc[list(element.data.index),
                  list(element.data.columns)] = (
      element.data)
  data = utilities.DictToObject(data)

  return data, codes



def extract_construction(config, struct_config, dirs):
  """RECON2024: read the 421-sector construction package
  (Use_revised_NAICS421_{year}.csv, Supply_revised_NAICS421_{year}.csv) into
  the same (data, codes) structure extract_excel returns for the 2017
  benchmark, with the raw BEA codes for final demand (F0xxxx), value added
  (V00100, T00OTOP, V00300) and supply transformations (MCIF ... SUB), so the
  benchmark code bridges apply unchanged. Titles: 2017 benchmark for the
  shared codes, the construction sheet for the 31 NAICS industries."""
  dir_ = dirs.__dict__[config.source.folder]
  use = pd.read_csv(dir_ / config.source.files.use, index_col=0)
  sup = pd.read_csv(dir_ / config.source.files.supply, index_col=0)
  use.index = [str(x).strip() for x in use.index]; use.columns = [str(x).strip() for x in use.columns]
  sup.index = [str(x).strip() for x in sup.index]; sup.columns = [str(x).strip() for x in sup.columns]

  fd = [c for c in use.columns if c.startswith('F')]
  va = [r for r in use.index if r in config.value_added_rows]
  tr = [c for c in sup.columns if c in config.trade_columns]
  com = [r for r in use.index if r not in va and r not in config.skip_rows]
  ind = [c for c in use.columns if c not in fd and c not in config.skip_columns]
  assert set(com) == set(sup.index), 'use and supply commodity sets differ'
  assert set(ind) <= set(sup.columns), 'industry sets differ'

  # titles
  titles = {}
  bench = utilities.get_config('sut').__dict__[config.titles.benchmark_config]
  for key in ['commodities', 'industries', 'final', 'primary', 'trade']:
    val = bench.codes.__dict__[key]
    col0, row0, col1, row1 = val.data
    el = pd.read_excel(dirs.__dict__[bench.source.folder] / bench.source.files.__dict__[val.file],
                       sheet_name=bench.source.sheet, header=None,
                       skiprows=row0 - 1, nrows=(1 + row1 - row0), usecols=f'{col0}:{col1}')
    if 'transpose' in val.__dict__:
      el = el.transpose()
    el.columns = val.columns
    for c, t in zip(el['code'].astype(str), el['title'].astype(str)):
      titles[c.strip()] = t.strip()
  con = pd.read_excel(utilities.INTERNAL_PATH / 'qcew' / config.titles.construction_file,
                      sheet_name=config.titles.construction_sheet, header=None)
  for c, t in zip(con.iloc[2, 3:].astype(str), con.iloc[3, 3:].astype(str)):
    titles[c.strip()] = t.strip()

  def code_df(codes):
    return pd.DataFrame({'title': [titles.get(c, c) for c in codes]},
                        index=pd.Index(codes, name='code'))
  codes = utilities.DictToObject({
    'commodities': code_df(com), 'industries': code_df(ind), 'final': code_df(fd),
    'primary': code_df(va), 'trade': code_df(tr)})

  data = {
    'use_int': use.loc[com, ind].astype(float),
    'use_fin': use.loc[com, fd].astype(float),
    'use_pri': use.loc[va, ind].astype(float),
    'sup_int': sup.loc[com, ind].astype(float),
    'sup_tra': sup.loc[com, tr].astype(float)}
  for k in data:
    data[k] = data[k].fillna(0.0)
  logger.info(f"construction SUT: {len(com)} commodities x {len(ind)} industries, "
              f"{len(fd)} final, {len(va)} primary, {len(tr)} trade columns; "
              f"output {data['sup_int'].values.sum():,.0f}")
  return utilities.DictToObject(data), utilities.DictToObject(codes.__dict__)


def process_sut(data, codes, config, struct_config):
  """Swaps some signs, reallocates margins, aggregates external."""

  (source_data, source_codes) = (data, codes)
  del data, codes

  # change negative to positive and merge code
  logger.info('Change sign of VA subsidies and merge with tax')
  table_ = config.subtract_subsidy.table
  subsidy_ = config.subtract_subsidy.subsidy
  if subsidy_ in source_data.__dict__[table_].index:
    tax_ = config.subtract_subsidy.tax
    source_data.__dict__[table_].loc[tax_] = (
      source_data.__dict__[table_].loc[tax_]
      - source_data.__dict__[table_].loc[subsidy_])
    source_data.__dict__[table_].loc[subsidy_] = 0
  else:
    logger.info('\tSkipping')

  # reset non-internal codes
  target_codes = deepcopy(source_codes)
  for key, val in config.bridge.__dict__.items():
    if key not in struct_config.internal_codes:
      target_codes.__dict__[key] = val.codes

  # aggregate primary, final and trade
  target_data = deepcopy(source_data)
  for key, val in struct_config.data.__dict__.items():
    target_data.__dict__[key] = source_data.__dict__[key]

    if val.index not in struct_config.internal_codes:
      target_data.__dict__[key] = (
        config.bridge.__dict__[val.index].matrix.dot(
          target_data.__dict__[key]))

    target_data.__dict__[key] = (
      target_data.__dict__[key].loc[
        target_codes.__dict__[val.index].index])

    if val.columns not in struct_config.internal_codes:
      target_data.__dict__[key] = (
        target_data.__dict__[key].dot(
          config.bridge.__dict__[val.columns].matrix.T))

    target_data.__dict__[key] = (
      target_data.__dict__[key][
        target_codes.__dict__[val.columns].index])

  # handle margins
  logger.info('Reallocate margins')

  # add new entries in codes and data
  target_codes.industries = pd.concat([
    target_codes.industries,
    target_codes.trade.loc[config.reallocate_margins.codes]], axis=0)

  for key, val in struct_config.data.__dict__.items():
    if val.index == 'industries':
      target_data.__dict__[
        key].loc[config.reallocate_margins.codes] = 0
    if val.columns == 'industries':
      target_data.__dict__[key][config.reallocate_margins.codes] = 0

  # reallocate margins
  codes_ = config.reallocate_margins.codes
  source_ = config.reallocate_margins.source
  target_ = config.reallocate_margins.target
  for target_, sign_ in target_.__dict__.items():
    tmp = sign_ * target_data.__dict__[source_][codes_]
    tmp[tmp < 0] = 0
    target_data.__dict__[target_][codes_] = tmp
  target_data.__dict__[source_].drop(codes_, axis=1, inplace=True)
  target_codes.trade.drop(codes_, axis=0, inplace=True)
  del codes_, source_, target_, sign_, tmp

  # reset index name
  for key in struct_config.data.__dict__.keys():
    target_data.__dict__[key].index.name = 'code'

  for key in struct_config.codes.__dict__.keys():
    target_codes.__dict__[key].index.name = 'code'

  # set nan to zero
  for key in target_data.__dict__.keys():
    target_data.__dict__[key].fillna(0, inplace=True)

  return target_data, target_codes


def create_aggregation_bridge_codes(parent, offspring,
                                    new_pairs):
  """Creates bridge using numerical codes plus adhoc pairs."""

  def extract_numbers(string):
    match = re.match(r'^[0-9]+', string)
    return match.group(0) if match else ""

  codes = {'parent': parent, 'offspring': offspring}
  del parent, offspring

  codes0 = {}
  for key, val in codes.items():
    df = codes[key]
    df.sort_index(inplace=True)
    df['trunc'] = [extract_numbers(x) for x in df.index]
    df.sort_values('trunc', inplace=True)
    df['len'] = [len(str_) for str_ in df.trunc]
    df['pos'] = range(len(df))
    codes[key] = df

    codes0[key] = deepcopy(codes[key])
    codes[key] = codes[key].loc[~ (codes[key].len == 0)]

  bmk = deepcopy(codes['offspring'])
  bmk0 = deepcopy(codes0['offspring'])
  agg = deepcopy(codes['parent'])
  agg0 = deepcopy(codes0['parent'])

  results = pd.DataFrame(
    columns=['offspring_code', 'parent_code',
             'match',
             'parent_title', 'offspring_title'],
    index=bmk0.index,
  )
  results['match'] = False
  results['offspring_code'] = bmk0.index
  results['offspring_title'] = bmk0.title

  # assign when sector prefix exists in parent
  logger.debug("assign when sector prefix exists in parent")
  agg.sort_values('len', ascending=False, inplace=True)
  found = []
  for index, row in bmk.iterrows():
    for parent_index, parent_row in agg.iterrows():
      if index[:parent_row.len] == parent_row.trunc:
        found.append({'offspring_code': index,
                      'parent_code': parent_index,
                      'offspring_title': row.title,
                      'parent_title': parent_row.title,
                      'match': True})
        break
  tmp = pd.DataFrame(found)
  tmp.index = list(tmp.offspring_code)
  results.loc[tmp.index] = tmp
  bmk.drop(tmp.index, inplace=True)
  agg.sort_values('pos', inplace=True)

  # assign when sector preceding prefix exists in parent
  logger.debug(
    "assign when sector preceding prefix exists in parent")
  found = []
  for index, row in bmk.iterrows():
    for pos in range(1, len(agg)):
      curr_len = min(agg.len.iloc[pos], row.len)
      prev_len = min(agg.len.iloc[pos-1], row.len)
      prev_test = (int(index[:prev_len]) > int(
        agg.trunc.iloc[pos - 1][:prev_len]))
      curr_test = (int(index[:curr_len]) < int(
        agg.trunc.iloc[pos][:curr_len]))
      # if index == '312110':
      #  logger.debug(
      #    f"{index}: {agg.index[pos - 1]
      #                } is {prev_test}; {agg.index[pos]}"
      #    f" is {curr_test}")
      if curr_test and prev_test:
        # logger.debug("Success!")
        found.append({'offspring_code': index,
                      'parent_code': agg.index[pos-1],
                      'offspring_title': row.title,
                      'parent_title': agg.title.iloc[pos-1],
                      'match': False})
        break
  tmp = pd.DataFrame(found)
  tmp.index = list(tmp.offspring_code)
  results.loc[tmp.index] = tmp
  bmk.drop(tmp.index, inplace=True)

  # assign remainder to first agg which matches truncated
  # benchmark
  logger.debug(
    "assign remainder to first agg which matches truncated "
    "benchmark")
  found = []
  for index, row in bmk.iterrows():
    for agg_index, agg_row in agg.iterrows():
      if index[:row.len] == agg_index[:row.len]:
        found.append({'offspring_code': index,
                      'parent_code': agg_index,
                      'offspring_title': row.title,
                      'parent_title': agg_row.title,
                      'match': False})
        break
  tmp = pd.DataFrame(found)
  tmp.index = list(tmp.offspring_code)
  results.loc[tmp.index] = tmp
  bmk.drop(tmp.index, inplace=True)

  # manual edits
  logger.debug("manual edits")
  for key, val in new_pairs.__dict__.items():
    key_exists = (key in results.index)
    val_exists = (val in agg0.index)
    logger.debug(
      f"{key_exists}, {val_exists}, {val_exists + val_exists}")
    if key_exists + val_exists == 2:
      logger.debug(f"editing: {key} -> {val}")
      results.loc[key, 'parent_code'] = val
      results.loc[key, 'parent_title'] = agg0.title.loc[val]
      results.loc[key, 'match'] = False
    elif key_exists + val_exists == 0:
      logger.debug(f"not editing: {key} -> {val}")
    else:
      logger.warning(
        f"Either {key} ({key_exists}) not in offspring "
        f"or {val} ({val_exists}) not in parent")

  return results


def load_raw_transport(config, dirs):
  logger.info("Loading source transport information")
  dir_ = dirs.__dict__[config.source.raw_transport.folder]
  raw_transport = pd.read_csv(dir_ / config.source.raw_transport.file)

  # convert to matrix format

  def get_transport_codes(transport, fields):
    sector_set = list(set(transport[fields.sector]))
    margin_set = list(set(transport[fields.margin]))
    sector_set.sort()
    margin_set.sort()
    sector_df = pd.DataFrame(columns=['code', 'title'])
    margin_df = pd.DataFrame(columns=['code', 'title'])
    for pos, str_ in enumerate(sector_set):
      tmp = str_.split('-')
      sector_df.at[pos, 'code'] = tmp[0].strip()
      sector_df.at[pos, 'title'] = tmp[1].strip()
    for pos, str_ in enumerate(margin_set):
      tmp = str_.split('-')
      margin_df.at[pos, 'code'] = tmp[0].strip()
      margin_df.at[pos, 'title'] = tmp[1].strip()
    sector_df.set_index('code', inplace=True)
    margin_df.set_index('code', inplace=True)
    transport_codes = {'margin': margin_df, 'sector': sector_df}

    return utilities.DictToObject(transport_codes)

  proxy_transport = get_transport_codes(
    raw_transport,
    config.source.raw_transport.fields)

  logger.info("Reshaping transport data")
  proxy_transport.data = pd.DataFrame(
    columns=proxy_transport.margin.index,
    index=proxy_transport.sector.index,
    data=0.0)

  config_fields = config.source.raw_transport.fields
  for index, row in raw_transport.iterrows():
    rowpos = row[config_fields.sector].split('-')[0].strip()
    colpos = row[config_fields.margin].split('-')[0].strip()
    value = row[config_fields.value]
    proxy_transport.data.at[rowpos, colpos] = value

  proxy_transport.data['title'] = proxy_transport.sector['title']
  proxy_transport.sector = proxy_transport.data
  del proxy_transport.data

  proxy_transport.margin['proxy'] = 'unknown'
  proxy_transport.margin = proxy_transport.margin[['proxy', 'title']]
  for key, val in config.transport_margins.__dict__.items():
    proxy_transport.margin.loc[key, 'proxy'] = val

  return proxy_transport


def extract_margins(data, codes, config):
  logger.info("Extract from SUT margin info for bridges")

  bridge = {}
  for margin in ['trade', 'transport']:
    bridge[margin] = {}

    tmp = data.sup_int[config.sut_codes.__dict__[margin]]
    bridge[margin]['sector'] = pd.DataFrame()
    bridge[margin]['sector']['total'] = tmp.loc[tmp != 0]
    bridge[margin]['sector']['title'] = codes.commodities.loc[
      bridge[margin]['sector'].index, 'title']
    bridge[margin]['sector'].sort_index(inplace=True)

    tmp = data.use_int[config.sut_codes.__dict__[margin]]
    bridge[margin]['margin'] = pd.DataFrame()
    bridge[margin]['margin']['total'] = tmp.loc[tmp != 0]
    bridge[margin]['margin']['title'] = codes.commodities.loc[
      bridge[margin]['margin'].index, 'title']
    bridge[margin]['margin'].sort_index(inplace=True)

  bridge = utilities.DictToObject(bridge)

  bridge.transport.sector['proxy'] = 'unknown'
  bridge.transport.sector = bridge.transport.sector[
    ['proxy', 'total', 'title']]

  bridge.trade.margin['not_allocated'] = True
  for key, val in config.trade_initial_digits.__dict__.items():
    tmp = deepcopy(bridge.trade)
    tmp.margin['two_digits'] = [x[:2] for x in list(tmp.margin.index)]
    tmp.margin = tmp.margin.loc[tmp.margin.two_digits.isin(val)]
    del tmp.margin['two_digits'], tmp.margin['not_allocated']
    bridge.trade.margin.loc[tmp.margin.index, 'not_allocated'] = False

    tmp.sector = pd.DataFrame()
    for column in tmp.margin.index:
      tmp.sector[column] = '0'
    tmp.sector['count'] = '0'
    tmp.sector['total'] = bridge.trade.sector.total
    tmp.sector['title'] = bridge.trade.sector.title

    setattr(bridge, key, tmp)

  logger.info("Trade margins not allocated to wholesale or retail: "
              f"{bridge.trade.margin.not_allocated.sum()}")

  return bridge


def biproportional_adjustment(matrix, rowsum, colsum, n=100, eps=1.0):
  logger.info(
    f"Adjust matrix to row/column sums; {n} steps, {eps} cutoff")

  obj_ = matrix

  def adhoc_multiply(obj_, sum_, axis):
    alt_ = obj_.sum(axis=axis)
    ret = obj_.multiply(sum_ * (alt_ != 0) / (alt_ + (alt_ == 0)),
                        axis=1-axis)
    return ret

  # simplified RAS
  for k in range(n):
    obj_ = adhoc_multiply(obj_, rowsum, 1)
    colsum_ = obj_.sum(axis=0)
    obj_ = adhoc_multiply(obj_, colsum, 0)
    rowsum_ = obj_.sum(axis=1)

    logger.debug(
      f"{k:2d}: "
      f"{abs(rowsum - rowsum_).max():.0f}, "
      f"{abs(colsum - colsum_).max():.0f}")
    if ((abs(rowsum - rowsum_).max() < eps) and
       (abs(colsum - colsum_).max() < eps)):
      break

  logger.info(
    f"Exit at {k:2d}; max abs row/col diff = "
    f"({abs(rowsum - rowsum_).max():.0f}, "
    f"{abs(colsum - colsum_).max():.0f})")

  return obj_


def load_margins():
  logger.info(
    "Loads configuration and returns source balanced margin splits")
  full_config = utilities.get_config('sut')
  config = full_config.apply_margins.balance

  margins = load_margins_config()

  matrix = {}
  for key, val in margins.__dict__.items():
    matrix[key] = pd.DataFrame(
      data=0.0,
      index=val.sector.index,
      columns=val.margin.index)
  matrix = utilities.DictToObject(matrix)
  del matrix.raw_transport

  # fixing type of proxy code
  margins.transport.sector.proxy = (
    margins.transport.sector.proxy.astype(str))
  margins.transport.sector.proxy = [
    x if len(x) > 1 else f"0{x}"
    for x in list(margins.transport.sector.proxy)]

  logger.info("Creating initial estimate")
  # create initial estimate of transport
  for k_margin, row_margin in margins.raw_transport.margin.iterrows():
    if row_margin.proxy in margins.transport.margin.index:
      for k_sector, row_sector in margins.transport.sector.iterrows():
        matrix.transport.loc[k_sector, row_margin.proxy] = (
          matrix.transport.loc[k_sector, row_margin.proxy]
          + margins.raw_transport.sector.loc[
            str(row_sector.proxy), str(k_margin)])

  # create initial estimate of trade
  for table in ['wholesale', 'retail']:
    for column, row in margins.__dict__[table].margin.iterrows():
      # populate with total sector use of all margins
      matrix.__dict__[table][column] = (
        margins.__dict__[table].sector[column]
        * margins.__dict__[table].sector.total)

  # normalize
  for table in matrix.__dict__.keys():
    for column, row in margins.__dict__[table].margin.iterrows():
      # logger.info(f"{table} {column} {row}")
      # divide by total of margin use in all sectors
      matrix.__dict__[table][column] = (
        matrix.__dict__[table][column]
        / matrix.__dict__[table][column].sum()
        * row.total)

  # combine trade
  matrix.trade = pd.concat([matrix.wholesale, matrix.retail],
                           axis=1)
  margins.trade = margins.wholesale
  margins.trade.sector = margins.trade.sector[['total', 'title']]
  margins.trade.margin = pd.concat([margins.wholesale.margin,
                                    margins.retail.margin],
                                   axis=0)
  del (matrix.wholesale, matrix.retail, margins.wholesale,
       margins.retail)
  # backup = {'source': deepcopy(matrix)}

  logger.info(
    "Balancing source margins: "
    f"nmax = {config.nmax}, eps = {config.eps}")
  for key, val in matrix.__dict__.items():
    logger.info(f"{key}")
    matrix.__dict__[key] = biproportional_adjustment(
      matrix.__dict__[key],
      margins.__dict__[key].sector.total,
      margins.__dict__[key].margin.total,
      config.nmax,
      config.eps)
  target_matrix = deepcopy(matrix)
  # backup['target'] = deepcopy(matrix)

  """
  for key0, val0 in backup.items():
    tmp = {}
    for key1, val1 in val0.__dict__.items():
      tmp[key1] = fill_totals(val1,
                              margins.__dict__[key1].sector.total,
                              margins.__dict__[key1].margin.total,)
    backup[key0] = tmp
  """

  # backup = utilities.DictToObject(backup)

  return target_matrix


def fill_totals(matrix, sector_total, margin_total):
  logger.debug("Add row and columns current and expected total")

  matrix = matrix.astype(float)
  sector_total = sector_total.astype(float)
  margin_total = margin_total.astype(float)

  matrix.loc['tot'] = matrix.sum(0)
  matrix.loc['ref'] = margin_total
  matrix.loc['dif'] = matrix.loc['ref'] - matrix.loc['tot']
  matrix['tot'] = matrix.sum(1)
  matrix['ref'] = 0.0
  matrix.loc[sector_total.index, 'ref'] = sector_total
  matrix['dif'] = matrix['ref'] - matrix['tot']

  return matrix


def apply_margins(sut, margins):
  logger.info("Endogenize margins to existing industries")
  full_config = utilities.get_config('sut')
  sut_codes = full_config.process_margins.sut_codes
  bal_config = full_config.apply_margins.balance

  logger.info(
    "Balancing target margins: "
    f"nmax = {bal_config.nmax}, eps = {bal_config.eps}")
  for key, code in sut_codes.__dict__.items():
    logger.info(f"{key}")
    margins.__dict__[key] = biproportional_adjustment(
      margins.__dict__[key],
      sut.data.sup_int.loc[margins.__dict__[key].index, code],
      sut.data.use_int.loc[margins.__dict__[key].columns, code],
      bal_config.nmax,
      bal_config.eps)

  logger.info(
    "Moving margins to existing industries")
  for key, code in sut_codes.__dict__.items():
    sut.data.sup_int.loc[
      margins.__dict__[key].index, margins.__dict__[key].columns] = (
        sut.data.sup_int.loc[
          margins.__dict__[key].index, margins.__dict__[key].columns]
        + margins.__dict__[key])

    tmp = margins.__dict__[key].sum(axis=0)
    for index, value in tmp.items():
      sut.data.use_int.at[index, index] = (
        sut.data.use_int.at[index, index]
        + value)

    sut.data.use_int.drop(code, axis=1, inplace=True)
    sut.data.sup_int.drop(code, axis=1, inplace=True)
    sut.data.use_pri.drop(code, axis=1, inplace=True)
    sut.codes.industries.drop(code, axis=0, inplace=True)

  sut.calculate_sums(endogenous=True)
  sut.sums_info()

  return sut


def estimate_requirements(reqs, bridge):
  reqs.target_det = deepcopy(reqs.source_det)
  for key, val in reqs.target_det.__dict__.items():
    agg_rows = list(bridge.commodities.loc[val.index, 'parent_code'])
    for col in val.columns:
      agg_col = bridge.industries.loc[col, 'parent_code']
      inv_val = reqs.source_agg.__dict__[key].loc[agg_rows, agg_col]
      val.loc[val.index, col] = (
        val.loc[val.index, col]
        * list(reqs.target_agg.__dict__[key].loc[agg_rows, agg_col])
        * list((inv_val != 0) / (inv_val + (inv_val == 0))))
    reqs.target_det.__dict__[key] = val

  return reqs


def load_external_codes_bridge(config, load_type, stage, codes):
  "Extract codes/bridges of value added/final demand from excel file"

  dir_ = utilities.INTERNAL_PATH / stage
  bridge = {}
  for key, val in config.sheets.__dict__.items():
    bridge[key] = {}
    bridge[key]['codes'] = pd.read_excel(
      dir_ / config.file,
      sheet_name=val.codes,
      index_col='code',
      dtype=str)
    bridge[key]['codes'].index = [x.strip() for x in
                                  bridge[key]['codes'].index]

    bridge[key]['table'] = pd.read_excel(
      dir_ / config.file,
      sheet_name=val.__dict__[load_type],
      index_col=None,
      dtype=str)
    for column in ['raw_code', 'clean_code']:
      bridge[key]['table'][column] = [x.strip() for x in
                                      bridge[key]['table'][column]]

    tmp = deepcopy(bridge[key]['table'])
    tmp['value'] = float(1)
    df_tmp = tmp.pivot(
      index='clean_code',
      columns='raw_code', values='value')
    df_tmp.fillna(float(0), inplace=True)
    df_tmp = df_tmp.astype(float)
    df = pd.DataFrame(index=bridge[key]['codes'].index,
                      columns=codes.__dict__[key].index,
                      data=float(0))
    df.loc[df_tmp.index, df_tmp.columns] = df_tmp
    bridge[key]['matrix'] = df

  return utilities.DictToObject(bridge)




def reconcile_va_output(vec_trunc, vec_ind, config, level_name, va_cols=None):
  """RECON2024 (Mike, 2026-09-21): regions whose value added exceeds output.

  Output is first set with the national (parent) output/compensation ratio.
  Where a region's value added VA = compensation + GOS + production taxes +
  noncomparable imports + product taxes exceeds (1 - f_min) x output:
    plan (2)  keep the region's GDP; raise its output to VA / v, v the
              parent's VA/output ratio (i.e. the national intermediate share),
              and take the extra output from the same industry in the other
              regions of the parent, pro rata to their headroom
              x - VA/(1 - f_min); if the headroom is short, raise only to
              VA/(1 - f_min);
    plan (1)  back-up when even that does not fit: keep the output the
              headroom allows and cut the region's GOS until VA =
              (1 - f_min) x; the GOS removed goes to the other regions of the
              parent pro rata to their GOS (it cannot be placed if they too
              are at the ceiling, and is then logged as dropped).
  Parent totals of output are conserved in every case."""
  f_min = float(config.f_min)
  c = 1.0 - f_min
  if va_cols is None:
    va_cols = ['wages', 'surplus', 'nettax', 'noncomparable', 'nettax_extra']
  vt = vec_trunc
  va = vt[va_cols].sum(axis=1).astype(float)
  x = vt['output'].astype(float).copy()
  surplus = vt['surplus'].astype(float).copy()
  par = vt['parent']
  pv = vec_ind.loc[par.unique(), va_cols].sum(axis=1).astype(float)
  px = vec_ind.loc[par.unique(), 'output'].astype(float)
  v_parent = (pv / px.replace(0, np.nan)).fillna(c)
  # ceiling per parent: 1 - f_min, or the parent's own VA/output ratio where
  # that is higher (private households, 814000: VA = output by definition)
  c_parent = v_parent.clip(lower=c, upper=1.0)
  v_parent = v_parent.clip(upper=1.0)
  stats = {'cells': 0, 'plan2_full': 0, 'plan2_min': 0, 'plan1': 0,
           'output_moved': 0.0, 'gos_moved': 0.0, 'gos_dropped': 0.0}
  c_cell = par.map(c_parent).fillna(c).astype(float)
  bad = va > c_cell * x + 0.5
  for p_code in par.loc[bad].unique():
    c = float(c_parent.get(p_code, 1.0 - f_min))
    idx = par.index[par == p_code]
    trig = idx[bad.loc[idx].values]
    rest = idx[~bad.loc[idx].values]
    stats['cells'] += len(trig)
    v = float(v_parent.get(p_code, c))
    head = (x.loc[rest] - (va.loc[rest] / c).clip(lower=0.0)).clip(lower=0.0)   # output never below 0
    H = float(head.sum())
    need_full = (va.loc[trig] / v - x.loc[trig]).clip(lower=0.0)
    need_min = (va.loc[trig] / c - x.loc[trig]).clip(lower=0.0)
    if need_full.sum() <= H:
      give = need_full; stats['plan2_full'] += len(trig)
    elif need_min.sum() <= H:
      give = need_min; stats['plan2_min'] += len(trig)
    else:
      give = need_min * (H / need_min.sum()) if need_min.sum() > 0 else need_min
      stats['plan1'] += len(trig)
    G = float(give.sum())
    if G > 0 and H > 0:
      x.loc[rest] = x.loc[rest] - head * (G / H)
      x.loc[trig] = x.loc[trig] + give
      stats['output_moved'] += G
    # plan (1) back-up: cut GOS where VA still exceeds the ceiling
    over = (va.loc[trig] - c * x.loc[trig]).clip(lower=0.0)
    over = over[over > 0.5]
    if len(over):
      cut = np.minimum(over, surplus.loc[over.index].clip(lower=0.0))
      surplus.loc[over.index] -= cut
      va.loc[over.index] -= cut
      cut_tot = float(cut.sum())
      room = (c * x.loc[rest] - va.loc[rest]).clip(lower=0.0)
      placeable = min(cut_tot, float(room.sum()))
      if placeable > 0:
        w = surplus.loc[rest].clip(lower=0.0)
        w = w.where(room > 0, 0.0)
        w = (w / w.sum()) if w.sum() > 0 else (room / room.sum())
        add = np.minimum(w * placeable, room)
        surplus.loc[rest] += add; va.loc[rest] += add
        placeable = float(add.sum())
      stats['gos_moved'] += cut_tot
      stats['gos_dropped'] += cut_tot - placeable
  if getattr(config, 'diagnostics', None):
    d = pd.DataFrame({'region': vt['region'], 'industry': vt['industry'], 'va': va,
                      'x_before': vt['output'].astype(float), 'x_after': x,
                      'wages': vt['wages'].astype(float), 'surplus_before': vt['surplus'].astype(float),
                      'surplus_after': surplus, 'triggered': bad})
    d.loc[d.triggered].to_csv(f"{config.diagnostics}_{level_name}.csv", index=False)
  # integer output that still covers value added (rounding must not turn a
  # cell at the ceiling negative)
  x_int = x.round(0)
  x_int = np.where(x_int < va, np.ceil(va), x_int)
  vt['output'] = x_int
  vt['surplus'] = surplus.round(0)
  if 'gdp' in vt.columns:
    vt['gdp'] = vt[['wages', 'nettax', 'surplus', 'nettax_extra']].sum(axis=1)
  logger.info(
    f"{level_name}: VA > (1-{f_min:g}) x output in {stats['cells']} cells -> plan 2 at the "
    f"parent's VA/output ratio {stats['plan2_full']}, plan 2 at the ceiling "
    f"{stats['plan2_min']}, plan 1 (GOS) {stats['plan1']}; output moved "
    f"${stats['output_moved'] / 1e6:,.1f}bn within parents; GOS moved "
    f"${stats['gos_moved'] / 1e6:,.2f}bn (dropped ${stats['gos_dropped'] / 1e6:,.2f}bn)")
  return vt


def services_import_shares(com, M_us, sut, x_st, pce_com_state, gov_share, goods_import_share,
                           config, dirs, year, states, w_st=None, D=None):
  """RECON2024: states' shares of international imports of services (and of
  the other commodities the Census goods pipeline does not cover).

  Import-use proportionality: a commodity's national imports serve its uses
  (household, intermediate, government, investment, export) in the same
  proportions as its total supply. Each use component is then put in the
  states with an allocator of its own:
    household     - a BEA SAPCE4 line by state where one describes the
                    purchase (foreign travel for noncomparable imports,
                    air fares, tuition, hospital care, insurance ...),
                    else the state's PCE of the commodity (413-commodity file)
    intermediate  - sum_j U(c,j) * x_j(s)/x_j  (the industries that buy c,
                    where they produce); port/freight services by the state's
                    goods imports instead
    government    - the state's share of the usaspending/Census government
                    proxy; investment and re-exports - by output share
  Returns (shares: commodities x states, method table)."""
  # ---- SAPCE4 lines by state (millions $)
  sap = pd.read_csv(dirs.__dict__[config.sapce.folder] / config.sapce.file,
                    dtype=str, encoding='latin-1')
  sap['GeoFIPS'] = sap.GeoFIPS.str.strip().str.replace('"', '')
  sap = sap.loc[sap.GeoFIPS.isin(states)]
  sap_val = sap.pivot(index='LineCode', columns='GeoFIPS', values=str(year)).apply(
    pd.to_numeric, errors='coerce').reindex(columns=states)

  def sapce_share(line):
    v = sap_val.loc[str(line)].fillna(0.0).clip(lower=0.0)
    return v / v.sum() if v.sum() > 0 else None

  # ---- use structure of each commodity (national, $M)
  U = sut.data.use_int.reindex(index=com).fillna(0.0)          # commodities x industries (+TRADE/TRANS)
  F = sut.data.use_fin.reindex(index=com).fillna(0.0)
  ind = [c for c in U.columns if c in x_st.index]
  # where the buying industries are: QCEW payroll (Mike's suggestion, the
  # location-quotient basis) or output, by state
  base = w_st if (getattr(config, 'intermediate_weight', 'payroll') == 'payroll' and w_st is not None) else x_st
  base_nat = base.loc[ind].sum(axis=1).replace(0, np.nan)
  ind_share = base.loc[ind].div(base_nat, axis=0).fillna(0.0)    # industries x states
  # producer location quotient of the commodity's main producing industry:
  # a state that lacks the producer imports more of the service per unit of
  # demand; propensity = LQ^-alpha, capped
  alpha = float(getattr(config, 'producer_lq_alpha', 0.0))
  state_tot = base.loc[ind].sum(axis=0); nat_tot = float(state_tot.sum())
  lq_adj = {}
  if alpha > 0 and D is not None:
    for c in com:
      if c not in D.index:
        continue
      prod = D.loc[c]; prod = prod[prod > 0]
      prod = prod.loc[[i for i in prod.index if i in ind]]
      if prod.sum() <= 0:
        continue
      pw = prod / prod.sum()
      st_share = ind_share.loc[pw.index].mul(pw.values, axis=0).sum(axis=0)     # producer's state shares
      lq = st_share / (state_tot / nat_tot).replace(0, np.nan)
      lq_adj[c] = (lq.clip(lower=0.2, upper=5.0) ** (-alpha)).fillna(1.0)

  pce_share_com = pce_com_state.reindex(index=com, columns=states).fillna(0.0).clip(lower=0.0)
  pce_share_com = pce_share_com.div(pce_share_com.sum(axis=1).replace(0, np.nan), axis=0)

  out_share_com = (U.loc[:, ind].sum(axis=1) * 0)  # placeholder index
  overrides = {str(k): v for k, v in config.household_allocator.__dict__.items()}
  port = list(config.goods_import_allocated)
  rows = []
  shares = pd.DataFrame(index=com, columns=states, data=np.nan)
  for c in com:
    if M_us.get(c, 0.0) <= 0:
      continue
    inter = U.loc[c, ind].clip(lower=0.0)
    parts = {'household': max(F.loc[c, 'pce'], 0.0),
             'intermediate': float(inter.sum()),
             'government': max(F.loc[c, 'government'], 0.0),
             'investment': max(F.loc[c, 'investment'] + F.loc[c, 'inventory'], 0.0),
             'exports': max(F.loc[c, 'exports'], 0.0)}
    tot = sum(parts.values())
    if tot <= 0:
      parts = {'intermediate': 1.0}; tot = 1.0
    vec = pd.Series(0.0, index=states); how = []
    # household
    w = parts['household'] / tot if 'household' in parts else 0.0
    if w > 0:
      hs = None
      if c in overrides:
        hs = sapce_share(overrides[c]); how.append(f"household {w:.0%}: SAPCE4 line {overrides[c]}")
      if hs is None:
        hs = pce_share_com.loc[c] if c in pce_share_com.index and pce_share_com.loc[c].notna().any() else None
        how.append(f"household {w:.0%}: state PCE of the commodity")
      if hs is None or hs.isna().all():
        hs = sapce_share(1); how[-1] = f"household {w:.0%}: total state PCE"
      vec += w * hs.fillna(0.0)
    # intermediate
    w = parts.get('intermediate', 0.0) / tot
    if w > 0:
      if c in port and goods_import_share is not None:
        vec += w * goods_import_share; how.append(f"intermediate {w:.0%}: state goods imports")
      else:
        buyers = inter / inter.sum() if inter.sum() > 0 else None
        if buyers is None:
          vec += w * sapce_share(1); how.append(f"intermediate {w:.0%}: total state PCE (no buyers)")
        else:
          vec += w * ind_share.loc[buyers.index].mul(buyers.values, axis=0).sum(axis=0)
          how.append(f"intermediate {w:.0%}: buying industries' output")
    # government
    w = parts.get('government', 0.0) / tot
    if w > 0:
      vec += w * gov_share; how.append(f"government {w:.0%}: government proxy")
    # investment, exports
    w = (parts.get('investment', 0.0) + parts.get('exports', 0.0)) / tot
    if w > 0:
      xs = ind_share.loc[buyers.index].mul(buyers.values, axis=0).sum(axis=0) if inter.sum() > 0 else sapce_share(1)
      vec += w * xs; how.append(f"investment/exports {w:.0%}: as intermediate")
    vec = vec.clip(lower=0.0)
    if c in lq_adj:
      vec = vec * lq_adj[c].reindex(states).fillna(1.0)
      how.append(f"x producer LQ^-{alpha:g}")
    if vec.sum() > 0:
      shares.loc[c] = (vec / vec.sum()).values
    rows.append({'commodity': c, 'imports_musd': M_us[c] / 1000.0, 'method': '; '.join(how)})
  method = pd.DataFrame(rows)
  return shares.dropna(how='all'), method


def apply_state_trade(vec_ind, sut, iot, bea, config, dirs, pce_com_state=None, year=None, out_dir=None):
  """RECON2024: state-specific international imports and exports.

  Goods commodities: the state's share of the national SUT import (export)
  vector is its share of retained imports (netted exports) in the Census
  state pipeline (raw/Trade/out_2024, by BEA commodity, SAMNJ22 rules).
  Services and other commodities without a state source: exports keep the
  output-share split already applied; imports use a demand proxy - the
  state's share of national absorption of the industry's output,
  sum_j A[i,j] x_j(s) + pce_s(i). Counties keep their within-state pattern
  (rescaled to the new state figure). Vectors are in thousand dollars."""
  regions = bea.regions
  states = regions.loc[regions.level == 1].index.tolist()
  st_codes = pd.read_excel(utilities.INTERNAL_PATH / 'sut' / config.state_codes.file,
                           sheet_name=config.state_codes.sheet, dtype=str)
  fips_of = dict(zip(st_codes.two_letter.str.strip(), st_codes.code.str.strip().str.zfill(5)))

  def load_shares(path):
    df = pd.read_csv(path, index_col=0)
    df = df.loc[[c for c in df.index if c in sut.codes.commodities.index]]
    df.columns = [fips_of.get(c, c) for c in df.columns]
    df = df[[c for c in df.columns if c in states]].astype(float).clip(lower=0.0)
    tot = df.sum(axis=1)
    shares = df.div(tot.replace(0, np.nan), axis=0)
    return shares.dropna(how='all')

  T = dirs.__dict__[config.folder]
  sh_m = load_shares(T / config.imports)
  sh_e = load_shares(T / config.exports)
  logger.info(f"state trade shares: imports {sh_m.shape}, exports {sh_e.shape} "
              f"(commodities with a state source)")

  D = iot.data.D                        # commodities x industries (market shares)
  ind = list(iot.codes.industries.index)
  com = list(D.index)
  M_us = (1000 * sut.data.sup_tra['imports'].astype(float)).reindex(com).fillna(0.0)
  E_us = (1000 * sut.data.use_fin['exports'].astype(float)).reindex(com).fillna(0.0)
  M_us = M_us.abs()

  # commodity-level state vectors, then to the industry axis through D
  def state_matrix(nat, shares, fallback_ind_shares):
    out = pd.DataFrame(index=com, columns=states, data=np.nan)
    hit = [c for c in com if c in shares.index]
    out.loc[hit] = shares.loc[hit, states].values * nat.loc[hit].values[:, None]
    m_ind = D.T.dot(out.fillna(0.0))                       # industries x states, sourced part
    covered_com = pd.Series(0.0, index=com); covered_com.loc[hit] = nat.loc[hit]
    resid_ind = D.T.dot(nat - covered_com)                 # national, industry axis, unsourced part
    m_ind = m_ind.add(fallback_ind_shares.multiply(resid_ind, axis=0), fill_value=0.0)
    return m_ind, len(hit), float(covered_com.sum()), float(nat.sum())

  # fallback shares on the industry axis
  x = vec_ind.pivot(index='industry', columns='region', values='output').astype(float)
  pce = vec_ind.pivot(index='industry', columns='region', values='pce').astype(float)
  x_st = x[states].reindex(ind).fillna(0.0); pce_st = pce[states].reindex(ind).fillna(0.0)
  out_share = x_st.div(x_st.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / len(states))
  A = iot.data.A.reindex(index=ind, columns=ind).fillna(0.0)
  absorb = A.dot(x_st) + pce_st                            # sum_j A[i,j] x_j(s) + pce_s(i)
  dem_share = absorb.div(absorb.sum(axis=1).replace(0, np.nan), axis=0).fillna(out_share)

  # ---- services and other commodities without a Census state source
  if hasattr(config, 'services') and pce_com_state is not None:
    gov = pd.read_csv(dirs.__dict__[config.services.government.folder] / config.services.government.file,
                      dtype={'code': str})
    gov['code'] = gov.code.str.zfill(5); gov = gov.set_index('code')
    gov_tot = gov.select_dtypes('number').sum(axis=1).reindex(states).fillna(0.0)
    gov_share = gov_tot / gov_tot.sum() if gov_tot.sum() > 0 else pd.Series(1.0 / len(states), index=states)
    goods_hit = [c for c in com if c in sh_m.index]
    g_imp = (sh_m.loc[goods_hit, states].mul(M_us.loc[goods_hit], axis=0)).sum(axis=0)
    goods_import_share = g_imp / g_imp.sum() if g_imp.sum() > 0 else None
    # commodities the Census pipeline only reaches through hold-outs (chapter
    # 98/99 -> noncomparable imports) are allocated as services instead
    force = [c for c in getattr(config.services, 'force_services', []) if c in sh_m.index]
    if force:
      sh_m = sh_m.drop(index=force)
    rest = [c for c in com if c not in sh_m.index]
    w = vec_ind.pivot(index='industry', columns='region', values='qcew_wages').astype(float)
    w_st = w[states].reindex(ind).fillna(0.0)
    sv_shares, sv_method = services_import_shares(
      rest, M_us, sut, x_st, pce_com_state, gov_share, goods_import_share,
      config.services, dirs, year, states, w_st=w_st, D=D)
    sh_m = pd.concat([sh_m, sv_shares.loc[[c for c in sv_shares.index if c not in sh_m.index]]])
    if out_dir is not None:
      sv_method.to_csv(out_dir / f"services_imports_method_{year}.csv", index=False)
      mat = sh_m.reindex(index=com).fillna(0.0).mul(M_us.reindex(com).fillna(0.0) / 1000.0, axis=0)
      mat.index.name = 'commodity'
      mat.to_csv(out_dir / f"imports_state_commodity_{year}.csv")
      mat.loc[rest].to_csv(out_dir / f"services_imports_state_commodity_{year}.csv")
      logger.info(f"services imports: {len(sv_shares)} commodities, ${M_us.loc[rest].sum() / 1e6:,.0f}bn, "
                  f"allocated by use component (method table and state x commodity matrices written)")

  # RECON2024 (2026-09-27): gateway service margin on re-exports (ESR vectors, benchmark
  # mu) is distribution-services output exported from the state where the goods pass
  # through. That part of national wholesale and transport exports is placed by the
  # states' gateway margins; the rest of those exports keeps the output-share split.
  if hasattr(config, 'gateway_margin'):
    gmc = config.gateway_margin
    gm = pd.read_csv(T / gmc.file, index_col=0).iloc[:, 0].astype(float) / 1000.0   # $ -> $ thousand
    gm.index = [fips_of.get(c, c) for c in gm.index]
    gm = gm.reindex(states).fillna(0.0)
    gshare = gm / gm.sum()
    whl = [c for c in com if any(str(c).startswith(p) for p in gmc.wholesale_prefix)
           and E_us.get(c, 0.0) > 0 and c not in sh_e.index]
    trn = [c for c in gmc.transport if c in com and E_us.get(c, 0.0) > 0 and c not in sh_e.index]
    placed = 0.0
    for group, part in ((whl, gmc.wholesale_share), (trn, 1.0 - gmc.wholesale_share)):
      if not group:
        continue
      Eg = E_us.loc[group]; vtot = min(gm.sum() * part, 0.95 * Eg.sum())
      for c in group:
        vc = vtot * Eg[c] / Eg.sum()
        os_ = out_share.loc[c, states] if c in out_share.index else pd.Series(1.0 / len(states), index=states)
        es = vc * gshare + (Eg[c] - vc) * os_
        sh_e.loc[c, states] = (es / Eg[c]).values
        placed += vc
    logger.info(f"gateway margin: ${gm.sum() / 1e6:,.1f}bn by state; ${placed / 1e6:,.1f}bn of national "
                f"wholesale ({len(whl)}) and transport ({len(trn)}) exports placed by it")

  m_ind, nm, cm, tm = state_matrix(M_us, sh_m, dem_share)
  e_ind, ne, ce, te = state_matrix(E_us, sh_e, out_share)
  logger.info(f"imports: {nm} commodities with a state source ({cm / tm:.1%} of national "
              f"imports), the rest by the demand proxy; exports: {ne} commodities "
              f"({ce / te:.1%}), the rest by output share")
  if out_dir is not None:
    emat = sh_e.reindex(index=com).fillna(0.0).mul(E_us.reindex(com).fillna(0.0) / 1000.0, axis=0)
    emat.index.name = 'commodity'
    emat.to_csv(out_dir / f"exports_state_commodity_{year}.csv")

  for name, mat in [('imports', m_ind), ('exports', e_ind)]:
    old_state = vec_ind.loc[vec_ind.region.isin(states), ['region', 'industry', name]]
    old = old_state.pivot(index='industry', columns='region', values=name).astype(float)
    new = mat.reindex(index=ind, columns=states).fillna(0.0)
    # national control: states sum to the national industry vector
    nat = vec_ind.loc['00000_' + pd.Index(ind), name].astype(float).values
    tot = new.sum(axis=1).values
    scale = np.where(tot > 0, nat / np.where(tot > 0, tot, 1.0), 1.0)
    new = new.multiply(scale, axis=0)
    if name == 'exports':
      # a state cannot export more than it produces (origin-of-movement
      # exports still carry some pass-through trade): cap exports at
      # `export_cap` x output and move the excess to states with headroom
      cap = config.export_cap * x_st.reindex(index=ind, columns=states).fillna(0.0)
      for _ in range(20):
        excess = (new - cap).clip(lower=0.0)
        if excess.values.sum() <= 1.0:
          break
        new = new - excess
        room = (cap - new).clip(lower=0.0)
        w = room.div(room.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        new = new + w.multiply(excess.sum(axis=1), axis=0)
      capped = ((new - cap).abs() < 1e-6) & (new > 0)
      logger.info(f"exports capped at {config.export_cap:.0%} of output in "
                  f"{int(capped.values.sum())} state x industry cells; "
                  f"US uncappable {float((new.sum(axis=1).values - nat).sum()) / 1e6:,.1f} bn")
    keys = [f"{r}_{i}" for r in states for i in ind]
    vals = new.T.values.reshape(-1)
    vec_ind.loc[keys, name] = vals
    # counties: keep the within-state pattern
    ratio = new.div(old.reindex(index=ind, columns=states).replace(0, np.nan)).fillna(1.0)
    cty = vec_ind.loc[regions.loc[vec_ind.region.values, 'level'].values == 2, ['region', 'industry', name]].copy()
    parent = regions.loc[cty.region.values, 'parent'].values
    r = [ratio.at[i, p] if (i in ratio.index and p in ratio.columns) else 1.0
         for i, p in zip(cty.industry.values, parent)]
    vec_ind.loc[cty.index, name] = cty[name].astype(float).values * np.array(r)
    # states whose old value was zero: split by county wages
    zero_old = (old.reindex(index=ind, columns=states).fillna(0.0) == 0) & (new > 0)
    if zero_old.values.any():
      w = vec_ind.pivot(index='industry', columns='region', values='wages').astype(float)
      for i, st in zip(*np.where(zero_old.values)):
        i_code, st_code = ind[i], states[st]
        kids = regions.loc[regions.parent == st_code].index
        ww = w.loc[i_code, kids].fillna(0.0) if i_code in w.index else pd.Series(0.0, index=kids)
        ww = ww / ww.sum() if ww.sum() > 0 else pd.Series(1.0 / len(kids), index=kids)
        vec_ind.loc[[f"{k}_{i_code}" for k in kids], name] = new.at[i_code, st_code] * ww.values
    logger.info(f"{name}: state vectors set; US {nat.sum() / 1e6:,.0f} $M, states sum "
                f"{new.values.sum() / 1e6:,.0f} $M")
  return vec_ind


def load_vectors():
  """Loads vectors with SUT and BEA data but not RPC or taxes"""
  full_config = utilities.get_config('sut')
  dirs = full_config.dirs
  vectors_config = full_config.create_vectors

  bea = bea_operations.load_bridged()

  dir_ = dirs.__dict__[vectors_config.target.folder]
  vec_ind = pd.read_csv(
    dir_ / vectors_config.target.file, index_col=None,
    dtype={'region': str, 'industry': str})
  vec_ind.index = vec_ind.region + '_' + vec_ind.industry
  del dir_
  vec_ind.insert(2, 'industry_title', bea.industries.loc[
    vec_ind.industry, 'title'].values)
  vec_ind.insert(3, 'region_title', bea.regions.loc[
    vec_ind.region, 'title'].values)

  return vec_ind


def load_bea_statistics():
  """Loads national_level PCE and GDP BEA data"""
  full_config = utilities.get_config('sut')
  dirs = full_config.dirs
  config = full_config.process_bea

  data = {}

  dir_ = dirs.__dict__[config.target.folder]
  with pd.ExcelFile(dir_ / config.target.file) as xls:
    for key, val in config.target.data.__dict__.items():
      data[key] = {}
      for key0 in val:
        data[key][key0] = {}
        for key1 in config.target.levels:
          data[key][key0][key1] = pd.read_excel(
            xls,
            sheet_name=key0 + '_' + key1,
            index_col='code',
            dtype={'code': str})

  data = utilities.DictToObject(data)

  return data.pce.pce, data.gdp


def calc_agg_errors(agg, det, bridge, source, target, show=True):
  """Used in BEA processing"""
  agg[target] = bridge.dot(
    det[[target]])
  agg[target + '_diff'] = (agg[target] - agg[source])
  agg[target + '_perc'] = (utilities.safe_division(
    agg[target + '_diff'], agg[source])
    * 100).round(2)
  agg[target + '_diff'] = agg[target + '_diff'].round(0).astype(int)
  if show:
    logger.info(
      f"Maximum {source} -> {target} absolute errors: "
      f"{abs(agg[target + '_diff']).max():.2f}; "
      f"{abs(agg[target + '_perc']).max():.2f} (%)")
  return agg


def load_government():
  logger.info("Load government proxy data")
  full_config = utilities.get_config('sut')
  config = full_config.process_government
  dirs = full_config.dirs

  dir_ = dirs.__dict__[config.target.folder]
  government = pd.read_csv(
    dir_ / config.target.file,
    index_col=0,
    dtype={'code': str})
  return government


def load_government_config():
  logger.info("Load government config")
  full_config = utilities.get_config('sut')
  config = full_config.process_government

  dir_ = utilities.INTERNAL_PATH / full_config.stage
  government = {}
  for key0 in config.source.sheets:
    government[key0] = pd.read_excel(
      dir_ / config.source.file,
      sheet_name=key0,
      index_col=0,
      dtype={'code': str})
  return utilities.DictToObject(government)


def load_margins_config():
  logger.info("Load margins")
  full_config = utilities.get_config('sut')
  config = full_config.process_margins

  dir_ = utilities.INTERNAL_PATH / full_config.stage
  margins = {}
  for key0, val in config.target.sheets.__dict__.items():
    margins[key0] = {}
    for key1, sheet_name in val.__dict__.items():
      margins[key0][key1] = pd.read_excel(
        dir_ / config.target.internal,
        sheet_name=sheet_name,
        index_col=0,
        dtype={'code': str})
  return utilities.DictToObject(margins)


def load_requirements():
  logger.info("Load requirements")
  full_config = utilities.get_config('sut')
  dirs = full_config.dirs
  config = full_config.create_requirements

  dir_ = dirs.__dict__[config.target.folder]
  reqs = {}
  for sheet in config.target.sheets:
    reqs[sheet] = pd.read_excel(
      dir_ / config.target.file,
      sheet_name=sheet,
      index_col=0,
      dtype={'code': str})
  return utilities.DictToObject(reqs)


def process_bea_pce(bea, pce_stat, bea_config):
  logger.info("Processing personal consumption expenditure")
  pce_matrix = bea.pce.pivot(index='commodity',
                             columns='region',
                             values='pce')
  # RECON2024: align the PCE commodity axis (2017 detail, 12 BEA construction
  # types) with the SUT axis (421, 31 NAICS construction industries)
  off = [c for c in pce_matrix.index if c not in pce_stat.det.index]
  if off:
    logger.info(f"PCE commodities off the SUT axis dropped "
                f"(${pce_matrix.loc[off].values.sum() / 1000:,.0f}M): {off}")
  pce_matrix = pce_matrix.reindex(pce_stat.det.index).fillna(0.0)

  logger.info("Resetting PCE religious, grant and civic services")
  reference_sum = 0
  edit_sum = 0 * pce_matrix.iloc[0]
  for index in bea_config.pce_other_service.control:
    reference_sum = (
      reference_sum + pce_stat.det.at[index, 'initial'])
    edit_sum = edit_sum + pce_matrix.loc[index]

  pce_matrix = pce_matrix.astype(float)
  for index in bea_config.pce_other_service.reset:
    pce_matrix.loc[index] = (pce_stat.det.at[
      index, 'initial'] * edit_sum / reference_sum)

  logger.info("Replacing inconsistent with hardcoded values")
  for key in bea_config.pce_adhoc.__dict__.keys():
    ratio = (pce_stat.det.at[key, 'initial']
             / pce_matrix.at[key, '00000'])
    pce_matrix.loc[index] = ratio * pce_matrix.loc[index]

  logger.info("Applying ratio and calculating statistics")
  ratio = utilities.safe_division(pce_stat.det.final,
                                  pce_matrix['00000'])
  ratio = ratio + 1 * (ratio == 0)
  ratio = pd.DataFrame(
    np.diag(ratio.to_numpy()), index=ratio.index, columns=ratio.index)
  pce_matrix = ratio.dot(pce_matrix)
  pce_matrix['commodity'] = pce_matrix.index
  pce_table = pd.melt(
    pce_matrix, id_vars=['commodity'], value_vars=pce_matrix.columns,
    var_name='region', value_name='pce')
  pce_table.index = (
    pce_table.region + '_' + pce_table.commodity)

  pce_tmp = pd.DataFrame(index=pce_matrix.index)
  pce_tmp['ref'] = pce_stat.det.final
  pce_tmp['calc'] = pce_matrix['00000']
  pce_tmp['dif'] = pce_tmp.ref - pce_tmp.calc
  pce_tmp['rel'] = 100 * \
    utilities.safe_division(pce_tmp.dif, pce_tmp.ref)
  pce_tmp.sort_values('rel', inplace=True)
  logger.info("National relative errors (%) are:\n"
              f"{pce_tmp}")

  pce_sum = bea.pce.pivot(index='region',
                          columns='commodity',
                          values='pce')
  pce_tmp = pd.DataFrame(index=pce_sum.columns)
  pce_tmp['ref'] = pce_sum.loc['00000']

  pce_sum.drop('00000', inplace=True)
  pce_tmp['calc'] = pce_sum.sum()

  pce_tmp['dif'] = pce_tmp.ref - pce_tmp.calc
  pce_tmp['rel'] = 100 * \
    utilities.safe_division(pce_tmp.dif, pce_tmp.ref)
  pce_tmp.sort_values('rel', inplace=True)
  logger.info("Difference between state sums and national (%) are:\n"
              f"{pce_tmp}")

  return pce_table


def process_bea_gdp(bea, gdp_stat, bea_config):
  logger.info("Processing value added, jobs and earnings")
  gdp = {}
  for column in bea.bea.columns:
    if column not in ['region', 'industry']:
      gdp[column] = bea.bea.pivot(index='industry',
                                  columns='region',
                                  values=column).astype(float)
  gdp = utilities.DictToObject(gdp)

  logger.info("Reassigning housing, no change in jobs/earnings")
  for line in bea_config.housing:
    ratio = utilities.safe_division(
      gdp_stat.__dict__[line.va].det.loc[line.source, 'alloc'],
      gdp.__dict__[line.va].at[line.source, '00000'])

    diff = (1 - ratio) * (gdp.__dict__[line.va].loc[line.source])

    gdp.__dict__[line.va].loc[line.source] = (
      gdp.__dict__[line.va].loc[line.source]
      - diff)

    gdp.__dict__[line.va].loc[line.target] = (
      gdp.__dict__[line.va].loc[line.target]
      + diff)

  logger.info("Reassigning other services, no change in jobs/earnings")
  for va in bea_config.gdp_other_service.va:
    diff = gdp.__dict__[va].loc[bea_config.gdp_other_service.remove]
    gdp.__dict__[va].loc[bea_config.gdp_other_service.remove] = 0

    tot = pd.Series(index=gdp.__dict__[va].columns, data=0)
    for index in bea_config.gdp_other_service.distribute:
      tot = tot + gdp.__dict__[va].loc[index]

    for index in bea_config.gdp_other_service.distribute:
      gdp.__dict__[va].loc[index] = gdp.__dict__[va].loc[index] + (
        diff * utilities.safe_division(
          gdp.__dict__[va].loc[index], tot))

  logger.info("Reassigning zero value added entries "
              "(wages proxy of jobs/earnings)")
  # neecessary for source value to stay constant
  gdp_tmp = deepcopy(gdp)
  for va in bea_config.va_zeros:
    for gdp_key in gdp.__dict__.keys():
      ratio_key = gdp_key
      if gdp_key in ['employment', 'earnings']:
        ratio_key = 'wages'

      ratio = utilities.safe_division(
        gdp_stat.__dict__[ratio_key].det.loc[va.source, 'alloc'],
        gdp_tmp.__dict__[ratio_key].loc[va.source, '00000'])

      diff = (1 - ratio) * (gdp.__dict__[gdp_key].loc[va.source])

      gdp.__dict__[gdp_key].loc[va.target] = (
        gdp.__dict__[gdp_key].loc[va.target]
        + diff)

      gdp.__dict__[gdp_key].loc[va.source] = (
        gdp.__dict__[gdp_key].loc[va.source]
        - diff)
  del gdp_tmp

  logger.info("Reassigning trade (wages proxy of jobs/earnings)")
  for gdp_key in gdp.__dict__.keys():
    ratio_key = gdp_key
    if gdp_key in ['employment', 'earnings']:
      ratio_key = 'wages'

    bulk_diff = pd.Series(index=gdp.__dict__[gdp_key].columns,
                          data=0.0)
    for individual in bea_config.trade.individual:
      ratio = (
        gdp_stat.__dict__[
          ratio_key].det.loc[individual.detailed, 'alloc']
        / gdp.__dict__[ratio_key].loc[individual.detailed, '00000'])

      individual_diff = (1 - ratio) * (
        gdp.__dict__[gdp_key].loc[individual.detailed])

      gdp.__dict__[gdp_key].loc[individual.detailed] = (
        gdp.__dict__[gdp_key].loc[individual.detailed]
        - individual_diff)

    bulk_sum = pd.Series(index=gdp.__dict__[gdp_key].columns, data=0)
    for key in bea_config.trade.bulk:
      bulk_sum = bulk_sum + gdp.__dict__[gdp_key].loc[key]

    for key in bea_config.trade.bulk:
      gdp.__dict__[gdp_key].loc[key] = (
        gdp.__dict__[gdp_key].loc[key]
        * (1 + utilities.safe_division(bulk_diff, bulk_sum)))

  logger.info(
    "Reassigning government (wages is proxy of jobs/earnings)")
  for gov_list in [bea_config.government_aggregate,
                   bea_config.government_detailed]:
    for gov_conf in gov_list:
      ratio_key = gov_conf.va
      if ratio_key == 'wages':
        va_list = ['employment', 'earnings', 'wages']  # wages at end
      else:
        va_list = [gov_conf.va]

      for gdp_key in va_list:
        ratio = (
          gdp_stat.__dict__[
            ratio_key].det.loc[gov_conf.source, 'alloc']
          / gdp.__dict__[ratio_key].loc[gov_conf.source, '00000'])
        # diff between alloc and result_final is small,
        # and 491000 is further split in next substep

        diff = (1 - ratio) * (
          gdp.__dict__[gdp_key].loc[gov_conf.source])

        gdp.__dict__[gdp_key].loc[gov_conf.source] = (
          gdp.__dict__[gdp_key].loc[gov_conf.source]
          - diff)

        gdp.__dict__[gdp_key].loc[gov_conf.target] = (
          gdp.__dict__[gdp_key].loc[gov_conf.target]
          + diff)

  for va in bea_config.government_scaled.va:
    ratio_key = va
    if ratio_key == 'wages':
      va_list = ['employment', 'earnings', 'wages']  # wages at end
    else:
      va_list = [va]

    gov_source = bea_config.government_scaled.source
    gov_target = bea_config.government_scaled.target

    for gdp_key in va_list:

      ratio = (
        gdp_stat.__dict__[
          ratio_key].det.loc[gov_source, 'result_total']
        / gdp.__dict__[ratio_key].loc[gov_source, '00000'])

      diff = (1 - ratio) * (
        gdp.__dict__[gdp_key].loc[gov_source])

      gdp.__dict__[gdp_key].loc[gov_source] = (
        gdp.__dict__[gdp_key].loc[gov_source]
        - diff)

      gdp.__dict__[gdp_key].loc[gov_target] = (
        gdp.__dict__[gdp_key].loc[gov_target]
        + diff)

  logger.info("Applying ratio and calculating statistics")
  gdp_table = {}
  for gdp_key in gdp.__dict__.keys():

    ratio_key = gdp_key
    if gdp_key in ['employment', 'earnings']:
      ratio_key = 'wages'
    logger.info(f"Variable {gdp_key} with proxy {ratio_key}")

    ratio = utilities.safe_division(
      gdp_stat.__dict__[ratio_key].det.result_total,
      gdp.__dict__[ratio_key]['00000'])
    ratio = ratio + 1 * (ratio == 0)
    ratio = pd.DataFrame(
      np.diag(ratio.to_numpy()),
      index=ratio.index,
      columns=ratio.index)

    gdp.__dict__[gdp_key] = ratio.dot(gdp.__dict__[gdp_key])
    gdp.__dict__[gdp_key]['industry'] = gdp.__dict__[gdp_key].index
    gdp_table[gdp_key] = pd.melt(
      gdp.__dict__[gdp_key],
      id_vars=['industry'],
      value_vars=gdp.__dict__[gdp_key].columns,
      var_name='region', value_name='total')
    gdp_table[gdp_key].index = (
      gdp_table[gdp_key].region + '_' + gdp_table[gdp_key].industry)
    gdp.__dict__[gdp_key].drop('industry', axis=1, inplace=True)

    if gdp_key == ratio_key:
      gdp_tmp = pd.DataFrame(index=gdp.__dict__[gdp_key].index)
      gdp_tmp['ref'] = gdp_stat.__dict__[gdp_key].det.result_total
      gdp_tmp['calc'] = gdp.__dict__[gdp_key]['00000']
      gdp_tmp['dif'] = gdp_tmp.ref - gdp_tmp.calc
      gdp_tmp['rel'] = 100 * \
        utilities.safe_division(gdp_tmp.dif, gdp_tmp.ref)
      gdp_tmp.sort_values('rel', inplace=True)
      logger.info(
        f"National relative errors (%) are:\n{gdp_tmp.round(2)}")

    gdp_sum = gdp.__dict__[gdp_key].transpose()
    gdp_tmp = pd.DataFrame(index=gdp_sum.columns)
    gdp_tmp['ref'] = gdp_sum.loc['00000']

    for string, level in {'state': 1, 'county': 2}.items():
      # gdp_sum.drop('00000', axis=0, inplace=True)
      gdp_tmp['calc'] = gdp_sum.loc[(bea.regions.loc[
        gdp_sum.index, 'level'] == level).values].sum()

      gdp_tmp['dif'] = gdp_tmp.ref - gdp_tmp.calc
      gdp_tmp['rel'] = 100 * utilities.safe_division(
        gdp_tmp.dif, gdp_tmp.ref)
      gdp_tmp.sort_values('rel', inplace=True)
      logger.info(
        f"Difference between {string} sums and national (%) "
        f"are:\n{gdp_tmp.round(2)}")

  gdp_results = deepcopy(gdp_table[list(gdp_table.keys())[0]])
  gdp_results.drop('total', axis=1, inplace=True)
  for key, val in gdp_table.items():
    gdp_results[key] = val['total']

  logger.info("Splitting value added and extra net taxes")
  gdp = {}
  for column in ['total', 'vasut', 'extra']:
    gdp[column] = gdp_results.pivot(index='industry',
                                    columns='region',
                                    values='nettax').astype(float)
  gdp = utilities.DictToObject(gdp)

  for gdp_key, gdp_col in {
      'vasut': 'nettax', 'extra': 'nettax_extra'}.items():
    ratio = utilities.safe_division(
      gdp_stat.nettax.det['result_' + gdp_key],
      gdp_stat.nettax.det.result_total)
    ratio = pd.DataFrame(
      np.diag(ratio.to_numpy()),
      index=ratio.index,
      columns=ratio.index)
    gdp.__dict__[gdp_key] = ratio.dot(gdp.__dict__['total'])

    gdp.__dict__[gdp_key]['industry'] = gdp.__dict__[gdp_key].index
    gdp_tmp = pd.melt(
      gdp.__dict__[gdp_key],
      id_vars=['industry'],
      value_vars=gdp.__dict__[gdp_key].columns,
      var_name='region', value_name='nettax')
    gdp_tmp.index = (
      gdp_tmp.region + '_' + gdp_tmp.industry)
    gdp_results[gdp_col] = gdp_tmp['nettax']

  return gdp_results


def apply_travel_exports(vec_ind, bea, config, dirs):
  """RECON2024 item 7: travel and education exports by state (Mike, 2026-09-21).

  BEA's spending in the US by nonresidents (SUT exports of S00900) is part of
  domestic-concept PCE of hotels, food services etc., and S00900 is not an
  industry, so without this step it stays inside state PCE, placed by
  resident-based SAPCE4 shares. The file from
  code/scripts/build_travel_exports_state.py (NTTO overseas spending, NAFSA
  students, Canada/Mexico residual; commodity x state, $k) is moved here from
  PCE to exports:
    exports(s, c) += T(s, c); counties by their share of the state's output of c
    pce(s, c)     -= T(US, c) x the state's current share of PCE of c
  (counties' PCE is split from the state afterwards, by income, as before).
  The US row changes by the national totals."""
  path = dirs.__dict__[config.folder] / config.file
  T = pd.read_csv(path, index_col=0)
  T.index = T.index.astype(str)
  T.columns = [str(c).zfill(5) for c in T.columns]
  regions = bea.regions
  states = [s for s in regions.loc[regions.level == 1].index if s in T.columns]
  counties = regions.loc[regions.level == 2]
  inds = [c for c in T.index if ('00000_' + c) in vec_ind.index]
  dropped = T.loc[[c for c in T.index if c not in inds]].values.sum()
  T = T.loc[inds, states]
  nat = T.sum(axis=1)
  out_st = pd.DataFrame({s: vec_ind.loc[s + '_' + pd.Index(inds), 'output'].values.astype(float)
                         for s in states}, index=inds)
  exp_st = pd.DataFrame({s: vec_ind.loc[s + '_' + pd.Index(inds), 'exports'].values.astype(float)
                         for s in states}, index=inds)
  # goods bought by visitors are produced elsewhere: the goods content is an
  # export of the producing states (state share of US output), not of the
  # state visited (the retail margin stays with the visited state's PCE)
  goods = [c for c in inds if c[0] in '123' and not c.startswith(('22', '23'))]
  if goods:
    osh = out_st.loc[goods].div(out_st.loc[goods].sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    T.loc[goods] = osh.mul(nat[goods], axis=0).values
  # all: exports capped at `cap` x output; the excess goes to the other
  # states in proportion to their headroom
  cap = float(getattr(config, 'export_cap', 0.9))
  head = (cap * out_st.clip(lower=0) - exp_st).clip(lower=0)
  n_cap = 0
  for c in inds:
    t = T.loc[c].values.copy(); h = head.loc[c].values
    for _ in range(20):
      ex = np.clip(t - h, 0, None)
      if ex.sum() < 1:
        break
      n_cap += int((ex > 0).sum())
      t = t - ex
      room = np.clip(h - t, 0, None)
      if room.sum() <= 0:
        break
      t = t + ex.sum() * room / room.sum()
    T.loc[c] = t
  logger.info(f"travel exports: goods content ({len(goods)} goods) placed by state output share; "
              f"{n_cap} cells capped at {cap:.0%} of output and moved to other states")
  logger.info(f"travel exports: ${nat.sum() / 1e6:,.1f}bn over {len(inds)} industries and "
              f"{len(states)} states (${dropped / 1e6:,.1f}bn on codes not in the vectors)")

  vec_ind.loc['00000_' + nat.index, 'exports'] = vec_ind.loc['00000_' + nat.index, 'exports'].values + nat.values
  vec_ind.loc['00000_' + nat.index, 'pce'] = vec_ind.loc['00000_' + nat.index, 'pce'].values - nat.values

  pce_st = pd.DataFrame({s: vec_ind.loc[s + '_' + pd.Index(inds), 'pce'].values.astype(float)
                         for s in states}, index=inds)
  share = pce_st.clip(lower=0).div(pce_st.clip(lower=0).sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
  dP = share.mul(nat, axis=0)
  for s in states:
    idx = s + '_' + pd.Index(inds)
    vec_ind.loc[idx, 'exports'] = vec_ind.loc[idx, 'exports'].values + T[s].values
    vec_ind.loc[idx, 'pce'] = vec_ind.loc[idx, 'pce'].values - dP[s].values

  # counties: exports by county share of the state's output of the industry
  cty = vec_ind.loc[vec_ind.region.isin(counties.index) & vec_ind.industry.isin(inds)]
  par = counties.loc[cty.region.values, 'parent'].values
  st_out = vec_ind.loc[pd.Index(par) + '_' + cty.industry.values, 'output'].values.astype(float)
  sh = np.where(st_out > 0, cty.output.values.astype(float) / np.where(st_out > 0, st_out, 1.0), 0.0)
  add = np.array([T.at[i, p] if p in T.columns else 0.0 for i, p in zip(cty.industry.values, par)])
  vec_ind.loc[cty.index, 'exports'] = cty.exports.values + sh * add

  st_rows = vec_ind.loc[vec_ind.region.isin(states) & vec_ind.industry.isin(inds)]
  over = st_rows.loc[st_rows.exports > st_rows.output]
  n_neg = int((st_rows.pce < 0).sum())
  logger.info(f"travel exports: {len(over)} state cells with exports > output "
              f"({sorted(set(over.industry))[:10]}); {n_neg} state cells with negative PCE")
  return vec_ind
