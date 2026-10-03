"""Transform imported data."""
import numpy as np
#  import scipy.sparse as sp
from .. import utilities
from recon2024.bea import operations as bea_operations
from recon2024.sut import operations as sut_operations
#  import re
import pandas as pd
from logging import getLogger
from copy import deepcopy
import pdb
import yaml
import statsmodels.api as sm
logger = getLogger('root')


def load_results(region, level='county'):
  """Loading vectors and A matrix given FIPS code"""
  full_config = utilities.get_config('rpc')
  task_str = "export_results"
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  dir_ = dirs.__dict__[config.target.folder]

  if region[-3:] == '000':
    level = 'state'

  if level == 'county':
    state_code = region[:2] + '000'
    vectors_folder = state_code
  elif level == 'state':
    state_code = region
    vectors_folder = 'states'
  else:
    logger.error(f'level is not in [county, state]: {level}')
    raise ValueError

  vectors = pd.read_csv(
    dir_ / vectors_folder
    / f'{config.target.prefix}_{region}.csv', index_col=None,
    dtype={'region': str, 'industry': str})
  vectors.index = vectors.industry

  for column in vectors.columns:
    if column in ['industry', 'region', 'industry_title',
                  'region_title', 'title']:
      vectors[column] = vectors[column].astype(str)
    elif column in [
        'intermediate_use',
        'supplydemand', 'employmentoutput', 'earningsoutput',
        'compensationoutput', 'surplusoutput', 'gdpoutput', 'rpc',
        'nettax_output_federal', 'nettax_output_state',
        'nettax_output_local']:
      vectors[column] = vectors[column].astype(float)
    else:
      vectors[column] = vectors[column].astype(int)

  Amatrix = pd.read_csv(
    dir_ / 'Amatrices'
    / f'{config.target.prefix}_Amatrix_{state_code}.csv',
    index_col=None,
    dtype={'industry': str})
  Amatrix.set_index('industry', inplace=True, drop=True)

  return vectors, Amatrix


def load_federal():
  """load federal vectors"""
  full_config = utilities.get_config('rpc')
  # dirs = full_config.dirs
  # task_str = "calculate_taxes"
  # config = full_config.__dict__[task_str]

  dir_ = utilities.INTERNAL_PATH / full_config.stage
  vectors = pd.read_csv(
    dir_ / 'recon2024_00000.csv', index_col=None,
    dtype={'region': str, 'industry': str})

  vectors.index = vectors.region + '_' + vectors.industry

  for column in vectors.columns:
    if column in ['industry', 'region', 'industry_title',
                  'region_title', 'title']:
      vectors[column] = vectors[column].astype(str)
    elif column in [
        'intermediate_use',
        'supplydemand', 'employmentoutput', 'earningsoutput',
        'compensationoutput', 'surplusoutput', 'gdpoutput', 'rpc',
        'nettax_output_federal', 'nettax_output_state',
        'nettax_output_local']:
      vectors[column] = vectors[column].astype(float)
    else:
      vectors[column] = vectors[column].astype(int)

  return vectors


def load_county_data():
  """load data related to counties"""
  full_config = utilities.get_config('rpc')
  # dirs = full_config.dirs
  # task_str = "calculate_taxes"
  # config = full_config.__dict__[task_str]

  dir_ = utilities.INTERNAL_PATH / full_config.stage
  counties = pd.read_csv(
    dir_ / 'county_data.csv', index_col=None,
    dtype={'region': str, 'parent': str, 'level': int})
  counties.set_index('region', inplace=True, drop=True)
  return counties


def calculate_households(vectors):
  # full_config = utilities.get_config('rpc')
  # dirs = full_config.dirs
  # config = full_config.__dict__[task_str]
  logger.info("Calculate households")

  logger.info("Loading data")
  area = load_area()
  earnings = load_earnings()
  total_income_tax = load_total_income_tax()
  household_income = load_household_income()
  revenue = load_state_local_revenue()

  logger.info("Performing calculations")
  new_vectors = vectors.drop_duplicates(subset=['region'])
  new_vectors['industry'] = 'Households'
  new_vectors['industry_title'] = 'Households'
  new_vectors.index = new_vectors.region + '_' + new_vectors.industry
  for col in new_vectors.columns:
    if col not in ['region', 'industry', 'region_title',
                   'industry_title']:
      if col in [
          'intermediate_use', 'supplydemand', 'employmentoutput',
          'earningsoutput', 'compensationoutput', 'surplusoutput',
          'gdpoutput', 'rpc', 'nettax_output_federal',
          'nettax_output_state', 'nettax_output_local']:
        new_vectors[col] = 0.0
      else:
        new_vectors[col] = 0

  logger.info("RPC")
  new_vectors['rpc'] = utilities.safe_division(
    earnings.loc[new_vectors.region.values,
                 'place_of_residence'].values,
    earnings.loc[new_vectors.region.values, 'place_of_work'].values)

  new_vectors = rpc_upper_bound(new_vectors)

  logger.info("Taxes")

  new_vectors['valid_region'] = deepcopy(new_vectors.region)
  new_vectors['region_level'] = deepcopy(
    area.loc[new_vectors.region.values].level.values)
  index = new_vectors.loc[new_vectors.region_level == 2].index
  new_vectors.loc[index, 'valid_region'] = area.loc[
    new_vectors.loc[index, 'region'].values, 'parent'].values
  new_vectors['households'] = utilities.safe_inverse(
    household_income.loc[
      new_vectors.valid_region.values, 'income'].values)

  new_vectors['nettax_output_federal'] = total_income_tax.loc[
    new_vectors.valid_region.values,
    'value'].values * new_vectors.households

  lines = {'18': 1.0, '9': 0.5, '20': 0.5, '21': 0.5}
  for gov in ['state', 'local']:
    new_vectors[f'nettax_output_{gov}'] = 0
    for key, val in lines.items():
      tmp = revenue.loc[
        new_vectors.valid_region.values + f'_{key}_{gov}', 'value']
      new_vectors[f'nettax_output_{gov}'] = new_vectors[
        f'nettax_output_{gov}'] + val * tmp.values

    new_vectors[f'nettax_output_{gov}'] = new_vectors[
      f'nettax_output_{gov}'] * new_vectors.households
  new_vectors.drop(['households'],
                   axis=1, inplace=True)

  return pd.concat([vectors, new_vectors], axis=0)


def calculate_taxes(vectors, drop=True):
  "Calculate taxes, drop intermediate variables by default"
  full_config = utilities.get_config('rpc')
  config = full_config.calculate_taxes
  #  dirs = full_config.dirs
  logger.info("Calculate taxes")

  logger.info("Loading data")
  area = load_area()
  revenue = load_state_local_revenue()
  business = load_business_income_tax()
  commodity_taxes = load_commodity_taxes()

  logger.info("Performing calculations")

  logger.info("Identifying relevant regions")
  regions = pd.DataFrame(
    index=list(set(vectors.region.values)),
    columns=[
      'level', 'valid_region', 'federal', 'state', 'local', 'total'])
  regions['level'] = deepcopy(area.loc[regions.index].level.values)
  index = regions.loc[regions.level < 2].index
  regions.loc[index, 'valid_region'] = index
  index = regions.loc[regions.level == 2].index
  regions.loc[index, 'valid_region'] = area.loc[index, 'parent']

  logger.info("Calculating production taxes government fractions")
  regions['federal'] = business.loc[
    regions.valid_region.values, 'value'].values.astype(float)
  state_local_lines = {
    '19': 1.0,
    '9': 0.5,
    '20': 0.5,
    '21': 0.5}
  for government in ['state', 'local']:
    regions[government] = 0
    for key, val in state_local_lines.items():
      revenue_tmp = deepcopy(revenue.loc[
        (revenue.line_code == key)
        & (revenue.government == government)])
      revenue_tmp.set_index('region_code', inplace=True, drop=True)
      regions[government] = regions[government] + val * (
        revenue_tmp.loc[regions.valid_region.values, 'value'].values
      )

  governments = ['federal', 'state', 'local']
  regions['total'] = regions[governments].sum(axis=1)
  for government in governments:
    regions[government] = utilities.safe_division(
      regions[government], regions['total'])

  logger.info("Calculating production taxes (subsidies all federal)")
  vectors['nettax_production_output'] = utilities.safe_division(
    vectors.nettax_production, vectors.output)
  index = vectors.loc[vectors.nettax_production_output < 0].index

  for government in governments:
    vectors[f'nettax_production_output_{government}'] = deepcopy(
      vectors.nettax_production_output.values *
      regions.loc[vectors.region, government].values)

    if government == 'federal':
      vectors.loc[
        index, f'nettax_production_output_{government}'] = deepcopy(
          vectors.loc[index, 'nettax_production_output'].values)
    else:
      vectors.loc[
        index, f'nettax_production_output_{government}'] = 0

  logger.info("Calculating commodity taxes")
  vectors['nettax_commodity_output'] = utilities.safe_division(
    vectors.nettax_commodity, vectors.output).astype(float)
  vectors['nettax_commodity_output_federal'] = deepcopy(
    vectors.nettax_commodity_output).astype(float)
  vectors['nettax_commodity_output_state'] = 0.0
  vectors['nettax_commodity_output_local'] = 0.0

  logger.info(f"Setting all tax to state: {config.all_state}")
  index = vectors.loc[vectors.industry.isin(
    config.all_state)].index
  vectors.loc[index, 'nettax_commodity_output_state'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output'].values)
  vectors.loc[index, 'nettax_commodity_output_federal'] = 0.0

  logger.info(
    f"Applying threshold (up to {config.threshold.value} "
    "federal, rest split state/local): "
    f"{config.threshold.industries}")
  index = deepcopy(vectors.loc[vectors.industry.isin(
    config.threshold.industries)].index)
  vectors['threshold'] = config.threshold.value
  tmp = deepcopy(vectors.loc[
    index,
    ['nettax_commodity_output', 'threshold']].min(axis=1).values)
  vectors.loc[index, 'nettax_commodity_output_federal'] = tmp

  tmp = deepcopy(
    (vectors.loc[index, 'nettax_commodity_output']
     - vectors.loc[index, 'nettax_commodity_output_federal']).values)
  vectors.loc[index, 'nettax_commodity_output_state'] = 0.5 * tmp
  vectors.loc[index, 'nettax_commodity_output_local'] = 0.5 * tmp
  del index, tmp, vectors['threshold']

  logger.info(
    "Applying state as fraction of total, local as fraction of state "
    "and federal remainder\n"
    f"{commodity_taxes.stat_loc_frac_fed_rest}")
  df = commodity_taxes.stat_loc_frac_fed_rest
  index = deepcopy(vectors.loc[vectors.industry.isin(
    df.index)].index)
  df_index = vectors.loc[index, 'industry'].values

  vectors.loc[index, 'nettax_commodity_output_state'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output'].values
    * df.loc[df_index, 'state_coeff'].values)
  vectors.loc[index, 'nettax_commodity_output_local'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output_state'].values
    * df.loc[df_index, 'local_coeff'].values)
  vectors.loc[index, 'nettax_commodity_output_federal'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output'].values
    - (vectors.loc[index, 'nettax_commodity_output_state'].values
       + vectors.loc[index, 'nettax_commodity_output_local'].values))
  del df, index

  logger.info(
    "Applying state and local fixed coefficients "
    "and federal remainder\n"
    f"{commodity_taxes.stat_loc_fix_fed_rest}")
  df = commodity_taxes.stat_loc_fix_fed_rest
  index = deepcopy(vectors.loc[vectors.industry.isin(
    df.index)].index)
  df_index = vectors.loc[index, 'industry'].values

  for government in ['state', 'local']:
    vectors.loc[index,
                f'nettax_commodity_output_{government}'] = deepcopy(
      df.loc[df_index, f'{government}_coeff'].values)

    vectors.loc[index,
                f'nettax_commodity_output_{government}'] = deepcopy(
      vectors.loc[index, [
        'nettax_commodity_output',
        f'nettax_commodity_output_{government}']].min(axis=1).values)

  vectors.loc[index, 'nettax_commodity_output_federal'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output'].values
    - (vectors.loc[index, 'nettax_commodity_output_state'].values
       + vectors.loc[index, 'nettax_commodity_output_local'].values))
  del df, index

  logger.info(
    "Applying state fixed coefficients, local remainder "
    "and zero federal\n"
    f"{commodity_taxes.stat_fix_loc_rest}")
  df = commodity_taxes.stat_fix_loc_rest
  index = deepcopy(vectors.loc[vectors.industry.isin(
    df.index)].index)
  df_index = vectors.loc[index, 'industry'].values

  vectors.loc[index, 'nettax_commodity_output_federal'] = 0.0

  vectors.loc[index, 'nettax_commodity_output_state'] = deepcopy(
    df.loc[df_index, 'state_coeff'].values
    * vectors.loc[index, 'nettax_commodity_output'].values)

  vectors.loc[index, 'nettax_commodity_output_local'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output'].values
    - vectors.loc[index, 'nettax_commodity_output_state'].values)
  del df, index

  logger.info(
    "Applying state as fraction of total, local as remainder "
    "and zero federal\n"
    f"{commodity_taxes.stat_frac_loc_rest}")
  df = commodity_taxes.stat_frac_loc_rest
  index = deepcopy(vectors.loc[vectors.industry.isin(
    df.index)].index)
  df_index = vectors.loc[index, 'industry'].values
  line_index = df.loc[df_index, 'state_line'].values
  region_index = regions.loc[vectors.loc[index, 'region'].values,
                             'valid_region'].values
  state_index = region_index + '_' + line_index + '_' + 'state'
  total_index = region_index + '_' + line_index + '_' + 'state_local'
  fraction = utilities.safe_division(
    revenue.loc[state_index, 'value'].values,
    revenue.loc[total_index, 'value'].values)

  vectors.loc[index, 'nettax_commodity_output_federal'] = 0.0
  vectors.loc[index, 'nettax_commodity_output_state'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output'].values
    * fraction)
  vectors.loc[index, 'nettax_commodity_output_local'] = deepcopy(
    vectors.loc[index, 'nettax_commodity_output'].values
    - vectors.loc[index, 'nettax_commodity_output_state'].values)
  del df, index, fraction

  logger.info("Combining production and commodity taxes")
  governments = ['federal', 'state', 'local']
  for government in governments:
    vectors[f'nettax_output_{government}'] = (
      vectors[f'nettax_production_output_{government}']
      + vectors[f'nettax_commodity_output_{government}'])

  if drop:
    for gov in ['_federal', '_state', '_local', '']:
      for type_ in ['production', 'commodity']:
        vectors.drop(f"nettax_{type_}_output{gov}",
                     axis=1, inplace=True)

  return vectors


def calculate_rpc(vectors, drop=True):
  "Calculate RPC, drop intermediate variables by default"
  full_config = utilities.get_config('rpc')
  task_str = "calculate_rpc"
  # dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info(f"Calculate prediction of {len(vectors)} vector")

  logger.info("Loading data")
  regression = load_regression()
  names = load_names()
  area = load_area()

  vectors['supply'] = vectors.output - vectors.exports
  vectors['demand'] = vectors.supply + vectors.imports
  vectors['supplydemand'] = utilities.safe_division(
    vectors.supply, vectors.demand)

  for var in [
      'employment', 'earnings', 'compensation', 'surplus', 'gdp']:
    vectors[f'{var}output'] = utilities.safe_division(
      vectors[var], vectors.output)
  vector_columns = list(vectors.columns)

  logger.info("Calculating dummy and repeated exogenous variables")
  vectors = convert_industry_sctg(vectors)

  # includes both fips and sctg dummies and us_s_wv_ratio
  vectors['fips'] = vectors.region
  vectors = set_dummy_repeated_vars(vectors)

  logger.info("Calculating new exogenous variables")

  vectors['emp_estab'] = utilities.safe_division(
    vectors.qcew_employment, vectors.qcew_establishments)
  vectors['emp_estab_ratio'] = vectors.loc[
    config.region_total + '_'
    + vectors.industry.values, 'emp_estab'].values
  vectors['emp_estab_ratio'] = utilities.safe_division(
    vectors.emp_estab, vectors.emp_estab_ratio)

  columns = ['supply', 'demand']
  for column in columns:
    tmp = deepcopy(vectors[column])
    index = tmp.loc[tmp == 0].index
    tmp.loc[index] = 1.0
    logger.info(f"Applying logarithm to {column}, {len(index)} "
                f"are zero, set log(val) = 0")
    vectors['l_' + column] = np.log(tmp.values)

  empshare_reg = vectors.groupby('region', as_index=False)[
    'qcew_employment'].sum()
  empshare_reg.set_index('region', inplace=True, drop=True)
  empshare_reg = empshare_reg['qcew_employment']
  vectors['empshare'] = 0.0
  vectors.loc[vectors.index, 'empshare'] = utilities.safe_division(
    deepcopy(vectors.qcew_employment.values),
    deepcopy(empshare_reg.loc[vectors.region.values].values))
  del empshare_reg

  vectors['empLQ'] = vectors.loc[
    config.region_total + '_'
    + vectors.industry.values, 'empshare'].values
  vectors['empLQ'] = utilities.safe_division(
    vectors.empshare, vectors.empLQ).values

  vectors['lodging_empLQ'] = vectors.loc[
    vectors.region.values + '_' + config.industry_lodging,
    'empLQ'].values

  wageshare_reg = vectors.groupby('region', as_index=False)[
    'qcew_wages'].sum()
  wageshare_reg.set_index('region', inplace=True, drop=True)
  wageshare_reg = wageshare_reg['qcew_wages']
  vectors['wageshare'] = 0.0
  vectors.loc[vectors.index, 'wageshare'] = utilities.safe_division(
    deepcopy(vectors.qcew_wages.values),
    deepcopy(wageshare_reg.loc[vectors.region.values].values))
  del wageshare_reg

  vectors['lodging_wageshare'] = vectors.loc[
    vectors.region.values + '_' + config.industry_lodging,
    'wageshare'].values

  if 'landshare' not in vectors.columns:
    vectors['landshare'] = utilities.safe_division(
      area.loc[vectors.region.values, 'km2'],
      area.loc['00000', 'km2']).values

  logger.info("Calculating prediction")

  vectors['const'] = 1.0
  X = deepcopy(vectors[names.exo])
  vectors['rpc'] = regression.predict(X)

  vectors['rpc_nomod'] = deepcopy(vectors.rpc)

  post = getattr(config, 'post_processing', None)
  def _on(flag, default=True):
    return default if post is None else bool(getattr(post, flag, default))

  logger.info("Post-prediction adjustments")
  from_sd = pd.Series(False, index=vectors.index)

  if _on('structural_zero'):
    # A region that produces none of a commodity cannot self-supply any of it: rpc = 0 is a
    # fact, not a missing value. The estimating equation cannot express this - supply enters
    # as log(supply) with zeros floored at log(1) = 0 - so it returns a positive rpc from the
    # employment structure alone. In the 2022 vectors this affects 19,340 of 28,363 county
    # goods cells (68%), which RECON2022 gave a mean rpc of 0.097.
    index = vectors.loc[vectors.supply <= 0].index
    vectors.loc[index, 'rpc'] = 0.0
    logger.info(f"Set rpc = 0 for {len(index)} cells with no local supply")

  if _on('services_supply_demand'):
    index = vectors.loc[vectors.sctg.isin(config.sctg_exclude)].index
    vectors.loc[index, 'rpc'] = deepcopy(vectors.loc[index, 'supplydemand'].values)
    from_sd.loc[index] = True
    logger.info(f"Replaced {len(index)} entries with supply demand ratio "
                f"sctg code in {config.sctg_exclude}")

  n_bad = int(vectors['rpc'].isna().sum())
  if n_bad:
    if _on('zero_fill', default=False):
      vectors['rpc'] = deepcopy(vectors['rpc'].fillna(0))
      logger.info(f"Replaced {n_bad} non-real entries with 0")
    else:
      # zero asserts the region imports all of it; the supply/demand ratio is the
      # better guess when a regressor is missing
      vectors['rpc'] = vectors['rpc'].fillna(
        vectors['supplydemand'].clip(lower=0.0, upper=1.0))
      logger.warning(f"{n_bad} cells had no prediction (missing regressors); "
                     "filled with the clipped supply/demand ratio")

  if _on('lodging_override'):
    index = vectors.loc[((vectors.industry == config.industry_lodging)
                         & (vectors.lodging_empLQ <= 1.42))].index
    vectors.loc[index, 'rpc'] = 0.5 * (deepcopy(
      vectors.loc[index, 'landshare'].values) / 0.015)
    index = vectors.loc[((vectors.industry == config.industry_lodging)
                         & (vectors.lodging_empLQ > 1.42))].index
    vectors.loc[index, 'rpc'] = deepcopy(
      vectors.loc[index, 'lodging_empLQ'].values) / 1.42
    logger.info(f"Lodging override applied to industry {config.industry_lodging}")
  else:
    logger.info("Lodging override off; 721000 keeps the supply/demand ratio")

  squeeze = 'services_only' if post is None else getattr(
    post, 'upper_bound_squeeze', 'services_only')
  if squeeze in (True, 'all'):
    vectors = rpc_upper_bound(vectors)
  elif squeeze == 'services_only':
    # the squeeze patched a model that could exceed 1 (no longer possible with a
    # fractional QMLE) and stops supply/demand ratios pinning rpc at exactly 1
    vectors = rpc_upper_bound(vectors, subset=from_sd)
  else:
    logger.info("Upper-bound squeeze off entirely; "
                f"max rpc = {float(vectors.rpc.max()):.4f}")
  vectors['rpc'] = vectors['rpc'].clip(lower=0.0, upper=1.0)

  if drop:
    logger.info("Dropping intermediate variables")
    vectors = vectors[vector_columns + ['rpc']]
  return vectors


def rpc_upper_bound(vectors, subset=None):

  mask = vectors.rpc > 0.95
  if subset is not None:
    mask = mask & subset
  index = vectors.loc[mask].index
  vectors.loc[index, 'rpc'] = 1 - np.exp(
    -3.154 * deepcopy(vectors.loc[index, 'rpc'].values))
  logger.info(
    f"Replacing {len(index)} entries with rpc > 0.95 "
    "with 1 - exp(-3.154 * rpc)")

  return vectors


def load_vectors(size='sample'):
  """Loading vectors with QCEW data, size in [full, sample]"""
  full_config = utilities.get_config('rpc')
  task_str = "process_vectors"
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  dir_ = dirs.__dict__[config.target.folder]

  sizes = ['full', 'sample']
  if size not in sizes:
    logger.error(f"Size {size} is not in {sizes}")
    raise ValueError

  logger.info(f"Loading RPC version of vectors of size {size}")
  if size == 'sample':
    vectors = pd.read_csv(
      dir_ / config.target.file, index_col=None,
      dtype={'region': str, 'industry': str})
    vectors.index = vectors.region + '_' + vectors.industry
  else:
    vectors = sut_operations.load_vectors()

  return vectors


def convert_area(area_raw, column):
  """df is assumed to have RPC area index"""
  full_config = utilities.get_config('rpc')
  task_str = "process_area"
  config = full_config.__dict__[task_str]

  logger.debug("Loading data codes and bridge")
  bea = bea_operations.BeaToSut(
    load_data_bool=False,
    load_proxy=False,
    create_additional_bridges=False)

  logger.debug("Converting area codes")

  area_raw = deepcopy(area_raw[[column]])
  area_raw['code'] = deepcopy(area_raw.index)
  area_clean = pd.DataFrame(columns=area_raw.columns)
  area_clean.loc['kludge_to_avoid_annoying_warning'] = {
    column: 0, 'code': 'kludge_to_avoid_annoying_warning'}
  for key, val in config.remove_append.__dict__.items():
    area_tmp = area_raw.loc[area_raw.code.str.startswith(key)]
    area_tmp.loc[area_tmp.index, 'code'] = deepcopy(
      area_tmp.code.str.replace(key, ''))
    area_tmp.loc[area_tmp.index, 'code'] = deepcopy(
      area_tmp.code) + val
    area_clean = pd.concat([area_clean, area_tmp], axis=0)
  area_clean.set_index('code', drop=True, inplace=True)
  area_clean.drop('kludge_to_avoid_annoying_warning', axis=0,
                  inplace=True)

  logger.debug("length of area before and after code fixing, "
               "should be the same: "
               f"{len(area_clean)} == {len(area_raw)}?")

  area_covered = pd.DataFrame(index=bea.codes.region.index,
                              columns=[column],
                              data=0)

  bea_match = area_clean.loc[area_clean.index.isin(
    bea.codes.region.index)]
  area_covered.loc[bea_match.index, column] = area_clean.loc[
    bea_match.index, column]
  del bea_match

  bea_no_match = area_clean.loc[~ area_clean.index.isin(
    bea.codes.region.index)]
  qcew_match = bea_no_match.loc[
    bea_no_match.index.isin(bea.codes.qcew.index)]

  qcew_vector = pd.DataFrame(index=bea.codes.qcew.index,
                             columns=[column],
                             data=0)
  qcew_vector.loc[qcew_match.index, column] = area_clean.loc[
    qcew_match.index, column]
  bea_vector = bea.bridge.region_qcew.matrix.dot(qcew_vector)
  area_covered[column] = area_covered[column] + bea_vector[column]

  for key, val in config.maui_kalawao_match.__dict__.items():
    area_covered.loc[key, column] = area_clean.loc[
      val, column]

  area_covered[bea.codes.region.columns] = bea.codes.region

  return area_covered


def load_business_income_tax():
  """load business income tax"""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_business_income_tax"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  result = pd.read_csv(dir_ / config.target.file,
                       dtype={'code': str})
  result.set_index('code', inplace=True, drop=True)
  return result


def load_household_income():
  """Loads household income from BEA calculations"""
  full_config = utilities.get_config('bea')
  dirs = full_config.dirs
  pce_config = full_config.process_pce_income

  dir_ = dirs.__dict__[pce_config.target.folder]
  with pd.ExcelFile(dir_ / pce_config.target.file) as xls:
    result = pd.read_excel(
      xls,
      sheet_name='regions',
      index_col=None,
      dtype={'code': str, 'commodity': str, 'region': str,
             'parent': str, 'level': int})
    result.set_index('code', inplace=True)

  return result


def load_state_local_revenue():
  """load state and local government revenue"""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_state_local_revenue"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  result = pd.read_csv(dir_ / config.target.file,
                       dtype={'line_code': str, 'region_code': str})
  result.index = (result.region_code + '_'
                  + result.line_code + '_'
                  + result.government)
  return result


def load_commodity_taxes():
  """load commodity taxes"""
  full_config = utilities.get_config('rpc')
  # dirs = full_config.dirs
  task_str = "calculate_taxes"
  config = full_config.__dict__[task_str]

  dir_ = utilities.INTERNAL_PATH / full_config.stage
  commodity_taxes = {}
  for key in config.source.sheets:
    commodity_taxes[key] = pd.read_excel(
      dir_ / config.source.file,
      sheet_name=key,
      index_col='sector_code',
      dtype={'sector_code': str, 'state_line': str})
  commodity_taxes = utilities.DictToObject(commodity_taxes)

  return commodity_taxes


def load_total_income_tax():
  """load total income tax"""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_total_income_tax"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  result = pd.read_csv(dir_ / config.target.file,
                       dtype={'region_code': str})
  result.set_index('region_code', inplace=True, drop=True)

  return result


def load_earnings():
  """load earnings by place of work and residence"""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_earnings"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  result = pd.read_csv(dir_ / config.target.file,
                       dtype={'region': str})
  result.set_index('region', inplace=True, drop=True)

  return result


def load_area():
  """load area (km2) of each region"""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_area"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  result = pd.read_csv(dir_ / config.target.file,
                       dtype={'code': str, 'parent': str, 'km2': int})
  result.set_index('code', inplace=True, drop=True)

  return result


def load_names():
  """load names of endo and exo regression vars"""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_rpc_regression"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  with open(dir_ / config.target.names, 'r') as file:
    names = yaml.safe_load(file)

  return utilities.DictToObject(names)


def load_regression():
  """load RPC beta regression"""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_rpc_regression"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  result = sm.load(dir_ / config.target.regression)

  return result


def convert_industry_sctg(data):
  """Convert codes, input df with col industry."""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_rpc_regression"
  config = full_config.__dict__[task_str]

  dir_ = dirs.__dict__[config.target.folder]
  conversion = pd.read_csv(
    dir_ / config.target.industry_sctg_conversion,
    index_col=None,
    dtype={'industry': str, 'sctg': str})
  conversion.set_index('industry', inplace=True, drop=True)

  data['sctg'] = conversion.loc[data.industry.values, 'sctg'].values

  return data


def set_dummy_repeated_vars(data):
  """Set dummy vars and rep vars, input df with cols fips and sctg."""
  full_config = utilities.get_config('rpc')
  dirs = full_config.dirs
  task_str = "process_rpc_regression"
  config = full_config.__dict__[task_str]

  index_vars = config.index_vars

  for index_var in index_vars:
    for key, cols in config.dummy_vars.__dict__[
        index_var].__dict__.items():
      data[key] = 0
      for col in cols:
        data.loc[data[index_var] == col, key] = 1

  sctg = config.repeated_var.index
  us_s_wv_ratio = config.repeated_var.value

  # Adding us_s_wv_ratio property from file
  dir_ = dirs.__dict__[config.target.folder]
  repeated = pd.read_csv(
    dir_ / config.target.repeated,
    index_col=None,
    dtype={sctg: str,
           us_s_wv_ratio: float})
  repeated.set_index(sctg, inplace=True, drop=True)
  data[us_s_wv_ratio] = 0.0

  tmp_data = data.loc[data[
    sctg].isin(repeated.index)]

  tmp_index = tmp_data[sctg].values

  data.loc[tmp_data.index, us_s_wv_ratio] = deepcopy(repeated.loc[
    tmp_index,
    us_s_wv_ratio].values)

  return data
