"""Steps of RPC pipeline."""
from .. import utilities
from . import operations
from recon2024.bea import operations as bea_operations
from recon2024.sut import operations as sut_operations
from recon2024 import __version__
import pandas as pd
import numpy as np
from logging import getLogger
from copy import deepcopy
import time
import pdb
import yaml
import os
from statsmodels.othermod.betareg import BetaModel
import statsmodels.api as sm
logger = getLogger('root')


def process_state_local_revenue(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Get state and local government revenue")

  logger.info("Loading data")
  area = operations.load_area()

  dir_ = dirs.__dict__[config.source.folder]
  raw_data = pd.read_excel(
    dir_ / config.source.file,
    sheet_name=config.source.sheet,
    header=None, index_col=None)
  raw_data.columns = list(raw_data.iloc[13].values)
  raw_data.Description = raw_data.Description.str.strip()

  logger.info("Extracting and organizing data")
  clean_data = pd.DataFrame(columns=[
    'line_code', 'region_code', 'government', 'value',
    'line_description', 'region_description'])
  for k in range(52):
    region = raw_data.loc[8, 'C' + str(k*5 + 1)]
    if region in area.title.values:
      region_code = area.loc[area.title == region].index[0]
    else:
      logger.info(f'{region} not valid region, used "00000"')
      region_code = '00000'

    tmp_data = raw_data.loc[raw_data.Line.isin(config.lines)]
    for key, val in config.government.__dict__.items():
      government = 'C' + str(k*5 + val)
      tmp0_data = pd.DataFrame(columns=clean_data.columns)
      tmp0_data[['line_code', 'line_description', 'value']] = tmp_data[[
        'Line', 'Description', government]].values
      tmp0_data['region_description'] = region
      tmp0_data['region_code'] = region_code
      tmp0_data['government'] = key
      clean_data = pd.concat([clean_data, tmp0_data], axis=0)
      clean_data.reset_index(inplace=True, drop=True)

  logger.info("Saving data")

  dir_ = dirs.__dict__[config.target.folder]
  clean_data.to_csv(dir_ / config.target.file, index=False)

  logger.info("Finished step")

  return None


def process_business_income_tax(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Get business income tax")

  logger.info("Loading data")
  area = operations.load_area()

  dir_ = dirs.__dict__[config.source.folder]
  raw_data = pd.read_excel(
    dir_ / config.source.file,
    sheet_name=config.source.sheet,
    usecols=config.columns)

  logger.info("Extracting and organizing data")
  raw_data.columns = config.names
  raw_data = raw_data.iloc[config.index.start:config.index.end]

  raw_data.rename({"region": "description"},
                  axis=1, inplace=True)
  raw_data["code"] = None
  for index, row in raw_data.iterrows():
    # pdb.set_trace()
    region = row['description']
    if region in area.title.values:
      region_code = area.loc[area.title == region].index[0]
    else:
      logger.info(f'{region} not valid region, used "00000"')
      region_code = '00000'
    raw_data.loc[index, "code"] = region_code

  raw_data = raw_data[['code', 'value', 'description']]

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  raw_data.to_csv(dir_ / config.target.file, index=False)

  logger.info("Finished step")

  return None


def process_total_income_tax(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Get total income tax")

  logger.info("Loading data")
  area = operations.load_area()

  dir_ = dirs.__dict__[config.source.folder]
  raw_data = pd.read_excel(
    dir_ / config.source.file,
    sheet_name='Sheet1',
    usecols=config.columns)

  logger.info("Extracting and organizing data")
  raw_data.columns = config.names
  raw_data = raw_data.iloc[config.index.start:config.index.end]

  raw_data.rename({"region": "region_description"},
                  axis=1, inplace=True)
  raw_data["region_code"] = None
  for index, row in raw_data.iterrows():
    # pdb.set_trace()
    region = row['region_description']
    if region in area.title.values:
      region_code = area.loc[area.title == region].index[0]
    else:
      logger.info(f'{region} not valid region, used "00000"')
      region_code = '00000'
    raw_data.loc[index, "region_code"] = region_code

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  raw_data.to_csv(dir_ / config.target.file, index=False)

  logger.info("Finished step")
  return None


def process_earnings(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Get variable 'earnings by place of work and residence'")

  logger.info("Loading data")
  area = operations.load_area()
  dir_ = dirs.__dict__[config.source.folder]
  try:
    raw_data = pd.read_csv(
      dir_ / config.source.file)  # , index_col=0,
    # dtype={'region': str, 'industry': str})
  except UnicodeDecodeError as e:
    logger.error(f"UnicodeDecodeError occurred: {e}")
    raw_data = pd.read_csv(
      dir_ / config.source.file,
      encoding='us-ascii',
      encoding_errors='ignore',
      dtype=str)  # , index_col=0,
    # dtype={'region': str, 'industry': str})

  clean_data = pd.DataFrame(index=raw_data.index)
  for key, val in config.columns.__dict__.items():
    clean_data[key] = raw_data[val]

  for key in config.replace.characters:
    clean_data[config.replace.column] = clean_data[
      config.replace.column].str.replace(key, '')

  for key, val in config.select.__dict__.items():
    tmp = clean_data.loc[
      clean_data.code == val]
    tmp.set_index('region', inplace=True, drop=True)
    area[key] = pd.to_numeric(
      tmp['value'], errors='coerce').fillna(0).astype(int)

  area = area[list(config.select.__dict__.keys())]
  area.index.name = 'region'

  dir_ = dirs.__dict__[config.target.folder]
  area.to_csv(dir_ / config.target.file, index=True)

  logger.info("Finished step")
  return None


def process_vectors(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Saving sample from full vectors to save time")

  logger.info("Loading original data")
  vectors = sut_operations.load_vectors()

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]

  vectors_sample = vectors.loc[vectors.region.isin(
    config.regions)]

  vectors_sample.to_csv(
    dir_ / config.target.file, index=False)

  del vectors_sample

  logger.info("Finished step")
  return None


def apply_rpc_tax_households(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Calculate RPC prediction")

  logger.info("Loading data")

  vectors = operations.load_vectors(size='full')
  vectors = vectors.loc[vectors.region.isin(
    ['00000', '39000', '39001'])]

  # vector_columns = list(vectors.columns)
  # rpc_results = operations.calculate_rpc(vectors)
  # vectors = rpc_results  # [config.new_columns]
  # del rpc_results

  vectors.to_csv('~/Desktop/test.csv', index=False)

  logger.info("Finished step")
  return None


def process_rpc_regression(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Processing RPC regression")

  logger.info("Loading raw regression data")
  dir_ = dirs.__dict__[config.source.folder]
  data = pd.read_excel(
    dir_ / config.source.file,
    sheet_name='Sheet1',
    index_col=None,
    dtype={'fips': str, 'sctg': str})

  logger.info("Importing industry-sctg conversion")
  dir_ = dirs.__dict__[config.source.folder]
  conversion = pd.read_csv(
    dir_ / config.source.industry_sctg_conversion,
    index_col=None,
    dtype={'industry': str, 'sctg': str})
  dir_ = dirs.__dict__[config.target.folder]
  conversion.to_csv(dir_ / config.target.industry_sctg_conversion,
                    index=False)
  del conversion, dir_

  logger.info("Extracting and saving sctg-specific us_s_wv_ratio")
  repeated_data = data[[
    config.repeated_var.index, config.repeated_var.value
  ]].drop_duplicates()
  repeated_data = repeated_data.dropna()
  dir_ = dirs.__dict__[config.target.folder]
  repeated_data.to_csv(dir_ / config.target.repeated, index=False)

  logger.info("Selecting exogenous variables")
  exo_vars = config.exo_vars
  endo_var = config.endo_var
  index_vars = config.index_vars
  data = data[index_vars + exo_vars + [endo_var]]
  data = data.dropna()
  X = sm.add_constant(data[exo_vars + index_vars])
  y = data[endo_var]

  logger.info("Adding dummy and repeated exogenous variables")
  X['fips'] = X['fips'].str.pad(width=5, side='left', fillchar='0')
  cluster = X['fips'].copy()
  X = operations.set_dummy_repeated_vars(X)
  X.drop(index_vars, axis=1, inplace=True)
  X = sm.add_constant(X)

  # RECON2024 (2026-09-20): fractional-response QMLE (Papke & Wooldridge) replaces
  # BetaModel. Evidence in RECON2024/_rpc_faf6_patches/README_rpc_estimator.md:
  # beta's information matrix is near-singular once 0/1 cells are squeezed, QMLE is
  # consistent for E[rpc|X] and bounded in (0,1) so no upper-bound patch is needed,
  # and it matches the saved Stata fracreg probit that prediction_stata_code.do uses.
  links = {'probit': sm.families.links.Probit(),
           'logit': sm.families.links.Logit(),
           'cloglog': sm.families.links.CLogLog()}
  link_name = getattr(config, 'link', 'probit')
  n_bound = int(((y <= 0) | (y >= 1)).sum())
  logger.info(f"Fitting fractional {link_name} QMLE on {len(y)} cells "
              f"({n_bound} at a 0/1 boundary, kept as observed); "
              "standard errors clustered by fips")
  model = sm.GLM(y, X, family=sm.families.Binomial(link=links[link_name]))
  result = model.fit(cov_type='cluster', cov_kwds={'groups': cluster})
  fitted = result.predict(X)
  logger.info("Fit on the rpc scale: R2 (corr^2 actual vs fitted) = "
              f"{np.corrcoef(fitted, y)[0, 1] ** 2:.4f}, "
              f"MAE = {np.mean(np.abs(fitted - y)):.4f}")

  dir_ = dirs.__dict__[config.target.folder]
  with open(dir_ / config.target.summary, "w") as file:
    file.write(result.summary().as_text())
  names = {'endo': model.endog_names,
           'exo': [x for x in model.exog_names if x not in ['rpc', 'precision']]}
  with open(dir_ / config.target.names, 'w') as file:
    yaml.dump(names, file)
  result.save(dir_ / config.target.regression)

  result0 = sm.load(dir_ / config.target.regression)

  logger.info("Checking results")

  y0 = result.predict(X.iloc[0])
  y1 = result0.predict(X.iloc[0])
  logger.info("Should be loaded == stored ~= original. Is it?\n"
              f"{y1.iloc[0]} = predicted loaded\n"
              f"{y0.iloc[0]} = predicted stored\n"
              f"{y.iloc[0]} = original")

  logger.info("Finished step")
  return None


def process_area(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Processing area")

  logger.info("Loading area data")
  dir_ = dirs.__dict__[config.source.folder]
  area_tmp = pd.read_csv(
    dir_ / config.source.file)
  area_raw = pd.DataFrame(columns=list(
    config.source.columns.__dict__.keys()))
  for key, val in config.source.columns.__dict__.items():
    area_raw[key] = area_tmp[val]
  del area_tmp

  logger.info("Converting numerical format")
  area_raw.set_index('code', inplace=True, drop=True)
  area_raw['m2'] = area_raw.m2.str.replace(',', '').astype(float)
  area_raw['km2'] = (1e-6 * area_raw.m2).round(0).astype(int)
  area_raw['ones'] = 1

  logger.info("Checking concordance between area and BEA region codes")
  area_clean = operations.convert_area(area_raw, 'ones')

  area_covered = 1 * (area_clean.ones > 0)
  n_match = (area_covered == 1).sum()
  is_match = (n_match == len(area_covered))
  logger.info("Are all target area categories covered?: {is_match}")
  del area_clean, area_covered, n_match, is_match

  logger.info("Converting to BEA region classification")
  area_clean = operations.convert_area(area_raw, 'km2')

  area_clean.km2 = area_clean.km2.astype(int)
  dir_ = dirs.__dict__[config.target.folder]
  area_clean.to_csv(dir_ / config.target.file, index=True)

  logger.info("Finished step")
  return None


def export_results_deprecated(task_str, full_config):
  dirs = full_config.dirs
  sym_config = full_config.create_symmetric
  vectors_config = full_config.create_vectors
  config = full_config.export_results
  logger.info("Exporting results")

  logger.info("Loading data")
  bea = bea_operations.load_bridged()

  dir_ = dirs.__dict__[sym_config.target.folder]
  iot = operations.Sut.read_excel(
    dir_ / sym_config.target.files.iot, 'iot', is_iot=True)
  del dir_

  logger.info("Loading vectors")
  dir_ = dirs.__dict__[vectors_config.target.folder]
  vec_ind = pd.read_csv(
    dir_ / vectors_config.target.file, index_col=None,
    dtype={'region': str, 'industry': str})
  vec_ind.index = vec_ind.region + '_' + vec_ind.industry
  del dir_
  vec_ind['title'] = bea.industries.loc[
    vec_ind.industry, 'title'].values

  qcew = bea_operations.load_qcew()
  qcew['wages'] = (qcew.wages / 1000).round(0).astype(int)
  vec_ind['qcew_' + qcew.columns] = qcew
  vec_ind = vec_ind[[
    *vec_ind.columns[:2],
    *vec_ind.columns[-3:],
    *vec_ind.columns[2:-3]]]
  del qcew

  dir_ = dirs.__dict__[config.folder]
  tmp = vec_ind.loc[vec_ind.region == '00000']
  tmp.index = tmp.industry
  interuse_ref = tmp.interuse
  del tmp
  metadata = pd.DataFrame(columns=['comment'])
  metadata.loc['description'] = (
    "2024/2025 RECON/IO revision")
  metadata.loc['reference year'] = str(full_config.year)
  metadata.loc['region_code'] = "PLACEHOLDER"
  metadata.loc['region_name'] = "PLACEHOLDER"
  metadata.loc['sheets'] = (
    "regions, vectors, A")

  metadata.loc['regions'] = (
    "General information about a region and its direct sub-regions")
  metadata.loc['regions.population'] = ("number of inhabitants (-)")
  metadata.loc['regions.income'] = (
    "personal income (thousand dollars)")

  metadata.loc['vectors'] = (
    "split by industry (thousand dollars except otherwise specified)")
  metadata.loc['industries.qcew_*'] = ("based on QCEW")
  metadata.loc['industries.employment'] = ("number of employees (-)")
  metadata.loc['industries.earnings'] = (
    "income from labor, land, and capital plus transfer receipts")
  metadata.loc['vectors.wages'] = (
    "compensation of employees")
  metadata.loc['vectors.surplus'] = (
    "gross operating surplus")
  metadata.loc['vectors.nettax'] = (
    "taxes less subsidies on production")
  metadata.loc['vectors.nettax_extra'] = (
    "taxes less subsidies on products and imports plus tariffs")
  metadata.loc['vectors.gdp'] = (
    "sum of wages, surplus, nettax and nettax_extra")
  metadata.loc['vectors.output'] = (
    "total domestic industry output")
  metadata.loc['vectors.noncomparable'] = (
    "noncomparable imports")
  metadata.loc['vectors.imports'] = (
    "total imports")
  metadata.loc['vectors.pce'] = (
    "personal consumption expenditure")
  metadata.loc['vectors.investment'] = (
    "fixed capital formation")
  metadata.loc['inventory'] = (
    "changes in stocks")
  metadata.loc['vectors.government'] = (
    "government expenditure")
  metadata.loc['vectors.exports'] = (
    "total exports")
  metadata.loc['vectors.interuse'] = (
    "ratio of intermediate use to total output")
  metadata.loc['vectors.supplydemand'] = (
    "ratio of output less exports by output less exports plus imports")
  metadata.loc['vectors.earningsgdp'] = (
    "ratio of earnings to gdp")

  metadata.loc['sources'] = (
    "Source data were BEA, BLS and BTS, see ancillary documentation")
  metadata.loc['authors'] = (
    "Joao  Rodrigues, Michael Lahr and Alexandru Voicu")
  metadata.loc['contact'] = ("lahr@rutgers.edu")
  metadata.loc['date'] = time.strftime('%l:%M%p %z on %b %d, %Y')
  metadata.loc['version'] = __version__

  logger.info("Generating results")
  regions_selection = bea.regions.loc[bea.regions.level < 2]
  for region_key, region_val in regions_selection.iterrows():
    logger.info(f"{region_key}")

    metadata.loc['region_code'] = region_key
    metadata.loc['region_name'] = region_val.title

    regions = pd.DataFrame(columns=bea.regions.columns)
    regions.loc[region_key] = region_val
    regions = pd.concat([
      regions,
      deepcopy(bea.regions.loc[bea.regions.parent == region_key])],
      axis=0)
    regions.drop(['parent', 'level'], inplace=True, axis=1)

    industries = deepcopy(vec_ind.loc[vec_ind.region == region_key])
    industries.set_index('industry', drop=True, inplace=True)
    industries.drop(['region'], inplace=True, axis=1)

    A = iot.data.A.multiply(industries.supplydemand, axis=0)
    A = A.loc[industries.index]
    A = A[industries.index]

    interuse_ratio = utilities.safe_division(
      industries.interuse, interuse_ref)
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

    file = (f"{config.prefix}"
            f"{region_key}"
            f"{config.suffix}")

    vectors = industries

    metadata.index.name = 'code'
    regions.index.name = 'code'
    A.index.name = 'code'
    vectors.index.name = 'code'

    @ utilities.open_excel(dir_ / file, makedir=True)
    def save_state_tables(writer, tables):
      for sheet, table in tables.items():
        table.to_excel(writer, sheet_name=sheet, index=True)

    tables = {'metadata': metadata, 'regions': regions,
              'vectors': vectors, 'A': A}
    save_state_tables('_', tables)

    # saving counties data
    if region_val.level == 1:
      county_index = bea.regions.loc[
        bea.regions.parent == region_key].index

      @ utilities.open_excel(dir_ / file, makedir=True)
      def save_county_tables(writer, values, county_index):
        for county_key in county_index:
          logger.info(f"{county_key}")
          results = deepcopy(values.loc[values.region == county_key])
          results.drop(['region'], inplace=True, axis=1)
          results.set_index('industry', drop=True, inplace=True)
          results.to_excel(writer, sheet_name=county_key, index=True)

      save_county_tables('_', vec_ind, county_index)
    logger.info("")

  logger.info("Finished step")
  return None


def export_results(task_str, full_config):
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  logger.info("Exporting results")

  vectors = operations.load_vectors(size='full')

  vectors = operations.calculate_rpc(vectors)
  vectors = operations.calculate_taxes(vectors)
  vectors = operations.calculate_households(vectors)

  regions = vectors[['region', 'valid_region', 'region_level']]
  vectors.drop(['valid_region', 'region_level'], axis=1, inplace=True)

  columns = [
    'supplydemand', 'employmentoutput', 'earningsoutput',
    'compensationoutput', 'surplusoutput', 'gdpoutput', 'rpc',
    'nettax_output_federal', 'nettax_output_state',
    'nettax_output_local']
  for col in columns:
    vectors[col] = vectors[col].round(5).astype(float)

  logger.info("Saving states data")
  states = list(set(
    regions.loc[regions.region_level < 2, 'region'].values))
  dir_ = dirs.__dict__[config.target.folder] / 'states/'
  if not os.path.exists(dir_):
    os.makedirs(dir_)
    logger.info("Created results 'states' directory")
  for region in states:
    logger.info(region)
    selection = vectors.loc[vectors.region == region]
    selection.to_csv(
      dir_ /
      f'{config.target.prefix}_{region}.csv',
      index=False)

  logger.info("Saving counties data")
  counties = regions.loc[regions.region_level == 2]
  counties.index = counties.region.values
  counties = counties.loc[counties.index.drop_duplicates(keep='first')]
  states = list(set(counties.valid_region.values))

  for state in states:
    logger.info(state)
    dir_ = dirs.__dict__[config.target.folder] / f'{state}/'
    if not os.path.exists(dir_):
      os.makedirs(dir_)
      logger.info(f"Created results '{state}' directory")
    regions = counties.loc[counties.valid_region == state].index
    for region in regions:
      selection = vectors.loc[vectors.region == region]
      selection.to_csv(
        dir_ /
        f'{config.target.prefix}_{region}.csv',
        index=False)

  logger.info("Finished step")

  return None
