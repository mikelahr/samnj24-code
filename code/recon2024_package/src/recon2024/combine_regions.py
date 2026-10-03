"""Combine regions."""
import argparse
import pdb
import os
import sys
import yaml
import pandas as pd
from time import time
from pathlib import Path
from copy import deepcopy
from . import utilities
from .rpc import operations as rpc_operations
import logging
logger = logging.getLogger('root')


def combine_regions(vectors, county_data):

  code_cols = ['region', 'industry', 'region_title',
               'industry_title']
  rpc_cols = [
    'intermediate_use', 'supplydemand', 'employmentoutput',
    'earningsoutput', 'compensationoutput', 'surplusoutput',
    'gdpoutput', 'rpc']

  tax_cols = [
    'nettax_output_federal',
    'nettax_output_state', 'nettax_output_local']

  logger.info('Creating combined vectors')
  first_region = list(vectors.__dict__.keys())[0]
  vector = deepcopy(vectors.__dict__[first_region])
  area_km2 = county_data.loc[first_region, 'km2']

  logger.info('Adding up extensive variables')
  for key, val in vectors.__dict__.items():
    if key != first_region:
      vector['region'] = (
        vector['region'].values + '_' + val['region'].values)
      vector['region_title'] = (
        vector['region_title'].values + '; '
        + val['region_title'].values)

      for col in vector.columns:
        if col not in code_cols + rpc_cols + tax_cols:
          vector[col] = vector[col].values + val[col].values

      area_km2 = area_km2 + county_data.loc[key, 'km2']

  logger.info('Adding up taxes excluding households')
  for col in tax_cols:
    vector[col] = 0.0
    for key, val in vectors.__dict__.items():
      vector[col] = vector[col] + val[col].values * val.output.values
    vector[col] = utilities.safe_division(vector[col], vector.output)

  vector.index = vector.region + '_' + vector.industry

  logger.info('Calculating ratio')
  vector['intermediate_use'] = (
    1 - utilities.safe_division(
      vector.gdp + vector.noncomparable,
      vector.output))

  logger.info('Adding stuff required for RPC calculation')

  vector_tax = deepcopy(vector)
  vector_tax = vector_tax.loc[vector_tax.industry != 'Households']
  vector_tax = vector_tax[tax_cols]

  federal = rpc_operations.load_federal()
  vector = pd.concat([vector, federal], axis=0)
  vector = vector.loc[vector.industry != 'Households']

  vector['landshare'] = utilities.safe_division(
    area_km2,
    county_data.loc['00000', 'km2'])
  vector.loc[vector.region == '00000', 'landshare'] = 1.0

  vector.drop('rpc', axis=1, inplace=True)
  vector.drop(tax_cols, axis=1, inplace=True)
  vector = rpc_operations.calculate_rpc(vector)
  vector.drop('landshare', axis=1, inplace=True)

  vector = vector.loc[vector.region != '00000']
  vector = pd.concat([vector, vector_tax], axis=1)

  logger.info('Adding households')

  households = {'region': vector.region.iloc[0],
                'region_title': vector.region_title.iloc[0],
                'industry': 'Households',
                'industry_title': 'Households'}
  for col in vector.columns:
    if col in code_cols:
      continue
    elif col in rpc_cols + tax_cols:
      households[col] = 0.0
    else:
      households[col] = 0

  vector.loc['Households'] = households
  rcp_num = 0.0
  rcp_dem = 0.0
  income = 0.0
  for key, val in vectors.__dict__.items():
    rcp_num = rcp_num + county_data.loc[key, 'place_of_residence']
    rcp_dem = rcp_dem + county_data.loc[key, 'place_of_work']
    income = income + county_data.loc[key, 'income']

  vector.loc['Households', 'rpc'] = utilities.safe_division(
    rcp_num, rcp_dem)

  vector = rpc_operations.rpc_upper_bound(vector)

  for col in tax_cols:
    vector.loc['Households', col] = 0.0
    for key, val in vectors.__dict__.items():
      vector.loc['Households', col] = (
        vector.loc['Households', col]
        + val.loc[key + '_Households', col]
        * county_data.loc[key, 'income'])
    vector.loc['Households', col] = utilities.safe_division(
      vector.loc['Households', col], income)

  for col in rpc_cols + tax_cols:
    vector[col] = vector[col].round(5).astype(float)

  return vector


def combine_regions_interface(inpath, outpath):
  """Combines RECON2024 regions provided input and output paths"""

  county_data = rpc_operations.load_county_data()

  try:
    logpath = Path(outpath)
    logfile = f"recon2024_combined_{utilities.datetime_now()}.log"
    utilities.set_logger(
      log_level=20,
      filename=logfile,
      path=logpath,
      create_path=True)
  except Exception as err:
    print(f"{err}")
    sys.exit()

  logger.info(f"Combining regions from folder '{inpath}'; "
              f"and placing resulting in folder '{outpath}'")

  try:
    files = [f for f in os.listdir(inpath)
             if os.path.isfile(os.path.join(inpath, f))]
  except Exception as err:
    logger.error(f"{err}")
    sys.exit()

  logger.info(f"Files in the input folder:'{files}'")

  vector_files = [f for f in files if (
    (f[-4:] == '.csv') and (f[:10] == 'recon2024_')
    and f[10:-4].isdigit() and (len(f[10:-4]) == 5))]
  logger.info(f"Potential input files are:'{vector_files}'")

  vector_files = {f[10:-4]: f for f in vector_files}
  invalid_files = {}
  valid_files = {}
  for key, val in vector_files.items():
    if key not in county_data.index:
      invalid_files[key] = val
    else:
      valid_files[key] = val

  if len(invalid_files) > 0:
    logger.info("The following files were ignored because their "
                f"FIPS code is not recognized: {invalid_files}")

  vector_files = valid_files
  if len(vector_files) > 0:
    logger.info("The following files will be combined in a new "
                f"region: {vector_files}")

  county_data['valid_region'] = deepcopy(county_data.index.values)
  index = county_data.loc[county_data.level == 2].index
  county_data.loc[index, 'valid_region'] = county_data.loc[
    index, 'parent'].values

  valid_regions = list(set(county_data.loc[
    vector_files.keys(), 'valid_region'].values))

  if '00000' in valid_regions:
    logger.error("One of the regions is the federal level. Exiting.")
    sys.exit()

  logger.info("Import vector files")
  vectors = {}
  for key, val in vector_files.items():
    try:
      vectors[key] = pd.read_csv(
        Path(inpath) / val, index_col=None,
        dtype={'region': str, 'industry': str})
      vectors[key].index = (
        vectors[key].region + '_' + vectors[key].industry)
    except Exception:
      logger.error(f"Could not load file {key}: {val}. Exiting.")
      sys.exit()

    try:
      regions = list(set(vectors[key].region.values))
    except Exception:
      logger.error(f"Could not read regions in file {key}: {val}.")
      sys.exit()

    error = False
    if len(regions) != 1:
      error = True
    else:
      if regions[0] != key:
        error = True
    if error:
      logger.error(f"Regions inside file {key}: {val} do not match: "
                   f"{regions}")
      sys.exit()
  vectors = utilities.DictToObject(vectors)

  vector = combine_regions(vectors, county_data)

  logger.info("Export vectors of combined region")
  filename = "recon2024_combined.csv"
  try:
    vector.to_csv(Path(outpath) / filename,
                  index=False)
  except Exception:
    logger.error(f"Could not write {filename} to {outpath}.")
    sys.exit()

  return None


def main():
  """Command-line interface"""

  # Receive arguments
  parser = argparse.ArgumentParser(
    description=(
      "Combines RECON2024 regions.\n"
      "\nReads all files names 'recon2024_XXXXX.csv' in INPUT "
      "where XXXXX is valid FIPS region code and creates combined "
      "region in OUTPUT called 'recon2024_combined.csv'.\n"
      "\nFile recon2002_combined.log lists original files used "
      "and additional information.\n"
      "\nIf input and output folders are not provided then the "
      "current working directory is used."),
    formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument(
    '-i', '--input', type=str, default=os.getcwd(),
    help='Path to folder with input files.')
  parser.add_argument(
    '-o', '--output', type=str, default=os.getcwd(),
    help='Path to folder with output files.')

  args = parser.parse_args()

  combine_regions_interface(args.input, args.output)

  return None


if __name__ == "__main__":
  main()
