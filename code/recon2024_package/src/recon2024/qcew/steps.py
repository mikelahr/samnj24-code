"""Steps of QCEW pipeline."""
from .. import utilities
from . import operations
import pandas as pd
import numpy as np
import itertools
from time import time
from logging import getLogger
from copy import deepcopy
import re
import pdb
logger = getLogger('root')


def dummy(task_str, full_config):
  logger.info("Performing dummy step")
  config = full_config.__dict__[task_str]
  logger.info(f"Reading dummy parameter: {config.parameter}")
  logger.info("Finished dummy step")
  return None


def process_codes(task_str, full_config):
  logger.info("Loading QCEW codes and creating trees")
  config = full_config.__dict__[task_str]
  dirs = full_config.dirs

  codes = {}
  for key, val in config.source.__dict__.items():
    logger.info(f"Loading classification '{key}'")
    codes[key] = pd.read_csv(dirs.source / val.source)
    codes[key].rename(columns={val.code: 'code',
                                         val.title: 'title'},
                      inplace=True)

  codes = utilities.DictToObject(codes)

  # area
  logger.info("Editing 'area' classification")
  strings = ['USCMS', 'USMSA', 'USNMS']
  logger.info(f"Removing records whose code is in {strings}")
  index = list(filter(
    lambda x: codes.area.code[x] in strings,
    codes.area.index))
  codes.area.drop(index, axis=0, inplace=True)

  strings = ['57', '72', '78']
  logger.info(
    f"Removing records whose first two digits are in {strings}")
  index = list(filter(
    lambda x: codes.area.code[x][:2] in strings,
    codes.area.index))
  codes.area.drop(index, axis=0, inplace=True)

  logger.info("Removing records whose code starts with 'C'")
  index = list(filter(
    lambda x: codes.area.code[x][0] == 'C',
    codes.area.index))
  codes.area.drop(index, axis=0, inplace=True)

  logger.info("Creating 'area' aggregation tree")
  codes.area['parent'] = [
    'US000' if x[-3:] == '000' else x[:-3] + '000' for x in
    codes.area.code]
  codes.area['level'] = [
    1 if x[-3:] == '000' else 2 for x in
    codes.area.code]
  codes.area.loc[
    codes.area['code'] == 'US000', 'parent'] = None
  codes.area.loc[
    codes.area['code'] == 'US000', 'level'] = 0

  # ownership
  logger.info("Creating 'ownership' aggregation tree")
  codes.ownership['parent'] = 0
  codes.ownership['parent'] = (
    codes.ownership['parent'].astype(str))
  codes.ownership['code'] = (
    codes.ownership['code'].astype(str))
  logger.info("Dropping intermediate totals")
  codes.ownership.drop([4, 6, 7], axis=0, inplace=True)
  codes.ownership['level'] = 1
  codes.ownership.loc[0, 'parent'] = None
  codes.ownership.loc[0, 'level'] = 0

  # industry
  logger.info("Dropping some 'industry' codes")
  index = [x for x in codes.industry.index if
           codes.industry.code[x][:2] == '10' and
           len(codes.industry.code[x]) > 2]
  codes.industry.drop(index, axis=0, inplace=True)

  logger.info("Creating 'industry' aggregation tree")
  logger.info("Handling hyphenated codes")
  # RECON2024 fix: this block clobbered `codes` (DictToObject) with a
  # DataFrame in the archived RECON2022 source; use a separate name.
  two_digit = pd.DataFrame(index=[str(x) for x in range(10, 100)])
  two_digit['string'] = two_digit.index
  hyphens = [x for x in codes.industry.code if '-' in x]
  for hyphen in hyphens:
    endpoints = hyphen.split('-')
    for x in range(int(endpoints[0]), int(endpoints[1])+1):
      two_digit.loc[str(x), 'string'] = hyphen
  codes.industry['parent'] = [
    x[:-1] if (len(x) > 2 and '-' not in x) else '10'
    for x in codes.industry['code']]
  codes.industry['parent'] = [
    two_digit['string'][x] if len(x) == 2 else x
    for x in codes.industry['parent']]

  codes.industry['level'] = [
    len(x) - 1 if '-' not in x else 1 for x in
    codes.industry['code']]

  codes.industry.loc[
    codes.industry['code'] == '10',
    'parent'] = None
  codes.industry.loc[
    codes.industry['code'] == '10',
    'level'] = 0

  # add NAICS prefix to title if missing
  for idx, row in codes.industry.iterrows():
    if 'NAICS' not in row.title:
      codes.industry.loc[
        idx, 'title'] = 'NAICS ' + codes.industry.loc[
          idx, 'title']

  # split version from title
  codes.industry['version'] = ''
  for idx, row in codes.industry.iterrows():
    tmp = row.title.split(' ')
    codes.industry.loc[
      idx, 'version'] = re.sub('NAICS', '', tmp[0])
    codes.industry.loc[idx, 'title'] = ' '.join(tmp[2:])

  codes.industry['version'] = [
    '22' if str_ == '' else str_ for str_ in
    codes.industry.version]

  # combining ownership and industry
  logger.info("Combining ownership and industry codes")
  merged = pd.DataFrame(
    columns=['code', 'parent', 'level', 'title'])
  tmp_own = codes.ownership.loc[
    codes.ownership['level'] == 0]
  tmp_ind = codes.industry.loc[
    codes.industry['level'] == 0]
  merged_tmp = {
    'code': tmp_own['code'][0] + '_' + tmp_ind['code'][0],
    'parent': None,
    'level': 0,
    'title': tmp_own['title'][0] + ' ' + tmp_ind['title'][0]}
  merged = pd.concat([merged, pd.DataFrame([merged_tmp])], axis=0)

  class_own = codes.ownership.loc[
    codes.ownership.level == 1]
  for level in range(0, 6):
    t0 = time()
    class_ind = codes.industry.loc[
      codes.industry.level == level]
    data = list(itertools.product(
      list(class_own.code), list(class_ind.code)))
    merged_tmp = []
    for index in data:
      row = {}
      own_code = index[0]
      ind_code = index[1]
      tmp_own = codes.ownership.loc[
        codes.ownership.code == own_code]
      tmp_ind = codes.industry.loc[
        codes.industry.code == ind_code]
      row['code'] = own_code + '_' + ind_code
      if level == 0:
        row['parent'] = '0_10'
      else:
        row['parent'] = (own_code + '_' +
                         list(tmp_ind['parent'])[0])
      row['level'] = level + 1
      row['title'] = (list(tmp_own['title'])[0] + ' ' +
                      list(tmp_ind['title'])[0])

      merged_tmp.append(row)
    merged = pd.concat([merged, pd.DataFrame(merged_tmp)], axis=0)
    logger.info(f"level: {level}; time: {time() - t0:.2f}s")

  merged['ownership'] = [oind.split('_')[0] for oind in merged.code]
  merged['industry'] = [
    '_'.join(oind.split('_')[1:]) for oind in merged.code]

  setattr(codes, 'oind', merged)

  # Calculate tree count
  logger.info(
    "Calculate number of bottom-level elements and minor edits")
  for key, val in codes.__dict__.items():
    logger.info(f"Classification {key}")
    val.set_index('code', inplace=True, drop=True)
    val['tree_count'] = int(0)
    nmax = val.level.max()
    val.loc[val.level == nmax, 'tree_count'] = int(1)
    for k in reversed(range(nmax)):
      offspring_index = list(val.loc[val.level == k + 1].index)
      current_index = list(val.loc[offspring_index, 'parent'])
      values = val.loc[offspring_index, 'tree_count']
      values.index = current_index
      values = values.groupby(values.index).sum()
      val.loc[values.index, 'tree_count'] = values
      logger.info(f"Level: {k}, number of categories: {values.sum()}")

    codes.__dict__[key] = val.reindex(
      columns=[col for col in val.columns if col != 'title']
      + ['title'])

  logger.info("Saving")

  @utilities.open_excel(dirs.target / config.target.file)
  def save_sheets(writer, codes, config):
    for key, val in config.target.sheets.__dict__.items():
      logger.info(f"Saving classification '{key}'")
      codes.__dict__[key].to_excel(
        writer,
        sheet_name=val,
        index=True)

  save_sheets('_', codes, config)

  logger.info("Finished step")

  return None


def process_data(task_str, full_config):
  logger.info("Loading relevant QCEW raw data")
  config = full_config.__dict__[task_str]
  dirs = full_config.dirs

  # This block duplicates the code for loading current data
  # To revise if time and money allow
  logger.info("Loading previous QCEW raw data")
  t0 = time()
  chunks = pd.read_csv(dirs.source / config.source_previous,
                       chunksize=10000)
  raw = pd.concat(chunks)
  logger.info(f"Read data in {time() - t0:.2f}s")

  for key, check in config.checks.__dict__.items():
    logger.info(f"Field '{key}', column '{check.column}' "
                f"has {(raw[check.column] != check.value).sum()} "
                f"entries with value different from {check.value}")

  logger.info("Renaming fields")
  t0 = time()
  source_previous = pd.DataFrame(index=raw.index)
  for key, val in config.columns.__dict__.items():
    source_previous[key] = raw[val]
  del raw

  logger.info("Keep only national level")
  source_previous = source_previous.loc[source_previous.area == 'US000']

  logger.info("Changing nondisclosed flag to boolean")
  source_previous.nondisclosed = (
    source_previous.nondisclosed == config.nondisclosed)

  logger.info(f"{source_previous.columns}")
  logger.info("Combining ownership and industry codes")
  source_previous.ownership = source_previous.ownership.astype(str)
  source_previous['oind'] = [
    x + '_' + y for x, y in zip(source_previous['ownership'],
                                source_previous['industry'])]
  source_previous = source_previous[
    ['area', 'oind', 'nondisclosed', 'establishments',
     'employment', 'wages']]

  source_previous.to_csv(dirs.target / config.target.data_previous,
                         index=False)

  logger.info("Finished saving previous data")
  # End of duplicated code

  t0 = time()
  chunks = pd.read_csv(dirs.source / config.source,
                       chunksize=10000)
  raw = pd.concat(chunks)
  logger.info(f"Read data in {time() - t0:.2f}s")

  for key, check in config.checks.__dict__.items():
    logger.info(f"Field '{key}', column '{check.column}' "
                f"has {(raw[check.column] != check.value).sum()} "
                f"entries with value different from {check.value}")

  logger.info("Renaming fields")
  t0 = time()
  source = pd.DataFrame(index=raw.index)
  for key, val in config.columns.__dict__.items():
    source[key] = raw[val]
  del raw

  logger.info("Changing nondisclosed flag to boolean")
  source.nondisclosed = (
    source.nondisclosed == config.nondisclosed)

  logger.info("Removing records not referenced in codes")

  logger.info("Loading codes")
  codes = operations.load_codes()

  logger.info(f"{len(source)} initial records")
  for key, val in codes.__dict__.items():
    if key != 'oind':
      allowed_values = list(val.code)
      source[key] = source[key].astype(str)
      source = source[source[key].isin(allowed_values)]
      logger.info(f"{len(source)} records after '{key}' parsing")

  logger.info(f"{source.columns}")
  logger.info("Combining ownership and industry codes")
  source['oind'] = [
    x + '_' + y for x, y in zip(source['ownership'],
                                source['industry'])]
  source = source[
    ['area', 'oind', 'nondisclosed', 'establishments',
     'employment', 'wages']]

  # create tree information
  logger.info("Constructing aggregation tree")
  qcew_source = source
  del source
  qcew_tree = pd.DataFrame()
  qcew_tree['current'] = [
    str(x) + '_' + str(y) for x, y in zip(qcew_source.area,
                                          qcew_source.oind)]

  qcew_tree['area_level'] = list(
    codes.area.level[qcew_source.area])

  qcew_tree['oind_level'] = list(
    codes.oind.level[qcew_source.oind])

  qcew_tree['area_parent'] = [
    str(x) + '_' + str(y) if z > 0 else None
    for x, y, z in zip(
      list(codes.area.parent[qcew_source.area]),
      qcew_source.oind,
      qcew_tree.area_level)]
  qcew_tree.area_parent = qcew_tree.area_parent.astype(str)

  qcew_tree['oind_parent'] = [
    str(x) + '_' + str(y) if z > 0 else None
    for x, y, z in zip(
      qcew_source.area,
      list(codes.oind.parent[qcew_source.oind]),
      qcew_tree.oind_level)]
  qcew_tree.oind_parent = qcew_tree.oind_parent.astype(str)
  qcew_tree = qcew_tree.replace('nan', None)

  # checking for missing records
  logger.info("Checking for missing area parent records")
  missing_area_parent = qcew_tree.loc[
    (~ qcew_tree.area_parent.isin(qcew_tree.current)) &
    (qcew_tree.area_level > 0)]
  logger.info(f"Found {len(missing_area_parent)}")
  if len(missing_area_parent) > 0:
    qcew_tree_tmp = []
    qcew_source_tmp = []
    for index, row in missing_area_parent.iterrows():
      area_code = str(row.area_parent.split('_')[0])
      oind_code = '_'.join(row.area_parent.split('_')[1:])
      area_level = row.area_level - 1
      oind_level = row.oind_level
      if row.area_level == 0:
        area_parent = None
      else:
        area_parent = '_'.join([
          str(codes.area.parent[area_code]),
          oind_code])
      if row.oind_level == 0:
        oind_parent = None
      else:
        oind_parent = '_'.join([
          area_code,
          str(codes.oind.parent[oind_code])])

      oind_parent = codes.oind.parent[oind_code]
      qcew_tree_tmp.append({
        'current': '_'.join([area_code, oind_code]),
        'area_level': area_level,
        'oind_level': oind_level,
        'area_parent': area_parent,
        'oind_parent': '_'.join([area_code, oind_parent]),
      })
      qcew_source_tmp.append({
        'area': area_code,
        'oind': oind_code,
        'nondisclosed': qcew_source.nondisclosed[index],
        'establishments': qcew_source.establishments[index],
        'employment': qcew_source.employment[index],
        'wages': qcew_source.wages[index],
      })
    qcew_tree = pd.concat([qcew_tree, pd.DataFrame(qcew_tree_tmp)],
                          axis=0)
    qcew_tree.reset_index(drop=True, inplace=True)
    qcew_source = pd.concat([qcew_source,
                             pd.DataFrame(qcew_source_tmp)],
                            axis=0)
    qcew_source.reset_index(drop=True, inplace=True)
    logger.info("Added missing area parents")

  logger.info("Checking for other missing records")
  tmp = qcew_tree.loc[
    (~ qcew_tree.area_parent.isin(qcew_tree.current)) &
    (qcew_tree.area_level > 0)]
  if len(tmp) > 0:
    logger.warning(f"Found {len(tmp)} missing parent tree records")

  tmp = qcew_tree.loc[
    (~ qcew_tree.oind_parent.isin(qcew_tree.current)) &
    (qcew_tree.oind_level > 0)]
  if len(tmp) > 0:
    logger.warning(f"Found {len(tmp)} missing parent oind records")

  tmp = qcew_tree.loc[
    (~ qcew_tree.current.isin(qcew_tree.area_parent)) &
    (qcew_tree.area_level < 2)]
  if len(tmp) > 0:
    logger.warning(
      f"Found {len(tmp)} missing offspring tree records")

  tmp = qcew_tree.loc[
    (~ qcew_tree.current.isin(qcew_tree.oind_parent)) &
    (qcew_tree.oind_level < 6)]
  if len(tmp) > 0:
    logger.warning(
      f"Found {len(tmp)} missing offspring oind records")

  logger.info("Saving source data and tree")
  qcew_source.to_csv(dirs.target / config.target.data,
                     index=False)
  qcew_tree.to_csv(dirs.target / config.target.tree,
                   index=False)

  logger.info("Calculating aggregates across levels")

  # create result dataframe
  indices = list(itertools.product(*[
    list(set(x.level)) for x in
    codes.__dict__.values()]))

  df = []
  # populate result dataframe
  for index in indices:
    t0 = time()
    lists = {
      'area': list(codes.area.index[
        codes.area.level == index[0]]),
      'oind': list(codes.oind.index[
        codes.oind.level == index[1]]),
    }
    lists = utilities.DictToObject(lists)

    select = qcew_source.loc[
      (qcew_tree.area_level == index[0])
      & (qcew_tree.oind_level == index[1])]
    df.append({
      'level_area': index[0],
      'level_oind': index[1],
      'n_area': len(lists.area),
      'n_oind': len(lists.oind),
      'n_potential': len(lists.area) * len(lists.oind),
      'n_total': len(select),
      'n_nondisclosed': select.nondisclosed.sum(),
      'val_establishments': select.establishments.sum(),
      'val_employment': select.employment.sum(),
      'val_wages': select.wages.sum(), })
  del index
  df = pd.DataFrame(df)

  @utilities.open_excel(dirs.target / config.target.stat_file)
  def save_sheets(writer, results, config):
    results.to_excel(writer,
                     sheet_name=config.target.stat_sheet,
                     index=False)
  save_sheets('_', df, config)

  logger.info("Finished step")

  return None



def import_filled(task_str, full_config):
  """RECON2024: replace process_data + balance with the v6 / stage-A panel.

  The disclosure-filled QCEW panel (private own 5 from the row-normalised
  exact-MAP solve, government own 1-3 from stage_a_gov_fill.py) is already
  laid out in the RECON2022 process format:
    qcew_source_{year}.parquet   area, oind, nondisclosed, est, emp, wages
    qcew_tree_{year}.parquet     current, area_level, oind_level, parents
    qcew_balanced_{year}.parquet est, emp, wages (row-aligned with source)
  This step writes them as the CSVs the later steps expect, and builds
  qcew_previous.csv (national rows of the {prev_year} balanced panel) for the
  wage-comparison sheet of process_qcew_bridge.
  """
  config = full_config.__dict__[task_str]
  dirs = full_config.dirs
  year = full_config.year
  prev = config.previous_year

  logger.info(f"Importing filled QCEW panel for {year} from {dirs.source}")
  def fname(key, yr):
    return config.source.__dict__[key] + str(yr) + config.source.suffix
  src = pd.read_parquet(dirs.source / fname('data', year))
  tree = pd.read_parquet(dirs.source / fname('tree', year))
  bal = pd.read_parquet(dirs.source / fname('balanced', year))
  if not (len(src) == len(tree) == len(bal)):
    raise ValueError("source, tree and balanced panels are not row-aligned")
  src['area'] = src['area'].astype(str)
  src['oind'] = src['oind'].astype(str)
  for col in ['establishments', 'employment', 'wages']:
    bal[col] = bal[col].round(0).astype('int64')
    src[col] = src[col].fillna(0).round(0).astype('int64')

  logger.info("Checking that published cells are reproduced")
  pub = ~src.nondisclosed.values
  for col in ['establishments', 'employment', 'wages']:
    dev = (bal.loc[pub, col].values - src.loc[pub, col].values)
    logger.info(f"  {col}: max |balanced - published| on disclosed cells "
                f"= {abs(dev).max()}")

  logger.info(f"Building previous ({prev}) national panel")
  psrc = pd.read_parquet(dirs.source / fname('data', prev))
  pbal = pd.read_parquet(dirs.source / fname('balanced', prev))
  psrc['area'] = psrc['area'].astype(str)
  keep = (psrc.area == config.total_area_code).values
  previous = psrc.loc[keep, ['area', 'oind', 'nondisclosed']].copy()
  for col in ['establishments', 'employment', 'wages']:
    previous[col] = pbal.loc[keep, col].round(0).astype('int64').values

  logger.info("Saving in RECON process format")
  dirs.target.mkdir(parents=True, exist_ok=True)
  src.to_csv(dirs.target / config.target.data, index=False)
  tree.to_csv(dirs.target / config.target.tree, index=False)
  bal[['establishments', 'employment', 'wages']].to_csv(
    dirs.target / config.target.balanced, index=False)
  previous.to_csv(dirs.target / config.target.data_previous, index=False)
  logger.info(f"Rows: {len(src)} ({year}), {len(previous)} previous national rows")
  logger.info("Finished step")
  return None


def balance(task_str, full_config):
  logger.info("Estimating nondisclosed values and aligning")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")

  qcew_data = operations.load_codes(load_results=False)
  qcew_tree = qcew_data.tree
  qcew_source = qcew_data.source
  del qcew_data

  max_area = max(qcew_tree.area_level)
  max_oind = max(qcew_tree.oind_level)

  results = pd.DataFrame(index=qcew_tree.index).astype(int)
  logger.info("Balancing establishments")

  logger.info("Creating variable vector")
  t = pd.DataFrame(index=qcew_tree.index,)
  t['pos'] = range(len(t))
  t['val'] = qcew_source.establishments
  t.loc[t.val == 0, 'val'] = int(1)
  t['flag'] = int(1)
  t.loc[qcew_source.nondisclosed, 'flag'] = int(2)
  t.loc['US000_0_10', 'flag'] = int(0)
  qcew_tree['pos'] = t['pos']

  logger.info("Creating aggregation matrix")
  g = pd.DataFrame()
  for pos, dim in enumerate(['area', 'oind']):
    g0 = pd.DataFrame()
    g0 = deepcopy(qcew_tree.loc[qcew_tree[dim + '_parent'].notna()])
    g0[dim + '_pos'] = list(t.loc[list(g0[dim + '_parent']), 'pos'])

    g1 = pd.DataFrame()
    g1['col'] = g0.pos
    g1['row'] = g0[dim + '_pos'] + pos * len(qcew_tree)
    g1['val'] = int(1)
    g = pd.concat([g, g1], axis=0)

    g0 = pd.Series(list(set(g0[dim + '_pos']))).astype(int)
    g1 = pd.DataFrame(index=list(qcew_tree.iloc[list(g0)].index))
    g1['col'] = list(g0)
    g1['row'] = list(g0 + pos * len(qcew_tree))
    g1['val'] = int(-1)
    g = pd.concat([g, g1], axis=0)
    del g0, g1, pos, dim
  g.index = range(len(g))

  bal = utilities.Balance.from_array(
    t, g, config.params.establishments)
  del g, qcew_tree['pos']
  bal.run()
  results['establishments'] = operations.rescale_integer(
    bal.tval,
    qcew_tree,
    eps=config.params.establishments.eps,
    nexit=config.params.establishments.nmax,)

  for dim, proxy in [['employment', 'establishments'],
                     ['wages', 'employment']]:

    logger.info(f"Balancing {dim} using {proxy} as proxy")

    t['val'] = operations.set_initial(
      qcew_source[dim],
      results[proxy],
      qcew_source.nondisclosed,
      qcew_tree,)

    bal.tval = np.array(t.val.astype(float))
    bal.param = config.params.__dict__[dim]
    bal.run()
    results[dim] = operations.rescale_integer(
      bal.tval,
      qcew_tree,
      eps=config.params.__dict__[dim].eps,
      nexit=config.params.__dict__[dim].nmax,)

  for dim in results.columns:
    logger.info(f"Bottom-up correction of {dim}")
    results[dim] = operations.bottom_up_correction(
      results[dim], qcew_tree,).astype(int)

  logger.info('Saving data')
  results.to_csv(
    dirs.target / config.target.data,
    index=False)

  logger.info('Calculating statistics')

  indices = list(itertools.product(*[
    list(set(range(n))) for n in
    [max_area + 1, max_oind + 1]]))

  df = []
  # populate result dataframe
  for index in indices:
    select_source = qcew_source.loc[
      (qcew_tree.area_level == index[0])
      & (qcew_tree.oind_level == index[1])]
    select_results = results.loc[select_source.index]

    dict_tmp = {
      'k_area': index[0],
      'k_oind': index[1],
      'n_total': len(select_source),
      'n_ndisc': select_source.nondisclosed.sum(), }

    for dim in select_results.columns:
      diff = abs(select_source.loc[~ qcew_source.nondisclosed, dim]
                 - select_results.loc[~ qcew_source.nondisclosed, dim])

      dict_tmp[dim[:3] + '_ndisc'] = select_results.loc[
        select_source.nondisclosed, dim].sum()
      dict_tmp[dim[:3] + '_median'] = np.rint(np.median(diff))
      dict_tmp[dim[:3] + '_mean'] = np.rint(diff.mean())
      dict_tmp[dim[:3] + '_max'] = diff.max()

    df.append(dict_tmp)
  del index
  df = pd.DataFrame(df)

  @utilities.open_excel(dirs.target / config.target.stat_file)
  def save_sheets(writer, results, config):
    results.to_excel(writer,
                     sheet_name=config.target.stat_sheet,
                     index=False)
  save_sheets('_', df, config)

  logger.info("Finished step")

  return None


def process_qcew_bridge(task_str, full_config):
  logger.info("Importing QCEW industry/ownership code pairs")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]
  val_cols = config.target.sheets.split

  logger.info("Loading codes and data")
  codes = operations.load_codes()
  data = operations.load_data()

  logger.info(
    "Selecting and rescaling national wages to million dollar")
  wages = deepcopy(codes.oind)
  wages[val_cols] = int(0)

  tmp_ = deepcopy(data.previous)
  tmp_.set_index('oind', inplace=True, drop=True)
  tmp_ = tmp_.loc[tmp_.index.isin(wages.index)]
  wages.loc[tmp_.index, 'previous'] = tmp_['wages']

  tmp_ = deepcopy(data.source[['area', 'oind']])
  tmp_['source'] = data.source['wages']
  tmp_['results'] = data.results['wages']
  tmp_ = tmp_.loc[tmp_.area == config.total_area_code]
  tmp_.set_index('oind', inplace=True, drop=True)
  wages.loc[tmp_.index, ['source', 'results']] = tmp_[[
    'source', 'results']]

  wages[val_cols] = (wages[val_cols]
                     * (1 / config.rescale_factor)).astype(int)

  wages = wages[val_cols + list(codes.oind.columns)]

  logger.info("Saving in ownership/industry format")

  dir_ = dirs.__dict__[config.target.folder]

  @utilities.open_excel(dir_ / config.target.file)
  def save_sheets(writer, results, config):
    results.to_excel(writer,
                     sheet_name=config.target.sheets.oind,
                     index=True)
  save_sheets('_', wages, config)

  logger.info("Splitting ownership from industry format")
  for col in val_cols:
    logger.info(f"Wages {col}")
    results = deepcopy(codes.industry)
    for own in codes.ownership.index:
      tmp_ = wages.loc[wages.ownership == own]
      results.loc[tmp_.industry.values, own] = tmp_[col].values

    results['0'] = 0
    results['0'] = results[list(codes.ownership.index)].sum(1)
    results = results[list(codes.ownership.index) +
                      list(codes.industry.columns)]

    dir_ = dirs.__dict__[config.target.folder]

    @utilities.open_excel(dir_ / config.target.file)
    def save_sheets(writer, results, col):
      results.to_excel(writer,
                       sheet_name=col,
                       index=True)

    save_sheets('_', results, col)

  logger.info("Finished step")

  return None


def process_naics_bridge(task_str, full_config):
  logger.info("Calculating QCEW industry/ownership code pairs")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  qcew = pd.read_excel(
    dirs.target / config.target.file,
    sheet_name='previous',
    index_col='code')

  # load bridge
  dir_ = dirs.__dict__[config.source.folder]
  col0, row0, col1, row1 = config.source.naics.range
  bridge = pd.read_excel(
    dir_ / config.source.file,
    sheet_name=config.source.naics.sheet,
    index_col=None,
    header=None,
    skiprows=row0 - 1, nrows=(1 + row1 - row0),
    usecols=f'{col0}:{col1}')
  bridge.columns = config.source.naics.columns
  bridge.sut = bridge.sut.astype(str)
  bridge.qcew = bridge.qcew.astype(str)
  bridge.drop(columns=['notes'], inplace=True)

  # load SUT industry classification
  dir_ = dirs.__dict__[config.source.folder]
  col0, row0, col1, row1 = config.source.industries.range

  sut = pd.read_excel(
    dir_ / config.source.file,
    sheet_name=config.source.industries.sheet,
    index_col=None,
    header=None,
    skiprows=row0 - 1, nrows=(1 + row1 - row0),
    usecols=f'{col0}:{col1}')
  sut = sut.T
  sut.columns = config.source.industries.columns
  sut.code = sut.code.astype(str)
  sut.set_index('code', drop=True, inplace=True)

  # load wages
  col0, row0, col1, row1 = config.source.wages.range
  wages = pd.read_excel(
    dir_ / config.source.file,
    sheet_name=config.source.wages.sheet,
    index_col=None,
    header=None,
    skiprows=row0 - 1, nrows=(1 + row1 - row0),
    usecols=f'{col0}:{col1}')
  wages = wages.T
  wages.columns = config.source.wages.columns
  wages.fillna(0, inplace=True)
  wages.wages = wages.wages.astype(int)
  wages.index = sut.index
  wages['title'] = sut.title
  sut = wages
  del wages, col0, row0, col1, row1

  logger.info("Clean bridge table")
  # remove extraneous rows
  bridge = bridge.loc[bridge.sut.isin(sut.index)]
  difference = list(set(sut.index).difference(bridge.sut))
  bridge.reset_index(drop=True, inplace=True)
  logger.info(
    f"{len(difference)} SUT codes were not found in bridge")
  if len(difference) > 0:
    logger.info(f"{difference}")
  del difference

  # expand commas
  logger.info("Expanding QCEW codes separated by columns")
  trunc = deepcopy(bridge.loc[bridge.qcew.str.contains(',')])
  bridge = deepcopy(bridge.loc[~ bridge.qcew.str.contains(',')])
  count = 0
  for _, row in trunc.iterrows():
    count += row.qcew.count(',') + 1

  bridge_tmp = pd.DataFrame(columns=bridge.columns,
                            index=range(count),)
  count = 0
  for _, row in trunc.iterrows():
    string = row.qcew.split(',')
    for substring in string:
      bridge_tmp.loc[count] = {'sut': row.sut,
                               'title': row.title,
                               'qcew': substring}
      count += 1
  bridge = pd.concat([bridge, bridge_tmp], axis=(0))
  bridge.reset_index(drop=True, inplace=True)
  del bridge_tmp, count, trunc, row, substring, string

  #  expand dashes
  logger.info("Expanding QCEW codes separated by dashes. Not found:")
  trunc = deepcopy(bridge.loc[bridge.qcew.str.contains('-')])
  bridge = deepcopy(bridge.loc[~ bridge.qcew.str.contains('-')])
  count = 0
  for _, row in trunc.iterrows():
    string = row.qcew.split('-')
    last_digit = int(string[-1])
    first_digit = int(string[0][-1])
    count += last_digit - first_digit + 1

  bridge_tmp = pd.DataFrame(columns=bridge.columns,
                            index=range(count),)
  count = 0
  for _, row in trunc.iterrows():
    string = row.qcew.split('-')
    last_digit = int(string[-1])
    first_digit = int(string[0][-1])
    for digit in range(first_digit, last_digit + 1):
      new_code = (string[0][:-1]).strip() + str(digit)
      if new_code in qcew.index:
        bridge_tmp.loc[count] = {'sut': row.sut,
                                 'title': row.title,
                                 'qcew': new_code}
        count += 1
      else:
        logger.info(f"\t{row.sut}: {new_code}")

  bridge = pd.concat([bridge, bridge_tmp.iloc[:count]], axis=(0))
  bridge.reset_index(drop=True, inplace=True)
  del (bridge_tmp, count, trunc, row, string, first_digit, last_digit,
       digit, new_code)

  # non-numerical
  logger.info("Removing non-numerical characters in QCEW codes")
  # bridge_na = bridge.loc[bridge.qcew == 'n.a.']
  bridge = deepcopy(bridge.loc[bridge.qcew != 'n.a.'])
  bridge.qcew = [re.sub("[^0-9]", "", str_) for str_ in
                 list(bridge.qcew)]  # remove \n and *

  # non-existent
  difference = bridge.loc[~ bridge.qcew.isin(qcew.index)]
  logger.info(f"{len(difference)} non-existing QCEW code(s) found")
  if len(difference) > 0:
    logger.info(f"To be removed:\n\t{list(difference.qcew)}")
    logger.info(f"In the 2022 revision:\n\t{config.non_existent}")
    logger.info("Do these sets match? "
                f"{list(difference.qcew) == config.non_existent}")
    bridge = deepcopy(bridge.drop(difference.index))
    bridge.reset_index(drop=True, inplace=True)
  del difference

  # duplicates
  bridge_duplicate = deepcopy(bridge.loc[bridge.qcew.isin(
    bridge.loc[bridge.qcew.duplicated()].qcew)])
  bridge = deepcopy(bridge.loc[~ bridge.qcew.isin(
    bridge.loc[bridge.qcew.duplicated()].qcew)])
  logger.info(f"{len(bridge_duplicate.qcew)} duplicated QCEW codes in "
              f"{len(bridge_duplicate)} benchmark SUT codes")
  if len(bridge_duplicate) > 0:
    logger.info(f"\n{bridge_duplicate[['sut', 'qcew']]}")
  logger.info(
    f"Expected duplicates are:\n{config.duplicates.__dict__}")
  del bridge_duplicate
  bridge.reset_index(drop=True, inplace=True)

  logger.info("Adding duplicate matches")
  duplicate_match = []
  for key, val in config.duplicate_match.__dict__.items():
    logger.info(f"\n\t{key}: {val}")
    for item in val:
      duplicate_match.append({
        'sut': str(key),
        'sut_title': sut.loc[key, 'title'],
        'qcew': str(item)})
  bridge.rename(columns={'title': 'sut_title'}, inplace=True)
  bridge = pd.concat([bridge, pd.DataFrame(duplicate_match)], axis=0)
  bridge.reset_index(drop=True, inplace=True)
  del duplicate_match

  logger.info("Adding reclassified industries (2017 to 2022)")
  reclassified = []
  for key, val in config.reclassified.__dict__.items():
    reclassified.append({
      'sut': str(val),
      'sut_title': sut.loc[str(val), 'title'],
      'qcew': str(key)})
    logger.info(f"\n\t{key} (qcew) -> {val} (sut)")
  bridge = pd.concat([bridge, pd.DataFrame(reclassified)], axis=0)
  bridge.reset_index(drop=True, inplace=True)
  del reclassified

  bridge['version'] = list(qcew.loc[list(bridge.qcew), 'version'])
  bridge['qcew_title'] = list(qcew.loc[list(bridge.qcew), 'title'])
  bridge.index.name = 'code'

  logger.info("Saving data")
  dir_ = dirs.__dict__[config.target.folder]
  with pd.ExcelWriter(dir_ / config.target.file,
                      engine='openpyxl',
                      mode='a',
                      if_sheet_exists='replace') as writer:
    bridge.to_excel(writer, sheet_name=config.target.sheets.bridge,
                    index=True)
    sut.to_excel(writer, sheet_name=config.target.sheets.sut,
                 index=True)

  logger.info("Finished step")

  return None



def retarget_sut_axis(task_str, full_config):
  """RECON2024: move the SUT industry universe from the 2017 detail axis (405
  industries, construction as 12 BEA structure types) to the 421 axis of the
  construction package (31 NAICS construction industries). Rewrites the 'sut'
  sheet of qcew_bridge.xlsx; the 'bridge' sheet has no construction rows
  (construction is handled by apply_bridge via construction.xlsx).
  Wages for the new rows are {year} compensation (V00100) from the 421 use
  table, in the same $ million unit as the 2017 column they replace."""
  config = full_config.__dict__[task_str]
  dirs = full_config.dirs
  axis = np.load(dirs.__dict__[config.axis.folder] / config.axis.file,
                 allow_pickle=True)
  ind421 = [str(x) for x in axis['ind']]
  naics31 = [str(x) for x in axis['naics']]
  use = pd.read_csv(dirs.__dict__[config.axis.folder] / config.axis.use_table,
                    index_col=0)
  wages = use.loc[config.axis.wages_row]

  path = dirs.__dict__[config.bridge.folder] / config.bridge.file
  sut = pd.read_excel(path, sheet_name=config.bridge.sheet, dtype=str)
  before = len(sut)
  keep = ~sut.code.str.startswith(tuple(config.drop_prefix))
  dropped = sut.loc[~keep, 'code'].tolist()
  sut = sut.loc[keep]
  new = pd.DataFrame({'code': naics31,
                      'wages': [int(round(wages[n])) for n in naics31],
                      'title': ['NAICS ' + n + ' construction' for n in naics31]})
  sut = pd.concat([sut, new], ignore_index=True)
  sut['wages'] = sut['wages'].astype(float).round(0).astype(int)
  missing = sorted(set(ind421) - set(sut.code))
  extra = sorted(set(sut.code) - set(ind421))
  logger.info(f"SUT universe: {before} -> {len(sut)} industries; dropped "
              f"{dropped}; not in 421 axis: {extra}; 421 codes absent: {missing}")
  order = {c: i for i, c in enumerate(ind421)}
  sut['_o'] = sut.code.map(order).fillna(10 ** 6)
  sut = sut.sort_values('_o').drop(columns='_o').reset_index(drop=True)

  # replace only the 'sut' sheet, leave the others untouched
  import openpyxl
  wb = openpyxl.load_workbook(path)
  if config.bridge.sheet in wb.sheetnames:
    del wb[config.bridge.sheet]
  ws = wb.create_sheet(config.bridge.sheet)
  ws.append(list(sut.columns))
  for row in sut.itertuples(index=False):
    ws.append(list(row))
  wb.save(path)
  logger.info("Finished step")
  return None


def test_bridge(task_str, full_config):
  logger.info("Testing QCEW to SUT bridge (complete and not duplicate")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info("Loading data")
  bridge = operations.QcewToSut()

  logger.info("Selecting nonzero oind entries")
  bridge.qcew['tot'] = bridge.qcew[[
    'previous', 'source', 'results']].sum(1)
  qcew_nnz = bridge.qcew.loc[bridge.qcew.tot != 0]
  logger.info(f"Reduced from {len(bridge.qcew)} to {len(qcew_nnz)}")

  bridge_nnz = pd.DataFrame(columns=qcew_nnz.index,
                            index=bridge.sut.index,
                            data=0)

  logger.info('Started looping over every nonzero entry')
  time_report_step = 10
  t0 = time()
  t1 = time()
  n = len(bridge_nnz.columns)
  for k, index in enumerate(bridge_nnz.columns):
    qcew = pd.Series(index=bridge.codes.oind.index,
                     data=int(0))
    qcew.loc[index] = int(1)
    bridge_nnz[index] = bridge.apply(qcew)
    t2 = time()
    if t2 - t1 > time_report_step:
      t1 = t2
      dt = t2 - t0
      tt = dt * (n - k + 1) / (k + 1)
      logger.info(f'{k}: {k/n*100:.1f}%, {dt:.1f}s, {tt:.1f}s')

  bridge_backup = deepcopy(bridge_nnz.T)
  bridge_nnz = bridge_nnz.loc[bridge_nnz.sum(1) > 0]
  bridge_nnz = bridge_nnz.T
  bridge_nnz = bridge_nnz.loc[bridge_nnz.sum(1) > 0]

  logger.info(f"Storing nonzeros, reduced from {bridge_backup.shape} "
              f"to {bridge_nnz.shape}")

  logger.info("Saving")

  dir_ = dirs.__dict__[config.target.folder]

  @utilities.open_excel(dir_ / config.target.file)
  def save_sheets(writer, results, config):
    results.to_excel(writer,
                     sheet_name=config.target.sheet,
                     index=True)

  save_sheets('_', bridge_nnz, config)

  logger.info("Finished step")

  return None


def apply_bridge(task_str, full_config):
  logger.info("Bridging QCEW wages to benchmark classification")
  dirs = full_config.dirs
  config = full_config.__dict__[task_str]

  logger.info('Loading bridge parameters')
  bridge = operations.QcewToSut()

  logger.info('Bridging national data for comparison')
  sut = pd.DataFrame(index=bridge.sut.index)
  for col in bridge.qcew.columns:
    sut[col] = bridge.apply(bridge.qcew[col])

  sut[bridge.sut.columns] = bridge.sut

  logger.info(
    'Wage totals comparison (million dollar): \ncolumns:'
    'previous = 2017 raw, source = 2022 raw, results = 2022 balanced, '
    'wages = 2017 SUT\nrows: bridge = SUT classification, '
    'grand = grand total in QCEW classification\n'
    'differences in "previous" and "source" might result from source '
    'discrepancies, "results" only from rounding of integers')

  total = pd.DataFrame(columns=sut.columns)
  total.loc['bridge'] = sut.sum()
  total.loc['grand', bridge.qcew.columns] = bridge.qcew.loc['0_10']
  total.drop('title', axis=1, inplace=True)
  total.fillna(0, inplace=True)

  logger.info(f'{total}')
  logger.info(
    'To further explore discrepancies run step "test_bridge" '
    'to identify QCEW sectors mapped to SUT (excluding construction '
    'and autos and trucks (see apply_bridge in pipeline.yaml) and '
    'execute in a script:'
    '\tfrom recon2024.qcew import operations\n'
    '\tbridge = operations.QcewToSut()\n'
    '\tbridge_sut = operations.load_qcew_sut_sparse_bridge(bridge)\n'
    '\tbridge_multi = operations.load_qcew_multilevel_sparse'
    '_bridge(bridge)\n'
    'The latter two are bridges from all QCEW-SUT and from the bottom '
    'QCEW level to all QCEW levels, they can be used to explore')

  logger.info('Saving comparison')
  dir_ = dirs.__dict__[config.statistics.folder]

  @utilities.open_excel(dir_ / config.statistics.file)
  def save_sheets(writer, results, config):
    results.to_excel(writer,
                     sheet_name=config.statistics.sheet,
                     index=True)

  save_sheets('_', sut, config)

  logger.info('Loading bulk data')
  qcew_data = operations.load_data()
  qcew = pd.DataFrame()
  qcew['area'] = qcew_data.source.area
  qcew['oind'] = qcew_data.source.oind
  qcew['establishments'] = qcew_data.results.establishments  # RECON2024
  qcew['employment'] = qcew_data.results.employment
  qcew['wages'] = qcew_data.results.wages
  del qcew_data

  logger.info("Converting every area, may take several minutes")
  results = pd.DataFrame(columns=[
    'area', 'industry', 'establishments', 'employment', 'wages'])
  results = results.astype(int)
  time_report_step = 10
  t0 = time()
  t1 = time()
  n = len(bridge.codes.area.index)
  for k, area in enumerate(bridge.codes.area.index):
    # logger.info(
    #   f"{area}: {bridge.codes.area.loc[area, 'title']}")
    results_tmp = pd.DataFrame(columns=[
      'area', 'establishments', 'employment', 'wages'],
      index=bridge.sut.index,
      data=int(0))
    results_tmp = results_tmp.astype(int)
    qcew_tmp = qcew.loc[(qcew.area == area)]
    qcew_tmp.index = qcew_tmp.oind
    for quantity in ['establishments', 'employment', 'wages']:
      tmp = bridge.apply(qcew_tmp[quantity])
      results_tmp.loc[tmp.index, quantity] = tmp.astype(int)
    results_tmp['industry'] = results_tmp.index
    results_tmp['area'] = area
    results = pd.concat([results, results_tmp], axis=0)
    t2 = time()
    if t2 - t1 > time_report_step:
      t1 = t2
      dt = t2 - t0
      tt = dt * (n - k + 1) / (k + 1)
      logger.info(f'{k}: {k/n*100:.1f}%, {dt:.1f}s, {tt:.1f}s')

  logger.info("Saving all data")
  dir_ = dirs.__dict__[config.target.folder]
  results.to_csv(dir_ /
                 config.target.file, index=False)

  logger.info("Finished step")

  return None
