"""Utilties."""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from time import time
from pathlib import Path
from typing import Union
import logging
import os
import yaml
from os.path import dirname
import sys
from datetime import datetime
logger = logging.getLogger('root')


ROOT_PATH = Path(os.getcwd())
INTERNAL_PATH = Path(dirname(__file__))


def safe_division(numerator, denominator):
  return numerator * safe_inverse(denominator)


def safe_inverse(var):
  return (var != 0) / (var + (var == 0))


def open_excel(filepath, makedir=False):
  """Decorator that saves to excel file, whether it exists or not"""
  def decorator(func):
    def wrapper(writer, *args, **kwargs):

      if not os.path.exists(os.path.dirname(filepath)):
        if makedir:
          os.makedirs(os.path.dirname(filepath))
        else:
          logger.error(f"Directory of {filepath} does not exist")
          raise OSError

      if os.path.exists(filepath):
        writer_args = {'mode': 'a', 'if_sheet_exists': 'replace'}
      else:
        writer_args = {'mode': 'w'}

      with pd.ExcelWriter(filepath,
                          engine='openpyxl',
                          **writer_args) as writer:
        return func(writer, *args, **kwargs)
    return wrapper
  return decorator



RUN_OVERRIDES = {}   # set by main() from the command line (year, prev_year, process_dir, results_dir)


def load_stages():
  """Read stages.yaml; pick the base directory for this OS."""
  import os
  with open(INTERNAL_PATH / 'stages.yaml', 'r') as file:
    d = yaml.safe_load(file)
  d.update({k: v for k, v in RUN_OVERRIDES.items() if v is not None})
  dirs = d['dirs']
  if 'base' not in dirs or dirs['base'] in (None, ''):
    dirs['base'] = dirs['base_windows'] if os.name == 'nt' else dirs['base_posix']
  if os.name != 'nt' and dirs['base'].startswith('~'):
    dirs['base'] = os.path.expanduser(dirs['base'])
  return d


def load_stage_yaml(stage, base_dict):
  """Read <stage>/pipeline.yaml, substituting {year} and {prev_year} tokens."""
  subs = {'year': str(base_dict['year']),
          'prev_year': str(base_dict.get('prev_year', int(base_dict['year']) - 2)),
          'process': str(base_dict.get('process_dir', 'process')),
          'results': str(base_dict.get('results_dir', 'results'))}
  with open(INTERNAL_PATH / stage / 'pipeline.yaml', 'r') as file:
    text = file.read()
  for k, v in subs.items():
    text = text.replace('{' + k + '}', v)
  return yaml.safe_load(text)


def get_config(stage):
  base_dict = load_stages()
  base_config = DictToObject(base_dict)
  config = DictToObject(load_stage_yaml(stage, base_dict))
  for key, val in config.dirs.__dict__.items():
    setattr(config.dirs, key, Path(base_config.dirs.base) / val)
  config.year = base_dict['year']
  return config


def datetime_now():
  return datetime.now().strftime("%Y%m%d_%H%M%S")


class Generic():
  def __init__(self):
    pass


class DictToObject:
  def __init__(self, data):
    for key, value in data.items():
      if isinstance(value, dict):
        setattr(self, key, DictToObject(value))
      elif isinstance(value, list):
        setattr(self, key, [DictToObject(item) if isinstance(
          item, dict) else item for item in value])
      else:
        setattr(self, key, value)

  def reverse(self):
    data = {}
    for key, value in self.__dict__.items():
      if isinstance(value, DictToObject):
        data[key] = value.reverse()
      elif isinstance(value, list):
        data[key] = [
          item.reverse() if isinstance(item, DictToObject)
          else item for item in value]
      else:
        data[key] = value

    return data


def set_logger(
    filename: Union[str, Path] = None,
    path: str = os.getcwd(),
    log_level: int = 20,
    log_format: str = (
      "%(asctime)s | [%(levelname)s]: %(message)s"
    ),
    overwrite=False,
    create_path=False,
) -> None:
  """Initialize the logger.

  This function creates and initializes a log file.
  Logging output is sent both to the file and standard output.
  If 'filename' == None, no file output is written
  To further write to this logger add in the script:

  import logging
  logger = logging.getLogger('root')
  logger.info(<info_string>)
  logger.warning(<warning_string>)
  logger.error(<error_string>)

  Parameters
  ----------
  filename : str
      name of output file
  path : str
      path to folder of output file
  log_level : int
      lowest log level to be reported. Options are:
        10=debug
        20=info
        30=warning
        40=error
        50=critical
  log_format : str
      format of the log
  overwrite : bool
    whether to overwrite existing log file
  create_path : bool
    whether to create path to log file if it does not exist
  """
  # logging.shutdown()
  # reload(logging)
  # logger = logging.getLogger('root')

  try:
    log_format_date = logging.Formatter(
      log_format, "%Y-%m-%dT%H:%M:%S")
  except ValueError:
    raise "Log format not allowed. Try using default instead."

  try:
    logging.basicConfig(level=log_level)
  except ValueError:
    raise "Log level not allowed. Try help(setlogger) for options."

  if logger.hasHandlers():
    logger.handlers.clear()

  # setup writing to console
  handler_stdout = logging.StreamHandler(sys.stdout)
  handler_stdout.setLevel(log_level)
  handler_stdout.setFormatter(log_format_date)
  logger.addHandler(handler_stdout)

  # setup writing to file
  if filename is not None:
    try:
      full_path = Path(path).joinpath(filename)
    except TypeError:
      raise (f"Cannot combine '{path}' and '{filename}' "
             "in one path, are both types and values valid?")

    if not os.path.exists(path):
      logger.info(f"Data output path '{path}' does not exist")
      if create_path:
        os.makedirs(path)
        logger.info("Data output path created")
      else:
        logger.error(
          " and 'create_path' option is disabled")
        raise FileNotFoundError
    if os.path.exists(full_path):
      logger.info(f"Data output full path {full_path} exists")
      if not overwrite:
        logger.error(" and 'overwrite' option is disabled")
        raise FileExistsError

    try:
      handler_file = logging.FileHandler(full_path, mode="w")
    except FileExistsError:
      raise (
        f"Error opening file path '{full_path}', does it exist?")
    handler_file.setLevel(log_level)
    handler_file.setFormatter(log_format_date)
    logger.addHandler(handler_file)

  # to avoid duplicates
  logger.propagate = False

  # To log uncaught exceptions when running as script
  def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
      sys.__excepthook__(exc_type, exc_value, exc_traceback)
      return
    logger.error("Uncaught exception", exc_info=(
      exc_type, exc_value, exc_traceback))

  sys.excepthook = handle_exception

  if filename is not None:
    logger.info(
      f"Logger initialized with log level {log_level} at {full_path}")
  else:
    logger.info("Logger initialized at console only")


class Balance():
  def __init__(self, tval, tflag, g, params):
    """
    params:
      lim: maximum allowed relative displacement in each step,
              default is 10%
      eps: maximum allowed discrepancy in every constraint,
              default is 1.0
      maxiter: maximum number of iterations.
              default is 100
      uscale: list whose length is the maximum value of tflag,
              should contain a positive float < 1.0 for each of those
              values
    kindex: list of constraint indices
    """

    self.param = params
    self.tval = tval
    self.tflag = tflag
    self.g = g

    if 'negative_proportional' not in self.param.__dict__.keys():
      setattr(self.param, 'negative_proportional', False)
      logger.info(
        'negative_proportional parameter not provided, assumed False')

    t = sp.csr_array(self.tval.reshape((len(self.tval), 1)))
    self.err = self.g.dot(t)

  @staticmethod
  def from_array(t, g, params, kindex=[]):
    logger.info('Started array system to balance format')

    if len(kindex) == 0:
      kindex = list(range(max(g.index)))
    k_index = pd.Series(index=range(len(kindex)), data=kindex)

    tval = np.array(t.val)
    tflag = np.array(t.flag)
    df = pd.DataFrame()
    df.index = list(set(g.row))
    df['constraint'] = df.index
    df['code'] = k_index.loc[df.constraint.values]
    df['position'] = list(range(len(df)))
    g['row'] = list(df.loc[list(g.row), 'position'])
    df.set_index('code', drop=True, inplace=True)

    gsparse = sp.coo_array(
      (np.array(g.val),
       (np.array(g.row),
        np.array(g.col)),
       ), shape=(len(df),
                 len(t))).tocsr()

    balance = Balance(tval, tflag, gsparse,
                      params)

    balance.k = df

    balance.t_index = pd.Index(t.index)

    logger.info('Finished array system to balance format')
    return balance

  def get_error_value(self, error='error', value='value'):
    """Only works with from_array instantiation."""

    t = sp.csr_array(self.tval.reshape((len(self.tval), 1)))
    self.err = self.g.dot(t)
    error_df = deepcopy(self.k.astype(int))
    error_df[error] = self.err.toarray()

    value_df = pd.DataFrame(index=self.t_index)
    value_df['position'] = range(len(value_df))
    value_df['position'] = value_df['position'].astype(int)
    value_df['flag'] = self.tflag.astype(int)
    value_df[value] = self.tval.astype(float)

    return error_df, value_df

  def nonzeros(self, positions: list[int] = [], axis: int = 0):
    """Only works with from_array instantiation."""

    if isinstance(positions, (int, np.int64)):
      positions = [positions]

    if axis == 0:
      nonzeros = self.g[positions, :].nonzero()
      const_pos = pd.Index(positions)[nonzeros[0]]
      value_pos = nonzeros[1]
      const_data = self.g[positions, :].data
    elif axis == 1:
      nonzeros = self.g[:, positions].nonzero()
      const_pos = nonzeros[0]
      value_pos = pd.Index(positions)[nonzeros[1]]
      const_data = self.g[:, positions].data

    df = pd.DataFrame()
    df['k_pos'] = const_pos
    df['t_pos'] = value_pos
    df['k_par'] = const_data
    df['t_par'] = self.tflag[value_pos]
    df['k_lab'] = self.k.index[const_pos]
    df['t_lab'] = self.t_index[value_pos]
    df['k_val'] = self.err.toarray()[const_pos]
    df['t_val'] = self.tval[value_pos]

    return df

  def run(self):
    """Performs balancing."""

    logger.info("Started balancing")
    t0 = time()

    t = sp.csr_array(self.tval.reshape((len(self.tval), 1)))
    tflag = self.tflag.reshape((len(self.tflag), 1))
    gT = self.g.T

    # format output
    n_iter = len(str(self.param.nmax)) - 1
    n_eps = len(str(self.param.eps)) - 1
    n_digit = - min(0, int(np.log10(self.param.eps))) + 1
    strformat = (
      f"%{n_iter}i %4.2f %4.2f %{n_eps}.{n_digit}f "
      f"%{n_eps}.{n_digit}f %{n_eps}.{n_digit}f")

    tu = np.zeros(t.shape)
    for (kpos, uval) in enumerate(self.param.uscale):
      tu = tu + uval * (tflag == kpos).astype(float)
    u = sp.csr_array(tu)
    del tflag, tu

    logger.info(
      f"\n\tNumber of variables: {t.shape[0]}\n"
      f"\tNumber of constraints: {self.g.shape[0]}\n"
      f"\tMaximum relative displacement: {self.param.maxstep}\n"
      f"\tStep displacement reduction: {self.param.stepreduce}\n"
      f"\tMaximum number of iterations: {self.param.nmax}\n"
      f"\tMaximum allowed error: {self.param.eps}\n"
      f"\tReliability scale: {self.param.uscale}")
    logger.info(
      f"{'Iteration'[:n_iter]}, {'reduce'[:5]}, {'displ'[:5]}, "
      f"median, mean, max")

    kpos = 0
    self.err = self.g.dot(t)
    ermedian = np.median(abs(self.err).toarray())
    ermean = np.mean(abs(self.err).toarray())
    ermax = np.max(abs(self.err).toarray())
    logger.info(strformat % (0, 0, 0, ermedian, ermean, ermax))
    step = self.param.stepreduce
    deltatmp = 0.0
    while ermax > self.param.eps:
      kpos = kpos + 1
      if self.param.negative_proportional:
        abs_t = t
      else:
        abs_t = abs(t)

      tmp = abs(self.g).dot(u.multiply(abs_t)).toarray()
      tmp = (1 * (tmp != 0)) / (tmp + 1 * (tmp == 0))
      a = (-self.g.dot(t)).multiply(sp.csr_array(tmp)).toarray()

      # This block handles the possibility of NaNs/Infty entries
      a = a * (1 * (np.isfinite(a)))
      where_are_NaNs = np.isnan(a)
      a[where_are_NaNs] = 0

      uga = u.multiply(abs_t).multiply(gT.dot(sp.csr_array(a)))

      # The following removes near-zeros from the variable check
      ugatmp = uga.toarray()[:, 0]
      tpos = np.where(ugatmp)[0]
      ugatmp = ugatmp[tpos]
      ttmp = t.toarray()[:, 0][tpos]
      tpos = np.where(abs(ttmp) > self.param.eps)[0]

      deltatmp = 0.0
      step = self.param.stepreduce
      # Displace check on variable
      if tpos.shape[0] > 0:
        ugatmp = ugatmp[tpos]
        ttmp = ttmp[tpos]
        deltatmp = max(abs(ugatmp / ttmp))
        if deltatmp > self.param.maxstep:
          step = step * self.param.maxstep / deltatmp

      t = (t + step * uga)
      self.err = self.g.dot(t)

      ermedian = np.median(abs(self.err).toarray())
      ermean = np.mean(abs(self.err).toarray())
      ermax = np.max(abs(self.err).toarray())
      # logger.info(
      #   strformat %
      #   (kpos, step, deltatmp, ermedian, ermean, ermax))
      if kpos >= self.param.nmax:
        break

    self.tval = t.toarray().reshape((len(self.tval)))
    logger.info(
      strformat %
      (kpos, step, deltatmp, ermedian, ermean, ermax))
    logger.info(f"Finished balancing in {(time() - t0):.2f} s")
