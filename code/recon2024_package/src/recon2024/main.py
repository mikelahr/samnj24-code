"""Main function, calls pipelines."""
from . import utilities, qcew, sut, bea, rpc
import pdb
import sys
import yaml
import pandas as pd
from time import time
from pathlib import Path
import logging
logger = logging.getLogger('root')


def pipeline(stage, base_config, steps=None):
  logger.info(f"Started stage {stage}")
  config_path = utilities.INTERNAL_PATH / stage / 'pipeline.yaml'
  logger.info(f"Reading stage configuration file: {config_path}")

  try:
    config_dict = utilities.load_stage_yaml(stage, base_config.__dict__['_raw'])
    config = utilities.DictToObject(config_dict)
    config.year = base_config.year
    if steps:                      # RECON2024: command-line step override
      config.steps = list(steps)
  except IOError:
    logger.info("Could not read stage config file, check path")

  logger.info("Adding base directory to stage directories")
  for key, val in config.dirs.__dict__.items():
    setattr(config.dirs, key, Path(base_config.dirs.base) / val)

  if config.steps is not None:
    if len(config.steps) > 0:
      logger.info("Executing steps")
      for step in config.steps:
        logger.info(f"Executing step '{step}'")
        globals()[stage].__dict__['steps'].__dict__[step](step, config)
    else:
      logger.info("No step found")
  else:
    logger.info("No step found")

  logger.info("Finished stage")
  return None


def main(stages=None, steps=None):
  """Run the stages listed in stages.yaml, or only those in `stages`
  (RECON2024: `python -m recon2024.main --stages bea sut`)."""
  if stages is None and len(sys.argv) > 1:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', nargs='+', default=None)
    ap.add_argument('--steps', nargs='+', default=None,
                    help='run only these steps (single stage)')
    ap.add_argument('--year', type=int, default=None)
    ap.add_argument('--prev-year', type=int, default=None)
    ap.add_argument('--process-dir', default=None,
                    help="process folder under RECON2024/ (default 'process')")
    ap.add_argument('--results-dir', default=None)
    args = ap.parse_args()
    stages, steps = args.stages, args.steps
    utilities.RUN_OVERRIDES.update({'year': args.year, 'prev_year': args.prev_year,
                                    'process_dir': args.process_dir,
                                    'results_dir': args.results_dir})

  config_dict = utilities.load_stages()
  if stages:
    config_dict['stages'] = {k: v for k, v in config_dict['stages'].items()
                             if k in stages}
  config = utilities.DictToObject(config_dict)
  config._raw = config_dict

  logpath = Path(config.dirs.base) / config.dirs.logs
  logfile = f"recon{config.year}_{utilities.datetime_now()}.log"
  utilities.set_logger(
    log_level=config.loglevel,
    filename=logfile,
    path=logpath)

  logger.info(f"Started RECON{config.year} update")
  logger.info("Main configuration file: 'main.yaml'")
  yaml_data = yaml.dump(config_dict, default_flow_style=False)
  logger.info(f"Configuration file content is:\n{yaml_data}")
  logger.info(
    f"Internal package directory is {utilities.INTERNAL_PATH}")

  if len(config.stages.__dict__) > 0:
    logger.info("Executing stages")
    for stage, str_ in config.stages.__dict__.items():
      logger.info(
        f"Executing stage {stage}: {str_}")
      if stage == 'rpc':
        from . import rpc_domestic
        rpc_domestic.install()
        rpc_domestic.install_taxes()
      pipeline(stage, config, steps)
    logger.info("Finished all stages, exiting")
  else:
    logger.info("No stage found")

  return None


if __name__ == "__main__":
  main()
