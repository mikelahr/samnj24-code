"""Run one pipeline step outside main(): python scripts/run_step.py <stage> <step> [<step> ...]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from recon2024 import utilities
import importlib
utilities.set_logger()
stage = sys.argv[1]
steps = importlib.import_module(f'recon2024.{stage}.steps')
cfg = utilities.get_config(stage)
for step in sys.argv[2:]:
    getattr(steps, step)(step, cfg)
