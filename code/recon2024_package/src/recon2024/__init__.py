"""Initializes recon2024 Python package."""

from importlib.metadata import version, PackageNotFoundError

try:
  __version__ = version(__name__)
except PackageNotFoundError:  # running from source without pip install
  __version__ = '2024.0.dev'

del version
