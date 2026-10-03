from setuptools import setup
import subprocess
import re


def get_version():
  try:
    cmd = ['git', 'describe', '--tags', '--abbrev=0']
    version = subprocess.check_output(cmd).decode().strip()
    version = re.sub('^v', '', version)
    cmd = ['git', 'status', '--porcelain']
    if subprocess.check_output(cmd).decode().strip():
      version += '.dev1'
  except subprocess.CalledProcessError:
    version = 'unknonwn'
  return version


if __name__ == "__main__":
  try:
    setup(
      version=get_version(),
      use_scm_version={"version_scheme": "no-guess-dev"})
  except:  # noqa
    print(
      "\n\nAn error occurred while building the project, "
      "please ensure you have the most updated version of setuptools, "
      "setuptools_scm and wheel with:\n"
      "   pip install -U setuptools setuptools_scm wheel\n\n"
    )
    raise
