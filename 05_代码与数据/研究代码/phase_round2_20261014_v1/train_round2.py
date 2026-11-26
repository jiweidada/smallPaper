"""Register the appended SOEP-lite module, then use unchanged train.py."""
import pathlib
import runpy
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
import engine.deim.soep_lite_round2  # noqa: F401

runpy.run_path(str(pathlib.Path.cwd()/'train.py'), run_name='__main__')
