"""Load the missing native Mosaic registration; original generator stays frozen."""
import pathlib,sys,runpy
ROOT=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'probe'),str(ROOT/'native')]
from engine.data.transforms.mosaic import Mosaic
runpy.run_path(str(ROOT/'probe/make_inputs.py'),run_name='__main__')
