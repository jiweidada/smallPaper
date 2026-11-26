"""Import the isolated native snapshot; never resolve the shared training engine."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for name in (('local_deps', 'native') if sys.platform == 'win32' else ('native',)):
    sys.path.insert(0, str(ROOT / name))
if 'engine' in sys.modules:
    path = Path(sys.modules['engine'].__file__).resolve()
    if ROOT / 'native' not in path.parents:
        raise RuntimeError('A different engine is loaded. Use a fresh process.')
