"""Run the recovered-lineage auditor with the existing local CPU tensor runtime."""

import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[3]
SITE = PAPER / '本地CPU核验_20261103/.runtime/site-packages'
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
require_paths = [SITE / 'torch', SITE / 'numpy', HERE / 'audit_recovered_ablation_lineage.py']
if not all(path.exists() for path in require_paths):
    raise RuntimeError('The existing CPU audit runtime or recovered auditor is missing')
handles = [os.add_dll_directory(str(path)) for path in
           (Path(sys.base_prefix), Path(sys.base_prefix) / 'DLLs', SITE / 'torch/lib') if path.is_dir()]
sys.path = [str(SITE), str(HERE), str(HERE.parent.parent), str(HERE.parents[2]),
            str(HERE.parents[2] / '断点续训_20261105')] + [
                entry for entry in sys.path if 'site-packages' not in entry.replace('\\', '/').split('/')]
sys.argv = [str(HERE / 'audit_recovered_ablation_lineage.py')] + sys.argv[1:]
runpy.run_path(str(HERE / 'audit_recovered_ablation_lineage.py'), run_name='__main__')
