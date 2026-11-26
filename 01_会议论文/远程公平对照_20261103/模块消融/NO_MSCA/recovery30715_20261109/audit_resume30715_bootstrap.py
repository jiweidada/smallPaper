"""Enter only the existing local CPU audit runtime after checking the sealed binding."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import sys


HERE = Path(__file__).resolve().parent
AUDITOR = HERE / 'audit_resume30715_lineage.py'
BINDING = HERE / 'segment30715_audit_binding.json'
PAPER = HERE.parents[3]
SITE = PAPER / '本地CPU核验_20261103/.runtime/site-packages'
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


locations = [index for index, value in enumerate(sys.argv) if value == '--binding-sha256']
require(len(locations) == 1 and locations[0] + 1 < len(sys.argv), 'Supply one externally sealed --binding-sha256')
expected = sys.argv[locations[0] + 1]
require(re.fullmatch(r'[0-9a-f]{64}', expected) and BINDING.is_file() and digest(BINDING) == expected,
        'Missing or changed controlling-agent binding; no audit was entered')
binding = json.loads(BINDING.read_text(encoding='utf-8'))
require(binding.get('schema') == 'second_no_msca_recovery_audit_binding_v1'
        and binding.get('bootstrap_sha256') == digest(Path(__file__))
        and binding.get('auditor_sha256') == digest(AUDITOR), 'Binding is not sealed to this exact CPU bootstrap/auditor')
require((SITE / 'torch').is_dir() and (SITE / 'numpy').is_dir(), 'Existing CPU tensor audit runtime is missing')
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  PYTHONDONTWRITEBYTECODE='1', ATEN_CPU_CAPABILITY='avx2')
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
handles = [os.add_dll_directory(str(path)) for path in
           (Path(sys.base_prefix), Path(sys.base_prefix) / 'DLLs', SITE / 'torch/lib') if path.is_dir()]
sys.path = [str(SITE), str(HERE), str(HERE.parent.parent), str(HERE.parents[2]),
            str(HERE.parents[2] / '断点续训_20261105')] + [
                entry for entry in sys.path if 'site-packages' not in entry.replace('\\', '/').split('/')]
import torch
require(Path(torch.__file__).resolve().is_relative_to(SITE.resolve()) and torch.version.cuda is None,
        'Unexpected tensor runtime; use only the pre-existing local CPU build')
torch.set_num_threads(1)
sys.argv = [str(AUDITOR)] + sys.argv[1:]
runpy.run_path(str(AUDITOR), run_name='__main__')
