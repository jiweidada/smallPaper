"""Freeze the reviewed one-shot diagnostic package, without model execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import json
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
AUTH = 'aa7698e43d3a7b09a24c0787c86610aa8ce87d8b8590c303ff72c375b1f4a107'
PLAN = 'fc8804407fcdbd9dd8210cd7f2f3c34ae503189c049659cbccf45a0023eff4fa'
STORAGE = '7dfb4c22fd467154713d9cf64451ecfe970358122b70b4197110b809b2ff4008'
CHECKPOINT = '33bb8012aa6b43a505b2f1a554189ed6abeeb4c39dc5cd6f20e83726fb922f65'
REMOTE_CODE = ['diagnostic_contract.py', 'remote_setup.py', 'diagnostic_worker.py',
               'diagnostic_controller.py', 'diagnostic_status.py']
LOCAL_CODE = ['backup_guard.py', 'diagnostic_backup_monitor.py', 'launch_setup.py',
              'launch_controller.py', 'initialize_remote_diagnostic.py', 'test_backup_guard.py',
              'test_diagnostic_contract.py']

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        __import__('os').fsync(f.fileno())

assert sha(HERE / 'diagnostic_user_authorization_20261105.json') == AUTH
assert sha(WORK / '基线骤降最小诊断方案_待授权_20261105.txt') == PLAN
assert sha(WORK / '诊断存储与中断保护补充_待授权_20261105.txt') == STORAGE
assert sha(WORK / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/best.pt') == CHECKPOINT
assert shutil.disk_usage(HERE).free >= 5 * 1024**3
for name in REMOTE_CODE + LOCAL_CODE:
    ast.parse((HERE / name).read_text(encoding='utf-8'), filename=name)
test = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'discover', '-s', str(HERE),
                       '-p', 'test_*.py', '-q'], capture_output=True, text=True, timeout=60)
assert test.returncode == 0, test.stdout + test.stderr
assert not (HERE / 'code_manifest.json').exists() and not (HERE / 'local_preflight.json').exists()
package = {'run_id': 'mpsma_diag_base_apdrop_20261105_v1', 'authorization_sha256': AUTH,
           'created_utc': datetime.now(timezone.utc).isoformat(),
           'files': [{'name': name, 'sha256': sha(HERE / name)} for name in REMOTE_CODE]}
write_new(HERE / 'code_manifest.json', package)
record = {'status': 'LOCAL_PREFLIGHT_PASSED_NO_MODEL_EXECUTION', 'created_utc': datetime.now(timezone.utc).isoformat(),
          'authorization_sha256': AUTH, 'plan_sha256': PLAN, 'storage_plan_sha256': STORAGE,
          'original_checkpoint_sha256': CHECKPOINT, 'code_manifest_sha256': sha(HERE / 'code_manifest.json'),
          'local_files': [{'name': name, 'sha256': sha(HERE / name)} for name in LOCAL_CODE],
          'free_bytes': shutil.disk_usage(HERE).free, 'synthetic_test_output': test.stdout + test.stderr,
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
write_new(HERE / 'local_preflight.json', record)
print(json.dumps({'status': record['status'], 'code_manifest_sha256': record['code_manifest_sha256'],
                  'tests': record['synthetic_test_output']}, ensure_ascii=False))
