"""Preserve the empty first-attempt staging and error before using Python3.10."""
from pathlib import Path
from hashlib import sha256
import json
import os

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
prep = root / 'recovery30715_preparation'
staging = Path('/tmp/mpsma_module_ablation_20261107_data_restore30715')
archived = Path('/tmp/mpsma_module_ablation_20261107_data_restore30715_attempt1_failed')
assert not Path('/tmp/mpsma_module_ablation_20261107_data').exists()
assert staging.is_dir() and not any(staging.iterdir()) and not archived.exists()
payload = (prep / 'data_status.json').read_bytes()
assert json.loads(payload)['status'] == 'RESTORATION_FAILED_STAGING_PRESERVED'
with (prep / 'data_status_attempt1_failed.json').open('xb') as handle:
    handle.write(payload)
os.rename(staging, archived)
print(json.dumps({'empty_failed_staging_preserved': str(archived),
    'failed_status_sha256': sha256(payload).hexdigest(), 'cache_created': False,
    'next_interpreter': '/25085400120/envs/deim/bin/python', 'training_started': False}))
