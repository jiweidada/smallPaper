from pathlib import Path
from hashlib import sha256
import json

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
assert sha256((root / 'execution_protocol.json').read_bytes()).hexdigest() == '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
path = root / 'recovery30715_preparation'
assert not path.exists()
path.mkdir(exist_ok=False)
print(json.dumps({'preparation_directory': str(path), 'training_started': False}))
