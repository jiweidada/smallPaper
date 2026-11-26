"""Create only the explicitly scoped new recovery preparation directory."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

parent = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1').resolve(strict=True)
target = parent / 'recovery_no_msca_r29_20261108'
assert target.parent == parent
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
target.mkdir(exist_ok=False)
value = {'created_utc': datetime.now(timezone.utc).isoformat(), 'path': str(target),
         'scope': 'NO_MSCA continuation preparation from locally accepted round29', 'training_started': False}
(target / 'preparation_identity.json').write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps(value))
