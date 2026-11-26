"""Run this bounded phase once; record its actual process exit outside the result manifest."""
from pathlib import Path
from datetime import datetime,timezone
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
receipt=root.parent/'a15_sampling_center_v1_exit.json'
assert not receipt.exists()
with (root.parent/'a15_sampling_center_v1_stdout.log').open('x',encoding='utf-8') as log:
    process=subprocess.run([sys.executable,str(root/'a15_sampling_intervention.py')],
        cwd=str(root),stdout=log,stderr=subprocess.STDOUT)
receipt.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
    'exit_code':process.returncode,'status':'TERMINAL'},indent=2)+'\n',encoding='utf-8')
sys.exit(process.returncode)
