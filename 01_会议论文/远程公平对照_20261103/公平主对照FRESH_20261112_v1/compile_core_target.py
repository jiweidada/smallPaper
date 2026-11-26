from pathlib import Path
import json
import sys

volume=Path('/25085400120') if Path('/25085400120').is_mount() else Path('/24085403014')
root=volume/'mpsma_fresh_pair_20261112_v1'
arm='BASE_FRESH' if volume.name=='25085400120' else 'FULL_FRESH'
files=[root/'driver/stage_arm.py']+[root/arm/name for name in ['fair_helpers.py','train_runner.py','backup_protocol.py','launch_arm.py','preflight.py','source_loader.py']]
for p in files:compile(p.read_text(),str(p),'exec')
print(json.dumps({'status':'ALL_CORE_COMPILED_ON_ACTUAL_INTERPRETER','arm':arm,'python':sys.version,'files':[str(p) for p in files],'model_imports':0,'forwards':0}))
