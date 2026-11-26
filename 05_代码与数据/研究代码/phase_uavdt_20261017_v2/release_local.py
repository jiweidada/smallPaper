"""Bind both independently accepted accounts and produce the one-time launcher."""
import ast
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'data/07_第二数据集_20261017/执行_v2'
proof={}
for account in ['A','B']:
    paths=sorted(DATA.glob('*_'+account+'_final_acceptance.json'))
    assert len(paths)==1,(account,paths)
    result=json.loads(paths[0].read_text(encoding='utf-8'))['results'][account]
    assert result['returncode']==0
    proof[account]=json.loads(result['stdout'])
assert len({p['initialization_sha256'] for p in proof.values()})==1
assert len({p['data_metadata_sha256'] for p in proof.values()})==1
assert len({p['model_engine_python_fingerprint'] for p in proof.values()})==1
text=(HERE/'launch_queue_template.py').read_text(encoding='utf-8').replace('__PROOF__',repr(proof))
ast.parse(text)
path=DATA/'launch_six_after_dual_acceptance_v2.py'
with path.open('x',encoding='utf-8') as f:f.write(text)
with (DATA/'dual_final_acceptance.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
print(json.dumps({'launcher':str(path),'maximum_concurrent_trainings':{a:p['maximum_concurrent_trainings'] for a,p in proof.items()}}))
