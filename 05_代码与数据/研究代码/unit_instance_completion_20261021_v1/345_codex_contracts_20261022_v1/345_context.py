"""345 CPU contracts; stop at any failing contract, no full training."""
import os,sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;SPEC=WORK.parents[1]/'持续寻找研究主线_20261020_v2'
S2=WORK/'342_codex_pretrain_20261022_v1';S3=WORK/'342_codex_stage3_20261022_v1';S343=WORK/'343_codex_stage4_3b_20261022_v1'
S344=WORK/'344_codex_closeout_20261022_v1';S5=WORK/'342_codex_stage5_review_20261022_v1'
for p in (S344,WORK,S2,S3,S343):
    if str(p) not in sys.path:sys.path.append(str(p))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(name,d):
    assert name.startswith(('345_','342_开训前审查包_v2'))
    with open(HERE/name,'x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(name,d):
    assert name.startswith(('345_','342_开训前审查包_v2','train345'))
    with open(HERE/name,'x',encoding='utf-8',newline='\n') as f:f.write(d)
def sources():
    paths=list(HERE.rglob('*.py'))+[SPEC/n for n in ('345_授权_继承合同重跑与正式入口验收.txt','296_A_单位实例残余补全_具体实现提案.txt','317_A_单位解释残余补全_四臂有限验证方案_v3.txt','317v4_G臂kappa_c学习率修订.txt')]
    paths += [S344/'train344.py',S344/'344_geometry/geom342.py',S2/'342_head.py',SPEC/'311_unit_instance_core_v2/core.py',SPEC/'320_geometry_calibrated_core_v2/gcal.py',WORK/'uic_losses.py']
    return {str(p):sha(p) for p in paths}
