"""343 CPU-only paths and exclusive-create evidence. No model forward entry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;SPEC=WORK.parents[1]/'持续寻找研究主线_20261020_v2'
S3=WORK/'342_codex_stage3_20261022_v1';S2=WORK/'342_codex_pretrain_20261022_v1'
NATIVE=WORK.parent/'dn_pre_frozen_diagnostic_20261020_v1'/'native'
DATA=WORK.parents[1]/'data'/'00_实验计划与证据索引'
EVAL_ROOT=DATA/'单一候选评价依赖补齐与构建核验_20261008_v1'/'isolated_runtime'
for p in (WORK,S3,S2,NATIVE,SPEC/'311_unit_instance_core_v2',SPEC/'320_geometry_calibrated_core_v2'):
    if str(p) not in sys.path:sys.path.append(str(p))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(name,obj):
    assert name.startswith('343_')
    with open(HERE/name,'x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def text(name,value):
    assert name.startswith('343_')
    with open(HERE/name,'x',encoding='utf-8',newline='\n') as f:f.write(value)

def load_geometry(path=None):
    path=S3/'342_geometry_stage3_v2'/'geom342.py' if path is None else Path(path)
    sp=importlib.util.spec_from_file_location('geom342',path)
    mod=importlib.util.module_from_spec(sp);sys.modules['geom342']=mod;sp.loader.exec_module(mod)
    return mod

def sources():
    paths=list(HERE.rglob('*.py'))+[SPEC/n for n in ('343_授权_阶段3放行_阶段4与阶段3b.txt',
        '342_总指令_修几何链路至开训前审查包.txt','317_A_单位解释残余补全_四臂有限验证方案_v3.txt',
        '296_A_单位实例残余补全_具体实现提案.txt')]
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and f.endswith('.py') and str(WORK.parents[1]) in f and Path(f).is_file():paths.append(Path(f))
    return {str(p):sha(p) for p in set(paths)}

def development_ids():
    idx=json.loads((SPEC/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
    ids=idx['sample_plan']['development_ids']
    digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    assert len(ids)==3235 and digest==idx['sample_plan']['development_ids_SHA256']
    return ids,digest
