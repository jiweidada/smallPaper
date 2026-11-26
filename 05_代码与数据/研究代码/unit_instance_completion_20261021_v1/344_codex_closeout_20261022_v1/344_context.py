"""344 CPU closeout; exclusive evidence, old 342/343 outputs read-only."""
import os,sys,json,hashlib,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;SPEC=WORK.parents[1]/'持续寻找研究主线_20261020_v2'
S2=WORK/'342_codex_pretrain_20261022_v1';S3=WORK/'342_codex_stage3_20261022_v1';S343=WORK/'343_codex_stage4_3b_20261022_v1'
TRAIN=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train')
VAL=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')
NATIVE=WORK.parent/'dn_pre_frozen_diagnostic_20261020_v1/native'
for p in (WORK,S2,S3,S343,NATIVE,SPEC/'311_unit_instance_core_v2',SPEC/'320_geometry_calibrated_core_v2'):
    if str(p) not in sys.path:sys.path.append(str(p))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    assert name.startswith('344_')
    with open(HERE/name,'x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(name,value):
    assert name.startswith('344_') or name=='train344.py'
    with open(HERE/name,'x',encoding='utf-8',newline='\n') as f:f.write(value)
def load_geometry():
    p=HERE/'344_geometry/geom342.py';s=importlib.util.spec_from_file_location('geom342',p)
    m=importlib.util.module_from_spec(s);sys.modules['geom342']=m;s.loader.exec_module(m);return m
def sources():
    files=list(HERE.rglob('*.py'))+[SPEC/n for n in ('344_授权_1466裁定_EMA诊断_3b收尾.txt',
        '296_A_单位实例残余补全_具体实现提案.txt','317_A_单位解释残余补全_四臂有限验证方案_v3.txt','327_训练实施输入索引.json')]
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and p.endswith('.py') and str(WORK.parents[1]) in p and Path(p).is_file():files.append(Path(p))
    return {str(p):sha(p) for p in set(files)}
def development_ids():
    d=json.loads((SPEC/'327_训练实施输入索引.json').read_text(encoding='utf-8'));ids=d['sample_plan']['development_ids']
    assert len(ids)==3235 and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==d['sample_plan']['development_ids_SHA256']
    return ids
