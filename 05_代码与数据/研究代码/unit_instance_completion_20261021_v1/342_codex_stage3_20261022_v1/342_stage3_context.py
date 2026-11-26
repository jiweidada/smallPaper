"""Immutable paths and receipts for stage 3. No training imports or entry point."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
PREV=WORK/'342_codex_pretrain_20261022_v1';SPEC=WORK.parents[1]/'持续寻找研究主线_20261020_v2'
VAL=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')
for p in (WORK,PREV,SPEC/'311_unit_instance_core_v2',
          WORK.parent/'dn_pre_frozen_diagnostic_20261020_v1'/'native'):
    if str(p) not in sys.path:sys.path.append(str(p))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def json_out(name,obj):
    assert name.startswith('342_')
    with open(HERE/name,'x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def text_out(name,text):
    assert name.startswith('342_')
    with open(HERE/name,'x',encoding='utf-8',newline='\n') as f:f.write(text)

def sources():
    paths=list(HERE.glob('*.py'))+[WORK/n for n in ('stage1_cache_v3.py','stage1_cache_v4.py','build_340.py','build_341_val.py','uic_m0.py')]
    paths+=[SPEC/n for n in ('296_A_单位实例残余补全_具体实现提案.txt','317_A_单位解释残余补全_四臂有限验证方案_v3.txt','342_总指令_修几何链路至开训前审查包.txt')]
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and f.endswith('.py') and str(WORK.parents[1]) in f and Path(f).is_file():paths.append(Path(f))
    return {str(p):sha(p) for p in set(paths)}

def metadata():
    idx=json.loads((SPEC/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
    annpath=VAL/'annotations'/'val.json';h=sha(annpath)
    assert h==next(a['expected_SHA256_from_existing_audit'] for a in idx['annotations'] if a['split']=='val')
    ann=json.loads(annpath.read_text(encoding='utf-8'))
    assert len(ann['images'])==548
    by={}
    for a in ann['annotations']:by.setdefault(a['image_id'],[]).append(a)
    return ann['images'],by,h
