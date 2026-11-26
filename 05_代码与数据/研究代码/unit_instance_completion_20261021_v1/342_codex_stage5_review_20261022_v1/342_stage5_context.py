"""Phase 5: evidence-only review package, exclusively new outputs, no train entry."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;SPEC=WORK.parents[1]/'持续寻找研究主线_20261020_v2'
S2=WORK/'342_codex_pretrain_20261022_v1';S3=WORK/'342_codex_stage3_20261022_v1'
S343=WORK/'343_codex_stage4_3b_20261022_v1';S344=WORK/'344_codex_closeout_20261022_v1'
for p in (S344,WORK,S2,S3,S343):
    if str(p) not in sys.path:sys.path.append(str(p))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(name,value):
    assert name.startswith('342_')
    with open(HERE/name,'x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(name,value,root=None):
    assert name.startswith('342_')
    with open((HERE if root is None else root)/name,'x',encoding='utf-8',newline='\n') as f:f.write(value)
def sources():
    files=list(HERE.rglob('*.py'))+[SPEC/n for n in ('342_总指令_修几何链路至开训前审查包.txt',
        '296_A_单位实例残余补全_具体实现提案.txt','317_A_单位解释残余补全_四臂有限验证方案_v3.txt',
        '317v4_G臂kappa_c学习率修订.txt','327_训练实施输入索引.json','343_授权_阶段3放行_阶段4与阶段3b.txt',
        '344_授权_1466裁定_EMA诊断_3b收尾.txt')]
    return {str(p):sha(p) for p in files}

def evidence(path,pointer=''):
    path=Path(path);assert path.is_file()
    return {'path':str(path),'sha256':sha(path),'json_pointer':pointer}
