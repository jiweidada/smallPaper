"""Build secret-free per-account preflight/preparation scripts from raw evidence."""
import ast
import base64
import hashlib
import json
import pathlib
import sys

here = pathlib.Path(__file__).resolve().parent
data = here.parents[2]/'英文小论文/data/06_第二轮_20261014'
destination = data/'恢复执行_20261016_v5'
controller = (here/'resume_output_fix_v5_draft.py').read_bytes()
template = (here/'repair_v5_template.py').read_text(encoding='utf-8')
marker = 'PAYLOAD = None  # Generated from the previously verified local deployment evidence.'
assert template.count(marker)==1
tag=sys.argv[1] if len(sys.argv)>1 else ''
assert not tag or tag.replace('_','').isalnum()
for account, port in [('A',30512),('B',30059)]:
    tree=ast.parse((data/'恢复执行_20261015_v4'/('prepare_'+account+'.py')).read_text(encoding='utf-8'))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
    assert isinstance(assignment.value,ast.Call) and isinstance(assignment.value.args[0],ast.Constant)
    old=json.loads(assignment.value.args[0].value)
    preflight=json.loads((data/'恢复执行_20261015_v4'/('preflight_'+str(port)+'.json')).read_text(encoding='utf-8'))
    phase='/25085400120/DEIM/experiments/phase_mechanism_20261012' if account=='A' else '/24085403014/experiments/phase_mechanism_20261012'
    payload={'account':account,'phase':phase,'completed_jobs':preflight['completed_jobs'],
             'common_init_sha256':preflight['common_init_sha256'],'resume_jobs':preflight['resume_jobs'],
             'code_hashes':preflight['code_hashes'],'protected_source_sha256':old['protected_source_sha256'],
             'controller_base64':base64.b64encode(controller).decode(),
             'controller_sha256':hashlib.sha256(controller).hexdigest()}
    payload['attempt_tag']=tag or 'initial'
    for action in ['preflight','prepare']:
        payload['action']=action
        source=template.replace(marker,'PAYLOAD = json.loads('+repr(json.dumps(payload,ensure_ascii=False))+')')
        ast.parse(source)
        with (destination/(action+'_'+account+('_'+tag if tag else '')+'.py')).open('x',encoding='utf-8') as handle:
            handle.write(source)
    print(account,'protected files',len(payload['protected_source_sha256']),'syntax verified')
