"""Read-only native environment preflight. Does not start a training run.

The default path uses stdlib inspection only. --import-native additionally
imports the real framework in a CPU-only child process, if pycocoeval exists.
An import pass does not verify GPU training, evaluation equivalence, or AP.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

API_KWARGS = {'using_tiny_metrice', 'using_vtiny_metrice', 'max_dets'}
API_METHODS = {'evaluate_', 'create_common_coco_eval'}
YOLO_FUNCTIONS = {'get_yolo_det_metrice', 'get_yolo_seg_metrice', 'get_yolo_obb_metrice'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluator_contract(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'COCOeval')
    methods = {n.name:n for n in cls.body if isinstance(n, ast.FunctionDef)}
    constructor = methods['__init__'].args
    kwargs = {n.arg for n in constructor.args + constructor.kwonlyargs}
    return {'source_sha256':sha(path), 'constructor_extra_keywords':sorted(API_KWARGS & kwargs),
            'missing_constructor_keywords':sorted(API_KWARGS - kwargs),
            'missing_custom_methods':sorted(API_METHODS - set(methods)),
            'required_interface_present':API_KWARGS <= kwargs and API_METHODS <= set(methods)}


def package_fingerprint(root):
    root = Path(root)
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and p.suffix.lower() in ('.py','.pyd','.so','.dll')}


def verify_experiment(root):
    root = Path(root).resolve()
    manifest_name = '03_交付回执与SHA.json' if (root/'geometry.py').exists() else '02_交付回执与SHA.json'
    manifest = read_json(root/manifest_name)
    errors=[]
    for rel,digest in manifest['files_sha256'].items():
        path=root/rel
        if not path.is_file() or sha(path)!=digest:errors.append(rel)
    prepared=list((root/'prepared_local').glob('*/receipt.json'))
    for receipt_file in prepared:
        receipt=read_json(receipt_file);bundle=receipt_file.parent
        if sha(bundle/'resolved.yml')!=receipt['resolved_sha256']:errors.append(str(bundle/'resolved.yml'))
        if sha(bundle/'plan.json')!=receipt['identity']['plan_sha256']:errors.append(str(bundle/'plan.json'))
    return {'root':str(root),'delivery_manifest_sha256':sha(root/manifest_name),
            'prepared_bundles':len(prepared),'modified_or_missing':errors,
            'status':'PASS' if not errors and len(prepared)==8 else 'FAIL'}


def import_native(root, timeout):
    # Explicit code imports only; it never reads a config, dataset or checkpoint.
    code = r'''import json,sys,traceback
sys.path.insert(0,sys.argv[1])
result={'mode':'native-import-only','explicit_model_constructions':0,'optimizer_constructions':0,'training_started':False}
try:
    from engine.core import YAMLConfig
    from engine.solver import TASKS
    from pycocoeval.cocoeval import COCOeval
    from pycocoeval.yoloeval import get_yolo_det_metrice,get_yolo_seg_metrice,get_yolo_obb_metrice
    import inspect,torch
    result.update(status='PASS',tasks=sorted(TASKS),torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),
                  evaluator_signature=str(inspect.signature(COCOeval)))
    if result['cuda_initialized']:result['status']='FAIL_UNEXPECTED_CUDA_INITIALIZATION'
except Exception:
    result.update(status='FAIL',traceback=traceback.format_exc())
print('PREFLIGHT_RESULT='+json.dumps(result,ensure_ascii=False))
'''
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='';environment['PYTHONDONTWRITEBYTECODE']='1'
    try:
        result=subprocess.run([sys.executable,'-X','utf8','-B','-c',code,str(root)],
                              env=environment,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'status':'TIMEOUT','limit_seconds':timeout,'mode':'native-import-only'}
    marker=[line for line in result.stdout.splitlines() if line.startswith('PREFLIGHT_RESULT=')]
    parsed=json.loads(marker[-1].split('=',1)[1]) if marker else {'status':'NO_RESULT'}
    parsed.update(exit_code=result.returncode,stderr=result.stderr[-8000:],other_stdout='\n'.join(
        line for line in result.stdout.splitlines() if not line.startswith('PREFLIGHT_RESULT='))[-8000:])
    return parsed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',required=True)
    parser.add_argument('--experiment',action='append',default=[])
    parser.add_argument('--expected-evaluator-fingerprint')
    parser.add_argument('--import-native',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists():raise FileExistsError(destination)
    root=Path(args.native_root).resolve()
    report={'date':time.strftime('%Y-%m-%d %H:%M:%S'),'python':sys.executable,
            'native_root':str(root),'mode':'stdlib-preflight','training_started':False,
            'AP_equivalence_verified':False,'CUDA_training_verified':False,
            'experiments':[verify_experiment(p) for p in args.experiment]}
    modules=('pycocoeval','torch','torchvision','yaml','numpy','scipy','cv2','tensorboard','calflops','thop','timm','einops')
    report['top_level_modules']={name:importlib.util.find_spec(name) is not None for name in modules}
    spec=importlib.util.find_spec('pycocoeval')
    report['official_baseline_evaluator_identity']='UNVERIFIED'
    if spec is None:
        report['evaluator']={'status':'MISSING'}
    else:
        locations=list(spec.submodule_search_locations or [])
        if len(locations)!=1:
            report['evaluator']={'status':'AMBIGUOUS_PACKAGE_LOCATION','locations':locations}
        else:
            package=Path(locations[0]);fingerprint=package_fingerprint(package)
            ev={'package_root':str(package),'fingerprint':fingerprint}
            if (package/'cocoeval.py').exists():ev['contract']=evaluator_contract(package/'cocoeval.py')
            if (package/'yoloeval.py').exists():
                tree=ast.parse((package/'yoloeval.py').read_text(encoding='utf-8'))
                names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
                ev['yolo_missing_functions']=sorted(YOLO_FUNCTIONS-names)
            else:ev['yolo_missing_functions']=sorted(YOLO_FUNCTIONS)
            ev['status']='STATIC_API_PRESENT' if ev.get('contract',{}).get('required_interface_present') and not ev['yolo_missing_functions'] else 'API_INCOMPLETE'
            report['evaluator']=ev
            if args.expected_evaluator_fingerprint:
                matched=fingerprint==read_json(args.expected_evaluator_fingerprint)
                report['official_baseline_evaluator_identity']='MATCHES_SUPPLIED_FINGERPRINT' if matched else 'MISMATCH'
    if args.import_native and report['evaluator']['status']=='STATIC_API_PRESENT':
        report['native_import']=import_native(root,45)
    else:
        report['native_import']={'status':'NOT_RUN','reason':'Missing/incomplete evaluator API' if args.import_native else 'Not requested'}
    report['status']='BLOCKED' if report['evaluator']['status']!='STATIC_API_PRESENT' else 'DEPENDENCY_INSPECTED_NOT_TRAINING_VERIFIED'
    if any(x['status']!='PASS' for x in report['experiments']):report['status']='EXPERIMENT_BINDING_FAIL'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
