"""Create an isolated TinyPerson phase and run CPU acceptance only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import base64
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

PAYLOAD = __PAYLOAD__
account = PAYLOAD['account']
storage = Path('/25085400120' if account == 'A' else '/24085403014')
old_phase = storage / ('DEIM/experiments/phase_mechanism_20261012' if account == 'A' else 'experiments/phase_mechanism_20261012')
phase = old_phase.parent / 'phase_second_dataset_tinyperson_20261017_v1'
data = storage / 'datasets_verified_20261017/TinyPerson_v1'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''): value.update(block)
    return value.hexdigest()


if account == 'A':
    archive_path = storage / 'datasets_verified_20261017/TinyPerson_v1_verified_portable.tar'
    manifest_path = storage / 'datasets_verified_20261017/TinyPerson_v1_portable_file_manifest.json'
    assert sha(archive_path) == '4971e8026613de535df23e625f967639a4a745fbced47f41f719c546030f3d1b'
    assert sha(manifest_path) == '9cb751e1fe3680b5816d578dbc99859bdaa1f5fe6d6b9e4dfdffb511b5b4ae9c'
    data.mkdir(exist_ok=False)
    with tarfile.open(archive_path) as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile(): raise RuntimeError('Unsafe portable member')
            target = data / relative; assert target.resolve().is_relative_to(data.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as incoming, target.open('xb') as outgoing: shutil.copyfileobj(incoming, outgoing, 2*1024*1024)
    for item in json.loads(manifest_path.read_text()):
        assert sha(data/item['relative']) == item['sha256'], item['relative']
    with (data/'replica_ready.json').open('x') as handle: json.dump({'source':'B verified portable package','all_files_verified':True,'files':1541},handle)
ready = json.loads((data/'data_ready.json').read_text())
assert ready['status'] == 'data_view_ready_no_training_started'
for item in ready['annotation_files']:
    assert sha(data/'annotations'/Path(item['path']).name) == item['sha256']
for split, folder in [('train','train_erased'),('val','train_original'),('test','test_original')]:
    annotation = json.loads((data/'annotations'/(split+'.json')).read_text())
    assert len(annotation['images']) == {'train':586,'val':160,'test':786}[split]
    assert all((data/folder/image['file_name']).is_file() for image in annotation['images'])

phase.mkdir(exist_ok=False)
for directory in ['artifacts','outputs','logs','jobs','results']: (phase/directory).mkdir()
source_code = old_phase/'artifacts/round2_20261014_v1/execution_v3/code'
code = phase/'code'
shutil.copytree(source_code, code, symlinks=True,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pth','dataset_visdrone','outputs','logs','.git'))
overlay_hashes = {}
for name, content in PAYLOAD['files'].items():
    assert Path(name).name == name
    path = code/name
    if path.exists(): raise FileExistsError('New adapter conflicts with frozen file: '+name)
    raw = base64.b64decode(content)
    with path.open('xb') as handle: handle.write(raw)
    overlay_hashes[name] = sha(path)
assert overlay_hashes['official_tiny_cocoeval.py'] == '222b3173510e7a89bd03d077dce5d4a11e23ea6a7cd22afbbe930817b0886557'
os.chdir(code);sys.path.insert(0,str(code))
import numpy as np
import torch
import yaml
torch.set_num_threads(1)
assert not torch.cuda.is_available(), 'Preparation must be CPU-only'
from tiny_protocol import install_native_adapter, METRIC_NAMES, TinyPersonEvaluator
install_native_adapter()
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.solver.det_solver import DetSolver

configs = {};config_directory = code/'configs/experiments/tinyperson'
config_directory.mkdir(exist_ok=False)
for arm in ['baseline','c','c_mean','lowpass']:
    original = load_config(str(code/'configs/experiments/phase'/(arm+'_full96.yml')))
    config = copy.deepcopy(original);config.pop('__include__',None)
    config.update(num_classes=1,remap_mscoco_category=False,use_swanlab=False,score_threshold=0.0)
    for key,value in config.items():
        if key.startswith('HGNetv2') and isinstance(value,dict):value['pretrained']=False
    config['evaluator']={'type':'TinyPersonEvaluator','ann_file':str(data/'annotations/val.json'),
                         'official_module_path':str(code/'official_tiny_cocoeval.py')}
    for split,folder in [('train','train_erased'),('val','train_original')]:
        config[split+'_dataloader']['dataset']['img_folder']=str(data/folder)
        config[split+'_dataloader']['dataset']['ann_file']=str(data/'annotations'/(split+'.json'))
        config[split+'_dataloader']['total_batch_size']=4
        config[split+'_dataloader']['num_workers']=2
    assert config['epoches']==96 and config['train_dataloader']['collate_fn']['stop_epoch']==84
    config['output_dir']=str(phase/'outputs'/arm)
    path=config_directory/(arm+'_full96.yml')
    with path.open('x') as handle:yaml.safe_dump(config,handle,sort_keys=False)
    configs[arm]=str(path)

source_init = old_phase/'artifacts/common_init_seed0.pth'
assert sha(source_init)=='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
random.seed(0);np.random.seed(0);torch.manual_seed(0)
baseline=YAMLConfig(configs['baseline']).model
checkpoint=torch.load(source_init,map_location='cpu',weights_only=False)
original_state=checkpoint['model']
target_state=baseline.state_dict()
shared={key:value for key,value in original_state.items() if key in target_state and value.shape==target_state[key].shape}
missing,unexpected=baseline.load_state_dict(shared,strict=False)
assert not unexpected
assert all(any(token in key for token in ['score_head','class_embed']) for key in missing),missing
reset_class_keys=list(missing)
init=phase/'artifacts/common_init_tinyperson_seed0.pth'
torch.save({'model':baseline.state_dict(),'dataset':'TinyPerson_v1','num_classes':1,
            'original_common_init_sha256':sha(source_init),'reset_class_parameter_keys':missing},init)
del baseline,checkpoint,original_state,target_state

with (phase/'logs/cpu_protocol_tests.log').open('xb') as log:
    result=subprocess.run([sys.executable,'-B',str(code/'test_tiny_protocol.py')],cwd=code,
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
        stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Official protocol tests failed; see CPU log'
shared_state=torch.load(init,map_location='cpu',weights_only=False)['model']
acceptance=[];reference=None;phase_state=None
for arm in ['baseline','c','c_mean','lowpass']:
    torch.manual_seed(123);cfg=YAMLConfig(configs[arm]);model=cfg.model
    missing,unexpected=model.load_state_dict(shared_state,strict=False)
    assert not unexpected and all('p2_skip' in key for key in missing)
    evaluator=cfg.evaluator;assert isinstance(evaluator,TinyPersonEvaluator) and not hasattr(evaluator,'coco_eval')
    sample,target=cfg.val_dataloader.dataset[0]
    assert list(sample.shape)==[3,640,640]
    assert all(int(label)==0 for label in target['labels'])
    branch=getattr(model.backbone,'p2_skip',None)
    count=sum(parameter.numel() for parameter in branch.parameters()) if branch is not None else 0
    assert count=={'baseline':0,'c':17440,'c_mean':17440,'lowpass':17376}[arm],(arm,count)
    model.eval()
    with torch.no_grad():prediction=model(sample.unsqueeze(0))
    current={key:prediction[key].detach().clone() for key in ['pred_logits','pred_boxes']}
    if reference is None:reference=current
    else:
        for key in reference:torch.testing.assert_close(current[key],reference[key],rtol=0,atol=0)
    if arm=='c':phase_state={key:value.clone() for key,value in model.state_dict().items()}
    if arm=='c_mean':
        assert phase_state.keys()==model.state_dict().keys()
        assert all(torch.equal(phase_state[key],value) for key,value in model.state_dict().items())
    acceptance.append({'arm':arm,'branch_parameters':count,'640_CPU_forward_finite':True,
                       'initial_prediction_equals_baseline':True,'validation_labels_contiguous_zero':True})
    del model,cfg,prediction,current,evaluator,sample,target
solver=DetSolver.__new__(DetSolver);solver.writer=None;solver.output_dir=None
solver.evaluator=TinyPersonEvaluator(data/'annotations/val.json',code/'official_tiny_cocoeval.py')
solver.train_dataloader=SimpleNamespace(collate_fn=SimpleNamespace(stop_epoch=84))
stats={'tiny_eval_bbox':[.45]+[.9]*41}
best=solver.save_best_model(stats,{},93)
assert best['epoch']==93 and best['tiny_eval_bbox']==.45 and best['avg_metric']==.45
plan={'account':account,'seed':0 if account=='A' else 1,'dataset':'TinyPerson_v1','data_root':str(data),
      'jobs':['baseline','c','c_mean','lowpass'],'epoches':96,'maximum_concurrent_trainings':2,
      'configs':configs,'initialization':str(init),'initialization_sha256':sha(init),
      'metric_names':METRIC_NAMES,'primary_metric':'AP50_tiny','official_evaluator_sha256':overlay_hashes['official_tiny_cocoeval.py'],
      'gate':'Both original execution_v5 queues complete, including independent evaluation and analysis; require release_after_round2.json.',
      'retries':0,'test_used_for_selection':False,'source_code':str(source_code)}
with (phase/'plan.json').open('x') as handle:json.dump(plan,handle,indent=2)
receipt={'status':'cpu_verified_ready_for_deferred_gpu_preflight','account':account,'phase':str(phase),
         'data':str(data),'plan_sha256':sha(phase/'plan.json'),'initialization_sha256':sha(init),
         'reset_class_parameter_keys':reset_class_keys,'acceptance':acceptance,'official_protocol_unit_tests_passed':True,
         'native_best_selects_AP50_tiny':True,'model_code_unchanged':True,'overlay_hashes':overlay_hashes,
         'GPU_work_started':False,'training_started':False,
         'whole_image640_adaptation':'Controlled same-compute adaptation; official original-coordinate metrics. No tiling/high-resolution literature superiority claim.'}
with (phase/'preparation_receipt.json').open('x') as handle:json.dump(receipt,handle,indent=2)
print(json.dumps(receipt),flush=True)
