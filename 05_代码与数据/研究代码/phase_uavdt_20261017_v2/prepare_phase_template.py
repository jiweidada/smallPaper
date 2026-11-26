"""Isolated new phase; original solver/backbone/recipe files remain byte-identical."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import base64
import copy
import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

PAYLOAD=__PAYLOAD__
old=Path(CONFIG['old']);root=Path(CONFIG['root']);data=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_v1'
assert root.is_absolute() and root.parent==old.parent,'Remote experiment root must be an absolute POSIX path'
ready=json.loads((data/'data_ready.json').read_text());assert ready['state']=='complete_data_view_verified'
source=old/'artifacts/round2_20261014_v1/execution_v3/code';code=root/'code'
if root.exists():
    assert not (root/'plan.json').exists() and not (root/'final_acceptance.json').exists(),'Only incomplete, unfrozen CPU preparation may resume'
    assert not any((root/'outputs').iterdir()) and not any((root/'jobs').iterdir()),'Never replace any training state'
else:
    root.mkdir()
    for name in ['artifacts','outputs','logs','jobs','results']:(root/name).mkdir()
    shutil.copytree(source,code,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pth','dataset_visdrone','outputs','logs','.git'))
hashes={}
for name,value in PAYLOAD['files'].items():
    assert Path(name).name==name
    path=code/name;raw=base64.b64decode(value)
    if path.exists():assert path.read_bytes()==raw,'An existing unfrozen overlay differs: '+name
    else:
        with path.open('xb') as f:f.write(raw)
    hashes[name]=hashlib.sha256(raw).hexdigest()
os.chdir(code);sys.path.insert(0,str(code))
import numpy as np
import torch
import yaml
torch.set_num_threads(1);assert not torch.cuda.is_available()
from uavdt_protocol import UAVDTEvaluator,install_native_adapter,METRIC_NAMES
install_native_adapter()
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
configs={};config_root=code/'configs/experiments/uavdt';config_root.mkdir(exist_ok=True)
normalized={}
for arm in ['baseline','c','c_mean']:
    config=copy.deepcopy(load_config(str(source/'configs/experiments/phase'/(arm+'_full96.yml'))));config.pop('__include__',None)
    config.update(num_classes=1,remap_mscoco_category=False,use_swanlab=False,score_threshold=0.,cache_imgsz=None,ram_cache=False)
    for key,value in config.items():
        if key.startswith('HGNetv2') and isinstance(value,dict):value['pretrained']=False
    config['evaluator']={'type':'UAVDTEvaluator','ann_file':str(data/'annotations/val.json')}
    for split in ['train','val']:
        settings=config[split+'_dataloader'];settings['dataset']['img_folder']=str(data/'images')
        settings['dataset']['ann_file']=str(data/'annotations'/(split+'.json'))
        settings['total_batch_size']=4;settings['num_workers']=2
    assert config['epoches']==96 and config['flat_epoch']==46 and config['no_aug_epoch']==12
    assert config['train_dataloader']['collate_fn']['stop_epoch']==84
    config['output_dir']=str(root/'outputs/UNASSIGNED')
    path=config_root/(arm+'_full96.yml')
    if path.exists():assert yaml.safe_load(path.read_text())==config,'Existing effective config changed'
    else:
        with path.open('x') as f:yaml.safe_dump(config,f,sort_keys=False)
    configs[arm]=str(path)
    recipe=copy.deepcopy(config);recipe.pop('DEIM',None);recipe.pop('HGNetv2_Light',None);recipe.pop('HGNetv2_Light_Phase',None)
    normalized[arm]=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
assert len(set(normalized.values()))==1,'Nonstructural recipe differs between the three arms'
source_init=old/'artifacts/common_init_seed0.pth'
assert hashlib.sha256(source_init.read_bytes()).hexdigest()=='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
random.seed(0);np.random.seed(0);torch.manual_seed(0)
baseline=YAMLConfig(configs['baseline']).model
source_state=torch.load(source_init,map_location='cpu',weights_only=False)['model'];target=baseline.state_dict()
shared={k:v for k,v in source_state.items() if k in target and v.shape==target[k].shape}
missing,unexpected=baseline.load_state_dict(shared,strict=False)
assert not unexpected and all(any(t in k for t in ['score_head','class_embed']) for k in missing),missing
init=root/'artifacts/common_init_uavdt_seed0.pth'
torch.save({'model':baseline.state_dict(),'dataset':'UAVDT_v1','num_classes':1,
            'public_source_sha256':hashlib.sha256(source_init.read_bytes()).hexdigest(),'reset_class_parameter_keys':list(missing)},init)
del baseline,source_state,target,shared
for script in ['test_uavdt_protocol.py','prove_uavdt_stage2.py']:
    with (root/'logs'/('CPU_'+script+'.log')).open('xb') as log:
        p=subprocess.run([sys.executable,'-B',str(code/script)],cwd=code,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,'CPU protocol/stage2 test failed: '+script
from check_author_reference import run as reference_check
reference=reference_check(data)
with (root/'artifacts/author_reference_check.json').open('x') as f:json.dump(reference,f,indent=2)
state=torch.load(init,map_location='cpu',weights_only=False)['model'];phase_state=None;initial_predictions=None;accepted=[]
for arm in ['baseline','c','c_mean']:
    torch.manual_seed(123);cfg=YAMLConfig(configs[arm]);model=cfg.model
    missing,unexpected=model.load_state_dict(state,strict=False);assert not unexpected and all('p2_skip' in k for k in missing)
    assert isinstance(cfg.evaluator,UAVDTEvaluator)
    sample,target=cfg.val_dataloader.dataset[0];assert list(sample.shape)==[3,640,640]
    assert (target['labels']==0).all()
    branch=getattr(model.backbone,'p2_skip',None);count=sum(p.numel() for p in branch.parameters()) if branch is not None else 0
    assert count=={'baseline':0,'c':17440,'c_mean':17440}[arm]
    model.eval()
    with torch.no_grad():prediction=model(sample.unsqueeze(0))
    current={k:prediction[k].clone() for k in ['pred_logits','pred_boxes']}
    assert all(torch.isfinite(v).all() for v in current.values())
    if initial_predictions is None:initial_predictions=current
    else:
        for k in current:torch.testing.assert_close(current[k],initial_predictions[k],rtol=0,atol=0)
    if arm=='c':phase_state={k:v.clone() for k,v in model.state_dict().items()}
    if arm=='c_mean':assert phase_state.keys()==model.state_dict().keys() and all(torch.equal(phase_state[k],v) for k,v in model.state_dict().items())
    accepted.append({'arm':arm,'branch_parameters':count,'CPU640_forward_finite':True,'initial_prediction_equals_baseline':True})
    del model,cfg,prediction,current,sample,target
plan={'account':CONFIG['account'],'seed':0 if CONFIG['account']=='A' else 1,'dataset':'UAVDT_v1','jobs':['c','c_mean','baseline'],
      'epoches':96,'data_root':str(data),'data_metadata_sha256':ready['data_metadata_sha256'],
      'annotations_sha256':ready['annotations_sha256'],'configs':configs,'initialization':str(init),
      'initialization_sha256':hashlib.sha256(init.read_bytes()).hexdigest(),'metric_names':METRIC_NAMES,
      'primary_metric':'AP70','maximum_concurrent_trainings':'to_be_measured_1_2_3','retries':0,'test_used_for_selection':False}
with (root/'plan.json').open('x') as f:json.dump(plan,f,indent=2)
receipt={'state':'CPU_verified_pending_capacity_and_finalization','account':CONFIG['account'],'root':str(root),
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_models':accepted,
         'C_Cmean_initial_parameters_equal':True,'official_reference':reference,'CPU_protocol_tests':10,'native_stage2_integration_cases':4,
         'initialization_sha256':plan['initialization_sha256'],'data_metadata_sha256':ready['data_metadata_sha256'],
         'overlay_sha256':hashes,'source_code_unchanged':True,'training_started':False}
with (root/'preparation_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
