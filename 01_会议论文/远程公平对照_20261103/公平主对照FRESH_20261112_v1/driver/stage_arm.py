"""Standard-library staging only. Never imports torch or starts a process.

prepare --root ORIGINAL_UNPACKED_FREEZE --data-root VERIFIED_CACHE --arm-dir NEW
        --arm BASE_FRESH|FULL_FRESH --gpu-uuid UUID --cuda-device0
        --authorization-file NEW_AUTH_JSON --environment-file CURRENT_ENV_JSON
finalize --arm-dir NEW
"""
from pathlib import Path
import argparse
import copy
import datetime as dt
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
PARENT_PROTOCOL='8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
PROPOSAL='fa4a0f72bec33b54218ba0948908f087f790c69b061108feff61e4bf3a6b33f7'
DRIVER_FILES=['fair_helpers.py','train_runner.py','backup_protocol.py','launch_arm.py','preflight.py']
INPUTS={
    'BASE':{'initial':'4fbd01639df2f19e9178bc536805649c5edb0c5221968fe73cc7f8351d642517','config':'1eed16436115ab62655c101e42f180e87fabb2f1a9e55d07b1d5241ce0070cdd'},
    'FULL':{'initial':'20624a8d3715ec0b31c14343e8a63f29ad90b695b4e6cd1296a7b1e2b971f598','config':'d1f5375a6385d135df894a555fcd53a8e7817bb1940c552a1ef7ff91347097aa'},
}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    assert not p.exists(),f'Refusing to overwrite {p}'
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def relative_checked(root,name):
    p=(root/name).resolve(strict=True)
    assert p.is_relative_to(root) and p.is_file(),name
    return p
def file_record(root,p):return {'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def copy_file(source,target):
    assert not target.exists(),str(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    assert sha(source)==sha(target)

def prepare(args):
    assert all([args.root,args.data_root,args.arm,args.gpu_uuid,args.authorization_file,args.environment_file,args.peer_environment_file])
    root=args.root.resolve(strict=True);data=args.data_root.resolve(strict=True)
    arm_dir=args.arm_dir.resolve()
    assert not arm_dir.exists(),'New isolated arm directory already exists'
    assert args.cuda_device==0 and args.arm in ('BASE_FRESH','FULL_FRESH')
    assert sha(root/'execution_protocol.json')==PARENT_PROTOCOL
    parent=read(root/'execution_protocol.json');assert len(parent['frozen_files'])==247
    auth=read(args.authorization_file)
    assert auth.get('authorized') is True or auth.get('training_authorized') is True,'New explicit authorization missing'
    for item in parent['frozen_files']:
        source=relative_checked(root,item['path'])
        assert sha(source)==item['sha256'],f'Original frozen file differs: {source}'
    env=read(args.environment_file);oldenv=read(root/'environment.current.json');peer=read(args.peer_environment_file)
    for name in ['torch','cuda','cudnn','packages']:assert env[name]==peer[name],f'Fresh paired runtimes differ: {name}'
    for name in ['torch','cuda','cudnn']:assert env[name]==oldenv[name],f'Runtime differs: {name}'
    package_differences=[]
    allowed_packaging_differences={'wheel':('0.48.0','0.46.3'),'tomli':('2.4.1','2.4.0'),'packaging':('26.3','26.0')}
    for name,version in oldenv['packages'].items():
        actual=env['packages'].get(name)
        if actual!=version:
            assert (version,actual)==allowed_packaging_differences.get(name),f'Scientific dependency differs: {name}'
            package_differences.append({'package':name,'historical':version,'fresh_pair':actual})
    dataset=read(root/'dataset_bundle_manifest.json');assert len(dataset['files'])==7021
    for item in dataset['files']:
        source=relative_checked(data,item['path'])
        assert sha(source)==item['sha256'],f'Dataset differs: {source}'
    original_arm=args.arm.removesuffix('_FRESH')
    assert sha(root/f'preflight/{original_arm}.initial.pt')==INPUTS[original_arm]['initial']
    assert sha(root/f'execution_configs/{original_arm}.yml')==INPUTS[original_arm]['config']
    arm_dir.mkdir(parents=True)
    (arm_dir/'inputs').mkdir()
    for item in parent['frozen_files']:
        copy_file(root/item['path'],arm_dir/'frozen_parent'/item['path'])
    copy_file(root/'execution_protocol.json',arm_dir/'frozen_parent/execution_protocol.json')
    for source in (arm_dir/'frozen_parent/source_snapshot').rglob('*'):
        if source.is_file():copy_file(source,arm_dir/source.relative_to(arm_dir/'frozen_parent'))
    for name in DRIVER_FILES:copy_file(HERE/name,arm_dir/name)
    for name in ['source_loader.py','source_manifest.json','dataset_bundle_manifest.json']:
        copy_file(root/name,arm_dir/name)
    copy_file(args.environment_file,arm_dir/'environment.current.json')
    copy_file(args.peer_environment_file,arm_dir/'environment.peer.json')
    copy_file(args.authorization_file,arm_dir/'user_authorization.json')
    copy_file(root/f'preflight/{original_arm}.initial.pt',arm_dir/'inputs/initial.pt')
    original=read(root/f'execution_configs/{original_arm}.yml')
    config=copy.deepcopy(original)
    relocations=[]
    for split in ['train','val']:
        for key in ['img_folder','ann_file']:
            old=Path(original[f'{split}_dataloader']['dataset'][key])
            suffix=old.relative_to('/tmp/mpsma_fair_20261103_data')
            config[f'{split}_dataloader']['dataset'][key]=str(data/suffix)
            relocations.append({'pointer':f'/{split}_dataloader/dataset/{key}','old':str(old),'new':str(data/suffix)})
    config['output_dir']=str(arm_dir/'run')
    relocations.append({'pointer':'/output_dir','old':original['output_dir'],'new':config['output_dir']})
    (arm_dir/'execution_configs').mkdir()
    cfgpath=arm_dir/f'execution_configs/{original_arm}.yml'
    write(cfgpath,config)
    for key in ['seed','use_amp','epoches','resume','tuning']:
        assert config[key]==original[key]
    write(arm_dir/'cache_verification.json',{'status':'ALL7021_DATASET_FILES_SHA256_MATCHED','cache_root':str(data),
          'verified_files':7021,'dataset_manifest_sha256':sha(root/'dataset_bundle_manifest.json'),'observed_utc':now()})
    write(arm_dir/'staged_inputs.json',{
        'status':'PREPARED_NOT_PREFLIGHTED_NOT_TRAINED','authorized':True,'arm':args.arm,
        'created_utc':now(),'original_root':str(root),'arm_dir':str(arm_dir),'data_root':str(data),
        'gpu_uuid':args.gpu_uuid,'cuda_visible_device':args.cuda_device,'parent_protocol_sha256':PARENT_PROTOCOL,
        'proposal_sha256':PROPOSAL,'configuration_sha256':sha(cfgpath),
        'original_configuration_sha256':INPUTS[original_arm]['config'],'initial_sha256':INPUTS[original_arm]['initial'],
        'authorization_sha256':sha(arm_dir/'user_authorization.json'),'original247_verified':True,
        'all7021_dataset_files_verified':True,'only_configuration_changes':relocations,
        'paired_runtime_package_versions_identical':True,'historical_packaging_tool_differences':package_differences,
        'new_training_runs':0,'model_forwards':0,'evaluator_runs':0,
    })
    print(json.dumps({'status':'PREPARED_NOT_PREFLIGHTED_NOT_TRAINED','arm':args.arm,'arm_dir':str(arm_dir),'configuration_sha256':sha(cfgpath)},ensure_ascii=False))

def finalize(args):
    arm_dir=args.arm_dir.resolve(strict=True)
    staged=read(arm_dir/'staged_inputs.json');pre=read(arm_dir/'inputs/preflight.json')
    assert not (arm_dir/'execution_protocol.json').exists()
    assert pre['status']=='PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE' and pre['optimizer_updates']==0
    assert pre['model_forwards']==1 and pre['full_validation_passes']==0 and pre['arm']==staged['arm']
    assert sha(arm_dir/'inputs/initial.pt')==staged['initial_sha256']==pre['initial_file_sha256']
    original_arm=staged['arm'].removesuffix('_FRESH')
    assert read(arm_dir/'inputs/first100.json')==read(arm_dir/f'frozen_parent/preflight/{original_arm}.first100.json')
    paths=[arm_dir/name for name in DRIVER_FILES+['source_loader.py','source_manifest.json','dataset_bundle_manifest.json',
              'environment.current.json','environment.peer.json','user_authorization.json','staged_inputs.json','cache_verification.json',
              'inputs/initial.pt','inputs/preflight.json','inputs/first100.json',f'execution_configs/{original_arm}.yml']]
    paths.extend(p for p in (arm_dir/'source_snapshot').rglob('*') if p.is_file())
    paths.extend(p for p in (arm_dir/'frozen_parent').rglob('*') if p.is_file())
    plan={
        'status':'FROZEN_AUTHORIZED_FRESH_READY_FOR_TRANSPORT_PROOF','authorized':True,'arm':staged['arm'],
        'epochs':320,'seed':0,'batch_size':4,'precision':'FP32','automatic_training_retry':False,
        'optimizer_update_limit_per_arm':517440,'evaluation_limit_per_arm':320,'hard_timeout_seconds':604800,
        'parent_protocol_sha256':PARENT_PROTOCOL,'proposal_sha256':PROPOSAL,
        'configuration_sha256':staged['configuration_sha256'],'original_configuration_sha256':staged['original_configuration_sha256'],
        'gpu_uuid':staged['gpu_uuid'],'cuda_visible_device':staged['cuda_visible_device'],
        'probe_files':{'initial.pt':sha(arm_dir/'inputs/initial.pt'),'preflight.json':sha(arm_dir/'inputs/preflight.json')},
        'frozen_files':[file_record(arm_dir,p) for p in sorted(set(paths))],
        'evaluated_and_next_roles':'Identical full checkpoint only when no epoch-end rollback; otherwise distinct complete state files and EMA hashes.',
        'authorized_arms':['BASE_FRESH','FULL_FRESH'],'authorization_sha256':staged['authorization_sha256'],
        'created_utc':now(),'primary_endpoint':'Actual epoch319 EMA before any following rollback',
        'runtime_note':'Both fresh arms have identical actual package versions. Three explicit packaging-tool versions differ from the historical inventory; scientific dependencies match.',
        'historical_packaging_tool_differences':staged['historical_packaging_tool_differences'],
        'secondary_endpoint':'Highest AP within320 recorded epochs; earliest ties and all12 metrics from same row',
    }
    write(arm_dir/'execution_protocol.json',plan)
    print(json.dumps({'status':plan['status'],'protocol_sha256':sha(arm_dir/'execution_protocol.json'),'arm':plan['arm']},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','finalize'])
    p.add_argument('--root',type=Path);p.add_argument('--data-root',type=Path);p.add_argument('--arm-dir',required=True,type=Path)
    p.add_argument('--arm',choices=['BASE_FRESH','FULL_FRESH']);p.add_argument('--gpu-uuid');p.add_argument('--cuda-device',type=int,default=0)
    p.add_argument('--authorization-file',type=Path);p.add_argument('--environment-file',type=Path);p.add_argument('--peer-environment-file',type=Path)
    a=p.parse_args();{'prepare':prepare,'finalize':finalize}[a.mode](a)

if __name__=='__main__':main()
