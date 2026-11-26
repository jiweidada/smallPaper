"""Read-only verification of the actual new trainer, resolved args, and data IDs."""
import hashlib
import json
import pathlib

phase=pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012')
execution=phase/'artifacts/round2_20261014_v1/execution_v2'
state=json.loads((execution/'queue_state.json').read_text())
assert state['active_job']=='c_mean_full96_seed0' and state['state']=='training_c_mean_full96_seed0'
args=json.loads((phase/'outputs/c_mean_full96_seed0/args.json').read_text())
cfg=args['yaml_cfg']
assert args['seed']==0 and cfg['epoches']==96 and cfg['flat_epoch']==46 and cfg['no_aug_epoch']==12
assert cfg['HGNetv2_Light_Phase']['p2_skip_mode']=='c_mean'
assert cfg['train_dataloader']['total_batch_size']==4 and args['use_amp'] and args['use_ema']
pid=state['child_pid'];proc=pathlib.Path('/proc',str(pid))
command=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
assert 'train.py' in command and '--seed 0' in command
annotation=pathlib.Path(cfg['val_dataloader']['dataset']['ann_file'])
images=len(json.loads(annotation.read_text())['images']);assert images==548
result={'trainer_pid':pid,'trainer_start_ticks':int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        'queue_pid':state['queue_pid'],'command':command,'seed':args['seed'],'epochs':cfg['epoches'],
        'flat_epoch':cfg['flat_epoch'],'no_aug_epoch':cfg['no_aug_epoch'],'warmup_iter':cfg['warmup_iter'],
        'augmentation_epochs':cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'],
        'mixup_epochs':cfg['train_dataloader']['collate_fn']['mixup_epochs'],
        'backbone':cfg['DEIM']['backbone'],'p2_skip_mode':cfg['HGNetv2_Light_Phase']['p2_skip_mode'],
        'batch':4,'AMP':args['use_amp'],'EMA':args['use_ema'],'val_images':images,
        'val_annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
        'source_sha256':{name:hashlib.sha256((phase/'code/engine/backbone'/name).read_bytes()).hexdigest() for name in ['hgnetv2_light.py','hgnetv2_light_control.py','hgnetv2_light_phase.py']},
        'status':'actual_new_training_contract_verified'}
print(json.dumps(result,indent=2))
