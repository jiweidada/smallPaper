"""Prepare an isolated native ECDet source and reuse verified local assets."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

D = Path(__file__).resolve().parent
A42 = D.parent / 'A42_ECDet三组最小验证_20261031_v1'
PIN = 'd8752b75c9beaecfcdd0e5cd207bae3b6946ae23'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def put(p, text):
    assert not p.exists(), p
    p.write_text(text, encoding='utf-8', newline='\n')


def main():
    source = D / 'source' / ('EdgeCrafter-' + PIN) / 'ecdetseg'
    assert not source.exists(), 'Local preparation is one-shot; preserve existing working source'
    shutil.copytree(A42 / 'author_source' / ('EdgeCrafter-' + PIN) / 'ecdetseg', source)
    changes = []
    p = source / 'engine/edgecrafter/ecvit.py'
    before = sha(p)
    p.write_text('from __future__ import annotations\n' + p.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    changes.append({'file': str(p.relative_to(D)), 'before': before, 'after': sha(p), 'reason': 'Python3.9 annotation compatibility'})
    p = source / 'engine/core/_config.py'
    before = sha(p)
    old = 'from torch.utils.tensorboard import SummaryWriter'
    t = p.read_text(encoding='utf-8')
    assert t.count(old) == 1
    t = t.replace(old, 'try:\n    from torch.utils.tensorboard import SummaryWriter\nexcept ImportError:\n    SummaryWriter = None  # Unused by the JSON-logging pilot runner.')
    p.write_text(t, encoding='utf-8', newline='\n')
    changes.append({'file': str(p.relative_to(D)), 'before': before, 'after': sha(p), 'reason': 'Optional unused logger'})
    p = source / 'engine/edgecrafter/criterion.py'
    before = sha(p)
    old = '        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}'
    t = p.read_text(encoding='utf-8')
    assert t.count(old) == 1
    t = t.replace(old, "        nonfinite = [k for k, v in losses.items() if not torch.isfinite(v).all()]\n        if nonfinite:\n            raise FloatingPointError('A43 raw native loss nonfinite: ' + str(nonfinite))\n" + old)
    p.write_text(t, encoding='utf-8', newline='\n')
    changes.append({'file': str(p.relative_to(D)), 'before': before, 'after': sha(p), 'reason': 'Fail before native nan_to_num; finite native math unchanged'})
    # The native decoder is copied byte-for-byte; none of the A42 reader patches.
    assert sha(source / 'engine/edgecrafter/decoder.py') == sha(A42 / 'author_source' / ('EdgeCrafter-' + PIN) / 'ecdetseg/engine/edgecrafter/decoder.py')
    runtime = (A42 / 'runtime_common.py').read_text(encoding='utf-8')
    runtime = runtime.replace('from detail_reader import StemDetailReader\n', '')
    runtime = runtime.replace('SEED=20261031', 'SEED=20261101')
    runtime = runtime.replace("MODEL_COUNTER={'forwards':0}", "MODEL_COUNTER={'forwards':0,'teacher_forwards':0}")
    start, end = runtime.index('def create_components('), runtime.index('\ndef fp32_tree(')
    replacement = '''def create_components(arm='B', role='student'):
    assert arm in ['B','M','Q'] and role in ['student','teacher','cpu_check']
    seed_all()
    cfg=YAMLConfig(str(SOURCE/'configs/ecdet/ecdet_s.yml'))
    cfg.yaml_cfg.update(num_classes=10,remap_mscoco_category=False)
    cfg.yaml_cfg['ViTAdapter']['skip_load_backbone']=True
    model=cfg.model
    if role!='cpu_check':
        def count_forward(module,args):
            key='teacher_forwards' if role=='teacher' else 'forwards'
            MODEL_COUNTER[key]+=1
        model.register_forward_pre_hook(count_forward)
    initial=torch.load(R/'common_init_10class.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(initial['model'],strict=True)
    if arm!='B':
        torch.manual_seed(SEED+1)
        model.aux_projector=nn.Linear(48,192)
    return model,cfg.criterion,cfg.postprocessor

def build_teacher():
    model,criterion,_=create_components('B',role='teacher')
    obj=torch.load(R/'teacher_B_epoch12.pth',map_location='cpu',weights_only=True)
    assert obj['epoch']==12 and obj['arm']=='B'
    model.load_state_dict(obj['ema']['module'],strict=True)
    model.requires_grad_(False);model.eval()
    return model,criterion.matcher
'''
    runtime = runtime[:start] + replacement + runtime[end:]
    put(D / 'runtime_common.py', runtime)
    shutil.copyfile(A42 / 'evaluate.py', D / 'evaluate.py')
    for name in ['train_data_manifest.json', 'val_data_manifest.json']:
        shutil.copyfile(A42 / 'completed_results/B' / name, D / name)
    assert sha(A42 / 'common_init_10class.pth') == '2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6'
    shutil.copyfile(A42 / 'common_init_10class.pth', D / 'common_init_10class.pth')
    teacher = A42 / 'completed_results/B/arm_B/epoch12.pth'
    assert sha(teacher) == 'bbd5712fe42bc7b11e3b01d46e930f7baa3992dd76832a913b367a58626a6ad6'
    receipt = {'status': 'LOCAL_PREPARED_NOT_RUN', 'source_pin': PIN, 'changes': changes,
               'native_decoder_exact_original': True,
               'common_init_sha256': sha(D / 'common_init_10class.pth'),
               'teacher_local_path': str(teacher), 'teacher_sha256': sha(teacher),
               'teacher_selection': 'ema.module', 'new_model_forwards': 0, 'new_optimizer_updates': 0}
    put(D / 'local_source_preparation.json', json.dumps(receipt, indent=2) + '\n')
    for p in D.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))
    print(json.dumps({'status': receipt['status'], 'source_changes': len(changes), 'native_decoder_exact_original': True}))


if __name__ == '__main__':
    main()
