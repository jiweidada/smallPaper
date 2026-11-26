"""Static runner checks. No torch import, checkpoint load, data replay or fitting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback
from experiment import HERE, ARMS, check_code, read_json, validate_plan, resolved_config


def run():
    root = HERE.parent/'p2_candidate_seed1_full132_20261007_v1'
    checks = [check_code(root)]
    paths = {k: Path('/review-only')/k for k in
             ('train_images','train_ann','val_images','val_ann','common_init','output')}
    for arm in ARMS:
        for seed in (1,2):
            plan = read_json(HERE/'configs'/f'{arm}_seed{seed}.json')
            cfg = resolved_config(plan, paths)
            original = deepcopy(plan['native_baseline_config'])
            # Compare the complete recipe after removing only declared integration/path changes.
            a, b = deepcopy(cfg), original
            for k in ('__include__','config','model','criterion','resume','tuning','output_dir','device',
                      'use_swanlab','BudgetNormDEIM','BudgetNormCriterion'):
                a.pop(k,None);b.pop(k,None)
            for split in ('train','val'):
                for k in ('img_folder','ann_file'):
                    a[f'{split}_dataloader']['dataset'].pop(k)
                    b[f'{split}_dataloader']['dataset'].pop(k)
            assert a == b, (arm,seed,'native recipe changed')
            assert cfg['BudgetNormDEIM']['arm'] == cfg['BudgetNormCriterion']['arm'] == arm
            assert '__include__' not in cfg and cfg['resume'] is None
            broken = deepcopy(plan)
            broken['native_baseline_config']['optimizer']['lr'] *= 2
            try: validate_plan(broken)
            except ValueError: pass
            else: raise AssertionError('Unreviewed learning rate was accepted')
    checks.extend(['All six resolved configs preserve the complete native recipe outside declared substitutions',
                   'Unreviewed learning-rate edits rejected in all six plans',
                   'No includes, source config, trained resume or active logging service silently reused'])
    assert 'torch' not in sys.modules
    checks.append('Static checking does not import torch')
    return checks


if __name__ == '__main__':
    receipt = {'status':'RUNNING','optimizer_constructions':0,'detector_forwards':0}
    target = HERE/('runner_checks_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    try:
        receipt.update(checks=run(),status='PASS')
    except Exception:
        receipt.update(status='FAIL',traceback=traceback.format_exc())
        raise
    finally:
        with target.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
        print(json.dumps(receipt,ensure_ascii=False,indent=2))
