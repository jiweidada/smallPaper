"""Create a disabled, review-only runner candidate by reversible text patches.

The candidate cannot be launched and this builder never imports or executes it.
Explicit new authorization and additional validation are required before any
future enabled driver can be prepared.
"""
from hashlib import sha256
from pathlib import Path
import ast
import json

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'fair_runner.py'
OUT = HERE / 'guarded_fair_runner_candidate_v2.py.txt'
REPORT = HERE / 'guarded_runner_candidate_build_v2.json'
SOURCE_SHA = 'cf054375eb10d9522f6e638fd6e85d985adf80cec8ae359eacf9dd2af3f9bef5'
HEADER = '''# REVIEW-ONLY CANDIDATE: this file is intentionally disabled.
def _evidence_require_authorization():
    raise RuntimeError('Review-only candidate: new authorization and validation are required')

if __name__ == '__main__':
    _evidence_require_authorization()

'''


def digest(data):
    return sha256(data).hexdigest()


def main():
    assert not OUT.exists() and not REPORT.exists()
    source_bytes = SOURCE.read_bytes()
    assert digest(source_bytes) == SOURCE_SHA
    source = source_bytes.decode('utf-8-sig').replace('\r\n', '\n')
    patches = []

    def add(label, old, new):
        assert old != new
        patches.append((label, old, new))

    add('disable_preflight', 'def preflight(arm):\n',
        'def preflight(arm):\n    _evidence_require_authorization()\n')
    add('disable_training', 'def train_run(arm):\n',
        'def train_run(arm):\n    _evidence_require_authorization()\n')
    add('reject_persistent_workers',
        '        cfg,model,criterion,post,train,val,evaluator,lrfn=build(arm)\n'
        "        model.load_state_dict(torch.load(ROOT/'preflight'/f'{arm}.initial.pt',map_location='cpu',weights_only=True),strict=True)\n",
        '        cfg,model,criterion,post,train,val,evaluator,lrfn=build(arm)\n'
        "        assert train.persistent_workers is False and val.persistent_workers is False, 'Persistent workers lack a complete RNG snapshot'\n"
        "        model.load_state_dict(torch.load(ROOT/'preflight'/f'{arm}.initial.pt',map_location='cpu',weights_only=True),strict=True)\n")
    add('capture_context',
        "        expected=readj(ROOT/'preflight'/f'{arm}.first100.json')\n        seeds()\n\n        def state(loop_epoch):\n",
        "        expected=readj(ROOT/'preflight'/f'{arm}.first100.json')\n"
        '        import epoch_evidence_stager as _evidence_stager\n'
        '        import epoch_backup_barrier as _evidence_barrier\n'
        "        _evidence_context=readj(ROOT/'guarded_trial_context.json')\n"
        "        assert _evidence_context['status']=='AUTHORIZED_NEW_PAIR'\n"
        "        _evidence_run_id=_evidence_context['run_id']\n"
        "        _evidence_identity={'configuration_sha256':receipt['config_sha256'],\n"
        "          'environment_sha256':receipt['environment_sha256'],'runner_sha256':sha(Path(__file__)),\n"
        "          'dataset_manifest_sha256':sha(ROOT/'dataset_bundle_manifest.json'),\n"
        "          'initialization_sha256':receipt['initialization_sha256']}\n"
        '        seeds()\n\n'
        '        def _evidence_capture_rng():\n'
        '            name,values,pos,has_gauss,cached=np.random.get_state()\n'
        "            return {'python':random.getstate(),'numpy':{'name':name,'values':values.tolist(),\n"
        "              'pos':int(pos),'has_gauss':int(has_gauss),'cached_gaussian':float(cached)},\n"
        "              'torch_cpu':torch.get_rng_state().tolist(),\n"
        "              'torch_cuda_all':[item.tolist() for item in torch.cuda.get_rng_state_all()],\n"
        "              'train_loader_generator':train.generator.get_state().tolist(),\n"
        "              'train_sampler_generator':train.sampler.generator.get_state().tolist(),\n"
        "              'validation_loader_generator':val.generator.get_state().tolist()}\n\n"
        '        def _evidence_control():\n'
        "            return {'last_epoch':int(last_epoch),'ema_updates':int(ema.updates),\n"
        "              'ema_decay':float(ema.decay),'best_epoch':int(best_epoch),\n"
        "              'best_ap':None if best_epoch<0 else float(best_ap)}\n\n"
        '        def state(loop_epoch):\n')
    add('store_rng_in_checkpoint',
        "                    'configuration_sha256':receipt['config_sha256'],'environment_sha256':receipt['environment_sha256']}\n",
        "                    'configuration_sha256':receipt['config_sha256'],'environment_sha256':receipt['environment_sha256'],\n"
        "                    'rng_state':_evidence_capture_rng()}\n")
    add('initialize_epoch_rates',
        '            for epoch in range(320):\n                started=time.monotonic(); prepare_epoch(train,epoch)\n',
        '            for epoch in range(320):\n                _evidence_lr_rows=[]\n'
        '                started=time.monotonic(); prepare_epoch(train,epoch)\n')
    add('record_epoch_rates',
        "                    lrlog.write(f'{attempts},{epoch},{step},'+','.join(repr(g['lr']) for g in optimizer.param_groups)+'\\n')\n",
        "                    _evidence_lr_rows.append((attempts,epoch,step,[g['lr'] for g in optimizer.param_groups]))\n"
        "                    lrlog.write(f'{attempts},{epoch},{step},'+','.join(repr(g['lr']) for g in optimizer.param_groups)+'\\n')\n")
    add('stage_evaluated_state',
        '                # Persist actual terminal evaluated state before late-stage rollback.\n',
        '                _evidence_before_rng=_evidence_capture_rng()\n'
        '                _evidence_before_control=_evidence_control()\n'
        '                _evidence_stage=_evidence_stager.begin(\n'
        "                    ROOT,_evidence_run_id,arm,receipt['protocol_sha256'],epoch,_evidence_identity,metrics,\n"
        '                    pred_sha,array_sha,weight_sha,_evidence_before_rng,_evidence_before_control,\n'
        '                    _evidence_lr_rows,lambda stream: torch.save(state(epoch),stream))\n'
        '                _evidence_restored=False\n'
        '                # Persist actual terminal evaluated state before late-stage rollback.\n')
    add('mark_post_evaluation_restore',
        "                    restore(out/'best_stg1.pt',epoch,'late_epoch_without_new_global_best')\n",
        "                    restore(out/'best_stg1.pt',epoch,'late_epoch_without_new_global_best')\n"
        '                    _evidence_restored=True\n')
    add('wait_for_verified_backup',
        "                appendj(out/'epochs.jsonl',row);lrlog.flush()\n",
        "                appendj(out/'epochs.jsonl',row);lrlog.flush()\n"
        "                receipt.update({'phase':'waiting_local_backup','actual_loop_epoch':epoch,'updated_utc':now()});savej(out/'receipt.json',receipt)\n"
        '                _evidence_continuation_sha=tensor_hash(ema.module.state_dict())\n'
        '                _evidence_writer=(lambda stream: torch.save(state(epoch),stream)) if _evidence_restored else None\n'
        '                _evidence_stager.finish(_evidence_stage,_evidence_restored,_evidence_continuation_sha,\n'
        '                    _evidence_capture_rng(),_evidence_control(),_evidence_capture_rng,\n'
        '                    continuation_writer=_evidence_writer)\n'
        "                _evidence_ack=_evidence_barrier.await_ack(_evidence_stage['stage'],_evidence_run_id,arm,\n"
        "                    receipt['protocol_sha256'],epoch,timeout_seconds=600)\n"
        "                receipt['latest_backup_manifest_sha256']=_evidence_ack['manifest_sha256']\n")

    candidate = HEADER + source
    for label, old, new in patches:
        assert candidate.count(old) == 1, label
        candidate = candidate.replace(old, new, 1)
    ast.parse(candidate)
    recovered = candidate
    for label, old, new in reversed(patches):
        assert recovered.count(new) == 1, label
        recovered = recovered.replace(new, old, 1)
    assert recovered == HEADER + source
    tree = ast.parse(candidate)
    auth = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_evidence_require_authorization')
    assert len(auth.body) == 1 and isinstance(auth.body[0], ast.Raise)
    for name in ('preflight', 'train_run'):
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
        assert isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Call)
        assert isinstance(fn.body[0].value.func, ast.Name) and fn.body[0].value.func.id == '_evidence_require_authorization'
    data = candidate.encode('utf-8')
    with OUT.open('xb') as stream:
        stream.write(data)
    report = {'status': 'DISABLED_REVIEW_ONLY_CANDIDATE_CREATED', 'source_sha256': SOURCE_SHA,
              'candidate_sha256': digest(data), 'original_text_recovered_by_inverse_patches': True,
              'patch_labels': [x[0] for x in patches], 'preflight_and_training_unconditionally_disabled': True,
              'limits': ['The candidate is not executable as delivered and has not been imported or run.',
                         'Reversible text patches do not prove runtime equivalence or checkpoint semantics.',
                         'A new authorization, integration review, actual preflight and server/monitor validation remain required.']}
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'reversible_patches': len(patches)}))


if __name__ == '__main__':
    main()
