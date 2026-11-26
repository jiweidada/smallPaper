"""Bounded D/E pair. No retries, no other experiment ownership, no signal sends."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2)

def run(root, expected_manifest):
    jobs = root/'jobs'; code = root/'code'
    lock = (jobs/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = root/'bundle_manifest.json'
    def verify():
        assert sha(manifest) == expected_manifest, 'Bundle manifest changed'
        for row in json.loads(manifest.read_text())['files']:
            assert sha(root/row['path']) == row['sha256'], row['path']
    verify()
    settings = json.loads((root/'runtime.json').read_text())
    protocol = json.loads((code/'protocol_v1.json').read_text())
    assert protocol['launch_scope']['formal_order'] == ['D', 'E']
    assert json.loads((code/'cpu_validation_v1.json').read_text())['check_count'] == 32
    uuid, python = settings['GPU_uuid'], settings['python']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None)
    def child(name, command):
        verify()
        assert shutil.disk_usage(root).free > 70*2**30, 'Insufficient disk before stage'
        assert not subprocess.check_output(['nvidia-smi', '--id', uuid, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU busy; never stop another process'
        with (jobs/(name+'.log')).open('x') as log:
            proc = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            ticks = (Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
            save(jobs/(name+'_started.json'), {'pid':proc.pid,'start_ticks':ticks,'command':command,'time':time.time()})
            (jobs/'status.json').write_text(json.dumps({'status':'running','stage':name,'pid':proc.pid,'time':time.time()}))
            rc = proc.wait()
        save(jobs/(name+'_exit.json'), {'exit_code':rc, 'time':time.time()})
        if rc != 0:
            raise RuntimeError(f'{name} exit={rc}; no retry or later stage')
    common = ['--common-init', settings['init'], '--train-images', settings['train_images'],
              '--train-annotations', settings['train_ann'], '--val-images', settings['val_images'],
              '--val-annotations', settings['val_ann'], '--gpu-uuid', uuid]
    capacities = {}
    for arm in ['D', 'E']:
        output = root/'outputs'/('capacity_'+arm)
        child(arm+'_capacity', [python, '-u', str(code/'run_candidate.py'), 'capacity', '--config', str(code/f'configs/distribution_{arm}.yml'), *common, '--output', str(output)])
        cap = json.loads((output/'capacity.json').read_text())
        assert cap['status'] == 'passed_capacity_only'
        capacities[arm] = cap
    # Input hashes and true optimizer calls must agree; losses need not agree.
    for d, e in zip(capacities['D']['phases'], capacities['E']['phases']):
        assert (d['phase'],d['case']) == (e['phase'],e['case'])
        assert [s['input_and_targets_sha256'] for s in d['steps']] == [s['input_and_targets_sha256'] for s in e['steps']]
        assert all(s['optimizer_step_calls'] == 1 for s in d['steps']+e['steps'])
    save(jobs/'paired_capacity_acceptance.json', {'status':'both_passed_matched_inputs','time':time.time(),
         'arms':{a:{'allocated_peak_GiB':max(r['allocated_peak_GiB'] for r in cap['phases']),
                    'reserved_peak_GiB':max(r['reserved_peak_GiB'] for r in cap['phases']),
                    'receipt_sha256':sha(root/'outputs'/('capacity_'+a)/'capacity.json')} for a,cap in capacities.items()}})
    for arm in ['D', 'E']:
        output = root/'outputs'/(arm+'_96')
        cfg = code/f'configs/distribution_{arm}.yml'
        child(arm+'_train96', [python, '-u', str(code/'run_candidate.py'), 'train', '--config', str(cfg), *common,
             '--capacity-receipt', str(root/'outputs'/('capacity_'+arm)/'capacity.json'), '--output', str(output)])
        complete = json.loads((output/'complete.json').read_text())
        assert complete['status'] == 'candidate96_pending_independent_acceptance'
        epochs = []
        best = None
        with (output/'training_audit.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                assert row['batches'] == 808 and len(row['steps']) == 808
                assert [x['step'] for x in row['steps']] == list(range(808))
                assert sum(x['optimizer_step_calls'] for x in row['steps']) == row['optimizer_steps']
                assert row['consumed_originals'] == 6464 and row['unique_images'] == 6464
                assert sha(row['checkpoint']) == row['checkpoint_sha256']
                assert row['metrics'] == json.loads((output/'val_epochs'/f"{row['epoch']:03}"/'evaluation.json').read_text())['metrics']
                if best is None or row['metrics']['AP'] > best['metrics']['AP']:
                    best = {k:v for k,v in row.items() if k != 'steps'}
                epochs.append(row['epoch'])
        assert epochs == list(range(96))
        assert complete['best_epoch'] == best['epoch']
        selected = complete['selected_checkpoint']
        assert selected == best['checkpoint']
        eval_dir = root/'outputs'/(arm+'_selected_independent_eval')
        child(arm+'_acceptance_eval', [python, '-u', str(code/'evaluate.py'), '--config', str(cfg), '--checkpoint', selected,
              '--images', settings['val_images'], '--annotations', settings['val_ann'], '--output', str(eval_dir), '--device', 'cuda:0'])
        result = json.loads((eval_dir/'evaluation.json').read_text())
        assert result['images'] == 548 and result['forward_views'] == 3390
        assert all(abs(result['metrics'][k]-best['metrics'][k]) < 1e-8 for k in best['metrics'])
        save(jobs/(arm+'_accepted.json'), {'status':'96_epochs_and_separate_process_eval_verified','selected_epoch':best['epoch'],
             'checkpoint_sha256':sha(selected),'metrics':result['metrics'],'time':time.time(),'seed1_not_scientific_proof':True})
    # Compare complete paired input stream, not just first batch each epoch.
    with (root/'outputs/D_96/batch_audit.jsonl').open() as fd, (root/'outputs/E_96/batch_audit.jsonl').open() as fe:
        import itertools
        count = 0
        for ld,le in itertools.zip_longest(fd,fe):
            assert ld is not None and le is not None
            d,e = json.loads(ld),json.loads(le)
            for key in ['epoch','step','input_and_targets_sha256','image_ids','windows','annotation_ids']:
                assert d[key] == e[key], (count,key)
            count += 1
        assert count == 96*808
    save(jobs/'pair_complete.json', {'status':'D_E_pair_complete_pending_scientific_analysis','matched_batches':count,'time':time.time(),'no_further_arms_launched':True})
    (jobs/'status.json').write_text(json.dumps({'status':'pair_complete_pending_analysis','time':time.time()}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.manifest_sha256)
    except BaseException as error:
        (args.root/'jobs/failure.json').write_text(json.dumps({'status':'stopped_preserving_all_materials','error':str(error),
             'traceback':traceback.format_exc(),'time':time.time()}, indent=2))
        (args.root/'jobs/status.json').write_text(json.dumps({'status':'technical_failure_no_next_stage','error':str(error),'time':time.time()}))
        raise
