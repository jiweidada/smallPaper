"""Fresh candidate EMA replay after exact12 training; no BASE or optimizer calls."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    with Path(p).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def verify_tools():
    protocol = read(HERE/'closeout_protocol.json')
    assert protocol['schema'] == 'native_stem_scratch12_closeout_v1'
    for name,digest in protocol['files'].items():
        assert sha(HERE/name) == digest, name
    return protocol


def same_live(record, pid_key, tick_key):
    proc = Path(f"/proc/{record[pid_key]}/stat")
    if not proc.exists(): return False
    fields = proc.read_text().split()
    return fields[21] == str(record[tick_key]) and fields[2] != 'Z'


def assert_trainers_stopped(screen):
    for name,pid,tick in [('controller_start.json','pid','start_ticks'),
                          ('candidate_start.json','child_pid','child_start_ticks')]:
        assert not same_live(read(screen/name),pid,tick), 'Owned training process is still live'


def bound_runtime(bundle):
    code = bundle.parent.parent/'code'
    assert code.is_dir()
    sys.path.insert(0,str(code))
    from runtime import verify_bundle
    receipt,plan,paths = verify_bundle(bundle)
    assert plan['arm'] == 'native_stem_scratch'
    return code,receipt,plan,paths


def worker(bundle):
    protocol = verify_tools()
    code,receipt,plan,paths = bound_runtime(bundle)
    screen,run = paths['screen_root'],paths['output']
    assert_trainers_stopped(screen)
    done = read(run/'training_completion.json')
    assert done['status'] == 'NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs'] == 12
    assert done['entered_epochs'] == list(range(12)) and done['identity'] == receipt['identity']
    assert read(screen/'controller_completion.json')['status'] == 'COMPLETE_FIXED12'
    assert read(screen/'candidate_completion.json')['exit_code'] == 0
    assert sha(run/'last.pth') == done['last_checkpoint_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    native_protocol = read(code/'training_protocol.json')
    uuid = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid == native_protocol['GPU_uuid'][str(plan['seed'])]
    from runtime import activate,register_model,resolved_config
    from results import load_endpoint,capture
    from stage_trace import StageTrace,TracedLoader,verify_trace
    from evaluator_binding import activate as bind_evaluator,same_result_evaluator
    primary,primary_hits,_ = load_endpoint(run/'endpoint_epoch12')
    assert sha(run/'endpoint_epoch12/manifest.json') == done['endpoint_manifest_sha256']
    assert sha(run/'stages_epoch12/manifest.json') == done['stage_manifest_sha256']
    verify_trace(run/'stages_epoch12',receipt['identity'])
    activate(paths['native_root'])
    bind_evaluator(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    register_model()
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver.det_engine import evaluate
    dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
    out = screen/'candidate_reload_v1'
    config = resolved_config(plan,dict(paths,output=out))
    write(out/'reload_config.yml',config)
    cfg = YAMLConfig(str(out/'reload_config.yml'))
    model = cfg.model.cuda()
    payload = torch.load(run/'last.pth',map_location='cpu',weights_only=False)
    assert payload['last_epoch'] == 11 and payload['native_stem_scratch12_identity'] == receipt['identity']
    expected = payload['ema']['module']
    model.load_state_dict(expected,strict=True)
    loaded = model.state_dict()
    assert loaded.keys() == expected.keys() and len(loaded) == read(run/'runtime_start.json')['initialization']['state_tensors']
    assert all(torch.equal(v.detach().cpu(),expected[n].cpu()) for n,v in loaded.items())
    count = sum(p.numel() for p in model.parameters())
    assert count == native_protocol['expected_parameters'] and model.geometry_arm == 'baseline'
    del payload,expected,loaded
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    criterion = cfg.criterion.cuda(); criterion.set_model(model); criterion.set_epoch(11)
    post,loader,evaluator = cfg.postprocessor.cuda(),cfg.val_dataloader,cfg.evaluator
    forward_calls = 0
    def observe(module,args):
        nonlocal forward_calls
        assert not module.training and not torch.is_grad_enabled()
        forward_calls += 1
    handle = model.register_forward_pre_hook(observe)
    trace = StageTrace(model,receipt['identity'],paths['val_ann']); model.eval()
    started = time.monotonic()
    try:
        with torch.no_grad():
            stats,obj = evaluate(model,criterion,post,TracedLoader(loader,trace),evaluator,
                torch.device('cuda:0'),test_only=False,output_dir=out,yolo_metrice=config['yolo_metrice'])
        assert obj is evaluator and forward_calls == len(loader) == protocol['candidate_forward_batches']
        trace.finish(out/'stages_epoch12')
        capture(evaluator.coco_eval['bbox'],stats,receipt['identity'],paths['val_ann'],out/'endpoint_epoch12',
                epoch=11,environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0)))
        replay,replay_hits,_ = load_endpoint(out/'endpoint_epoch12')
        same_result_evaluator(primary,replay)
        assert primary['identity'] == replay['identity'] == receipt['identity']
        error = max(abs(a-b) for a,b in zip(primary['native_stats'],replay['native_stats']))
        hit_equal = all(np.array_equal(primary_hits[n],replay_hits[n]) for n in primary_hits)
        fields,all_equal = {},True
        for iid in range(protocol['validation_images']):
            name = f'image_{iid:06d}.npz'
            with np.load(run/'stages_epoch12'/name,allow_pickle=False) as a,np.load(out/'stages_epoch12'/name,allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    assert a[key].shape == b[key].shape
                    all_equal = all_equal and np.array_equal(a[key],b[key])
                    fields[key] = max(fields.get(key,0),float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64)))))
        passed = error <= protocol['stats_fraction_tolerance'] and hit_equal and all_equal
        result = dict(status='PASS_SAVED_CANDIDATE_RELOAD' if passed else 'CANDIDATE_RELOAD_MISMATCH_NO_RETRY',
            seed=plan['seed'],arm=plan['arm'],identity=receipt['identity'],checkpoint_sha256=done['last_checkpoint_sha256'],
            original_training_completion_sha256=sha(run/'training_completion.json'),strict_loaded_tensors=read(run/'runtime_start.json')['initialization']['state_tensors'],
            parameters=count,all_parameters_frozen_like_native_EMA=True,
            original_stats=primary['native_stats'],reloaded_stats=replay['native_stats'],
            stats_max_abs_fraction_difference=error,stats_tolerance=protocol['stats_fraction_tolerance'],
            native_GT_hits_exact=hit_equal,all548_stage_arrays_exact=bool(all_equal),stage_max_abs_difference=fields,
            images=protocol['validation_images'],candidate_model_forwards=forward_calls,
            baseline_model_forwards=0,optimizer_updates=0,epochs_added=0,automatic_retry=False,
            evaluation_seconds_with_trace_overhead=time.monotonic()-started,
            verification_script_sha256=sha(__file__),closeout_protocol_sha256=sha(HERE/'closeout_protocol.json'),
            completed_remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'verification.json',result)
        assert passed, 'Saved candidate replay mismatch; preserve primary/replay results; no automatic retry'
        print(json.dumps(dict(status=result['status'],seed=plan['seed'],arm=plan['arm'],
                              all548_stage_arrays_exact=bool(all_equal))),flush=True)
    finally:
        handle.remove(); trace.close(); dist_utils.cleanup()


def supervise(bundle):
    protocol = verify_tools()
    _,receipt,plan,paths = bound_runtime(bundle)
    screen = paths['screen_root']
    assert read(screen/'controller_completion.json')['status'] == 'COMPLETE_FIXED12'
    assert_trainers_stopped(screen)
    out = screen/'candidate_reload_v1'; out.mkdir(exist_ok=False)
    started,timed_out = time.monotonic(),False
    command = [sys.executable,'-B',str(Path(__file__).resolve()),'--bundle',str(bundle),'--worker']
    with (out/'worker.log').open('xb') as log:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        write(out/'start.json',dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            controller_pid=os.getpid(),controller_start_ticks=Path('/proc/self/stat').read_text().split()[21],
            command=command,timeout_seconds=protocol['reload_timeout_seconds'],automatic_retry=False))
        while child.poll() is None:
            elapsed = time.monotonic()-started
            (out/'live.json').write_text(json.dumps(dict(pid=child.pid,elapsed_seconds=elapsed)))
            if elapsed >= protocol['reload_timeout_seconds']:
                timed_out = True; os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
                break
            time.sleep(5)
    write(out/'completion.json',dict(exit_code=child.returncode,timed_out=timed_out,
        elapsed_seconds=time.monotonic()-started,automatic_retry=False,baseline_model_forwards=0,epochs_added=0))
    assert child.returncode == 0 and not timed_out, 'Saved candidate verification failed; no auto retry'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle',required=True); p.add_argument('--worker',action='store_true')
    a = p.parse_args(); bundle = Path(a.bundle).resolve()
    worker(bundle) if a.worker else supervise(bundle)
