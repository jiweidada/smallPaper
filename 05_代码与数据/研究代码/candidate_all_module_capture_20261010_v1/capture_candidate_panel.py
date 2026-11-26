"""Two fixed12-candidate validation forwards, only after original GPU jobs finish."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from raw_module_recorder import RawModuleRecorder, write


def prepare(args):
    sys.path.insert(0, str(Path(args.closeout_tools).resolve()))
    from verify_candidate_reload import bound_runtime, read, sha, verify_tools, same_live, assert_trainers_stopped
    verify_tools()
    protocol = read(args.protocol)
    assert protocol['schema'] == 'candidate_all_module_panel_protocol_v1'
    here = Path(__file__).resolve().parent
    assert all(sha(here / name) == digest for name, digest in protocol['files'].items())
    code, receipt, plan, paths = bound_runtime(Path(args.bundle).resolve())
    assert plan['arm'] == 'target_protect'
    assert receipt['identity']['val_annotation_sha256'] == protocol['annotation_sha256']
    root = paths['experiment_root']
    states = {}
    for arm in ('target_protect', 'random_protect'):
        screen = root / f"seed{plan['seed']}_{arm}"
        try:
            assert_trainers_stopped(screen)
            completion = read(root / f"post_training_seed{plan['seed']}_{arm}_v1/completion.json")
            assert completion['status'] == 'POST_TRAINING_RELOAD_AND_COLLECTION_FINISHED'
            assert completion['reload_exit_code'] == completion['collection_exit_code'] == 0
            suffix = '_random' if arm == 'random_protect' else ''
            launch = read(root / f"post_training_tools_v1/launch_seed{plan['seed']}{suffix}.json")
            assert not same_live(launch, 'pid', 'start_ticks')
            assert read(screen / 'candidate_reload_v1/verification.json')['status'] == 'PASS_SAVED_CANDIDATE_RELOAD'
            states[arm] = 'ALL_OWNED_ORIGINAL_JOBS_STOPPED_AND_RELOAD_VERIFIED'
        except (AssertionError, FileNotFoundError) as exc:
            states[arm] = 'WAITING_ORIGINAL_CLOSEOUT: ' + str(exc)
    ready = all(value == 'ALL_OWNED_ORIGINAL_JOBS_STOPPED_AND_RELOAD_VERIFIED' for value in states.values())
    result = dict(status='READY_FOR_CANDIDATE_PANEL' if ready else 'WAITING_ORIGINAL_GPU_JOBS',
        seed=plan['seed'], states=states, candidate_model_forwards=0, baseline_model_forwards=0,
        epochs_added=0, optimizer_updates=0, automatic_retry=False)
    return code, receipt, plan, paths, protocol, ready, result


def capture(args):
    code, receipt, plan, paths, protocol, ready, readiness = prepare(args)
    if args.check_ready:
        print(json.dumps(readiness), flush=True)
        return
    assert ready, json.dumps(readiness)
    from verify_candidate_reload import read, sha
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert uuid == read(code / 'training_protocol.json')['GPU_uuid'][str(plan['seed'])]
    destination = Path(args.destination).resolve()
    assert shutil.disk_usage(destination.parent).free >= 12 * 1024 ** 3, 'Insufficient disk for uncapped raw module capture'
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    from runtime import activate, register_model, resolved_config
    from evaluator_binding import activate as bind_evaluator
    activate(paths['native_root'])
    bind_evaluator(paths['evaluator_root'], receipt['identity']['evaluator_binding'])
    register_model()
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver.det_engine import move_samples_to_device, select_model_input_for_model
    dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=plan['seed'])
    config = resolved_config(plan, dict(paths, output=destination))
    write(destination / 'resolved.yml', config)
    cfg = YAMLConfig(str(destination / 'resolved.yml'))
    model = cfg.model.cuda()
    done = read(paths['output'] / 'training_completion.json')
    checkpoint = paths['output'] / 'last.pth'
    assert sha(checkpoint) == done['last_checkpoint_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['last_epoch'] == 11 and payload['scale_protection12_identity'] == receipt['identity']
    state = payload['ema']['module']
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    assert len(loaded) == len(state) == 794 and loaded.keys() == state.keys()
    assert all(torch.equal(value.detach().cpu(), state[name].cpu()) for name, value in loaded.items())
    assert sum(p.numel() for p in model.parameters()) == 10231997 and model.geometry_arm == 'baseline'
    del payload, state, loaded
    model.requires_grad_(False)
    model.eval()
    loader = cfg.val_dataloader
    assert len(loader) == 137 and loader.batch_size == 4
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    selected = {row['native_batch_index']: row for row in protocol['selected_batches']}
    reports, forwards = [], 0
    try:
        with torch.no_grad():
            for batch_index, (samples, targets) in enumerate(loader):
                ids = [int(target['image_id'].item()) for target in targets]
                assert ids == list(range(4 * batch_index, 4 * batch_index + len(ids)))
                if batch_index not in selected:
                    continue
                assert ids == selected[batch_index]['image_ids'] and len(ids) == 4
                samples = move_samples_to_device(samples, torch.device('cuda:0'), non_blocking=True)
                model_input = select_model_input_for_model(samples, model=model, key='rgb')
                assert isinstance(model_input, torch.Tensor) and tuple(model_input.shape) == (4, 3, 640, 640)
                recorder = RawModuleRecorder(model, destination / f'batch_{batch_index:03d}')
                try:
                    recorder.begin(dict(seed=plan['seed'], arm=plan['arm'], image_ids=ids,
                        native_batch_index=batch_index, identity=receipt['identity'],
                        checkpoint_sha256=done['last_checkpoint_sha256'], epoch_zero_based=11,
                        evaluated_weights='EMA', autocast_enabled=False, source_panel=selected[batch_index]),
                        model_input, targets)
                    # Same precision as bound native evaluate; training AMP is irrelevant here.
                    assert not torch.is_autocast_enabled()
                    output = model(model_input)
                    forwards += 1
                    for offset, iid in enumerate(ids):
                        with np.load(paths['output'] / 'stages_epoch12' / f'image_{iid:06d}.npz', allow_pickle=False) as reference:
                            assert np.array_equal(output['pred_boxes'][offset].detach().float().cpu().numpy(), reference['final_boxes'])
                            assert np.array_equal(output['pred_logits'][offset].detach().float().cpu().numpy(), reference['final_logits'])
                    record = recorder.finish()
                    reports.append(dict(batch_index=batch_index, image_ids=ids,
                        module_calls=record['total_module_calls'], called_modules=record['called_module_count'],
                        registered_modules=record['registered_module_count'],
                        unique_blobs=len(record['unique_physical_blobs']), saved_final_boxes_and_logits_exact=True))
                finally:
                    recorder.close()
                if len(reports) == len(selected):
                    break
        assert forwards == len(reports) == 2
        assert not any(p.grad is not None for p in model.parameters())
        result = dict(status='PASS_ALL_CALLED_MODULE_RAW_OUTPUTS_SELECTED_PANEL',
            seed=plan['seed'], checkpoint_sha256=done['last_checkpoint_sha256'],
            protocol_sha256=sha(args.protocol), identity=receipt['identity'], selected_batches=reports,
            strict_loaded_tensors=794, candidate_model_forwards=forwards,
            baseline_model_forwards=0, evaluator_calls=0, matcher_calls=0, optimizer_updates=0, epochs_added=0,
            native_evaluation_autocast_enabled=False, panel_images=8,
            saved_final_boxes_and_logits_exact=True, all_called_module_outputs_recorded=True,
            automatic_retry=False, population_efficacy_claim=False, causal_root_claim=False,
            elapsed_seconds=time.monotonic() - started,
            completed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(destination / 'verification.json', result)
        files = {p.relative_to(destination).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
                 for p in destination.rglob('*') if p.is_file()}
        write(destination / 'manifest.json', dict(files=files, total_bytes=sum(v['bytes'] for v in files.values()),
            capture_source_sha256=sha(__file__), recorder_source_sha256=sha(Path(__file__).with_name('raw_module_recorder.py'))))
        print(json.dumps(result), flush=True)
    finally:
        dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--closeout-tools', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--check-ready', action='store_true')
    capture(parser.parse_args())
