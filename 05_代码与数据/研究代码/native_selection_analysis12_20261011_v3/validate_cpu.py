"""New six-record binding and inclusive area-boundary fixtures; no efficacy claim."""
from copy import deepcopy
from pathlib import Path
import argparse, json, sys
import numpy as np
import analyze as a


def validate(output):
    p, helper, parts, aggregate = a.prepared()
    jobs = [dict(arm=arm, kind=kind, seed=s) for arm, kind in a.KINDS.items() for s in (1, 2)]
    a.validate_job_set(jobs); results = {'six_correct_bindings': True}

    def rejected(name, operation):
        try: operation()
        except AssertionError: results[name] = True
        else: raise AssertionError('Unsafe acceptance: ' + name)

    rejected('reject_four_record_partial', lambda: a.validate_job_set(jobs[:4]))
    bad = deepcopy(jobs); bad[-1] = deepcopy(bad[-2])
    rejected('reject_duplicate_seed', lambda: a.validate_job_set(bad))
    bad_kind = deepcopy(jobs); bad_kind[-1]['kind'] = 'full'
    rejected('reject_wrong_reader_kind', lambda: a.validate_job_set(bad_kind))
    boolean_seed = deepcopy(jobs); boolean_seed[0]['seed'] = True
    rejected('reject_boolean_seed', lambda: a.validate_job_set(boolean_seed))
    identity = dict(arm='highfreq_rgb', seed=2)
    v = dict(status='PASS_SELECTION_DIAGNOSTIC_EXACT_SAVED_EMA_REPLAY', identity=identity, arm='highfreq_rgb', seed=2,
        checkpoint_sha256='fixture-checkpoint', original_training_completion_sha256='fixture-completion',
        diagnostic_protocol_sha256=p['diagnostic_protocol_sha256'], verification_script_sha256=p['diagnostic_replay_sha256'],
        stats_max_abs_fraction_difference=0, stats_tolerance=0, original_stats=[0.] * 12, reloaded_stats=[0.] * 12,
        native_GT_hits_exact=True, all548_stage_arrays_exact=True, stage_max_abs_difference={'fixture': 0},
        candidate_model_forwards=137, actual_selector_calls=137, existing_selected_bbox_head_calls=137,
        images=548, selection_images=548, additional_bbox_head_or_topk_calls=0, baseline_model_forwards=0,
        optimizer_updates=0, epochs_added=0, automatic_retry=False, all_parameters_frozen_like_native_EMA=True)
    done = dict(exit_code=0, timed_out=False, automatic_retry=False, baseline_model_forwards=0, epochs_added=0)
    check = lambda x, d=done: a.validate_replay(x, d, identity, 'fixture-checkpoint', 'fixture-completion', p)
    check(v); results['exact_v2_replay_binding'] = True
    for name, key, value in (
        ('reject_old_diagnostic_protocol', 'diagnostic_protocol_sha256', 'v1-fixture'),
        ('reject_wrong_checkpoint', 'checkpoint_sha256', 'other-fixture'),
        ('reject_metric_mismatch', 'stats_max_abs_fraction_difference', 1e-12),
        ('reject_stage_mismatch', 'all548_stage_arrays_exact', False),
        ('reject_extra_head_call', 'additional_bbox_head_or_topk_calls', 1),
        ('reject_baseline_forward', 'baseline_model_forwards', 1)):
        bad = deepcopy(v); bad[key] = value
        rejected(name, lambda bad=bad: check(bad))
    bad_done = dict(done, timed_out=True)
    rejected('reject_timeout_even_if_exit_zero', lambda: check(v, bad_done))
    annotation = dict(images=[dict(id=0, width=640, height=640)], annotations=[
        dict(id=i, image_id=0, bbox=[0, 0, width, 2]) for i, width in enumerate((3, 5, 9, 20))])
    values = dict(annotation_id=np.arange(4), original_area=np.array([1023., 1024., 1024.1, 2048.]))
    masks = a.group_masks(values, annotation)
    assert masks['COCO_small'].tolist() == [True, True, False, False]
    assert masks['input_under4'].tolist() == [True] * 4
    results['include_exact_1024_native_small_upper_boundary'] = True
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    output = Path(output); output.mkdir(exist_ok=False)
    report = dict(status='PASS_SIX_RECORD_BINDING_AND_NATIVE_AREA_BOUNDARY_FIXTURES', fixtures=results,
        protocol_sha256=a.sha(a.HERE / 'protocol.json'), dependencies_and_formal_reader_source_hashes_verified=True,
        actual_six_record_evidence_consumed=False, actual_model_performance_tested=False,
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0)
    a.write(output / 'verification.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(validate(args.output)['status'])
