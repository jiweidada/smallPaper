"""Reject incomplete diagnostic proof and unsafe archive member names. Fixtures only."""
from copy import deepcopy
from pathlib import Path
import argparse, sys
from collector import protocol, read, HERE, sha, validate_metadata, names_valid, write


def validate(output):
    p = protocol(); results = {}
    def reject(name, function):
        try: function()
        except AssertionError: results[name] = True
        else: raise AssertionError('Unsafe fixture accepted: ' + name)
    identity = dict(arm='highfreq_rgb', seed=2)
    plan = dict(arm='highfreq_rgb', seed=2, max_epochs=12, fixed_endpoint='native_epoch12_EMA')
    trained = dict(status='NATIVE_FIT_FINISHED_EXACT12', identity=identity, completed_epochs=12,
        entered_epochs=list(range(12)), automatic_retry=False, last_checkpoint_sha256='fixture-checkpoint')
    v = dict(status='PASS_SELECTION_DIAGNOSTIC_EXACT_SAVED_EMA_REPLAY', identity=identity, arm='highfreq_rgb', seed=2,
        checkpoint_sha256='fixture-checkpoint', original_training_completion_sha256='fixture-completion',
        diagnostic_protocol_sha256=p['diagnostic_protocol_sha256'], verification_script_sha256=p['diagnostic_source_sha256']['replay.py'],
        original_stats=[0.] * 12, reloaded_stats=[0.] * 12, stats_max_abs_fraction_difference=0, stats_tolerance=0,
        native_GT_hits_exact=True, all548_stage_arrays_exact=True, stage_max_abs_difference={'fixture': 0},
        candidate_model_forwards=137, actual_selector_calls=137, existing_selected_bbox_head_calls=137,
        images=548, selection_images=548, additional_bbox_head_or_topk_calls=0, baseline_model_forwards=0,
        optimizer_updates=0, epochs_added=0, automatic_retry=False, all_parameters_frozen_like_native_EMA=True)
    done = dict(exit_code=0, timed_out=False, automatic_retry=False, baseline_model_forwards=0, epochs_added=0)
    check = lambda proof=v, training=trained: validate_metadata(identity, plan, training, proof, done, p, 'highfreq_rgb', 2, 'fixture-checkpoint', 'fixture-completion')
    check(); results['positive_exact12_v2_binding'] = True
    for name, key, value in (('reject_primary_only', 'status', 'PRIMARY_ONLY'),
            ('reject_checkpoint_mismatch', 'checkpoint_sha256', 'other'),
            ('reject_inexact_metric', 'stats_max_abs_fraction_difference', 1e-12),
            ('reject_inexact_stage', 'all548_stage_arrays_exact', False),
            ('reject_baseline_forward', 'baseline_model_forwards', 1),
            ('reject_extra_head', 'additional_bbox_head_or_topk_calls', 1)):
        bad = deepcopy(v); bad[key] = value; reject(name, lambda bad=bad: check(proof=bad))
    bad = deepcopy(trained); bad['entered_epochs'] = list(range(11))
    reject('reject_incomplete12_training', lambda: check(training=bad))
    names_valid(['diagnostic/verification.json', 'binding/receipt.json']); results['safe_distinct_member_names'] = True
    for name, members in (('reject_parent_traversal', ['../outside']), ('reject_absolute_path', ['/outside']),
                          ('reject_windows_drive', ['C:/outside']), ('reject_case_collision', ['a.json', 'A.json'])):
        reject(name, lambda members=members: names_valid(members))
    assert 'torch' not in sys.modules
    output = Path(output); output.mkdir(exist_ok=False)
    result = dict(status='PASS_READ_ONLY_COLLECTION_BINDING_AND_ARCHIVE_NAME_FIXTURES', cases=results,
        protocol_sha256=sha(HERE / 'protocol.json'), actual_diagnostic_records_collected=0, actual_model_performance_tested=False,
        new_model_forwards=0, new_training_epochs=0, existing_process_signals=0)
    write(output / 'verification.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(validate(args.output)['status'])
