"""Exercise numerical helpers on real BASE12 files; reject wrong evidence.

This does not manufacture or certify a new NATIVE_STEM_SCRATCH terminal packet.
"""
import argparse
from copy import deepcopy
import sys
from verify_terminal_local import CORE, ROOT, local_protocol, native_endpoint, read, reuse_sources, sha, stages, validate_identity, validate_metadata, validate_names, write
from decide_fixed12 import decision


def rejected(function, *args):
    try:
        function(*args)
    except (AssertionError, KeyError, ValueError):
        return True
    raise AssertionError('Invalid/ineligible evidence was accepted')


def main(output):
    p = local_protocol()
    results = []
    old_reports = []
    for seed in (1, 2):
        a = p['reused_local_evidence'][str(seed)]
        packet = ROOT / a['packet_relative_path']
        ann = read(packet / 'reference/val_annotations.json')
        baseline, flags, checks = native_endpoint(packet / 'baseline_eval/endpoint_epoch12', ann)
        assert baseline['identity']['seed'] == seed and baseline['identity']['arm'] == 'baseline'
        assert checks['GT'] == 38759 and flags['hits'].shape == (38759, 10)
        old = read(ROOT / a['report_relative_path'])
        old_reports.append(ROOT / a['report_relative_path'])
        assert baseline['overall_percent'] == old['baseline_overall_percent']
        assert reuse_sources(old['identity'], p) == packet.resolve()
        stages(packet / 'baseline_eval', baseline['identity'], ann)
        old_plan = read(packet / 'prepared/plan.json')
        assert rejected(validate_identity, old['identity'], old_plan, p)
        old_meta = read(packet.parent / f'terminal_target_protect_seed{seed}_v1_metadata.json')
        assert rejected(validate_metadata, old_meta)
        results.append(dict(seed=seed, baseline_overall_percent=baseline['overall_percent'],
            endpoint=checks, reused_reference_current_bytes_verified=True,
            existing_baseline_stage_arrays_verified=548, legacy_scale_packet_rejected_as_NATIVE_STEM_SCRATCH=True))
    for names in (['../escape'], ['a\\b'], ['C:/escape'], ['A', 'a'], ['a/./b'], ['/escape']):
        assert rejected(validate_names, names)
    assert rejected(validate_metadata, {'status':'STEM_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED', 'seed':3})
    assert rejected(decision, old_reports)
    assert rejected(decision, old_reports[:1])
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    value = dict(status='PASS_READ_ONLY_HELPERS_ON_REAL_BASE12_AND_WRONG_PACKET_REJECTION',
        local_reader_protocol_sha256=sha(CORE.parent / 'native_stem_scratch12_monitor_20261011_v2/local_reader_protocol.json'),
        actual_existing_BASE12_checks=results, unsafe_archive_names_rejected=6,
        legacy_and_incomplete_two_seed_decisions_rejected=True,
        new_NATIVE_STEM_SCRATCH_terminal_packets_validated=0, new_model_forwards=0, new_evaluator_calls=0,
        new_optimizer_updates=0, new_training_epochs=0,
        limitation='Only reader preparation and numerical helper/rejection checks have passed. Actual new NATIVE_STEM_SCRATCH terminal archives, saved-weight replay and efficacy gates are still pending.')
    write(output, value)
    print(value['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.output)
