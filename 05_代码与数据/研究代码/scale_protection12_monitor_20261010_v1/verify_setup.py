"""Independently verify downloaded setup evidence without Torch or remote calls."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

WORK = Path(__file__).resolve().parents[3]
OUT = WORK/'outputs/scale_protection12_20261010_v1'
CODE = WORK/'英文小论文/code/scale_protection12_20261010_v2'
EXPECTED = {1:'d22fd8ef1dfd40bc8901a37dd86d6e51fb240fa1e98ee188ed370894eee6cd88',
            2:'e870c1e74cf0c17cff8060db6631f9ea645d9b0282983fa08e6b74ecc473ad84'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    packets, checks = {}, []
    deployment = json.loads((OUT/'training_deployment_manifest_v2.json').read_text(encoding='utf8'))
    protocol = json.loads((CODE/'training_protocol.json').read_text(encoding='utf8'))
    for seed, digest in EXPECTED.items():
        archive = OUT/f'setup_seed{seed}_v2.zip'
        assert sha(archive.read_bytes()) == digest
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('setup_collection_manifest.json'))
            assert set(z.namelist()) == set(manifest['files'])|{'setup_collection_manifest.json'}
            for name, item in manifest['files'].items():
                n = PurePosixPath(name)
                assert not n.is_absolute() and '..' not in n.parts and ':' not in name
                b = z.read(name)
                assert len(b) == item['bytes'] and sha(b) == item['sha256'], name
            bmap = {n:z.read(n) for n in z.namelist()}
        js = lambda name: json.loads(bmap[name])
        screen = f'seed{seed}_target_protect/'
        receipt = js(screen+'bundle/receipt.json'); ident = receipt['identity']
        assert ident['seed'] == seed and ident['arm'] == 'target_protect'
        assert ident['protocol_sha256'] == sha((CODE/'training_protocol.json').read_bytes())
        assert ident['experiment_protocol_sha256'] == sha(bmap['protocol.json']) == protocol['experiment_protocol_sha256']
        assert ident['data_only_report_sha256'] == protocol['data_only_report_sha256']
        assert ident['training_profile_sha256'] == protocol['training_profile_sha256']
        assert ident['native_parameter_count'] == 10231997 and ident['added_parameter_count'] == 0
        for name, digest2 in ident['implementation'].items():
            b = bmap['code_v2/'+name]
            assert sha(b) == digest2 == deployment['files']['code_v2/'+name]['sha256']
            assert b == (CODE/name).read_bytes(), name
        for name in ('geometry.py','residual.py'):
            legacy = WORK/'英文小论文/code/fine_residual_screen24_20261009_v2'/name
            assert bmap['code_v2/'+name] == legacy.read_bytes()
        assert sha(bmap[screen+'bundle/plan.json']) == ident['plan_sha256']
        assert sha(bmap[screen+'bundle/resolved.yml']) == receipt['resolved_sha256']
        plan = js(screen+'bundle/plan.json'); cfg = js(screen+'bundle/resolved.yml')
        allowed = copy.deepcopy(plan['historical_resolved_config'])
        for k in ('resume','tuning','output_dir','device','use_swanlab'):
            allowed[k] = cfg[k]
        for split in ('train','val'):
            for k in ('img_folder','ann_file'):
                allowed[split+'_dataloader']['dataset'][k] = cfg[split+'_dataloader']['dataset'][k]
        assert cfg == allowed, 'Unexpected recipe drift'
        assert plan['max_epochs'] == 12 and cfg['epoches'] == 132
        assert cfg['flat_epoch'] == 64 and cfg['no_aug_epoch'] == 12
        assert cfg['ResidualGeometryDEIM']['arm'] == 'baseline'
        assert cfg['ResidualGeometryDEIM']['recipe'] == plan['model_recipe']
        assert cfg['seed'] == seed and cfg['use_amp'] and cfg['use_ema']
        assert not cfg.get('use_amp_fallback_fp32', False)
        assert cfg['baseline_log_monitor']['enabled'] is False
        for split, count in (('train',6471),('val',548)):
            image_manifest = js(f'image_manifest_seed{seed}.json')
            assert len(image_manifest[split]) == count
        assert sha(bmap[f'image_manifest_seed{seed}.json']) == ident['image_manifest_sha256']
        assert sha(bmap[f'image_verification_seed{seed}.json']) == ident['image_verification_sha256']
        assert js(f'image_verification_seed{seed}.json')['files_verified'] == 7019
        native = bmap['reference/baseline_first100_training_inputs.jsonl']
        assert sha(native) == ident['baseline_first100_sha256'] == protocol['references'][str(seed)]['baseline_first100_sha256']
        native_rows = [json.loads(x) for x in native.splitlines()]
        candidate = bmap[screen+'run/first100_training_inputs.jsonl']
        candidate_rows = [json.loads(x) for x in candidate.splitlines()]
        assert candidate_rows == native_rows and len(candidate_rows) == 100
        assert js(screen+'run/first100_parity.json')['all100_rows_exact'] is True
        runtime = js(screen+'run/runtime_start.json')
        assert runtime['identity'] == ident and runtime['initialization']['strict_keys'] == 794
        assert runtime['initialization']['parameters'] == 10231997
        assert runtime['initialization']['extra_parameters'] == 0
        opt = runtime['optimizer_coverage']
        assert opt['trainable_tensors'] == 488 and opt['all_parameters_exactly_once']
        assert opt['all_native_parameter_groups_unchanged'] and opt['added_parameters'] == 0
        assert runtime['evaluator_runtime']['binding'] == ident['evaluator_binding'] == protocol['references'][str(seed)]['evaluator_binding']
        reuse = js(screen+'baseline_reuse.json')
        assert reuse['new_model_forwards'] == reuse['new_optimizer_steps'] == reuse['new_epochs'] == 0
        assert reuse['evidence_files_verified'] == 553
        assert reuse['original_source'] == protocol['baseline_reuse'][str(seed)]['source']
        assert js(f'prepare_seed{seed}_completion.json')['exit_code'] == 1
        assert js(f'prepare_seed{seed}_v2_completion.json')['exit_code'] == 0
        assert 'baseline_log_monitor' in bmap[f'prepare_seed{seed}.log'].decode()
        evidence = js('setup_evidence.json')
        assert evidence['endpoint_AP_not_available'] and evidence['first100_all_rows_exact']
        packets[seed] = bmap
        checks.append(dict(seed=seed,archive_sha256=EXPECTED[seed],immutable_files_verified=len(manifest['files']),
            first100_input_and_pre_forward_RNG_rows_equal=100,parameters=10231997,added_parameters=0,
            native_recipe_changes_except_paths_and_logging=0,baseline_forwards_or_epochs_added=0,
            current_images_verified=7019,prepared_before_training_repair_retained=True))
    # Materialize only after both complete packets pass.
    destination = OUT/'setup_collection_v2'
    assert not destination.exists()
    destination.mkdir()
    for seed, bmap in packets.items():
        d = destination/f'seed{seed}'; d.mkdir()
        for name,b in bmap.items():
            p = d/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)
    report = dict(material_passport=dict(skill='academic-research-suite',workflow='experiment-agent',
        mode='validate',verification_status='ENGINEERING_ONLY_NOT_TERMINAL',engineering_status='TWO_DOWNLOADED_SETUP_PACKETS_VERIFIED'),
        status='BOTH_NATIVE_SETUP_AND_FIRST100_PARITY_PASS',seeds=checks,
        scientific_scope='Two ongoing fixed12 screens; no candidate AP, efficacy, causal root or full training trajectory equality established.',
        collector_print_issue=dict(seed1='Archive closed successfully; stdout JSON serializer used flush incorrectly and exited1. All41 archived files independently verified; original source preserved.',seed2='Print corrected before collection; exited0.'),
        fallacy_scan=dict(coverage='11/11',items={
            '1_Simpson':'Separate seed gates; no averaged positive outcome substitutes for either seed.',
            '2_Ecological':'Data-panel instance counts and setup parity do not establish detection AP.',
            '3_Berkson':'Fixed128 physical panel and full7019 byte verification are separate populations; neither is a selected success case.',
            '4_Collider':'Conditioning protection on training-source size is an explicit policy, not a causal root identification.',
            '5_Base_rates':'Protection995/6471; physical panel128; evaluation548; first100 parity100rows per seed are distinct denominators.',
            '6_Regression_to_mean':'Fixed12 endpoint and two seeds; no best epoch selection.',
            '7_Survivorship':'Both preparation failures and the non-training collector print error are preserved; no candidate update budget reset.',
            '8_Look_elsewhere':'Exploratory follow-up after rejected earlier mechanisms; no significance or novelty claim.',
            '9_Forking_paths':'Data and training protocol bytes unchanged by the disabled-monitor check repair; candidate source frozen before updates.',
            '10_Correlation_causation':'Input geometry improvement changes density/context too; conditional random control is required after native gates.',
            '11_Reverse_causality':'Early pre-forward RNG equality does not show later learned representations or unique error cause.'}))
    with (OUT/'setup_validation_report.json').open('x',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],seeds=checks),ensure_ascii=False))


if __name__ == '__main__':
    main()
