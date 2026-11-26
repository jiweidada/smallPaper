"""Independently verify downloaded early-run evidence, not final efficacy."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/small_object_encoder_decoder12_analysis_20261010_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
PREFIXES=('encoder.geometry_field.','decoder.enc_bbox_head.geometry_proposal.',
          'decoder.decoder.geometry_layers.','decoder.decoder.geometry_attention.')


def main():
    reports=[];metadata=read(OUT/'collection/archive_metadata.json')
    for seed,version in ((1,4),(2,3)):
        archive=OUT/f'collection/early_audit_seed{seed}.zip';assert sha(archive)==metadata[f'seed{seed}']['sha256']
        target=OUT/f'collection/seed{seed}';assert not target.exists()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
            for name in z.namelist():assert (target/name).resolve().is_relative_to(target.resolve())
            target.mkdir();z.extractall(target)
        manifest=read(target/'manifest.json')
        assert set(z.namelist())==set(manifest['files'])|{'manifest.json'}
        for name,item in manifest['files'].items():
            p=target/name;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name
        receipt=read(target/'prepared/receipt.json');identity=receipt['identity']
        local=ROOT/f'英文小论文/code/small_object_encoder_decoder12_20261010_v{version}'
        for name,expected in identity['implementation'].items():
            assert sha(target/'code'/name)==expected==sha(local/name),name
        assert sha(target/'prepared/plan.json')==identity['plan_sha256']
        assert sha(target/'prepared/image_manifest.json')==identity['image_manifest_sha256']
        images=read(target/'prepared/image_manifest.json');assert len(images['train'])==6471 and len(images['val'])==548
        protocol=read(target/'code/protocol.json');assert protocol['max_total_epochs_each_configuration']==12
        actual=[json.loads(s) for s in (target/'run/first100_training_inputs.jsonl').read_text().splitlines()]
        reference=[json.loads(s) for s in (target/'reference/baseline_first100.jsonl').read_text().splitlines()]
        assert len(actual)==100 and actual==reference
        assert sha(target/'reference/baseline_first100.jsonl')==identity['baseline_first100_sha256']
        runtime=read(target/'run/runtime_start.json');assert runtime['identity']==identity
        assert runtime['optimizer_coverage']['all_parameters_exactly_once']
        assert len(runtime['optimizer_coverage']['added_GroupNorm_zero_decay'])==6
        eight=read(target/'run/first8_optimizer_steps.json');assert len(eight)==8
        assert [r['actual_optimizer_step'] for r in eight]==list(range(1,9))
        assert all(v is not None and v['finite'] for r in eight for v in r['gradients_after_native_unscale_clip'].values())
        epochs=[json.loads(s) for s in (target/'run/training_audit_epochs.jsonl').read_text().splitlines()]
        assert [e['epoch_zero_based'] for e in epochs]==list(range(len(epochs)))
        first=epochs[0];assert first['attempts_total']==1617 and first['EMA_updates']==1617
        assert all(e['EMA_updates']==e['attempts_total'] for e in epochs)
        groups={p:sum(v for n,v in first['added_changed_elements_from_initial'].items() if n.startswith(p)) for p in PREFIXES}
        assert all(v>0 for v in groups.values())
        assert all(v['finite'] for e in epochs for v in e['new_parameters'].values())
        logs=[json.loads(s) for s in (target/'run/log.txt').read_text().splitlines() if s.strip()]
        base={r['epoch']:r for r in [json.loads(s) for s in (target/'reference/baseline_log.txt').read_text().splitlines()]}
        comparisons=[]
        for row in logs:
            original=base[row['epoch']]
            comparisons.append({'epoch_one_based':row['epoch']+1,**{
                name:{'candidate_percent':row['test_coco_eval_bbox'][i]*100,
                      'baseline_percent':original['test_coco_eval_bbox'][i]*100,
                      'delta_pp':(row['test_coco_eval_bbox'][i]-original['test_coco_eval_bbox'][i])*100}
                for name,i in [('AP',0),('APs',3),('AP75',2)]}})
        cache=read(target/'screen/baseline_reuse.json') if seed==1 else None
        if cache:assert cache['new_model_forwards']==0 and cache['evidence_files_verified']==553
        reports.append({'seed':seed,'runtime_version':version,'archive_sha256':sha(archive),
            'files_verified':len(manifest['files'])+1,'all100_input_target_RNG_rows_exact':True,
            'all_new_gradient_tensors_finite_at_first8_actual_steps':True,
            'first_epoch_attempts':first['attempts_total'],'first_epoch_actual_optimizer_updates':first['actual_optimizer_steps_total'],
            'first_epoch_EMA_updates':first['EMA_updates'],'first_epoch_changed_elements_by_group':groups,
            'first_epoch_all_new_groups_updated':True,'completed_epochs_in_snapshot':len(logs),
            'first_epoch_peak_allocated_GiB':first['CUDA_peak_allocated_bytes']/2**30,
            'same_epoch_comparisons':comparisons,'seed1_baseline_cache_reused':bool(cache),
            'captured_remote_utc':manifest['captured_remote_utc']})
    geometry=read(OUT/'baseline_seed1/report.json')
    assert geometry['native_TP_geometry_disagreement_cells']==0
    tests=read(Path(__file__).with_name('engineering_synthetic')/'report.json');assert tests['check_count']==30
    result={'status':'PASS_DOWNLOADED_EARLY_AUDIT_AND_BASELINE_STAGE_ANALYSIS',
        'runs':reports,'early_results_are_not_fixed12_decisions':True,'fixed12_results_available':False,
        'prior_failed_seed1_engineering_attempt_retained':{'attempts':10,'actual_optimizer_updates':8},
        'CPU_geometry_tests':30,'baseline_stage_analysis':str((OUT/'baseline_seed1/report.json').resolve()),
        'baseline_stage_analysis_images':548,'baseline_stage_analysis_small_GT':26586,
        'zero_extra_baseline_model_calls':True,'geometry_native_TP_conflicts':0,
        'candidate_reload_verifier':'Prepared but not executed; can only run after exact12 supervisor completion; zero training or baseline model calls.',
        'material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate',
            'scope':'Early engineering integration verified; native fixed12 efficacy,capacity controls and checkpoint reload pending.'},
        'fallacy_scan':{'checked':11,'total':11,'boundaries':[
            'Seeds and epochs remain separate; no pooled improvement.',
            'Per-query geometry is not a per-image or per-dataset causal claim.',
            'All548 baseline images and38759GT retained.',
            'Native GT identities retained; geometry-best query is not treated as matched TP.',
            'COCO-small denominator26586; geometry support andnativeTP reported separately.',
            'Final12 fixed; no early best-epoch adoption.',
            'Failed10attempt/8update trial retained.',
            'No additional seed,arm,hyperparameter sweep or training.',
            'Negative-size CPU reader handling recorded with originalfailedread; native outputs andprimarygate unchanged.',
            'Passive stage relations do not prove a causal mechanism.',
            'Early positiveAP is not used to retroactively validate architecture rationale.']}}
    with (OUT/'early_validation_report.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'seeds':len(reports),'files_verified':sum(r['files_verified'] for r in reports),
        'all100_parity_both':True,'epoch1_all_groups_moved_both':True,'fixed12_results_available':False}))


if __name__=='__main__':main()
