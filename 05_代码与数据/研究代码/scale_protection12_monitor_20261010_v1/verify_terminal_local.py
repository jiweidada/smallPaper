"""Independent downloaded-file checks using NumPy; zero model/evaluator calls."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STATES = ('FAILED_NO_FIXED12_ACCEPTANCE','FIXED12_RELOAD_FAILED','FIXED12_AND_RELOAD_COMPLETE')


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines()] if Path(p).exists() else []


def extract(archive, metadata, destination):
    assert sha(archive)==metadata['sha256'] and Path(archive).stat().st_size==metadata['bytes']
    assert metadata['status'] in STATES
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names)==len(set(names))==len({n.casefold() for n in names})==metadata['files']
        assert z.testzip() is None
        for name in names:
            assert '\\' not in name and (destination/name).resolve().is_relative_to(destination.resolve())
        manifest = json.loads(z.read('manifest.json'))
        assert set(names)==set(manifest['files'])|{'manifest.json'}
        assert all(manifest[k]==metadata[k] for k in ('status','seed','arm'))
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            assert shutil.disk_usage(destination.parent).free > sum(x.file_size for x in z.infolist())+64*1024**2
            destination.mkdir(); z.extractall(destination)
        else:
            assert set(names)=={p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    for name,item in manifest['files'].items():
        assert (destination/name).stat().st_size==item['bytes'] and sha(destination/name)==item['sha256'],name
    return manifest


def endpoint(p):
    from results import load_endpoint
    summary, flags, manifest = load_endpoint(p)
    with np.load(p/'precision_recall.npz',allow_pickle=False) as z:
        precision, recall = z['precision'], z['recall']
    assert precision.shape==(10,101,10,4,3) and recall.shape==(10,10,4,3)
    assert np.isfinite(precision).all() and np.isfinite(recall).all()
    assert ((precision>=-1)&(precision<=1)).all() and ((recall>=-1)&(recall<=1)).all()
    def mean(x):
        good=x[x>-1]; return float(good.mean()) if len(good) else -1.
    rebuilt = [mean(precision[:,:,:,0,2]),mean(precision[0,:,:,0,2]),mean(precision[5,:,:,0,2])]
    rebuilt += [mean(precision[:,:,:,a,2]) for a in (1,2,3)]
    rebuilt += [mean(recall[:,:,0,m]) for m in (0,1,2)]
    rebuilt += [mean(recall[:,:,a,2]) for a in (1,2,3)]
    assert np.allclose(rebuilt,summary['native_stats'],atol=1e-12,rtol=0)
    expected=('AP','AP50','AP75','APs','APm','APl')
    for k,v in zip(expected,rebuilt[:6]): assert abs(100*v-summary['overall_percent'][k])<=1e-9
    assert abs(100*mean(precision[5,:,:,1,2])-summary['overall_percent']['small_AP75'])<=1e-9
    for category in summary['categories_percent']:
        k=category['class_id']
        vals=[mean(precision[:,:,k,0,2]),mean(precision[0,:,k,0,2]),mean(precision[5,:,k,0,2])]
        vals += [mean(precision[:,:,k,a,2]) for a in (1,2,3)]
        for key,v in zip(expected,vals): assert abs(100*v-category[key])<=1e-9
    ids=flags['annotation_id']; index={int(g):i for i,g in enumerate(ids)}
    hits=np.zeros_like(flags['hits']); fp=np.zeros(10,dtype=np.int64);seen=set();cells=set()
    with gzip.open(p/'matching_all_area.json.gz','rt',encoding='utf8') as f: records=json.load(f)
    for row in records:
        assert row['aRng']==[0,1e10] and row['maxDet']==100
        cell=(row['image_id'],row['category_id']);assert cell not in cells;cells.add(cell)
        assert not any(row['gtIgnore'])
        for g in row['gtIds']:
            assert g in index and g not in seen; seen.add(g)
            assert flags['category_id'][index[g]]==row['category_id']
        match,ignore=np.asarray(row['dtMatches']),np.asarray(row['dtIgnore'],dtype=bool)
        assert match.shape==ignore.shape==(10,len(row['dtIds']))
        assert np.isfinite(match).all() and (match==np.floor(match)).all()
        for ti in range(10):
            fp[ti]+=int(((match[ti]==0)&~ignore[ti]).sum())
            for g in match[ti][(match[ti]!=0)&~ignore[ti]].astype(np.int64):
                assert int(g) in row['gtIds'] and not hits[index[int(g)],ti]
                hits[index[int(g)],ti]=True
    assert seen==set(index) and np.array_equal(hits,flags['hits'])
    assert hits.sum(0).tolist()==summary['TP_by_IoU'] and fp.tolist()==summary['FP_by_IoU']
    return summary,flags,dict(PR_native12_exact=True,matching_native_GT_flags_exact=True,
                              FP_exact=True,matching_cells=len(cells),GT=len(ids))


def events(packet,identity):
    profile={int(k):v for k,v in read(packet/'deployment/training_profile.json').items()}
    assert len(profile)==6471 and sum(v['protect'] for v in profile.values())==995
    random_members={}; counts={};seen=set()
    for path in sorted((packet/'run/augmentation_events').glob('*.jsonl')):
        for row in rows(path):
            epoch,iid=row['epoch'],row['image_id'];assert 4<=epoch<=11 and iid in profile
            assert row['arm']==identity['arm'] and row['source_GT']==profile[iid]['raw_valid_GT']
            assert row['risky_fraction']==profile[iid]['risky_fraction']
            assert (epoch,iid) not in seen;seen.add((epoch,iid))
            if identity['arm']=='target_protect': expected=profile[iid]['protect']
            else:
                if epoch not in random_members:
                    ordered=sorted(profile,key=lambda k:hashlib.sha256(
                        f"scale_protection_control:{identity['seed']}:{epoch}:{k}".encode()).digest())
                    random_members[epoch]=set(ordered[:995])
                expected=iid in random_members[epoch]
            assert row['applied']==bool(expected)
            assert row['used_GT']==(row['source_GT'] if expected else row['native_mosaic_GT'])
            assert min(row['source_GT'],row['native_mosaic_GT'],row['used_GT'])>=0
            c=counts.setdefault(epoch,Counter())
            c.update(events=1,substitutions=int(expected),source_GT=row['source_GT'],
                     native_mosaic_GT=row['native_mosaic_GT'],used_GT=row['used_GT'])
    assert set(counts)==set(range(4,12))
    sizes=rows(packet/'run/sampled_training_target_sizes.jsonl')
    assert len(sizes)==20
    assert {(x['epoch_zero_based'],x['epoch_attempt']) for x in sizes}=={
        (e,b) for e in (0,4,8,11) for b in (1,101,501,1001,1501)}
    for row in sizes:
        assert len(row['target_histogram'])==6 and sum(row['target_histogram'])==row['target_exposures']
        assert row['samples_shape'][0]==4 and min(row['target_histogram'])>=0
    return dict(Mosaic_events_by_epoch={str(k):dict(v) for k,v in counts.items()},
                all_policy_decisions_exact=True,post_collate_size_snapshots_verified=20,
                event_target_counts_are_before_sanitation_and_MixUp=True,
                no_full_detector_RNG_equivalence_inferred=True)


def verify(archive,metadata_path,destination,report):
    destination,report=Path(destination),Path(report)
    assert not report.exists()
    manifest=extract(archive,read(metadata_path),destination)
    receipt=read(destination/'prepared/receipt.json');identity=receipt['identity']
    assert manifest['identity']==identity and identity['arm']==manifest['arm'] and identity['seed']==manifest['seed']
    assert identity['arm'] in ('target_protect','random_protect') and identity['seed'] in (1,2)
    local=ROOT/'英文小论文/code/scale_protection12_20261010_v2'
    for name,h in identity['implementation'].items(): assert sha(destination/'code'/name)==h==sha(local/name)
    assert sha(destination/'prepared/plan.json')==identity['plan_sha256']
    assert sha(destination/'prepared/resolved.yml')==receipt['resolved_sha256']
    protocol=read(destination/'code/training_protocol.json')
    assert sha(destination/'code/training_protocol.json')==identity['protocol_sha256']
    spec=protocol['baseline_reuse'][str(identity['seed'])]
    for name,h in spec['pinned_files'].items(): assert sha(destination/'baseline_eval'/name)==h
    original=read(destination/'baseline_eval/completion.json')
    assert original['checkpoint_sha256']==identity['baseline_checkpoint_sha256']
    assert original['images']==548 and original['optimizer_steps']==0 and original['historical_log_reproduction_pass']
    image_list=read(destination/f"deployment/image_manifest_seed{identity['seed']}.json")
    assert (len(image_list['train']),len(image_list['val']))==(6471,548)
    source=read(destination/'code/source_binding.json')['native_files_sha256'];assert len(source)==468
    for name,h in source.items(): assert sha(destination/'native'/name)==h
    for name,h in identity['evaluator_binding']['files'].items(): assert sha(destination/'evaluator/pycocoeval'/name)==h
    for name,key in [('baseline_epoch12.pth','baseline_checkpoint_sha256'),('baseline_first100.jsonl','baseline_first100_sha256'),
                     ('train_annotations.json','train_ann_sha256'),('val_annotations.json','val_ann_sha256')]:
        assert sha(destination/'reference'/name)==identity[key]
    for name,key in [('training_profile.json','training_profile_sha256'),('data_probe.json','data_only_report_sha256'),
                     (f"image_manifest_seed{identity['seed']}.json",'image_manifest_sha256'),
                     (f"image_verification_seed{identity['seed']}.json",'image_verification_sha256')]:
        assert sha(destination/'deployment'/name)==identity[key]
    common=manifest['common_initialization_reused_not_duplicated'];initial=(ROOT/common['local_relative_path']).resolve()
    assert initial.is_relative_to(ROOT) and sha(initial)==common['sha256']==identity['common_init_sha256']
    actual=rows(destination/'run/first100_training_inputs.jsonl');reference=rows(destination/'reference/baseline_first100.jsonl')
    assert len(reference)==100 and len(actual)<=100 and actual==reference[:len(actual)]
    result=dict(status='FAILED_OR_INCOMPLETE_EVIDENCE_VERIFIED_NO_ACCEPTANCE',identity=identity,
        terminal_state=manifest['status'],archive_sha256=sha(archive),files_verified=len(manifest['files'])+1,
        native_source_files_verified=468,evaluator_source_files_verified=7,first100_rows_exact=len(actual),
        local_common_initialization_reused=str(initial),new_model_forwards=0,new_evaluator_calls=0,
        new_optimizer_updates=0,automatic_promotion=False,verification_script_sha256=sha(__file__))
    if manifest['status']=='FIXED12_AND_RELOAD_COMPLETE':
        assert len(actual)==100
        run=destination/'run';done=read(run/'training_completion.json');runtime=read(run/'runtime_start.json')
        assert done['identity']==runtime['identity']==identity
        assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
        assert sha(run/'last.pth')==done['last_checkpoint_sha256']
        assert runtime['initialization']['parameters']==10231997 and runtime['initialization']['extra_parameters']==0
        assert runtime['initialization']['strict_keys']==794
        assert runtime['optimizer_coverage']==dict(trainable_tensors=488,all_native_parameter_groups_unchanged=True,
                                                    all_parameters_exactly_once=True,added_parameters=0)
        reuse=read(destination/'screen/baseline_reuse.json')
        assert reuse['new_model_forwards']==reuse['new_optimizer_steps']==reuse['new_epochs']==0
        assert read(run/'args.json')['yaml_cfg']==read(destination/'prepared/resolved.yml')
        audit=rows(run/'training_audit_epochs.jsonl')
        assert len(audit)==12 and [r['epoch_zero_based'] for r in audit]==list(range(12))
        previous=0
        for e,r in enumerate(audit):
            assert r['attempts_total']==r['EMA_updates']==1617*(e+1)
            assert previous<=r['actual_optimizer_steps_total']<=r['attempts_total']
            assert r['attempts_without_optimizer_step']==r['attempts_total']-r['actual_optimizer_steps_total']
            assert r['scale_protection_active']==(e>=4) and r['no_new_model_parameters']
            previous=r['actual_optimizer_steps_total']
        replay=destination/'candidate_reload_v1';proof=read(replay/'verification.json')
        assert read(replay/'completion.json')['exit_code']==0 and proof['status']=='PASS_SAVED_CANDIDATE_RELOAD'
        assert proof['identity']==identity and proof['strict_loaded_tensors']==794 and proof['parameters']==10231997
        assert proof['checkpoint_sha256']==done['last_checkpoint_sha256']
        assert proof['original_training_completion_sha256']==sha(run/'training_completion.json')
        tools=read(HERE/'closeout_protocol.json');assert read(destination/'tools/closeout_protocol.json')==tools
        for name,h in tools['files'].items(): assert sha(destination/'tools'/name)==h==sha(HERE/name)
        assert proof['verification_script_sha256']==tools['files']['verify_candidate_reload.py']
        assert proof['closeout_protocol_sha256']==sha(HERE/'closeout_protocol.json')
        assert proof['optimizer_updates']==proof['epochs_added']==proof['baseline_model_forwards']==0
        assert proof['candidate_model_forwards']==137 and proof['images']==548
        sys.path.insert(0,str(destination/'code'))
        from results import pair
        from evaluator_binding import same_result_evaluator
        base,bh,bcheck=endpoint(destination/'baseline_eval/endpoint_epoch12')
        candidate,ch,ccheck=endpoint(run/'endpoint_epoch12');loaded,lh,lcheck=endpoint(replay/'endpoint_epoch12')
        same_result_evaluator(base,candidate);same_result_evaluator(candidate,loaded)
        assert candidate['identity']==loaded['identity']==identity
        assert np.allclose(candidate['native_stats'],loaded['native_stats'],atol=1e-8,rtol=0)
        assert all(np.array_equal(ch[k],lh[k]) for k in ch)
        for folder in (run,replay,destination/'baseline_eval'):
            m=read(folder/'stages_epoch12/manifest.json')
            assert m['trace_code_sha256']==identity['implementation']['stage_trace.py']
            assert set(m['files'])=={'summary.json'}|{f'image_{i:06d}.npz' for i in range(548)}
            for name,h in m['files'].items(): assert sha(folder/'stages_epoch12'/name)==h
            stage=read(folder/'stages_epoch12/summary.json')
            assert stage['image_ids']==list(range(548)) and stage['epoch_zero_based']==11
            assert stage['stored_original_hw_order']=='height_width' and stage['query_count']==300
            assert stage['additional_model_or_head_calls']==stage['additional_IoU_or_matcher_calls']==0
        for i in range(548):
            name=f'image_{i:06d}.npz'
            with np.load(run/'stages_epoch12'/name,allow_pickle=False) as a,np.load(replay/'stages_epoch12'/name,allow_pickle=False) as b:
                assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
        comparison=pair(base,candidate,bh,ch);saved=read(run/'fixed12_comparison.json')
        for k,v in comparison.items(): assert saved[k]==v,k
        delta=comparison['delta_AP_metrics_pp'];passed=delta['APs']>0 and delta['AP']>=0
        assert passed==saved['native_gate_passed']==done['native_gate_passed']
        prefix=[];br=rows(destination/'reference/baseline_log.txt');cr=rows(run/'log.txt')
        assert len(cr)==12 and [int(r['epoch']) for r in cr]==list(range(12))
        for e in range(4):
            x=next(r for r in br if int(r['epoch'])==e)['test_coco_eval_bbox'];y=cr[e]['test_coco_eval_bbox']
            prefix.append(dict(epoch=e+1,baseline=x,candidate=y,all12_exact=x==y,
                               delta_AP_pp=100*(y[0]-x[0]),delta_APs_pp=100*(y[3]-x[3]),protection_active=False))
        result.update(status='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD',
            baseline_overall_percent=base['overall_percent'],candidate_overall_percent=candidate['overall_percent'],
            delta_AP_metrics_pp=delta,native_gate_passed_this_seed=passed,paired_native_results=comparison,
            endpoint_checks=dict(baseline=bcheck,candidate=ccheck,reloaded=lcheck),all548_reloaded_stage_arrays_exact=True,
            augmentation=events(destination,identity),pre_intervention_prefix=prefix,
            final_training_counts={k:audit[-1][k] for k in ('attempts_total','actual_optimizer_steps_total','EMA_updates','attempts_without_optimizer_step')},
            training_elapsed_seconds=read(destination/'screen/candidate_completion.json')['elapsed_seconds'],
            scope='One-seed12-epoch screening. Historical pre-intervention drift limits causal attribution. Conditional matched-rate controls and both-seed gates remain required; no uniqueness, significance or mature-model claim.')
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('archive','metadata','destination','report'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=verify(a.archive,a.metadata,a.destination,a.report)
    print(json.dumps(dict(status=r['status'],seed=r['identity']['seed'],arm=r['identity']['arm'])),flush=True)
