"""Describe paired saved outputs, without constructing a detector or evaluator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import geometry_core as geometry


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def write(p, value):
    with Path(p).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def onset(a, mask, threshold, ti):
    support = a['max_iou'] >= threshold
    tp = a['native_TP'][:,ti]
    patterns = support[:,0].astype(int)*4+support[:,1].astype(int)*2+support[:,2].astype(int)
    r = dict(GT=int(mask.sum()),
        geometric_support_patterns={f'{k:03b}':int((mask & (patterns==k)).sum()) for k in range(8)},
        no_support_at_any_observed_boundary=int((mask & ~support.any(1)).sum()),
        first_support_at_initial=int((mask & support[:,0]).sum()),
        first_support_at_pre=int((mask & ~support[:,0] & support[:,1]).sum()),
        first_support_at_final=int((mask & ~support[:,:2].any(1) & support[:,2]).sum()),
        never_supported_and_not_native_TP=int((mask & ~support.any(1) & ~tp).sum()),
        early_supported_lost_final_and_not_native_TP=int((mask & support[:,:2].any(1) & ~support[:,2] & ~tp).sum()),
        final_geometry_without_native_TP=int((mask & support[:,2] & ~tp).sum()),
        native_all_area_TP_filtered_GT=int((mask & tp).sum()),
        native_TP_below_recomputed_final_geometry=int((mask & tp & ~support[:,2]).sum()))
    assert sum(r['geometric_support_patterns'].values()) == r['GT']
    assert sum(r[k] for k in ('no_support_at_any_observed_boundary','first_support_at_initial',
                             'first_support_at_pre','first_support_at_final')) == r['GT']
    assert sum(r[k] for k in ('never_supported_and_not_native_TP','early_supported_lost_final_and_not_native_TP',
                             'final_geometry_without_native_TP','native_all_area_TP_filtered_GT')) == r['GT']
    return r


def analyze(packet, output, verification_report):
    packet,output = Path(packet),Path(output)
    assert not output.exists()
    verification = read(verification_report)
    monitor = Path(__file__).resolve().parents[3] / '英文小论文/code/fdr_tail12_monitor_20261011_v1'
    assert verification['status'] == 'PASS_FDR_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_EMA_RELOAD'
    assert verification['local_reader_protocol_sha256'] == sha(monitor/'local_reader_protocol.json')
    assert verification['verification_script_sha256'] == sha(monitor/'verify_terminal_local.py')
    assert verification['terminal_manifest_sha256'] == sha(packet/'terminal_manifest.json')
    sys.path.insert(0,str(packet/'code'))
    from results import load_endpoint,pair
    from evaluator_binding import same_result_evaluator
    identity = read(packet/'bundle/receipt.json')['identity']
    arm,seed = identity['arm'],identity['seed']
    assert arm == 'endpoint2' and seed in (1,2)
    assert identity == verification['identity'] and identity['endpoint_factor'] == 2.0
    protocol = read(Path(__file__).parent/'analysis_protocol.json')
    assert sha(Path(__file__).parent/'geometry_core.py') == protocol['files']['geometry_core.py']
    ann_path = packet/'reference/val_annotations.json'
    assert sha(ann_path) == identity['val_annotation_sha256']
    annotation = read(ann_path)
    images = {int(x['id']):x for x in annotation['images']}
    anns = {int(x['id']):x for x in annotation['annotations']}
    assert len(images)==548 and len(anns)==38759
    output.mkdir()
    values,summaries,hits,provenance = {},{},{},{}
    for name,folder in (('baseline',packet/'baseline_eval'),(arm,packet/'run')):
        summary,flags,_ = load_endpoint(folder/'endpoint_epoch12')
        assert summary['identity']['seed'] == seed
        assert summary['identity']['arm'] == ('baseline' if name=='baseline' else arm)
        manifest = read(folder/'stages_epoch12/manifest.json')
        assert manifest['trace_code_sha256'] == identity['implementation']['stage_trace.py']
        files = {}
        invalid = {k:0 for k in geometry.STAGES}
        def reader(iid):
            p = folder/f'stages_epoch12/image_{iid:06d}.npz'
            assert sha(p) == manifest['files'][p.name]
            files[p.name] = manifest['files'][p.name]
            with np.load(p,allow_pickle=False) as z:
                observed = {k:z[k].copy() for k in z.files}
            for k in geometry.STAGES:
                assert observed[k].shape==(300,4)
                invalid[k] += int((observed[k][:,2:]<=0).any(1).sum())
            return observed
        values[name] = geometry.collect(annotation,reader,flags)
        summaries[name],hits[name] = summary,flags
        provenance[name] = dict(endpoint_manifest_sha256=sha(folder/'endpoint_epoch12/manifest.json'),
            stage_manifest_sha256=sha(folder/'stages_epoch12/manifest.json'),images_read=len(files),
            invalid_raw_size_queries_assigned_zero_geometry=invalid,stage_files=files)
        assert len(files)==548
        np.savez_compressed(output/f'{name}_per_GT.npz',**values[name])
    base,trial = values['baseline'],values[arm]
    for k in ('annotation_id','image_id','category_id','original_area'):
        assert np.array_equal(base[k],trial[k]),k
    assert set(map(int,base['annotation_id']))==set(anns)
    same_result_evaluator(summaries['baseline'],summaries[arm])
    paired = pair(summaries['baseline'],summaries[arm],hits['baseline'],hits[arm])
    saved = read(packet/'run/fixed12_comparison.json')
    for k,v in paired.items(): assert saved[k]==v,k
    side = np.array([min(anns[int(i)]['bbox'][2]*640/images[int(anns[int(i)]['image_id'])]['width'],
                        anns[int(i)]['bbox'][3]*640/images[int(anns[int(i)]['image_id'])]['height'])
                     for i in base['annotation_id']])
    groups = dict(all=np.ones(len(side),dtype=bool),COCO_small=base['original_area']<=1024,
        input_under4=side<4,input_4_to_under8=(side>=4)&(side<8),
        input_8_to_under16=(side>=8)&(side<16),input_ge16=side>=16)
    assert groups['input_under4'].sum()==5096 and groups['COCO_small'].sum()==26586
    by_group,table = {},[]
    conflicts = 0
    for group,mask in groups.items():
        rows = {}
        for threshold,ti in ((.5,0),(.75,5)):
            native = onset(base,mask,threshold,ti); candidate = onset(trial,mask,threshold,ti)
            support_changes = [dict(stage=k,
                new_supported=int((mask & (base['max_iou'][:,s]<threshold) & (trial['max_iou'][:,s]>=threshold)).sum()),
                lost_supported=int((mask & (base['max_iou'][:,s]>=threshold) & (trial['max_iou'][:,s]<threshold)).sum()))
                for s,k in enumerate(geometry.STAGES)]
            rows[str(threshold)] = dict(baseline=native,candidate=candidate,geometric_support_changes=support_changes,
                new_native_TP=int((mask & ~base['native_TP'][:,ti] & trial['native_TP'][:,ti]).sum()),
                lost_native_TP=int((mask & base['native_TP'][:,ti] & ~trial['native_TP'][:,ti]).sum()))
            if group=='all':
                conflicts += native['native_TP_below_recomputed_final_geometry']+candidate['native_TP_below_recomputed_final_geometry']
            for name,data in (('baseline',native),(arm,candidate)):
                table.append(dict(seed=seed,arm=name,group=group,IoU=threshold,**{k:v for k,v in data.items() if isinstance(v,int)}))
        by_group[group] = rows
    report = dict(status='PAIRED_SAVED_STAGE_DESCRIPTION' if not conflicts else 'STAGE_NUMERICAL_REVIEW_REQUIRED',
        seed=seed,arm=arm,identity=identity,provenance=provenance,groups=by_group,
        native_fixed12_delta_AP_metrics_pp=paired['delta_AP_metrics_pp'],native_gate_passed=saved['native_gate_passed'],
        source_verification_report_sha256=sha(verification_report),analysis_protocol_sha256=sha(Path(__file__).parent/'analysis_protocol.json'),
        boundary_locations=protocol['stage_boundary_locations'],native_TP_geometry_disagreements=conflicts,
        geometry_reader_sha256=sha(Path(__file__).parent/'geometry_core.py'),analysis_code_sha256=sha(__file__),
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
        scope=protocol['stage_analysis_scope'])
    write(output/'report.json',report)
    with (output/'onset_counts.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    write(output/'manifest.json',dict(files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--packet',required=True); p.add_argument('--output',required=True);p.add_argument('--verification-report',required=True)
    a = p.parse_args(); r = analyze(a.packet,a.output,a.verification_report)
    print(json.dumps(dict(status=r['status'],seed=r['seed'],arm=r['arm'])),flush=True)
