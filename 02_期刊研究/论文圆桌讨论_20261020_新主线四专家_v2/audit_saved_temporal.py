"""Existing-prediction temporal description; neither new inference nor a method."""
import hashlib, json, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'英文小论文/data/07_第二数据集_20261017'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def describe(fail, tracks):
    runs=[]; changes=edges=isolated=eligible_centers=0; frag=[]
    for indices in tracks:
        v=fail[indices]; n=len(v)
        if n>1:
            c=int(np.count_nonzero(v[1:]!=v[:-1]));changes+=c;edges+=n-1;frag.append(c/(n-1))
        if n>2:
            eligible_centers+=n-2
            isolated+=int(np.count_nonzero(v[1:-1]&~v[:-2]&~v[2:]))
        pad=np.r_[False,v,False]; starts=np.where(~pad[:-1]&pad[1:])[0];ends=np.where(pad[:-1]&~pad[1:])[0]
        runs.extend((ends-starts).tolist())
    return {'GT_frames':sum(len(t) for t in tracks),'fail_frames':int(sum(np.count_nonzero(fail[t]) for t in tracks)),
            'contiguous_segments':len(tracks),'adjacent_changes':changes,'adjacent_pairs':edges,
            'frame_edge_fragment_rate':changes/edges if edges else None,'segment_equal_fragment_rate':float(np.mean(frag)) if frag else None,
            'eligible_triples':eligible_centers,'isolated_miss_frames':isolated,
            'failure_runs':len(runs),'failure_run_length_quantiles':np.quantile(runs,[.5,.9,.99]).tolist() if runs else [],
            'fail_frames_in_runs_ge5':sum(n for n in runs if n>=5)}

def main():
    spec={'status':'exploratory_CPU_description','scope':'six archived fixed36 EMA UAVDT val predictions; no new inference',
          'unit':'same sequence+official trackID, strictly consecutive frame IDs, split segments at gaps and at size640>=32',
          'states':['best_iou<.70: all-pool per-GT geometry absence, ignores competition','native_match_score_rank sentinel: all-300 native unmatched; no score cutoff, not AP'],
          'comparisons':'within seed baseline/C/Cmean, both frame edge and segment equal descriptions',
          'isolated_pair_flip':'center models disagree, immediately adjacent frames agree across models; geometry and native separately',
          'no_significance_or_causal_claim':True,'no_temporal_method_or_AP_gain_claim':True,
          'new_forwards':0,'new_training_epochs':0,'script_sha256':sha(Path(__file__))}
    (OUT/'08_CPU时序探索口径.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
    annpath=DATA/'独立中期评估_20261018_v1/data/val.json'; gt=json.loads(annpath.read_text(encoding='utf-8-sig'))
    images={x['id']:x for x in gt['images']}; anns={x['id']:x for x in gt['annotations']}
    archive=DATA/'原始下载/UAVDT/UAV-benchmark-MOTD_v1.0.zip'
    raw={}
    with zipfile.ZipFile(archive) as z:
        readme=z.read('UAV-benchmark-MOTD_v1.0/README.md').decode('utf8',errors='replace')
        assert '<target_id>' in readme and 'temporal corresponding' in readme
        for seq in sorted({x['uav_sequence'] for x in images.values()}):
            for line in z.read(f'UAV-benchmark-MOTD_v1.0/GT/{seq}_gt_whole.txt').decode('utf8').splitlines():
                a=list(map(float,line.split(',')));key=(seq,int(a[0]),int(a[1])); assert key not in raw;raw[key]=a
    folder=DATA/'第36轮候选覆盖与排序筛查_20261019_v1/models'
    models={};sources={str(annpath.relative_to(ROOT)):sha(annpath)}
    for seed in [0,1]:
        for arm in ['baseline','c','c_mean']:
            key=f'seed{seed}_{arm}';p=folder/key/'per_GT_geometry_and_ranks.npz';sources[str(p.relative_to(ROOT))]=sha(p)
            with np.load(p,allow_pickle=False) as z:models[key]={k:z[k] for k in z.files}
    base=models['seed0_baseline'];ids=base['annotation_ids'];N=len(ids);assert N==87719
    groups=defaultdict(list);raw_deltas=[]
    for i,aid in enumerate(ids):
        a=anns[int(aid)];im=images[a['image_id']];seq,frame,tid=im['uav_sequence'],im['uav_frame'],a['uav_track_id']
        assert int(base['image_ids'][i])==a['image_id'];rr=raw[(seq,frame,tid)]
        raw_deltas.append(np.asarray(a['bbox'])-np.asarray(rr[2:6]))
        groups[(seq,tid)].append((frame,i))
    delta=np.asarray(raw_deltas);assert np.max(np.abs(delta[:,2:]))==0
    # Converter may subtract a fixed pixel origin offset; track identity and box sizes must agree.
    assert len(np.unique(delta[:,:2],axis=0))==1
    segments=[];seq_segments=defaultdict(list);frame_intervals=[]
    for (seq,tid),items in groups.items():
        items.sort();assert len({x[0] for x in items})==len(items)
        frame_intervals.extend(np.diff([x[0] for x in items]).tolist());seg=[];prev=None
        for frame,i in items:
            small=base['size640'][i]<32
            if (prev is not None and frame!=prev+1) or not small:
                if seg:segments.append(np.array(seg));seq_segments[seq].append(np.array(seg));seg=[]
            if small:seg.append(i)
            prev=frame
        if seg:segments.append(np.array(seg));seq_segments[seq].append(np.array(seg))
    result={'spec':spec,'source_sha256':sources,'mapping':{'GT_frames':N,'official_tracks':len(groups),'annotation_to_raw_bbox_delta':delta[0].tolist(),'frame_intervals':dict(zip(*[x.tolist() for x in np.unique(frame_intervals,return_counts=True)]))},'models':{},'pair_flips':{}}
    for key,m in models.items():
        assert np.array_equal(m['annotation_ids'],ids);assert np.array_equal(m['image_ids'],base['image_ids']);assert np.array_equal(m['size640'],base['size640'])
        ix=int(np.flatnonzero(np.isclose(m['thresholds'],.7))[0]);sentinel=int(m['absent_rank_sentinel'])
        states={'geometry':m['best_iou']<.7,'native':m['native_match_score_rank'][ix]>=sentinel}
        result['models'][key]={'summary':{},'per_sequence':{}}
        for label,fail in states.items():
            result['models'][key]['summary'][label]=describe(fail,segments)
            result['models'][key]['per_sequence'][label]={seq:describe(fail,ts) for seq,ts in seq_segments.items()}
        m['_states']=states
    for seed in [0,1]:
        for other in ['c','c_mean']:
            key=f'seed{seed}_{other}-baseline';result['pair_flips'][key]={}
            for label in ['geometry','native']:
                b=models[f'seed{seed}_baseline']['_states'][label];c=models[f'seed{seed}_{other}']['_states'][label]
                total=eligible=isolated=extra=rescued=0
                for indices in segments:
                    d=c[indices]!=b[indices];total+=int(np.count_nonzero(d));extra+=int(np.count_nonzero(c[indices]&~b[indices]));rescued+=int(np.count_nonzero(b[indices]&~c[indices]))
                    if len(indices)>2:
                        eligible+=int(np.count_nonzero(d[1:-1]));isolated+=int(np.count_nonzero(d[1:-1]&~d[:-2]&~d[2:]))
                result['pair_flips'][key][label]={'all_disagree_frames':total,'extra_fail':extra,'rescued_fail':rescued,'eligible_disagree_centers':eligible,'isolated_disagree_centers':isolated,'isolated_fraction_of_eligible_disagreements':isolated/eligible if eligible else None}
    (OUT/'09_CPU现有UAVDT36_轨迹描述.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'mapping':result['mapping'],'model_summaries':{k:v['summary'] for k,v in result['models'].items()},'pair_flips':result['pair_flips']},ensure_ascii=False))

if __name__=='__main__':main()
