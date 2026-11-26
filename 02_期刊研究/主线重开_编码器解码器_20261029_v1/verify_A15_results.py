"""Verify archive lineage and independently recompute fixed-pair geometry (no Torch)."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib
import json
import math
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'A15_固定query的采样中心干预_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (RUN/'independent_verification.json').exists(), 'Inspect completed verification instead of rerunning.'
r=read(RUN/'backup'/'receipt.json')
archive=RUN/'backup'/'A15_sampling_intervention_results.tar.gz'
assert archive.stat().st_size==r['bytes'] and sha(archive)==r['sha256']
with tarfile.open(archive,'r:gz') as tf:
    assert len(tf.getmembers())==r['files']
    assert len({x.name for x in tf.getmembers()})==r['files']
    for member in tf.getmembers():
        rel=PurePosixPath(member.name)
        assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and '\\' not in member.name
        dest=RUN.joinpath(*rel.parts);assert dest.resolve().is_relative_to(RUN.resolve())
        payload=tf.extractfile(member).read()
        if dest.exists():assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(payload).digest(),member.name
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
done=read(RUN/'completed.json')
for f in done['files']:
    path=RUN/f['path'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
p=read(RUN/'protocol.json');summ=read(RUN/'summary.json');analysis=read(RUN/'analysis_summary.json')
assert summ['new_full_model_forwards']==13 and summ['last_layer_suffix_forwards']==72
assert summ['state_digest_before']==summ['state_digest_after']==p['state_digest']
assert read(RUN/'process'/'a15_sampling_center_v1_exit.json')['exit_code']==0
calls=[json.loads(x) for x in (RUN/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
assert sum(x['kind']=='LAST_LAYER_SUFFIX' for x in calls)==72
assert sum(x['kind']!='LAST_LAYER_SUFFIX' for x in calls)==13
assert len({(x['case'],x['arm']) for x in calls if x['kind']=='LAST_LAYER_SUFFIX'})==72

def scalar_iou(b,g):
    cx,cy,w,h=map(float,b)
    left=(cx-.5*w)*800;right=(cx+.5*w)*800
    top=(cy-.5*h)*800;bottom=(cy+.5*h)*800
    ix=max(0.,min(right,float(g[2]))-max(left,float(g[0])))
    iy=max(0.,min(bottom,float(g[3]))-max(top,float(g[1])))
    union=max(0.,right-left)*max(0.,bottom-top)+max(0.,float(g[2]-g[0]))*max(0.,float(g[3]-g[1]))-ix*iy
    return ix*iy/max(union,1e-30)

saved={(row['case'],row['gt']):row['ious'] for row in read(RUN/'per_GT_analysis.json')}
effects=[];totals=[0]*6;checked=0;untouched=0
prefixes=[]
for case in range(12):
    with np.load(RUN/'predictions'/('case%02d.npz'%case),allow_pickle=False) as z, \
         np.load(RUN/'captures'/('case%02d.npz'%case),allow_pickle=False) as c:
        tiny=np.flatnonzero(z['gt_sizes800']<8);mapping=dict(zip(z['match_gt'].tolist(),z['match_q'].tolist()))
        assert np.array_equal(z['arm_boxes'][0],z['raw_boxes'])
        assert np.array_equal(z['arm_logits'][0],z['raw_logits'])
        qidx=np.asarray([mapping[j] for j in tiny if j in mapping])
        assert np.array_equal(np.sort(qidx),np.sort(c['gt_query_indices']))
        gt_for_q={mapping[j]:j for j in tiny if j in mapping}
        for q,centre in zip(c['gt_query_indices'],c['gt_centres']):
            g=z['gt_boxes800'][gt_for_q[int(q)]]
            assert np.allclose(centre,(g[:2]+g[2:])/(2*800),atol=1e-8,rtol=0)
        keep=np.ones(z['raw_boxes'].shape[0],dtype=bool);keep[qidx]=False
        assert np.array_equal(z['arm_boxes'][0,keep],z['arm_boxes'][1,keep])
        assert np.array_equal(z['arm_logits'][0,keep],z['arm_logits'][1,keep]);untouched+=int(keep.sum())
        counts=[0]*6;means=[0.]*6
        for j in tiny:
            q=mapping.get(int(j));values=[]
            g=np.asarray(z['gt_boxes800'][j],dtype=np.float64)
            for a in range(6):
                v=scalar_iou(z['arm_boxes'][a,q],g) if q is not None else 0.
                assert abs(v-saved[(case,int(j))][a])<2e-14
                values.append(v);counts[a]+=int(v>=.75);means[a]+=v;checked+=1
            assert all(math.isfinite(v) for v in values)
        n=len(tiny);fractions=[v/n for v in counts]
        effects.append({'sample75':fractions[1]-fractions[0],
            'content_interaction75':(fractions[1]-fractions[0])-(fractions[5]-fractions[4]),
            'position75':fractions[2]-fractions[0],'both75':fractions[3]-fractions[0],
            'sample_mean_iou':(means[1]-means[0])/n})
        totals=[a+b for a,b in zip(totals,counts)];prefixes.append(str(z['prefix'].item()))
assert checked==1578 and len(set(prefixes))==12 and totals==analysis['tiny_iou75_counts']

def quantile(a,q):
    a=sorted(a);t=(len(a)-1)*q;lo=math.floor(t);hi=math.ceil(t)
    return a[lo]+(a[hi]-a[lo])*(t-lo)

plan=np.random.default_rng(p['bootstrap_seed']).integers(0,12,size=(2000,12))
independent={}
for name in effects[0]:
    values=[x[name] for x in effects]
    sampled=[sum(values[int(k)] for k in row)/12 for row in plan]
    stat={'mean':sum(values)/12,'low':quantile(sampled,.025),'high':quantile(sampled,.975)}
    assert all(abs(stat[k]-analysis['image_primary_effects'][name][k])<1e-14 for k in stat)
    independent[name]=stat
assert analysis['gate_passed'] is False
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_ARCHIVE_METRICS_AND_SCOPED_INVARIANTS',
    'archive_sha256':r['sha256'],'archive_members_verified':r['files'],'completion_files_verified':len(done['files']),
    'independent_scalar_ious':checked,'independent_bootstrap_effects':independent,
    'unchanged_nonintervened_queries_checked':untouched,'tiny_iou75_counts':totals,
    'full_model_forwards':13,'last_layer_suffix_forwards':72,'cumulative_full_model_forwards':226,
    'actual_process_exit_code':0,'new_full_model_forwards_for_verification':0,'new_fits':0,
    'limits':['Arithmetic/lineage verification,notsource-authorreproductionorindependentdata.',
        'Cross-queryandstateidentityexecutionassertionsarenotindependentlyrerun;frozencodeandreportedgatesinspected.',
        'NoAP,causalnatural-failureattribution,newalgorithmortrainingfeasibilityproved.']}
write(RUN/'independent_verification.json',receipt)
print(json.dumps({k:receipt[k] for k in ['status','archive_members_verified','independent_scalar_ious',
    'unchanged_nonintervened_queries_checked','tiny_iou75_counts','cumulative_full_model_forwards']}))
