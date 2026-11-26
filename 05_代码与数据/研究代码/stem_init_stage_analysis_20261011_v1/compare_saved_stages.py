"""Fixed12 saved-tensor description; no network/evaluator or causal root claims."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'protocol.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepared():
    p=read(PROTOCOL)
    assert p['schema']=='stem_init_stage_analysis_v1'
    for n,h in p['analysis_files_sha256'].items():assert sha(HERE/n)==h,n
    for n,h in p['dependency_files_sha256'].items():assert sha(ROOT/n)==h,n
    old=ROOT/'英文小论文/code/scale_protection12_monitor_20261010_v1'
    geometry=module('stem_init_geometry',old/'geometry_core.py');sys.modules['geometry_core']=geometry
    onset=module('stem_init_onset',old/'analyze_stages.py').onset
    rank=module('stem_init_rank',HERE/'ranking_core.py')
    readers={}
    for kind,path in p['readers'].items():
        r=module('stem_init_reader_'+kind,ROOT/path/'verify_terminal_local.py');r.local_protocol();readers[kind]=r
    return p,geometry,onset,rank,readers

def sorted_flags(flags):
    order=np.argsort(flags['annotation_id'])
    return {k:flags[k][order] for k in ('annotation_id','category_id','hits')}

def groups(annotation,values):
    anns={int(a['id']):a for a in annotation['annotations']};images={int(i['id']):i for i in annotation['images']}
    side=np.array([min(anns[int(g)]['bbox'][2]*640/images[int(anns[int(g)]['image_id'])]['width'],
                       anns[int(g)]['bbox'][3]*640/images[int(anns[int(g)]['image_id'])]['height']) for g in values['annotation_id']])
    result=dict(all=np.ones(len(side),dtype=bool),COCO_small=values['original_area']<=1024,
        input_under4=side<4,input_4_to_under8=(side>=4)&(side<8),
        input_8_to_under16=(side>=8)&(side<16),input_ge16=side>=16)
    assert len(side)==38759 and result['COCO_small'].sum()==26586 and result['input_under4'].sum()==5096
    return result

def cached_baseline(seed,p,reader):
    ev=reader.local_protocol()['reused_local_evidence'][str(seed)]
    packet=ROOT/ev['packet_relative_path'];report=ROOT/ev['report_relative_path']
    assert sha(report)==ev['report_sha256'] and sha(packet/'manifest.json')==ev['manifest_sha256']
    reader.reuse_sources(read(report)['identity'],reader.local_protocol())
    annotation=read(packet/'reference/val_annotations.json')
    summary,flags,_=reader.native_endpoint(packet/'baseline_eval/endpoint_epoch12',annotation)
    assert summary['identity']['arm']=='baseline' and summary['identity']['seed']==seed
    stage=ROOT/p['baseline_caches'][str(seed)]['stage'];rank=ROOT/p['baseline_caches'][str(seed)]['rank']
    for key,folder in (('stage',stage),('rank',rank)):
        assert sha(folder/'manifest.json')==p['baseline_caches'][str(seed)][key+'_manifest_sha256']
        for n,h in read(folder/'manifest.json')['files'].items():assert sha(folder/n)==h,n
    sr,rr=read(stage/'report.json'),read(rank/'report.json')
    assert sr['provenance']['baseline']['stage_manifest_sha256']==sha(packet/'baseline_eval/stages_epoch12/manifest.json')
    assert rr['stage_manifest_sha256']==sr['provenance']['baseline']['stage_manifest_sha256']
    assert rr['identity']==summary['identity'] and not rr['score_mismatch_images'] and rr['native_TP_proxy_conflicts']==0
    with np.load(stage/'baseline_per_GT.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    f=sorted_flags(flags)
    assert np.array_equal(values['annotation_id'],f['annotation_id']) and np.array_equal(values['category_id'],f['category_id'])
    assert np.array_equal(values['native_TP'],f['hits'])
    return dict(packet=packet,annotation=annotation,summary=summary,values=values,rank_report=rr,
        original_stage_report=sr,source_report_sha256=sha(report),cache_stage_manifest_sha256=sha(stage/'manifest.json'),
        cache_rank_manifest_sha256=sha(rank/'manifest.json'))

def inspect(item,p,geometry,onset,rank,readers,base,output):
    kind=item['kind'];reader=readers[kind];protocol=reader.local_protocol();report=read(item['report'])
    assert report['status']==reader.PASS and report['fixed_endpoint']==protocol['endpoint']
    assert report['verification_script_sha256']==protocol['reader_files_sha256']['verify_terminal_local.py']
    assert report['local_reader_protocol_sha256']==sha(reader.PROTOCOL)
    assert report['identity']['implementation']==protocol['implementation_sha256']
    assert report['all548_stage_arrays_reloaded_exact'] and report['first100_inputs_and_pre_forward_RNG_exact']
    assert report['new_model_forwards']==report['new_evaluator_calls']==report['new_optimizer_updates']==report['new_training_epochs']==0
    identity=report['identity'];assert identity['seed']==item['seed']
    allowed={'full':['original_stem'],'reference':['native_stem_scratch'],'control':p['control_arms']}
    assert identity['arm'] in allowed[kind]
    packet=Path(item['packet']);assert sha(packet/'terminal_manifest.json')==report['terminal_manifest_sha256']
    if kind in ('reference','control'):assert packet.resolve()==Path(report['downloaded_packet']).resolve()
    annotation=base['annotation'];summary,flags,_=reader.native_endpoint(packet/'run/endpoint_epoch12',annotation)
    assert summary['identity']==identity and summary['overall_percent']==report['candidate_overall_percent']
    for key in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
        assert identity[key]==base['summary']['identity'][key],key
    reader.stages(packet/'run',identity,annotation)
    def image(iid):
        with np.load(packet/f'run/stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    values=geometry.collect(annotation,image,flags)
    label=identity['arm']+'_seed'+str(identity['seed']);dest=output/label;dest.mkdir()
    np.savez_compressed(dest/'per_GT.npz',**values)
    counts={name:{str(t):onset(values,mask,t,ti) for t,ti in ((.5,0),(.75,5))}
            for name,mask in groups(annotation,values).items()}
    ranking=rank.analyze(packet,identity['arm'],HERE/'ranking_protocol.json',dest/'ranking',
        base['packet']/'native/engine/deim/postprocessor.py',base['packet']/'reference/val_annotations.json')
    rank_report=read(dest/'ranking/report.json')
    assert rank_report['identity']==identity
    with np.load(dest/'ranking/per_GT.npz',allow_pickle=False) as z:
        assert np.array_equal(z['annotation_id'],values['annotation_id'])
        assert np.array_equal(z['native_TP_50'],values['native_TP'][:,0])
        assert np.array_equal(z['native_TP_75'],values['native_TP'][:,5])
    description=dict(identity=identity,counts=counts,ranking=rank_report['groups'],
        score_mismatch_images=rank_report['score_mismatch_images'],ranking_ambiguous_images=rank_report['ambiguous_images'],
        native_TP_proxy_conflicts=rank_report['native_TP_proxy_conflicts'],
        AP_percent=summary['overall_percent'],report_sha256=sha(item['report']),
        stage_manifest_sha256=sha(packet/'run/stages_epoch12/manifest.json'),
        endpoint_manifest_sha256=sha(packet/'run/endpoint_epoch12/manifest.json'))
    write(dest/'description.json',description)
    return values,description

def pair(left,right,annotation,geometry):
    for n in ('annotation_id','image_id','category_id','original_area'):assert np.array_equal(left[n],right[n]),n
    out={}
    for name,mask in groups(annotation,left).items():
        row=geometry.pair_metrics(left,right,mask);row.pop('transition_annotation_ids')
        first={}
        for threshold,ti in ((.5,0),(.75,5)):
            changed=(left['max_iou']>=threshold)!=(right['max_iou']>=threshold)
            any_changed=changed.any(1);earliest=changed.argmax(1)
            first[str(threshold)]=dict(unchanged_support_pattern=int((mask&~any_changed).sum()),
                first_observed_support_change={stage:int((mask&any_changed&(earliest==i)).sum()) for i,stage in enumerate(geometry.STAGES)},
                new_native_TP=int((mask&~left['native_TP'][:,ti]&right['native_TP'][:,ti]).sum()),
                lost_native_TP=int((mask&left['native_TP'][:,ti]&~right['native_TP'][:,ti]).sum()))
            assert first[str(threshold)]['unchanged_support_pattern']+sum(first[str(threshold)]['first_observed_support_change'].values())==int(mask.sum())
        row['earliest_changed_geometric_boundary']=first;out[name]=row
    return out

def analyze(spec_path,output):
    p,geometry,onset,rank,readers=prepared();spec=read(spec_path);output=Path(output);assert not output.exists()
    jobs=spec['jobs'];required={(k,s) for k in ('full','reference') for s in (1,2)}
    assert {(x['kind'],x['seed']) for x in jobs if x['kind']!='control'}==required
    assert len([x for x in jobs if x['kind']!='control'])==4
    controls=[x for x in jobs if x['kind']=='control']
    if controls:assert len(controls)==8 and {(read(x['report'])['identity']['arm'],x['seed']) for x in controls}=={(a,s) for a in p['control_arms'] for s in (1,2)}
    # Reject missing/unverified terminal evidence before creating an analysis output.
    for x in jobs:assert read(x['report'])['status']==readers[x['kind']].PASS
    bases={s:cached_baseline(s,p,readers['full']) for s in (1,2)}
    output.mkdir();values={};descriptions={}
    for item in jobs:
        v,d=inspect(item,p,geometry,onset,rank,readers,bases[item['seed']],output)
        key=(d['identity']['arm'],item['seed']);assert key not in values;values[key]=v;descriptions[str(key)]=d
    comparisons={}
    for seed in (1,2):
        b=bases[seed];full=values[('original_stem',seed)];ref=values[('native_stem_scratch',seed)]
        comparisons[str(seed)]={'pretrained_BASE_to_full':pair(b['values'],full,b['annotation'],geometry),
            'pretrained_BASE_to_scratch_reference':pair(b['values'],ref,b['annotation'],geometry),
            'scratch_reference_to_full':pair(ref,full,b['annotation'],geometry)}
        for arm in p['control_arms']:
            if (arm,seed) in values:comparisons[str(seed)]['control_'+arm+'_to_full']=pair(values[(arm,seed)],full,b['annotation'],geometry)
    report=dict(status='SAVED_FIXED12_STAGE_AND_SELECTION_DESCRIPTIONS',comparisons=comparisons,descriptions=descriptions,
        baseline_sources={str(s):dict(source_report_sha256=b['source_report_sha256'],stage_cache_manifest_sha256=b['cache_stage_manifest_sha256'],rank_cache_manifest_sha256=b['cache_rank_manifest_sha256']) for s,b in bases.items()},
        spec_sha256=sha(spec_path),protocol_sha256=sha(PROTOCOL),causal_mechanism_proven=False,
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,scope=p['scope'])
    if any(d['native_TP_proxy_conflicts'] or d['score_mismatch_images'] for d in descriptions.values()):report['status']='SAVED_FIXED12_DESCRIPTION_NUMERICAL_REVIEW_REQUIRED'
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files={f.relative_to(output).as_posix():sha(f) for f in output.rglob('*') if f.is_file()}))
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--spec',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    print(analyze(args.spec,args.output)['status'])
