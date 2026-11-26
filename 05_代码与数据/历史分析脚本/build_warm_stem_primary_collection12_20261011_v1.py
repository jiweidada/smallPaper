from pathlib import Path
import ast,difflib,hashlib,json
root=Path('英文小论文/code/warm_stem_primary_collection12_20261011_v1');root.mkdir(exist_ok=False)
old=Path('英文小论文/code/stem_component_primary_collection_20261011_v2/collect.py')
s=old.read_text(encoding='utf8')
replacements=[
("assert arm in ('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')","assert arm=='warm_stem'"),
("rootname='stem_component12_20261011_v1'","rootname='warm_stem12_20261011_v1'"),
("protocol['component_files_sha256']","protocol['warm_files_sha256']"),
("output=base/'stem_component_primary_collection_20261011_v2'","output=base/'warm_stem_primary_collection12_20261011_v1'"),
("'baseline_reuse.json','preflight.json'","'baseline_reuse.json','detector_preflight_start.json','detector_preflight_completion.json','training_resource_receipt.json'"),
("schema='exact12_component_primary_artifact_collection_v2'","schema='exact12_warm_primary_artifact_collection_v1'"),
("choices=('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')","choices=('warm_stem',)")
]
for before,after in replacements:
 assert s.count(before)==1,before
 s=s.replace(before,after)
before="    for label,p in pairs.items():"
after="""    for p in (screen/'detector_preflight_v1').rglob('*'):
        if p.is_file():pairs['screen/detector_preflight_v1/'+p.relative_to(screen/'detector_preflight_v1').as_posix()]=p
    for key in ('image_manifest','image_verification'):
        pairs['provenance/'+key+'.json']=paths[key]
    for label,p in pairs.items():"""
assert s.count(before)==1;s=s.replace(before,after)
before="    a=p.parse_args();print(json.dumps(collect(a.prefix,a.seed,a.arm)),flush=True)"
after="""    a=p.parse_args()
    assert a.prefix=={1:'/25085400120',2:'/24085403014'}[a.seed]
    collection_root=Path(__file__).resolve().parent.parent
    start=collection_root/f'seed{a.seed}_collect_start.json'
    done=collection_root/f'seed{a.seed}_collect_completion.json'
    failure=collection_root/f'seed{a.seed}_collect_failure.json'
    assert not any(p.exists() for p in (start,done,failure)), 'Inspect existing attempt; no automatic retry'
    import os,traceback
    receipt=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        seed=a.seed,source_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).parent/'protocol.json'),
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False,
        new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
    write(start,receipt)
    try:
        metadata=collect(a.prefix,a.seed,a.arm)
        result=dict(status='STOPPED_WARM_PRIMARY12_PACKAGE_COLLECTED',exit_code=0,receipt=receipt,metadata=metadata,
                    automatic_retry=False,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
        write(done,result);print(json.dumps(result),flush=True)
    except BaseException:
        write(failure,dict(status='READ_ONLY_COLLECTION_FAILURE_NO_RETRY',receipt=receipt,traceback=traceback.format_exc()))
        raise"""
assert s.count(before)==1;s=s.replace(before,after)
ast.parse(s);(root/'collect.py').write_text(s,encoding='utf8',newline='\n')
expected=json.loads(Path('outputs/warm_stem12_20261011_v1/local_code_manifest.json').read_bytes())
protocol=dict(schema='warm_stem_primary_collection12_v1',collector_sha256=hashlib.sha256((root/'collect.py').read_bytes()).hexdigest(),
 warm_files_sha256=expected,permitted_seeds=[1,2],permitted_arm='warm_stem',
 independent_saved_EMA_replay_included=False,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,
 upstream_collector_sha256=hashlib.sha256(old.read_bytes()).hexdigest())
(root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8',newline='\n')
(root/'adaptation.diff').write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf8').splitlines(True),s.splitlines(True),fromfile=str(old),tofile=str(root/'collect.py'))),encoding='utf8')
print(json.dumps({'status':'READ_ONLY_WARM_COLLECTOR_PREPARED','files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('collect.py','protocol.json')},
'scope':'Existing candidate artifacts only; original frozen collectors and training code remain unchanged.'}))
