from pathlib import Path
import ast,json
D=Path(__file__).resolve().parent;OLD=D.parent/'A38_细尺度与查询语义成对诊断_v1'
source=(OLD/'verify_support_independent.py').read_text(encoding='utf-8');new=source
def replace(a,b):
    global new
    assert new.count(a)==1,(a,new.count(a));new=new.replace(a,b)
replace("records=read(A/'pair_records.json');source_sample=read(D.parent/'A37_局部差分编码与语义判别证据核查_v1/semantic_probe_sample_FROZEN.json')",
"records=read(A/'pair_records.json');source_sample=read(D/'combined_cases_FROZEN.json')")
replace('by_case={i:[] for i in range(112)}','by_case={i:[] for i in range(224)}')
replace("    i=row['index'];c=dict(np.load(E/('captures/case_%03d.npz'%i),allow_pickle=False))",
"    i=row['index'];path=Path(row['capture_path']) if row['capture_phase']=='A38' else E/row['capture_relative'];c=dict(np.load(path,allow_pickle=False));assert int(c['index'])==row['capture_index']")
replace("counts=Counter();train_images=set();val_images=set();max_head_error=0.;max_iou_error=0.",
"counts=Counter();train_images=set();val_images=set();train_prefixes=set();val_prefixes=set();max_head_error=0.;max_iou_error=0.")
replace("        if row['split']=='train':train_images.add(i)\n        elif not native:counts['primary_val_pairs']+=1;val_images.add(i)",
"        if row['split']=='train':train_images.add(i);train_prefixes.add(row['prefix'])\n        elif not native:counts['primary_val_pairs']+=1;val_images.add(i);val_prefixes.add(row['prefix'])")
start=new.index("assert counts['train_pairs']==290")
new=new[:start]+'''result=read(A/'semantic_analysis_result.json')
assert counts['train_pairs']==result['eligibility']['train_pairs']==508
assert counts['primary_val_pairs']==result['eligibility']['primary_val_pairs']==199
assert len(train_images)==result['eligibility']['train_images_with_pairs']==100
assert len(val_images)==result['eligibility']['val_images_with_primary_pairs']==60
assert len(train_prefixes)==60 and len(val_prefixes)==45
fits=[json.loads(x) for x in (A/'probe_fit_entries.jsonl').read_text().splitlines()]
assert len(fits)==result['probe_fit_entries']==7
out=dict(status='PASS_INDEPENDENT_MATCHING_SUPPORT_AND_NATIVE_HEAD_AUDIT',cases=224,
    rows_checked=len(records['rows']),train_pairs=counts['train_pairs'],all_val_pairs=counts['val_pairs'],primary_val_pairs=199,
    train_images_with_pairs=100,val_images_with_primary_pairs=60,train_prefixes_with_pairs=60,val_prefixes_with_primary_pairs=45,
    iou_reference_max_error=max_iou_error,all_head_reconstruction_max_error=max_head_error,
    independent_checks=['2Dcell-unionignorefractions','Allquerybackgroundexclusionincludingclass11','HopcroftKarpgeometryandbackgroundcardinality',
        'Everychosenpairlabel/visibility/scorebin/native-support','AllnativeTopNscoreprovenance','All224queryLinearheadreconstructions'],
    data_probe_fits_observed=7,verifier_new_probe_fits=0,new_author_model_calls=0,mainline_validated=False)
with (D/'independent_support_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
'''
ast.parse(new,feature_version=(3,9))
with (D/'verify_support_independent.py').open('x',encoding='utf-8') as f:f.write(new)
print(json.dumps(dict(status='INDEPENDENT_SUPPORT_VERIFIER_CREATED',new_probe_fits=0,new_model_calls=0)))
