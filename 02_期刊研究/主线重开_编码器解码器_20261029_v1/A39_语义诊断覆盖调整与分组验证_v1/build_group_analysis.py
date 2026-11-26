from pathlib import Path
import ast,hashlib,json,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;OLD=D.parent/'A38_细尺度与查询语义成对诊断_v1'
source=(OLD/'analyze_features.py').read_text(encoding='utf-8');new=source
def replace(a,b):
    global new
    assert new.count(a)==1,(a,new.count(a));new=new.replace(a,b)
replace("def save(name,value):\n    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)",
"""def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def save(name,value):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,default=scalar)""")
replace("assert collection['status']=='COMPLETE_114_NATIVE_FROZEN_ENTRIES_NO_OPTIMIZER'","assert collection['status']=='COMPLETE_113_NEW_NATIVE_ENTRIES_WITH_FULL_FEATURE_REPEAT'")
replace("assert collection['entries']==collection['completed']==114","assert collection['entries']==collection['completed']==113 and collection['all_first_new_case_arrays_bitwise_repeat_equal']")
replace("first=dict(np.load(E/'captures/case_000.npz',allow_pickle=False))",
"first_new=read(P/'collector_inputs.json')['cases'][0]\nfirst=dict(np.load(E/('captures/case_%03d.npz'%first_new['index']),allow_pickle=False))")
replace("cases={s:[] for s in ['train','val']}; unsupported={s:[] for s in ['train','val']}",
"cases={s:[] for s in ['train','val']}; unsupported={s:[] for s in ['train','val']}; groups={s:[] for s in ['train','val']}")
replace("inputs=read(P/'collector_inputs.json')","inputs=read(P/'combined_cases_FROZEN.json')")
replace("    c=dict(np.load(E/('captures/case_%03d.npz'%row['index']),allow_pickle=False))\n    assert int(c['index'])==row['index'] and c['file'].item()==row['file'] and c['split'].item()==row['split']",
"""    path=Path(row['capture_path']) if row['capture_phase']=='A38' else E/row['capture_relative']
    if row['capture_phase']=='A38':assert sha(path)==row['capture_sha256']
    c=dict(np.load(path,allow_pickle=False))
    assert int(c['index'])==row['capture_index'] and c['file'].item()==row['file'] and c['split'].item()==row['split']""")
replace("    cases[s].extend([row['index']]*(2*m))","    cases[s].extend([row['index']]*(2*m)); groups[s].extend([row['prefix']]*(2*m))")
replace("    cases[s]=np.array(cases[s],dtype=np.int64);unsupported[s]=np.array(unsupported[s],dtype=bool)",
"    cases[s]=np.array(cases[s],dtype=np.int64);unsupported[s]=np.array(unsupported[s],dtype=bool);groups[s]=np.array(groups[s],dtype=str)")
replace("    val_images_with_primary_pairs=len(np.unique(cases['val'][primary])))",
"    val_images_with_primary_pairs=len(np.unique(cases['val'][primary])),train_prefixes_with_pairs=len(np.unique(groups['train'])),val_prefixes_with_primary_pairs=len(np.unique(groups['val'][primary])))")
replace("passed=(train_pairs>=200 and primary_pairs>=100 and eligibility['train_images_with_pairs']>=32\n        and eligibility['val_images_with_primary_pairs']>=20)",
"passed=(train_pairs>=200 and primary_pairs>=100 and eligibility['train_prefixes_with_pairs']>=32\n        and eligibility['val_prefixes_with_primary_pairs']>=20)")
replace("val_ids=np.array([r['index'] for r in inputs['cases'] if r['split']=='val'],dtype=np.int64)\ndraws=np.load(P/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws']\nassert draws.shape==(1000,48)\nindices={int(i):np.flatnonzero((cases['val']==i)&primary) for i in val_ids}",
"""group_order=read(P/'bootstrap_groups_FROZEN.json')['groups']
draws=np.load(P/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws']
assert draws.shape==(1000,64) and len(group_order)==64
indices={g:np.flatnonzero((groups['val']==g)&primary) for g in group_order}""")
replace("    idx=np.concatenate([indices[int(val_ids[i])] for i in draw])","    idx=np.concatenate([indices[group_order[int(i)]] for i in draw])")
replace("labels=labels['val'],cases=cases['val'],primary_mask=primary,","labels=labels['val'],cases=cases['val'],groups=groups['val'],primary_mask=primary,")
replace("    inference_limit='Accessibility under fixed linear/RFF probes only.",
"    bootstrap_unit='64filename-prefixproxygroupswithallframesandliteralrepeats',\n    inference_limit='Accessibility under fixed linear/RFF probes only.")
new=new.replace("json.dumps(dict(status='INSUFFICIENT_MATCHED_SUPPORT',eligibility=eligibility))","json.dumps(dict(status='INSUFFICIENT_MATCHED_SUPPORT',eligibility=eligibility),default=scalar)")
replace("fine_minus_query_AUC=result['fine_minus_query_AUC'],CI95=ci,gates=gates,probe_fit_entries=fits),ensure_ascii=False))",
"fine_minus_query_AUC=result['fine_minus_query_AUC'],CI95=ci,gates=gates,probe_fit_entries=fits),ensure_ascii=False,default=scalar))")
ast.parse(new,feature_version=(3,9))
old_fit=ast.get_source_segment(source,next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fit'))
new_fit=ast.get_source_segment(new,next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='fit'))
assert old_fit==new_fit
for marker in ["fine_auc_at_least_point75=point['FINE']>=.75", "fine_minus_query_at_least_point05=point['FINE']-point['QUERY']>=.05", "fine_minus_query_RFF_at_least_point02=point['FINE']-point['QUERY_RFF']>=.02", "fine_minus_geometry_at_least_point05=point['FINE']-point['GEOMETRY']>=.05"]:
    assert marker in source and marker in new
with (D/'analyze_features.py').open('x',encoding='utf-8') as f:f.write(new)
with (D/'group_analysis_preservation.json').open('x',encoding='utf-8') as f:
    json.dump(dict(status='PRIMARY_METRIC_FITTER_AND_THRESHOLDS_RETAINED_GROUP_BOOTSTRAP_ADAPTED',
        frozen_A38_analysis_sha256=hashlib.sha256(source.encode()).hexdigest(),new_analysis_sha256=hashlib.sha256(new.encode()).hexdigest(),
        numerical_ridge_function_unchanged=True,primary_gates_unchanged=True,
        changes=['Combined224cached/newcases','Training119/validation105frames','64prefixgroupbootstrap','Groupcountsforcoverage','PredefinedNumPyscalarJSONexport'],
        new_data_probe_fits=0),f,indent=2)
print(json.dumps(dict(status='A39_GROUP_ANALYSIS_CREATED_STATIC_ONLY',data_probe_fit_cap=7,bootstrap_groups=64)))
