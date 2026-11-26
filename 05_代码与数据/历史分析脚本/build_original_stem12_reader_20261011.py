from pathlib import Path
import ast,json,hashlib
ROOT=Path.cwd()
SOURCE=ROOT/'英文小论文/code/support_matching12_monitor_20261011_v1'
OUT=ROOT/'英文小论文/code/original_stem12_monitor_20261011_v1'
OUT.mkdir(exist_ok=True)
def save(n,s): (OUT/n).write_text(s,encoding='utf8',newline='\n')
def change(s,a,b):
    assert a in s,a[:120]
    return s.replace(a,b)
def common(s):
    return (s.replace('support_matching12_monitor_20261011_v1','original_stem12_monitor_20261011_v1')
        .replace('support_matching12_20261011_v1','original_stem12_20261011_v1')
        .replace('support_matching12_','original_stem12_').replace('support_cost_fixed12','original_stem_fixed12')
        .replace('SUPPORT_COST','ORIGINAL_STEM').replace('support-cost','original-Stem')
        .replace('support12_','stem12_').replace("'support_cost'","'original_stem'")
        .replace('FDR_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED','STEM_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED'))
s=common((SOURCE/'verify_terminal_local.py').read_text())
s=change(s,"    assert identity['matching_coefficient'] == plan['matching_coefficient'] == 2.0\n",'')
s=s.replace("identity['added_parameter_count'] == 0","identity['added_parameter_count'] == 5325")
s=s.replace("plan['extra_parameters'] == 0","plan['extra_parameters'] == 5325")
s=change(s,"    cfg.update(resume=None, tuning=paths['common_init']", "    cfg['model']='OriginalStemDEIM'\n    cfg['OriginalStemDEIM']=cfg.pop('ResidualGeometryDEIM')\n    cfg.update(resume=None, tuning=paths['common_init']")
a=s.index('def training_snapshots(run):');b=s.index('\n\ndef verify(',a)
s=s[:a]+'''def training_snapshots(run):
    expected={(e,a) for e in (0,4,8,11) for a in (1,101,501,1001,1501)}
    records=rows(run/'sampled_training_target_sizes.jsonl')
    assert len(records)==20
    seen=set()
    for r in records:
        key=(r['epoch_zero_based'],r['epoch_attempt'])
        assert key in expected and key not in seen;seen.add(key)
        assert r['samples_shape'][:2]==[4,3]
        assert len(r['target_histogram'])==6 and all(type(n) is int and n>=0 for n in r['target_histogram'])
        assert sum(r['target_histogram'])==r['target_exposures']
        assert r['bands']==['under2','2_to_under4','4_to_under8','8_to_under16','16_to_under32','ge32']
    assert seen==expected
    return dict(scheduled_size_histograms=20,structural_count_checks_passed=True,
        scope='Passive target-size counts, not evidence for a causal mechanism.')
'''+s[b:]
a=s.index("    assert runtime['initialization']['parameters']");b=s.index("    assert runtime['resume']",a)
s=s[:a]+'''    init=runtime['initialization']
    assert init['parameters']==10237322 and init['extra_parameters']==5325
    assert init['exact_shared_tensors']==754 and init['state_tensors']==831
    assert init['old_stem_parameters']==6474 and init['new_stem_parameters']==11799
    assert init['only_missing_prefix']=='backbone.stem.' and init['common_init_sha256']==identity['common_init_sha256']
    assert runtime['optimizer_coverage']==dict(trainable_tensors=510,all_native_parameter_groups_unchanged=True,
        all_parameters_exactly_once=True,added_parameters=5325)
    assert runtime['stem_source']==read(CORE/'original_stem_source.json')
    assert sha(CORE/'original_stem.py')==native['original_stem_source_sha256']
    assert runtime['added_parameters']==5325 and 'matching_cost' not in runtime
'''+s[b:]
s=s.replace("row['FDR_matching_coefficient'] == 2.0", "row['original_stem_only'] is True")
s=s.replace("row['no_new_model_parameters'] is True", "row['no_new_model_parameters'] is False and row['added_parameters']==5325")
s=s.replace("proof['strict_loaded_tensors'] == 794 and proof['parameters'] == 10231997", "proof['strict_loaded_tensors'] == 831 and proof['parameters'] == 10237322")
save('verify_terminal_local.py',s)
s=common((SOURCE/'decide_fixed12.py').read_text())
s=s.replace(" and r['identity']['matching_coefficient'] == 2.0",'')
s=s.replace('this specific bounded support-cost configuration','the user original Stem-only configuration')
save('decide_fixed12.py',s)
s=common((SOURCE/'validate_cpu.py').read_text())
save('validate_cpu.py',s)
save('register_protocol.py','''"""Bind local result readers to the immutable Stem experiment."""
import ast,sys
from verify_terminal_local import CORE,HERE,ROOT,PROTOCOL,read,sha,write
assert not PROTOCOL.exists()
files=read(ROOT/'outputs/original_stem12_20261011_v1/local_code_manifest.json')
assert len(files)==23 and all(sha(CORE/n)==h for n,h in files.items())
tree=ast.parse((CORE/'runtime.py').read_text(encoding='utf8'))
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IMPLEMENTATION' for t in n.targets))
assert set(ast.literal_eval(node.value))==set(files)
old=read(ROOT/'英文小论文/code/support_matching12_monitor_20261011_v1/local_reader_protocol.json')
names=('verify_terminal_local.py','decide_fixed12.py','register_protocol.py','validate_cpu.py','compare_live_history.py')
for n in names:ast.parse((HERE/n).read_text(encoding='utf8'))
p=dict(schema='original_stem_fixed12_local_reader_v1',endpoint=old['endpoint'],acceptance=old['acceptance'],
    core_files_sha256=files,implementation_sha256=files,reader_files_sha256={n:sha(HERE/n) for n in names},
    reused_local_evidence=old['reused_local_evidence'],common_init_relative_path=old['common_init_relative_path'],
    prepared_before_new_terminal_file_consumption=True,new_stem_terminal_packets_validated=0,
    new_model_forwards=0,new_training_epochs=0,source_reader_sha256=sha(ROOT/'英文小论文/code/support_matching12_monitor_20261011_v1/verify_terminal_local.py'),
    notes=['Only backbone.stem is replaced; native matching, criterion and augmentation remain.',
        'Shared initialization outside Stem verified; the original Stem uses seed-specific default initialization.',
        'Positive fixed12 results permit mechanism followup but do not themselves explain which Stem branch caused improvement.'])
assert 'torch' not in sys.modules
write(PROTOCOL,p);print('ORIGINAL_STEM_READER_FROZEN',sha(PROTOCOL))
''')
save('compare_live_history.py','''"""Compare completed log epochs to existing same-seed baseline logs, without models."""
import argparse,json,sys
from pathlib import Path
from verify_terminal_local import ROOT,CORE,local_protocol,read,sha,write
p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
o=read(a.observation);seed=o['seed'];assert seed in (1,2) and not o['read_errors']
protocol=local_protocol();ev=protocol['reused_local_evidence'][str(seed)]
old=ROOT/ev['packet_relative_path'];report=ROOT/ev['report_relative_path']
assert sha(report)==ev['report_sha256'] and sha(old/'manifest.json')==ev['manifest_sha256']
assert read(report)['identity']['seed']==seed
item=read(old/'manifest.json')['files']['reference/baseline_log.txt'];path=old/'reference/baseline_log.txt'
assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
baseline=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x]
assert [r['epoch'] for r in baseline]==list(range(24))
assert o['protocol_sha256']==sha(CORE/'training_protocol.json')
candidate=o['metrics'];assert [r['epoch'] for r in candidate]==list(range(len(candidate))) and len(candidate)<=12
rows=[]
for r in candidate:
    e=r['epoch'];b=baseline[e]['test_coco_eval_bbox'];c=r['test_coco_eval_bbox'];assert len(b)==len(c)==12
    rows.append(dict(epoch_one_based=e+1,baseline_AP=100*b[0],candidate_AP=100*c[0],delta_AP_pp=100*(c[0]-b[0]),
        baseline_APs=100*b[3],candidate_APs=100*c[3],delta_APs_pp=100*(c[3]-b[3])))
result=dict(status='PROVISIONAL_SAME_EPOCH_NATIVE_LOG_COMPARISON',seed=seed,rows=rows,observed_remote_UTC=o['observed_remote_UTC'],
    historical_log_sha256=sha(path),observation_sha256=sha(a.observation),completed_epochs=len(candidate),
    fixed12_saved_EMA_verification_passed=False,new_model_forwards=0,new_training_commands=0,
    interpretation='Same seed/epoch log comparison only. Final decision requires fixed12 saved-EMA verification; early positive values are not an efficacy conclusion.')
assert 'torch' not in sys.modules
write(a.output,result);print(json.dumps(dict(seed=seed,latest=rows[-1] if rows else None,fixed12_verified=False)))
''')
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
print(json.dumps({'reader':str(OUT),'files':len(list(OUT.glob('*.py')))}))
