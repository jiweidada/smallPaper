from pathlib import Path
import ast,difflib,hashlib,json,sys

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
old=(D/'collect_features.py').read_text(encoding='utf-8');new=old
def replace(a,b):
    global new
    assert new.count(a)==1,(a,new.count(a));new=new.replace(a,b)
replace('from pool_tensor import pool_tensor','from pool_tensor import pool_tensor\nfrom npz_export import save_npz')
replace('args=parser.parse_args()',"parser.add_argument('--recovery-manifest-sha256',required=True)\nargs=parser.parse_args()")
replace("protocol=json.loads((D/'semantic_protocol_FROZEN.json').read_text(encoding='utf-8'))",
"""assert sha(D/'export_recovery_execution_manifest.json')==args.recovery_manifest_sha256
recovery_manifest=json.loads((D/'export_recovery_execution_manifest.json').read_text(encoding='utf-8'))
for name,digest in recovery_manifest['files_sha256'].items(): assert sha(D/name)==digest,name
amendment=json.loads((D/'export_recovery_amendment_FROZEN.json').read_text(encoding='utf-8'))
assert amendment['old_completed_entries']==2 and amendment['remaining_native_entries']==112
for name,record in amendment['preserved_partial_files'].items(): assert sha(D/name)==record['sha256'],name
protocol=json.loads((D/'semantic_protocol_FROZEN.json').read_text(encoding='utf-8'))""")
replace("D/'collector.lock'","D/'collector_recovery.lock'")
replace("receipt_path=D/'native_load_receipt.json'","receipt_path=D/'recovery_native_load_receipt.json'")
replace("assert before==load_receipt['state_digest']","assert before==load_receipt['state_digest']\nassert before==json.loads((D/'native_load_receipt.json').read_text())['state_digest']")
replace("captures=D/'captures'; captures.mkdir(exist_ok=False)","captures=D/'captures'; assert captures.is_dir() and not list(captures.iterdir())")
replace("ledger=(D/'forward_entries.jsonl').open('x',encoding='utf-8')","ledger=(D/'recovery_forward_entries.jsonl').open('x',encoding='utf-8')")
replace("entries=0; completed=0; current=''; observed={}; hooks=[]","entries=2; completed=2; current=''; observed={}; hooks=[]")
start=new.index('torch.cuda.reset_peak_memory_stats()')
stop=new.index('hooks=[',start)
new=new[:start]+"""torch.cuda.reset_peak_memory_stats()
first=inputs['cases'][0]
baseline=dict(np.load(D/'baseline_first_no_capture_hooks.npz',allow_pickle=False))
original_control=dict(np.load(D/'first_control_memory_and_head.npz',allow_pickle=False))
"""+new[stop:]
replace('query_counts=[]; first_saved=None','query_counts=[]')
block="""            first_saved={name:value.copy() for name,value in arrays.items()}
            np.savez_compressed(D/'first_control_memory_and_head.npz',fine_memory=fine_map[0].cpu().numpy(),
                head_weight=model.decoder.dec_score_head[2].weight.detach().cpu().numpy(),
                head_bias=model.decoder.dec_score_head[2].bias.detach().cpu().numpy())"""
replacement="""            np.testing.assert_array_equal(fine_map[0].cpu().numpy(),original_control['fine_memory'])
            np.testing.assert_array_equal(model.decoder.dec_score_head[2].weight.detach().cpu().numpy(),original_control['head_weight'])
            np.testing.assert_array_equal(model.decoder.dec_score_head[2].bias.detach().cpu().numpy(),original_control['head_bias'])
            save_json(D/'recovery_first_fidelity.json',dict(status='NATIVE_OUTPUTS_AND_FULL_FINE_MAP_BITWISE_EQUAL_PRESERVED_ORIGINAL',
                recovered_entry=entries,reserved_repeat_used_here=True,full_query_feature_repeat_available=False,
                scope='Native prediction arrays,entire256x200x200fine map andheadparameters equal preserved originals;originalqueryfeatureswere not exported.'))"""
replace(block,replacement)
replace("np.savez_compressed(captures/(current+'.npz'),**arrays)","save_npz(captures/(current+'.npz'),arrays)")
start=new.index("    current='repeat_first_with_hooks'")
stop=new.index('assert entries==completed==114',start)
new=new[:start]+new[stop:]
replace("repeated_fidelity='All repeated prediction and sampled feature arrays bitwise equal first case.'",
"repeated_fidelity='Recovery firstnativepredictionandwholefinemapagreewithpreservedoriginal;fullqueryfeaturebitwiserepeatunavailableandnotclaimed.'")
replace("detector_optimizer_updates=0,diagnostic_data_probe_fits=0,",
"detector_optimizer_updates=0,diagnostic_data_probe_fits=0,export_recovery_amendment='export_recovery_amendment_FROZEN.json',\n    original_failed_entries=2,recovery_process_entries=112,full_query_feature_repeat_available=False,")
replace("D/'repeat_first_with_hooks.npz',D/'native_load_receipt.json'",
"D/'native_load_receipt.json',D/'recovery_native_load_receipt.json',D/'recovery_first_fidelity.json'")
ast.parse(new,feature_version=(3,9))
assert new.count('output=model(x)')==1 and 'np.savez_compressed(' not in new
with (D/'collect_features_export_recovery.py').open('x',encoding='utf-8') as f:f.write(new)
with (D/'collector_export_recovery.diff').open('x',encoding='utf-8') as f:
    f.writelines(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozen_original_collect_features.py',tofile='export_recovery_collect_features.py'))
print(json.dumps(dict(status='SEPARATE_RECOVERY_COLLECTOR_GENERATED_ORIGINAL_UNCHANGED',remaining_calls=112,
    original_sha256=hashlib.sha256((D/'collect_features.py').read_bytes()).hexdigest(),
    recovery_sha256=hashlib.sha256((D/'collect_features_export_recovery.py').read_bytes()).hexdigest())))
