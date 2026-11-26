from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    with (D/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
original=read(D/'execution_inputs_manifest.json')
for name,digest in original['files_sha256'].items():assert sha(D/name)==digest,name
failure=read(D/'export_failure_terminal_observation.json')
assert failure['old_worker']['author_entries']==failure['old_worker']['author_completed']==2
assert not failure['old_collector_process_present'] and not failure['captures']
assert read(D/'export_fixture_local.json')['status']=='PASS_STANDARD_NPZ_RESERVED_FILE_KEY_ROUNDTRIP'
log=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\前期调研_码率受限航拍小目标_20261028_v1\远端验证_20261029_v1\ssh_operation_receipts.jsonl')
rows=[]
for line in log.open(encoding='utf-8'):
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('id')=='A38_verify_npz_writer_fix_cpu_only_20' and r.get('action')=='exec':rows.append(r)
assert len(rows)==1 and rows[0]['exit_code']==0
remote=json.loads(rows[0]['stdout']);assert remote['status']=='PASS_STANDARD_NPZ_RESERVED_FILE_KEY_ROUNDTRIP'
save('export_fixture_remote_receipt.json',dict(receipt=rows[0],result=remote))
old=(D/'collect_features.py').read_text(encoding='utf-8')
new=(D/'collect_features_export_recovery.py').read_text(encoding='utf-8')
trees=[ast.parse(s,feature_version=(3,9)) for s in [old,new]]
names=['native_arrays','load_case','memory_hook','query_hook','score_hook']
for name in names:
    nodes=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
    assert ast.dump(nodes[0])==ast.dump(nodes[1]),name
start='        x=load_case(row); output=model(x)'
stop="        if row['index']==0:"
assert old[old.index(start):old.index(stop)]==new[new.index(start):new.index(stop)]
amendment=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
    status='FROZEN_EXPORT_ONLY_RECOVERY_BEFORE_ANY_VAL_FEATURE_OR_DATA_PROBE',
    trigger="Original exporter failed on NumPy savez_compressed destination argument named 'file' colliding with unchanged array metadata key 'file'.",
    source_original_protocol_sha256=sha(D/'semantic_protocol_FROZEN.json'),
    source_original_execution_manifest_sha256=sha(D/'execution_inputs_manifest.json'),
    old_completed_entries=2,remaining_native_entries=112,combined_cap=114,
    order='Originalhook-freebaseline andfailedfirstinstrumentedcallareconsumed. Recoveredfirsttraincaseusespreviouslyallocatedrepeatcall,thenremaining111cases. No extra final repeat.',
    preserve=['Same64train/48valimages','Samecheckpoint/source/transform','Samequeryandmemoryfeatures','Samesamplegeometryandignores','Sameprobeheads/lambda/seeds','Sameprimarymetric/CI/gates','Total114whole-detectorentries','Detectoroptimizer0','Data-probefitcap7'],
    procedural_deviation='One recovery process reinstantiates the identical frozen checkpoint after authoritative terminal export failure; this deviates from the original no-restart clause and moves the reserved repetition earlier. It is not a performance-driven rerun.',
    measurement_deviation='Original firstqueryfeatures were not exported. Full query-feature bitwise repetition cannot be verified. Preserved nativeprediction arrays,entirefine-memorymap andheadparameters must match recoveredfirstcase bitwise; independent NumPy feature/Linear verification remains mandatory.',
    serialization_change='Standard nonpickleNPZ writer accepts array keyfile without changing array names,dtypes,shapes or values; frozen analysis script stays unchanged.',
    preserved_partial_files=failure['files'],old_worker_exit=failure['old_worker'],
    old_collector_pid=failure['old_collector_pid'],original_model_process_observed_absent=True,
    preserved_numerical_functions=names,loop_numerical_feature_block_identical=True,
    before_val_model_entries=True,before_data_probe_fits=True,performance_outcomes_seen=False,
    mainline_validated=False,automatic_further_restart=False)
save('export_recovery_amendment_FROZEN.json',amendment)
files=['npz_export.py','artifact_export_fixture.py','collect_features_export_recovery.py',
       'run_recovery_and_record.py','launch_export_recovery.py','export_recovery_amendment_FROZEN.json',
       'collector_export_recovery.diff','export_failure_terminal_observation.json']
for name in files:
    if name.endswith('.py'):ast.parse((D/name).read_text(encoding='utf-8'),feature_version=(3,9))
save('export_recovery_execution_manifest.json',dict(original_execution_manifest_sha256=sha(D/'execution_inputs_manifest.json'),
    files_sha256={n:sha(D/n) for n in files},old_entries=2,remaining_entries=112,total_cap=114,data_probe_fit_cap=7))
z=D/'export_recovery_bundle.zip'
with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED) as archive:
    for name in files+['export_recovery_execution_manifest.json']:archive.write(D/name,name)
print(json.dumps(dict(status='EXPORT_RECOVERY_FROZEN_WITHIN_ORIGINAL114',bytes=z.stat().st_size,
    bundle_sha256=sha(z),recovery_manifest_sha256=sha(D/'export_recovery_execution_manifest.json'),
    original_frozen_files_unchanged=len(original['files_sha256']),query_feature_repeat_unavailable=True)))
