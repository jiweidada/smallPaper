"""Read-only exact12 native Warm endpoint verification; no network/evaluator call."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
root=Path('/24085403014/DEIM/experiments/warm_stem12_20261011_v1')
code=root/'code';screen=root/'seed2_warm_stem'
read=lambda p:json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def state(r,pid='pid',ticks='start_ticks'):
 try:
  f=Path('/proc/'+str(r[pid])+'/stat').read_text().split()
  return dict(r,same_live=f[21]==str(r[ticks]) and f[2]!='Z',state=f[2])
 except (FileNotFoundError,ProcessLookupError):return dict(r,same_live=False)
expected={"closeout_protocol.json":"49454271684f19dd487b53857a6912384bf0222513e342463f6493257ed0be4e","collect_terminal.py":"818a574f59b836766d4788848d2937f5d3af78a0aefca32f41a481804544c3a5","dispatch.py":"06e44e1163451fab9eef7b62e7d15355baac9c583ee7008fecffefda407f2c16","evaluator_binding.py":"7c9262de399b85895c5996f49c24ea24ca42cdca98c791a67d5ff564d85e7782","evaluator_sources.json":"fd37b26487848369be53c02b80fe3f58c9b67b90d973d64b7d209965ccc968e0","experiment.py":"d9b122c5beda39af500807b380e279836b3fad86e559e005ab204ba6a5792c2c","finish.py":"1f639a7b1026c8d60faa6019f0babc4c13c5c7f670398eb0dfcf56fce0eb7372","geometry.py":"f537d8ad50a54fd1b063db5e86d6c1cf4534304588db3c5008cfe69fac1100c9","initialization.py":"a0658cb895923055d832d04f683128969ae91a53e3f78a770616ec1fe1d0200c","original_backbone_source.txt":"8d229219cce4ff8213e38b9f4fc3e6d72f227061b9ecca6a7a322e15d30195bc","original_stem.py":"ca9884aebe8fe3b2bbc1f3f36ea2fed17f2c817b464a2581f22c2f4d3ffb2c74","original_stem_source.json":"967f5d9283c71ebb854354e19fe765166983f47d23c2212555d0ff4bc9e49193","preflight.py":"66af7d8134fbd89f206908ee7ad3ee8587269a8fbb949f9aeff549096dbcf550","prepare_training.py":"71154964015c43efa0b158d00776466da89d6f952d3671097d13993f904f9c24","residual.py":"3f60403011f8e999c77dcc4a31aeb9974149fda9dbc7aa9216eca486704f30a2","results.py":"9c6cca1e563cdb50877ac37d26e2098743b19c3a85d86d7e03005f8bffe1a02f","reuse_baseline.py":"e1cb9e0b0594a427af08813a04ed49660ce045183d929e64909f2545f54f335e","runtime.py":"494491abad53d5f433d693e964ac635011f1e816d4b40b38c49856236f96a6a4","source_binding.json":"0c65e7cfda7fd294809d284020e26a5a92a1c9bb6580bec45ffbe6015a1498f5","stage_trace.py":"d9668777da440da708ecb3779e6cefc4810abfe65480b6b11520463ac7b896a5","supervise.py":"d80867b7dafaad875498b8f1a25e310f0ccd710e6f941be6cc27bcf397bfe4f8","training_audit.py":"0a13858be69bffd7a932e25b7f06bcfd7466fb8b080fea4ee07a28aa68950254","training_protocol.json":"b7a9f4e8db9cf463e5858c69eb22a4b7ad30866f19d507bc4b7113b4648a7064","verify_candidate_reload.py":"bbd10cc73101cf6d8a01aa516508ff3b8902d3d13cb9262f8cf3be410720d4ee","warm_stem.py":"107a3e2718cf690df5e39fdc15f1b98f63b14de860ce81b83da377875a45c105","warm_stem_model.py":"58c94ea2b7dd019f93714e85a9fd88121173c96a53fed67cdd8e74cdf02a9232"}
assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==expected
sys.path.insert(0,str(code))
from runtime import verify_bundle
from stage_trace import verify_trace
from results import load_endpoint
receipt,plan,paths=verify_bundle(screen/'bundle')
identity=receipt['identity'];run=paths['output']
controller=state(read(screen/'controller_start.json'))
trainer=state(read(screen/'candidate_start.json'),'child_pid','child_start_ticks')
assert not controller['same_live'] and not trainer['same_live']
assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
exit=read(screen/'candidate_completion.json')
assert exit['exit_code']==0 and not exit['timed_out'] and not exit['automatic_retry']
done=read(run/'training_completion.json')
assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12
assert done['entered_epochs']==list(range(12)) and done['identity']==identity
assert sha(run/'last.pth')==done['last_checkpoint_sha256']
assert sha(run/'stages_epoch12/manifest.json')==done['stage_manifest_sha256']
assert sha(run/'endpoint_epoch12/manifest.json')==done['endpoint_manifest_sha256']
stage=verify_trace(run/'stages_epoch12',identity)
endpoint,flags,_=load_endpoint(run/'endpoint_epoch12')
assert endpoint['identity']==identity and endpoint['evaluated_weights']=='EMA' and endpoint['mode']=='native_runtime'
assert endpoint['GT_count']==38759 and flags['hits'].shape==(38759,10)
assert flags['hits'].dtype==np.dtype(bool) and int(np.unique(flags['annotation_id']).size)==38759
assert flags['hits'].sum(0).tolist()==endpoint['TP_by_IoU']
annotation=read(paths['val_ann']);gt=sorted(annotation['annotations'],key=lambda x:x['id'])
assert np.array_equal(flags['annotation_id'],np.asarray([a['id'] for a in gt]))
assert np.array_equal(flags['category_id'],np.asarray([a['category_id'] for a in gt]))
assert not any(a.get('ignore',False) or a.get('iscrowd',False) for a in gt)
with np.load(run/'endpoint_epoch12/precision_recall.npz',allow_pickle=False) as a:
 precision=a['precision'];recall=a['recall']
 assert precision.shape==(10,101,10,4,3) and recall.shape==(10,10,4,3)
 assert np.isfinite(precision).all() and np.isfinite(recall).all()
 def avg(v):
  v=v[v>-1];assert len(v);return float(v.mean()*100)
 metrics=[avg(precision[:,:,:,0,2]),avg(precision[0,:,:,0,2]),avg(precision[5,:,:,0,2])]+[avg(precision[:,:,:,i,2]) for i in (1,2,3)]
 assert np.allclose(metrics,np.asarray(endpoint['native_stats'][:6])*100,atol=1e-9,rtol=0)
rows=[json.loads(x) for x in (run/'log.txt').read_text().splitlines() if x]
assert [r['epoch'] for r in rows]==list(range(12)) and rows[-1]['test_coco_eval_bbox']==endpoint['native_stats']
assert read(run/'first100_parity.json')['all100_rows_exact']
base=paths['baseline_cached'];spec=read(code/'training_protocol.json')['baseline_reuse']['2']
for n,h in spec['pinned_files'].items():assert sha(base/n)==h
b=read(base/'endpoint_epoch12/summary.json')
for name in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
 assert identity[name]==b['identity'][name],name
assert np.allclose(b['native_stats'][[0,3]] if isinstance(b['native_stats'],np.ndarray) else [b['native_stats'][0],b['native_stats'][3]],[.13711356414272169,.08636856006407324],atol=1e-12,rtol=0)
AP=endpoint['native_stats'][0]*100;APs=endpoint['native_stats'][3]*100
baseAP=b['native_stats'][0]*100;baseAPs=b['native_stats'][3]*100
gate=APs>baseAPs and AP>=baseAP
assert gate==done['native_gate_passed']==False
assert 'torch' not in sys.modules and not any(n=='pycocoeval' or n.startswith('pycocoeval.') for n in sys.modules)
assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==expected
o=dict(status='PASS_WARM_SEED2_EXACT12_PRIMARY_INTEGRITY_AND_NATIVE_AP_ARITHMETIC_ONLY',seed=2,
 UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity=identity,
 source26_before_after_exact=True,native_bundle_and468source_verified=True,
 actual_controller=controller,actual_trainer=trainer,completed_epochs=12,evaluated_weights='EMA',
 last_checkpoint_sha256=done['last_checkpoint_sha256'],training_completion_sha256=sha(run/'training_completion.json'),
 stage_manifest_sha256=done['stage_manifest_sha256'],stage_images_each_SHA_verified=stage['images'],
 endpoint_manifest_sha256=done['endpoint_manifest_sha256'],GT_flags_and38759_identities_verified=True,
 primary_saved_precision_six_AP_metrics_percent=metrics,AP=AP,APs=APs,BASE12=dict(AP=baseAP,APs=baseAPs),
 delta_points=dict(AP=AP-baseAP,APs=APs-baseAPs),native_gate_passed=False,
 independent_saved_EMA_replay_verified=False,formal_terminal_reader_passed=False,
 same_epoch12_log_and_endpoint_exact=True,first100_exact=True,baseline_cached_pinned_files_verified=True,
 new_model_forwards=0,new_BASE_forwards=0,new_training_epochs=0,new_optimizer_updates=0,new_evaluator_calls=0,
 scope='Existing native output integrity, 548 stage-file SHA values, all GT flags and saved precision arithmetic only; no independent saved-EMA replay, no raw stage-array shape analysis or matching-record audit.')
