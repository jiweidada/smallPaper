from pathlib import Path
import hashlib,json,py_compile
ROOT=Path(__file__).resolve().parent
A49=ROOT.parent/'A49_下一主线有界筛选_20261101_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['activation_A50.py','edge_shape_matching.py','runtime_A50.py']
for name in names:py_compile.compile(str(ROOT/name),doraise=True)
expect=json.loads((A49/'asset_expectations.json').read_text(encoding='utf-8'))
checkpoint=A49/'completed_B_v1/arm_B/epoch12.pth'
external={
 '/root/uav_a47_20261101_v1/assets_A43_baseline/common_init_10class.pth':expect['common_init_sha256'],
 '/root/uav_a47_20261101_v1/assets_A43_baseline/runtime_common.py':expect['runtime_common_sha256'],
 '/root/uav_a49_20261101_v1/arm_B/epoch12.pth':sha(checkpoint)}
prefix='/root/uav_a47_20261101_v1/assets_A43_baseline/source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/'
for name in ['engine/edgecrafter/decoder.py','engine/edgecrafter/matcher.py','engine/edgecrafter/utils.py','engine/edgecrafter/criterion.py']:
    external[prefix+name]=expect['native_source_sha256'][name]
p=ROOT/'activation_protocol_FROZEN_v1.json'
assert not p.exists()
plan={'phase':'A50','type':'BOUNDED_ACTIVATION_ONLY','goal_status':'active','mainline_confirmed':False,
 'checkpoint':'/root/uav_a49_20261101_v1/arm_B/epoch12.pth',
 'checkpoint_role':'already observed native baseline fixed12 EMA; no new trained candidate outcomes used',
 'images':64,'selection':'first64 by sha256(A50-shape:+filename) over native trainset',
 'split':'train','batch_size':4,'max_whole_model_forwards':16,'optimizer_updates':0,
 'input':'native dataset epoch0 deterministic flip,640stretch; original-image area defines small',
 'model_mode':'literal eval FP32; hook reads inner decoder returns without changing outputs or BN',
 'base_identity':'per-image reimplementation must return native Hungarian indices exactly',
 'decoded_support_identity_max_absolute_error':2e-6,
 'M':'native cost plus native cost_giou*(1-exp(-mean4(E_Gaussian |edge-GTedge| /GTwidth_or_height)))',
 'Q':'same expression, exact native categorical marginal expectation replaces Gaussian expectation',
 'shared_settings':'native cost_cls,L1,GIoU retained; same bins, affine support, normalization, bounded weight; no new params',
 'activation_gate':{'minimum_small_GT_Q_vs_M_owner_changes':20,'minimum_fraction_of_small_GT':.01},
 'gate_scope':'an operation-distinctness check only; not sufficient for AP benefit or journal novelty',
 'on_weak_activation':'no threshold/weight/site sweep; stop exact proposal and continue different mechanism',
 'new_detector_forward_cap_from_preceding_terminal':144957,
 'no_auto_retry':True,'files_sha256':{n:sha(ROOT/n) for n in names},'external_sha256':external}
p.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(p),'checkpoint_sha256':sha(checkpoint),'max_forwards':16,'updates':0}))
