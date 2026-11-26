from pathlib import Path
import hashlib,json,py_compile,shutil,time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
A43=BASE/'A43_ECDet三组蒸馏最小验证_20261031_v1'
A47=BASE/'A47_微小框匹配稳定性选题与验证_20261101_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__=='__main__':
    assert not (ROOT/'protocol_FROZEN.json').exists()
    mathcheck=json.loads((ROOT/'norm_math_check.json').read_text())
    assert mathcheck['status']=='PASS' and mathcheck['operator_sha256']==sha(ROOT/'instance_moment_norm.py')
    audit=json.loads((ROOT/'data_and_site_audit.json').read_text())
    assert audit['annotation_hashes_verified']==6471 and audit['model_parameters']==9803112
    for name in ['evaluate.py','train_data_manifest.json','val_data_manifest.json']:
        assert not (ROOT/name).exists()
        shutil.copyfile(A43/name,ROOT/name)
    shutil.copyfile(A47/'asset_expectations.json',ROOT/'asset_expectations.json')
    files=['runtime_A48.py','training_utils_A48.py','instance_moment_norm.py','prepare_server_A48.py',
           'capacity_A48.py','train_A48.py','supervise_A48.py','evaluate.py','asset_expectations.json',
           'train_data_manifest.json','val_data_manifest.json','norm_math_check.json','data_and_site_audit.json']
    for name in files:
        if name.endswith('.py'):py_compile.compile(str(ROOT/name),doraise=True)
    protocol={'status':'FROZEN_BEFORE_GPU','version':'A48_v1','unix':time.time(),'user_date':'2026-11-01',
      'question':'Does per-instance rather than box-area-weighted training normalization improve UAV small-object detection beyond ordinary pixel-weighted foreground statistics?',
      'hypothesis_status':'unvalidated task-specific extension of prior weighted normalization, not a novel BN formula',
      'dataset':'VisDrone2019-DET train6471/val548, 10 classes',
      'host':'ECDet-S fixed EdgeCrafter d8752b75c9beaecfcdd0e5cd207bae3b6946ae23; existing A43 common init',
      'arms':['B','M','Q'],'B':'original BN','M':'uniform + summed bbox pixel mass',
      'Q':'uniform + per-box unit mass, each valid instance counts once before overlap aggregation',
      'sites':list(audit['chosen_encoder_sites']),
      'weights':'fractional box-cell overlap; mixture lambda=0.5*min(1,global_step/1618); total batch mass1; emptyGT uniform; no labels at inference',
      'variance':'training weighted central second moment (PyTorch BN biased-variance convention); running var correction 1/(1-sum(w^2)); FP32 moments; original affine/state names',
      'inference':'materialize ordinary BatchNorm at four sites; same parameter count and native inference graph, no mask prediction',
      'epochs':12,'microbatch':8,'accumulation':1,'steps_per_epoch':809,'updates_per_arm':9708,'seed':20261101,
      'optimizer':'same A43 AdamW backbone2.5e-5/rest5e-4; wd1e-4 except norm/bn/bias0; gradient norm clip0.1',
      'schedule':'500-step linear warmup, cosine to0.1*initial at9708; separate fresh initialization for each arm',
      'augmentation':'same deterministic hash horizontal flip and stretch640x640, ImageNet normalization; no Mosaic; all raw images/annotations SHA checked before launch',
      'precision':'BF16 forward, FP32 native criterion; no GradScaler skipped updates; no teacher',
      'EMA':'same A43 decay .9999*(1-exp(-updates/1000)); integer buffers copied; fixed epoch12 EMA primary',
      'validation_epochs':[4,8,12],'primary_metrics':['APs','AP'],
      'evaluation':'A43 byte-identical evaluator: COCO-style bbox maxDets300, original clipped GT area, class0 union-IoA filtering; NOT official VisDrone evaluator',
      'practical_gates_pp':{'Q_minus_B_APs_min':.5,'Q_minus_M_APs_min':.3,'Q_minus_B_AP_min':-.2},
      'gates_interpretation':'resource allocation only, not statistical significance, causality or novelty proof; no best-epoch selection',
      'capacity_budget':{'model_forwards':12,'optimizer_updates':9,'training_steps_per_arm':3,'labels_mixture':.5,'inference_check_per_arm':1},
      'stop':'nonfinite or infrastructure failure stops queue; no automatic retry. At fixed12, any practical gate failure retires this exact recipe; no lambda/site/seed rescue sweep.',
      'next_steps':'No longer schedule or other candidate automatically. Broad goal stays active if this candidate fails.',
      'old_overlap_boundary':{'20261027_BN_cross_swap':'stopped frozen raw/EMA running-state replacement; current method changes training statistics measure from common initialization, no old state swapping',
                              'D025':'attention scale/point weight normalization; no BN moments and no H training endpoint',
                              'A42_A43_A44_A46_A47':'not resumed; no P2 readout, distillation, foreground quota, class-negative reweighting or jitter matching'},
      'known_risks':['generic weighted BN and foreground-aware normalization pre-exist; innovation may be too incremental',
                     'box footprint includes background/occlusion and overlaps; instance balance may amplify noisy tiny regions',
                     'train/running-stat distribution effects may hurt calibration; short one-seed pilot is not generalization evidence'],
      'files_sha256':{name:sha(ROOT/name) for name in files}}
    (ROOT/'protocol_FROZEN.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'FROZEN','sha256':sha(ROOT/'protocol_FROZEN.json'),'files':len(files),'new_GPU_forwards':0,'new_optimizer_updates':0}))
