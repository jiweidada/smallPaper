"""Verify both downloaded archives, every file, saved arrays and AP arithmetic."""
from pathlib import Path
import hashlib,json,zipfile,numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/small_object_encoder_decoder12_20261010_v1'
EXPECTED={1:'fc1b0055cc1cdb31256081f5943b5938a7a96be5e00f85e809481ea44a3fe1c7',2:'eeb27f8ee33910764e333f396866fa9a1949a50f498c9cad194519f1c72eee33'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=[]
for seed in (1,2):
    archive=OUT/f'collection/baseline_replay_evidence_seed{seed}.zip';assert sha(archive)==EXPECTED[seed]
    dest=OUT/f'collection/seed{seed}';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():assert (dest/name).resolve().is_relative_to(dest.resolve()),name
        z.extractall(dest)
    manifest=json.loads((dest/f'evidence_manifest_seed{seed}.json').read_text())['files']
    for name,v in manifest.items():assert sha(dest/name)==v['sha256'] and (dest/name).stat().st_size==v['bytes'],name
    baseline=dest/f'seed{seed}_fine/baseline_eval';comparison=json.loads((baseline/'historical_comparison.json').read_text());completion=json.loads((baseline/'completion.json').read_text())
    assert not completion['historical_log_reproduction_pass'] and completion['optimizer_steps']==0 and completion['epochs_added']==0
    for iid in range(548):
        with np.load(baseline/f'stages_epoch12/image_{iid:06d}.npz') as x:
            assert x['final_boxes'].shape==(300,4) and x['final_logits'].shape==(300,10)
            assert all(np.isfinite(x[k]).all() for k in x.files)
    with np.load(baseline/'endpoint_epoch12/precision_recall.npz') as x:
        p=x['precision'];assert p.shape==(10,101,10,4,3)
        def mean(v):v=v[v>-1];return float(v.mean())*100
        current={'AP':mean(p[:,:,:,0,2]),'AP50':mean(p[0,:,:,0,2]),'AP75':mean(p[5,:,:,0,2]),'APs':mean(p[:,:,:,1,2]),'small_AP75':mean(p[5,:,:,1,2])}
    for key,index in [('AP',0),('AP50',1),('AP75',2),('APs',3)]:assert abs(current[key]-comparison['current_stats'][index]*100)<1e-9
    probe=json.loads((dest/f'diagnosis_seed{seed}_v2/summary.json').read_text())
    with np.load(dest/f'diagnosis_seed{seed}_v2/outputs.npz') as x:
        for field in ('pred_boxes','pred_logits'):
            for i in range(1,4):assert np.array_equal(x[f'forward_0_{field}'],x[f'forward_{i}_{field}'])
            for i in range(4):
                with np.load(baseline/f'stages_epoch12/image_{i:06d}.npz') as y:assert np.array_equal(x[f'forward_0_{field}'][i],y['final_'+field[5:]])
    results.append({'seed':seed,'archive_sha256':EXPECTED[seed],'verified_files':len(manifest)+1,'full_stage_images':548,'current_BASE12_percent':current,'historical_BASE12_percent':{'AP':comparison['historical_stats'][0]*100,'APs':comparison['historical_stats'][3]*100,'AP75':comparison['historical_stats'][2]*100},'deltas_pp':{'AP':current['AP']-comparison['historical_stats'][0]*100,'APs':current['APs']-comparison['historical_stats'][3]*100},'historical_gate_pass':False,'four_forward_native_wrapper_repeat_exact':True,'candidate_optimizer_steps':0})
report={'status':'LOCAL_ARCHIVE_ARRAY_AND_AP_VERIFICATION_PASS','seeds':results,'new_training_epochs':0,'scientific_claim':'Engineering archive and array checks only; historical reproduction failed and no candidate AP exists.'}
(OUT/'collection_validation.json').open('x',encoding='utf8').write(json.dumps(report,indent=2));print(json.dumps(report))
