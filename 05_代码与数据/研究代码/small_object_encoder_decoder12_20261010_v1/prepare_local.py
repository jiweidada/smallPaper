"""Freeze model/runtime/scientific protocol and four bounded plans; no Torch."""
from pathlib import Path
import ast,copy,datetime,difflib,hashlib,io,json,zipfile,base64
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'outputs/small_object_encoder_decoder12_20261010_v1';PREV=HERE.with_name('p3_band_attention12_20261010_v1');MODEL=HERE.with_name('small_object_encoder_decoder_20261010_v2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert not (HERE/'protocol.json').exists()
    for n in ('small_object_model.py','geometry_memory.py'):(HERE/n).write_bytes((MODEL/n).read_bytes())
    for n in ('evaluator_binding.py','evaluator_sources.json','stage_trace.py'):(HERE/n).write_bytes((PREV/n).read_bytes())
    s=(PREV/'results.py').read_text(encoding='utf8');s=s[:s.index('\ndef compare(paths):')]
    assert "ARMS = ('spatial', 'wavelet')" in s
    s=s.replace("ARMS = ('spatial', 'wavelet')","ARMS = ('baseline', 'fine', 'degraded')")
    s=s.replace('return dict(zip(METRICS, [avg(', 'out = dict(zip(METRICS, [avg(',1)
    marker='*[avg(precision[:, :, k, a, 2]) for a in (1, 2, 3)]]))'
    assert marker in s;s=s.replace(marker,marker+'\n        out[\'small_AP75\'] = avg(precision[5, :, k, 1, 2])\n        return out',1)
    s=s.replace("for k in METRICS},","for k in (*METRICS, 'small_AP75')},",1)
    (HERE/'results.py').write_text(s,encoding='utf8')
    binding=read(ROOT/'outputs/encoder_pool_diagnostic_20261010_v1/collection_seed1/capture_v1/source_identity.json')
    save(HERE/'source_binding.json',{'native_files_sha256':binding['native_files_sha256']})
    scientific=ROOT/'outputs/small_object_encoder_decoder_20261010_v2/prospective_screen_plan.json'
    protocol={'schema':'small_object_encoder_decoder12_20261010_v1','status':'FROZEN_BEFORE_BASELINE_REEVALUATION_AND_ALL_NEW_TRAINING','frozen_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scientific_plan_sha256':sha(scientific),'prototype_delivery_sha256':sha(ROOT/'outputs/small_object_encoder_decoder_20261010_v2/delivery.json'),'prototype_model_sha256':{n:sha(MODEL/n) for n in ('small_object_model.py','geometry_memory.py')},'GPU_uuid':{'1':'GPU-d29c55c5-02ab-1e69-c978-0b28f2102003','2':'GPU-b8084aa9-0176-df73-41fb-03307ded1be3'},'stage1':'Fine geometry field,seed1 and2,max12total epochs each,original132 prefix0..11.','stage2':'Degraded samecapacity control only if BOTHseed fine APs>BASE12 and overallAP>=BASE12. Never automatically queued.','primary':'Fixed12EMA standard COCO APs; no best checkpoint or later epoch.','mechanism':'Fine vs samecapacity degraded APs>0 bothseeds required for fine spatial-field attribution; exploratory only.','baseline':'Existing same-seed native BASE12 EMA checkpoint,first reproduced with unchanged evaluator/source/data.','baseline_reproduction_gate':'All12native stats maxabs difference<=1e-8 fraction relative historical epoch11 log. Record both current/historical arrays; failure blocks automatic candidate phase.','max_training_configurations':4,'max_total_epochs_each_configuration':12,'no_external_warmup_learning_steps':True,'sampling':'No new hyperparameter/seed/variant/radius/chart search; negative historical routes stay closed.','optim':'Native groups unchanged;6 added GroupNorm parameters explicitly zero weight decay. Original clipping,AMP/EMA/lr/augmentation/query budget/losses retained.','training_integrity':'First100 pre-forward input/target/RNG hashes exact against existing baseline traces, optimizer coverage exact,real8 updates andEMA counts within12prefix. No independent old DN tensor replay claim.','evaluation_artifacts':'Full precision/recall,all GT hits andnative matching records; AP/APs/AP75/smallAP75/classAP/new-lostTP/FP. Passive initial/pre/final outputs on548images.','cost':'Actual forward+backward CUDA memory from real first8 optimizer updates; no standalone extra training. Runtime includes observer overhead;not claimed benchmark FPS.','failure':'No automatic retries/resume/rollback/recipe change/extensions. Preserve one-shot locks/logs/partial weights/failed endpoints. Hardbaseline timeout1800s,each candidate64800s.','material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate'},'scientific_scope':'ANALYZED_EXPLORATORY_SCREEN_NOT_CAUSAL_ROOT_OR_NOVELTY_PROOF'}
    save(HERE/'protocol.json',protocol);save(OUT/'protocol.json',protocol)
    plans=HERE/'configs';plans.mkdir(exist_ok=False)
    for seed in (1,2):
        reference=read(OUT/f'reference_seed{seed}.json');old=read(PREV/f'configs/wavelet_seed{seed}.json');native=old['native_baseline_config'];actual=reference['yaml_cfg']
        # Native and ResidualGeometryDEIM baseline recipes differ only by the
        # old empty wrapper and non-learning paths/logging/include metadata.
        allowed={'ResidualGeometryDEIM','model','output_dir','use_swanlab','__include__','config'}
        assert all(native.get(k)==actual.get(k) for k in set(native)|set(actual) if k not in allowed)
        assert actual['ResidualGeometryDEIM']['arm']=='baseline'
        assert reference['runtime']['identity']['common_init_sha256']=='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
        for variant in ('fine','degraded'):
            plan={'schema':protocol['schema'],'variant':variant,'seed':seed,'native_baseline_config':native,'model_recipe':{k:native[k] for k in ('HGNetv2','HybridEncoder','DFINETransformer')},'criterion_recipe':native['DEIMCriterion'],'historical_baseline_stats':reference['historical_epoch12'][0]['test_coco_eval_bbox'],'baseline_first100_sha256':reference['first100_sha256'],'evaluator_binding':reference['runtime']['evaluator_runtime']['binding'],'evaluator_root':reference['runtime']['evaluator_runtime']['package_root'],'baseline_checkpoint_sha256':{1:'ba3b6351fa8a4f7c769698c616ed6d20acf6a60ba119cf8edc8611851c4d27d4',2:'6198ee5348630ce97c322c16aa4499a50a2f7af0d08dea1ec3a4cb632d9d5046'}[seed],'common_init':actual['tuning'],'reference_metadata_sha256':sha(OUT/f'reference_seed{seed}.json'),'screen_protocol_sha256':sha(HERE/'protocol.json')}
            save(plans/f'{variant}_seed{seed}.json',plan)
    from runtime import IMPLEMENTATION
    for n in HERE.glob('*.py'):ast.parse(n.read_text(encoding='utf8'),filename=n.name)
    expected={n:sha(HERE/n) for n in IMPLEMENTATION};expected.update({'configs/'+p.name:sha(p) for p in plans.iterdir()})
    save(HERE/'deployment_manifest.json',expected)
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for n in list(expected)+['deployment_manifest.json']:z.writestr(n,(HERE/n).read_bytes())
    data=b.getvalue();(OUT/'deploy_bundle.zip').write_bytes(data);(OUT/'deploy_bundle.zip.b64').write_text(base64.b64encode(data).decode())
    meta={'status':'FROZEN_NOT_LAUNCHED','files':len(expected)+1,'zip_bytes':len(data),'zip_sha256':hashlib.sha256(data).hexdigest(),'protocol_sha256':sha(HERE/'protocol.json')};save(OUT/'deployment_local.json',meta)
    print(json.dumps(meta));print(base64.b64encode(data).decode())
if __name__=='__main__':main()
