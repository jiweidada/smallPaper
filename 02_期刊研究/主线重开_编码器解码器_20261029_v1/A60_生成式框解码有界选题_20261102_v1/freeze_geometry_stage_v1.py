from pathlib import Path
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
R54=P.parent/'A54_场景尺度约束候选_20261101_v1'
source=P/'geometry_exposure_D0_v1.py'
scope=dict(phase='A60_GEOMETRY_D0',sample_images=128,seed=20261102,prior_draws_per_GT=4,
    priors=['specified_truncated_gaussian','uniform'],relative_edge_threshold=.1,midpoint_time_nodes=64,
    train_root='/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone/VisDrone2019-DET-train',
    label_policy='score>0,class1..10,positive raw width/height; selected GT additionally clipped to image bounds',
    sample_selection='128 smallest SHA256(seed|jpg_name), no validation images',
    prior_specification='Independent normalized cx,cy,w,h, N(0,1) truncated [0,1]; uniform[0,1] sensitivity. Positive widths/heights, boxes can extend beyond image boundaries. Own explicit protocol, not verified native FlowDet sampler.',
    coordinate_representation='cxcywh prior converted to xyxy; GT xyxy normalized by original dimensions',
    d0='max absolute corresponding xyxy-edge error normalized by GT width/height',
    linear_clock='alpha=u',global_clock='alpha=1-expm1((1-u)*log1p(D))/D, D=median(d0) in training sample',
    instance_clock='same alpha with D=d0 per pair; alpha=u for D->0',
    primary_quantity='Exact fraction of uniform clock time with relative max edge error <=0.1; not recall or AP',
    secondary_quantity='Fraction of fixed64 midpoint times with paired IoU>=0.5; geometry only, no matching',
    seconds_cap=120,annotation_dataset_passes=1,model_constructors=0,model_forwards=0,
    optimizer_updates=0,checkpoint_loads=0,AP_evaluations=0,training_jobs=0,
    no_automatic_retry_or_sample_change=True,script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    limits='Training-input geometry and oracle endpoint clock only. No detector failure cause, deployed clock accuracy, learned gain, statistical benefit or novelty claim.')
target=P/'geometry_scope_FROZEN_v1.json'
with target.open('x',encoding='utf-8') as f:json.dump(scope,f,ensure_ascii=False,indent=2)
stage=R54/'A60_uploads';stage.mkdir(exist_ok=False)
for src in [source,target]:shutil.copyfile(src,stage/src.name)
print(json.dumps(dict(scope_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    script_sha256=scope['script_sha256'],new_model_calls_allocated=0,stage=str(stage))))
