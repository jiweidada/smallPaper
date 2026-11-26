"""Freeze sources and one matched <=12-epoch experiment. Does not train."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
D=P.parent/'A54_场景尺度约束候选_20261101_v1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def pure_functions(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(body)==len(names)
    namespace={};exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def main():
    assert not (P/'protocol_FROZEN_v1.json').exists()
    assert json.loads((P/'capacity_local_verification_v1.json').read_text())['source_bindings_verified']
    manifest=D/'A60_downloads/dataset_manifest_v1.json'
    assert sha(manifest)=='350fcef03e5ae28cfb1fe06b8483500c17524ef5ff8297c269cf3ac415015639'
    data=json.loads(manifest.read_text());splits=data['splits']
    assert splits['train']['images']==6471 and splits['val']['images']==548 and splits['train']['max_GT']==902
    # Compare the entire cached GT construction against the established evaluator,
    # without importing either model/runtime or executing another COCO evaluation.
    old=pure_functions(P.parent/'A58_区域范围路由候选_20261102_v1/evaluate_A58.py',
        ['union_overlap_fraction','coco_ground_truth'])
    new=pure_functions(P/'evaluate_A60_v1.py',['union_overlap_fraction','ground_truth'])
    class Rows:pass
    rows=Rows();rows.rows=splits['val']['rows']
    expected=old['coco_ground_truth'](rows);actual=new['ground_truth'](rows.rows)
    for key in ['images','annotations','ignored_valid_gt','categories']:assert actual[key]==expected[key],key
    assert len(actual['annotations'])==38602 and actual['ignored_valid_gt']==157
    assert sum(a['area']<=1024 for a in actual['annotations'])==26444
    metric_proof=dict(status='PASS_FULL_CACHED_VAL_GT_PARITY_WITH_ESTABLISHED_EVALUATOR',
        images=548,scored_GT=38602,ignored_valid_GT=157,small_GT=26444,
        annotation_array_sha256=hashlib.sha256(json.dumps(actual['annotations'],sort_keys=True).encode()).hexdigest(),
        new_model_forwards=0,new_optimizer_updates=0,new_COCO_evaluations=0,
        scope='GT/ignore/area/categories parity on cached annotations, not a detector or postprocess efficacy test.')
    names=['box_progress_conditioning_v1.py','torch_detector_parts_v1.py','head_query_time_A60_v1.py',
        'model_A60_v1.py','criterion_A60_v1.py','box_ops_A60_v1.py','runtime_A60_v1.py',
        'evaluate_A60_v1.py','train_A60_v1.py','supervise_formal_A60_v1.py']
    capacity=json.loads((P/'capacity_scope_FROZEN_v1.json').read_text())
    for name in names:
        ast.parse((P/name).read_text(encoding='utf-8'),filename=name)
        if name in capacity['source_sha256']:assert sha(P/name)==capacity['source_sha256'][name],name
    init=P/'CPU_constructor_stage_v1/common_init_A60_v1.pth'
    assert sha(init)==capacity['initialization_sha256']
    scope=dict(phase='A60_MATCHED_FIXED12_RAW_PILOT',material_passport=dict(workflow='ARS experiment run, inline single assistant',
        user_authorization='Continue finding the mainline and run necessary tests; <=12 epochs per arm; no subagents',
        received_materials='A59 primary review, A60 selected literature/geometry/capacity and shared public pretrained initialization',
        deliverable='Bounded matched pilot and evidence report, not a submission or established novelty claim'),
        arms=dict(B='Linear CFM interpolation; physical clock u',
            M='Relative-geometry-biased interpolation; physical clock alpha',
            Q='Exactly M noisy inputs; normalized geometric-progress clock u'),
        seed=20261102,epochs_per_arm=12,evaluation_epochs=[4,8,12],endpoint='fixed12 raw model; no EMA or best epoch',
        source_sha256={name:sha(P/name) for name in names},initialization_sha256=sha(init),dataset_manifest_sha256=sha(manifest),
        train_images=6471,val_images=548,train_valid_GT=343204,maximum_GT_per_image=902,
        image_size=[640,640],resize='RGB PIL bilinear stretch',augmentation='one common seeded horizontal flip probability .5; no mosaic/crops/mixup',
        train_microbatch=1,gradient_accumulation_images=8,optimizer_updates_per_epoch=809,
        gradient_reduction='Mean of native per-image matched-positive-normalized losses, last accumulation group has seven images; not claimed equivalent to a native batch8 matcher',
        training_query_policy='max(300,32*ceil(GT_in_microbatch/32)); cap1024, never drop GT',
        inference_queries=300,inference_decoder_calls=1,internal_decoder_stages=6,eval_batch=2,
        inference_inputs='Image + fixed filename-seeded prior + time999; no GT, target scale or oracle endpoint',
        common_model='Controlled Torch/TorchVision R50-FPN + selected DiffusionDet DynamicHead; linear CFM inspired by FlowDet; not a full author FlowDet reproduction',
        trainable_parameters=110565396,
        common_matcher_correction='Retain native DynamicK initial proposals; if uncovered GT remain, keep one existing positive per covered GT, Hungarian-assign free queries to uncovered GT, restore optional nonconflicting native positives; shared engineering correction, not candidate increment',
        optimizer=dict(type='AdamW',backbone_LR=1e-5,head_FPN_LR=1e-4,new_classifiers_LR=5e-4,
            weight_decay=1e-4,bias_norm_weight_decay=0.,betas=[.9,.999],clip_grad_norm=1.,
            schedule='500 update linear warmup factor .1->1, then cosine to .1 at update9708'),
        backend='BF16 backbone; FP32 ROI/head/loss; TF32 off; cudnn deterministic=True, benchmark=False',
        evaluation='COCO-style10 classes, maxDets300, clipped original GT area small<=1024, class0 union-IoA>=.5 filtering; not official VisDrone evaluator',
        postprocess='sigmoid global top300 class entries -> classwise NMS .6 -> original-image clip -> degenerate/ignore filtering; no refill',
        latency=dict(warmup=10,timed=30,batch=1,model='full same eager model, no decode/postprocess'),
        B_learning_gate=dict(AP_min=8.,APs_min=3.,failure='Cancel unstarted M/Q; candidate remains untested'),
        continuation_gates=dict(Q_minus_B_APs_min=.5,Q_minus_M_APs_min=.3,Q_minus_B_AP_min=-.2,Q_over_M_median_latency_max=1.10),
        maximum_seconds_per_arm=43200,maximum_seconds_total=129600,queue=['B','M','Q'],
        no_automatic_retry=True,no_resume=True,no_new_seed=True,no_extensions=True,no_method_rescue=True,
        maximum_detector_forwards_per_arm=78514,maximum_optimizer_updates_per_arm=9708,
        forward_breakdown_per_arm=dict(train=77652,evaluation=822,latency=40),
        maximum_new_formal_forwards=235542,maximum_new_formal_updates=29124,maximum_new_formal_epochs=36,
        global_before=dict(model_forwards=276643,all_optimizer_updates=233606,formal_optimizer_updates=233518,
            discarded_capacity_optimizer_updates=88,formal_epochs=288),
        global_with_full_reserved_cap=dict(model_forwards=512185,all_optimizer_updates=262730,
            formal_optimizer_updates=262642,discarded_capacity_optimizer_updates=88,formal_epochs=324),
        data_manifest_new_read_scope='7019 JPEG headers and annotation files, no full JPEG byte hashing; 0 model calls',
        limitations=['One exploratory short seed; no significance/convergence guarantee',
            'CEOSOD full-method overlap unresolved; only abstract/README read',
            'Geometric exposure and capacity do not establish detection gain',
            'No first flow/one-step/UAV/size-aware noise/time-warp claim; no new architecture primitive or likelihood theorem'])
    stage=D/'A60_formal_uploads';assert not stage.exists();stage.mkdir()
    save(P/'formal_metric_contract_CPU_v1.json',metric_proof)
    shutil.copyfile(manifest,P/'dataset_manifest_v1.json')
    save(P/'protocol_FROZEN_v1.json',scope)
    for name in names+['protocol_FROZEN_v1.json']:shutil.copyfile(P/name,stage/name)
    record=dict(status='FORMAL_SOURCE_AND_PROTOCOL_FROZEN_NOT_LAUNCHED',protocol_sha256=sha(P/'protocol_FROZEN_v1.json'),
        source_files=len(names),dataset_manifest_sha256=sha(manifest),initialization_sha256=sha(init),
        stage=str(stage),new_model_forwards=0,new_optimizer_updates=0,reserved_forwards=235542,reserved_updates=29124,
        global_cap=scope['global_with_full_reserved_cap'],metric_contract=metric_proof['status'])
    save(P/'formal_stage_preparation_v1.json',record);print(json.dumps(record))

if __name__=='__main__':main()
