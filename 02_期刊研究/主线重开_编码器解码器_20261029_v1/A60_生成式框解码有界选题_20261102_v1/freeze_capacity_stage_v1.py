from pathlib import Path
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
STAGE=P.parent/'A54_场景尺度约束候选_20261101_v1/A60_uploads'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(4<<20),b''):h.update(v)
    return h.hexdigest()

def main():
    proof=json.loads((P/'CPU_constructor_stage_v1/result.json').read_text())
    assert proof['status']=='PASS_CPU_CONSTRUCTOR_WARM_MAPPING_AND_STANDARD_COMPONENT_CHECKS'
    assert json.loads((P/'loss_interface_CPU_v1.json').read_text())['status']=='PASS_NONEMPTY_AND_EMPTY_GT_LOSS_INTERFACE'
    geometry=json.loads((P/'geometry_D0_remote_v1/result.json').read_text())
    meta=json.loads((P/'geometry_D0_remote_v1/selected_images.json').read_text())
    counts=sorted(v['clipped_valid_GT'] for v in meta);median=counts[len(counts)//2]
    typical=min(meta,key=lambda z:(abs(z['clipped_valid_GT']-median),z['name']))['name']
    names=['box_progress_conditioning_v1.py','torch_detector_parts_v1.py','head_query_time_A60_v1.py',
        'model_A60_v1.py','criterion_A60_v1.py','box_ops_A60_v1.py','runtime_A60_v1.py','capacity_A60_v1.py']
    scope=dict(phase='A60_CAPACITY_ONLY',arms=['B','M','Q'],real_train_images=[geometry['largest_counts'][0][1],typical],
        source_sha256={n:sha(P/n) for n in names},initialization_sha256=proof['common_init_sha256'],
        maximum_detector_forwards=9,maximum_optimizer_updates=6,formal_epochs=0,maximum_seconds=600,
        training_microbatch=1,formal_gradient_accumulation_proposed=8,
        train_query_policy='max(300,32*ceil(max_GT_in_microbatch/32)); maximum1024, never drop GT',
        inference_queries=300,inference_steps=1,inference_capacity_batch=2,
        backend='BF16 backbone, FP32 RoI/head/loss, TF32 off, same flags across arms',
        purpose='Two real finite-gradient/optimizer steps per mode plus one GT-free inference call per mode; all resulting weights discarded',
        no_automatic_retry=True,formal_training_allocated=False,global_before=dict(model_forwards=276634,optimizer_updates_including_capacity=233600),
        global_with_full_capacity_cap=dict(model_forwards=276643,optimizer_updates_including_capacity=233606),
        limits='Capacity and input-interface evidence only, not scientific training, learned efficacy or novelty proof')
    path=P/'capacity_scope_FROZEN_v1.json'
    with path.open('x',encoding='utf-8') as f:json.dump(scope,f,indent=2)
    files=[P/n for n in names]+[path,P/'CPU_constructor_stage_v1/common_init_A60_v1.pth']
    for source in files:
        dest=STAGE/source.name;assert not dest.exists();shutil.copyfile(source,dest)
    record=dict(source_files=len(names),scope_sha256=sha(path),initialization_sha256=proof['common_init_sha256'],
        real_train_images=scope['real_train_images'],maximum_detector_forwards=9,maximum_updates=6,
        local_stage=str(STAGE),staged_files=[f.name for f in files])
    with (P/'capacity_stage_preparation_v1.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record))

if __name__=='__main__':main()
