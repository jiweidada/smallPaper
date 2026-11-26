"""Freeze sources, owners, intervention, controls and budget before execution."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
inputs=ROOT/'outputs/sampling_location_intervention_20261011_v1/inputs'
meta=read(inputs/'inputs_metadata.json')
assert meta['status']=='PASS_VERIFIED_CPU_INPUT_PACKAGE_WITH_NATIVE_GT_OWNERS'
assert sha(inputs/'inputs.zip')==meta['archive_sha256']
assert (inputs/'inputs.zip').stat().st_size==meta['archive_bytes']
assert sha(HERE/'prepare_inputs.py')==meta['prepare_script_sha256']
assert sha(inputs/'owners.json')==meta['owners_sha256']
owners=read(inputs/'owners.json')
source=read(HERE.parent/'decoder_sampling_geometry_20261011_v5/protocol.json')
for name in ('runner.py','controller.py','analyze_downloaded.py'):
    compile((HERE/name).read_text(encoding='utf8'),name,'exec')
seeds={}
for seed,prefix,gpu in ((1,'/25085400120','GPU-d29c55c5-02ab-1e69-c978-0b28f2102003'),(2,'/24085403014','GPU-b8084aa9-0176-df73-41fb-03307ded1be3')):
    old=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1/extracted'
    receipt=read(old/'prepared/receipt.json')
    assert receipt['paths']['screen_root']==f'{prefix}/DEIM/experiments/scale_protection12_20261010_v1/seed{seed}_target_protect'
    rows=owners['seeds'][str(seed)]['owners']
    assert len(rows)==meta['owner_counts'][str(seed)]=={1:88,2:85}[seed]
    assert len({(r['image_id'],r['query']) for r in rows})==len(rows)
    seeds[str(seed)]=dict(bundle=receipt['paths']['screen_root']+'/bundle',
        closeout_tools=receipt['paths']['experiment_root']+'/post_training_tools_v1',gpu_uuid=gpu,
        checkpoint_sha256=source['sources'][str(seed)]['checkpoint_sha256'],
        owner_count=len(rows),max_suffix_calls=2+3*len(rows),
        actual_scale_uint32=source['actual_EMA_sampling_buffers'][str(seed)][0]['uint32_bits'])
p=dict(schema='sampling_location_suffix_intervention_v1',created_local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    seeds=seeds,code_sha256={n:sha(HERE/n) for n in ('runner.py','controller.py')},
    analysis_sha256=sha(HERE/'analyze_downloaded.py'),registration_sha256=sha(__file__),
    prepare_inputs_sha256=sha(HERE/'prepare_inputs.py'),inputs_sha256=meta['archive_sha256'],inputs_bytes=meta['archive_bytes'],
    inputs_manifest_sha256=meta['manifest_sha256'],owners_sha256=meta['owners_sha256'],
    plan_sha256=sha(ROOT/'outputs/sampling_audit_20261011_v1_next_diagnostic/下一项诊断_一页计划.txt'),
    source_sampling_protocol_sha256=sha(HERE.parent/'decoder_sampling_geometry_20261011_v5/protocol.json'),
    intervention='Only first-layer sampling_locations of one unique owner query are translated, all heads and levels equally. Saved norm1, offsets, weights, values and original initial reference stay fixed; downstream candidate decoder computes normally.',
    arms=['N','T','Cplus','Cminus'],delta='GT center minus original encoder-reference center, normalized XY; Cplus/Cminus rotate delta by +/-90 degrees.',
    owners_rule='All native input-short-side<4px GT with positive best-pre-query IoU; lowest query index breaks equal IoU, lowest annotation ID owns a shared query; no selection by final success.',
    controls=dict(cross_maxabs=.0025,cross_relative_L2=1e-5,boxes_maxabs=1e-6,logits_maxabs=1e-5,logits_relative_L2=1e-5,GT_geometry_boolean_disagreements=0),
    control_gate_scope='Each seed must pass both original batch controls before any intervention in that seed. Seeds run independently; no cross-seed four-control barrier is claimed.',
    timeout_seconds=900,total_suffix_call_limit=sum(s['max_suffix_calls'] for s in seeds.values()),
    planned_control_calls=4,planned_intervention_calls=519,original_batch_size=4,
    forbidden_calls=['Full detector','Backbone','Encoder','First self attention','BASE forward','Evaluator','Optimizer','Training'],
    primary_outcomes=['Fixed owner-query IoU change vs same-batch N','All-GT geometry gains and losses within each individual counterfactual, thresholds .5/.75, all size strata'],
    secondary_outcomes=['Owner true-class logit change','Cross-attention output change','Bilinear support mass and boundary condition'],
    boundary_match='T and each direction control have equal normalized_inside_after and abs(bilinear_valid_mass_after difference)<=1e-6; report matched and unmatched, do not discard.',
    no_cross_case_additive_whole_image_gain_claim=True,no_independent_GT_significance_claim=True,
    no_AP_or_causal_root_or_deployable_method_claim=True,no_oracle_to_architecture_promotion_gate=True,
    no_automatic_retry=True,no_epochs_added=True)
assert p['total_suffix_call_limit']==523
with (HERE/'protocol.json').open('x',encoding='utf8') as f:json.dump(p,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(status='REGISTERED_BEFORE_SUFFIX_INTERVENTION',protocol_sha256=sha(HERE/'protocol.json'),total_suffix_call_limit=523,training_epochs=0)))
