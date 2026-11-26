"""Preserve deployed queue helper proof and current same-epoch component evidence."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';MAIN=OUT/'stem_component12_20261011_v1';QUEUE=OUT/'stem_component_queue_pulse_20261011_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
meta=read(QUEUE/'observation_transport_v1.json');observations=[]
for seed in (1,2):
    p=QUEUE/f'remote_observation_seed{seed}_v1.json';m=meta[str(seed)]
    assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
    o=read(p);assert o['code_SHA_verified'] and o['component_SHA_verified'] and o['failure'] is None
    assert o['helper']['same_live'] and o['conductor']['same_live'] and o['conductor']['state']=='T' and o['reference']['same_live']
    assert o['CPU_validation']['status']=='PASS_DISPOSABLE_CPU_ONE_DISPATCH_AND_REFERENCE_HANDOFF_FIXTURE'
    assert o['CPU_validation']['real_trainer_signals']==o['CPU_validation']['model_forwards']==0
    assert o['live']['GPU']['compute_processes']==4 and not o['pulses'] and o['completion'] is None
    observations.append(o)
write(QUEUE/'execution_state_v1.json',dict(status='LIVE_CPU_HELPERS_WAIT_FOR_NATIVE_RESOURCE_AUTHORIZATION',evidence='remote_observation_seed1/2_v1.json',
    helpers={str(o['seed']):o['helper'] for o in observations},goal_complete=False,
    training_code_changed=False,model_or_training_configuration_changed=False,queued_arms_started_by_helper=0,
    minimum_condition='Actual GPU compute apps<=2 and free>=6144MiB; unchanged native authorizer max3 remains active.',
    explanation='The existing conductor was held for the full reference training. Helpers can pulse its existing pending dispatches sooner when native capacity allows, then hand final resume back to the reference coordinator.',
    no_training_retry_or_extension=True,baseline_forwards_added=0,CPU_fixture_status='PASS_BOTH_SERVERS',
    next='Revalidate helper/conductor/reference PID+start ticks and GPU state; when a native dispatch appears verify its actual trainer, input parity and epoch<=12. Preserve failures rather than restarting training.'))
progress=read(MAIN/'progress_v20.json');verification=read(MAIN/'training_observation_v20_transport_verification.json')
assert progress['snapshot_sha256']==sha(MAIN/'training_observation_v20.json')==verification['verified_snapshot_sha256']
latest=[]
for seed in (1,2):
    for arm in ('original_stem','detail_lowres','highfreq_rgb'):
        latest.append(max([r for r in progress['rows'] if r['seed']==seed and r['arm']==arm],key=lambda r:r['epoch']))
paper=ROOT/'英文小论文/文献_相关原始论文_20261004/14_DEIM_CVPR2025_improved-matching.pdf'
criterion=ROOT/'英文小论文/code/p2_candidate_seed1_full132_20261007_v1/engine/deim/deim_criterion.py'
assert sha(criterion)=='16cdd80b2197b8def8ee3f5f19f6311f5221c3da08abacb99f50d346610a2c30'
write(QUEUE/'research_and_progress_note_v1.json',dict(verification_status='ANALYZED',latest_same_epoch_rows=latest,
    paper=dict(primary='https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html',local=str(paper),sha256=sha(paper),
        examined='Abstract/introduction and DenseO2O/MAL formulation; no new paper downloaded',lesson='Distinguish positive matching quantity from matching quality and inspect both localization and classification response.'),
    criterion_source_sha256=sha(criterion),native_positive_target='matched IoU^1.5 in this bound recipe',
    boundary='This is a problem-discovery lesson and source-level explanation. Earlier loss audit already showed the confound; no new training matching history or unique Stem cause established.',
    fixed12_efficacy_established=False,novelty_established=False,new_detector_forwards=0,new_training_epochs=0))
main=read(MAIN/'execution_state_v15.json');live=read(MAIN/'training_observation_v20.json')
main.update(supersedes='execution_state_v15.json',evidence='training_observation_v20.json',baseline_comparison='progress_v20.json',
    previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    queue_helper_state='../stem_component_queue_pulse_20261011_v2/execution_state_v1.json',
    finding='Actual unchanged queue source verified; two live CPU helpers now allow safe native queued dispatches before reference12 ends when the native max3 capacity allows. No queued training has started yet; both cards still have4 compute apps.',
    next='Continue bounded training. Revalidate helper dispatch events, collect fixed12 outcomes and then perform the independently prepared exact EMA/selection replay; do not infer structure efficacy from early logs.')
for seed in (1,2):
    main['current_live_trainers']['seed'+str(seed)]={a:live['seed'+str(seed)]['jobs'][a]['processes'] for a in ('original_stem','detail_lowres','highfreq_rgb')}
    assert all(p['same_live'] and p['state']!='T' for ps in main['current_live_trainers']['seed'+str(seed)].values() for p in ps)
write(MAIN/'execution_state_v16.json',main)
full=read(OUT/'original_stem12_20261011_v1/execution_state_v18.json');full.update(supersedes='execution_state_v18.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v16.json')
write(OUT/'original_stem12_20261011_v1/execution_state_v19.json',full)
print(json.dumps(dict(status='PROGRESS_SAFE_EXISTING_QUEUE_HELPERS_DEPLOYED',helpers={str(o['seed']):o['helper'] for o in observations},latest_rows=latest),ensure_ascii=False))
