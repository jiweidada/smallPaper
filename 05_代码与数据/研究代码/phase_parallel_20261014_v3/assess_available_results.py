"""Recompute descriptive, same-seed comparisons from checkpoint-bound evidence."""
import datetime
import hashlib
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[2]/'英文小论文/data'
DEST = DATA/'06_第二轮_20261014/结果评估_20261014'
NAMES = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
INIT = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
sources = []


def read(path):
    raw = path.read_bytes()
    sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    return json.loads(raw)


def log_rows(path):
    raw = path.read_bytes()
    sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [row for row in rows if 'test_coco_eval_bbox' in row]
    assert sorted(row['epoch'] for row in rows) == list(range(96))
    return rows


def make_row(seed,arm,epoch,values,source,checkpoint_sha=None):
    assert len(values)==12 and all(-1 <= value <= 1 for value in values)
    return {'seed':seed,'arm':arm,'best_round_one_based':epoch+1,'metrics_raw_0_to_1':dict(zip(NAMES,values)),
            'metrics_percent':dict(zip(NAMES,[value*100 for value in values])),
            'selection':'actual best_stg2 epoch; selected on AP, not APs; training EMA validation',
            'source':str(source),'checkpoint_sha256':checkpoint_sha,'training_complete':True,
            'unified_round2_independent_evaluation_complete':False}


rows=[]
audit_path=DATA/'01_C主线_已完成/C正式配对审计_远端.json'
audit=read(audit_path)
assert audit['status']=='passed'
for run in audit['runs']:
    arm='C' if run['run_id']=='p2_skip_main_seed0' else 'baseline'
    rows.append(make_row(0,arm,run['best_epoch'],run['best_validation_metrics'],audit_path))
for folder in (DATA/'05_相位机制96轮_20261012').iterdir():
    path=folder/'collection_receipt.json'
    if not path.is_file():
        continue
    receipt=read(path)
    seed=int(receipt['job'].split('seed')[-1])
    arm=receipt['job'].split('_full96_seed')[0]
    raw_rows=log_rows(folder/'log.txt')
    selected=[r for r in raw_rows if r['epoch']==receipt['best_epoch_zero_based']]
    assert len(selected)==1 and selected[0]['test_coco_eval_bbox']==receipt['metrics_raw_0_to_1']
    rows.append(make_row(seed,arm,receipt['best_epoch_zero_based'],receipt['metrics_raw_0_to_1'],path,receipt['best_checkpoint_sha256']))
for folder in ['C_seed1','C_mean_seed0']:
    path=DEST/folder/'checkpoint_log_binding.json'
    receipt=read(path)
    raw_rows=log_rows(path.parent/'log.txt')
    selected=[r for r in raw_rows if r['epoch']==receipt['checkpoint_epoch_zero_based']]
    assert len(selected)==1 and selected[0]['test_coco_eval_bbox']==receipt['metrics_raw_0_to_1']
    assert receipt['best_stat']['epoch']==receipt['checkpoint_epoch_zero_based']
    rows.append(make_row(receipt['seed'],receipt['arm'],receipt['checkpoint_epoch_zero_based'],receipt['metrics_raw_0_to_1'],path,receipt['checkpoint_sha256']))
index={(row['seed'],row['arm']):row for row in rows}
assert len(index)==len(rows)==9
for row in rows:
    baseline=index[row['seed'],'baseline']
    row['minus_same_seed_baseline_pp']={name:row['metrics_percent'][name]-baseline['metrics_percent'][name] for name in NAMES}
pairs=[]
for seed in [0,1]:
    for other in ['lowpass','singlephase']+(['C-mean'] if seed==0 else []):
        first,second=index[seed,'C'],index[seed,other]
        pairs.append({'seed':seed,'comparison':'C_minus_'+other,
                      'delta_pp':{name:first['metrics_percent'][name]-second['metrics_percent'][name] for name in NAMES}})
gate=[]
for seed in [0,1]:
    gain=index[seed,'C']['minus_same_seed_baseline_pp']['APs']
    checks=[]
    for other in ['lowpass','singlephase']:
        other_gain=index[seed,other]['minus_same_seed_baseline_pp']['APs']
        checks.append({'control':other,'C_minus_control_APs_pp':gain-other_gain,
                       'gap_at_least_0_5':gain-other_gain>=.5,'control_below_half_C_gain':other_gain<gain/2})
    gate.append({'seed':seed,'C_gain_APs_pp':gain,'checks':checks,
                 'passes_historical_gate':all(x['gap_at_least_0_5'] and x['control_below_half_C_gain'] for x in checks)})
descriptive={}
for arm in ['C','lowpass','singlephase']:
    values=[index[seed,arm]['minus_same_seed_baseline_pp']['APs'] for seed in [0,1]]
    descriptive[arm]={'n_training_seeds':2,'mean_delta_APs_pp':statistics.mean(values),'sample_sd_delta_APs_pp':statistics.stdev(values),'seed_values_pp':values,
                      'inference':'descriptive only; no significance or equivalence test'}
pair_path=DATA/'01_C主线_已完成/C正式配对结果_远端.json'
pair=read(pair_path)
testdev=[row for row in pair['metrics'] if row['split']=='test-dev']
testdev_delta={name:(next(r for r in testdev if r['variant']=='p2_skip')[name]-next(r for r in testdev if r['variant']=='baseline')[name])*100 for name in NAMES}
placement_path=DATA/'00_实验计划与证据索引/C完整重复验收辅助_20261003/dual_view_execution_20261004_v1/main_pair_closeout_v1/collection_actual_20261004_183202_868498/C8_O8_root_scientific_acceptance_actual_v1.json'
placement=read(placement_path)
placement_delta={name:(placement['metrics_C'][name]-placement['metrics_O'][name])*100 for name in NAMES}
neighbor_path=DATA/'00_实验计划与证据索引/C近邻UAV_DETR先例_原始方法核对_20261003_0343_v1.json'
neighbor=read(neighbor_path)
fallacies=[
    {'name':'Simpson reversal','status':'checked','note':'Keep seeds separate. Lowpass overall AP changes sign across seeds; C APs gains have the same sign.'},
    {'name':'Ecological inference','status':'limitation','note':'Aggregate AP/AP75/AR cannot establish per-object localization or subpixel position preservation.'},
    {'name':'Berkson selection','status':'scope_limit','note':'One selected drone dataset and detector; do not infer universal applicability.'},
    {'name':'Collider selection','status':'diagnostic_limit','note':'Future common-detection localization statistics are conditional on joint detection, not population causal effects.'},
    {'name':'Base-rate neglect','status':'pending_diagnostic','note':'Tiny input-size bin GT counts and class composition await D; no sensitivity/prevalence claim made.'},
    {'name':'Regression to mean','status':'caution','note':'C was selected during earlier screens; paired controls and test-dev help but do not remove selection bias.'},
    {'name':'Survivorship','status':'checked_with_limits','note':'All five first-round arms included; unfinished new arms retained as pending, not discarded or called negative.'},
    {'name':'Look elsewhere','status':'caution','note':'Many exploratory variants exist; no p-value or significance claim from this selected comparison.'},
    {'name':'Forking paths','status':'checked_with_limits','note':'Preserve historical thresholds and AP-selected stg2 checkpoint rule; no choosing the maximum APs epoch.'},
    {'name':'Causal overstatement','status':'caution','note':'Mean removes subcell differences and changes gradient structure; rank controls are pending. Shuffle tests establish learned dependence only.'},
    {'name':'Reverse causality','status':'not_applicable_to_intervention_order','note':'Architecture intervention precedes training; this does not by itself establish the proposed positional mechanism.'},
]
report={'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','origin_mode':'validate','verification_status':'ANALYZED','version':'available_results_20261014_v1'},
        'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'common_init_sha256':INIT,
        'scope':'Nine completed 96-epoch trajectories; exact checkpoint-bound training EMA metrics, awaiting unified independent evaluation. No new training or GPU inference.',
        'rows':rows,'same_seed_pairs':pairs,'two_seed_descriptives':descriptive,'historical_first_round_gate':gate,
        'historical_gate_passes_both_seeds':all(r['passes_historical_gate'] for r in gate),
        'seed0_independent_testdev_delta_pp':testdev_delta,'seed0_independent_testdev_raw_rows':testdev,
        'cost_evidence':{'training_parameter_delta':pair['costs']['p2_skip']['training_parameters']-pair['costs']['baseline']['training_parameters'],
                         'relative_FLOPs_overhead':pair['estimated_flops_overhead'],'latency':pair['latency'],'source':str(pair_path)},
        'placement_C8_minus_O8_pp':placement_delta,'placement_limitations':placement['limitations'],
        'prior_art_audit_status':neighbor['status'],
        'limitations':['Two seeds for C/lowpass/singlephase; one completed seed for C-mean. No significance/equivalence inference.',
                       'Historical C seed1 resumed after epoch3 without complete RNG restoration; accepted effective recipe match is not uninterrupted bitwise reproducibility.',
                       'C-mean best_stg2 is round94; its round96 APs17.92 is not the selected checkpoint score.',
                       'No completed rank16/rank16-mean pair, SOEP-lite accuracy, second-dataset study or new D/A results yet.',
                       'Original seed0 independent val/testdev evidence is separate from this uniform training-EMA table.'],
        'fallacy_scan':{'coverage':'11/11','items':fallacies},'source_manifest':sources}
(DEST/'同种子完整指标与判据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('BEST_CHECKPOINT_TRAINING_EMA_TABLE')
for arm in ['baseline','C','lowpass','singlephase','C-mean']:
    print(arm,[(seed,index[seed,arm]['best_round_one_based'],
        *[round(index[seed,arm]['metrics_percent'][k],4) for k in ['AP','AP75','APs','ARs']],
        *[round(index[seed,arm]['minus_same_seed_baseline_pp'][k],4) for k in ['AP','APs']]) for seed in [0,1] if (seed,arm) in index])
print('PAIR_GAPS',[(r['seed'],r['comparison'],round(r['delta_pp']['APs'],4)) for r in pairs])
print('OLD_GATE',report['historical_gate_passes_both_seeds'])
print('TESTDEV_DELTA',json.dumps({k:round(v,4) for k,v in testdev_delta.items()}))
print('C_MEAN_MINUS_CONTROLS',[(arm,index[0,'C-mean']['metrics_percent']['APs']-index[0,arm]['metrics_percent']['APs']) for arm in ['lowpass','singlephase']])
print('DESCRIPTIVE',json.dumps(descriptive,ensure_ascii=False))
print('SOURCE_FILE',DEST/'同种子完整指标与判据.json')
