"""One-time protocol creation before real prediction/GT task evaluation."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent
ENG=NEW.parent
AP='/root/c10_region_restore_20261029_v1/architecture_probe_v1'
REMOTE=AP+'/a27_same_budget_task_cpu_20261030_v1'
A26=NEW/'A26_查询扩容接口与缓存核验_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    p=ROOT/name;assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    assert not (ROOT/'task_protocol_FROZEN.json').exists()
    for name in ['o27_metrics.py','metrics_preflight.py','run_decoder_task.py','analyze_decoder_task.py']:
        ast.parse((ROOT/name).read_text(encoding='utf-8'))
    pre=read(ROOT/'metric_preflight_result.json')
    assert pre['status']=='SYNTHETIC_METRIC_PREFLIGHT_PASS'
    assert pre['metric_script_sha256']==sha(ROOT/'o27_metrics.py')
    assert pre['self_sha256']==sha(ROOT/'metrics_preflight.py')
    receipts=ENG/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'/'ssh_operation_receipts.jsonl'
    matches=[json.loads(x) for x in receipts.read_text(encoding='utf-8').splitlines() if x.strip()]
    matches=[x for x in matches if x.get('id')=='a27_runtime_source_versions' and x.get('action')=='exec']
    assert len(matches)==1 and matches[0]['exit_code']==0
    runtime=json.loads(matches[0]['stdout'])
    save('runtime_source_versions.json',runtime)
    p26=read(A26/'cpu_interface_protocol_FROZEN.json')
    result26=read(A26/'verified_engineering_evidence'/'result.json')
    raw=read(ROOT/'raw_annotations_verified.json')
    inventory=read(A26/'cache_memory_roundtrip_inventory.json')['rows']
    toolkit=read(ROOT/'official_toolkit_source_manifest.json')
    for f in toolkit['files']:
        data=(Path(toolkit['source_root'])/f['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==f['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==f['git_blob_sha1']
    source_audit={
      'repo':'https://github.com/VisDrone/VisDrone2018-DET-toolkit','commit':toolkit['commit'],
      'tree':toolkit['tree'],'files_byte_and_git_blob_verified':len(toolkit['files']),
      'reading_scope':['evalDET.m','utils/saveAnnoRes.m','utils/dropObjectsInIgr.m',
        'utils/createIntImg.m','utils/evalRes.m','utils/compOas.m','utils/VOCap.m','utils/calcAccuracy.m'],
      'source_conclusions':[
        'Category0 rectangles form a union ignore bitmap. Both GT and detections with union-region fraction>=0.5 are dropped before evaluation.',
        'Integral-image formula uses1-based inclusive bitmap but differences of same-size cumulative image; rounding half away from zero; pos coordinates and dimensions min1.',
        'Evaluation categories1..10; category11 is not scored and does not itself create ignored-region geometry.',
        'Native maxDets truncates globally per image before class filtering. Continuous VOCap integral across IoU0.5:0.05:0.95, maxDets500.',
        'calcAccuracy appends a category once per image containing that category and does not unique the evalClass list; its reported average is presence-weighted.',
      ],
      'our_policy':'COCO101 AP@500 and APsmall with source-aligned ignored-region preprocessing, not claimed as official VisDrone benchmark reproduction. A source-inspired continuous presence-weighted score is descriptive only.',
      'reading_limitations':'Pinned MATLAB source fully inspected for these functions; MATLAB runtime not executed. COCO evaluator is separately source-pinned and checked with synthetic fixtures.',
      'real_predictions_evaluated_before_freeze':False,
    }
    save('official_evaluation_source_audit.json',source_audit)
    save('A26_decimal_transcription_correction.json',{
      'old_report':str(NEW/'62_A26查询扩容接口与完整解码器对照核验.txt'),
      'correct_source':str(A26/'verified_engineering_evidence'/'result.json'),
      'observed_native_prefix_box_max_abs':result26['prefix_max_absolute_differences']['NATIVE500']['boxes'],
      'reported_decimal_transcription_issue':True,
      'effect':'Only long decimal in readable report62 was mistyped. Receipt63 and raw JSON were correct; no scientific status changes or model rerun.',
      'historical_artifacts_modified':False,
    })
    cases=[]
    for i,(r,c) in enumerate(zip(raw['cases'],inventory)):
        assert Path(c['file']).name=='case%02d.npz'%i
        cases.append({'index':i,'file':r['file'],'prefix':r['prefix'],'width':r['width'],'height':r['height'],
          'cache_file':AP+'/a15_sampling_center_v1/captures/case%02d.npz'%i,
          'cache_sha256':c['sha256'],'memory_sha256':c['memory_sha256'],
          'annotation_sha256':r['annotation_sha256']})
    assert len(cases)==12
    deps=[x for x in p26['dependencies'] if x['path']!=p26['cache_file']]
    for name in ['result.json','predictions_no_GT.npz']:
        deps.append({'path':AP+'/a26_query_prefix_interface_cpu_20261030_v1/'+name,
                     'sha256':sha(A26/'verified_engineering_evidence'/name)})
    deps.append({'path':runtime['COCOeval_source'],'sha256':runtime['COCOeval_sha256']})
    file_names=['run_decoder_task.py','analyze_decoder_task.py','o27_metrics.py','metrics_preflight.py',
      'metric_preflight_result.json','raw_annotations_verified.json','official_toolkit_source_manifest.json',
      'official_evaluation_source_audit.json','annotation_QC_before_task_effects.json','runtime_source_versions.json']
    protocol={'status':'FROZEN_EXPLORATORY_SAME_BUDGET_TASK',
      'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'candidate':'O_QUERY_BUDGET_EXPANSION_CONTEXT','mainline_validated':False,
      'design':'Exploratory paired decoder-only task on12 previously reused cached images. No independent confirmation, training, image encoding or benchmark reproduction.',
      'arms':['BASE250','NATIVE500','TWOWAY500','ONEWAY500'],
      'query_initializer':'Exact native fixedTopK500 from unchanged encoded memory and native heads; first250 are exact prefix. PAQI density selection bypassed for this controlled pool.',
      'mask_contract':{'BASE250':'Unmasked250 prefix reference',
        'NATIVE500':'Unmasked500 same pool',
        'TWOWAY500':'Old/new groups mutually blocked. A standard grouping control, not a trained GroupEnsemble reproduction.',
        'ONEWAY500':'Old queries cannot read new queries; new queries may read all500. Known adjacency pattern, not novelty by itself.'},
      'new_complete_decoder_call_cap':45,'decoder_wall_seconds':180,
      'new_whole_detector_call_cap':0,'new_optimizer_update_cap':0,'new_GPU_call_cap':0,
      'reuse_A26_arms':['BASE250','NATIVE500','ONEWAY500'],
      'A26_result':AP+'/a26_query_prefix_interface_cpu_20261030_v1/result.json',
      'A26_predictions':AP+'/a26_query_prefix_interface_cpu_20261030_v1/predictions_no_GT.npz',
      'decoder_seed':p26['seed'],'CPU_threads':4,'prefix_absolute_tolerance':p26['prefix_absolute_tolerance'],
      'loader':p26['loader'],'source_commit':p26['source_commit'],'state_digest':p26['state_digest'],
      'checkpoint_sha256':p26['checkpoint_sha256'],'cases':cases,
      'metrics':{
        'postprocess':'Same fixed Top500 over flattened sigmoid class scores across all12 classes for every arm, includingBASE250; cxcywh->xyxy then original[W,H,W,H]; filterlabels1..10 afterTopK without refill; source-pinned region exclusion. No score threshold or NMS.',
        'primary':['COCO101 APsmall original-area<=1024 at maxDets500','COCO101 APall at maxDets500'],
        'harm':'One argmax class per raw query, no confidence threshold, labels1..10 and same region filter. BASE250 score-desc class-aware greedy one-to-one GT matching atIoU0.5. Among its matched smallGT, count absence of ANY sameclass raw-query IoU>=0.5 inNATIVE500 (or ONEWAY500). Conservative uncompensated localization/class coverage loss, not confidence-calibrated detection success.',
        'secondary':'AP50 and source-inspired continuous presence-weighted AP only; no official benchmark score claim.',
        'GT':'Original raw xywh, not clipped cache boxes800; GT score>0/category1..10, source-aligned ignored-region filtering.',
      },
      'statistics':{'paired_bootstrap_replicates':10000,'seed':427031,'analysis_wall_seconds':1800,
        'unit':'12 file prefixes, one image each; prefix is a sampling block, not proven independent flight identity.',
        'interval':'Paired multinomial12-draw percentile95%, linear quantile. Same draws for all arms. Available-GT classes averaged per replicate; class support may change.',
        'ties':'Canonical stable global score order fixed once per matching block; integer multiplicities duplicate adjacent detection records. Conditional fixed-rank bootstrap; tie-order alternatives not evaluated.',
        'zero_harm_denominator':'Retain replicate with harm rate0; disclose count.',
        'interpretation':'Exploratory stability intervals on a reused cohort; not power calculation, confirmatory significance or generalization proof.'},
      'screen_gates_all_required':{
        'baseline_small_exposure':'base_correct_small>=24 AND prefixes_with_base_small_TP>=6',
        'native_uncompensated_harm':'harm_rate>=0.02 AND paired_lower95>0',
        'APsmall_vs_native':'O-N>=0.005 AND paired_lower95>0',
        'AP_no_material_loss':'APall O-N>=0 AND paired_lower95>=-0.002',
        'specificity_vs_group_mask':'APsmall O-T>=0.002 AND paired_lower95>0',
        'coverage_vs_base':'APsmall O-B>=0.003 AND paired_lower95>0'},
      'threshold_scope':'Engineering screening choices made before task outcomes, not literature-derived effect sizes or publication criteria.',
      'stop_and_decision':'First integrity, finiteness, cap, wall or prefix-identity failure stops; no model retry. Any task gate failure does not advance this increment. All gates passing still requires novelty/independent validation, not a found mainline. No automatic training or new candidate.',
      'annotations_sha256':sha(ROOT/'raw_annotations_verified.json'),
      'files':[{'path':name,'sha256':sha(ROOT/name)} for name in file_names],
      'dependencies':deps,
    }
    save('task_protocol_FROZEN.json',protocol)
    save('protocol_creation_receipt.json',{'status':'FROZEN_BEFORE_REAL_TASK_EFFECTS',
      'protocol_sha256':sha(ROOT/'task_protocol_FROZEN.json'),'files':len(file_names),'cases':12,
      'new_decoder_calls_executed_at_freeze':0,'new_real_task_AP_evaluations_at_freeze':0,
      'synthetic_preflight_failed_v1_retained':True,'synthetic_preflight_corrected_v2_pass':True})
    print(json.dumps({'status':'FROZEN_BEFORE_REAL_TASK_EFFECTS','cases':12,'new_decoder_call_cap':45,
      'mainline_validated':False,'protocol_sha256':sha(ROOT/'task_protocol_FROZEN.json')}))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
