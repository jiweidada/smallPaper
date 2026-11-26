from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;P=B/'A35_新架构与查询生存最近工作筛查_v1'
REMOTE='/root/c10_region_restore_20261029_v1/architecture_probe_v1/a36_lqe_scores_cpu_20261031_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    if Path(p).exists():assert read(p)==x;return
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
prior=read(B/'81_A35原文源码与原生前缀核验回执.json')
for n,h in prior['files_sha256'].items():assert sha(n)==h,n
assert len(prior['files_sha256'])==115
save(D/'handoff_verified.json',dict(previous_goal_turn='progress:A35 auditedclosestquerymethods,completed37CPUdecoderprefixcalls and rejected thequery-retentioncandidate;posthocmissandLQEmetadata changednextaction.',
    prior_A35_files_verified=115,prior_receipt_sha256=sha(B/'81_A35原文源码与原生前缀核验回执.json'),
    whole_detector_forwards_before=3168,optimizer_updates_before=528,complete_decoder_only_calls_before=61,
    A35_37_cap_remains_closed=True,initial_new_model_allocation=0))
for old,new in [(P/'pinned_o27_metrics.py',D/'pinned_o27_metrics.py'),(P/'raw_annotations_verified.json',D/'raw_annotations_verified.json'),
    (B/'A30_YOLO26双头原生读出核查_v1/independent_coco_reaggregation.py',D/'pinned_A30_independent.py')]:
    if new.name=='pinned_A30_independent.py':assert sha(old)==read(B/'71_A30双头任务与原始支持核验回执.json')['files_sha256'][str(old)]
    if new.exists():assert sha(new)==sha(old)
    else:shutil.copy2(old,new)
cfg35=read(P/'native_prefix_protocol_FROZEN.json');old_manifest=read(P/'downloaded_evidence/file_manifest.json')['files_sha256']
cases=[]
for c in cfg35['cases']:
    name='captures/case%02d.npz'%c['index'];assert sha(P/'downloaded_evidence'/name)==old_manifest[name]
    cases.append(dict(index=c['index'],file=c['file'],width=c['width'],height=c['height'],
                      capture=cfg35['remote_output']+'/'+name,capture_sha256=old_manifest[name],
                      local_capture=str(P/'downloaded_evidence'/name)))
source_zip=B/'official_assets_v1/Dome_DETR_2dde3bc_v1/author_source.zip'
with zipfile.ZipFile(source_zip) as z:
    names=[n for n in z.namelist() if n.endswith(('dome_decoder.py','dome_criterion.py','Dome-S-VisDrone.yml','dome_hgnetv2.yml'))]
    source=[]
    for name in names:
        data=z.read(name);target=D/'source'/name.split('/',1)[1];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:
            with target.open('xb') as f:f.write(data)
        source.append(dict(path=str(target),sha256=sha(target)))
save(D/'source_operator_audit.json',dict(status='DIAGNOSIS_SOURCE_AUDIT_NOT_NOVELTY_APPROVAL',source_commit=cfg35['source_commit'],files=source,
    facts=['Finaldec_score_head[2]isLinear;LQE adds a singlelearned distribution-statistic offset to everyclass logit.',
           'LQE uses per-side softmax,topk probabilities andtheirmean;it doesnot use binpositions directly.',
           'Selected sourceconfig usesVFL on finalcombined pred_logits;gamma0.8 inSconfig isLRscheduler gamma,notVFL gamma.',
           'GFLV2 Sec3.2 multiplies class andIoU representations andsupervisesjointscore;Dome uses additive logits initsderivedhead.'],
    inferences=['RemovingLQEis a functionalablation at thepinnedjoint-trainedparameterization,not a separatelytrainedclassificationbaseline.',
                'Classifierbias+g andLQElastbias-g preservefinalsum;absolutecomponentvalues cannotbeinterpretedas independentlycalibrated probabilities.',
                'ExactTopk-statistic/scale invariance alone isexistingdesign,not a newalgorithm.'],
    limits=['Sourceconfig is not a completeauthortrainingreceipt.','Noencoderorbackground-interferencecauseestablished.','GFLV2andD-FINEbenchmarknotreproduced.']))
seed=361031;draws=np.random.default_rng(seed).integers(0,12,size=(1000,12))
if (D/'bootstrap_draws_FROZEN.npz').exists():assert np.array_equal(np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws'],draws)
else:np.savez_compressed(D/'bootstrap_draws_FROZEN.npz',draws=draws)
fixture=subprocess.run([sys.executable,'-X','utf8',str(D/'analyze_scores.py'),'--fixture'],check=True,capture_output=True,text=True,encoding='utf-8')
save(D/'metric_fixture_receipt.json',json.loads(fixture.stdout))
cfg=dict(status='FROZEN_LQE_SCORE_DECOMPOSITION_TWO_ARMS',phase='A36',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 question='Doesremovingthelearnedqualityoffset improve small-objectranking atthe same nativepostprocessing andevaluation,orshouldquality-offsetharmbedeprioritized?',
 remote_output=REMOTE,arms=['NATIVE','RAW_CLASS'],cases=cases,
 checkpoint=cfg35['dependencies'][2]['path'],checkpoint_sha256=cfg35['checkpoint_sha256'],
 native_prediction_reference=cfg35['remote_output']+'/raw_native_predictions_without_region_filter.json',
 native_prediction_reference_sha256=old_manifest['raw_native_predictions_without_region_filter.json'],
 caps=dict(head_linear_calls=12,whole_detector_calls=0,decoder_calls=0,image_encoder_calls=0,optimizer_updates=0,GPU_calls=0,wall_seconds=180),
 execution=dict(device='CPU',dtype='float32',threads=4,no_detector_construction=True,source_weights='Originaldecoder.dec_score_head.2fromunchangedcheckpoint',
                same_boxes_and_query_features=True,no_GTin_collector=True),
 metric=dict(type='Source-alignedignorepreprocessing plusCOCO101@500,notfullofficialVisDronescore.',
             original_area_small='area<=1024',AP_primary='APsmallaveragedIoU0.5:0.95',AP75_required=True,smallAP75_required=True,
             native_cap='Torchsigmoid,class-pairTop500 beforevalid-label1..10andignore-regionfilter;no refill.',
             FP_budget=111,bootstrap=dict(draws=1000,seed=seed,unit='pairedimage',
                 AP='Literalreplicatedimage-records withfreshglobalstable scoreorder;matchingrecomputedperarm.',
                 FP='Eacharmre-estimatesitsown global complete-score-block prefix perdraw.'),
             all_question_gates=['RAWminusNATIVEAPsmall>=0.005','APsmall95%lower>0','overallAPdelta>=-0.002','APsmall75delta>=-0.002','smallTP@FP111delta>=0']),
 cocoeval_source='/root/c10_region_restore_20261029_v1/venv/lib/python3.9/site-packages/pycocotools/cocoeval.py',
 cocoeval_sha256='988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267',
 interpretation_limits=['12 reusedcontexts;not independentblindorcrosssequencegeneralization.',
   'Ablation removesqualityoffset onlyatfinalscore;queryfeatures/boxes andlatentstates do notchange.',
   'Joint-trained rawhead isnot a cleanclassprobability estimator;absolute componentoffsets haveanadditivegauge.',
   'Qualityheadbenefitorharm doesnot byitselfprove novelalgorithm,encoderlossorbackgroundinterference.',
   'Noweight/gamma/area-stratumsearch ifthisscreen fails;twoarms only.',
   'Passingpermitsconsideringindependentconfirmation andclosestmethodclearance,notnewtrainingoralgorithmqueue.'],
 source_commit=cfg35['source_commit'],automatic_followups=False,new_algorithm_approved=None,historical_caps_remain_closed=True)
save(D/'score_protocol_FROZEN.json',cfg)
remote_files=['collect_head_scores.py','score_protocol_FROZEN.json','pinned_o27_metrics.py']
save(D/'execution_inputs_manifest.json',dict(remote_files_sha256={n:sha(D/n) for n in remote_files},analysis_sha256=sha(D/'analyze_scores.py'),
    source_audit_sha256=sha(D/'source_operator_audit.json'),pinned_A30_sha256=sha(D/'pinned_A30_independent.py'),
    bootstrap_sha256=sha(D/'bootstrap_draws_FROZEN.npz'),coco_reference_sha256=sha(D/'coco_reference.py'),
    archive_verifier_sha256=sha(D/'verify_score_archive.py'),metric_verifier_sha256=sha(D/'verify_score_metrics.py')))
print(json.dumps(dict(status='A36_FROZEN_BEFORE_RAW_HEAD_OUTCOMES',head_call_cap=12,whole_model_cap=0,optimizer_cap=0,protocol_sha256=sha(D/'score_protocol_FROZEN.json'))))
