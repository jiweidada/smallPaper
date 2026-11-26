"""Predeclare a single remaining-cohort replication; keep A33 unchanged and closed."""
from pathlib import Path
import ast,copy,datetime,difflib,hashlib,json,runpy,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;OLD=B/'A33_固定可见内容的采样相位诊断_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
prior=read(B/'77_A33原生相位实验与独立核验回执.json')
for p,h in prior['files_sha256'].items():assert sha(p)==h,p
assert prior['A33_cap_closed'] and prior['new_whole_detector_forwards']==169
replacements={
 'native_phase_worker.py':[("assert COUNT['whole_detector_forwards']==169 and COUNT['extra_native_O_decodes']==168 and COUNT['all_native_head_decodes']==337","assert COUNT['whole_detector_forwards']==533 and COUNT['extra_native_O_decodes']==532 and COUNT['all_native_head_decodes']==1065")],
 'analyze_native_phase.py':[("assert complete['counts']['whole_detector_forwards']==169","assert complete['counts']['whole_detector_forwards']==533")],
 'verify_archive.py':[("len(proof['proof'])==168","len(proof['proof'])==532"),("proof['zero_fidelity_cases']==24","proof['zero_fidelity_cases']==76"),("zero_native_image_branch_records_compared=48","zero_native_image_branch_records_compared=152"),("pixel_preservation_conditions=168","pixel_preservation_conditions=532")],
 'launch_native_once.py':[], 'verify_primary_scalar.py':[]}
reuse=[]
for name,changes in replacements.items():
 original=(OLD/name).read_text(encoding='utf-8');new=original
 for a,b in changes:assert new.count(a)==1,(name,a);new=new.replace(a,b,1)
 ast.parse(new)
 with (D/name).open('x',encoding='utf-8') as f:f.write(new)
 reuse.append(dict(file=name,from_file=str(OLD/name),from_sha256=sha(OLD/name),to_sha256=sha(D/name),changes=changes,diff=list(difflib.unified_diff(original.splitlines(),new.splitlines(),lineterm=''))))
save(D/'reused_code_change_audit.json',dict(scope='Only nativecount/fidelity-casecount assertions changed;no modeloperator,metricformula,gates,preprocessing,postprocessing changes.',files=reuse))
c33=read(OLD/'phase_protocol_FROZEN.json');c30=read(B/'A30_YOLO26双头原生读出核查_v1/native_dual_head_protocol_FROZEN.json')
all_cases=sorted([c for c in c30['cases'] if min(c['geometry']['pad_ltrb'][1],c['geometry']['pad_ltrb'][3])>=32],key=lambda c:hashlib.sha256(('A33fixedpixels-v1|'+c['file_name']).encode()).hexdigest())
assert [c['image_id'] for c in all_cases[:24]]==[c['image_id'] for c in c33['cases']]
cases=all_cases[24:];assert len(cases)==76
for c in cases:
 local=Path(r'D:\数据集\visdrone2019\VisDrone2019-DET-train\images')/c['file_name'];assert sha(local)==c['sha256']
 c['original_A30_index']=c['index'];ref=B/'A30_YOLO26双头原生读出核查_v1/downloaded_evidence/captures'/('case_%03d.npz'%c['index'])
 c['reference_capture']=c30['remote_output']+'/captures/'+ref.name;c['reference_capture_sha256']=sha(ref)
analysis=runpy.run_path(str(D/'analyze_native_phase.py'));save(D/'metric_fixture_receipt.json',analysis['fixture']())
ann=[g for g in read(c33['local_inputs']['GT']['path'])['annotations'] if g['image_id'] in {c['image_id'] for c in cases}]
e0=analysis['exposure_flags'](ann,cases,0);e8=analysis['exposure_flags'](ann,cases,8);where={g['id']:g['image_id'] for g in ann}
sep=[g for g in e0 if e0[g] and not e8[g]];merge=[g for g in e0 if not e0[g] and e8[g]]
support=dict(small_GT=len(e0),separate_GT=len(sep),merge_GT=len(merge),separate_images=len({where[g] for g in sep}),merge_images=len({where[g] for g in merge}))
assert min(len(sep),len(merge))>=50 and min(support['separate_images'],support['merge_images'])>=5
cfg=copy.deepcopy(c33)
cfg.update(phase='A34',status='FROZEN_REPLICATION_BEFORE_REMAINING76_PHASE_OUTCOMES',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_goal_turn='progress:A33fixedpixel169forwardsscreen;generalsensitivityobserved,butprimarydirectionalCIcrosseszero;newalgorithmnotapproved.',remote_output=c33['remote_root']+'/architecture_probe_v1/a34_remaining76_phase_20261030_v1',cases=cases,case_selection='Exactlyranks25-100of the original A33SHAorder;the24screen casesareexcluded. Selectionuses no A34phaseoutcomes.',prelaunch_support=support,caps=dict(whole_detector_forwards=533,extra_native_O_decodes=532,all_native_head_decodes=1065,optimizer_updates=0,standalone_TAL_calls=0,new_training_jobs=0,wall_seconds=1800),execution='76imagesx7same shifts plusonefirstzero repeat;new533cap. IdenticalFP32nativepipeline;A33cap169 remainsclosed. Noautomaticretry/followup.',expansion_rationale='A33primaryOcontrast+0.07827but95CI[-0.03458,0.16250]. Resolveprecisiononceonremaining76;not a changedgate,shiftsearch or algorithmtrial.',cohort_scope='Theseimageswerepartofearlierdevelopmentdata,but havenotreceivedA33phaseinterventions. Thisisremaining-cohortreplication,not an independentblinddataset.',replication_decision='Same5gates on76only. ReportA33separately;no poolingorbestphasecanoverridefailureofthisreplication. Ifnotpassed,do notextendthisphase-instancebranchfurtherwithoutnewmechanisticevidence.',A33_169_cap_remains_closed=True,architecture_forwards_before=2635,prior_A33_files_verified=len(prior['files_sha256']))
cfg['local_inputs']['analysis']=dict(path=str(D/'analyze_native_phase.py'),sha256=sha(D/'analyze_native_phase.py'))
cfg['metrics']['FP_budget']=703;cfg['metrics']['FP_budget_rule']='floor(9.26 x76)=703;sameFP/imagebudget asA33tointegerprecision;reestimateeachconditionthresholdoneverypairedbootstrapdraw.'
cfg['metrics']['bootstrap_seed']=434103
cfg['metrics']['bootstrap']='Resample76imagesjointly1000times;reestimateeachconditionglobalFPthresholdoneverydraw;no pooled100imageprimaryanalysis.'
save(D/'phase_protocol_FROZEN.json',cfg)
save(D/'execution_inputs_manifest.json',dict(worker_sha256=sha(D/'native_phase_worker.py'),protocol_sha256=sha(D/'phase_protocol_FROZEN.json'),launcher_sha256=sha(D/'launch_native_once.py'),analysis_sha256=sha(D/'analyze_native_phase.py')))
print(json.dumps(dict(status=cfg['status'],cases=76,support=support,cap=533,prior_A33_verified=len(prior['files_sha256']),no_A34_model_calls_yet=True),ensure_ascii=False))
