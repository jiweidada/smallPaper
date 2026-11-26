"""Archive old mutable state and seal this research snapshot without changing evidence."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PAIR=ROOT/'raw_EMA逐GT对照_20261026_v1';PARTNER=ROOT/'讨论伙伴_候选保留主线取证_20261027_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
 with open(p,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 assert read(HERE/'delivery_checks.json')['status']=='PASS'
 sources={};probes=[]
 for d in HERE.iterdir():
  p=d/'attempt01/input_manifest.json'
  if not p.exists():continue
  probes.append(d.name)
  for path,h in read(p).items():
   k=str(Path(path).resolve());assert k not in sources or sources[k]==h,(k,'source conflict');sources[k]=h
 support=HERE/'class_history_information/support_denominators_v1/input_manifest.json'
 for p,h in read(support).items():
  k=str(Path(p).resolve());assert k not in sources or sources[k]==h;sources[k]=h
 assert len(probes)==7
 for p,h in sources.items():assert sha(p)==h,(p,'source changed')
 extra={}
 external=['00_原文取证与已有方法边界.txt','02_H1证伪后_匹配见证层级与新增近邻.txt','03_当前主线与pre见证反证意见.txt','05_Hpre冻结前的逻辑复核与代码绑定.txt','06_Hpre关闭与类别身份假说_新增原文边界.txt','07_类别身份信息阳性及关键性反证_交换前.txt','08_简单跨层分类集成的反解释与原文范围.txt','09_本轮最新结论与下一条双向信息命题.txt','10_两对话本轮续议实际原文.txt','11_冻结计划与分母补全_续议追加记录.txt','sources_and_reading.json','sources_extension_after_H1.json','claim_to_source_v3.csv','discussion_record_final.json','frozen_plan_addendum.json','frozen_plan_message_addendum.json','final_source_audit_v2.json','delivery_manifest.json']
 for n in external:
  p=PARTNER/n;assert p.exists(),n;extra[str(p)]=sha(p)
 pa=read(PARTNER/'final_source_audit_v2.json');assert pa['status']=='PASS'
 checked=0
 for r in pa['unchanged_source_checks']:
  if 'path' in r and 'sha256' in r:assert sha(r['path'])==r['sha256'];checked+=1
 write(HERE/'source_integrity_final.json',{'status':'PASS','completed_probe_input_manifests':len(probes),'unique_inputs_rehashed':len(sources),'partner_source_entries_rehashed':checked,'model_forwards':0,'evaluator_calls':0,'training':0,'meaning':'Integrity only, no analysis or evaluation rerun.','unique_input_sha256':sources,'partner_evidence_sha256':extra})
 statepath=PAIR/'work_state_v2.json';old=read(statepath);snapshot=PAIR/'work_state_v2_20261027前快照.json';write(snapshot,old)
 state={'version':3,'client_date':'2026-10-27','latest_user_instruction':'继续和另一个对话找主线；先缩小可检验原因，查已有方法缺口，提出可证伪创新假说；不训练、不提前提模块或损失，一个失败继续下一条。','current_objective':'候选保留问题的双向风险：补入收益与反向覆盖损伤；下一条为Hlineage信息假说，而非已验证新方法。','constraints_still_active':['No training','No network module/loss proposals before mechanism gate','Use existing local data/CPU','No GPU/remote/test/test-dev','Preserve checkpoints/caches/source/failed protocols','Single assistant, no subagents'],'discussion_thread_id':'01a17b5a-0a45-7921-9591-3ac370e081b5','discussion_title':'两个检测问题的研究讨论','discussion_host':'local','last_wait_cursor':read(HERE/'两对话实际讨论_来源索引.json').get('latest_cursor'),'discussion_status':'Actual multi-round literature/mechanism discussion completed for this snapshot; partner checked final Hlineage plan SHA and denominators.','current_authority':str(HERE/'00_当前主线候选_问题卡与最小验证.txt'),'current_problem_card':str(HERE/'问题卡_01_候选保留.json'),'hypothesis_ledger':str(HERE/'假说状态账本.json'),'original_allGT_and_4x4_cohorts_preserved':True,'completed_current_counts':old.get('completed_current_counts',{}),'completed_current_stage':'AllGT map retained; seven frozen zero-training probes PASS, cap/prefix explanations and pre graph/rule branches closed; qualification information separated from criticality; bounded primary literature gap analysis; one problem card and next falsifiable plan.','root_execution_this_stage':{'CPU_evaluator_calls':20,'new_model_forwards':0,'head_calls':0,'training':0,'GPU':False,'remote':False,'test_reads':0},'next_hypothesis':{'id':'Hlineage','status':'FROZEN_PLAN_NOT_EXECUTED','path':str(HERE/'Hlineage_下一最小零训练检验_冻结计划.json'),'sha256':sha(HERE/'Hlineage_下一最小零训练检验_冻结计划.json'),'code_implemented':False,'question':'Does preserved class-conditioned supporting-query correspondence add BOTH insertion-gain and deletion-risk information beyond frozen fusion/marginal/geometric references?','do_not_advance_without':['Feasibility and all literal denominators','Qualified-domain marginal-capacity information on both sides, both seeds/states','Strong identity-permutation and geometric/score-fusion references','Separate genuine equal-budget native evaluation','Nearest-method novelty/fidelity checks']},'important_limits':['BASE24 only, two seeds with correlated raw/EMA states','GT matching bounds are not AP attainability','Qualification positive but criticality not supported by pre winner','One-swap AP/APs negative allfour; rule closed','Hlineage has no result and no established novelty','Other hypotheses remain open when one fails'],'previous_state_snapshot':str(snapshot)}
 statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 items=[]
 for p in sorted(HERE.rglob('*')):
  if p.is_file() and p.name not in ('delivery_manifest.json','manifest_verification.json'):
   items.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 manifest={'status':'SEALED_RESEARCH_SNAPSHOT','created_utc':datetime.now(timezone.utc).isoformat(),'client_date':'2026-10-27','current_report':str(HERE/'00_当前主线候选_问题卡与最小验证.txt'),'cards':1,'Hlineage_executed':False,'file_count':len(items),'files':items,'external_evidence_sha256':extra,'mutable_state_excluded':str(statepath),'mutable_state_history_snapshot_sha256':sha(snapshot),'no_method_or_novelty_success_claim':True}
 write(HERE/'delivery_manifest.json',manifest)
 assert all(sha(r['path'])==r['sha256'] for r in items)
 write(HERE/'manifest_verification.json',{'status':'PASS','file_count':len(items),'all_snapshot_files_rehashed':True,'unique_probe_inputs_rehashed':len(sources),'external_source_checks':checked,'manifest_sha256':sha(HERE/'delivery_manifest.json'),'Hlineage_not_run':True,'training':0})
 print('Final source and delivery integrity PASS',len(items),'files;',len(sources),'unique inputs;',checked,'partner source entries. No experiment rerun.')
if __name__=='__main__':main()
