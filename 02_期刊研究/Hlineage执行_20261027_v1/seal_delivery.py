"""Seal actual executed evidence; do not alter historical experiment outputs."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PARTNER=ROOT/'讨论伙伴_候选保留主线取证_20261027_v1';STATE=ROOT/'raw_EMA逐GT对照_20261026_v1/work_state_v2.json'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
 with open(p,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 r=read(HERE/'attempt01/results.json');v=read(HERE/'attempt01/verification.json');a=read(HERE/'audit_v1/results.json');q=read(HERE/'argmax_parity_v1/results.json')
 assert r['status']=='COMPLETE' and v['status']==a['status']==q['status']=='PASS' and not r['strict_information_advance_gate_pass']
 h=sha(HERE/'protocol.json');assert r['protocol_sha256']==h==(HERE/'protocol.sha256').read_text('utf-8').strip()
 inputs={}
 for source in (read(HERE/'attempt01/input_manifest.json'),a['input_sha256'],q['input_sha256']):
  for p,h0 in source.items():assert p not in inputs or inputs[p]==h0;inputs[p]=h0
 for p,h0 in inputs.items():assert sha(p)==h0,(p,'input changed')
 peers={}
 for p in sorted(PARTNER.glob('*')):
  if p.is_file() and (p.name.startswith('12_') or p.name.startswith('13_') or 'Hlineage' in p.name or 'hlineage' in p.name or 'actual_execution' in p.name):peers[str(p)]=sha(p)
 assert any(Path(p).name.startswith('13_') for p in peers),'Partner actual-result conclusion not saved yet'
 state=read(STATE);state['discussion_status']='Both chats read actual all8-domain negative results and both audits; scientific limitations and no-supported-mainline conclusion agreed. No experiment pending.';state['last_wait_cursor']='574151d0-7511-4597-9f3b-5eab2bbcf277:32';STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 write(HERE/'source_integrity_final.json',{'status':'PASS','unique_inputs_rehashed':len(inputs),'all_completed_checks_PASS':True,'no_domain_advances':True,'argmax_queries_checked':q['queries_checked_total'],'argmax_mismatches':0,'partner_public_evidence_sha256':peers,'operations':{'model_forwards':0,'evaluator_calls':0,'training':0}})
 files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ('delivery_manifest.json','manifest_verification.json')]
 manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'client_date':'2026-10-27','status':'SEALED_ACTUAL_NEGATIVE_RESULT','current_report':str(HERE/'00_当前结论_未找到可保留主线.txt'),'file_count':len(files),'files':files,'source_input_sha256':inputs,'partner_public_evidence_sha256':peers,'current_state_path_excluded':str(STATE),'innovation_line_established':False,'mainline_result':'None found this round; no universal impossibility claim','training':0}
 write(HERE/'delivery_manifest.json',manifest);assert all(sha(e['path'])==e['sha256'] for e in files)
 write(HERE/'manifest_verification.json',{'status':'PASS','snapshot_files_rehashed':len(files),'unique_inputs_rehashed':len(inputs),'manifest_sha256':sha(HERE/'delivery_manifest.json'),'protocol_sha256':r['protocol_sha256'],'all8_domains_not_advanced':True,'argmax_parity_pass':True,'integer_entropy_audit_pass':True,'new_training':0})
 print('Actual negative-result delivery sealed PASS;',len(files),'files;',len(inputs),'input hashes. No experiment rerun.')
if __name__=='__main__':main()
