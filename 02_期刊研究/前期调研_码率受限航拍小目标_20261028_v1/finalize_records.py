"""Consolidate document provenance and verify local report bookkeeping only."""
from pathlib import Path
import json, hashlib, os, subprocess, sys, re
from source_fetch import ROOT, ARCHIVE, fetch

records=json.loads((ROOT/'literature_records.json').read_text(encoding='utf-8'))
logpath=ROOT/'02_检索与证据记录.json'
log=json.loads(logpath.read_text(encoding='utf-8'))
for row in json.loads((ROOT/'search_supplement.json').read_text(encoding='utf-8')):
 if row['id'] not in {b['id'] for b in log['batches']}: log['batches'].append(row)
log['counts']={'all_query_attempts':sum(len(b['queries']) for b in log['batches']),
 'completed_targeted_batches':sum(b['status'].startswith('completed') and not b['id'].startswith('S00') for b in log['batches']),
 'completed_targeted_queries':sum(len(b['queries']) for b in log['batches'] if not b['id'].startswith('S00') and b['status'].startswith('completed')),
 'warning':'Queries are attempts, not unique papers or complete database hit counts.'}
logpath.write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')

inventory=json.loads((ROOT/'local_pdf_inventory.json').read_text(encoding='utf-8'))
preflight=Path(r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py')
for entry in inventory:
 path=Path(entry['file']); sidecar=Path(entry['preflight_sidecar'])
 result=json.loads(sidecar.read_text(encoding='utf-8'))
 if result.get('file')!=str(path):
  old=sidecar.with_name(sidecar.stem+'_initial_stdout_encoding.json')
  if not old.exists(): old.write_bytes(sidecar.read_bytes())
  p=subprocess.run([sys.executable,str(preflight),str(path)],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
  result=json.loads(p.stdout.decode('utf-8'))
  assert result['sha256']==entry['sha256']
  sidecar.write_bytes(p.stdout)
 entry['structural_verdict']=result.get('verdict','UNAVAILABLE')
 entry['structure_scope']='STRUCTURE_ONLY; not claim validation or full-reading attestation'
 for record in records:
  if record['pdf_pattern']==path.name: entry['ref_id']=record['id']; entry['read_scope']=record['reading']
(ROOT/'local_pdf_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')

last_receipts=json.loads((ROOT/'retrieval_last.json').read_text(encoding='utf-8')) if (ROOT/'retrieval_last.json').exists() else []
for key,url in {
 'Masked_feature_publisher_page':'https://www.mdpi.com/2227-7390/12/12/1848',
 'Tu_knowledge_transfer_publisher_page':'https://dl.acm.org/doi/10.1145/3678471',
}.items():
 if key in {r['key'] for r in last_receipts}: continue
 item=fetch(key,url); last_receipts.append(item)
 print(json.dumps({'source':key,'status':item['status'],'error':item.get('error')},ensure_ascii=True),flush=True)
(ROOT/'retrieval_last.json').write_text(json.dumps(last_receipts,ensure_ascii=False,indent=2),encoding='utf-8')

attempts=[]
for filename in ['retrieval_receipts.json','retrieval_extra.json','retrieval_tail.json','retrieval_last.json','retrieval_mainline.json','retrieval_yolo26.json']:
 attempts+=json.loads((ROOT/filename).read_text(encoding='utf-8'))
byname={Path(x['file']).name:x for x in inventory}
for record in records:
 record['human_read_attestation']=False
 record['source_check']='Primary title/metadata or stated primary excerpts verified by browsing; not a retraction or independent replication audit.'
 record['local_pdf']=byname.get(record['pdf_pattern']) if record['pdf_pattern'] else None
 record['archive_status']='pdf_archived' if record['local_pdf'] else 'full_pdf_not_obtained; reading is limited to the recorded online scope'
summary={'date':'2026-10-28','timezone':'America/Los_Angeles','records':records,'retrieval_attempts':attempts,
 'pdf_inventory':inventory,'counts':{'core_studies':8,'total_references':len(records),'local_pdfs':len(inventory),'new_local_pdfs':sum('20261028' in Path(i['file']).name for i in inventory)},
 'execution':{'model_construction':0,'model_weight_loads':0,'model_forwards':0,'training_runs':0,'GPU_jobs':0,'remote_jobs':0,'evaluation_cache_rebuilds':0,'test_or_test_dev_evaluations':0,'subagents':0},
 'notes':['Text extraction is not full reading.','PDF structural PASS is not scientific validation.','The Lost in Compression PDF was pre-existing and was reused, not a new download.','An initial Windows stdout encoding artifact in four sidecars was preserved, then corrected by re-running the same structural tool in UTF-8 mode.']}
out=ROOT/'03_来源核验与原件存档.json'; out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
central=ARCHIVE/'_sources_码率调研_20261028_v1.json'
if not central.exists(): central.write_bytes(out.read_bytes())
central_update=ARCHIVE/'_sources_码率调研_主线追加_20261028_v1.json'
if not central_update.exists(): central_update.write_bytes(out.read_bytes())
central_model=ARCHIVE/'_sources_码率调研_YOLO26基线更正_20261028_v1.json'
if not central_model.exists(): central_model.write_bytes(out.read_bytes())

report=(ROOT/'00_调研报告.txt').read_text(encoding='utf-8')
ids=set(re.findall(r'\[R(\d+)\]',report))
assert ids=={str(i) for i in range(1,len(records)+1)},ids
assert len(log['batches'])==24 and log['counts']['all_query_attempts']==83
assert all(i['structural_verdict']=='PASS' for i in inventory)
assert all(hashlib.sha256(Path(i['file']).read_bytes()).hexdigest()==i['sha256'] for i in inventory)
audit={'date':'2026-10-28','report_characters':len(report),'bibliography_ids':sorted(ids,key=int),
 'query_counts':log['counts'],'pdfs':summary['counts'],'scope':'Bookkeeping and source artifact checks; no research experiments.'}
(ROOT/'verification_receipt.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=True),flush=True)
