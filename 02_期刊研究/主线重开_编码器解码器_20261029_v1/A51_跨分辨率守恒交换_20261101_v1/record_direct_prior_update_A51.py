from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parent;LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
registry=load(ROOT/'05_核心文献矩阵与阅读范围_v1.json')
message=load(ROOT/'FMRNet_Crossref_response_v1.json')['message']
assert message['DOI'].lower()=='10.1109/tgrs.2026.3744667'
assert message['container-title']==['IEEE Transactions on Geoscience and Remote Sensing']
registry['supplement_recorded_utc']=datetime.now(timezone.utc).isoformat()
registry['previous_registry_preserved']=str(ROOT/'05_核心文献矩阵与阅读范围_v1.json')
registry['unverified_discovery_only']=[]
registry['entries'].append({'title':message['title'][0],'authors':[x['given']+' '+x['family'] for x in message['author']],
 'DOI':message['DOI'],'venue':message['container-title'][0],'year':message['issued']['date-parts'][0][0],
 'primary_registry_url':'https://api.crossref.org/works/10.1109/TGRS.2026.3744667',
 'publisher_url':message['resource']['primary']['URL'],'reading_scope':'verified primary registry metadata ONLY; no primary abstract/full method obtained',
 'pdf_obtained':False,'full_paper_read':False,'acronym_FMRNet_status':'from secondary index, not confirmed by primary metadata',
 'metadata_archive':str(ROOT/'FMRNet_Crossref_response_v1.json'),
 'metadata_sha256':hashlib.sha256((ROOT/'FMRNet_Crossref_response_v1.json').read_bytes()).hexdigest(),
 'full_text_access_receipt':str(ROOT/'FMRNet_publisher_access_v1.json'),
 'comparison_limit':'exact operation overlap unresolved; metadata does not confirm secondary method description'})
registry['theory_attribution_supplement']=str(ROOT/'13_理论归属与直接检测先例补充_v1.txt')
registry['novelty_risk']='Q is expressible as mass-preconditioned connection/sheaf diffusion with gate tying. Only task-specific cross-grid implementation/constraints and empirically demonstrated benefit may support an incremental paper; no new diffusion principle claim.'
for p in [ROOT/'14_核心文献矩阵与阅读范围_v2.json',LIB/'A51_核心方法阅读范围_20261101_v2.json']:
    assert not p.exists();p.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
current=load(ROOT/'CURRENT_A51.json');current['latest_research_registry']=str(ROOT/'14_核心文献矩阵与阅读范围_v2.json')
current['latest_theory_attribution']=registry['theory_attribution_supplement']
current['latest_observation']=str(ROOT/'observer_09.json')
(ROOT/'CURRENT_A51.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PRIMARY_METADATA_VERIFIED_FULL_METHOD_UNREAD','title':message['title'][0],'previous_frozen_reports_unchanged':True,'model_ops_added':0}))
