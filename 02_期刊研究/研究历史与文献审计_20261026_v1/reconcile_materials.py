"""Reconcile PDF identities, legacy reading notes, and saved retrieval records."""
import hashlib,json,re,sys
from collections import defaultdict,Counter
from pathlib import Path
from pypdf import PdfReader
from prepare_audit import norm_url
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;EN=OUT.parent;ROOT=EN.parent;LIB=EN/'文献_相关原始论文_20261004'
d=json.loads((OUT/'literature_preparation.json').read_text(encoding='utf-8'))
inv=json.loads((OUT/'filesystem_inventory.json').read_text(encoding='utf-8'))
manifest={}
notes=[]
for p in sorted(LIB.glob('*')):
    if p.suffix.lower() not in ('.txt','.md') or p.name.startswith(('文献统一管理入口','150_','26_','27_','28_','30_')):continue
    blob=p.read_bytes();s=blob.decode('utf-8-sig',errors='replace')
    manifest[str(p)]=hashlib.sha256(blob).hexdigest()
    # Reading-log blocks with explicit numerical bibliography identifiers.
    starts=list(re.finditer(r'(?m)^\s*(?:#+\s*)?(\d{1,3})\s+([^\n]+)',s))
    for j,m in enumerate(starts):
        body=s[m.start():starts[j+1].start() if j+1<len(starts) else len(s)]
        if len(body)>4500:body=body[:4500]
        notes.append({'bibliography_number':int(m[1]),'source':str(p),
                      'line':s.count('\n',0,m.start())+1,'text':body})
extra=[]
hash_to_lib=defaultdict(list)
for i,r in enumerate(d['pdf_records'],1):hash_to_lib[r['sha256']].append(i)
for r in inv['files']:
    p=Path(r['path'])
    if r['ext']!='.pdf' or p.is_relative_to(LIB) or not p.is_relative_to(EN/'data/00_实验计划与证据索引'):continue
    b=p.read_bytes();h=hashlib.sha256(b).hexdigest();manifest[str(p)]=h
    reader=PdfReader(p,strict=False)
    extra.append({'path':str(p),'sha256':h,'same_as_library_records':hash_to_lib.get(h,[]),
                  'metadata_title':str((reader.metadata or {}).get('/Title','')),
                  'first_page_head':' '.join((reader.pages[0].extract_text() or '').split())[:600]})
retrieval=[]
# Select saved scientific-source documents, not code bodies, prediction arrays or runtime manuals.
for r in inv['files']:
    p=Path(r['path']);rel=r['relative']
    if r['ext'] not in ('.txt','.md','.json','.bib') or r['bytes']>2_000_000:continue
    if any(x in rel for x in ('\\runtime\\','\\_text\\','\\.claude\\','\\extracted\\','\\verified\\','\\collection\\','\\packet\\')):continue
    if not re.search(r'文献|查新|阅读|原文|近邻|literature|triage|novelty|sources|source_manifest|references|refs\.bib',p.name,re.I):continue
    try:blob=p.read_bytes();s=blob.decode('utf-8-sig',errors='replace')
    except OSError:continue
    if 'http' not in s:continue
    manifest[str(p)]=hashlib.sha256(blob).hexdigest()
    for m in re.finditer(r'https?://[^\s<>"\x27\\]+',s):
        url=m[0].rstrip('.,;，；。)）]')
        if not re.search(r'arxiv|doi\.org|openaccess\.thecvf|openreview|sciencedirect|mdpi\.com|nature\.com/articles|springer\.com|proceedings\.|papers\.ni?ps|pmlr\.press|scitepress|pinnaclepubs|ieeexplore',url,re.I):continue
        retrieval.append({'key':norm_url(url),'url':url,'source':str(p),'line':s.count('\n',0,m.start())+1,
                          'context':s[max(0,m.start()-260):min(len(s),m.end()+300)]})
source_only=[]
for r in d['index_rows']:
    if not any(x['filename'] in r.get('完整文件','') for x in d['pdf_records']):source_only.append(r)
out={'outside_library_scientific_pdfs':extra,'legacy_reading_note_blocks':notes,
     'index_records_without_pdf':source_only,'saved_retrieval_mentions':retrieval,
     'duplicate_sha_groups':[v for v in hash_to_lib.values() if len(v)>1],
     'document_sha256':manifest}
with (OUT/'material_reconciliation.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'outside_library_scientific_pdfs':extra,'duplicate_sha_groups':out['duplicate_sha_groups'],
                  'legacy_note_blocks':len(notes),'old_index_records_without_pdf':len(source_only),
                  'saved_scientific_url_mentions':len(retrieval),'normalized_scientific_urls':len({x['key'] for x in retrieval}),
                  'source_documents':len(manifest)},ensure_ascii=False,indent=2))
