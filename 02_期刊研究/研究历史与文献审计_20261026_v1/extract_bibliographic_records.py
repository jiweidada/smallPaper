import hashlib,json,re,sys
from pathlib import Path
from collections import defaultdict
from prepare_audit import norm_url
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'material_reconciliation.json').read_text(encoding='utf-8'))
prep=json.loads((OUT/'literature_preparation.json').read_text(encoding='utf-8'))
files=list(d['document_sha256'])
events=[]
for path in files:
    p=Path(path)
    if p.suffix.lower()!='.json':continue
    try:obj=json.loads(p.read_text(encoding='utf-8-sig'))
    except (ValueError,OSError):continue
    def visit(v,loc='$'):
        if isinstance(v,dict):
            title=next((v.get(k) for k in ('title','paper_title','论文名称','name') if isinstance(v.get(k),str) and len(v[k])>3),None)
            ur=[]
            for k,x in v.items():
                if isinstance(x,str) and re.search('url|page|doi|link|arxiv',k,re.I):
                    ur.extend(re.findall(r'https?://[^\s<>"\x27]+',x))
            ur=[u.rstrip('.,;）)') for u in ur if re.search('arxiv|doi.org|cvf|openreview|mdpi|nature.com/articles|sciencedirect|springer|pmlr|nips|neurips|ieeexplore|scitepress|pinnaclepubs',u,re.I)]
            if title and ur:
                local=[]
                for k,x in v.items():
                    if isinstance(x,str) and re.search('file|path|local|pdf',k,re.I):
                        local.extend(r['filename'] for r in prep['pdf_records'] if r['filename'] in x)
                scopes={k:str(x)[:2000] for k,x in v.items() if re.search('scope|read|verified|verification|limit|finding|implication|note|status',k,re.I) and not isinstance(x,(dict,list))}
                events.append({'title':title,'urls':ur,'url_keys':[norm_url(u) for u in ur],
                               'local_pdf_files':sorted(set(local)),'source':str(p),'locator':loc,'reading_fields':scopes})
            for k,x in v.items():
                if isinstance(x,(dict,list)):visit(x,loc+'.'+k)
        elif isinstance(v,list):
            for i,x in enumerate(v):visit(x,loc+f'[{i}]')
    visit(obj)
with (OUT/'bibliographic_record_events.json').open('x',encoding='utf-8') as f:json.dump(events,f,ensure_ascii=False,indent=2)
named=defaultdict(list)
for e in events:
    key=re.sub('[^a-z0-9\u4e00-\u9fff]','',e['title'].lower())
    named[key].append(e)
unbound=[]
for k,es in named.items():
    if not any(e['local_pdf_files'] for e in es):
        unbound.append({'title':es[0]['title'],'keys':sorted({k for e in es for k in e['url_keys']}),
                        'source':es[0]['source'],'events':len(es)})
with (OUT/'unbound_named_record_candidates.json').open('x',encoding='utf-8') as f:json.dump(unbound,f,ensure_ascii=False,indent=2)
print(json.dumps({'events':len(events),'normalized_names':len(named),'unbound_name_candidates':len(unbound)}))
for i,e in enumerate(unbound,1):
    print(json.dumps({'n':i,'title':e['title'],'keys':e['keys'][:2]},ensure_ascii=False))
