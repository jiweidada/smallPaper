import json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
mode=sys.argv[1]
if mode=='papers':
    d=json.loads((OUT/'literature_preparation.json').read_text(encoding='utf-8'))
    lo=int(sys.argv[2]);hi=int(sys.argv[3])
    for i,r in enumerate(d['pdf_records'],1):
        if not lo<=i<=hi:continue
        text=r.get('first_two_pages_text','')
        m=re.search(r'(?:Abstract|ABSTRACT|A B S T R A C T|AB S T R A C T)\s*[—–:.-]?\s*',text)
        ab=' '.join(text[m.end():m.end()+1400].split())[:550] if m else ''
        print(json.dumps({'n':i,'file':r['filename'],
                          'title_head':r['metadata_title'] or r.get('first_page_head','')[:270],
                          'abstract':ab[:320],
                          'index_read':r['index'].get('阅读范围_非用户已读声明','')[:160] if r['index'] else None},ensure_ascii=False))
elif mode=='scope':
    d=json.loads((OUT/'filesystem_inventory.json').read_text(encoding='utf-8'))
    paths=[r for r in d['files'] if r['ext']=='.pdf' and '文献_相关原始论文_20261004' not in r['path']]
    print(json.dumps(Counter(str(Path(r['relative']).parent) for r in paths),ensure_ascii=False,indent=2))
    print(json.dumps({'errors':d['errors'][:12]},ensure_ascii=False,indent=2))
elif mode=='find':
    d=json.loads((OUT/'filesystem_inventory.json').read_text(encoding='utf-8'))
    pat=sys.argv[2]
    rows=[r for r in d['files'] if re.search(pat,r['relative'],re.I)]
    for r in rows[:int(sys.argv[3]) if len(sys.argv)>3 else 80]:
        print(f"{r['bytes']}\t{r['relative']}")
elif mode=='sources':
    d=json.loads((OUT/'literature_preparation.json').read_text(encoding='utf-8'))
    for k,v in d['url_records'].items():
        print(json.dumps({'key':k,'title':v[0]['title'],'file':Path(v[0]['source_file']).name,'loc':v[0]['locator']},ensure_ascii=False))
elif mode=='index_missing':
    d=json.loads((OUT/'literature_preparation.json').read_text(encoding='utf-8'))
    for r in d['index_rows']:
        fs=[x['filename'] for x in d['pdf_records'] if x['filename'] in r.get('完整文件','')]
        if not fs:
            print(json.dumps(r,ensure_ascii=False))
    print('substring_matches',sum(any(x['filename'] in r.get('完整文件','') for r in d['index_rows']) for x in d['pdf_records']))
elif mode=='unmatched_arxiv':
    from prepare_audit import norm_url
    d=json.loads((OUT/'literature_preparation.json').read_text(encoding='utf-8'))
    rec=json.loads((OUT/'material_reconciliation.json').read_text(encoding='utf-8'))
    known=set()
    for r in d['pdf_records']:
        for a in re.findall(r'\d{4}\.\d{4,5}',r['filename']+' '+r.get('first_two_pages_text','')):known.add('arxiv:'+a)
        if r['index']:
            for u in re.findall('https?://[^ ;；]+',r['index'].get('来源URL','')):known.add(norm_url(u))
    unmatched={}
    for r in rec['saved_retrieval_mentions']:
        if r['key'].startswith('arxiv:') and r['key'] not in known:unmatched.setdefault(r['key'],r)
    for k,r in unmatched.items():
        print(json.dumps({'key':k,'source':str(Path(r['source']).relative_to(OUT.parent.parent)),'line':r['line'],'context':r['context'][:400]},ensure_ascii=False))
    print('UNMATCHED_ARXIV',len(unmatched))
