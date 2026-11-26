"""Local document inventory and bibliographic preparation. No model/data experiment."""
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
logging.getLogger('pypdf').setLevel(logging.ERROR)
OUT = Path(__file__).resolve().parent
EN = OUT.parent
ROOT = EN.parent
LIB = EN/'文献_相关原始论文_20261004'
SKIP = {'node_modules','__pycache__','.git','.agents','.codex','.aws','site-packages',
        'python_deps','isolated_runtime','build_tools','analysis_tools','.latex_build_check',
        '.latex_build_check3','.latex_build_check4','_claude_runner'}
TEXT_EXT = {'.txt','.md','.json','.csv','.yaml','.yml','.log','.tex','.bib'}
READS = {}


def read(path):
    b = Path(path).read_bytes()
    READS[str(path)] = {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
    return b.decode('utf-8-sig',errors='replace')


def write(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)


def norm_url(url):
    s = url.strip('.,;\n\r\t <>"\'')
    a = re.search(r'(\d{4}\.\d{4,5})(?:v\d+)?',s)
    if 'arxiv' in s.lower() and a:
        return 'arxiv:'+a[1]
    a = re.search(r'10\.\d{4,9}/[^\s<>"\']+',s,re.I)
    if a:
        return 'doi:'+a[0].rstrip('.,;)').lower()
    if 'openreview.net/' in s and 'id=' in s:
        return 'openreview:'+s.split('id=',1)[1].split('&',1)[0]
    return s.split('#',1)[0].rstrip('/')


def walk():
    files = [];errors = [];pruned=[]
    for folder,dirs,names in os.walk(ROOT,onerror=lambda e:errors.append(str(e)),followlinks=False):
        p = Path(folder)
        keep=[]
        for d in dirs:
            q=p/d
            if d in SKIP or q == OUT or q.is_symlink():
                pruned.append(str(q))
            else:
                keep.append(d)
        dirs[:] = keep
        for n in names:
            q=p/n
            if q.suffix.lower() not in TEXT_EXT|{'.pdf','.docx','.py','.mjs','.html'}:
                continue
            try:
                st=q.stat()
                files.append({'path':str(q),'relative':str(q.relative_to(ROOT)),
                              'bytes':st.st_size,'mtime':st.st_mtime,'ext':q.suffix.lower()})
            except OSError as e:
                errors.append(str(e))
    write('filesystem_inventory.json',{'files':files,'errors':errors,'pruned':pruned})
    return files,errors,pruned


def main():
    tick=time.monotonic()
    files,errors,pruned=walk()
    indexes=sorted(LIB.glob('文献统一总索引_*.csv'))
    index_rows=list(csv.DictReader(read(indexes[-1]).splitlines()))
    byfile={}
    for r in index_rows:
        for name in re.split('[;；]',r.get('完整文件','')):
            byfile[Path(name.strip()).name]=r
    pdfs=[x for x in files if x['ext']=='.pdf' and Path(x['path']).is_relative_to(LIB)]
    prep=[]
    (OUT/'pdf_text').mkdir(exist_ok=False)
    for j,item in enumerate(pdfs):
        path=Path(item['path']);blob=path.read_bytes();h=hashlib.sha256(blob).hexdigest()
        READS[str(path)]={'sha256':h,'bytes':len(blob)}
        record={'path':str(path),'filename':path.name,'sha256':h,'bytes':len(blob),
                'index':byfile.get(path.name),'metadata_title':None}
        try:
            reader=PdfReader(path,strict=False)
            record['pages']=len(reader.pages)
            record['metadata_title']=str((reader.metadata or {}).get('/Title',''))
            texts=[reader.pages[i].extract_text() or '' for i in range(min(2,len(reader.pages)))]
            record['first_two_pages_text']='\n\n---PAGE---\n\n'.join(texts)
            record['first_page_head']=' '.join(texts[0].split())[:1300] if texts else ''
            record['arxiv_ids_firstpages']=sorted(set(re.findall(r'\b\d{4}\.\d{4,5}(?:v\d+)?\b',' '.join(texts))))
            record['doi_firstpages']=sorted(set(re.findall(r'10\.\d{4,9}/[^\s<>"\']+',' '.join(texts),re.I)))
            record['text_file']=str(OUT/'pdf_text'/f'{j+1:03d}_{h[:12]}.txt')
            Path(record['text_file']).write_text(record['first_two_pages_text'],encoding='utf-8')
        except Exception as e:
            record['error']=f'{type(e).__name__}: {e}'
        prep.append(record)
        if (j+1)%50==0:
            print(json.dumps({'pdf_prepared':j+1},ensure_ascii=True),flush=True)
    sources=[];urls=defaultdict(list)
    for path in sorted(LIB.rglob('*.json')):
        if path.stat().st_size>6_000_000 or not path.name.startswith(('_sources','_source','_download','_reading')):
            continue
        s=read(path)
        try:
            obj=json.loads(s)
        except Exception:
            continue
        sources.append({'path':str(path),'root_type':type(obj).__name__})
        def visit(v,loc='$'):
            if isinstance(v,dict):
                for k,x in v.items():
                    if isinstance(x,str) and re.search('url|link|source',k,re.I) and re.match('https?://',x):
                        title=next((v.get(t) for t in ('title','name','paper_title','query','filename','file') if isinstance(v.get(t),str)),None)
                        urls[norm_url(x)].append({'url':x,'source_file':str(path),'locator':loc+'.'+k,'title':title})
                    elif isinstance(x,(dict,list)):
                        visit(x,loc+'.'+k)
            elif isinstance(v,list):
                for k,x in enumerate(v):visit(x,loc+f'[{k}]')
        visit(obj)
    write('literature_preparation.json',{'latest_index':str(indexes[-1]),'index_rows':index_rows,
          'pdf_records':prep,'source_manifests':sources,'url_records':dict(urls)})
    write('input_manifest_preparation.json',READS)
    summary={'files_by_ext':dict(Counter(x['ext'] for x in files)),'walk_errors':len(errors),'pruned_directories':len(pruned),
             'library_pdf_files':len(pdfs),'library_unique_pdf_sha256':len({r['sha256'] for r in prep}),
             'library_root_pdfs':sum(Path(r['path']).parent==LIB for r in prep),
             'library_pdf_parse_errors':sum('error' in r for r in prep),'latest_index_rows':len(index_rows),
             'pdfs_without_latest_index':len([r for r in prep if not r['index']]),
             'source_manifest_files':len(sources),'normalized_url_records':len(urls),'wall_seconds':time.monotonic()-tick}
    write('preparation_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
