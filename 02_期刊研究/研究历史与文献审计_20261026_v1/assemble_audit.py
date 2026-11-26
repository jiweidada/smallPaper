"""Assemble auditable tables from reviewed local records, without new scientific experiments."""
import hashlib,json,re,sys,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import direction_decisions as D
import literature_decisions as L
import literature_project_uses as U
from prepare_audit import norm_url
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent;LIB=OUT.parent/'文献_相关原始论文_20261004'
P=json.loads((OUT/'literature_preparation.json').read_text(encoding='utf-8'))
R=json.loads((OUT/'material_reconciliation.json').read_text(encoding='utf-8'))
EVENTS=json.loads((OUT/'bibliographic_record_events.json').read_text(encoding='utf-8'))
INPUTS={}

def read(p):
    p=Path(p);b=p.read_bytes();INPUTS[str(p)]=hashlib.sha256(b).hexdigest()
    return b.decode('utf-8-sig',errors='replace')

def title_key(s):
    return re.sub('[^a-z0-9\u4e00-\u9fff]','',unicodedata.normalize('NFKD',s).lower())

def ukey(u):
    u=u.strip().rstrip('.,;；）)')
    if u.startswith(('arxiv:','doi:','openreview:')):
        return re.sub(r'(?:\.pdf|/pdf)$','',u.lower()) if u.startswith('doi:') else u
    k=norm_url(u)
    if k.startswith(('arxiv:','doi:','openreview:')):return k
    if 'mdpi.com/' in u:
        m=re.search(r'mdpi\.com/([^/]+)/(\d+)/(\d+)/(\d+)',u)
        if m:
            journals={'1424-8220':'s','2072-4292':'rs','2076-3417':'app','2079-9292':'electronics',
                      '2313-433X':'jimaging','2078-2489':'info','1999-4893':'a','2227-7390':'math','2073-8994':'sym'}
            if m[1] in journals:
                return f'doi:10.3390/{journals[m[1]]}{int(m[2])}{int(m[3]):02d}{int(m[4]):04d}'
    if 'sciencedirect.com/' in u:
        m=re.search(r'/pii/([A-Za-z0-9]+)',u)
        if m:return 'pii:'+m[1].lower()
    if 'ieeexplore.ieee.org/' in u:
        m=re.search(r'(?:document/|arnumber=)(\d+)',u)
        if m:return 'ieee:'+m[1]
    if 'openaccess.thecvf.com/' in u:
        stem=u.rsplit('/',1)[-1].split('?')[0]
        stem=re.sub(r'_(?:paper|supplemental|supplement)\.(?:pdf|html)$','',stem,flags=re.I)
        return 'cvf:'+stem.lower()
    return re.sub(r'/(?:pdf|html|pdfft)$','',k).rstrip('.pdf') if 'nature.com/articles/' in k else re.sub(r'/(?:pdf|html|pdfft)$','',k)


def source_ref(rel,anchor=''):
    p=ROOT/rel
    assert p.is_file(),f'Missing evidence source: {p}'
    s=read(p)
    if anchor:
        assert anchor in s,(p,anchor)
        n=s.count('\n',0,s.index(anchor))+1
    else:n=1
    return {'path':str(p),'line':n,'sha256':INPUTS[str(p)]}


def make_directions():
    out=[]
    for r in D.rows:
        row=dict(r)
        row['evidence']=source_ref(r['source'],r['anchor'])
        row['secondary_evidence']=[source_ref(x) for x in r['secondary_sources']]
        row['evidence_level']='终点/专题报告核查，未重跑实验'
        if r['source'] in (D.HIST,D.REV,D.OLD):
            row['evidence_level']='历史复盘与其来源索引；未逐项重新复算原始预测'
        if r['source'].endswith('.json'):
            row['evidence_level']='已保存结构化结果/执行记录，未重跑实验'
        row['count_note']='种子、对照臂、代码修订不另计方向；本条主状态唯一'
        out.append(row)
    return out


class DSU:
    def __init__(self,n):self.p=list(range(n))
    def find(self,a):
        if self.p[a]!=a:self.p[a]=self.find(self.p[a])
        return self.p[a]
    def union(self,a,b):self.p[self.find(b)]=self.find(a)


def classify_record(title):
    s=title.lower()
    if re.search('delay|video|track-before|track before|seq-nms|vid:',s):return 'T10'
    if re.search('uncertain|noise|annotation|label error|bounding box|localiz|imprecise|hausdorff',s):return 'T3'
    if re.search('frequency|wavelet|detail-preserv|ese-net|cfba|sfcnet',s):return 'T1'
    if re.search('mosaic|augment|scale match',s):return 'T6'
    if re.search('query|matching|assignment|deim|detector failures',s):return 'T4'
    if re.search('resolution|zoom|crop|fusion|glsdet|glnet|sfenet|wacfn|arf-yolo',s):return 'T2'
    if re.search('rank|calibr|classif|semantic|confidence|sad et|sadet',s):return 'T5'
    if re.search('efficient|lightweight|real-time|detr|yolo',s):return 'T7'
    return '待核'


def reading_grade(text):
    if not text:return '历史阅读证据未知'
    # A local method/formula locator supports partial reading, never full-read claims.
    if re.search(r'§\s*\d|[Ss]ections?\s*\d|第\s*\d+\s*[页节]|(?:Eq|equation|式)\.?\s*\(?\d|源码.*(?:核对|静态|阅读)|方法.*(?:核读|局部|复读|补读)',text):
        return '有局部方法/公式阅读记录'
    if re.search('摘要|abstract|引言|introduction|Highlights',text,re.I):return '有摘要/引言阅读记录'
    return '历史阅读证据未知'


def make_literature(directions):
    records=[]
    for n,r in enumerate(P['pdf_records'],1):
        title=L.TITLE_OVERRIDES.get(n) or r['metadata_title'] or r.get('first_page_head','')[:220]
        keys=[];urls=[]
        for a in re.findall(r'\d{4}\.\d{4,5}',r['filename']):keys.append('arxiv:'+a)
        for a in re.findall(r'arXiv\s*:\s*(\d{4}\.\d{4,5})',r.get('first_two_pages_text',''),re.I):keys.append('arxiv:'+a)
        text=r.get('first_two_pages_text','').split('---PAGE---')[0]
        ds=re.findall(r'10\.\d{4,9}/[^\s<>"\x27]+',text,re.I)
        if ds:
            common=Counter(x.rstrip('.,;）)').lower() for x in ds).most_common(1)[0][0]
            keys.append('doi:'+common)
        indexes=[i for i in P['index_rows'] if r['filename'] in i.get('完整文件','')]
        for ix in indexes:
            urls.extend(re.findall('https?://[^ ;；]+',ix.get('来源URL','')))
        keys.extend(ukey(u) for u in urls)
        m=re.match(r'(\d{1,3})[_ ]',r['filename'])
        note=[x for x in R['legacy_reading_note_blocks'] if m and x['bibliography_number']==int(m[1])]
        scopes=[{'text':ix.get('阅读范围_非用户已读声明',''),'source':P['latest_index'],'locator':'编号='+ix['编号']} for ix in indexes]
        scopes.extend({'text':x['text'],'source':x['source'],'locator':f"L{x['line']}"} for x in note)
        # Later explicit rechecks that the old unified index did not contain.
        if n in (34,113,146):
            scopes.append({'text':'局部章节复核：SPD-Conv §3.1–3.2；UAV-DETR SOEP §3.3.1–3.3.2；HMF-DEIM §3.3.3 式12；各条仅对应本论文。',
                          'source':str(ROOT/D.MAIN/'360_线程A_补核_三篇章节号_20261026_v1.md'),'locator':'§1–3'})
        records.append({'kind':'pdf','n':n,'title':title,'keys':keys,'urls':urls,'pdf':r,
                        'local':r['filename'],'scopes':scopes,'index_numbers':[ix['编号'] for ix in indexes]})
    for e in EVENTS:
        records.append({'kind':'event','title':e['title'],'keys':[ukey(k) for k in e['url_keys']],
                        'urls':e['urls'],'local_files':e['local_pdf_files'],
                        'scopes':[{'text':' '.join(f'{k}: {v}' for k,v in e['reading_fields'].items()),
                                   'source':e['source'],'locator':e['locator']}]})
    for ix in R['index_records_without_pdf']:
        urls=re.findall('https?://[^ ;；]+',ix.get('来源URL',''))
        records.append({'kind':'index_only','title':ix['论文名称_来源记录'],'keys':[ukey(u) for u in urls],
                        'urls':urls,'index_numbers':[ix['编号']],
                        'scopes':[{'text':ix.get('阅读范围_非用户已读声明',''),'source':P['latest_index'],'locator':'编号='+ix['编号']}]})
    additions=[
      ('Learning from Imprecise and Fuzzy Observations','https://arxiv.org/pdf/1305.0698','T8','第4.1节式9、第4.3节、第5节式32'),
      ('Can We Trust Bounding Box Annotations for Object Detection?','https://openaccess.thecvf.com/content/CVPR2022W/VDU/html/Murrugarra-Llerena_Can_We_Trust_Bounding_Box_Annotations_for_Object_Detection_CVPRW_2022_paper.html','T9','标注/量化评估相关段；未声明全文'),
      ('Noise-Aware Evaluation of Object Detectors','https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf','T9','第3.4节式10'),
      ('End-to-End Object Detection with Transformers','https://arxiv.org/abs/2005.12872','T7','题名/摘要及相关方法片段；非全文阅读声明'),
      ('FaPN: Feature-Aligned Pyramid Network for Dense Image Prediction','https://arxiv.org/html/2108.07058','T2','第3.1–3.2节式1–5及第4.1–4.2节'),
      ('TOOD: Task-Aligned One-Stage Object Detection','https://arxiv.org/html/2108.07755','T5','第3.1–3.2节式1–12'),
    ]
    for title,url,theme,scope in additions:
        src=ROOT/'英文小论文/code/annotation_precision_novelty_review_20261026_v1/367c_集合监督候选_独立创新性审查.txt'
        if title.startswith('End-to-End'):
            src=ROOT/'英文小论文/相位机制论文_20261018/修订_20261018_v4/sources/added_sources.json'
        elif title.startswith(('FaPN','TOOD')):
            src=ROOT/'英文小论文/code/experiment_environment_preflight_20261008_v1/03_定位方案原文复核与交付边界.txt'
        assert src.is_file()
        records.append({'kind':'manual_record','title':title,'keys':[ukey(url)],'urls':[url],'theme':theme,
                        'scopes':[{'text':scope,'source':str(src),'locator':'题名/URL相关段'}]})
    dsu=DSU(len(records));key_owner={};file_owner={r['local']:i for i,r in enumerate(records) if r['kind']=='pdf'}
    for a,b in L.SAME_WORK:dsu.union(a-1,b-1)
    for group in R['duplicate_sha_groups']:
        for n in group[1:]:dsu.union(group[0]-1,n-1)
    titles={}
    for i,r in enumerate(records):
        for key in r['keys']:
            key=ukey(key)
            if key in key_owner:dsu.union(i,key_owner[key])
            else:key_owner[key]=i
        tk=title_key(r['title'])
        if len(tk)>22:
            if tk in titles:dsu.union(i,titles[tk])
            else:titles[tk]=i
        for fn in r.get('local_files',[]):
            if fn in file_owner:dsu.union(i,file_owner[fn])
    # Aliases whose identity is explicit in saved title/DOI records.
    aliases={'tolfnoiserobusttinyobjectlocalizationwithflows':228,
             'towardsefficientuseofmultiscalefeaturesintransformerbasedobjectdetectorsimfa':131,
             'decoupleddetr':144,
             '补查20261008saliencedetrarxiv240316131v1':254}
    for i,r in enumerate(records):
        if title_key(r['title']) in aliases:dsu.union(i,aliases[title_key(r['title'])]-1)
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[dsu.find(i)].append(r)
    items=[]
    for group in groups.values():
        pdfs=[r for r in group if r['kind']=='pdf']
        scopes=[x for r in group for x in r['scopes']]
        # Remove duplicated evidence without turning archive size into reading volume.
        unique={(x['source'],x['locator'],x['text']):x for x in scopes};scopes=list(unique.values())
        gs=[reading_grade(x['text']) for x in scopes]
        grade=('有局部方法/公式阅读记录' if '有局部方法/公式阅读记录' in gs else
               '有摘要/引言阅读记录' if '有摘要/引言阅读记录' in gs else '历史阅读证据未知')
        if pdfs:
            lead=min(pdfs,key=lambda r:r['n']);n=lead['n'];theme=L.ASSIGN[n];title=lead['title']
            filenames=[r['local'] for r in pdfs]
            if n==122:title=L.TITLE_OVERRIDES.get(n) or lead['title']
            scope=L.SCOPE_BY_ID.get(n,'二维目标检测相关；仍须按数据集/模型与本项目比较')
            availability='仅补充材料' if all('supplement' in x.lower() for x in filenames) else '有本地主文PDF'
            ident='首面题名/摘要核查（扫描页另视觉核对）；版本/重复已归并'
            if availability=='仅补充材料':ident='保存补充材料及CVF来源，主文题名/内容未在本地复核'
            current='首1–2页题名/摘要；不是全文评审'
            nums=[r['n'] for r in pdfs]
            excerpt=lead['pdf'].get('first_page_head','')[:400]
            if n==98:excerpt='扫描首面视觉核对：Hierarchical Mixtures of Experts and the EM Algorithm，Jordan/Jacobs，1994。'
        else:
            # Prefer an actual article title over an archival shorthand or filename.
            def score(r):return (not bool(re.search(r'\.pdf|\.html|reading_update|identity_recheck',r['title'])),len(r['title']))
            lead=max(group,key=score);title=lead['title'];theme=lead.get('theme') or classify_record(title)
            # Archival acronyms/filenames alone do not identify a research theme.
            if re.search(r'^\d+_|\.download$|\.html$|原检索简称|^补查_',title):theme='待核'
            explicit_theme={
              'AnyUp: Universal Feature Upsampling':'T2',
              'CZ Detector':'T2',
              'Consistent and Comprehensive Scale Aggregation Network for Drone-View Small Object Detection':'T2',
              'FSENet: Feature suppression and enhancement network for tiny object detection':'T1',
              'Parallel-Circuitized distillation for dense object detection':'T4',
              'Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning':'T4',
              'ESE-Net':'T1','SARES-DEIM v2':'T1',
            }
            theme=explicit_theme.get(title,theme)
            nums=[];scope='仅依已有来源记录核对；任务范围以记录为准，未新增全文读取'
            availability='无本地主文PDF（有历史检索/引用记录）'
            ident='记录题名及来源标识；未重新联网核验'
            current='核对来源/阅读记录；未取得新全文'
            excerpt=''
        uses=[]
        for c in U.match(pdfs,title):
            ev=source_ref(c['source'],c['anchor'])
            assert set(c['direction_ids']) <= {r['id'] for r in directions}
            uses.append({'direction_ids':c['direction_ids'],'use_and_limit':c['text'],'evidence':ev})
            if c['reading_scope']:
                scopes.append({'text':c['reading_scope'],'source':ev['path'],'locator':f"L{ev['line']}"})
        gs=[reading_grade(x['text']) for x in scopes]
        grade=('有局部方法/公式阅读记录' if '有局部方法/公式阅读记录' in gs else
               '有摘要/引言阅读记录' if '有摘要/引言阅读记录' in gs else '历史阅读证据未知')
        comparison=[x for x in scopes if re.search(r'作者.*(?:代码|源码).*(?:静态|核对|阅读)|(?:代码|源码).*(?:逐项|核对|静态)|逐式|逐项同构|实现.*对照|源文件核对',x['text'])]
        title_status='本地主文首面题名已核'
        if not pdfs:
            title_status='题名据历史记录；本轮未重新联网核验'
            if theme=='待核' or title in ['CFBA-FPN / CFCI','ESE-Net','SARES-DEIM v2','GLNet','GLSDet','CZ Detector']:
                title_status='原记录简称/文件名；完整题名待核，来源标识保留'
        elif availability=='仅补充材料':
            title_status='仅补充材料；主文完整题名及内容待核'
        items.append({'title':title,'theme':theme,'theme_name':L.THEMES.get(theme,'题名/主题待核'),
                      'scope':scope,'availability':availability,'identity_evidence':ident,'pdf_record_numbers':nums,
                      'title_verification':title_status,
                      'theme_basis':'按已保存主文首1–2页题名/摘要及阅读记录归类；未作全文方法综述' if pdfs else '仅据已存题名/摘要/局部阅读记录；身份或主题未定者单列，未新增检索',
                      'pdf_files':[r['pdf']['path'] for r in pdfs],
                      'pdf_sha256':[r['pdf']['sha256'] for r in pdfs],
                      'urls':sorted({u for r in group for u in r['urls']}),
                      'stable_keys':sorted({ukey(k) for r in group for k in r['keys']}),
                      'historical_reading_grade':grade,'historical_reading_evidence':scopes,
                      'has_formula_or_code_comparison_record':bool(comparison),
                      'comparison_evidence':comparison,'current_audit_scope':current,'firstpage_excerpt':excerpt,
                      'related_problem_families':L.THEME_FAMILIES.get(theme,[]),
                      'related_direction_ids':[r['id'] for r in directions if r['family'] in L.THEME_FAMILIES.get(theme,[])],
                      'mapping_basis':'按研究问题主题的关联；不表示每一篇都实际用于每个关联方案',
                      'documented_project_uses':uses,
                      'actual_related_direction_ids':sorted({i for u in uses for i in u['direction_ids']}),
                      'use_coverage_note':'其余用途未逐项对齐；空白不表示未使用或未阅读',
                      'full_read_attestation':'未替用户确认全文已读；局部记录不升级为全文',
                      'record_kinds':sorted({r['kind'] for r in group})})
    items.sort(key=lambda r:(not bool(r['pdf_record_numbers']),min(r['pdf_record_numbers'] or [10000]),r['title']))
    for i,r in enumerate(items,1):r['id']=f'L{i:03d}'
    return items,records


def main():
    directions=make_directions();papers,records=make_literature(directions)
    withpdf=[r for r in papers if r['pdf_record_numbers']]
    summary={'research_units':len(directions),'problem_families':len(D.FAMILIES),
             'method_proposals':sum(r['unit_type']=='方法方案' for r in directions),
             'diagnostic_hypotheses':sum(r['unit_type']=='诊断假设' for r in directions),
             'current_check_complete':sum(r['current_check_complete'] for r in directions),
             'with_planned_training_endpoint':sum(r['has_planned_training_endpoint'] for r in directions),
             'direction_status_counts':dict(Counter(r['status'] for r in directions)),
             'family_counts':dict(Counter(r['family'] for r in directions)),
             'main_library_pdf_files':260,'unique_pdf_sha256':255,'represented_pdf_works':len(withpdf),
             'works_with_main_pdf':sum(r['availability']=='有本地主文PDF' for r in papers),
             'supplement_only_works':sum(r['availability']=='仅补充材料' for r in papers),
             'bibliographic_rows_total':len(papers),'no_pdf_record_rows':len(papers)-len(withpdf),
             'pdf_work_themes':dict(Counter(r['theme_name'] for r in withpdf)),
             'pdf_work_reading_grades':dict(Counter(r['historical_reading_grade'] for r in withpdf)),
             'pdf_work_code_comparison_records':sum(r['has_formula_or_code_comparison_record'] for r in withpdf),
             'pdf_works_with_explicit_project_use_examples':sum(bool(r['documented_project_uses']) for r in withpdf),
             'all_records_with_explicit_project_use_examples':sum(bool(r['documented_project_uses']) for r in papers),
             'records_with_title_pending':sum('待核' in r['title_verification'] for r in papers),
             'all_record_reading_grades':dict(Counter(r['historical_reading_grade'] for r in papers)),
             'history_full_read_count':'无法确认；未把局部或下载记录升级为全文阅读',
             'historical_total_searched_papers':'无法从现存记录完整恢复；文献记录行数不是全部历史检索篇数'}
    for name,obj in (('direction_table_data.json',directions),('literature_table_data.json',papers),
                     ('assembled_summary.json',summary),('document_evidence_hashes.json',INPUTS)):
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('NO_PDF_RECORDS')
    for r in papers:
        if not r['pdf_record_numbers']:print(json.dumps({'id':r['id'],'title':r['title'],'theme':r['theme'],'keys':r['stable_keys'][:2]},ensure_ascii=False))


if __name__=='__main__':main()
