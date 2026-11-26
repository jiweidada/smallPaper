"""Promote the rendered citation-only revision and synchronize source records."""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import json,os,re,shutil
HERE=Path(__file__).resolve().parent
RECORDS=HERE.parent;PAPER=RECORDS.parent
SOURCE=PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
CANDIDATE=HERE/'MPSMA-DEIM_conference_EN_revised_第八轮候选.docx'
OLD='a494f6b7ca2b1c200232ab49fa7f3d3831b161e2552990da3b88109f7b599de2'
NEW='37026584852e657c15bce14d522d261762acf2695c7bd681948b56ff0ccc66ac'
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert sha(SOURCE)==OLD and sha(CANDIDATE)==NEW
patch=read(HERE/'eighth_revision_pending_qa.json');qa=read(HERE/'eighth_revision_visual_qa.json')
assert qa['status']=='VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['docx_sha256']==NEW
assert qa['inspected_pages']==list(range(1,11))
for name,digest in qa['page_png_sha256'].items():assert sha(HERE/'rendered_word'/name)==digest
archive=RECORDS/'citation_archive.json';log=RECORDS/'引用核查记录.txt'
assert sha(archive)==sha(HERE/'citation_archive_第七轮备份.json')
assert sha(log)==sha(HERE/'引用核查记录_第七轮备份.txt')
entries=read(archive)
assert [e['reference'] for e in entries]==list(range(1,11))
tool_sources=read(RECORDS/'dataset_eval_audit/official_toolkit/sources.json')
for s in tool_sources:assert sha(Path(s['path']))==s['sha256']
entries.append({'reference':11,'source_url':'https://github.com/VisDrone/VisDrone2018-DET-toolkit',
 'version':'Official source snapshots accessed 2026-11-03; identified by file SHA, not an unverified commit.',
 'reading_scope':'Earlier targeted static inspection of evalDET.m and dropObjectsInIgr.m; never executed.',
 'download_status':'Existing source snapshots reused','source_files':tool_sources,
 'metadata_added_in_revision8':'The seventh manuscript already cited this source; its snapshot metadata was stored separately from the ten-paper archive.'})
mapping={int(k):v for k,v in patch['citation_number_mapping_old_to_new'].items()}
for e in entries:
    previous=e['reference'];e['reference_in_seventh_revision']=previous;e['reference']=mapping[previous]
    if previous==1:
        e['followup_20261104']={'reading_scope':'Publisher abstract and indexed HTML excerpts only; no full-text PDF obtained.',
          'retrieval_receipt':str(HERE/'reference1_retrieval.json'),'pdf_status':'HTTP403; no PDF archived',
          'claim_scope':'General small-object/complex-background challenges. Occlusion now has a direct VisDrone primary citation, new reference 2.'}
    if previous==2:
        e['conference_program_url']='https://2024.ieeeigarss.org/view_paper.php?PaperNum=5618'
        e['author_lab_url']='https://uwaterloo.ca/geospatial-intelligence/publication/presentations-conferences'
        e['previous_program_link_correction']={'wrong_url':'https://2024.ieeeigarss.org/view_paper.php?PaperNum=1464',
          'reason':'The old main heading is another paper; the target title is only in the shared session footer. Correct detail page: PaperNum=5618.'}
        e['followup_20261104']={'reading_scope':'Publisher abstract, correct conference heading/authors, and author-lab publication list. Full paper not obtained.',
          'supported_claim':'Convolutional local-context enhancement combined with Swin Transformer.',
          'primary_html_receipts':str(HERE/'primary_web_retrieval.json')}
    if previous==9:
        e['followup_20261104']={'reading_scope':'Targeted existing original, Occlusion paragraph; archived extracted text lines 330-381.',
          'supports':'Occlusion by other objects and background obstacles; cited in the first introduction paragraph as new reference 2.'}
entries.sort(key=lambda e:e['reference'])
assert [e['reference'] for e in entries]==list(range(1,12))
oldlog=log.read_text(encoding='utf-8-sig');body='[1] '+oldlog.split('[1] ',1)[1]
body,footer=body.split('\n原文库：',1);footer='\n原文库：'+footer
starts=list(re.finditer(r'(?m)^\[(\d+)\] ',body));blocks={}
for i,m in enumerate(starts):
    oldnum=int(m.group(1));chunk=body[m.start():starts[i+1].start() if i+1<len(starts) else len(body)].rstrip()
    chunk=re.sub(r'^\[\d+\]',f'[{mapping[oldnum]}]',chunk,count=1)
    if oldnum==1:chunk+='\n2026-11-04补核：出版社摘要/索引HTML节选支持一般背景；官方PDF仍返回403，未取得全文。遮挡事实另以新文献[2]直接支持。'
    if oldnum==2:chunk+='\n2026-11-04更正：原程序PaperNum=1464主体是另一篇论文，目标题名仅见同场页脚。现改为https://2024.ieeeigarss.org/view_paper.php?PaperNum=5618，并与作者实验室清单交叉核对。'
    if oldnum==9:chunk+='\n2026-11-04定向核对Occlusion段落（归档提取文本330–381行），支持引言中的遮挡论述；编号从旧[9]提前为新[2]。'
    blocks[mapping[oldnum]]=chunk
assert sorted(blocks)==list(range(1,12))
footer=footer.replace('PaperNum=1464','PaperNum=5618')
footer+='\n作者实验室交叉核对：https://uwaterloo.ca/geospatial-intelligence/publication/presentations-conferences\n补核范围见引用补核_20261104/引用补核报告.txt；未将有限核查表述为全文通读。\n'
newlog='引用核查记录（2026-11-04，第八轮编号与来源补核）\n共11条引用，按首次出现编号并均在正文使用。旧编号及范围保存在本轮第七轮备份中。\n\n'+'\n\n'.join(blocks[n] for n in range(1,12))+'\n'+footer
ref2=next(e for e in entries if e['reference']==2)
report={'status':'CITATION_SUPPORT_AND_NUMBERING_REVISED','checked_local_utc':datetime.now(timezone.utc).isoformat(),
 'manuscript_sha256':NEW,'previous_manuscript_sha256':OLD,'reference_count':11,
 'citation_archive_records':11,'first_appearance_order':patch['first_appearance_order'],
 'citation_number_mapping_old_to_new':mapping,'orphan_citations':0,'orphan_references':0,
 'source_link_correction':{'old_url':'https://2024.ieeeigarss.org/view_paper.php?PaperNum=1464',
    'correct_url':'https://2024.ieeeigarss.org/view_paper.php?PaperNum=5618','title_authors_doi_unchanged':True},
 'occlusion_primary_evidence':{'new_reference':2,'source_text':ref2['text'],'source_text_sha256':sha(Path(ref2['text'])),
    'locator':'Occlusion paragraph, extracted lines 330-381'},
 'pdf_access_limit':'Tang survey PDF and Liang full paper not obtained. Only actual abstract/indexed excerpt/metadata reading is claimed.',
 'scope_limit':'Other references retain earlier targeted checks. No new complete full-text, plagiarism or retraction-screening claim.',
 'visual_qa':str(HERE/'eighth_revision_visual_qa.json'),'experiments_added':False,'submission_ready':False}

# Prepare records, then promote the exact reviewed bytes.
save(archive,entries);log.write_text(newlog,encoding='utf-8')
save(HERE/'citation_followup_report.json',report)
(HERE/'引用补核报告.txt').write_text('''引用补核与第八轮修订（2026-11-04）
引言遮挡论述已增加VisDrone-DET2019原文的直接支持（Occlusion段）。该文由旧[9]提前为新[2]；全部11条参考文献及正文按首次出现同步编号。数值向量没有作引用替换。
来源记录的IGARSS PaperNum=1464主体是同场另一篇论文，目标只在页脚。现已沿官方链接更正为PaperNum=5618，按主标题和作者块核对，并与作者实验室清单交叉验证。稿件中的Liang题名、作者、页码和DOI未变，现为[3]。
原PDF检索仍未取得Tang综述和Liang全文，实际阅读范围限于已获得摘要、索引节选和元数据；不声称全文通读。11条归档记录现包含原先单独存储的官方评价源码快照与SHA，源码未执行。
本轮只改引文及文献顺序。全部非引用段落、表格数值、配置向量、6式、7图和5组批注锚点保持；10页经Word只读导出和逐页图像检查。第七轮稿件与来源记录备份保留。
新公平主对照完整训练、四个最终端点核验及相应论文结果修订仍待完成，尚不能投稿。
''',encoding='utf-8')
temp=SOURCE.with_name(SOURCE.stem+'_promotion.tmp.docx');assert not temp.exists()
shutil.copy2(CANDIDATE,temp);os.replace(temp,SOURCE)
assert sha(SOURCE)==NEW and sha(HERE/'MPSMA-DEIM_conference_EN_revised_第七轮备份.docx')==OLD
patch.update(status='PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA',current_manuscript_sha256=NEW,
             visual_qa_sha256=sha(HERE/'eighth_revision_visual_qa.json'))
save(HERE/'eighth_revision_patch.json',patch)
print(json.dumps({'status':patch['status'],'manuscript_sha256':NEW,'pages':10,'references':11,'only_document_xml_changed':True}))
