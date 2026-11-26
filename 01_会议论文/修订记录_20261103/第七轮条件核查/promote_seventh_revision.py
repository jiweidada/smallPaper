from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from datetime import datetime, timezone
from docx import Document
import json
import shutil

here = Path(__file__).resolve().parent
rev, paper = here.parent, here.parent.parent
source = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_第六轮备份.docx'
candidate = here / 'MPSMA-DEIM_conference_EN_revised_第七轮候选_v2.docx'
old_sha = 'dc74f01ef224d9a76ae58f262b88929214dc81881b41d314123c538ac3af1ed3'
new_sha = 'a494f6b7ca2b1c200232ab49fa7f3d3831b161e2552990da3b88109f7b599de2'

def digest(path): return sha256(path.read_bytes()).hexdigest()

assert digest(source) == digest(backup) == old_sha and digest(candidate) == new_sha
doc = Document(candidate)
assert len(doc.tables) == 6 and len(doc.inline_shapes) == 7 and len(doc.element.xpath('.//m:oMath')) == 6
full_text = '\n'.join(p.text for p in doc.paragraphs)
assert 'from a local legacy code tree' in full_text and 'GPU timing in Table 5 comes from a server package' in full_text
assert 'The source of the difference between code trees has not been isolated' in full_text
assert 'A separately archived module breakdown' in full_text
anchors = {}
for tag in ('commentRangeStart', 'commentRangeEnd', 'commentReference'):
    nodes = doc.element.xpath('.//w:' + tag)
    anchors[tag] = sorted(n.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id') for n in nodes)
assert len(anchors['commentRangeStart']) == 5 and anchors['commentRangeStart'] == anchors['commentRangeEnd'] == anchors['commentReference']
with ZipFile(source) as a, ZipFile(candidate) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']

images, first_images = here / 'rendered_word_v2', here / 'rendered_word'
previous_images = rev / '第六轮结构核查/rendered_word_v2'
pngs = sorted(images.glob('page-*.png'))
assert [p.name for p in pngs] == [f'page-{n:02}.png' for n in range(1, 11)]
same_first = [int(p.stem.split('-')[-1]) for p in pngs if digest(p) == digest(first_images / p.name)]
assert same_first == [1, 2, 3, 4, 5, 6, 7, 10], same_first
assert all(digest(images / f'page-{n:02}.png') == digest(previous_images / f'page-{n:02}.png') for n in [1,2,3,4,5,6,10])
qa = {'status':'VISUAL_AND_STRUCTURAL_QA_PASSED', 'revision':7, 'docx_sha256':new_sha, 'previous_docx_sha256':old_sha,
      'pdf_sha256':digest(images / 'manuscript.pdf'), 'renderer':'Word read-only ExportAsFixedFormat and bundled Poppler pdftoppm',
      'packaged_renderer_failure':'render_docx.py could not find soffice.exe on PATH', 'pages':10, 'inspected_pages':list(range(1,11)),
      'review_method':'Pages 1–6 and 10 are byte-identical to the previously reviewed sixth revision; page 7 is identical to the reviewed first candidate; pages 8–9 were reopened after the final text edit.',
      'page_png_sha256':{p.name:digest(p) for p in pngs},
      'structural_counts':{'tables':6,'equations':6,'inline_images':7,'comment_starts':5,'comment_ends':5,'comment_references':5},
      'only_document_xml_changed':True,
      'warnings':'Poppler reported no display font for Symbol and ArialUnicode; visible content was checked in PNGs.',
      'scope':'Clarification of historical source conditions and document layout; fair-pair results and submission readiness remain pending.'}

entry = paper / '当前修订入口.txt'
recovery = paper / '远程公平对照_20261103/操作恢复说明.txt'
integration = paper / '远程公平对照_20261103/结果并入与验收清单.txt'
notes = rev / '修订说明与未解决问题.txt'
acceptance = rev / '目标验收与阻塞记录.txt'
texts = {p:p.read_text(encoding='utf-8') for p in [entry,recovery,integration,notes,acceptance]}
assert old_sha in texts[entry] and old_sha in texts[recovery] and old_sha in texts[integration]
assert '十五、第七轮' not in texts[notes]
top = ('【最新入口：第七轮实验条件区分，公平对照仍在运行】\n'
       f'当前稿件：{source}\nSHA256：{new_sha}\n'
       '复杂度表明确采用本地旧代码树归档值，GPU时延来自服务器包；另一树FLOPs不同的原因未隔离，不能混作同一测量。'
       '保留加载链限制；第六轮备份与10页版式核验见修订记录_20261103/第七轮条件核查。\n'
       '主要结论证据索引：修订记录_20261103/主要结论证据索引/主要结论证据索引_第七轮.txt。'
       '新公平对照未完成，尚不能认定可投稿；实时状态见远程公平对照_20261103/training_execution_ledger.json。\n'
       '以下旧入口及状态为历史记录，由本段覆盖。\n\n')
texts[entry] = top + texts[entry]
old_recovery = f'第六轮稿件MPSMA-DEIM_conference_EN_revised.docx SHA256为{old_sha}；第五轮备份及结构图核验见修订记录_20261103/第六轮结构核查。'
new_recovery = f'第七轮稿件MPSMA-DEIM_conference_EN_revised.docx SHA256为{new_sha}；第六轮备份及测量条件核验见修订记录_20261103/第七轮条件核查。'
assert old_recovery in texts[recovery]
texts[recovery] = texts[recovery].replace(old_recovery,new_recovery,1)
texts[integration] = texts[integration].replace(f'当前稿件SHA256：{old_sha}',f'当前稿件SHA256：{new_sha}',1)
texts[notes] += ('\n十五、第七轮：复杂度与时延来源区分\n'
                 '旧归档明确把24.86/29.63 GFLOPs标为本地旧树测量，25.37/30.14 GFLOPs标为服务器新包测量。'
                 '第4.4节明确表4使用前者，表5为单独的服务器GPU时延记录；参数分解也注明单独归档。'
                 '两树差异原因未隔离，不再将差值归因于计数工具，也不以模型标签相同认定代码字节或加载权重相同。\n'
                 f'第六轮备份SHA256 {old_sha}；当前稿件SHA256 {new_sha}。'
                 '仅document.xml的三处文字改变，6表、7图、6式及5组批注锚点保留，10页完成版式检查。未新增测量或写入早期训练结果。\n')
texts[acceptance] += f'\n第七轮条件修订：当前稿件SHA256 {new_sha}。已分开旧树复杂度与服务器时延来源；最终公平对照及投稿验收仍待完成。\n'

(here / 'seventh_revision_visual_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
tmp = source.with_name(source.stem + '_seventh_promote.tmp')
assert not tmp.exists()
shutil.copy2(candidate,tmp)
assert digest(tmp)==new_sha
tmp.replace(source)
assert digest(source)==new_sha
for p, text in texts.items(): p.write_text(text,encoding='utf-8')
patch = json.loads((here / 'seventh_revision_pending_qa.json').read_text(encoding='utf-8'))
patch.update({'status':'PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA','candidate_sha256':new_sha,
              'additional_change':'Identify the separately archived module breakdown.', 'promoted_utc':datetime.now(timezone.utc).isoformat()})
(here / 'seventh_revision_patch.json').write_text(json.dumps(patch,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PROMOTED','sha256':new_sha,'pages':10,'changed_pages_reviewed':[7,8,9]},ensure_ascii=False))
