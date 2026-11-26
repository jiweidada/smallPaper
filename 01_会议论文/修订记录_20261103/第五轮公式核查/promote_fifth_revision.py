from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from datetime import datetime, timezone
from docx import Document
import json
import shutil

here = Path(__file__).resolve().parent
paper = here.parent.parent
source = paper / '稿件' / 'MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_第四轮备份.docx'
candidate = here / 'MPSMA-DEIM_conference_EN_revised_第五轮候选.docx'
old_sha = '3aeb70332b9a4dc9d66720a1ded4b5b067ce6eeba47a6f489a280e444bf79984'
new_sha = '3fdd80a2cac6eaddd9e79c9ee416b5f9bb5a6561ab5c240e47253b3bf743f033'
source_code = here.parent / 'source_snapshot/current_code/engine/extre_module/custom_nn/attention/MSCA.py'
source_sha = '04a3ebb82954c060843493a27a582694d9f056e7badc0629ffe0718674a6e888'

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

assert digest(source) == digest(backup) == old_sha
assert digest(candidate) == new_sha
assert digest(source_code) == source_sha
protocol = json.loads((paper / '远程公平对照_20261103/execution_protocol.json').read_text(encoding='utf-8'))
assert source_sha in json.dumps(protocol)
doc = Document(candidate)
assert len(doc.tables) == 6 and len(doc.inline_shapes) == 7
equations = doc.element.xpath('.//m:oMath')
assert len(equations) == 6
equation_four = ''.join(equations[3].itertext())
assert 'W₁ GAP(F) + b₁' in equation_four and '+ b₂' in equation_four
anchors = {}
for tag in ('commentRangeStart', 'commentRangeEnd', 'commentReference'):
    nodes = doc.element.xpath('.//w:' + tag)
    anchors[tag] = sorted(n.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id') for n in nodes)
assert len(anchors['commentRangeStart']) == 5
assert anchors['commentRangeStart'] == anchors['commentRangeEnd'] == anchors['commentReference']
with ZipFile(source) as a, ZipFile(candidate) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']

images = here / 'rendered_word'
prior_images = here.parent / '第四轮摘要核查/render_word_v2'
pngs = sorted(images.glob('page-*.png'))
assert [p.name for p in pngs] == [f'page-{n:02}.png' for n in range(1, 11)]
same = [int(p.stem.split('-')[-1]) for p in pngs if digest(p) == digest(prior_images / p.name)]
assert same == [1, 2, 3, 5, 10], same
qa = {
    'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED',
    'revision': 5,
    'docx_sha256': new_sha,
    'previous_docx_sha256': old_sha,
    'source_file_sha256': source_sha,
    'source_matches_frozen_training_manifest': True,
    'bias_confirmation': 'Both nn.Conv2d layers have non-None biases in the retained PyTorch 2.6.0 CPU environment; no model forward or optimizer update was performed.',
    'pdf_sha256': digest(images / 'manuscript.pdf'),
    'renderer': 'Word read-only ExportAsFixedFormat and bundled Poppler pdftoppm',
    'packaged_renderer_failure': 'render_docx.py could not find soffice.exe on PATH',
    'pages': 10,
    'inspected_pages': list(range(1, 11)),
    'review_method': 'Pages 1, 2, 3, 5 and 10 are byte-identical to the previously inspected fourth revision. Pages 4, 6, 7, 8 and 9 were reopened and visually inspected after this edit.',
    'page_png_sha256': {p.name: digest(p) for p in pngs},
    'structural_counts': {'tables': 6, 'equations': 6, 'inline_images': 7, 'comment_starts': 5, 'comment_ends': 5, 'comment_references': 5},
    'only_document_xml_changed': True,
    'warnings': 'Poppler reported no display font for Symbol and ArialUnicode; the altered formula and visible symbols were checked in the page PNG.',
    'scope': 'Formula fidelity, layout and structural integrity only. Fair-pair results and submission readiness remain pending.'
}

entry = paper / '当前修订入口.txt'
recovery = paper / '远程公平对照_20261103/操作恢复说明.txt'
integration = paper / '远程公平对照_20261103/结果并入与验收清单.txt'
notes = here.parent / '修订说明与未解决问题.txt'
acceptance = here.parent / '目标验收与阻塞记录.txt'
contents = {p: p.read_text(encoding='utf-8') for p in (entry, recovery, integration, notes, acceptance)}
assert old_sha in contents[entry] and old_sha in contents[recovery] and old_sha in contents[integration]
assert '十二、第五轮' not in contents[notes]

top = ('【最新入口：第五轮公式修订，公平对照仍在运行】\n'
       f'当前稿件：{source}\nSHA256：{new_sha}\n'
       '通道门公式显式补入两层1×1卷积的偏置，并说明权重与偏置符号；现存源码与冻结实验源码哈希一致。'
       '候选稿10页版式及文档结构已核查，第四轮备份和本轮核验保留在修订记录_20261103/第五轮公式核查。\n'
       '新对照尚未完成，不能据早期指标宣布达到投稿要求；最新状态以远程公平对照_20261103/training_execution_ledger.json为准。\n'
       '以下旧入口及状态为历史记录，由本段覆盖。\n\n')
contents[entry] = top + contents[entry]
old_recovery = f'第四轮稿件MPSMA-DEIM_conference_EN_revised.docx SHA256为{old_sha}；第三轮稿件备份见修订记录_20261103/第三轮留档。'
new_recovery = f'第五轮稿件MPSMA-DEIM_conference_EN_revised.docx SHA256为{new_sha}；第四轮备份及公式核验见修订记录_20261103/第五轮公式核查。'
assert old_recovery in contents[recovery]
contents[recovery] = contents[recovery].replace(old_recovery, new_recovery, 1)
contents[integration] = contents[integration].replace(f'当前稿件SHA256：{old_sha}', f'当前稿件SHA256：{new_sha}', 1)
contents[notes] += ('\n十二、第五轮：通道门公式与实现对齐\n'
                    '现存MSCA_Dilated.channel_att的两个1×1卷积均使用偏置。式(4)显式补入b₁和b₂，'
                    '并在相邻正文说明W₁/W₂与b₁/b₂。已用保存的源码和PyTorch 2.6.0 CPU层构造核实偏置存在；'
                    '未做模型前向、评价或训练。当前源码SHA与新实验冻结清单一致，不能因此认证历史训练源码。\n'
                    f'第四轮备份SHA256 {old_sha}；当前稿件SHA256 {new_sha}。'
                    '仅document.xml的两处内容发生变化，6表、7图、6式及5个完整批注锚点保留。'
                    '10页已完成渲染核查；新公平对照未写入早期结果，尚不能认定可投稿。\n')
contents[acceptance] += (f'\n第五轮公式修订：当前稿件SHA256 {new_sha}。已补充通道门两层卷积偏置并检查10页；'
                         '公平对照仍在运行，目标active，最终比较与投稿验收尚待完成。\n')

(here / 'fifth_revision_visual_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
temporary = source.with_name(source.stem + '_fifth_promote.tmp')
assert not temporary.exists()
shutil.copy2(candidate, temporary)
assert digest(temporary) == new_sha
temporary.replace(source)
assert digest(source) == new_sha
for path, text in contents.items():
    path.write_text(text, encoding='utf-8')
patch = json.loads((here / 'fifth_revision_pending_qa.json').read_text(encoding='utf-8'))
patch.update({'status': 'PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA', 'promoted_utc': datetime.now(timezone.utc).isoformat()})
(here / 'fifth_revision_patch.json').write_text(json.dumps(patch, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status':'PROMOTED','sha256':new_sha,'pages':len(pngs),'changed_pages_reviewed':[4,6,7,8,9]}, ensure_ascii=False))
