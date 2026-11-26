from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from datetime import datetime, timezone
from docx import Document
import json
import shutil

here = Path(__file__).resolve().parent
rev = here.parent
paper = rev.parent
source = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_第五轮备份.docx'
candidate = here / 'MPSMA-DEIM_conference_EN_revised_第六轮候选_v2.docx'
old_sha = '3fdd80a2cac6eaddd9e79c9ee416b5f9bb5a6561ab5c240e47253b3bf743f033'
new_sha = 'dc74f01ef224d9a76ae58f262b88929214dc81881b41d314123c538ac3af1ed3'
old_fig_sha = '216bef948774510d8e168a65977729cb4b691d006b32239ed1c168f89545f911'
new_fig_sha = '4845bfab42ecfaac89ebbfc8b814287fd225bea2b607f25f001420cbc62ac168'

def digest(path): return sha256(path.read_bytes()).hexdigest()

assert digest(source) == digest(backup) == old_sha and digest(candidate) == new_sha
fig_dir = rev / 'figures'
fig_png, fig_svg = fig_dir / 'fig2_mps.png', fig_dir / 'fig2_mps.svg'
new_png, new_svg = here / 'fig2_mps_sixth_candidate.png', here / 'fig2_mps_sixth_candidate.svg'
assert digest(fig_png) == old_fig_sha and digest(new_png) == new_fig_sha
diagram_builder = rev / 'make_figures.py'
builder_text = diagram_builder.read_text(encoding='utf-8')
old_legend = "a.text(.2,.15,'Conv blocks include BN and the configured activation. DWConv denotes depthwise convolution.',fontsize=8.6)"
new_legend = ("a.text(.2,.32,'ConvBNAct: convolution → BN → activation → learned scalar affine (use_lab=True).',fontsize=8.6)\n"
              "a.text(.2,.13,'Detail, main, fusion and final use ConvBNAct; high-pass and gate omit the affine. DWConv: depthwise.',fontsize=8.6)")
assert builder_text.count(old_legend) == 1

doc = Document(candidate)
assert len(doc.tables) == 6 and len(doc.inline_shapes) == 7 and len(doc.element.xpath('.//m:oMath')) == 6
assert 'learned scalar scale and bias after activation' in '\n'.join(p.text for p in doc.paragraphs)
intro = [p for p in doc.paragraphs if p.text.startswith('For each channel of a feature map T, define')]
assert len(intro) == 1 and intro[0].paragraph_format.keep_with_next is True
anchors = {}
for tag in ('commentRangeStart', 'commentRangeEnd', 'commentReference'):
    nodes = doc.element.xpath('.//w:' + tag)
    anchors[tag] = sorted(n.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id') for n in nodes)
assert len(anchors['commentRangeStart']) == 5 and anchors['commentRangeStart'] == anchors['commentRangeEnd'] == anchors['commentReference']
with ZipFile(source) as a, ZipFile(candidate) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert sorted(n for n in a.namelist() if a.read(n) != b.read(n)) == ['word/document.xml', 'word/media/image9.png']
    assert sha256(b.read('word/media/image9.png')).hexdigest() == new_fig_sha

images = here / 'rendered_word_v2'
prior_images = rev / '第五轮公式核查/rendered_word'
pngs = sorted(images.glob('page-*.png'))
assert [p.name for p in pngs] == [f'page-{n:02}.png' for n in range(1, 11)]
same = [int(p.stem.split('-')[-1]) for p in pngs if digest(p) == digest(prior_images / p.name)]
assert same == [1, 2, 5, 9, 10], same
qa = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'revision': 6, 'docx_sha256': new_sha,
      'previous_docx_sha256': old_sha, 'old_figure_sha256': old_fig_sha, 'new_figure_sha256': new_fig_sha,
      'pdf_sha256': digest(images / 'manuscript.pdf'), 'renderer': 'Word read-only ExportAsFixedFormat and bundled Poppler pdftoppm',
      'packaged_renderer_failure': 'render_docx.py could not find soffice.exe on PATH', 'pages': 10,
      'inspected_pages': list(range(1, 11)),
      'review_method': 'Pages 1, 2, 5, 9 and 10 are byte-identical to the previously inspected fifth revision; pages 3, 4, 6, 7 and 8 were reopened and visually inspected.',
      'page_png_sha256': {p.name: digest(p) for p in pngs},
      'structural_counts': {'tables': 6, 'equations': 6, 'inline_images': 7, 'comment_starts': 5, 'comment_ends': 5, 'comment_references': 5},
      'changed_parts': ['word/document.xml', 'word/media/image9.png'],
      'equation_2_intro_kept_with_equation': True,
      'warnings': 'Poppler reported no display font for Symbol and ArialUnicode; visible symbols and the updated diagram were checked in PNGs.',
      'scope': 'Method fidelity and layout only; fair-pair completion and submission readiness remain pending.'}

entry = paper / '当前修订入口.txt'
recovery = paper / '远程公平对照_20261103/操作恢复说明.txt'
integration = paper / '远程公平对照_20261103/结果并入与验收清单.txt'
notes = rev / '修订说明与未解决问题.txt'
acceptance = rev / '目标验收与阻塞记录.txt'
texts = {p: p.read_text(encoding='utf-8') for p in [entry, recovery, integration, notes, acceptance]}
assert old_sha in texts[entry] and old_sha in texts[recovery] and old_sha in texts[integration]
assert '十四、第六轮' not in texts[notes]
top = ('【最新入口：第六轮结构图与方法说明修订，公平对照仍在运行】\n'
       f'当前稿件：{source}\nSHA256：{new_sha}\n'
       '图2和正文补充启用的LAB：detail/main/fusion/final的ConvBNAct在激活后有可学习标量仿射，高频和空间门路径没有。'
       '前一稿和原图备份及10页版式核验见修订记录_20261103/第六轮结构核查。\n'
       '主要结论证据索引：修订记录_20261103/主要结论证据索引/主要结论证据索引_第六轮.txt。'
       '新对照未完成，尚不能认定可投稿；实时状态见远程公平对照_20261103/training_execution_ledger.json。\n'
       '以下旧入口及状态为历史记录，由本段覆盖。\n\n')
texts[entry] = top + texts[entry]
old_recovery = f'第五轮稿件MPSMA-DEIM_conference_EN_revised.docx SHA256为{old_sha}；第四轮备份及公式核验见修订记录_20261103/第五轮公式核查。'
new_recovery = f'第六轮稿件MPSMA-DEIM_conference_EN_revised.docx SHA256为{new_sha}；第五轮备份及结构图核验见修订记录_20261103/第六轮结构核查。'
assert old_recovery in texts[recovery]
texts[recovery] = texts[recovery].replace(old_recovery, new_recovery, 1)
texts[integration] = texts[integration].replace(f'当前稿件SHA256：{old_sha}', f'当前稿件SHA256：{new_sha}', 1)
texts[notes] += ('\n十四、第六轮：MPS仿射层与图文一致性\n'
                 '现存N1/N4/S1配置及本次FULL冻结配置均启用use_lab；当前EnhancedStemBlock把该设置传入detail、main、fusion及final-downsample的ConvBNAct。'
                 '其激活后有每块一组可学习标量scale和bias，高频及空间门直接卷积路径不含LAB。N4/S1保存的checkpoint元信息也含对应LAB张量；这不证明历史运行时源码哈希。\n'
                 '图2可编辑SVG及嵌入PNG的图例和第3.2节同步说明该结构，并将式(2)引导段与公式保持同页。'
                 f'第五轮备份SHA256 {old_sha}；当前稿件SHA256 {new_sha}。'
                 '仅document.xml及图2媒体改变，6表、7图、6式、5组批注锚点保留；10页完成可视检查。公平对照结果仍待完成与核验。\n')
texts[acceptance] += f'\n第六轮结构修订：当前稿件SHA256 {new_sha}。图2和正文补充LAB路径，10页版式核查通过；固定预算对照仍在运行，尚不能投稿。\n'

(here / 'sixth_revision_visual_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
for original, name in [(fig_png, 'fig2_mps_第五轮备份.png'), (fig_svg, 'fig2_mps_第五轮备份.svg'), (diagram_builder, 'make_figures_第五轮备份.py')]:
    out = here / name
    if out.exists(): assert digest(out) == digest(original)
    else: shutil.copy2(original, out)
temporary = source.with_name(source.stem + '_sixth_promote.tmp')
assert not temporary.exists()
shutil.copy2(candidate, temporary)
assert digest(temporary) == new_sha
temporary.replace(source)
shutil.copy2(new_png, fig_png)
shutil.copy2(new_svg, fig_svg)
diagram_builder.write_text(builder_text.replace(old_legend, new_legend, 1), encoding='utf-8')
assert digest(source) == new_sha and digest(fig_png) == new_fig_sha
for p, content in texts.items(): p.write_text(content, encoding='utf-8')
patch = json.loads((here / 'sixth_revision_pending_qa.json').read_text(encoding='utf-8'))
patch.update({'status': 'PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA', 'candidate_sha256': new_sha,
              'additional_change': 'Keep equation 2 with its introduction.', 'promoted_utc': datetime.now(timezone.utc).isoformat()})
(here / 'sixth_revision_patch.json').write_text(json.dumps(patch, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PROMOTED', 'sha256': new_sha, 'pages': 10, 'changed_pages_reviewed': [3, 4, 6, 7, 8]}, ensure_ascii=False))
