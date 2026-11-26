"""Clarify source-proven capacity and the exact baseline comparison scope."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
SOURCE_SHA = '2653ab4164cc454b3fad4d21c4c2e02e80febe00e80e4fc3e13759a7173114ac'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_backup.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev35.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
value = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
q = lambda s: '{' + W + '}' + s
assert digest(MASTER) == SOURCE_SHA and not BACKUP.exists() and not TARGET.exists()
gate_path = BUNDLE / 'msca_parameter_decomposition_20261108.json'
capacity_path = BUNDLE / 'base_full_capacity_changes_20261108.json'
gate, capacity = read(gate_path), read(capacity_path)
assert gate['removed_gate_trainable_parameters'] == 3841280
assert gate['trainable_parameters_by_component']['context_conv'] == 3146752
assert capacity['FULL_minus_BASE_parameters_by_category'] == {
    'stem': 5325, 'aggregation_projections': -47434, 'stage1_ESE': 4160,
    'stage4_MSCA_Dilated': 3841280, 'other_backbone': 0}
assert not gate['model_constructed'] and not capacity['model_constructed']
with ZipFile(MASTER) as source:
    document = etree.fromstring(source.read('word/document.xml'))
    comments = etree.fromstring(source.read('word/comments.xml'))
paragraphs = document.findall('w:body/w:p', NS)
changes = []

def replace(old, new):
    matches = [p for p in paragraphs if old in value(p)]
    assert len(matches) == 1, old
    paragraph = matches[0]
    before = value(paragraph)
    after = before.replace(old, new)
    nodes = paragraph.findall('.//w:t', NS)
    nodes[0].text = after
    for n in nodes[1:]:
        n.text = ''
    changes.append({'old': before, 'new': after})
    return paragraph

replace('The four stages use ESE, SimAM, MSCA-Lite and MSCA-Dilated (Fig. 3).',
        'The modified backbone uses ESE, SimAM, MSCA-Lite and MSCA-Dilated in its four stages (Fig. 3).')
replace('The local baseline uses two 1 × 1 ConvBNAct blocks with intermediate width C/2; the change therefore includes the projection path.',
        'The baseline uses two 1 × 1 ConvBNAct blocks with intermediate width C/2 and no separate ESE or SimAM gates; the modified backbone also changes this projection path.')
replace('Fig. 3. Stage-specific gates within the B0 backbone.',
        'Fig. 3. Stage-specific gates in the modified B0 backbone.')
replace('Table 1 compares the complete configuration and its additional capacity; it does not isolate the effects of the stem, contextual gates, aggregation projections or activations.',
        'Table 1 compares the complete configuration, including its ESE/SimAM gates, aggregation paths and activation choices. The gain cannot be assigned to MPS or MSCA alone.')
replace('BASE has different aggregation paths and activations, so it is not a joint-removal control.',
        'BASE lacks the ESE/SimAM gates and uses different aggregation paths and activations, so it is not a joint-removal control.')

cost_matches = [p for p in paragraphs if value(p).startswith('With k = 3, the effective sizes')]
assert len(cost_matches) == 1
cost_paragraph = cost_matches[0]
cost_sentence = ('In stage 4, the Dilated gate has 3,841,280 trainable parameters; '
                 'its 3072-to-1024 context projection accounts for 3,146,752.')
assert not cost_paragraph.findall('.//w:commentRangeStart', NS)
start = etree.SubElement(cost_paragraph, q('commentRangeStart'))
start.set(q('id'), '5')
run = etree.SubElement(cost_paragraph, q('r'))
properties = cost_paragraph.find('w:r/w:rPr', NS)
if properties is not None:
    run.append(deepcopy(properties))
node = etree.SubElement(run, q('t'))
node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
node.text = ' ' + cost_sentence
end = etree.SubElement(cost_paragraph, q('commentRangeEnd'))
end.set(q('id'), '5')
reference_run = etree.SubElement(cost_paragraph, q('r'))
reference_properties = etree.SubElement(reference_run, q('rPr'))
style = etree.SubElement(reference_properties, q('rStyle'))
style.set(q('val'), 'CommentReference')
reference = etree.SubElement(reference_run, q('commentReference'))
reference.set(q('id'), '5')
assert not any(c.get(q('id')) == '5' for c in comments)
comment = etree.SubElement(comments, q('comment'))
comment.set(q('id'), '5')
comment.set(q('author'), 'Codex')
comment.set(q('initials'), 'CDX')
comment.set(q('date'), datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
comment_p = etree.SubElement(comment, q('p'))
comment_r = etree.SubElement(comment_p, q('r'))
etree.SubElement(comment_r, q('t')).text = (
    '依据冻结源码与留存初始化张量静态计数：Dilated 门 3,841,280 个参数，其中 3C→C 投影 3,146,752。'
    'BASE 默认 se 分支为两次 ConvBNAct，并非独立 ESE；FULL 另含阶段1 ESE 和阶段2 SimAM。'
    '源证据见 msca_parameter_decomposition_20261108.json 与 base_full_capacity_changes_20261108.json。'
    '未构建或执行模型，参数计数不作为精度、时延或显存结论。')

with BACKUP.open('xb') as stream:
    stream.write(MASTER.read_bytes())
assert digest(BACKUP) == SOURCE_SHA
with ZipFile(MASTER) as source, ZipFile(TARGET, 'x') as target:
    for item in source.infolist():
        if item.filename == 'word/document.xml':
            content = etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
        elif item.filename == 'word/comments.xml':
            content = etree.tostring(comments, encoding='UTF-8', xml_declaration=True, standalone=True)
        else:
            content = source.read(item.filename)
        target.writestr(item, content)
texts = [value(p) for p in paragraphs]
abstract = next(p for p in texts if p.startswith('Small objects in unmanned aerial vehicle'))
for name, text in [('body_for_acronym_check.txt', '\n\n'.join(texts[texts.index('1. Introduction'):texts.index('References')])),
                   ('abstract_for_acronym_check.txt', abstract), ('candidate_text.txt', '\n\n'.join(texts))]:
    with (HERE / name).open('x', encoding='utf-8') as stream:
        stream.write(text + '\n')
report = {'status': 'CAPACITY_AND_BASELINE_SCOPE_CANDIDATE_PENDING_QA',
          'source_sha256': SOURCE_SHA, 'candidate_sha256': digest(TARGET),
          'evidence_sha256': {str(p.relative_to(PAPER)): digest(p) for p in [gate_path, capacity_path]},
          'paragraph_changes': changes, 'added_method_sentence': cost_sentence,
          'new_comment_id': 5, 'abstract_changed': False,
          'performance_tables_figures_equations_references_or_styles_changed': False,
          'new_training_or_evaluation': False, 'submission_ready': False}
with (HERE / 'capacity_scope_candidate.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
renderer_source = PAPER / '修订记录_20261103/第三十四轮MPS移除终点_20261108/render_v4.ps1'
renderer = renderer_source.read_text(encoding='utf-8-sig').replace('_rev34_v4.docx', '_rev35.docx').replace('rendered_word_v4', 'rendered_word').replace('render_metadata_v4.json', 'render_metadata.json')
with (HERE / 'render_candidate.ps1').open('x', encoding='utf-8') as stream:
    stream.write(renderer)
print(json.dumps({k: report[k] for k in ['status', 'candidate_sha256', 'new_comment_id', 'submission_ready']}))
