"""Add the verified BASE/FULL endpoints while preserving the existing OOXML."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

from lxml import etree


sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev31_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev32.docx'
PREVIOUS_SHA = 'cb213ccd8553e8965485ce4b588b1a32752dffccd3f706581979df59b23dfe98'
PAIR_PATH = PAPER / '远程公平对照_20261103/断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
METHOD_PATH = PAPER / '远程公平对照_20261103/模块消融/method_source_binding_20261108.json'
PART = 'word/document.xml'
COMMENTS_PART = 'word/comments.xml'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

helper_path = HERE.parent / '第三十轮主对照口径_20261106/patch_controlled_comparison.py'
spec = importlib.util.spec_from_file_location('prior_controlled_patch', helper_path)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def q(name):
    return f'{{{W}}}{name}'


def paragraph_like(template, value):
    paragraph = etree.Element(q('p'))
    properties = template.find('w:pPr', NS)
    if properties is not None:
        paragraph.append(deepcopy(properties))
    run = etree.SubElement(paragraph, q('r'))
    run_properties = template.find('w:r/w:rPr', NS)
    if run_properties is not None:
        run.append(deepcopy(run_properties))
    etree.SubElement(run, q('t')).text = value
    return paragraph


def endpoint_table(template, rows):
    table = deepcopy(template)
    header = deepcopy(table.find('w:tr', NS))
    body = deepcopy(table.findall('w:tr', NS)[1])
    for row in table.findall('w:tr', NS):
        table.remove(row)
    widths = [1300, 900, 700, 1050, 1050, 1050, 1050, 1050]
    properties = table.find('w:tblPr', NS)
    width = properties.find('w:tblW', NS)
    width.set(q('type'), 'dxa')
    width.set(q('w'), str(sum(widths)))
    layout = properties.find('w:tblLayout', NS)
    if layout is None:
        layout = etree.SubElement(properties, q('tblLayout'))
    layout.set(q('type'), 'fixed')
    grid = table.find('w:tblGrid', NS)
    grid.clear()
    for value in widths:
        etree.SubElement(grid, q('gridCol')).set(q('w'), str(value))
    for index, values in enumerate(rows):
        row = deepcopy(header if index == 0 else body)
        while len(row.findall('w:tc', NS)) < len(widths):
            row.append(deepcopy(row.findall('w:tc', NS)[-1]))
        for cell, value, cell_width in zip(row.findall('w:tc', NS), values, widths):
            cell_properties = cell.find('w:tcPr', NS)
            cell_properties.find('w:tcW', NS).set(q('w'), str(cell_width))
            template_paragraph = cell.find('w:p', NS)
            paragraph = paragraph_like(template_paragraph, value)
            for child in list(cell):
                if child.tag != q('tcPr'):
                    cell.remove(child)
            cell.append(paragraph)
            run_properties = paragraph.find('w:r/w:rPr', NS)
            if run_properties is None:
                run_properties = etree.Element(q('rPr'))
                paragraph.find('w:r', NS).insert(0, run_properties)
            size = run_properties.find('w:sz', NS)
            if size is None:
                size = etree.SubElement(run_properties, q('sz'))
            size.set(q('val'), '19')
        row_properties = row.find('w:trPr', NS)
        if row_properties is None:
            row_properties = etree.Element(q('trPr'))
            row.insert(0, row_properties)
        if row_properties.find('w:cantSplit', NS) is None:
            etree.SubElement(row_properties, q('cantSplit'))
        if index == 0 and row_properties.find('w:tblHeader', NS) is None:
            etree.SubElement(row_properties, q('tblHeader'))
        table.append(row)
    return table


def main():
    assert prior.digest(CURRENT) == PREVIOUS_SHA
    assert not BACKUP.exists() and not CANDIDATE.exists()
    pair = json.loads(PAIR_PATH.read_text(encoding='utf-8'))
    assert pair['status'] == 'RESUMED_PAIR_ENDPOINTS_VERIFIED_PENDING_MANUSCRIPT_REVIEW'
    assert pair['result_generated'] and not pair['submission_ready']
    assert pair['comparisons']['terminal']['epoch_indices'] == {'BASE': 319, 'FULL': 319}
    assert all(value['joined_completed_epochs'] == 320 for value in pair['lineages'].values())
    method = json.loads(METHOD_PATH.read_text(encoding='utf-8'))
    assert all(item['identical_bytes'] for item in method['files'])

    primary = pair['comparisons']['terminal']
    best = pair['comparisons']['best']
    base = {key: value / 100 for key, value in primary['absolute_percent']['BASE'].items()}
    full = {key: value / 100 for key, value in primary['absolute_percent']['FULL'].items()}
    gain = primary['FULL_minus_BASE_percentage_points']
    parameters = pair['shared_conditions']['trainable_parameters']
    assert parameters == {'BASE': 3737611, 'FULL': 7540942}

    with ZipFile(CURRENT) as source:
        document = etree.fromstring(source.read(PART))
        comments = etree.fromstring(source.read(COMMENTS_PART))
    paragraphs = document.findall('w:body/w:p', NS)
    originals = [etree.tostring(paragraph) for paragraph in paragraphs]
    existing_tables = document.findall('w:body/w:tbl', NS)
    original_tables = [etree.tostring(table) for table in existing_tables]
    changes = []

    def replace(index, replacement):
        old = prior.text(paragraphs[index])
        prior.edit(paragraphs, old, old, replacement)
        changes.append({'paragraph_index_before_edit': index, 'old': old, 'new': replacement})

    replace(5,
        'Small objects in unmanned aerial vehicle (UAV) images provide limited visual evidence and occur amid clutter '
        'and scale variation. We present MPSMA-DEIM, an adaptation of DEIM-D-FINE that combines a Multi-branch '
        'Progressive Stem (MPS) with Multi-scale Context-aware Attention (MSCA). The stem merges detail, downsampling '
        'and high-frequency branches before producing stride-four features. The attention design uses pooling-based '
        'context at an intermediate stage and dilated context at the deepest stage, with energy-based gates derived '
        'from SimAM. A comparison of the complete n configuration and its original baseline uses seed 0, FP32, batch '
        'size 4 and a 320-epoch budget on all 548 VisDrone2019 validation images. Under the frozen local COCO-style '
        f'evaluation, the final-round EMA achieves average precision (AP) of {full["AP"]:.4f} and small-object AP (APs) '
        f'of {full["APs"]:.4f}, compared with {base["AP"]:.4f} and {base["APs"]:.4f} for the baseline. The absolute '
        f'differences are {gain["AP"]:.2f} and {gain["APs"]:.2f} percentage points. The comparison includes documented '
        'checkpoint continuations. Component-specific effects are being evaluated separately.')
    introduction = prior.text(paragraphs[10])
    introduction = introduction.replace(
        'We also use the sigmoid-weighted linear unit (SiLU), without claiming a new activation mechanism.',
        'We use the sigmoid-weighted linear unit (SiLU) in configurable backbone blocks.')
    introduction = introduction.replace(
        'We report traceable configuration-level results, distinguish the n and s protocols, and identify the '
        'comparisons that require matched controls before a causal performance claim can be made.',
        'We evaluate the complete n configuration under a common training and evaluation recipe, reporting '
        'fixed-budget and validation-selected endpoints separately. The archived n and s runs provide additional '
        'configuration-level observations.')
    replace(10, introduction)

    heading = paragraph_like(paragraphs[78], '4.6 Comparison of the n Configuration')
    protocol_text = (
        'BASE is the original n baseline and FULL is the complete n model. Both use the frozen 6,471-image training '
        'split and 548-image validation split, seed 0, FP32 and batch size 4 for 320 epochs, with 1,617 optimizer '
        'update attempts per retained epoch. A predefined set of 554 shared tensors has identical initial values. '
        'The optimizer, augmentation, loss, EMA and evaluator follow the common frozen recipe. The primary endpoint '
        'is the EMA evaluated at actual loop index 319 before any validation-dependent rollback. The secondary '
        'endpoint is each run\'s highest-AP row, with the earliest round selected on ties. APs and the other metrics '
        'use the same checkpoint as AP. Table 7 reports completed rounds using one-based numbering.')
    protocol = paragraph_like(paragraphs[81], protocol_text)
    caption = paragraph_like(paragraphs[76],
        'Table 7. Fixed-budget primary endpoints and validation-selected secondary endpoints for the n configuration. '
        'AP and recall values are fractions. Rounds are one-based.')
    caption_properties = caption.find('w:pPr', NS)
    if caption_properties.find('w:keepNext', NS) is None:
        etree.SubElement(caption_properties, q('keepNext'))
    rows = [['Endpoint', 'Model', 'Round', 'AP', 'AP50', 'AP75', 'APs', 'AR@100']]
    for role, comparison in [('Primary', primary), ('Secondary', best)]:
        for arm in ['BASE', 'FULL']:
            metrics = comparison['absolute_percent'][arm]
            rows.append([role, arm, str(comparison['epoch_indices'][arm] + 1)]
                        + [f'{metrics[name] / 100:.4f}' for name in ['AP', 'AP50', 'AP75', 'APs', 'AR100']])
    table = endpoint_table(existing_tables[1], rows)
    results_text = (
        f'At the prespecified primary endpoint, FULL reaches AP {full["AP"]:.4f} and APs {full["APs"]:.4f}, compared '
        f'with BASE values of {base["AP"]:.4f} and {base["APs"]:.4f}. The absolute differences are '
        f'{gain["AP"]:.2f} and {gain["APs"]:.2f} percentage points. The highest-AP secondary endpoints occur at '
        f'completed rounds {best["epoch_indices"]["BASE"] + 1} for BASE and {best["epoch_indices"]["FULL"] + 1} '
        f'for FULL. FULL has {parameters["FULL"]:,} trainable parameters, compared with {parameters["BASE"]:,} '
        'for BASE. The comparison measures the entire changed backbone configuration, including its additional '
        'capacity and aggregation and activation settings. Component-removal experiments are in progress.')
    results = paragraph_like(paragraphs[81], results_text)
    cursor = paragraphs[78]
    for element in [heading, protocol, caption, table, results]:
        cursor.addprevious(element)
    replace(78, '4.7 Evidence Limits')
    replace(79,
        'All metrics use the validation set. The primary row is fixed at 320 completed rounds, and the native '
        'training policy includes validation-dependent restoration. The secondary endpoints select the highest '
        'AP within one run. One seed provides no independent test estimate or uncertainty across training runs.')
    replace(81,
        'BASE and FULL initially resumed after 214 and one completed round, respectively. Their anchor checkpoints '
        'omitted main-process random states, which were reinitialized with seed 0. BASE later resumed after round '
        '308 with its recorded random states, and the interrupted partial round 309 was rerun. The FULL per-batch '
        'learning-rate log begins at round 2. The comparison therefore documents the observed continuations under '
        'a common recipe. The retained records, saved endpoints and process-exit receipts have been checked together.')
    replace(83,
        'MPSMA-DEIM combines a three-path shallow stem with stage-specific contextual attention in a DEIM-D-FINE '
        'detector. It uses residual stem weighting, parameter-free pooled-context gates and a learned '
        'dilated-context gate. At the prespecified 320-round endpoint, the complete n configuration reaches '
        f'AP {full["AP"]:.4f} and APs {full["APs"]:.4f}, exceeding the original baseline by {gain["AP"]:.2f} and '
        f'{gain["APs"]:.2f} percentage points under the frozen local evaluation recipe. These results describe '
        'one comparison with matched data, budget and shared-tensor initialization and documented checkpoint '
        'continuations. The archived n and s runs and the separate CPU checks supply additional configuration '
        'observations. Conditional removals of MPS and MSCA are in progress, and their individual effects remain open.')

    comment_updates = {
        '0': '方法图文所依据的原始骨干、改进骨干、MSCA 和 SimAM 四份源码，已与新 BASE/FULL 冻结协议逐字节核对一致，'
             '见 method_source_binding_20261108.json。历史训练时代码身份仍受原有证据限制。',
        '1': '历史 N0 使用 batch 16，其余历史配置使用 batch 4。新 BASE/FULL 均使用 batch 4、FP32、seed 0 和相同冻结'
             '训练评价方案，其完整主次端点已核验，见 Table 7。初始随机状态缺失及后续恢复限制见第 4.7 节。',
        '4': '新 BASE/FULL 使用冻结的数据文件、源码与本地 COCO-style 评价口径，终点预测、数组、权重和退出回执已绑定。'
             '历史数据文件身份仍按既有证据范围说明，本地口径与官方 ignore/maxDets 的差异保留在第 4.1 节。',
    }
    for comment in comments.findall('w:comment', NS):
        identifier = comment.get(q('id'))
        if identifier in comment_updates:
            nodes = comment.findall('.//w:t', NS)
            assert len(nodes) == 1
            nodes[0].text = comment_updates[identifier]
            comment.set(q('date'), datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))

    changed = [index for index, paragraph in enumerate(paragraphs)
               if etree.tostring(paragraph) != originals[index]]
    assert changed == [5, 10, 78, 79, 81, 83]
    assert [etree.tostring(table) for table in existing_tables] == original_tables
    counts = {name: len(document.findall('.//' + name, NS)) for name in
              ['w:tbl', 'm:oMath', 'wp:inline', 'w:commentRangeStart', 'w:commentRangeEnd', 'w:commentReference']}
    assert counts == {'w:tbl': 7, 'm:oMath': 6, 'wp:inline': 7,
                      'w:commentRangeStart': 5, 'w:commentRangeEnd': 5, 'w:commentReference': 5}
    visible = '\n'.join(prior.text(paragraph) for paragraph in document.findall('w:body/w:p', NS))
    assert 'Paired results are pending.' not in visible and 'The ongoing n-scale comparison' not in visible
    assert visible.count('Table 7.') == 1 and '\ufffd' not in visible
    replacements = {
        PART: etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True),
        COMMENTS_PART: etree.tostring(comments, encoding='UTF-8', xml_declaration=True, standalone=True),
    }
    shutil.copy2(CURRENT, BACKUP)
    assert prior.digest(BACKUP) == PREVIOUS_SHA
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, replacements.get(item.filename, source.read(item.filename)))
    with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
        assert after.testzip() is None and before.namelist() == after.namelist()
        assert {name for name in before.namelist() if before.read(name) != after.read(name)} == set(replacements)
    report = {
        'status': 'CANDIDATE_PENDING_VISUAL_QA', 'revision': 32, 'source_sha256': PREVIOUS_SHA,
        'candidate_sha256': prior.digest(CANDIDATE), 'structure': counts, 'paragraph_changes': changes,
        'new_table_rows': rows, 'comment_ids_updated': list(comment_updates),
        'evidence_sha256': {path.relative_to(PAPER).as_posix(): prior.digest(path)
                            for path in [PAIR_PATH, METHOD_PATH]},
        'submission_ready': False,
    }
    with (HERE / 'paired_revision_candidate.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'revision': 32,
                      'candidate_sha256': report['candidate_sha256'], 'structure': counts}))


if __name__ == '__main__':
    main()
