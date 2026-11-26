"""Add the audited completed MPS replacement without claiming unfinished outcomes."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
from zipfile import ZipFile

from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34.docx'
SOURCE_SHA = 'b3aae17cf12667f34797179f399dcfd608edfab724633f023921c9703b0d4cee'
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
COMPARISON_PATH = BUNDLE / 'NO_MPS_FULL_endpoint_comparison_20261108.json'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def text(element):
    return ''.join(element.xpath('.//w:t/text()', namespaces=NS))


def q(value):
    return '{' + W + '}' + value


def paragraph_like(template, value):
    result = etree.Element(q('p'))
    properties = template.find('w:pPr', NS)
    if properties is not None:
        result.append(deepcopy(properties))
    run = etree.SubElement(result, q('r'))
    run_properties = template.find('w:r/w:rPr', NS)
    if run_properties is not None:
        run.append(deepcopy(run_properties))
    etree.SubElement(run, q('t')).text = value
    return result


def main():
    assert digest(CURRENT) == SOURCE_SHA and not BACKUP.exists() and not CANDIDATE.exists()
    comparison = read(COMPARISON_PATH)
    assert comparison['status'] == 'NO_MPS_ENDPOINT_COMPARISON_VERIFIED_NO_MSCA_PENDING'
    assert comparison['NO_MPS_best_and_primary_coincide'] is True
    primary = comparison['comparisons']['primary_fixed_round320']
    assert primary['rounds'] == {'FULL': 320, 'NO_MPS': 320}
    gain = primary['FULL_minus_NO_MPS_percentage_points']
    values = primary['fractions']
    parameters = comparison['trainable_parameters']
    full_config_path = PAPER / '远程公平对照_20261103/remote_evidence/execution_configs/FULL.yml'
    assert digest(full_config_path) == read(BUNDLE / 'module_shared_conditions_verified_20261108.json')['full_config_sha256']
    full_config = read(full_config_path)
    assert full_config['HybridEncoder']['num_encoder_layers'] == 1
    assert full_config['HybridEncoder']['nhead'] == 8
    assert full_config['DFINETransformer']['num_layers'] == 3
    assert full_config['DFINETransformer']['num_queries'] == 300
    assert full_config['DFINETransformer']['num_points'] == [6, 6]
    assert full_config['train_dataloader']['drop_last'] is True
    with ZipFile(CURRENT) as package:
        document = etree.fromstring(package.read('word/document.xml'))
        comments = etree.fromstring(package.read('word/comments.xml'))
    paragraphs = document.findall('w:body/w:p', NS)
    tables = document.findall('.//w:tbl', NS)
    assert len(tables) == 1
    original_table = etree.tostring(tables[0])
    original_math = [etree.tostring(node) for node in document.findall('.//m:oMath', NS)]
    original_figures = [etree.tostring(node) for node in document.findall('.//wp:inline', NS)]
    replacements = []

    def change(prefix, old, new):
        candidates = [paragraph for paragraph in paragraphs if text(paragraph).startswith(prefix)]
        assert len(candidates) == 1, prefix
        nodes = [node for node in candidates[0].findall('.//w:t', NS) if old in (node.text or '')]
        assert len(nodes) == 1 and text(candidates[0]).count(old) == 1
        nodes[0].text = nodes[0].text.replace(old, new)
        replacements.append({'old': old, 'new': new})
        return candidates[0]

    change('Small objects in unmanned aerial vehicle',
           'Component-specific effects are being evaluated separately.',
           f'Replacing MPS with the standard SiLU stem gives AP {values["NO_MPS"]["AP"]:.4f} and APs '
           f'{values["NO_MPS"]["APs"]:.4f}. The MSCA-removal run is ongoing.')
    change('MPSMA-DEIM uses an HGNetv2-B0 backbone',
           'The evaluated n configuration supplies stride-16 and stride-32 features to an encoder of width 128.',
           'The evaluated n configuration supplies stride-16 and stride-32 features to an encoder of width 128. '
           'The encoder has one eight-head layer; the three-layer decoder uses 300 regular queries and six '
           'sampling points per feature level.')
    change('Evaluation follows a local bounding-box protocol',
           'Average recall (AR) is reported at maxDets=100 as AR@100.',
           'Average recall (AR) is reported at maxDets=100 as AR@100. Its large-object counterpart, ARl, uses '
           'original-image area of at least 9,216 square pixels.')
    change('Each retained trajectory contains',
           'or 517,440 attempts. We use one-based round numbers.',
           'or 517,440 attempts. The training loader drops its final incomplete batch. We use one-based round numbers.')
    change('Fig. 6. All 320 retained validation observations',
           'Fig. 6. All 320 retained validation observations, without smoothing.',
           'Fig. 6. All 320 retained validation observations for BASE and FULL, without smoothing.')
    ongoing = next(paragraph for paragraph in paragraphs if text(paragraph).startswith('Both removal runs are ongoing'))
    template = next(paragraph for paragraph in paragraphs if text(paragraph).startswith('At the primary endpoint, FULL reaches'))
    caption_template = next(paragraph for paragraph in paragraphs if text(paragraph).startswith('Table 1.'))
    caption = paragraph_like(caption_template,
        'Table 2. Conditional MPS replacement at the fixed round-320 endpoint. Metrics are fractions; '
        'trainable parameters are in millions. All other FULL settings are retained.')
    table = deepcopy(tables[0])
    row_templates = table.findall('w:tr', NS)
    header_template, data_template = deepcopy(row_templates[0]), deepcopy(row_templates[1])
    for row in row_templates:
        table.remove(row)
    widths = [1350, 1150, 1150, 1400, 1150, 1950]
    properties = table.find('w:tblPr', NS)
    properties.find('w:tblW', NS).set(q('w'), str(sum(widths)))
    grid = table.find('w:tblGrid', NS)
    grid.clear()
    for width in widths:
        etree.SubElement(grid, q('gridCol')).set(q('w'), str(width))
    rows = [['Model', 'AP', 'APs', 'AR@100', 'ARl', 'Train params/M']]
    for arm in ['FULL', 'NO_MPS']:
        rows.append([arm] + [f'{values[arm][name]:.4f}' for name in ['AP', 'APs', 'AR100', 'ARl']]
                    + [f'{parameters[arm] / 1e6:.4f}'])
    for index, values_row in enumerate(rows):
        row = deepcopy(header_template if index == 0 else data_template)
        for extra in row.findall('w:tc', NS)[len(widths):]:
            row.remove(extra)
        for cell, value, width in zip(row.findall('w:tc', NS), values_row, widths):
            cell.find('w:tcPr/w:tcW', NS).set(q('w'), str(width))
            paragraph = paragraph_like(cell.find('w:p', NS), value)
            for child in list(cell):
                if child.tag != q('tcPr'):
                    cell.remove(child)
            cell.append(paragraph)
        table.append(row)
    results = paragraph_like(template,
        f'NO_MPS completes all 320 rounds with AP {values["NO_MPS"]["AP"]:.4f} and APs '
        f'{values["NO_MPS"]["APs"]:.4f}. Relative to this replacement, FULL is higher by '
        f'{gain["AP"]:.2f} percentage points in AP and {gain["APs"]:.2f} in APs at the prespecified primary '
        'endpoint (Table 2). NO_MPS also reaches its highest AP at round 320, so its primary and secondary '
        'checkpoints coincide.')
    tradeoff = paragraph_like(template,
        f'The stem replacement removes {comparison["FULL_minus_NO_MPS_trainable_parameters"]:,} trainable '
        f'parameters. FULL has large-object recall {values["FULL"]["ARl"]:.4f}, compared with '
        f'{values["NO_MPS"]["ARl"]:.4f} for NO_MPS, a difference of {gain["ARl"]:.2f} percentage points. '
        'These recall values are descriptive observations from the same two endpoints.')
    caveat = paragraph_like(template,
        'This comparison concerns the stem while the remaining FULL settings are present. NO_MPS starts '
        'from round zero, whereas FULL continued after round 1 with reinitialized main-process random states. '
        'The results describe these realized trajectories under a common recipe and do not estimate seed '
        'variability or establish a mechanism independent of capacity.')
    pending = paragraph_like(template,
        'The NO_MSCA run is ongoing, so its outcome is not reported. BASE is not a joint-removal arm because '
        'its aggregation paths and activations also differ. The design cannot establish an interaction or '
        'synergy between MPS and MSCA.')
    for element in [caption, table, results, tradeoff, caveat, pending]:
        ongoing.addprevious(element)
    ongoing.getparent().remove(ongoing)
    change('The comparison uses one seed and one validation split.',
           'Capacity is not matched, the module-removal runs are incomplete, and no current n-model latency '
           'or onboard benchmark is available.',
           'Capacity is not matched, the MSCA-removal run is incomplete, and no current n-model latency '
           'or onboard benchmark is available.')
    change('MPSMA-DEIM combines a multi-branch shallow stem',
           'The two conditional removal runs remain in progress; their results are needed before drawing '
           'component-specific conclusions.',
           f'Relative to FULL, the completed MPS replacement is lower by {gain["AP"]:.2f} percentage points in AP and '
           f'{gain["APs"]:.2f} in APs, while its large-object recall is higher. These are conditional '
           'observations from one seed. The MSCA-removal run remains in progress.')
    comment_updates = {
        '1': '新 BASE/FULL 与 NO_MPS 使用同一冻结配方。NO_MPS 已完成 320 轮并通过全轨迹和主次端点核验；共享代码、数据与保留张量初值已交叉核对。FULL 第 1 轮后恢复、NO_MPS 从零开始的差异在正文保留。',
        '3': 'NO_MPS 固定第 320 轮 AP/APs 为 0.2098591466311687/0.13620306232013496；FULL 分别高 1.565409/1.765416 个百分点，但 ARl 低 1.686903 个百分点。替换减少 5,325 个训练参数。只能说明本次条件观察，MSCA 组仍待完成。',
    }
    for comment in comments.findall('w:comment', NS):
        identifier = comment.get(q('id'))
        if identifier in comment_updates:
            nodes = comment.findall('.//w:t', NS)
            assert len(nodes) == 1
            nodes[0].text = comment_updates[identifier]
            comment.set(q('date'), datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    assert etree.tostring(tables[0]) == original_table
    assert [etree.tostring(node) for node in document.findall('.//m:oMath', NS)] == original_math
    assert [etree.tostring(node) for node in document.findall('.//wp:inline', NS)] == original_figures
    assert len(document.findall('.//w:tbl', NS)) == 2
    for name in ['commentRangeStart', 'commentRangeEnd', 'commentReference']:
        assert sorted(node.get(q('id')) for node in document.findall('.//w:' + name, NS)) == ['0', '1', '2', '3', '4']
    visible = '\n'.join(text(paragraph) for paragraph in document.findall('w:body/w:p', NS))
    assert 'Both removal runs are ongoing' not in visible and 'The two conditional removal runs remain in progress' not in visible
    assert 'The NO_MSCA run is ongoing' in visible and 'Table 2.' in visible and '\ufffd' not in visible
    shutil.copy2(CURRENT, BACKUP)
    assert digest(BACKUP) == SOURCE_SHA
    replacements_xml = {
        'word/document.xml': etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True),
        'word/comments.xml': etree.tostring(comments, encoding='UTF-8', xml_declaration=True, standalone=True),
    }
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, replacements_xml.get(item.filename, source.read(item.filename)))
    with ZipFile(CANDIDATE) as result:
        assert result.testzip() is None
    report = {'status': 'MPS_ENDPOINT_CANDIDATE_PENDING_QA', 'revision': 34,
              'source_sha256': SOURCE_SHA, 'candidate_sha256': digest(CANDIDATE),
              'comparison_sha256': digest(COMPARISON_PATH), 'new_table_rows': rows,
              'primary_differences_pp': gain, 'paragraph_replacements': replacements,
              'preserved_table_count': 1, 'preserved_equations': 6, 'preserved_figures': 6,
              'updated_comment_ids': list(comment_updates), 'submission_ready': False}
    with (HERE / 'mps_endpoint_candidate.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (HERE / 'candidate_text.txt').open('x', encoding='utf-8') as stream:
        stream.write(visible + '\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'table_rows': rows}))


if __name__ == '__main__':
    main()
