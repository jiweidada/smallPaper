"""Focus the manuscript on verified, comparable observations and preserve history."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import ast
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
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev32_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33.docx'
SOURCE_SHA = 'f8c5739e2394375160a7f606aec3aa47d632d1dab5376d90f14f148db3914c79'
REMOTE = PAPER / '远程公平对照_20261103'
BUNDLE = REMOTE / '模块消融'
PAIR_PATH = REMOTE / '断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'w': W, 'r': R,
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def text(element):
    return ''.join(element.xpath('.//w:t/text()', namespaces=NS))


def q(name):
    return '{' + W + '}' + name


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


def comment_anchor(paragraph, identifier):
    start = etree.Element(q('commentRangeStart'))
    start.set(q('id'), identifier)
    paragraph.insert(1 if paragraph.find('w:pPr', NS) is not None else 0, start)
    end = etree.SubElement(paragraph, q('commentRangeEnd'))
    end.set(q('id'), identifier)
    run = etree.SubElement(paragraph, q('r'))
    properties = etree.SubElement(run, q('rPr'))
    etree.SubElement(properties, q('rStyle')).set(q('val'), 'CommentReference')
    etree.SubElement(run, q('commentReference')).set(q('id'), identifier)


def replace_single_text(paragraph, value):
    nodes = paragraph.findall('.//w:t', NS)
    assert len(nodes) == 1
    old = nodes[0].text
    nodes[0].text = value
    return {'old': old, 'new': value}


def main():
    assert digest(MASTER) == SOURCE_SHA and not BACKUP.exists() and not CANDIDATE.exists()
    pair = read(PAIR_PATH)
    assert pair['lineages']['BASE']['resume_boundaries'] == [214, 308]
    assert pair['lineages']['BASE']['original_rows_after_breakpoint_excluded'] == 11
    assert pair['status'] == 'RESUMED_PAIR_ENDPOINTS_VERIFIED_PENDING_MANUSCRIPT_REVIEW'
    plot = read(HERE / 'learning_curve_provenance.json')
    assert plot['status'] == 'ALL_320_OBSERVED_ROUNDS_BOUND_TO_VERIFIED_PAIR'
    figure_path = HERE / 'paired_validation_curves.png'
    assert digest(figure_path) == plot['figure_sha256'][figure_path.name]
    full_config = read(REMOTE / 'remote_evidence/execution_configs/FULL.yml')
    base_config = read(REMOTE / 'remote_evidence/execution_configs/BASE.yml')
    assert full_config['optimizer']['betas'] == [0.9, 0.999]
    assert full_config['optimizer']['lr'] == 0.0008
    assert [p.get('lr', full_config['optimizer']['lr']) for p in full_config['optimizer']['params']] == [0.0004, 0.0004, 0.0008]
    assert full_config['warmup_iter'] == 2000 and full_config['lr_gamma'] == 1.0
    assert full_config['lr_warmup_scheduler']['warmup_duration'] == 500
    assert full_config['clip_max_norm'] == 0.1 and not full_config['use_amp']
    assert full_config['ema']['decay'] == 0.9999 and full_config['ema']['warmups'] == 1000
    assert full_config['train_dataloader']['dataset']['transforms']['policy']['epoch'] == [4, 156, 296]
    assert full_config['train_dataloader']['collate_fn']['mixup_epochs'] == [4, 156]
    source_binding = read(BUNDLE / 'method_source_binding_20261108.json')
    original_binding = next(item for item in source_binding['files']
                            if item['method_evidence_path'].endswith('/hgnetv2.py'))
    original_backbone_path = PAPER / original_binding['method_evidence_path']
    assert digest(original_backbone_path) == original_binding['sha256']
    original_backbone_source = original_backbone_path.read_text(encoding='utf-8')
    original_class = next(node for node in ast.parse(original_backbone_source).body
                          if isinstance(node, ast.ClassDef) and node.name == 'HGNetv2')
    constructor = next(node for node in original_class.body
                       if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in
                zip(constructor.args.args[-len(constructor.args.defaults):], constructor.args.defaults)}
    assert base_config['HGNetv2'].get('agg', defaults['agg']) == 'se'
    assert 'self.act = nn.ReLU()' in original_backbone_source
    parameters = pair['shared_conditions']['trainable_parameters']
    assert parameters == {'BASE': 3737611, 'FULL': 7540942}
    assert f'{parameters["FULL"] / parameters["BASE"]:.2f}' == '2.02'
    for arm, count in [('NO_MPS', 7535617), ('NO_MSCA', 3699662)]:
        preflight = read(BUNDLE / arm / 'inputs/preflight.json')
        assert preflight['initialization']['trainable_parameters'] == count
    with ZipFile(MASTER) as package:
        document = etree.fromstring(package.read('word/document.xml'))
        comments = etree.fromstring(package.read('word/comments.xml'))
        relationships = etree.fromstring(package.read('word/_rels/document.xml.rels'))
        content_types = etree.fromstring(package.read('[Content_Types].xml'))
    body = document.find('w:body', NS)
    paragraphs = document.findall('w:body/w:p', NS)
    tables = document.findall('.//w:tbl', NS)
    assert len(paragraphs) == 102 and len(tables) == 7
    assert text(paragraphs[48]) == '4. Experiments' and text(paragraphs[86]) == '5. Conclusion'
    original_math = [etree.tostring(n) for n in document.findall('.//m:oMath', NS)]
    original_core_figures = [etree.tostring(n) for n in document.findall('.//wp:inline', NS)[:5]]
    original_references = [text(p) for p in paragraphs[89:]]
    old_elements = list(body)[list(body).index(paragraphs[49]):list(body).index(paragraphs[86])]
    archive = {'source_sha256': SOURCE_SHA,
               'scope': 'Full revision-32 experimental context, including historical observations and the then-current controlled comparison. The six historical tables are excluded from revision-33 comparisons; no archived value is substituted into the new comparison.',
               'blocks': []}
    archive_text = ['Experimental context preserved from revision 32', 'Source SHA-256: ' + SOURCE_SHA, '']
    for element in old_elements:
        if element.tag == q('tbl'):
            cells = [[text(cell) for cell in row.findall('w:tc', NS)] for row in element.findall('w:tr', NS)]
            archive['blocks'].append({'type': 'table', 'rows': cells})
            archive_text.extend('\t'.join(row) for row in cells)
        else:
            value = text(element)
            archive['blocks'].append({'type': 'paragraph', 'text': value})
            archive_text.append(value)
        archive_text.append('')
    with (HERE / 'historical_experimental_context_rev32.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(archive, ensure_ascii=False, indent=2) + '\n')
    with (HERE / 'historical_experimental_context_rev32.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(archive_text))

    replacements = []
    replacements.append(replace_single_text(paragraphs[8],
        'Unmanned aerial vehicle (UAV) imagery poses a challenging object-detection setting because many objects '
        'occupy few pixels, overlap with neighbouring objects, and appear against complex backgrounds [1], [2]. '
        'Small-object performance depends on both the features retained during early downsampling and the context '
        'available for classification and localization. We therefore study shallow feature paths and context-dependent '
        'gates within a DEIM-D-FINE backbone.'))
    replacements.append(replace_single_text(paragraphs[10],
        'MPSMA-DEIM combines two backbone modifications. The Multi-branch Progressive Stem (MPS) aligns three '
        'shallow feature paths, applies residual spatial weighting, and fuses them before the final stem downsampling. '
        'Multi-scale Context-aware Attention (MSCA) uses stage-specific gates, including pooled-context and '
        'dilated-context variants. The constituent operations and SimAM-style weighting are inherited techniques. '
        'The complete configuration also uses full-width aggregation projections and the sigmoid-weighted linear '
        'unit (SiLU) in configurable backbone blocks. We compare this complete n configuration with the original '
        'baseline under a common frozen recipe, reporting fixed-round and validation-selected endpoints together '
        'with the increase in trainable parameters.'))
    replacements.append(replace_single_text(paragraphs[12],
        'Real-time detectors seek a useful balance between detection accuracy and computation. YOLOv10 uses '
        'consistent dual assignments to support inference without non-maximum suppression (NMS) [6]. DEIM [4] '
        'and D-FINE [5] instead develop the training and localization of detection transformers. MPSMA-DEIM '
        'retains the encoder and decoder design of its corresponding DEIM-D-FINE configuration and modifies '
        'the backbone. The present evaluation focuses on detection accuracy and trainable parameter count.'))
    replacements.append(replace_single_text(paragraphs[16],
        'MPSMA-DEIM uses an HGNetv2-B0 backbone, a hybrid feature encoder and a D-FINE decoder trained within '
        'the DEIM framework [4,5]. Figure 1 shows the implemented n and s configurations. Both use the same B0 '
        'stage widths and modified backbone. The evaluated n configuration supplies stride-16 and stride-32 '
        'features to an encoder of width 128. The implementation also supports an s configuration with an '
        'additional stride-8 input and width 256; it is outside the present controlled comparison. MPS and MSCA '
        'modify the backbone, while the detection losses and decoder design are inherited.'))
    replacements.append(replace_single_text(paragraphs[47],
        'The full configuration selects SiLU, f(x) = x sigmoid(x) [11], for configurable convolution blocks in '
        'the stem and backbone stages. The internal branch and channel-gate activations of MSCA-Dilated remain '
        'ReLU. The original baseline uses ReLU in its configurable backbone blocks. '
        'Both configurations retain the same encoder and decoder activation settings. SiLU is a configuration '
        'choice, rather than a new activation mechanism.'))

    template = paragraphs[79]
    heading_template = paragraphs[49]
    caption_template = paragraphs[80]
    blocks = []
    prose = []

    def heading(value):
        blocks.append(paragraph_like(heading_template, value))

    def paragraph(value, comment=None):
        element = paragraph_like(template, value)
        if comment is not None:
            comment_anchor(element, comment)
        blocks.append(element)
        prose.append(value)

    heading('4.1 Dataset and Evaluation')
    paragraph(
        'We use the VisDrone2019 detection training and validation splits [2], with 6,471 and 548 images, '
        'respectively, across ten categories. Inputs are resized to 640 × 640. The frozen converted annotations '
        'contain 343,205 training boxes and 38,759 validation boxes. All validation images are retained, including '
        'two byte-identical images with different annotation records. The same data files are used in every arm.')
    paragraph(
        'Evaluation uses the local converted COCO-style bounding-box protocol [12]: intersection-over-union '
        '(IoU) thresholds 0.50–0.95 in steps of 0.05, 101 recall thresholds, class-aware matching and '
        'maxDets=[1, 10, 100]. The postprocessor returns 300 predictions per image across categories. Average '
        'precision (AP) averages the ten IoU thresholds; AP50 and AP75 use 0.50 and 0.75. Small-object AP '
        '(APs) uses original-image box area up to 1,024 square pixels. AR@100 denotes average recall at '
        'maxDets=100.', '4')
    paragraph(
        'The converted annotations have iscrowd=0 and no ignore field. The 1,378 ignored-region entries '
        'and 32 entries labelled others in the original validation annotations are absent. The official '
        'VisDrone toolkit filters detections using ignored regions and uses an AP detection limit of 500 [13]. '
        'The local protocol therefore differs from the official challenge evaluation, and the reported values '
        'are not used to rank against official test-dev results.')

    heading('4.2 Training and Endpoint Protocol')
    paragraph(
        'BASE is the original n detector with its standard stem, two-convolution aggregation paths and ReLU '
        'backbone activations. FULL is the complete n configuration described in Section 3, including its '
        'aggregation projections and activation choices. Both use seed 0, FP32 with mixed precision and TF32 '
        'disabled, batch size 4, and one NVIDIA RTX 3090 GPU per run. The software versions are PyTorch '
        '2.5.1+cu124, CUDA 12.4 and faster-coco-eval 1.8.0. Pretrained-backbone loading and tuning are disabled. '
        'A predefined set of 554 shared encoder, decoder, stage-downsampling and HG-block tensors has identical '
        'initial values. The first 100 training input batches were checked for exact agreement.', '1')
    paragraph(
        'Each retained trajectory contains 320 completed rounds with 1,617 optimizer update attempts per round, '
        'or 517,440 attempts. We use one-based round numbers. AdamW uses betas (0.9, 0.999), base learning '
        'rates of 4 × 10⁻⁴ for the backbone and 8 × 10⁻⁴ otherwise, weight decay 10⁻⁴, and the frozen '
        'zero-decay normalization and encoder/decoder bias groups. The gradient norm is clipped at 0.1. '
        'Learning rates are assigned after each update: the first update uses the base rate divided by 500, '
        'followed by the selected quadratic warm-up over 2,000 steps. The rates remain at their base values '
        'after warm-up because lr_gamma=1.0. The exponential moving average (EMA) uses decay 0.9999 and '
        'a ramp time constant of 1,000 updates.')
    paragraph(
        'Mosaic and batch MixUp each have selection probability 0.5 in rounds 5–156. Photometric distortion, '
        'zoom-out and IoU cropping are eligible in rounds 5–296; Mosaic-selected samples skip zoom-out and '
        'IoU cropping. From round 297, these strong augmentations are disabled, while horizontal flipping '
        'and resizing remain. Batch multi-scale resizing is disabled. The native late-stage policy restores '
        'the best-AP checkpoint from the first 296 rounds at stage entry and after a later round that does '
        'not set a new global best. This policy is identical in all arms.')
    paragraph(
        'The primary endpoint is the EMA evaluated at actual round 320 before any subsequent restore. '
        'The secondary endpoint is the highest-AP round within each run, choosing the earliest round on ties. '
        'APs and all other metrics follow that same AP-selected checkpoint. Stored endpoint weights, prediction '
        'files, evaluator arrays and metric rows are bound by hashes and were checked together.')

    heading('4.3 Complete Configuration Results')
    blocks.append(paragraph_like(caption_template,
        'Table 1. Fixed-round primary endpoints and validation-selected secondary endpoints. AP and recall '
        'values are fractions; rounds are one-based.'))
    blocks.append(deepcopy(tables[6]))
    paragraph(
        'At the primary endpoint, FULL reaches AP 0.2255 and APs 0.1539, compared with BASE values of '
        '0.1902 and 0.1215. The differences are 3.53 and 3.23 percentage points, respectively. The secondary '
        'best-AP rounds are 303 for BASE and 305 for FULL. These results describe the complete changed '
        'backbone configuration under the common local recipe.')
    paragraph(
        'FULL has 7,540,942 trainable parameters, compared with 3,737,611 for BASE, approximately 2.02 times '
        'as many. The observed accuracy gain therefore comes with a substantial capacity increase. The '
        'comparison does not separate this capacity effect from the stem, contextual gates, aggregation '
        'projections or activation changes. Trainable parameter counts do not establish deployment latency '
        'or onboard throughput.', '3')

    figure_paragraph = deepcopy(paragraphs[52])
    drawings = figure_paragraph.findall('.//wp:inline', NS)
    assert len(drawings) == 2
    first_drawing = deepcopy(figure_paragraph.find('.//w:drawing', NS))
    for child in list(figure_paragraph):
        if child.tag != q('pPr'):
            figure_paragraph.remove(child)
    run = etree.SubElement(figure_paragraph, q('r'))
    run.append(first_drawing)
    new_rid = 'rId' + str(max(int(r.get('Id')[3:]) for r in relationships if re.fullmatch(r'rId\d+', r.get('Id', ''))) + 1)
    relation = etree.SubElement(relationships, '{' + PR + '}Relationship')
    relation.set('Id', new_rid)
    relation.set('Type', R + '/image')
    relation.set('Target', 'media/paired_validation_curves.png')
    figure_paragraph.find('.//a:blip', NS).set('{' + R + '}embed', new_rid)
    inline = figure_paragraph.find('.//wp:inline', NS)
    width, height = 6217920, 2703443
    inline.find('wp:extent', NS).set('cx', str(width))
    inline.find('wp:extent', NS).set('cy', str(height))
    for extent in inline.findall('.//a:xfrm/a:ext', NS):
        extent.set('cx', str(width)); extent.set('cy', str(height))
    properties = inline.find('wp:docPr', NS)
    properties.set('name', 'BASE and FULL validation curves')
    properties.set('descr', 'All 320 observed validation AP and APs values; fixed-round and best-AP markers; late restoration stage shaded.')
    blocks.append(figure_paragraph)
    blocks.append(paragraph_like(paragraphs[53],
        'Fig. 6. All 320 retained validation observations, without smoothing. Circles mark round 320; stars '
        'mark the best-AP round, including the APs value from that same checkpoint. Shading marks the '
        'late restoration stage, rounds 297–320. BASE continued after rounds 214 and 308; FULL continued '
        'after round 1.'))
    paragraph(
        'Figure 6 shows sharp validation drops in the observed BASE trajectory, while the observed FULL '
        'trajectory is smoother. These curves document this pair of runs. Different continuation histories '
        'and a single seed prevent an architectural claim about training stability or a prediction of '
        'run-to-run variability.')

    heading('4.4 Conditional Module Removal')
    paragraph(
        'Two additional runs examine removals from FULL under the same frozen data, optimizer, augmentation, '
        'evaluation and 320-round schedule. NO_MPS replaces only MPS with the original BASE StemBlock '
        'using SiLU and retains the remaining FULL structure. NO_MSCA replaces the two MSCA-Lite gates and '
        'one MSCA-Dilated gate with identities, retaining their aggregation projections, ESE, SimAM, MPS '
        'and SiLU. All retained tensors start from their corresponding FULL initial values; the replacement '
        'stem starts from the original BASE stem tensors. The trainable parameter counts are 7,535,617 '
        'and 3,699,662, respectively.')
    paragraph(
        'Both removal runs are ongoing, so no component-specific outcome is reported here. These conditional '
        'comparisons will concern each removal while the remaining FULL settings are present. BASE is not '
        'a joint-removal arm because its aggregation paths and activations also differ. This design cannot '
        'establish an interaction or synergy between MPS and MSCA.')

    heading('4.5 Limitations')
    paragraph(
        'The comparison uses one seed and one validation split. The same validation set guides the native '
        'restoration policy and the secondary checkpoint selection. The fixed final round limits one form '
        'of endpoint selection but does not provide an independent test estimate or uncertainty across '
        'training runs. Capacity is not matched, the module-removal runs are incomplete, and no current '
        'n-model latency or onboard benchmark is available.')
    paragraph(
        'The retained trajectories include interrupted training. BASE first continued from round 214, '
        'excluding 11 later rows from the interrupted original branch, and FULL continued from round 1. '
        'Those first anchors lacked main-process random states, which were reset with seed 0. BASE later '
        'continued from round 308 with saved random states and reran an interrupted partial round 309. '
        'The FULL per-batch learning-rate record starts at round 2. The reported results therefore describe '
        'these documented continuations, rather than uninterrupted replications.', '2')

    stop = paragraphs[86]
    for element in old_elements:
        body.remove(element)
    for element in blocks:
        stop.addprevious(element)
    replacements.append(replace_single_text(paragraphs[87],
        'MPSMA-DEIM combines a multi-branch shallow stem with stage-specific contextual gating in a '
        'DEIM-D-FINE backbone. In the seed-0 n comparison, the fixed round-320 EMA endpoint achieves '
        'AP 0.2255 and APs 0.1539, exceeding BASE by 3.53 and 3.23 percentage points under the common '
        'local recipe. FULL also uses approximately 2.02 times as many trainable parameters. These '
        'observations apply to the documented continuation histories and local COCO-style evaluation. '
        'The two conditional removal runs remain in progress; their results are needed before drawing '
        'component-specific conclusions.'))
    comment_text = {
        '0': '本轮正文集中于已核验的 n 对照；图 1 中 s 是实现支持的配置，不再用于本轮受控结果。四份方法核心源码与冻结实验逐字节一致的绑定继续适用。',
        '1': '历史 batch 16/4 及聚合路径混杂的六张表已移出正文并完整归档于本轮 historical_experimental_context_rev32.* 和第三十二轮备份。新正文明确 BASE/FULL 的全部结构差异、共同配方、共享初始化及恢复限制。',
        '2': '已补充 BASE 第 214 轮锚点之后原中断分支的 11 行不纳入保留轨迹；另披露第 308 轮恢复与部分第 309 轮重跑。320 轮是保留轨迹长度，不表示总计算消耗完全相等，也不证明连续从头复现。',
        '3': '正文仅报告新 n 组训练参数 3,737,611/7,540,942 及其代价。旧 s 速度与复杂度没有移用于当前权重，旧记录及限制保存在归档中。条件移除结果尚待完成。',
        '4': '冻结本地数据及评价口径用于新四组。保留官方 ignore/maxDets 差异及验证集同图不同标注事实；548 张全部保留。当前曲线和主次端点均绑定已核验日志，不是重新运行模型得到的结果。',
    }
    for comment in comments.findall('w:comment', NS):
        identifier = comment.get(q('id'))
        nodes = comment.findall('.//w:t', NS)
        assert len(nodes) == 1
        nodes[0].text = comment_text[identifier]
        comment.set(q('date'), datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))

    assert [etree.tostring(n) for n in document.findall('.//m:oMath', NS)] == original_math
    assert [etree.tostring(n) for n in document.findall('.//wp:inline', NS)[:5]] == original_core_figures
    assert len(document.findall('.//wp:inline', NS)) == 6
    assert len(document.findall('.//w:tbl', NS)) == 1
    assert [text(p) for p in document.findall('w:body/w:p', NS) if re.match(r'^\[\d+\]', text(p))] == original_references
    for tag in ['w:commentRangeStart', 'w:commentRangeEnd', 'w:commentReference']:
        assert sorted(n.get(q('id')) for n in document.findall('.//' + tag, NS)) == ['0', '1', '2', '3', '4']
    visible = '\n'.join(text(p) for p in document.findall('w:body/w:p', NS))
    assert 'Table 7' not in visible and 'Section 4.7' not in visible and '\ufffd' not in visible
    assert '44.30' not in visible and '0.2876' not in visible and 'Table 1.' in visible
    assert any(n.get('Extension') == 'png' for n in content_types)
    changed_parts = {
        'word/document.xml': etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True),
        'word/comments.xml': etree.tostring(comments, encoding='UTF-8', xml_declaration=True, standalone=True),
        'word/_rels/document.xml.rels': etree.tostring(relationships, encoding='UTF-8', xml_declaration=True, standalone=True),
    }
    shutil.copy2(MASTER, BACKUP)
    assert digest(BACKUP) == SOURCE_SHA
    with ZipFile(MASTER) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, changed_parts.get(item.filename, source.read(item.filename)))
        target.writestr('word/media/paired_validation_curves.png', figure_path.read_bytes())
    with ZipFile(CANDIDATE) as result:
        assert result.testzip() is None
        assert sha256(result.read('word/media/paired_validation_curves.png')).hexdigest() == digest(figure_path)
    report = {'status': 'CANDIDATE_PENDING_VISUAL_AND_CONTENT_QA', 'revision': 33,
              'source_sha256': SOURCE_SHA, 'candidate_sha256': digest(CANDIDATE),
              'preserved_core_equations': 6, 'preserved_core_diagrams': 5,
              'new_observed_learning_curve': plot, 'table_count': 1, 'comment_anchors': 5,
              'historical_tables_archived': 6, 'paragraph_replacements': replacements,
              'new_experiment_paragraphs': prose,
              'pair_report_sha256': digest(PAIR_PATH),
              'historical_context_sha256': digest(HERE / 'historical_experimental_context_rev32.json'),
              'submission_ready': False}
    with (HERE / 'reorganization_candidate.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (HERE / 'candidate_text.txt').open('x', encoding='utf-8') as stream:
        stream.write(visible + '\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'tables': 1, 'inline_figures': 6, 'preserved_equations': 6}))


if __name__ == '__main__':
    main()
