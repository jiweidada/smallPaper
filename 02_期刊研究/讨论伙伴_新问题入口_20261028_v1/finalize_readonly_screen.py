"""Archive source identities and validate delivered records. No model imports."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LIB = PROJECT / '文献_相关原始论文_20261004'
OLD = PROJECT / 'data/00_实验计划与证据索引'
RAW = OLD / '原始遮挡忽略与像素对照独立复核_20261006_v1'
ROOT_RUN = PROJECT / '新问题入口重寻_20261028_v1'
FOUR = PROJECT / 'raw_EMA逐GT对照_20261026_v1/跨状态逐GT响应4x4_20261026_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def js(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def local(path, scope, **extra):
    assert path.is_file(), str(path)
    item = {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': sha(path), 'actual_read_scope': scope}
    item.update(extra)
    return item


papers = [
    ('20261028_新问题_Motion_Blur_CVPR2021_Publisher.pdf',
     'https://openaccess.thecvf.com/content/CVPR2021/papers/Sayed_Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR_2021_paper.pdf',
     'CVPR2021 Publisher open-access version, 11 PDF pages',
     'PDF4 sections3.2/3.3; PDF5 relevant3.4/4.2/4.4 material; PDF6 Fig4, box-support paragraph and section5 opening; PDF8 Table1 caption and first discussion paragraph. PDF6 visually inspected. Not full-paper reading.'),
    ('20261028_新问题_Position_Padding_arXiv2101.12322v1.pdf',
     'https://arxiv.org/pdf/2101.12322v1',
     'arXiv2101.12322v1, 2021-01-28, 19 PDF pages',
     'Abstract/introduction opening via primary web PDF; local PDF6 section4.1 and PDF10 sections6.1/6.2 read, PDF10 visually inspected.'),
    ('20261028_新问题_Mind_the_Pad_arXiv2010.02178v1.pdf',
     'https://arxiv.org/pdf/2010.02178v1',
     'arXiv2010.02178v1, 2020-10-05, under-review draft, 15 pages; not the accepted OpenReview version',
     'Primary web PDF1-5 problem/condition passages; local PDF3-5 selected text, actual focus PDF3/4 examples and PDF5 section5/Eq1/2. PDF5 visually inspected. No claim of full paper or accepted-version parity.'),
    ('20261028_新问题_Zone_Evaluation_arXiv2310.13215v2.pdf',
     'https://arxiv.org/pdf/2310.13215v2',
     'arXiv2310.13215v2, 2024-06-01, 16 PDF pages; original archived by peer',
     'Primary web abstract and PDF5 section4.1 result paragraphs and section4.2 shift construction. Section4.3 only title/opening context, not full mechanism audit.'),
    ('早期分类复核_20261006_Stable_DINO_ICCV2023.pdf',
     'https://arxiv.org/pdf/2304.04742v1',
     'Actual internal header arXiv2304.04742v1 2023-04-10, 15 PDF pages; filename is not version proof',
     'PDF1/2 abstract/introduction/Fig2 read; PDF1-3 extracted for lookup. Extraction is not full reading. No new matching computation.'),
]
sources = []
for filename, url, version, scope in papers:
    path = LIB / filename
    sources.append(local(path, scope, source_type='primary_paper', url=url,
                         version=version, pdf_pages=len(PdfReader(path).pages)))

sources += [
    {'source_type': 'primary_paper_web_only',
     'title': 'Human-Aware Motion Deblurring',
     'url': 'https://arxiv.org/pdf/2001.06816v1',
     'official_publication_url': 'https://openaccess.thecvf.com/content_ICCV_2019/html/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.html',
     'version': 'arXiv2001.06816v1, 2020-01-19, 10 PDF pages; official venue ICCV2019',
     'actual_read_scope': 'Primary arXiv PDF1-3 introduction/Fig1/related data context returned by browser; official abstract also read. No full methods or experiments review.',
     'sha256': None, 'local_copy': None,
     'claim_limit': 'Only supports the distinct heterogeneous FG/BG blur condition; does not establish a detection effect in this project.'},
    {'source_type': 'official_dataset_format_web',
     'url': 'https://aiskyeye.com/evaluate/results-format_2021/',
     'actual_read_scope': 'Object-detection eight-field format, truncation and occlusion definitions; checked against actual selected raw TXT rows.',
     'sha256': None},
]

history = [
    (PROJECT / 'code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/backbone/hgnetv2.py',
     'ConvBNAct padding code92-119; stem setup179-228 and forward230-258. Frozen local source copy, not a freshly loaded checkpoint.'),
    (PROJECT / '持续寻找研究主线_20261020_v2/19_边界分支冻结核查事前协议.txt',
     'Read protocol; identifies box-edge strip features rather than image-border cohort.'),
    (PROJECT / '持续寻找研究主线_20261020_v2/24_边界差分冻结结果完整回交.txt',
     'Read result scope and beginning; no re-evaluation.'),
    (PROJECT / '持续寻找研究主线_20261020_v2/222_双BASE24既有开发缓存修复_事前协议.txt',
     'Read GO source/constraints, own Hungarian parity statement and native_cost_face specification.'),
    (PROJECT / '持续寻找研究主线_20261020_v2/349_候选1_保持宽高比_零训练预检_Codex执行计划.txt',
     'Read frozen input-shape protocol to exclude synonym duplication; no execution.'),
    (OLD / 'CPU局部细节简测_20261009_v1/10_简测结果与边界.txt',
     'Read existing A/B/C results and interpretation limits. Did not compute new crop results.'),
    (RAW / 'pixel_gallery.py',
     'Selected rendering and sampling lines26-64; verifies 64pixel raw crop and nearest-neighbor5x display.'),
    (RAW / '像素对照图_v3/抽样与匹配规则.json',
     'Read fixed case/control sampling and caveats; not used for outcome-dependent image selection.'),
    (RAW / '像素对照图_v3/样本追溯与分组密钥.json',
     'After pre-unblinding observations, selected only P09/P11 source/GT/raw-attribute metadata; did not use group/prediction values to select witnesses.'),
    (RAW / '逐GT原始属性与轨迹对齐.csv',
     'Header and first two contract examples read; verification scans IDs to retrieve only the two selected witnesses and compare raw identity/attributes. No prediction fields used or new population summary.'),
    (ROOT_RUN / '01_LQE作用检验_事前协议.json',
     'Read arm definitions and comparison scope; no evaluator execution.'),
    (ROOT_RUN / 'LQE_effect_attempt02/results.json',
     'Read COMPLETED status and prespecified harmful-correction flags; use peer evaluation, no independent metric re-evaluation.'),
    (ROOT_RUN / 'boundary_condition_attempt01/results.json',
     'Read GT/join/groups/standardized output only. Observational results generated by peer; not our computation.'),
    (ROOT_RUN / '05_边界平移_前向前协议.json',
     'Read for discussion; peer subsequently explicitly did not execute. No forward authorization inferred.'),
    (FOUR / 'protocol.json', 'Read hash identity only for original4x4 task closure.'),
    (FOUR / 'attempt01/verification.json', 'Read existing verification status/schema for original4x4 closure.'),
]
sources.extend(local(path, scope, source_type='project_source_or_existing_record')
               for path, scope in history)
for page in ['盲态对照_01.png', '盲态对照_02.png']:
    sources.append(local(RAW / '像素对照图_v3' / page,
                         'Visually inspected at original resolution; fixed P01-P16 only.',
                         source_type='existing_pixel_panel'))

metadata = js(HERE / 'selected_raw_image_metadata.json')
selected_ids = {x['annotation_id'] for x in metadata}
with (RAW / '逐GT原始属性与轨迹对齐.csv').open(encoding='utf-8-sig', newline='') as f:
    selected_rows = {int(row['annotation_id']): row for row in csv.DictReader(f)
                     if int(row['annotation_id']) in selected_ids}
identity_checks = []
for item in metadata:
    image = Path(item['source_image'])
    ann = image.parent.parent / 'annotations' / (image.stem + '.txt')
    raw_line = ann.read_text(encoding='utf-8-sig').splitlines()[item['raw_row_1based'] - 1]
    values = [int(v) for v in raw_line.strip().strip(',').split(',')]
    row = selected_rows[item['annotation_id']]
    assert int(row['image_id']) == item['image_id']
    assert row['filename'] == item['filename']
    assert int(row['raw_row_1based']) == item['raw_row_1based']
    assert values[6] == item['truncation'] == int(row['truncation'])
    assert values[7] == item['occlusion'] == int(row['occlusion'])
    sources.append(local(image, 'Viewed entire original image after fixed-panel observation; no prediction overlay.',
                         source_type='original_val_image_witness', code=item['code']))
    sources.append(local(ann, 'Only exact mapped raw row inspected; no new outcome-derived sampling.',
                         source_type='original_val_annotation_witness', code=item['code'],
                         row_1based=item['raw_row_1based']))
    identity_checks.append({'code': item['code'], 'annotation_id': item['annotation_id'],
                            'image_id': item['image_id'], 'raw_row_1based': item['raw_row_1based'],
                            'raw_line': raw_line, 'identity_and_flags_equal': True})

assert sha(FOUR / 'protocol.json') == 'a1ea55445363e9db94d4902b4afd434c07d82c568f8099f86150ded2e766a275'
four_verify = js(FOUR / 'attempt01/verification.json')
assert four_verify['status'] == 'PASS'
assert four_verify['GT_rows'] == 77518
assert four_verify['GT0_present_both_seeds'] is True
assert four_verify['protocol_unchanged'] is True
assert not four_verify['changed_inputs']

public_rows = []
for name in ['06_另一对话本轮公开消息_过滤记录.json', '07_另一对话新增公开消息_过滤记录.json']:
    public_rows.extend(js(HERE / name)['records'])
assert len({x['item_id'] for x in public_rows}) == len(public_rows)
assert all(x['kind'] in {'source_thread_public_user_message', 'source_thread_public_agent_message',
                         'public_interthread_message_sent'} for x in public_rows)
save('source_ledger.json', {'date': '2026-10-28 America/Los_Angeles',
                          'scope': 'Read-only discussion partner; actual reading ranges, not full-paper claims.',
                          'sources': sources,
                          'bounded_history_search': {'directories': ['持续寻找研究主线_20261020_v2', 'raw_EMA逐GT对照_20261026_v1', '讨论伙伴_文献条件新入口重筛_20261027_v2'],
                                                     'pattern_family': 'HIDE / Human-Aware Motion Deblurring / heterogeneous FG-BG blur',
                                                     'exact_matches': 0, 'limit': 'Not a claim that all project history is new or absent.'}})
save('verification.json', {'status': 'PASS', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                         'selected_raw_identity_checks': identity_checks,
                         'original_4x4_protocol_hash_equal': True,
                         'existing_4x4_verification_status': four_verify['status'],
                         'existing_4x4_rows': four_verify['GT_rows'],
                         'public_records': len(public_rows),
                         'sources_exist_and_hashed': sum('path' in x for x in sources),
                         'new_executions_this_partner': {'model_loads': 0, 'forwards': 0, 'training': 0,
                                                         'gradients_or_optimizers': 0, 'COCO_evaluations': 0,
                                                         'GPU_remote_test_reads': 0, 'subagents': 0},
                         'actual_work': {'existing_pixel_GT_panels_viewed': 16,
                                         'original_val_images_viewed': 2,
                                         'first_convolution_integer_geometry_only': True,
                                         'paper_reading_and_pdf_page_rendering': True},
                         'implementation_note': 'One requests-based download helper failed on import before any download/write; replaced with standard urllib. No experimental tolerance or outcome rule changed.'})
outputs = [local(f, 'New partner artifact', source_type='delivery_artifact')
           for f in sorted(HERE.rglob('*')) if f.is_file() and f.name != 'delivery_manifest.json']
save('delivery_manifest.json', {'status': 'PASS', 'scope': 'This partner directory only; primary PDFs are source-ledger inputs in shared literature library.',
                               'artifacts': outputs, 'artifact_count': len(outputs)})
print(json.dumps({'status': 'PASS', 'source_records': len(sources),
                  'selected_identity_checks': len(identity_checks), 'artifacts': len(outputs),
                  'new_model_forwards': 0, 'new_training': 0}, ensure_ascii=False))
