"""Bind the current references and their uses to existing source archives."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
RECORDS = HERE.parent
PAPER = RECORDS.parent
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
EXPECTED = '2653ab4164cc454b3fad4d21c4c2e02e80febe00e80e4fc3e13759a7173114ac'
ARCHIVE = RECORDS / '第十五轮相关工作补核_20261105/citation_archive_revision15.json'
NOW = datetime.now(timezone.utc).isoformat()
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
relative = lambda p: p.relative_to(PAPER).as_posix() if p.is_relative_to(PAPER) else str(p)

def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

assert digest(MASTER) == EXPECTED
archive = read(ARCHIVE)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(MASTER) as docx:
    xml = etree.fromstring(docx.read('word/document.xml'))
paragraphs = [''.join(p.xpath('.//w:t/text()', namespaces=ns)) for p in xml.findall('w:body/w:p', ns)]
refs = {int(re.match(r'^\[(\d+)\]', p).group(1)): p for p in paragraphs if re.match(r'^\[\d+\]', p)}
assert set(refs) == set(range(1, 14))
body = paragraphs[paragraphs.index('1. Introduction'):paragraphs.index('References')]
contexts = {n: [] for n in refs}
first_order = []
for index, text in enumerate(body):
    numbers = []
    for match in re.findall(r'(?<![=\w])\[([\d, ]+)\]', text):
        numbers.extend(int(v) for v in match.split(',') if v.strip())
    for number in sorted(set(numbers)):
        assert number in contexts
        contexts[number].append({'body_paragraph_index': index, 'text': text})
    for number in numbers:
        if number not in first_order:
            first_order.append(number)
assert first_order == list(range(1, 14))

checks = []
def check(path, expected, role):
    path = Path(path)
    actual = digest(path) if path.is_file() else None
    record = {'path': str(path), 'role': role, 'expected_sha256': expected,
              'actual_sha256': actual, 'matches': actual == expected}
    checks.append(record)
    assert record['matches'], record
    return record

rows = []
for entry in archive['references']:
    n = entry['reference']
    assert refs[n] == entry['manuscript_entry'], {'reference': n, 'archive': entry['manuscript_entry'], 'current': refs[n]}
    source = entry.get('inherited_source_record') or entry['new_source_record']
    files = []
    if source.get('sha256'):
        files.append(check(source['pdf'], source['sha256'], 'archived PDF'))
    if source.get('text_sha256'):
        files.append(check(source['text'], source['text_sha256'], 'extracted text'))
    for file in source.get('source_files', []):
        files.append(check(file['path'], file['sha256'], 'official evaluation source; static inspection only'))
    rows.append({'reference': n, 'current_entry': refs[n],
                 'current_entry_sha256': sha256(refs[n].encode('utf-8')).hexdigest(),
                 'bibliographic_entry_identical_to_revision15': True,
                 'reading_scope': source.get('reading_scope', source.get('read_scope')),
                 'download_status': source.get('download_status', 'No full PDF obtained'),
                 'verified_local_source_files': files, 'current_citation_contexts': contexts[n],
                 'source_url': source.get('source_url', source.get('publisher_url')),
                 'full_text_unavailable': n in [1, 3, 9, 10]})

retrieval_path = RECORDS / '引用补核_20261104/primary_web_retrieval.json'
for record in read(retrieval_path):
    if record.get('target_in_main_heading') or record.get('file') == 'Waterloo_author_publication_list.html':
        check(retrieval_path.parent / record['file'], record['sha256'], 'reference 3 correct official program or author-institution page')

refresh = {
    'observed_local_utc': NOW,
    'method': 'Direct publisher opens followed by publisher-domain-restricted search-index retrieval.',
    'access_limits': 'Direct MDPI opens returned 429 and ScienceDirect returned 403. Indexed publisher excerpts were available. No full PDF was obtained, no paywall bypass was attempted, and this is not a full-text-reading certificate.',
    'sources': [
        {'reference': 1, 'url': 'https://www.mdpi.com/2072-4292/16/1/149',
         'direct_open_status': 429, 'scope': 'Indexed publisher abstract and background sections',
         'supported_current_claim': 'The survey covers tiny objects, difficult backgrounds and changing object scales in drone imagery.',
         'current_use': 'General introduction background only; occlusion also has the VisDrone primary citation.'},
        {'reference': 9, 'url': 'https://www.mdpi.com/2073-8994/18/5/763',
         'direct_open_status': 429, 'scope': 'Indexed publisher abstract, method excerpts and publication metadata',
         'supported_current_claim': 'The detector uses parallel paths for detailed backbone features and attention during feature fusion.',
         'metadata_verified': {'doi': '10.3390/sym18050763', 'volume': 18, 'issue': 5, 'article': 763, 'year': 2026},
         'current_use': 'One related-work sentence describing DMF-Net; no imported performance or detailed formula claim.'},
        {'reference': 10, 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S026288562600003X',
         'direct_open_status': 403, 'scope': 'Indexed publisher abstract, introduction and section snippets',
         'supported_current_claim': 'The detector combines a Mamba-related backbone, an attention encoder and a tiered feature pyramid.',
         'metadata_verified': {'doi': '10.1016/j.imavis.2026.105897', 'volume': 167, 'article': 105897, 'year': 2026},
         'current_use': 'One related-work sentence describing TF-DEIM-DFINE; no imported performance or detailed formula claim.'}
    ],
    'novelty_limit': 'This targeted source check is not an exhaustive literature search or a proof of novelty.',
    'new_training_or_evaluation': False}
refresh_path = HERE / 'publisher_index_refresh_20261108.json'
write(refresh_path, refresh)
report = {'status': 'CURRENT_REV34_REFERENCES_AND_ARCHIVED_SOURCE_HASHES_VERIFIED_WITH_READING_LIMITS',
          'observed_local_utc': NOW, 'manuscript_sha256': EXPECTED, 'content_revision': 34,
          'archive': relative(ARCHIVE), 'archive_sha256': digest(ARCHIVE),
          'reference_count': 13, 'first_citation_order': first_order,
          'all_current_entries_identical_to_audited_archive': True,
          'verified_source_file_count': len(checks), 'verified_source_files': checks,
          'full_text_PDF_and_text_references': [2, 4, 5, 6, 7, 8, 11, 12],
          'full_text_unavailable_references': [1, 3, 9, 10],
          'official_toolkit_reference': 13,
          'publisher_refresh': relative(refresh_path), 'publisher_refresh_sha256': digest(refresh_path),
          'references': rows,
          'interpretation': 'Current references are structurally and bibliographically bound to the archived checks. Existing source bytes were rehashed. Current citation contexts are retained for review; a byte match alone is not a new semantic or full-text-reading certificate.',
          'manuscript_changed': False, 'new_model_forwards': 0, 'new_evaluations': 0,
          'submission_ready': False}
report_path = HERE / 'citation_binding_current_rev34.json'
write(report_path, report)
with (HERE / '当前十三条引用绑定与阅读范围.txt').open('x', encoding='utf-8') as stream:
    stream.write('当前第三十四轮十三条引用绑定与阅读范围\n\n'
                 '稿件 SHA-256: ' + EXPECTED + '\n'
                 '十三条文献与已核查归档逐条一致，首次出现顺序为 1–13，每条仍在正文使用。\n'
                 '重新核对八篇已留存 PDF 及各自提取文本、两份官方评价源码、IGARSS 正确论文页和作者机构页，共 ' + str(len(checks)) + ' 个文件，哈希全部匹配。\n'
                 '参考文献 1、3、9、10 仍没有全文 PDF，不将摘要、索引节选或元数据核查称为全文阅读。文献 13 仅作官方评价源码静态核对，未执行。\n'
                 '重新查询文献 1、9、10 的出版方页面；直接访问仍被限流或拒绝，出版方索引摘要/节选可读，当前一般背景及两句相关方法概要未见超出该阅读范围。没有从外文献引入性能比较或详细公式，也不宣称穷尽新颖性检索。\n'
                 '本次不修改稿件，不新增训练、模型推理或评价。MSCA 终点及相应条件结果分析仍待完成，尚不能投稿。\n')
assert digest(MASTER) == EXPECTED
print(json.dumps({k: report[k] for k in ['status', 'reference_count', 'verified_source_file_count', 'full_text_unavailable_references', 'manuscript_changed', 'submission_ready']}))
