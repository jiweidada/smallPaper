"""Add a limited profiling-provenance disclosure without executing model code."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import ast
import json
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
ROOT = PAPER.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十六轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十七轮候选.docx'
REPORT = HERE / 'complexity_disclosure_pending_qa.json'
OLD_SHA = '185cbe8ffca81f86b10e8c12fab1e68eda90b722629617e870068e14a60f8993'
ARCHIVE_SHA = '4706b4e4d7032e5264e981c9d59fc9771807cb8a72c5d814363d8054ab29216e'
SUMMARY_SHA = 'f93b3cf5d2cb0dd1551559fa393b795780b4d3237864ebe5963591b1370e9210'
OLD_SENTENCE = 'The floating-point operation (FLOP) counts are expressed in billions (GFLOPs).'
NEW_SENTENCE = (OLD_SENTENCE + ' The archive summary names get_info.py but does not record the profiler version '
                'or operator coverage; the GFLOPs are therefore tool-dependent estimates.')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert digest(CURRENT) == OLD_SHA
    assert not any(p.exists() for p in (BACKUP, CANDIDATE, REPORT))
    summary = REC / 'source_snapshot/archived_benchmark/零训练数据汇总.md'
    original_summary = ROOT / '05_代码与数据/原始论文数据/01_零训练数据/零训练数据汇总.md'
    assert digest(summary) == digest(original_summary) == SUMMARY_SHA
    s = summary.read_text(encoding='utf-8')
    for x in ('tools/benchmark/get_info.py', '本机（旧树）', '24.86 G', '29.63 G', '服务器（新包）', '25.37 G', '30.14 G'):
        assert x in s
    old_source = Path(r'C:\Users\jiwei\Desktop\DEIM\tools\benchmark\get_info.py')
    current_bytes = old_source.read_bytes()
    current_text = current_bytes.decode('utf-8-sig')
    ast.parse(current_text)
    assert 'from calflops import calculate_flops' in current_text and 'calculate_flops(model=model' in current_text
    archive = ROOT / '90_历史与工具/旧期刊投稿材料/《计算机应用与软件》投稿/_work/upload/DEIM_migrated.zip'
    assert digest(archive) == ARCHIVE_SHA
    member = 'DEIM_1572862469/tools/benchmark/get_info.py'
    with ZipFile(archive) as z:
        archived_bytes = z.read(member)
    archived_text = archived_bytes.decode('utf-8-sig')
    ast.parse(archived_text)
    assert b'calculate_flops' in archived_bytes and b'thop.profile' in archived_bytes
    assert b'except Exception:' in archived_bytes and current_bytes != archived_bytes
    for name, data in [('get_info_current_tree.py.txt', current_bytes), ('get_info_migrated_archive.py.txt', archived_bytes)]:
        target = HERE / name
        assert not target.exists()
        target.write_bytes(data)
    with ZipFile(CURRENT) as z:
        original_xml = z.read('word/document.xml')
    root = etree.fromstring(original_xml)
    paragraphs = root.findall('.//w:p', NS)
    matching = []
    for i, paragraph in enumerate(paragraphs):
        nodes = paragraph.findall('.//w:t', NS)
        text = ''.join(x.text or '' for x in nodes)
        if text.startswith('Table 4 reports archived 640'):
            matching.append((i, nodes, text))
    assert len(matching) == 1
    index, nodes, before = matching[0]
    assert len(nodes) == 1 and before.count(OLD_SENTENCE) == 1
    assert original_xml.count(OLD_SENTENCE.encode()) == 1
    revised_xml = original_xml.replace(OLD_SENTENCE.encode(), NEW_SENTENCE.encode())
    revised = etree.fromstring(revised_xml)
    revised_paragraphs = revised.findall('.//w:p', NS)
    assert len(paragraphs) == len(revised_paragraphs)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, revised_paragraphs)) if etree.tostring(a) != etree.tostring(b)] == [index]
    counts = {'tables': len(revised.findall('.//w:tbl', NS)), 'equations': len(revised.findall('.//m:oMath', NS)),
              'inline_images': len(revised.findall('.//wp:inline', NS)),
              **{key: len(revised.findall('.//w:' + key, NS)) for key in ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7,
                      'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
    shutil.copy2(CURRENT, BACKUP)
    assert digest(BACKUP) == OLD_SHA
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, revised_xml if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
        assert b.testzip() is None and a.namelist() == b.namelist()
        assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 27,
              'source_sha256': OLD_SHA, 'backup_sha256': digest(BACKUP), 'candidate_sha256': digest(CANDIDATE),
              'paragraph_index_xml': index, 'before': before, 'after': before.replace(OLD_SENTENCE, NEW_SENTENCE),
              'structural_counts': counts, 'only_document_xml_changed': True,
              'evidence': {'archived_summary_sha256': SUMMARY_SHA, 'local_get_info_sha256': sha256(current_bytes).hexdigest(),
                           'migrated_archive_sha256': ARCHIVE_SHA, 'migrated_get_info_sha256': sha256(archived_bytes).hexdigest()},
              'limits': ['The summary does not identify the executed script bytes, profiler version or operator coverage.',
                         'The migrated fallback does not establish which profiler produced an archived number.',
                         'No model, profiler, evaluator or training was executed.']}
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'], 'xml_paragraph': index}))


if __name__ == '__main__':
    main()
