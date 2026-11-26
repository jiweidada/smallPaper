from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json

here = Path(__file__).resolve().parent
source = here / 'MPSMA-DEIM_conference_EN_revised_第六轮候选.docx'
target = here / 'MPSMA-DEIM_conference_EN_revised_第六轮候选_v2.docx'
assert not target.exists()
with ZipFile(source) as src, ZipFile(target, 'x') as dst:
    for item in src.infolist():
        data = src.read(item.filename)
        if item.filename == 'word/document.xml':
            needle = b'For each channel of a feature map T, define'
            assert data.count(needle) == 1
            pos = data.index(needle)
            start = data.rfind(b'<w:p>', 0, pos)
            props = data.index(b'<w:pPr>', start, pos) + len(b'<w:pPr>')
            assert start >= 0 and b'<w:keepNext' not in data[props:pos]
            data = data[:props] + b'<w:keepNext/>' + data[props:]
        dst.writestr(item, data)
with ZipFile(source) as a, ZipFile(target) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
report = {'candidate_sha256': sha256(target.read_bytes()).hexdigest(),
          'change': 'Keep the paragraph introducing equation (2) with the following equation.',
          'render_status': 'PENDING_REVIEW'}
(here / 'sixth_revision_v2_pending_qa.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
