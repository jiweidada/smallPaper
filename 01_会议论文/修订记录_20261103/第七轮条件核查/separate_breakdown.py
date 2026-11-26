from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json

here = Path(__file__).resolve().parent
source = here / 'MPSMA-DEIM_conference_EN_revised_第七轮候选.docx'
target = here / 'MPSMA-DEIM_conference_EN_revised_第七轮候选_v2.docx'
assert not target.exists()
old = b'The archived module breakdown assigns approximately 3.84 M parameters'
new = b'A separately archived module breakdown assigns approximately 3.84 M parameters'
with ZipFile(source) as a, ZipFile(target, 'x') as b:
    for item in a.infolist():
        data = a.read(item.filename)
        if item.filename == 'word/document.xml':
            assert data.count(old) == 1
            data = data.replace(old, new)
        b.writestr(item, data)
with ZipFile(source) as a, ZipFile(target) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
report = {'candidate_sha256': sha256(target.read_bytes()).hexdigest(),
          'change': 'Identify the separately archived module breakdown.', 'render_status': 'PENDING_REVIEW'}
(here / 'seventh_revision_v2_pending_qa.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
