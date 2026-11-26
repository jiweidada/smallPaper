from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json
import shutil

here = Path(__file__).resolve().parent
paper = here.parent.parent
source = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_第六轮备份.docx'
candidate = here / 'MPSMA-DEIM_conference_EN_revised_第七轮候选.docx'
old_sha = 'dc74f01ef224d9a76ae58f262b88929214dc81881b41d314123c538ac3af1ed3'

def digest(p): return sha256(p.read_bytes()).hexdigest()

assert digest(source) == old_sha and not candidate.exists()
summary = (here.parent / 'source_snapshot/archived_benchmark/零训练数据汇总.md').read_text(encoding='utf-8')
assert all(x in summary for x in ['本机（旧树）', '服务器（新包）', '24.86 G', '29.63 G', '25.37 G', '30.14 G'])
if backup.exists(): assert digest(backup) == old_sha
else: shutil.copy2(source, backup)
assert digest(backup) == old_sha

changes = [
    ('Table 4 retains the archived deployment-complexity measurements at 640 × 640 input and batch size one. The floating-point operation (FLOP) counts are expressed in billions (GFLOPs). These counts describe the benchmarked structures, while Tables 1 and 3 report the parameter counts logged during training.',
     'Table 4 reports archived 640 × 640, batch-one deployment-complexity measurements from a local legacy code tree. The floating-point operation (FLOP) counts are expressed in billions (GFLOPs). The separate GPU timing in Table 5 comes from a server package. Tables 1 and 3 instead report parameter counts from the historical training logs.'),
    ('Absolute FLOP counts are tool-dependent: a separate archived code-tree measurement reports 25.37 and 30.14 GFLOPs for the same two model labels.',
     'The archive also reports 25.37 and 30.14 GFLOPs for the same model labels from the server package. The source of the difference between code trees has not been isolated, so those figures are not substituted for the local-tree values in Table 4.')
]
with ZipFile(source) as src, ZipFile(candidate, 'x') as dst:
    for item in src.infolist():
        content = src.read(item.filename)
        if item.filename == 'word/document.xml':
            for before, after in changes:
                old, new = before.encode('utf-8'), after.encode('utf-8')
                assert content.count(old) == 1, before
                content = content.replace(old, new)
        dst.writestr(item, content)
with ZipFile(source) as a, ZipFile(candidate) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
record = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'source_sha256': old_sha,
          'candidate_sha256': digest(candidate), 'changes': [{'old': a, 'new': b} for a,b in changes],
          'evidence': 'Archived complexity summary distinguishes 本机（旧树） and 服务器（新包）; latency JSON records an RTX 3090 GPU benchmark.',
          'limit': 'The archive does not establish why FLOPs differ between the two trees or prove exact trained-checkpoint loading.'}
(here / 'seventh_revision_pending_qa.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False))
