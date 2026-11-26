from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json
import shutil


here = Path(__file__).resolve().parent
root = here.parent.parent
source = root / '稿件' / 'MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_第四轮备份.docx'
candidate = here / 'MPSMA-DEIM_conference_EN_revised_第五轮候选.docx'
expected = '3aeb70332b9a4dc9d66720a1ded4b5b067ce6eeba47a6f489a280e444bf79984'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


assert digest(source) == expected, 'The current draft changed; review before editing.'
assert not candidate.exists(), 'Review the existing candidate before creating another.'
if backup.exists():
    assert digest(backup) == expected
else:
    shutil.copy2(source, backup)
    assert digest(backup) == expected

old_paragraph = ('A separate channel gate uses global average pooling (GAP) of F and two 1 × 1 '
                 'convolutions, reducing C to C/8 and restoring C:')
new_paragraph = ('A separate channel gate uses global average pooling (GAP) of F and two 1 × 1 '
                 'convolutions, reducing C to C/8 and restoring C. Their weights and biases are '
                 'W₁, W₂ and b₁, b₂, respectively:')
old_formula = 'Ac = sigmoid{W₂ ReLU[W₁ GAP(F)]}.                                           (4)'
new_formula = 'Ac = sigmoid{W₂ ReLU[W₁ GAP(F) + b₁] + b₂}.                               (4)'
with ZipFile(source) as src, ZipFile(candidate, 'x') as dst:
    for item in src.infolist():
        content = src.read(item.filename)
        if item.filename == 'word/document.xml':
            for before, after in ((old_paragraph, new_paragraph), (old_formula, new_formula)):
                old, new = before.encode('utf-8'), after.encode('utf-8')
                assert content.count(old) == 1, before
                content = content.replace(old, new)
        dst.writestr(item, content)

with ZipFile(source) as src, ZipFile(candidate) as dst:
    assert src.namelist() == dst.namelist()
    changed = [n for n in src.namelist() if src.read(n) != dst.read(n)]
    assert changed == ['word/document.xml'], changed
    assert dst.testzip() is None

record = {
    'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW',
    'source_sha256': expected,
    'candidate_sha256': digest(candidate),
    'changed_parts': changed,
    'changes': [
        {'old': old_paragraph, 'new': new_paragraph},
        {'old': old_formula, 'new': new_formula},
    ],
    'evidence': 'source_snapshot/current_code/engine/extre_module/custom_nn/attention/MSCA.py: MSCA_Dilated.channel_att uses two nn.Conv2d layers with default bias=True.',
    'note': 'No experimental result was changed or added; historical version binding is unchanged.'
}
(here / 'fifth_revision_pending_qa.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False))
