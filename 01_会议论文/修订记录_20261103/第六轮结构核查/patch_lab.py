from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json
import shutil

here = Path(__file__).resolve().parent
paper = here.parent.parent
source = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_第五轮备份.docx'
candidate = here / 'MPSMA-DEIM_conference_EN_revised_第六轮候选.docx'
diagram = here / 'fig2_mps_sixth_candidate.png'
old_doc = '3fdd80a2cac6eaddd9e79c9ee416b5f9bb5a6561ab5c240e47253b3bf743f033'
old_figure = '216bef948774510d8e168a65977729cb4b691d006b32239ed1c168f89545f911'

def digest(p): return sha256(p.read_bytes()).hexdigest()

assert digest(source) == old_doc and not candidate.exists()
for run in ['N1', 'N4', 'S1']:
    args = json.loads((here.parent / 'source_snapshot' / run / 'args.json').read_text(encoding='utf-8'))['yaml_cfg']
    cfg = args[args['DEIM']['backbone']]
    assert cfg['stem_type'] == 'enhanced' and cfg['use_lab'] is True
full = json.loads((paper / '远程公平对照_20261103/remote_evidence/execution_configs/FULL.yml').read_text(encoding='utf-8'))
assert full[full['DEIM']['backbone']]['use_lab'] is True
if backup.exists():
    assert digest(backup) == old_doc
else:
    shutil.copy2(source, backup)
    assert digest(backup) == old_doc

before = 'The output stride is four, as in the original stem. The name progressive refers to the two downsampling steps;'
after = ('The output stride is four, as in the original stem. The reported configurations enable the learnable affine block (LAB): '
         'each ConvBNAct in the detail, main, fusion and final downsampling paths applies a learned scalar scale and bias after activation. '
         'The high-frequency and spatial-gate paths do not contain this block. The name progressive refers to the two downsampling steps;')
with ZipFile(source) as src, ZipFile(candidate, 'x') as dst:
    for item in src.infolist():
        data = src.read(item.filename)
        if item.filename == 'word/document.xml':
            old, new = before.encode('utf-8'), after.encode('utf-8')
            assert data.count(old) == 1
            data = data.replace(old, new)
        elif item.filename == 'word/media/image9.png':
            assert sha256(data).hexdigest() == old_figure
            data = diagram.read_bytes()
        dst.writestr(item, data)
with ZipFile(source) as src, ZipFile(candidate) as dst:
    assert src.namelist() == dst.namelist() and dst.testzip() is None
    changed = [n for n in src.namelist() if src.read(n) != dst.read(n)]
    assert sorted(changed) == ['word/document.xml', 'word/media/image9.png']
record = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'source_sha256': old_doc,
          'candidate_sha256': digest(candidate), 'changed_parts': changed,
          'old_figure_sha256': old_figure, 'new_figure_sha256': digest(diagram),
          'paragraph_old': before, 'paragraph_new': after,
          'evidence': 'Current EnhancedStemBlock passes use_lab to detail/main/fusion/final ConvBNAct; enabled configurations and retained checkpoint LAB tensors corroborate the intended structure. High-frequency and spatial gates use separate direct convolutions.',
          'limit': 'Current source and configuration agree; this does not establish an immutable historical training-source binding.'}
(here / 'sixth_revision_pending_qa.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False))
