"""Shorten the correction so its following table note stays with the table."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json
import runpy

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十三轮备份.docx'
FIRST = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮候选.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮候选_v2.docx'
REPORT = HERE / 'fourteenth_revision_v2_pending_qa.json'
BRIEF = ('AdamW has betas (0.9, 0.999), default decay 10⁻⁴ and name-based exemptions. '
         'The n rates are 4 × 10⁻⁴ for backbone names and 8 × 10⁻⁴ otherwise. '
         'In the inspected code, s uses 2 × 10⁻⁴ for backbone names without bn and 4 × 10⁻⁴ '
         'for other names; some BatchNorm parameters consequently receive the lower rate and default decay.')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    prior = json.loads((HERE / 'fourteenth_revision_pending_qa.json').read_text(encoding='utf-8'))
    constants = runpy.run_path(str(HERE / 'patch_optimizer_description.py'))
    assert not CANDIDATE.exists() and not REPORT.exists()
    assert sha(CURRENT) == sha(BACKUP) == prior['source_sha256']
    assert sha(FIRST) == prior['candidate_sha256']
    with ZipFile(FIRST) as z:
        raw = z.read('word/document.xml')
    root = etree.fromstring(raw)
    paragraphs = root.findall('.//w:p', NS)
    text = ''.join(paragraphs[54].xpath('.//w:t/text()', namespaces=NS))
    assert text.count(constants['NEW']) == 1 and raw.count(text.encode('utf-8')) == 1
    revised = text.replace(constants['NEW'], BRIEF)
    updated = raw.replace(text.encode('utf-8'), revised.encode('utf-8'))
    new_paragraphs = etree.fromstring(updated).findall('.//w:p', NS)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, new_paragraphs)) if etree.tostring(a) != etree.tostring(b)] == [54]
    with ZipFile(FIRST) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as before, ZipFile(CANDIDATE) as after:
        assert before.namelist() == after.namelist() and after.testzip() is None
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ['word/document.xml']
    report = dict(prior, status='CANDIDATE_V2_PENDING_RENDER_AND_REVIEW',
                  candidate_sha256=sha(CANDIDATE), first_candidate_sha256=sha(FIRST),
                  after=revised, explanation='Shortens the optimizer correction to preserve the table note placement; full scheduler details remain in the evidence record.')
    with REPORT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256']}))

if __name__ == '__main__':
    main()
