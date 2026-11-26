"""Shorten the new related-work text to keep the following expression together."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from lxml import etree
import json
import runpy

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮备份.docx'
FIRST = HERE / 'MPSMA-DEIM_conference_EN_revised_第十五轮候选.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十五轮候选_v2.docx'
REPORT = HERE / 'fifteenth_revision_v2_pending_qa.json'
SHORT = (' DMF-Net pairs multi-branch detail extraction with dual-attention fusion [9]; '
         'TF-DEIM-DFINE combines a Mamba backbone, attention and a feature pyramid [10].')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    prior = json.loads((HERE / 'fifteenth_revision_pending_qa.json').read_text(encoding='utf-8'))
    constants = runpy.run_path(str(HERE / 'patch_related_work.py'))
    assert not CANDIDATE.exists() and not REPORT.exists()
    assert sha(CURRENT) == sha(BACKUP) == prior['source_sha256']
    assert sha(FIRST) == prior['candidate_sha256']
    with ZipFile(FIRST) as z:
        raw = z.read('word/document.xml')
    root = etree.fromstring(raw)
    paragraphs = root.findall('.//w:p', NS)
    before = ''.join(paragraphs[13].xpath('.//w:t/text()', namespaces=NS))
    assert before.count(constants['ADDED']) == 1 and raw.count(before.encode('utf-8')) == 1
    after = before.replace(constants['ADDED'], SHORT)
    updated = raw.replace(before.encode('utf-8'), after.encode('utf-8'))
    revised = etree.fromstring(updated).findall('.//w:p', NS)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, revised)) if etree.tostring(a) != etree.tostring(b)] == [13]
    with ZipFile(FIRST) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
        assert a.namelist() == b.namelist() and b.testzip() is None
        assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
    report = dict(prior, status='CANDIDATE_V2_PENDING_RENDER_AND_REVIEW',
                  candidate_sha256=sha(CANDIDATE), first_candidate_sha256=sha(FIRST),
                  related_work_replacement=SHORT.strip(),
                  explanation='Shortens source-supported descriptions to retain the stem expression with its lead-in.')
    with REPORT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256']}))

if __name__ == '__main__':
    main()
