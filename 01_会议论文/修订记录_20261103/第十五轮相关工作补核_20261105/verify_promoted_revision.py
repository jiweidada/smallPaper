"""Verify the promoted manuscript, citation mapping and evidence references."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import json
import re

here = Path(__file__).resolve().parent
rec, paper = here.parent, here.parent.parent
current = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
previous = here / 'MPSMA-DEIM_conference_EN_revised_第十四轮备份.docx'
original = paper / '稿件/MPSMA-DEIM_conference_EN.docx'
original_backup = paper / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
index_path = rec / '主要结论证据索引/主要结论证据索引_第十五轮.json'
digest = lambda path: sha256(path.read_bytes()).hexdigest()
load = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
index = load(index_path)
qa = load(Path(index['visual_qa']['path']))
catalog = load(Path(index['citation_archive']['path']))
evidence = load(Path(index['literature_evidence']['path']))
assert digest(current) == index['manuscript']['sha256'] == qa['candidate_sha256'] == catalog['manuscript_sha256']
assert digest(previous) == qa['previous_docx_sha256']
assert digest(original) == digest(original_backup) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
for key in ['literature_evidence', 'plain_language_explanation', 'citation_archive',
            'visual_qa', 'manuscript_version_locations', 'diagnostic_completion']:
    assert digest(Path(index[key]['path'])) == index[key]['sha256']
for ref in index['previous_indices']:
    assert digest(Path(ref['path'])) == ref['sha256']
for item in load(Path(index['manuscript_version_locations']['path']))['versions']:
    assert digest(Path(item['retained_file']['path'])) == item['sha256']
with ZipFile(previous) as before, ZipFile(current) as after:
    assert after.testzip() is None and before.namelist() == after.namelist()
    assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ['word/document.xml']
    root = ET.fromstring(after.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    pars = root.findall('.//w:p', ns)
    text = ''.join(pars[13].itertext())
    assert 'DMF-Net pairs multi-branch detail extraction with dual-attention fusion [9]' in text
    assert 'TF-DEIM-DFINE combines a Mamba backbone, attention and a feature pyramid [10]' in text
    body = root.find('w:body', ns)
    refs = [''.join(p.itertext()) for p in body.findall('w:p', ns)]
    refs = [p for p in refs if re.match(r'^\[\d+\] ', p)]
    assert [int(re.match(r'^\[(\d+)\]', p).group(1)) for p in refs] == list(range(1, 14))
    assert refs == [item['manuscript_entry'] for item in catalog['references']]
    assert '10.3390/sym18050763' in refs[8] and '10.1016/j.imavis.2026.105897' in refs[9]
    assert len(root.findall('.//w:tbl', ns)) == 6
assert [x['reference_in_revision14'] for x in catalog['references']] == list(range(1, 9)) + [None, None, 9, 10, 11]
assert qa['inspected_pages'] == list(range(1, 11)) and qa['citation_first_appearance_order'] == list(range(1, 14))
assert not evidence['cross_paper_metric_comparisons'] and not evidence['new_experiments'] and not index['submission_ready']
assert any(digest(current) in line for line in (paper / '当前修订入口.txt').read_text(encoding='utf-8-sig').splitlines()[:4])
assert '当前稿件SHA256：' + digest(current) in (paper / '远程公平对照_20261103/结果并入与验收清单.txt').read_text(encoding='utf-8-sig')
assert '当前稿共13条引用' in (rec / '引用核查记录.txt').read_text(encoding='utf-8-sig').splitlines()[1]
print(json.dumps({'status': 'PROMOTED_REVISION_AND_CITATIONS_VERIFIED', 'current_sha256': digest(current),
                  'references': len(refs), 'pages_inspected': len(qa['inspected_pages']),
                  'original_and_backup_match': True, 'submission_ready': index['submission_ready']}, ensure_ascii=False))
