"""Independently check the promoted manuscript and its immediately relevant evidence."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import json

here = Path(__file__).resolve().parent
rec, paper = here.parent, here.parent.parent
current = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
previous = here / 'MPSMA-DEIM_conference_EN_revised_第十二轮备份.docx'
original = paper / '稿件/MPSMA-DEIM_conference_EN.docx'
original_backup = paper / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
index_path = rec / '主要结论证据索引/主要结论证据索引_第十三轮.json'
digest = lambda path: sha256(path.read_bytes()).hexdigest()
load = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
index = load(index_path)
qa = load(Path(index['visual_qa']['path']))
audit = load(Path(index['static_restore_and_collate_evidence']['path']))
assert digest(current) == index['manuscript']['sha256'] == qa['candidate_sha256']
assert digest(previous) == qa['previous_docx_sha256']
assert digest(original) == digest(original_backup) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
for key in ['static_restore_and_collate_evidence', 'visual_qa', 'plain_language_explanation',
            'prior_audit_correction_note', 'manuscript_version_locations', 'diagnostic_completion']:
    assert digest(Path(index[key]['path'])) == index[key]['sha256']
for ref in index['previous_indices']:
    assert digest(Path(ref['path'])) == ref['sha256']
for item in load(Path(index['manuscript_version_locations']['path']))['versions']:
    assert digest(Path(item['retained_file']['path'])) == item['sha256']

with ZipFile(previous) as before, ZipFile(current) as after:
    assert after.testzip() is None
    assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ['word/document.xml']
    root = ET.fromstring(after.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = [''.join(node.itertext()) for node in root.findall('.//w:p', ns)]
    text = paragraphs[228]
    assert 'selects a stage-one checkpoint by validation AP' in text
    assert 'whether this solver was used historically is unverified' in text
    assert len(root.findall('.//w:tbl', ns)) == 6

for item in audit['configs']:
    path = rec / 'source_snapshot' / item['run'] / 'args.json'
    assert digest(path) == item['args_sha256']
    cfg = load(path)['yaml_cfg']
    assert cfg['train_dataloader']['collate_fn']['base_size_repeat'] is None
    assert 'base_size_repeat' not in cfg['val_dataloader']['collate_fn']
assert len(audit['configs']) == 6
assert not audit['historical_solver_identity_certified'] and not index['submission_ready']
assert all(audit[x] == 0 for x in ['model_executions', 'evaluator_executions', 'training_updates'])
assert '第十三轮核查更正' in Path(index['prior_audit_correction_note']['path']).read_text(encoding='utf-8')
assert any(digest(current) in line for line in (paper / '当前修订入口.txt').read_text(encoding='utf-8-sig').splitlines()[:4])
assert '当前稿件SHA256：' + digest(current) in (paper / '远程公平对照_20261103/结果并入与验收清单.txt').read_text(encoding='utf-8-sig')
assert qa['inspected_pages'] == list(range(1, 11))
print(json.dumps({'status': 'PROMOTED_REVISION_AND_EVIDENCE_VERIFIED', 'current_sha256': digest(current),
                  'saved_configs_checked': len(audit['configs']), 'pages_inspected': len(qa['inspected_pages']),
                  'original_and_backup_match': True, 'submission_ready': index['submission_ready']}, ensure_ascii=False))
