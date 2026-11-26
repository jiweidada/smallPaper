"""Independently verify the promoted manuscript and optimizer evidence links."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import json
import re

here = Path(__file__).resolve().parent
rec, paper = here.parent, here.parent.parent
current = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
previous = here / 'MPSMA-DEIM_conference_EN_revised_第十三轮备份.docx'
original = paper / '稿件/MPSMA-DEIM_conference_EN.docx'
original_backup = paper / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
index_path = rec / '主要结论证据索引/主要结论证据索引_第十四轮.json'
digest = lambda path: sha256(path.read_bytes()).hexdigest()
load = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
index = load(index_path)
qa = load(Path(index['visual_qa']['path']))
audit = load(Path(index['optimizer_config_and_saved_group_audit']['path']))
assert digest(current) == index['manuscript']['sha256'] == qa['candidate_sha256']
assert digest(previous) == qa['previous_docx_sha256']
assert digest(original) == digest(original_backup) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
for key in ['optimizer_config_and_saved_group_audit', 'semantic_verification', 'visual_qa',
            'plain_language_explanation', 'manuscript_version_locations', 'diagnostic_completion']:
    assert digest(Path(index[key]['path'])) == index[key]['sha256']
for ref in index['previous_indices'] + index['frozen_fair_configs']:
    assert digest(Path(ref['path'])) == ref['sha256']
for item in load(Path(index['manuscript_version_locations']['path']))['versions']:
    assert digest(Path(item['retained_file']['path'])) == item['sha256']
with ZipFile(previous) as before, ZipFile(current) as after:
    assert after.testzip() is None and before.namelist() == after.namelist()
    assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ['word/document.xml']
    root = ET.fromstring(after.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    text = ''.join(root.findall('.//w:p', ns)[54].itertext())
    assert 'backbone names without bn' in text
    assert 'some BatchNorm parameters consequently receive the lower rate and default decay' in text
    assert 'backbone parameters outside batch normalization' not in text
    assert len(root.findall('.//w:tbl', ns)) == 6
for item in audit['runs']:
    args_path = Path(item['args']['path'])
    assert digest(args_path) == item['args']['sha256']
    groups = load(args_path)['yaml_cfg']['optimizer']['params']
    name = item['named_bn_example']['parameter_name']
    assert [i for i, group in enumerate(groups) if re.search(group['params'], name)] == ([1] if item['run'] == 'S1' else [0])
assert len(audit['runs']) == 6 and qa['inspected_pages'] == list(range(1, 11))
assert not index['historical_training_source_identity_certified'] and not index['submission_ready']
assert all(audit[key] == 0 for key in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
assert any(digest(current) in line for line in (paper / '当前修订入口.txt').read_text(encoding='utf-8-sig').splitlines()[:4])
assert '当前稿件SHA256：' + digest(current) in (paper / '远程公平对照_20261103/结果并入与验收清单.txt').read_text(encoding='utf-8-sig')
proposal = (paper / '远程公平对照_20261103/公平主对照重做提案_20261105.md').read_text(encoding='utf-8-sig')
assert '优化器规则补充核查（不改变方案或授权状态）' in proposal and '不是运行授权' in proposal
print(json.dumps({'status': 'PROMOTED_REVISION_AND_OPTIMIZER_EVIDENCE_VERIFIED',
                  'current_sha256': digest(current), 'runs_checked': len(audit['runs']),
                  'pages_inspected': len(qa['inspected_pages']), 'original_and_backup_match': True,
                  'submission_ready': index['submission_ready']}, ensure_ascii=False))
