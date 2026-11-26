"""Resolve historical manuscript hashes to retained files without changing DOCX bytes."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
INDICES = REC / '主要结论证据索引'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
INDEX = INDICES / '主要结论证据索引_第九轮.json'
TEXT = INDEX.with_suffix('.txt')
NEW = '41cbe7bd9c42fda9ab5fac813207209d591250076e15d62f3097647d1716fc2f'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def ref(path):
    return {'path': str(path), 'sha256': sha(path)}

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

assert sha(CURRENT) == NEW
current_index = read(INDEX)
assert current_index['manuscript_sha256'] == NEW
assert 'manuscript_version_locations' not in current_index
for item in current_index['inherited_evidence']:
    assert sha(Path(item['path'])) == item['sha256']
previous_index = read(INDICES / '主要结论证据索引_第八轮.json')
for item in previous_index['inherited_numerical_and_implementation_checks']:
    assert sha(Path(item['path'])) == item['sha256']

json_backup = HERE / '主要结论证据索引_第九轮_位置补注前.json'
text_backup = HERE / '主要结论证据索引_第九轮_位置补注前.txt'
catalog = HERE / 'manuscript_version_locations.json'
assert not any(p.exists() for p in [json_backup, text_backup, catalog])
shutil.copy2(INDEX, json_backup)
shutil.copy2(TEXT, text_backup)

versions = []
specifications = [
    ('original', PAPER / '稿件/MPSMA-DEIM_conference_EN.docx',
        PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx',
        'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323', None),
    (7, CURRENT, REC / '引用补核_20261104/MPSMA-DEIM_conference_EN_revised_第七轮备份.docx',
        'a494f6b7ca2b1c200232ab49fa7f3d3831b161e2552990da3b88109f7b599de2',
        INDICES / '主要结论证据索引_第七轮.json'),
    (8, CURRENT, HERE / 'MPSMA-DEIM_conference_EN_revised_第八轮备份.docx',
        '37026584852e657c15bce14d522d261762acf2695c7bd681948b56ff0ccc66ac',
        INDICES / '主要结论证据索引_第八轮.json'),
    (9, CURRENT, HERE / 'MPSMA-DEIM_conference_EN_revised_第九轮候选.docx',
        NEW, json_backup),
]
for revision, canonical, retained, expected, index_path in specifications:
    assert sha(retained) == expected
    item = {'revision': revision, 'sha256': expected,
            'canonical_path_recorded_at_the_time': str(canonical),
            'retained_file': ref(retained),
            'current_canonical_has_these_bytes': sha(canonical) == expected}
    if index_path:
        assert read(index_path)['manuscript_sha256'] == expected
        item['index_at_the_time'] = ref(index_path)
    versions.append(item)
result = {
    'status': 'CURRENT_AND_INHERITED_MANUSCRIPT_VERSIONS_LOCATED',
    'checked_local_utc': datetime.now(timezone.utc).isoformat(),
    'current_manuscript': ref(CURRENT),
    'rule': 'Resolve the manuscript expected SHA in a historical index to retained_file with that exact SHA. The canonical revised path is a mutable current entry point.',
    'versions': versions,
    'index_preimages': [ref(json_backup), ref(text_backup)],
    'scope': 'Original manuscript and revisions 7-9 used by the current inherited evidence chain. Artifact identity only; no new numerical/source-version certification or model execution.',
    'historical_indices_modified': False, 'docx_files_modified': False,
    'model_executions': 0, 'evaluator_reruns': 0,
}
save(catalog, result)
current_index['manuscript_version_locations'] = ref(catalog)
current_index['version_location_followup_utc'] = result['checked_local_utc']
save(INDEX, current_index)
with TEXT.open('a', encoding='utf-8') as handle:
    handle.write('\n历史稿件位置补注（2026-11-04）：旧索引中的统一稿件路径是当时入口；该路径现保存第九轮，不能用它核验旧版稿件SHA。按增强时序核查_20261104/manuscript_version_locations.json，将当前继承链中的第七、八轮SHA解析到留存备份，并定位第九轮同字节留档及原稿备份。四份留档均已逐文件核对SHA。旧索引保持原样，补注前第九轮TXT/JSON另存备份。此补注只证明稿件文件身份，不扩大任何数值、历史源码或实验结论的核查范围。\n')
receipt = {
    'status': result['status'], 'catalog': ref(catalog),
    'current_index_after': ref(INDEX), 'current_index_text_after': ref(TEXT),
    'manuscript_sha256_before_and_after': NEW,
    'retained_version_count': len(versions),
    'historical_index_canonical_paths_resolved': 2,
    'historical_indices_modified': False, 'model_executions': 0,
}
assert sha(CURRENT) == NEW
save(HERE / 'manuscript_version_location_receipt.json', receipt)
print(json.dumps(receipt, ensure_ascii=False))
