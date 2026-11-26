"""Pin mutable exports to immutable snapshots and retain the rev36 source binding."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
bundle, paper = here.parents[1], here.parents[3]
data = paper / '实验数据汇总_20261108'
index_root = paper / '修订记录_20261103/主要结论证据索引'
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return sha256(path.read_bytes()).hexdigest()
def verified_manifest(folder, expected_sha=None):
    path = folder / '交付清单.json'
    if expected_sha: assert sha(path) == expected_sha
    value = read(path)
    assert value['status'] == 'COMPLETE'
    for item in value['files']:
        file = (folder / item['file']).resolve(strict=True)
        assert file.is_relative_to(folder.resolve(strict=True))
        assert file.stat().st_size == item['bytes'] and sha(file) == item['sha256'], item['file']
    return value

original_index = index_root / '主要结论证据索引_第三十六轮.json'
old_index = read(original_index)
old_binding = next(item for item in old_index['new_evidence'] if item['file'] == '实验数据汇总_20261108/交付清单.json')
old_snapshot = data / 'snapshots/NO_MSCA_round041_before_update_20261109'
old_data = verified_manifest(old_snapshot, old_binding['sha256'])
assert old_data['NO_MSCA_accepted_rounds'] == 41
current_manifest_path = data / '交付清单.json'
current_data = verified_manifest(data)
assert current_data['NO_MSCA_accepted_rounds'] == 140
update = read(data / '第140轮更新核查.json')
assert update['per_epoch_csv_rows'] == 1100 and update['raw_metric_values_rechecked'] == 13200
assert update['NO_MSCA_final_endpoint_blank_cells'] == 24
assert update['xlsx_sha256'] == 'fe1557c094cea009485de50e89537ee67da60785b4706a6b8e99168b2f5703b1'
assert update['per_epoch_csv_sha256'] == '26e51fd6995b928a2e9c65ed6cd96a75be9ddc6de5315f9340de140fc0fbfa35'
destination = data / 'verified_snapshots/NO_MSCA_round140_verified_20261109'
assert not destination.exists()
destination.mkdir(parents=True, exist_ok=False)
for item in current_data['files']:
    target = destination / item['file']
    assert target.resolve().is_relative_to(destination.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with (data / item['file']).open('rb') as source, target.open('xb') as dest:
        shutil.copyfileobj(source, dest)
with current_manifest_path.open('rb') as source, (destination / '交付清单.json').open('xb') as dest:
    shutil.copyfileobj(source, dest)
verified_manifest(destination, sha(current_manifest_path))
assert sha(paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx') == old_index['manuscript_sha256']
supplement = {'status': 'REV36_SOURCE_RESOLVED_TO_IMMUTABLE_SNAPSHOT_AND_NEW_PROGRESS_PINNED',
    'observed_utc': datetime.now(timezone.utc).isoformat(), 'revision': 36,
    'manuscript_sha256': old_index['manuscript_sha256'],
    'original_index': {'file': str(original_index), 'sha256': sha(original_index), 'unchanged': True},
    'rev36_mutable_path_resolution': {'original_path': old_binding['file'],
        'expected_sha256': old_binding['sha256'],
        'resolved_snapshot_file': str(old_snapshot / '交付清单.json'),
        'resolved_sha256': sha(old_snapshot / '交付清单.json'),
        'snapshot_artifacts_individually_verified': len(old_data['files']),
        'reason': 'The continuous delivery manifest advances as experiments run; the original rev36 hash resolves to its retained round41 snapshot.'},
    'current_progress_snapshot': {'folder': str(destination),
        'manifest_sha256': sha(destination / '交付清单.json'), 'NO_MSCA_verified_round': 140,
        'per_epoch_rows': 1100, 'artifacts_individually_verified': len(current_data['files'])},
    'independent_audit': {'file': str(here / 'lineage_recovered_through_round140.json'),
        'sha256': sha(here / 'lineage_recovered_through_round140.json')},
    'source_reading_limits_and_scientific_endpoints_unchanged': True,
    'submission_ready': False, 'local_model_forwards': 0, 'local_evaluator_runs': 0}
supplement_path = index_root / '主要结论证据索引_第三十六轮_数据快照路径补充_20261109.json'
with supplement_path.open('x', encoding='utf-8') as handle:
    json.dump(supplement, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
state_path = bundle / 'current_execution_state.json'
state = read(state_path)
arm = state['NO_MSCA']
arm['data_snapshot_independently_verified_round'] = 140
arm['data_handoff'] = {'path': str(destination / '交付清单.json'),
    'sha256': sha(destination / '交付清单.json'), 'data_rows': 1100, 'NO_MSCA_verified_round': 140,
    'scope': 'Immutable validated snapshot'}
arm['current_data_delivery_manifest'] = {'path': str(current_manifest_path),
    'sha256': sha(current_manifest_path), 'scope': 'Continuous current delivery; resolve historic hashes via snapshots'}
state['latest_evidence_index_supplement'] = {'file': str(supplement_path), 'sha256': sha(supplement_path)}
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('【数据快照与证据回查更新：2026-11-09】\n'
    '当前Excel/JSON/CSV已整理至NO_MSCA独立核验140轮，共1100条逐轮记录；完整320轮终点仍空。'
    '140轮已验证交付另存不可变快照 verified_snapshots/NO_MSCA_round140_verified_20261109。'
    '第三十六轮旧证据索引所绑定的41轮清单，已核对并补充其保留快照精确路径，原索引字节保持。\n'
    '稿件仍第三十六轮，现有已完成三组的表值/公式/来源保持。当前训练和备份进度见模块消融/current_execution_state.json。'
    '数据快照路径补充见主要结论证据索引_第三十六轮_数据快照路径补充_20261109.json。以下为此前记录。\n\n')
entrance = paper / '当前修订入口.txt'
entrance.write_text(note + entrance.read_text(encoding='utf-8'), encoding='utf-8')
with (bundle / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write('\n\n## 2026-11-09 第140轮数据快照与旧索引路径补充\n\n' + note + '\n')
print(json.dumps({'status': supplement['status'], 'data_rows': 1100, 'verified_round': 140,
                  'old_rev36_index_unchanged': True}, ensure_ascii=False))
