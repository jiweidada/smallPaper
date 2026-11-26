"""Collect existing verified experiment records without executing models or evaluators."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent.parent
PAPER = OUT.parent
REMOTE = PAPER / '远程公平对照_20261103'
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
SOURCE_PATHS = {
    'BASE_FULL': '断点续训_20261105/BASE_FULL_pair_verified_20261108.json',
    'NO_MPS_FULL': '模块消融/NO_MPS_FULL_endpoint_comparison_20261108.json',
    'NO_MPS_LINEAGE': '模块消融/NO_MPS/lineage_through_round320.json',
    'NO_MSCA_ROUND29': '模块消融/NO_MSCA/round029_verified_from_D.json',
    'NO_MSCA_LINEAGE': '模块消融/NO_MSCA/lineage_through_round029.json',
    'SHARED_CONDITIONS': '模块消融/module_shared_conditions_verified_20261108.json',
    'PARENT_PROTOCOL': 'execution_protocol.json',
    'NO_MPS_PROTOCOL': '模块消融/NO_MPS/execution_protocol.json',
    'NO_MSCA_PROTOCOL': '模块消融/NO_MSCA/execution_protocol.json',
}
CHECKS = []
SOURCES = []
ARTIFACTS = []
HASH_CACHE = {}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    key = (str(path.resolve(strict=True)), path.stat().st_size, path.stat().st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while data := stream.read(4 * 1024**2):
                digest.update(data)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def check(name, condition, **details):
    if not condition:
        raise RuntimeError(name)
    CHECKS.append({'check': name, 'status': 'PASS', **details})


def write_json(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(name, fields, rows):
    with (OUT / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_artifact(arm, role, path, expected):
    actual = sha(path)
    check(f'{arm}/{role}/{path.name} SHA-256', actual == expected)
    ARTIFACTS.append({'arm': arm, 'endpoint_role': role, 'path': str(path),
                      'bytes': path.stat().st_size, 'expected_sha256': expected,
                      'actual_sha256': actual, 'status': 'PASS'})


def endpoint(arm, role, data, source_id, pointer, params, config_sha, protocol_sha, folder):
    check(f'{arm}/{role} metric names', set(data['metrics']) == set(METRICS))
    check(f'{arm}/{role} finite fractions', all(type(data['metrics'][m]) in (int, float)
          and math.isfinite(data['metrics'][m]) and 0 <= data['metrics'][m] <= 1 for m in METRICS))
    epoch = data['epoch_index']
    check(f'{arm}/{role} epoch', 0 <= epoch < 320 and (role != 'primary' or epoch == 319))
    check(f'{arm}/{role} coverage', data['validation_images'] == 548 and data['predictions'] == 164400)
    prefix = 'terminal' if role == 'primary' else 'best'
    weight = 'terminal_epoch0319.pt' if role == 'primary' else 'best.pt'
    for name, key in [(weight, 'checkpoint_sha256'), (f'{prefix}.metrics.json', 'metrics_sha256'),
                      (f'{prefix}.predictions.json', 'prediction_sha256'),
                      (f'{prefix}.evaluation_arrays.npz', 'evaluation_arrays_sha256')]:
        verify_artifact(arm, role, folder / name, data[key])
    saved = read(folder / f'{prefix}.metrics.json')
    check(f'{arm}/{role} saved metric equality', saved['epoch'] == epoch and saved['metrics'] == data['metrics']
          and saved['checkpoint_sha256'] == data['checkpoint_sha256'])
    return {
        'arm': arm, 'endpoint_role': role, 'endpoint_label': '固定320轮主端点' if role == 'primary' else 'AP最佳次端点',
        'round_1based': epoch + 1, 'epoch_index_0based': epoch, 'seed': 0, 'completed_rounds': 320,
        'optimizer_step_attempts_at_evaluation': (epoch + 1) * 1617,
        'total_optimizer_step_attempts': 517440, 'trainable_parameters': params, 'weight_source': 'EMA',
        'checkpoint_sha256': data['checkpoint_sha256'], 'ema_tensor_sha256': data['ema_tensor_sha256'],
        'metrics_sha256': data['metrics_sha256'], 'prediction_sha256': data['prediction_sha256'],
        'evaluation_arrays_sha256': data['evaluation_arrays_sha256'],
        'protocol_sha256': protocol_sha, 'configuration_sha256': config_sha,
        'metrics_fraction': data['metrics'], 'metrics_percent': {m: data['metrics'][m] * 100 for m in METRICS},
        'validation_images': 548, 'predictions': 164400, 'source_id': source_id, 'source_json_pointer': pointer,
        'local_artifact_folder': str(folder),
    }


def main():
    sources = {}
    for source_id, relative in SOURCE_PATHS.items():
        path = REMOTE / relative
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        value = json.loads(payload)
        sources[source_id] = value
        SOURCES.append({'source_id': source_id, 'relative_to_remote_root': relative, 'path': str(path),
                        'bytes': len(payload), 'sha256': digest, 'status_in_source': value['status']})
    hashes = {item['source_id']: item['sha256'] for item in SOURCES}
    pair = sources['BASE_FULL']
    comparison = sources['NO_MPS_FULL']
    no_mps = sources['NO_MPS_LINEAGE']
    progress = sources['NO_MSCA_LINEAGE']
    round29 = sources['NO_MSCA_ROUND29']
    shared = sources['SHARED_CONDITIONS']
    parent_protocol = sources['PARENT_PROTOCOL']
    check('BASE/FULL verified completion', pair['result_generated'] is True and
          all(pair['lineages'][a]['joined_completed_epochs'] == 320 for a in ('BASE', 'FULL')))
    check('NO_MPS verified completion', no_mps['completed_rounds'] == 320)
    for relative, expected in comparison['evidence_sha256'].items():
        path = REMOTE / relative.replace('\\', '/')
        check('comparison source binding ' + relative, sha(path) == expected, expected_sha256=expected)
    check('BASE/FULL frozen protocol', hashes['PARENT_PROTOCOL'] == pair['shared_conditions']['protocol_sha256']
          == shared['parent_protocol_sha256'])
    for arm, lineage in [('NO_MPS', no_mps), ('NO_MSCA', progress)]:
        protocol = sources[f'{arm}_PROTOCOL']
        check(arm + ' frozen protocol', hashes[f'{arm}_PROTOCOL'] == lineage['protocol_sha256']
              == shared['arms'][arm]['protocol_sha256'])
        check(arm + ' frozen budget', protocol['epochs'] == 320 and protocol['seed'] == 0
              and protocol['batch_size'] == 4 and protocol['optimizer_update_limit_per_arm'] == 517440)
        check(arm + ' frozen configuration', protocol['configuration_sha256'] ==
              shared['arms'][arm]['configuration_sha256'])
    check('NO_MSCA progress cursor', progress['completed_rounds'] == round29['round'] == 29
          and progress['best_epoch_index'] == 28)
    check('NO_MSCA progress metrics agree', progress['latest_metrics'] == round29['metrics']
          == progress['endpoints']['best']['metrics'])
    check('NO_MSCA progress coverage', round29['validation_images'] == 548 and round29['predictions'] == 164400
          and round29['learning_rate_records'] == 29 * 1617 and round29['rng_states_present'] is True)

    frozen = {item['path']: item['sha256'] for item in parent_protocol['frozen_files']}
    records = []
    base_folder = REMOTE / '断点续训_20261105/BASE_30166_r308/terminal_evidence/run'
    full_folder = REMOTE / '断点续训_20261105/FULL_30848_20261106/backups/epochs/epoch0319'
    for arm in ('BASE', 'FULL'):
        lineage = pair['lineages'][arm]
        for role, source_role in [('primary', 'terminal'), ('secondary', 'best')]:
            data = lineage['endpoints'][source_role]
            folder = base_folder if arm == 'BASE' else full_folder
            if arm == 'FULL' and role == 'secondary':
                folder = folder / 'selected' / f"epoch{data['epoch_index']:04d}"
            records.append(endpoint(arm, role, data, 'BASE_FULL', f'/lineages/{arm}/endpoints/{source_role}',
                            pair['shared_conditions']['trainable_parameters'][arm],
                            frozen[f'execution_configs/{arm}.yml'], hashes['PARENT_PROTOCOL'], folder))
    no_mps_folder = Path(no_mps['backup_root']) / 'epochs/epoch0319'
    for role, source_role in [('primary', 'terminal'), ('secondary', 'best')]:
        data = no_mps['endpoints'][source_role]
        folder = no_mps_folder if role == 'primary' else no_mps_folder / 'selected/epoch0319'
        records.append(endpoint('NO_MPS', role, data, 'NO_MPS_LINEAGE', f'/endpoints/{source_role}',
                        shared['arms']['NO_MPS']['trainable_parameters'],
                        shared['arms']['NO_MPS']['configuration_sha256'], hashes['NO_MPS_PROTOCOL'], folder))
    index = {(record['arm'], record['endpoint_role']): record for record in records}
    differences = []
    for role, pair_key, ablation_key in [('primary', 'terminal', 'primary_fixed_round320'),
                                        ('secondary', 'best', 'secondary_highest_AP')]:
        for reference in ('BASE', 'NO_MPS'):
            for metric in METRICS:
                full_value = index['FULL', role]['metrics_fraction'][metric]
                reference_value = index[reference, role]['metrics_fraction'][metric]
                expected = (pair['comparisons'][pair_key]['FULL_minus_BASE_percentage_points'][metric]
                            if reference == 'BASE' else
                            comparison['comparisons'][ablation_key]['FULL_minus_NO_MPS_percentage_points'][metric])
                check(f'{role} FULL minus {reference} {metric}', abs(100 * (full_value - reference_value) - expected) < 1e-12)
                differences.append({'endpoint_role': role, 'comparison': f'FULL-{reference}', 'metric': metric,
                                    'FULL_fraction': full_value, 'reference_fraction': reference_value,
                                    'difference_percentage_points': expected})
        for arm in ('FULL', 'NO_MPS'):
            check(f'{role} {arm} cross-report values', comparison['comparisons'][ablation_key]['fractions'][arm]
                  == index[arm, role]['metrics_fraction'])
            check(f'{role} {arm} cross-report checkpoint', comparison['comparisons'][ablation_key]['endpoint_checkpoint_sha256'][arm]
                  == index[arm, role]['checkpoint_sha256'])
    check('NO_MPS primary and selected best coincide', index['NO_MPS', 'primary']['checkpoint_sha256']
          == index['NO_MPS', 'secondary']['checkpoint_sha256'] and comparison['NO_MPS_best_and_primary_coincide'])
    check('negative ARl difference retained', next(row for row in differences if row['endpoint_role'] == 'primary'
          and row['comparison'] == 'FULL-NO_MPS' and row['metric'] == 'ARl')['difference_percentage_points'] < 0)

    no_msca_folder = Path(progress['backup_root']) / 'epochs/epoch0028'
    verify_artifact('NO_MSCA', 'progress_only', no_msca_folder / 'next_epoch.pt', round29['rolling_checkpoint_sha256'])
    partial = progress['endpoints']['best']
    for name, key in [('best.pt', 'checkpoint_sha256'), ('best.metrics.json', 'metrics_sha256'),
                      ('best.predictions.json', 'prediction_sha256'), ('best.evaluation_arrays.npz', 'evaluation_arrays_sha256')]:
        verify_artifact('NO_MSCA', 'progress_only', no_msca_folder / 'selected/epoch0028' / name, partial[key])
    progress_record = {
        'arm': 'NO_MSCA', 'status': '仅已核验进度', 'completed_rounds': 29, 'epoch_index_0based': 28,
        'target_rounds': 320, 'remaining_rounds': 291, 'optimizer_step_attempts': 46893, 'seed': 0,
        'trainable_parameters': shared['arms']['NO_MSCA']['trainable_parameters'],
        'latest_metrics_fraction': progress['latest_metrics'],
        'latest_metrics_percent': {m: progress['latest_metrics'][m] * 100 for m in METRICS},
        'rolling_checkpoint_sha256': round29['rolling_checkpoint_sha256'],
        'latest_best_checkpoint_sha256': partial['checkpoint_sha256'], 'ema_tensor_sha256': partial['ema_tensor_sha256'],
        'primary_endpoint': None, 'secondary_endpoint_after_320_rounds': None,
        'protocol_sha256': hashes['NO_MSCA_PROTOCOL'], 'source_ids': ['NO_MSCA_ROUND29', 'NO_MSCA_LINEAGE'],
    }
    limitations = [
        '所有已完成配置均为单个 seed 0，仅提供已观测数值，不作显著性、方差或多种子推断。',
        'BASE 在第214和308轮后续训，FULL 在第1轮后续训。最初的 BASE/FULL 锚点缺少主进程随机数状态。',
        'FULL 的历史第1轮缺少逐批学习率日志，后续记录的515823次更新已由来源核验。',
        'NO_MPS 从第0索引开始运行。相同冻结配方、输入流和初始共享张量不能建立完全相同的随机轨迹。',
        '采用本地转换后的 COCO bbox 验证口径，548张验证图像，maxDets为1/10/100，每张保留top-300预测。',
        '主端点为实际第320轮 EMA 评估状态，记录于晚期验证依赖回滚之前。次端点为同一验证集选出的最高AP单轮。',
        'NO_MPS 是其他FULL设置存在时的stem替换，容量也随替换改变。BASE并非两模块联合移除，不作析因交互推断。',
        'NO_MSCA本快照仅核验29轮，320轮终点为空，不参与最终端点差值。',
    ]
    recipe = {
        'budget': pair['shared_conditions']['budget'], 'evaluation': pair['shared_conditions']['evaluation'],
        'preserved_native_behavior': parent_protocol['preserved_native_behavior'],
        'dataset_manifest_sha256': pair['shared_conditions']['dataset_manifest_sha256'],
        'dataset_archive_sha256': pair['shared_conditions']['dataset_archive_sha256'],
        'arm_architectures': {'BASE': '冻结BASE骨干定义', 'FULL': 'EnhancedStem与MSCA完整配置',
                             'NO_MPS': shared['arms']['NO_MPS']['phenotype'], 'NO_MSCA': shared['arms']['NO_MSCA']['phenotype']},
    }
    for item in SOURCES:
        check(item['source_id'] + ' unchanged during collection', sha(Path(item['path'])) == item['sha256'])
    summary = {
        'status': 'COLLECTED_FROM_VERIFIED_LOCAL_RECORDS', 'snapshot_label': '20261108',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'metric_order': METRICS, 'metric_fraction_unit': '0至1原始比例', 'metric_percent_unit': '原始比例乘100，单位%',
        'difference_unit': '百分点，100乘以两原始比例之差', 'relative_percentage_change_computed': False,
        'completed_endpoints': records, 'differences': differences, 'NO_MSCA_progress': progress_record,
        'recipe_and_evaluation': recipe, 'limitations': limitations, 'input_sources': SOURCES,
        'source_metadata_notes': [
            'BASE/FULL总报告的lineages子报告中full_pair_complete为false。按总报告result_generated=true及各320轮核验收录，保留来源原值。',
        ],
    }
    write_json('实验汇总.json', summary)
    metric_rows = []
    info_rows = []
    for record in records:
        info = {key: value for key, value in record.items() if not key.startswith('metrics_')}
        info_rows.append(info)
        for metric in METRICS:
            metric_rows.append({'arm': record['arm'], 'endpoint_role': record['endpoint_role'],
                'round_1based': record['round_1based'], 'epoch_index_0based': record['epoch_index_0based'],
                'metric': metric, 'value_fraction': record['metrics_fraction'][metric],
                'value_percent': record['metrics_percent'][metric], 'source_id': record['source_id'],
                'source_json_pointer': record['source_json_pointer'] + '/metrics/' + metric})
    write_csv('终点指标.csv', list(metric_rows[0]), metric_rows)
    write_csv('终点信息.csv', list(info_rows[0]), info_rows)
    write_csv('FULL差值.csv', list(differences[0]), differences)
    progress_rows = [{'arm': 'NO_MSCA', 'status': 'progress_only', 'completed_rounds': 29, 'target_rounds': 320,
                      'metric': metric, 'latest_value_fraction': progress_record['latest_metrics_fraction'][metric],
                      'latest_value_percent': progress_record['latest_metrics_percent'][metric],
                      'round320_primary_value': None, 'round320_secondary_value': None,
                      'source_ids': 'NO_MSCA_ROUND29,NO_MSCA_LINEAGE'} for metric in METRICS]
    write_csv('NO_MSCA进度.csv', list(progress_rows[0]), progress_rows)
    write_csv('输入来源.csv', list(SOURCES[0]), SOURCES)
    write_json('输入核查.json', {
        'status': 'PASSED', 'check_count': len(CHECKS), 'checks': CHECKS,
        'artifact_file_hashes': ARTIFACTS, 'model_forwards': 0, 'training_updates': 0,
        'evaluator_runs': 0, 'remote_connections': 0, 'new_statistical_inference': False,
    })
    print(json.dumps({'status': 'COLLECTED', 'endpoint_records': len(records), 'metric_rows': len(metric_rows),
                      'source_count': len(SOURCES), 'artifact_bindings': len(ARTIFACTS), 'checks': len(CHECKS)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
