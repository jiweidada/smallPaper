"""Inventory accessible saved args files outside the original DEIM outputs.

Only files returned by the bounded ripgrep search and adjacent log metadata are
read. This script does not run any model, predictor, evaluator or training.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REC = HERE.parent
ROOT = REC.parent.parent
OUT = HERE / 'archived_args_inventory.json'
ERRORS = HERE / 'rg_scan_stderr.txt'
REFERENCE = REC / 'source_snapshot/N4/args.json'
SCOPES = ['04_实验记录', '05_代码与数据', '90_历史与工具']
EXCLUSIONS = ['!**/runtime*/**', '!**/local_deps/**', '!**/build_tools/**']
MISSING = '<missing>'


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def select(args):
    cfg = args.get('yaml_cfg', args)
    if not isinstance(cfg, dict):
        raise TypeError('yaml_cfg is not a dictionary')
    backbone_type = get(cfg, 'DEIM', 'backbone')
    backbone = cfg.get(backbone_type, {}) if isinstance(backbone_type, str) else {}
    train, val = cfg.get('train_dataloader', {}), cfg.get('val_dataloader', {})
    pick = lambda key: args.get(key, cfg.get(key, MISSING))
    return {'backbone_class': backbone_type, 'backbone_name': get(backbone, 'name'),
            'aggregation': get(backbone, 'agg'), 'stem_type': get(backbone, 'stem_type'),
            'backbone_activation': get(backbone, 'act_type'), 'return_idx': get(backbone, 'return_idx'),
            'encoder_hidden_dim': get(cfg, 'HybridEncoder', 'hidden_dim'),
            'epochs_configured': pick('epoches'), 'seed': pick('seed'), 'resume': pick('resume'),
            'tuning': pick('tuning'), 'use_amp': pick('use_amp'), 'use_ema': pick('use_ema'),
            'train_batch': train.get('total_batch_size', train.get('batch_size', MISSING)),
            'val_batch': val.get('total_batch_size', val.get('batch_size', MISSING)),
            'train_ann': get(train, 'dataset', 'ann_file'), 'val_ann': get(val, 'dataset', 'ann_file'),
            'lrsheduler': pick('lrsheduler'), 'lr_gamma': pick('lr_gamma'),
            'warmup_iter': pick('warmup_iter'), 'flat_epoch': pick('flat_epoch'),
            'no_aug_epoch': pick('no_aug_epoch')}


def log_summary(path):
    epochs, parse_errors, nonempty = [], [], 0
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            nonempty += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    parse_errors.append(line_number)
                elif isinstance(row.get('epoch'), int) and not isinstance(row['epoch'], bool):
                    epochs.append(row['epoch'])
            except ValueError:
                parse_errors.append(line_number)
    return {'path': str(path), 'sha256': digest(path), 'nonempty_lines': nonempty,
            'epoch_rows': len(epochs), 'min_epoch': min(epochs) if epochs else None,
            'max_epoch': max(epochs) if epochs else None, 'unique_epochs': len(set(epochs)),
            'contains_all_0_319': set(range(320)).issubset(epochs),
            'parse_error_count': len(parse_errors), 'first_parse_error_lines': parse_errors[:10]}


def main():
    assert not OUT.exists() and not ERRORS.exists()
    exe = shutil.which('rg')
    assert exe
    args = [exe, '--files', *SCOPES, '-g', 'args.json']
    for exclusion in EXCLUSIONS:
        args.extend(['-g', exclusion])
    result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace', check=False)
    assert result.returncode in (0, 1, 2)
    ERRORS.write_text(result.stderr, encoding='utf-8')
    files = sorted({ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()}, key=str)
    reference = select(json.loads(REFERENCE.read_text(encoding='utf-8-sig')))
    records = []
    for path in files:
        item = {'args_path': str(path)}
        try:
            item['args_sha256'] = digest(path)
            parsed = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(parsed, dict):
                raise TypeError('args JSON is not an object')
            selected = select(parsed)
            item['selected_config'] = selected
            item['differences_from_N4'] = {k: {'reference': reference[k], 'candidate': v}
                                           for k, v in selected.items() if v != reference[k]}
            item['coarse_320_batch4_fp32'] = (selected['epochs_configured'] == 320 and
                                               selected['train_batch'] == 4 and selected['use_amp'] is False)
            item['coarse_original_b0'] = selected['backbone_class'] == 'HGNetv2' and selected['backbone_name'] == 'B0'
        except (OSError, ValueError, TypeError, KeyError) as exc:
            item['args_error'] = type(exc).__name__ + ': ' + str(exc)
        log = path.parent / 'log.txt'
        item['adjacent_log_present'] = log.is_file()
        if item['adjacent_log_present']:
            try:
                item['adjacent_log'] = log_summary(log)
            except (OSError, UnicodeError) as exc:
                item['log_error'] = type(exc).__name__ + ': ' + str(exc)
        try:
            item['checkpoint_names_present'] = sorted(p.name for p in path.parent.iterdir()
                                                       if p.is_file() and p.suffix == '.pth' and
                                                       (p.name in ['best_stg1.pth', 'best_stg2.pth', 'last.pth'] or p.name.startswith('checkpoint')))
        except OSError as exc:
            item['directory_error'] = type(exc).__name__ + ': ' + str(exc)
        records.append(item)
    errored = [item for item in records if 'args_error' in item]
    parsed = [item for item in records if 'selected_config' in item]
    candidates = [item for item in parsed if item['coarse_320_batch4_fp32']]
    result_obj = {'status': 'ACCESSIBLE_ARCHIVED_ARGS_INVENTORIED', 'checked_utc': datetime.now(timezone.utc).isoformat(),
                  'search_roots': [str(ROOT / scope) for scope in SCOPES],
                  'file_name_glob': 'args.json', 'excluded_path_globs': EXCLUSIONS,
                  'rg_exit_code': result.returncode, 'rg_stderr_path': str(ERRORS), 'rg_stderr_sha256': digest(ERRORS),
                  'rg_error_line_count': len(result.stderr.splitlines()), 'files_returned': len(files),
                  'args_parsed': len(parsed), 'args_errors': len(errored),
                  'distinct_args_sha256': len({item['args_sha256'] for item in records if 'args_sha256' in item}),
                  'with_adjacent_log': sum(item['adjacent_log_present'] for item in records),
                  'configured_epoch_counts': dict(Counter(str(item['selected_config']['epochs_configured']) for item in parsed)),
                  'coarse_320_batch4_fp32_count': len(candidates), 'reference_N4': {'path': str(REFERENCE), 'sha256': digest(REFERENCE), 'selected': reference},
                  'records': records,
                  'limits': ['Only accessible non-hidden args.json files returned by the bounded rg search were parsed; environment directories were excluded and access errors remain.',
                             'Files inside compressed archives, other filenames, remote systems and experiments lacking args.json were not searched.',
                             'Selected config fields and adjacent log presence are coarse screens, not proof of same data bytes, source, initialization, checkpoint or evaluator.',
                             'This search makes no accuracy comparison and does not instantiate or execute models.'],
                  'model_instantiations': 0, 'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result_obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result_obj['status'], 'rg_exit_code': result.returncode,
                      'rg_error_line_count': result_obj['rg_error_line_count'], 'files_returned': len(files),
                      'args_parsed': len(parsed), 'args_errors': len(errored), 'with_adjacent_log': result_obj['with_adjacent_log'],
                      'configured_epoch_counts': result_obj['configured_epoch_counts'],
                      'coarse_320_batch4_fp32_count': len(candidates),
                      'candidates': [{'path': x['args_path'], 'selected': x['selected_config'],
                                      'log': x.get('adjacent_log')} for x in candidates],
                      'audit_sha256': digest(OUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
