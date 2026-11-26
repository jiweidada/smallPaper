"""Inventory the original project's saved runs without executing models or evaluators."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
OUTPUTS = Path(r'C:\Users\jiwei\Desktop\DEIM\outputs')
REFERENCE = REC / 'source_snapshot/N4/args.json'
REPORT = HERE / 'existing_outputs_inventory.json'
MISSING = '<missing>'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def get(d, *path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return MISSING
        d = d[key]
    return d

def selected(args):
    cfg = args.get('yaml_cfg', args)
    model = get(cfg, 'DEIM', 'backbone')
    backbone = cfg.get(model, {}) if isinstance(model, str) else {}
    pick = lambda key: args.get(key, cfg.get(key, MISSING))
    train, val = cfg.get('train_dataloader', {}), cfg.get('val_dataloader', {})
    return {'backbone_class': model, 'backbone_name': get(backbone, 'name'),
            'stem_type': get(backbone, 'stem_type'), 'aggregation': get(backbone, 'agg'),
            'backbone_activation': get(backbone, 'act_type'), 'pretrained': get(backbone, 'pretrained'),
            'return_idx': get(backbone, 'return_idx'), 'freeze_at': get(backbone, 'freeze_at'),
            'freeze_norm': get(backbone, 'freeze_norm'), 'seed': pick('seed'),
            'epochs_configured': pick('epoches'), 'resume': pick('resume'), 'tuning': pick('tuning'),
            'use_amp': pick('use_amp'), 'train_batch': train.get('total_batch_size', train.get('batch_size', MISSING)),
            'val_batch': val.get('total_batch_size', val.get('batch_size', MISSING)),
            'train_drop_last': get(train, 'drop_last'), 'train_ann': get(train, 'dataset', 'ann_file'),
            'val_ann': get(val, 'dataset', 'ann_file'), 'optimizer': cfg.get('optimizer', MISSING),
            'train_transforms': get(train, 'dataset', 'transforms'),
            'val_transforms': get(val, 'dataset', 'transforms'),
            'train_collate': get(train, 'collate_fn'), 'val_collate': get(val, 'collate_fn'),
            'lrsheduler': pick('lrsheduler'), 'lr_gamma': pick('lr_gamma'), 'warmup_iter': pick('warmup_iter'),
            'flat_epoch': pick('flat_epoch'), 'no_aug_epoch': pick('no_aug_epoch'), 'use_ema': pick('use_ema')}

def main():
    assert OUTPUTS.is_dir() and not REPORT.exists()
    n4_args = json.loads(REFERENCE.read_text(encoding='utf-8'))
    reference = selected(n4_args)
    records = []
    for folder in sorted(OUTPUTS.iterdir(), key=lambda p: p.name.casefold()):
        if not folder.is_dir():
            continue
        args_path, log_path = folder / 'args.json', folder / 'log.txt'
        item = {'directory': str(folder), 'name': folder.name,
                'args_present': args_path.is_file(), 'log_present': log_path.is_file(),
                'checkpoint_presence': {name: (folder / name).is_file() for name in ['best_stg1.pth', 'best_stg2.pth', 'last.pth']}}
        if args_path.is_file():
            item['args_sha256'] = sha(args_path)
            try:
                args = json.loads(args_path.read_text(encoding='utf-8-sig'))
                current = selected(args)
                item['selected_config'] = current
                item['differences_from_N4'] = {k: {'N4': reference[k], 'run': current[k]}
                                                for k in current if current[k] != reference[k]}
                item['candidate_original_n_host'] = current['backbone_class'] == 'HGNetv2' and current['backbone_name'] == 'B0'
            except (OSError, ValueError, TypeError, KeyError) as exc:
                item['args_error'] = type(exc).__name__ + ': ' + str(exc)
        if log_path.is_file():
            item['log_sha256'] = sha(log_path)
            rows, errors = [], []
            try:
                with log_path.open(encoding='utf-8-sig') as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                rows.append(row)
                            else:
                                errors.append(line_number)
                        except ValueError:
                            errors.append(line_number)
                epochs = [row['epoch'] for row in rows if isinstance(row.get('epoch'), int)]
                item['log'] = {'rows': len(rows), 'parse_error_lines': errors[:20], 'parse_error_count': len(errors),
                               'epoch_min': min(epochs) if epochs else None, 'epoch_max': max(epochs) if epochs else None,
                               'unique_epoch_count': len(set(epochs)),
                               'all_epochs_monotone_strict': all(a < b for a, b in zip(epochs, epochs[1:]))}
            except (OSError, UnicodeError) as exc:
                item['log_error'] = type(exc).__name__ + ': ' + str(exc)
        records.append(item)
    report = {'status': 'TOP_LEVEL_ORIGINAL_OUTPUTS_INVENTORIED',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'scope': str(OUTPUTS),
              'reference_N4_args': {'path': str(REFERENCE), 'sha256': sha(REFERENCE)},
              'directory_count': len(records), 'runs': records,
              'limits': ['Only immediate output directories and their directly contained args.json/log.txt/checkpoint names were inspected.',
                         'Declared fields, labels or matching names do not prove historical source, initialization, data bytes, or a paired valid endpoint.',
                         'An absent args file or unavailable artifact leaves its conditions unknown.',
                         'No accuracy comparison is made and no historical checkpoint is executed.'],
              'model_instantiations': 0, 'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
    with REPORT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    for item in records:
        cfg, log = item.get('selected_config', {}), item.get('log', {})
        print(json.dumps({'name': item['name'], 'original_n_host': item.get('candidate_original_n_host'),
                          'backbone_class': cfg.get('backbone_class'), 'backbone_name': cfg.get('backbone_name'),
                          'epochs': cfg.get('epochs_configured'), 'rows': log.get('rows'),
                          'epoch_range': [log.get('epoch_min'), log.get('epoch_max')],
                          'train_batch': cfg.get('train_batch'), 'val_batch': cfg.get('val_batch'),
                          'seed': cfg.get('seed'), 'train_ann': cfg.get('train_ann'),
                          'differences_from_N4': list(item.get('differences_from_N4', {}).keys()),
                          'errors': [item.get('args_error'), item.get('log_error')]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
