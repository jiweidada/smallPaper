"""Display material metadata differences without loading checkpoints or models."""
from pathlib import Path
import json

ROOT = Path(r'C:\Users\jiwei\Desktop\DEIM\outputs')
NAMES = ['deim_hgnetv2_n_custom160', 'deim_hgnetv2_n_custom320',
         'deim_hgnetv2_n_Simam_3202', 'deim_hgnetv2_MSCA_EnhancedStem_SiLU_320']

def diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from diff(a.get(k, '<missing>'), b.get(k, '<missing>'), path + '/' + str(k))
    elif a != b:
        yield (path, a, b)

def main():
    cfgs = {n: json.loads((ROOT/n/'args.json').read_text(encoding='utf-8-sig')) for n in NAMES}
    for n in NAMES:
        cfg = cfgs[n].get('yaml_cfg', cfgs[n])
        print('\nDIRECTORY', n)
        print('metadata files:', [(p.name, p.stat().st_size) for p in (ROOT/n).iterdir() if p.is_file()])
        print('top-level configuration keys:', sorted(cfg.keys()))
        for key in ['DEIM', 'HybridEncoder', 'DFINETransformer', 'DFINECriterion', 'HGNetv2',
                    'HGNetv2_MSCA_EnhancedStem', 'tuning', 'resume']:
            if key in cfg:
                print(key, json.dumps(cfg[key], ensure_ascii=False))
        epochs = []
        with (ROOT/n/'log.txt').open(encoding='utf-8-sig') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get('epoch'), int):
                        epochs.append(row['epoch'])
                except ValueError:
                    pass
        print('epoch irregularities:', {'missing_in_range': sorted(set(range(min(epochs), max(epochs)+1)) - set(epochs)),
                                          'duplicates': sorted({x for x in epochs if epochs.count(x) > 1}),
                                          'nonincreasing_pairs': [(i, a, b) for i, (a, b) in enumerate(zip(epochs, epochs[1:]), 1) if b <= a]})
    ref = cfgs[NAMES[-1]].get('yaml_cfg', cfgs[NAMES[-1]])
    for n in NAMES[:-1]:
        print('\nCONFIG_DIFF_VS_N4', n)
        cfg = cfgs[n].get('yaml_cfg', cfgs[n])
        for path, a, b in diff(ref, cfg):
            print(json.dumps({'path': path, 'N4': a, 'candidate': b}, ensure_ascii=False))

if __name__ == '__main__':
    main()
