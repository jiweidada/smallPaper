import argparse
import copy
import json
from pathlib import Path
import yaml
from engine.core import YAMLConfig
from screen_runtime import canonical_cfg, file_sha, INIT_SHA


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a)|set(b)):
            p = f'{path}.{k}' if path else k
            if k not in a or k not in b: out.append(p)
            else: out.extend(differences(a[k], b[k], p))
        return out
    return [] if a == b else [path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True); parser.add_argument('--init', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    cfg = YAMLConfig('configs/test/deim_hgnetv2_s_visdrone.yml').yaml_cfg
    cfg.update(epoches=24, flat_epoch=12, no_aug_epoch=4, seed=2, use_amp=True,
               use_amp_fallback_fp32=False, device='cuda:0')
    cfg['HGNetv2']['pretrained'] = False
    cfg['evaluator'].update(using_tiny_metrice=False, using_vtiny_metrice=False, max_dets=100)
    cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'] = [2, 12, 20]
    cfg['train_dataloader']['collate_fn'].update(mixup_epochs=[2, 12], stop_epoch=20)
    for split in ('train', 'val'):
        dl = cfg[f'{split}_dataloader']; dl['total_batch_size'] = 4; dl['num_workers'] = 4
        dl['dataset'].update(img_folder=f'{args.data}/VisDrone2019-DET-{split}/images',
                             ann_file=f'{args.data}/VisDrone2019-DET-{split}/annotations/{split}.json')
    cfg.pop('__include__', None)
    # Both arms keep all original model schemas; only the selected model changes.
    cfg['P2CandidateDEIM'] = dict(cfg['DEIM'], c2_channels=64, p2_budget=60, anchor_wh_px=8.2)
    arms = {}
    for arm in ('base', 'p2_quota60'):
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg['model'] = 'DEIM' if arm == 'base' else 'P2CandidateDEIM'
        arm_cfg['output_dir'] = str(root/arm/'run')
        p = root/f'{arm}.yml'; p.write_text(yaml.safe_dump(arm_cfg, sort_keys=False))
        arms[arm] = arm_cfg
    diff = differences(canonical_cfg(arms['base']), canonical_cfg(arms['p2_quota60']))
    assert diff == ['model'], diff
    assert file_sha(args.init) == INIT_SHA
    data_shas = {s: file_sha(arms['base'][f'{s}_dataloader']['dataset']['ann_file']) for s in ('train','val')}
    assert data_shas == {'train': '32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883',
                         'val': '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'}
    (root/'configuration_alignment.json').write_text(json.dumps({'semantic_differences': diff,
        'arms': {k: canonical_cfg(v) for k,v in arms.items()}, 'data_sha256': data_shas,
        'common_init_sha256': INIT_SHA}, indent=2))
    print(json.dumps({'config_diff': diff, 'data_sha256': data_shas, 'init': INIT_SHA}))


if __name__ == '__main__': main()
