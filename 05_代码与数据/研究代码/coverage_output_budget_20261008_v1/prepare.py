"""Prepare experiment configurations from the actual 132-epoch baseline args."""
from pathlib import Path
from copy import deepcopy
import json, hashlib

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
EVIDENCE = WORKSPACE / '英文小论文/data/00_实验计划与证据索引'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out = HERE / 'configs'
    out.mkdir(exist_ok=False)
    sources = {}
    for seed in (1, 2):
        args_path = EVIDENCE / f'正式132轮基线模型下载_20261008_v1/seed{seed}/run/args.json'
        manifest_path = EVIDENCE / f'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2/seed{seed}/manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert sha(args_path) == manifest['args_sha256_local']
        cfg = json.loads(args_path.read_text(encoding='utf-8'))['yaml_cfg']
        assert cfg['DFINETransformer']['num_queries'] == 300
        assert cfg['train_dataloader']['total_batch_size'] == 4
        assert cfg['epoches'] == 132 and cfg['DEIMCriterion']['losses'] == ['mal', 'boxes', 'local']
        recipe = {k: deepcopy(cfg[k]) for k in ('HGNetv2', 'HybridEncoder', 'DFINETransformer')}
        for arm in ('baseline', 'admission_quality', 'admission_cover', 'output_budget'):
            plan = {'schema': 'coverage_output_budget_v1', 'arm': arm, 'seed': seed,
                    'native_baseline_config': deepcopy(cfg), 'model_recipe': recipe,
                    'criterion_recipe': deepcopy(cfg['DEIMCriterion']),
                    'budget_loss': {'margin': .1, 'weight': .1, 'residual_weight': 1e-4,
                                    'threshold': .5, 'budget': 300},
                    'native_args_sha256': sha(args_path), 'execution_status': 'PREPARED_NOT_TRAINED'}
            (out / f'{arm}_seed{seed}.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        sources[str(args_path)] = sha(args_path)
        for rel, digest in manifest['source_hashes'].items():
            path = HERE.parent / 'local_candidates_20261004_v2' / rel
            assert sha(path) == digest, rel
            sources[str(path)] = digest
    (HERE/'source_binding.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Prepared eight configurations. No model, data transform, or optimizer executed.')

if __name__ == '__main__': main()
