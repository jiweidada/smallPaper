"""Audit retained configuration layers and inspected training-source semantics.

This reads JSON and Python source only; it does not instantiate a model or
execute training, inference, or an evaluator.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import ast
import json
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
CODE = Path(r'C:\Users\jiwei\Desktop\DEIM')
CPU_SNAPSHOT = PAPER / '本地CPU核验_20261103/source_snapshot'
OUT = HERE / 'training_parameters_audit.json'
FILES = [
    'engine/core/yaml_config.py', 'engine/core/workspace.py', 'engine/core/yaml_utils.py',
    'engine/core/_config.py', 'engine/solver/_solver.py', 'engine/solver/det_solver.py',
    'engine/solver/det_engine.py', 'engine/optim/ema.py', 'engine/optim/lr_scheduler.py',
    'engine/optim/warmup.py',
]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    sources = {}
    for relative in FILES:
        current = CODE / relative
        archived = CPU_SNAPSHOT / relative
        snapshot = HERE / 'source_snapshot' / relative
        assert current.is_file() and archived.is_file() and digest(current) == digest(archived)
        text = current.read_text(encoding='utf-8')
        ast.parse(text)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            assert digest(current) == digest(snapshot)
        else:
            shutil.copy2(current, snapshot)
        assert digest(current) == digest(snapshot)
        sources[relative] = {'path': str(current), 'cpu_snapshot': str(archived),
                             'round_snapshot': str(snapshot), 'sha256': digest(current)}

    code = {path: (CODE / path).read_text(encoding='utf-8') for path in FILES}
    yaml_config = code['engine/core/yaml_config.py']
    workspace = code['engine/core/workspace.py']
    base_config = code['engine/core/_config.py']
    ema = code['engine/optim/ema.py']
    solver = code['engine/solver/det_solver.py']
    trainer = code['engine/solver/det_engine.py']
    lr = code['engine/optim/lr_scheduler.py']
    warmup = code['engine/optim/warmup.py']
    checks = {
        'yaml_ema_created_from_nested_config': "create('ema', self.global_cfg, model=self.model)" in yaml_config,
        'global_cfg_preserves_yaml_values': 'merge_config(self.yaml_cfg, inplace=False, overwrite=False)' in yaml_config,
        'factory_overrides_defaults_with_nested_arguments': workspace.index("_cfg.update(_cfg['_kwargs'])") < workspace.index('_cfg.update(cfg)') < workspace.index('_cfg.update(kwargs)'),
        'base_ema_fallback_uses_outer_attributes': 'ModelEMA(self.model, self.ema_decay, self.ema_warmups)' in base_config,
        'solver_uses_cfg_ema': 'self.ema = self.to(cfg.ema, device)' in code['engine/solver/_solver.py'],
        'ema_exponential_ramp_and_update': 'decay * (1 - math.exp(-x / warmups))' in ema and 'd = self.decay_fn(self.updates)' in ema,
        'ema_state_saves_module_and_updates_only': 'return dict(module=self.module.state_dict(), updates=self.updates)' in ema,
        'trainer_clips_gradient_norm': 'max_norm=args.clip_max_norm' in solver and 'torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)' in trainer,
        'solver_selects_custom_scheduler': 'if args.lrsheduler is not None:' in solver and 'FlatCosineLRScheduler(' in solver,
        'custom_scheduler_called_after_optimizer_step': trainer.index('optimizer.step()') < trainer.index('lr_scheduler.step(cur_iters + i, optimizer)'),
        'gamma_sets_min_lr_and_constant_after_warmup': 'base_lr * lr_gamma' in lr and 'return init_lr * (current_iter / float(warmup_iter)) ** 2' in lr and 'min_lr + (init_lr - min_lr) * cosine_decay' in lr,
        'separate_linear_warmup_initialized': "create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)" in yaml_config and 'self.step()' in warmup and '(step + 1) / self.warmup_duration' in warmup,
    }
    assert all(checks.values()), checks
    runs = []
    for run in ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']:
        path = REC / 'source_snapshot' / run / 'args.json'
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        cfg = data['yaml_cfg']
        observed = {'use_amp': cfg['use_amp'], 'use_ema': cfg['use_ema'],
                    'ema_type': cfg['ema']['type'], 'ema_decay': cfg['ema']['decay'],
                    'ema_warmups': cfg['ema']['warmups'], 'ema_start': cfg['ema']['start'],
                    'outer_ema_warmups': data['ema_warmups'], 'clip_max_norm': cfg['clip_max_norm'],
                    'lrsheduler': cfg['lrsheduler'], 'warmup_iter': cfg['warmup_iter'],
                    'lr_gamma': cfg['lr_gamma'], 'linear_warmup_duration': cfg['lr_warmup_scheduler']['warmup_duration'],
                    'train_batch': cfg['train_dataloader']['total_batch_size'],
                    'ema_restart_decay': cfg['train_dataloader']['collate_fn']['ema_restart_decay']}
        assert observed == {'use_amp': False, 'use_ema': True, 'ema_type': 'ModelEMA', 'ema_decay': 0.9999,
                            'ema_warmups': 1000, 'ema_start': 0, 'outer_ema_warmups': 2000,
                            'clip_max_norm': 0.1, 'lrsheduler': 'flatcosine', 'warmup_iter': 2000,
                            'lr_gamma': 1.0, 'linear_warmup_duration': 500,
                            'train_batch': 16 if run == 'N0' else 4, 'ema_restart_decay': 0.9999}
        runs.append({'run': run, 'args_path': str(path), 'args_sha256': digest(path), 'retained': observed})
    result = {'status': 'SIX_CONFIGS_AND_INSPECTED_PARAMETER_RESOLUTION_CHECKED',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'sources': sources,
              'static_checks': checks, 'runs': runs,
              'interpretation': 'The inspected YAMLConfig creates ModelEMA from nested ema values, so the nested warmups=1000 applies in that code path; the outer BaseConfig ema_warmups=2000 is a fallback value. The EMA ramp is exponential, not a linear ramp ending at 1000 updates. The custom LR schedule uses warmup_iter=2000 and gamma=1; a separate 500-step LinearWarmup object also initializes the optimizer before custom scheduling.',
              'limits': ['Saved configuration and current code do not authenticate the exact historical source version or complete per-update optimizer trajectory.',
                         'The inspected EMA state dictionary saves module and update count, not constructor decay or warmups.',
                         'These settings do not remove the batch-size or architecture differences in historical comparisons.'],
              'historical_source_identity_certified': False,
              'model_instantiations': 0, 'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'runs': len(runs), 'static_checks': len(checks),
                      'audit_sha256': digest(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
