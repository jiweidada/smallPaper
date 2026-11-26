"""CPU-only native import probe; guards prohibit training-related operations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CHILD = r'''
import json,sys,traceback
from pathlib import Path
from experiment import verify_bundle
from runtime import activate_training_snapshot, register_components
from evaluator_binding import activate, loaded_fingerprint
report={'status':'NOT_STARTED','guard_attempts':[], 'mode':'imports_only_no_training'}
try:
    receipt, plan, paths = verify_bundle(sys.argv[1])
    activate_training_snapshot(paths['native_root'])
    import torch
    def forbidden(name):
        def block(*args, **kwargs):
            report['guard_attempts'].append(name)
            raise RuntimeError('Import probe forbids: '+name)
        return block
    torch.load = forbidden('checkpoint_load')
    torch.nn.Parameter.__new__ = staticmethod(forbidden('parameter_construction'))
    torch.nn.Module._call_impl = forbidden('module_forward')
    torch.optim.Optimizer.__init__ = forbidden('optimizer_construction')
    torch.cuda._lazy_init = forbidden('CUDA_initialization')
    report['evaluator'] = activate(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    from engine.core import YAMLConfig
    from engine.solver.det_solver import DetSolver
    from engine.misc import dist_utils
    register_components()
    loaded_fingerprint(receipt['identity']['evaluator_binding'],paths['evaluator_root'])
    from engine.data.dataset import coco_eval
    from pycocoeval.cocoeval import COCOeval
    if coco_eval.COCOeval is not COCOeval:
        raise RuntimeError('Native evaluator uses a different COCOeval class')
    report.update(status='PASS_NATIVE_IMPORTS_ONLY',
        engine_root=str(Path(sys.modules['engine'].__file__).resolve().parent),
        solver_module=str(Path(sys.modules[DetSolver.__module__].__file__).resolve()),
        native_COCOeval_is_bound=True)
    if sys.argv[2] == 'read-configs':
        from engine.core.workspace import GLOBAL_CONFIG
        if 'ResidualGeometryDEIM' not in GLOBAL_CONFIG:
            raise RuntimeError('Candidate class was not registered')
        report['configuration_reads'] = []
        for bundle in sorted(Path(sys.argv[1]).resolve().parent.iterdir()):
            if not bundle.is_dir():
                continue
            item, item_plan, item_paths = verify_bundle(bundle)
            cfg = YAMLConfig(str(bundle/'resolved.yml'))
            expected = json.loads((bundle/'resolved.yml').read_text(encoding='utf-8'))
            if expected.get('use_amp_fallback_fp32',False):
                expected['use_amp'] = True
            if cfg.yaml_cfg != expected or cfg.epoches != 132 or cfg.seed != item_plan['seed']:
                raise RuntimeError('Native configuration parsing changed the prepared recipe')
            lazy = ('_model','_criterion','_postprocessor','_optimizer','_ema','_evaluator',
                    '_train_dataloader','_val_dataloader','_scaler','_writer')
            if any(cfg.__dict__[name] is not None for name in lazy):
                raise RuntimeError('Reading a configuration constructed a training component')
            report['configuration_reads'].append({'arm':item_plan['arm'], 'seed':item_plan['seed'],
                'unchanged_after_native_parse':True, 'training_components_constructed':False})
        if len(report['configuration_reads']) != 6:
            raise RuntimeError('Expected all six prepared configurations')
        report['status'] = 'PASS_NATIVE_IMPORTS_AND_SIX_CONFIGURATION_READS_ONLY'
except Exception:
    report.update(status='FAILED_NATIVE_IMPORTS',traceback=traceback.format_exc())
if 'torch' in sys.modules:
    report['CUDA_initialized']=torch.cuda.is_initialized()
report.update(model_calls=0,optimizer_constructions=0,checkpoint_loads=0,
              parameter_updates=0,training_started=False)
print('NATIVE_IMPORT_REPORT='+json.dumps(report,ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--extra-python-path', action='append', default=[])
    parser.add_argument('--read-configs', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    extra = [str(Path(p).resolve()) for p in args.extra_python_path]
    if extra:
        env['PYTHONPATH'] = os.pathsep.join(extra + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    process = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', CHILD,
                              str(Path(args.bundle).resolve()),
                              'read-configs' if args.read_configs else 'imports-only'], cwd=HERE, env=env,
                             capture_output=True, text=True, encoding='utf-8', timeout=45)
    marker = [line.split('=', 1)[1] for line in process.stdout.splitlines()
              if line.startswith('NATIVE_IMPORT_REPORT=')]
    report = json.loads(marker[-1]) if marker else {'status': 'NO_STRUCTURED_RESULT'}
    report.update(exit_code=process.returncode, stderr=process.stderr, extra_python_path=extra,
                  other_stdout='\n'.join(line for line in process.stdout.splitlines()
                     if not line.startswith('NATIVE_IMPORT_REPORT=')))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
