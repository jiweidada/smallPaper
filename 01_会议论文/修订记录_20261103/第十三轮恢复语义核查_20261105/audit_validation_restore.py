"""Audit saved configuration paths and validation-dependent restoration without executing a model."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import copy
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
CPU = PAPER / '本地CPU核验_20261103'
SOURCE = CPU / 'source_snapshot/engine'
PREV = REC / '第十二轮Stem尺寸核查_20261105'
OUTPUT = HERE / 'validation_restore_static_audit.json'

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def module(path):
    return ast.parse(path.read_text(encoding='utf-8'))

def klass(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)

def method(node, name):
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)

def main():
    assert not OUTPUT.exists()
    solver_path = SOURCE / 'solver/det_solver.py'
    base_path = SOURCE / 'solver/_solver.py'
    engine_path = SOURCE / 'solver/det_engine.py'
    collate_path = SOURCE / 'data/dataloader.py'
    solver = method(klass(module(solver_path), 'DetSolver'), 'fit')
    solver_text = ast.unparse(solver)
    assert "test_stats[k][0] > best_stat[k]" in solver_text
    assert "best_stat[k] = max(best_stat[k], test_stats[k][0])" in solver_text
    assert "dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')" in solver_text
    assert solver_text.count("self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))") == 2
    assert 'self.val_dataloader' in solver_text and 'module = self.ema.module if self.ema else self.model' in solver_text
    ifs = [n for n in ast.walk(solver) if isinstance(n, ast.If)]
    boundary = next(n for n in ifs if ast.unparse(n.test) == 'epoch == self.train_dataloader.collate_fn.stop_epoch')
    assert "self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))" in ast.unparse(boundary)
    late = next(n for n in ifs if ast.unparse(n.test) == 'epoch >= self.train_dataloader.collate_fn.stop_epoch'
                and "self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))" in ast.unparse(n))
    assert 'test_stats[k][0]' in solver_text
    engine = method(klass(module(base_path), 'BaseSolver'), 'load_state_dict')
    base_text = ast.unparse(engine)
    assert "if 'last_epoch' in state" in base_text and "self.last_epoch = state['last_epoch']" in base_text
    evaluate = next(n for n in module(engine_path).body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    assert "stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()" in ast.unparse(evaluate)

    collate = klass(module(collate_path), 'BatchImageCollateFunction')
    init = method(collate, '__init__')
    defaults = dict(zip([x.arg for x in init.args.args[-len(init.args.defaults):]], init.args.defaults))
    assert ast.literal_eval(defaults['base_size_repeat']) is None
    assert 'self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None' in ast.unparse(init)
    configs = []
    for name in ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']:
        path = REC / 'source_snapshot' / name / 'args.json'
        cfg = json.loads(path.read_text(encoding='utf-8'))['yaml_cfg']
        train = cfg['train_dataloader']['collate_fn']
        val = cfg['val_dataloader']['collate_fn']
        assert train['type'] == val['type'] == 'BatchImageCollateFunction'
        assert 'base_size_repeat' in train and train['base_size_repeat'] is None
        assert 'base_size_repeat' not in val  # inspected constructor's default is None
        assert train['stop_epoch'] == 296
        assert cfg['evaluator'] == {'type': 'CocoEvaluator', 'iou_types': ['bbox']}
        mutated = copy.deepcopy(cfg)
        mutated['train_dataloader']['collate_fn']['base_size_repeat'] = 1
        old_check_passes_on_mutated = mutated.get('BatchImageCollateFunction', {}).get('base_size_repeat') is None
        assert old_check_passes_on_mutated
        assert mutated['train_dataloader']['collate_fn']['base_size_repeat'] is not None
        configs.append({'run': name, 'args_sha256': digest(path), 'train_base_size_repeat': train['base_size_repeat'],
                        'val_base_size_repeat': 'omitted; inspected default None', 'stop_epoch': train['stop_epoch'],
                        'evaluator_iou_types': cfg['evaluator']['iou_types']})
    prior_script = PREV / 'audit_stem_input_shape.py'
    assert "cfg.get('BatchImageCollateFunction', {}).get('base_size_repeat') is None" in prior_script.read_text(encoding='utf-8')
    prior_report = PREV / 'stem_shape_static_audit.json'
    assert json.loads(prior_report.read_text(encoding='utf-8'))['status'] == 'STEM_INPUT_SHAPE_STATICALLY_CHECKED_NO_MODEL_EXECUTION'
    evidence = {'status': 'RESTORE_SEMANTICS_AND_COLLATE_PATH_STATICALLY_CHECKED',
                'checked_utc': datetime.now(timezone.utc).isoformat(),
                'sources': [{'path': str(p), 'sha256': digest(p)} for p in [solver_path, base_path, engine_path, collate_path]],
                'source_lines': {'fit': [solver.lineno, solver.end_lineno], 'epoch_boundary_restore': boundary.lineno,
                                 'late_nonbest_restore': late.lineno, 'bbox_stats': [evaluate.lineno, evaluate.end_lineno],
                                 'load_state_dict': [engine.lineno, engine.end_lineno],
                                 'collate_init': [init.lineno, init.end_lineno]},
                'configs': configs,
                'restore_interpretation': 'In the inspected solver, bbox validation AP selects best_stg1 and may lead to restoring it during stage-two training; restoration can reset last_epoch. The exact historical training-source identity is not established.',
                'prior_audit_correction': {'previous_script': str(prior_script), 'sha256': digest(prior_script),
                                           'previous_report': str(prior_report), 'previous_report_sha256': digest(prior_report),
                                           'problem': 'The earlier check queried a missing top-level BatchImageCollateFunction key and therefore would pass regardless of the nested training collator value.',
                                           'synthetic_counterexample_old_check_passes': True,
                                           'corrected_result': 'All six saved training collators explicitly set base_size_repeat to null; all six validation collators omit it and the inspected implementation defaults to null. This confirms the conclusion for the inspected source/configuration, not the historical runtime.'},
                'model_executions': 0, 'evaluator_executions': 0, 'training_updates': 0,
                'historical_solver_identity_certified': False}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': evidence['status'], 'configs_checked': len(configs),
                      'prior_check_path_was_nondiscriminating': True, 'model_executions': 0}))

if __name__ == '__main__':
    main()
