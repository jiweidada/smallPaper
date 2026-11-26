"""Statically cross-check selected epoch semantics, without importing a model."""
from hashlib import sha256
from pathlib import Path
import ast
import json

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / 'fair_runner.py'
STAGER = HERE / 'epoch_evidence_stager.py'
OUT = HERE / 'stager_static_binding_audit_v2.json'
PRIOR = HERE / 'stager_static_binding_audit.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    source = RUNNER.read_text(encoding='utf-8-sig')
    assert digest(RUNNER) == 'cf054375eb10d9522f6e638fd6e85d985adf80cec8ae359eacf9dd2af3f9bef5'
    tree = ast.parse(source)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train_run')
    epoch_loops = [x for x in ast.walk(fn) if isinstance(x, ast.For) and isinstance(x.target, ast.Name)
                   and x.target.id == 'epoch' and ast.unparse(x.iter) == 'range(320)']
    assert len(epoch_loops) == 1
    loop = epoch_loops[0]
    boundary = [x for x in ast.walk(loop) if isinstance(x, ast.If) and ast.unparse(x.test) == 'epoch == 296']
    assert len(boundary) == 1 and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and
                                      x.func.id == 'restore' for x in ast.walk(boundary[0]))
    best = [x for x in ast.walk(loop) if isinstance(x, ast.If) and ast.unparse(x.test) == "metrics['AP'] > best_ap"]
    assert len(best) == 1 and len(best[0].orelse) == 1 and isinstance(best[0].orelse[0], ast.If)
    assert ast.unparse(best[0].orelse[0].test) == 'epoch >= 296'
    assert any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == 'restore'
               for x in ast.walk(best[0].orelse[0]))
    assignments = [ast.unparse(x) for x in best[0].body if isinstance(x, ast.Assign)]
    assert "best_ap = metrics['AP']" in assignments and 'best_epoch = epoch' in assignments
    assert 'epoch_step=1617' in source and 'attempts==517440' in source
    state_fn = next(x for x in fn.body if isinstance(x, ast.Try) for x in x.body
                    if isinstance(x, ast.FunctionDef) and x.name == 'state')
    ret = next(x for x in state_fn.body if isinstance(x, ast.Return))
    keys = {ast.literal_eval(k) for k in ret.value.keys}
    assert {'model', 'ema', 'optimizer', 'lr_warmup_scheduler', 'last_epoch'} <= keys and 'rng_state' not in keys
    stager_tree = ast.parse(STAGER.read_text(encoding='utf-8-sig'))
    finish = next(x for x in stager_tree.body if isinstance(x, ast.FunctionDef) and x.name == 'finish')
    statements = [ast.unparse(x) for x in finish.body]
    assert "improved = before['best_ap'] is None or handle['metrics']['AP'] > before['best_ap']" in statements
    assert any('restored == (epoch >= 296 and (not improved))' in x for x in statements)
    assert any("after_control['best_epoch'] == epoch and after_control['best_ap'] == handle['metrics']['AP']" in x
               and "after_control['best_epoch'] == before['best_epoch']" in x for x in statements)
    guard_tree = ast.parse((HERE / 'epoch_backup_guard.py').read_text(encoding='utf-8-sig'))
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in guard_tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('EPOCHS', 'BATCHES')}
    assert constants == {'EPOCHS': 320, 'BATCHES': 1617}
    imports = []
    for item in ast.walk(stager_tree):
        if isinstance(item, ast.Import):
            imports.extend(x.name.split('.')[0] for x in item.names)
        elif isinstance(item, ast.ImportFrom):
            imports.append((item.module or '').split('.')[0])
    assert not {'torch', 'numpy', 'paramiko', 'socket', 'subprocess'} & set(imports)
    assert PRIOR.is_file()
    report = {'status': 'SELECTED_SOURCE_BRANCHES_STATICALLY_MATCHED', 'previous_audit_sha256': digest(PRIOR),
              'original_runner_sha256': digest(RUNNER), 'stager_sha256': digest(STAGER),
              'checks': {'epochs_320': True, 'batches_1617': True, 'stage_entry_restore_at_296': True,
                         'best_selection_strict_AP_increase': True, 'nonbest_postevaluation_restore_at_or_after_296': True,
                         'stager_selection_and_restore_predicates_match': True, 'guard_epoch_and_batch_constants_match': True,
                         'old_state_lacks_rng_field': True, 'stager_imports_no_model_or_network_libraries': True},
              'limits': ['The first audit checked original branches but did not itself compare the stager predicates; this revision adds that check.',
                         'Selected static branches only; no training-driver integration or runtime equivalence is established.',
                         'No model, inference, evaluation, optimizer update or remote connection was performed.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'sha256': digest(OUT)}))


if __name__ == '__main__':
    main()
