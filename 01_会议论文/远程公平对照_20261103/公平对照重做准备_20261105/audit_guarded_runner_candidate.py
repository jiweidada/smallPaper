"""Independently inspect a disabled candidate without importing or executing it."""
from hashlib import sha256
from pathlib import Path
import ast
import json

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'fair_runner.py'
BUILDER = HERE / 'build_guarded_runner_candidate.py'
CANDIDATE = HERE / 'guarded_fair_runner_candidate_v2.py.txt'
PRIOR = HERE / 'guarded_fair_runner_candidate.py.txt'
BUILD_REPORT = HERE / 'guarded_runner_candidate_build_v2.json'
OUT = HERE / 'guarded_runner_static_audit.json'


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def call_name(node):
    if isinstance(node, ast.Expr):
        node = node.value
    if isinstance(node, ast.Assign):
        node = node.value
    return ast.unparse(node.func) if isinstance(node, ast.Call) else None


def train(tree):
    return next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train_run')


def main():
    assert not OUT.exists()
    report = json.loads(BUILD_REPORT.read_text(encoding='utf-8'))
    assert report['status'] == 'DISABLED_REVIEW_ONLY_CANDIDATE_CREATED'
    assert report['candidate_sha256'] == sha(CANDIDATE) and report['source_sha256'] == sha(SOURCE)
    builder_tree = ast.parse(BUILDER.read_text(encoding='utf-8'))
    header = ast.literal_eval(next(x.value for x in builder_tree.body if isinstance(x, ast.Assign)
                                   and any(isinstance(t, ast.Name) and t.id == 'HEADER' for t in x.targets)))
    build_main = next(x for x in builder_tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
    patches = []
    for node in build_main.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'add':
            patches.append(tuple(ast.literal_eval(arg) for arg in node.value.args))
    assert len(patches) == 10
    candidate = CANDIDATE.read_text(encoding='utf-8')
    recovered = candidate
    for label, old, new in reversed(patches):
        assert recovered.count(new) == 1, label
        recovered = recovered.replace(new, old, 1)
    source = SOURCE.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    assert recovered == header + source
    tree = ast.parse(candidate)
    auth = tree.body[0]
    assert isinstance(auth, ast.FunctionDef) and auth.name == '_evidence_require_authorization'
    assert len(auth.body) == 1 and isinstance(auth.body[0], ast.Raise)
    top_guard = tree.body[1]
    assert isinstance(top_guard, ast.If) and ast.unparse(top_guard.test) == "__name__ == '__main__'"
    assert len(top_guard.body) == 1 and call_name(top_guard.body[0]) == '_evidence_require_authorization'
    for name in ('preflight', 'train_run'):
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
        assert call_name(fn.body[0]) == '_evidence_require_authorization'
    fn = train(tree)
    trial = next(x for x in fn.body if isinstance(x, ast.Try))
    assert any(isinstance(x, ast.Raise) for handler in trial.handlers for x in handler.body)
    imports = [i for i, x in enumerate(trial.body) if isinstance(x, ast.Import) and
               any(y.name in ('epoch_evidence_stager', 'epoch_backup_barrier') for y in x.names)]
    seeds = [i for i, x in enumerate(trial.body) if call_name(x) == 'seeds']
    assert len(imports) == 2 and len(seeds) == 1 and max(imports) < seeds[0]
    previous_fn = train(ast.parse(PRIOR.read_text(encoding='utf-8')))
    previous_try = next(x for x in previous_fn.body if isinstance(x, ast.Try))
    old_imports = [i for i, x in enumerate(previous_try.body) if isinstance(x, ast.Import) and
                   any(y.name in ('epoch_evidence_stager', 'epoch_backup_barrier') for y in x.names)]
    old_seeds = [i for i, x in enumerate(previous_try.body) if call_name(x) == 'seeds']
    assert len(old_imports) == 2 and len(old_seeds) == 1 and old_seeds[0] < min(old_imports)
    state_fn = next(x for x in trial.body if isinstance(x, ast.FunctionDef) and x.name == 'state')
    state_return = next(x for x in state_fn.body if isinstance(x, ast.Return))
    assert 'rng_state' in {ast.literal_eval(k) for k in state_return.value.keys}
    loop = next(x for x in ast.walk(fn) if isinstance(x, ast.For) and isinstance(x.target, ast.Name)
                and x.target.id == 'epoch' and ast.unparse(x.iter) == 'range(320)')
    names = [call_name(x) for x in loop.body]
    eval_i, begin_i = names.index('evaluate'), names.index('_evidence_stager.begin')
    finish_i, wait_i = names.index('_evidence_stager.finish'), names.index('_evidence_barrier.await_ack')
    best_i = next(i for i, x in enumerate(loop.body) if isinstance(x, ast.If) and
                  ast.unparse(x.test) == "metrics['AP'] > best_ap")
    completed_i = next(i for i, x in enumerate(loop.body) if isinstance(x, ast.Expr) and
                       'epoch_complete' in ast.unparse(x) and call_name(x) == 'receipt.update')
    assert eval_i < begin_i < best_i < finish_i < wait_i < completed_i
    best = loop.body[best_i]
    assert len(best.orelse) == 1 and isinstance(best.orelse[0], ast.If)
    nonbest = best.orelse[0]
    assert ast.unparse(nonbest.test) == 'epoch >= 296'
    restore_i = next(i for i, x in enumerate(nonbest.body) if call_name(x) == 'restore')
    marker_i = next(i for i, x in enumerate(nonbest.body) if isinstance(x, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == '_evidence_restored' for t in x.targets) and
                    isinstance(x.value, ast.Constant) and x.value.value is True)
    assert restore_i < marker_i
    step_loop = next(x for x in loop.body if isinstance(x, ast.For) and isinstance(x.target, ast.Tuple))
    step_calls = [call_name(x) for x in step_loop.body]
    assert step_calls.index('_evidence_lr_rows.append') < step_calls.index('lrlog.write')
    result = {'status': 'DISABLED_CANDIDATE_STATICALLY_REVIEWED', 'candidate_sha256': sha(CANDIDATE),
              'source_sha256': sha(SOURCE), 'builder_sha256': sha(BUILDER), 'previous_candidate_sha256': sha(PRIOR),
              'checks': {'prior_training_text_recovered_by_inverse_patches': True,
                         'main_preflight_and_train_unconditionally_disabled': True,
                         'new_imports_before_final_seed_reset': True, 'state_captures_rng': True,
                         'evaluation_then_stage_then_restore_then_finish_then_ack': True,
                         'ack_before_epoch_complete': True, 'learning_rates_recorded_before_original_log': True,
                         'exceptions_propagate_out_of_training': True},
              'previous_candidate_issue': 'Its added imports followed the final seed reset; actual RNG impact was not tested. The candidate is disabled.',
              'limits': ['Static ordering and reversible source edits do not establish runtime or numerical equivalence.',
                         'Candidate and preflight were not executed. New authorization, real integration and server validation remain required.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'candidate_sha256': result['candidate_sha256'], 'sha256': sha(OUT)}))


if __name__ == '__main__':
    main()
