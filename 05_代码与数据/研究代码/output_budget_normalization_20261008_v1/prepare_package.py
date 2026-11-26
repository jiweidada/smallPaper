"""One-time standard-library scaffolding. Never imports model/training code."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'coverage_output_budget_20261008_v1'
ARMS = ('baseline', 'output_image', 'output_gt')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, content):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f'Unexpected source occurrence count: {before!r}')
    return source.replace(before, after)


def main():
    old_files = {str(p.relative_to(OLD)).replace('\\', '/'): sha(p)
                 for p in OLD.rglob('*') if p.is_file()}
    write('prior_package_binding.json', json.dumps(old_files, ensure_ascii=False, indent=2))
    for name in ('source_binding.json', 'training_source_binding.json'):
        write(name, (OLD / name).read_text(encoding='utf-8'))

    runtime = (OLD / 'runtime.py').read_text(encoding='utf-8')
    runtime = runtime.replace('BudgetDEIM', 'BudgetNormDEIM').replace('BudgetCriterion', 'BudgetNormCriterion')
    runtime = replace_once(runtime, "('matching.py', 'losses.py', 'native_adapter.py', 'runtime.py', 'experiment.py')",
                           "('losses.py', 'native_adapter.py', 'runtime.py', 'experiment.py')")
    start = runtime.index("    allowed = {'decoder.admission_weight'}")
    runtime = runtime[:start] + '''    if set(current) != set(state):
        raise RuntimeError('Initialization state keys do not match the native model')
    for key in state:
        if current[key].shape != state[key].shape or current[key].dtype != state[key].dtype:
            raise RuntimeError(f'Initialization tensor mismatch: {key}')
    model.load_state_dict(state, strict=True)
    return {'common_init_sha': COMMON_INIT_SHA, 'native_tensor_count': len(state),
            'new_zero_keys': []}
'''
    write('runtime.py', runtime)

    runner = (OLD / 'experiment.py').read_text(encoding='utf-8')
    runner = runner.replace('BudgetDEIM', 'BudgetNormDEIM').replace('BudgetCriterion', 'BudgetNormCriterion')
    runner = runner.replace('coverage_output_budget_v1', 'output_budget_normalization_v1')
    runner = runner.replace('coverage_output_budget_identity', 'output_budget_normalization_identity')
    runner = replace_once(runner, "('baseline', 'admission_quality', 'admission_cover', 'output_budget')", repr(ARMS))
    runner = replace_once(runner, 'dict(margin=.1, weight=.1, residual_weight=1e-4, threshold=.5, budget=300)',
                           'dict(margin=.1, weight=.1, threshold=.5, budget=300)')
    runner = replace_once(runner, 'if len(plans) != 8:', 'if len(plans) != 6:')
    runner = replace_once(runner, 'Expected all four arms and both seeds', 'Expected all three arms and both seeds')
    runner = replace_once(runner,
        "return {'trainable_tensors': len(actual),\n            'admission_head_present': any(n.endswith('decoder.admission_weight') for n in expected.values())}",
        "return {'trainable_tensors': len(actual), 'new_parameter_tensors': 0}")
    write('experiment.py', runner)

    static_test = (OLD / 'test_runner.py').read_text(encoding='utf-8')
    static_test = static_test.replace('BudgetDEIM', 'BudgetNormDEIM').replace('BudgetCriterion', 'BudgetNormCriterion')
    static_test = static_test.replace('eight', 'six')
    write('test_runner.py', static_test)

    (HERE / 'configs').mkdir(exist_ok=False)
    bindings = {}
    for seed in (1, 2):
        original = json.loads((OLD / 'configs' / f'baseline_seed{seed}.json').read_text(encoding='utf-8'))
        for arm in ARMS:
            plan = deepcopy(original)
            plan.update(schema='output_budget_normalization_v1', arm=arm)
            plan['budget_loss'].pop('residual_weight')
            text = json.dumps(plan, ensure_ascii=False, indent=2)
            write(f'configs/{arm}_seed{seed}.json', text)
            bindings[f'{arm}_seed{seed}'] = hashlib.sha256(
                json.dumps(plan, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    write('plan_binding.json', json.dumps(bindings, indent=2))
    assert old_files == {str(p.relative_to(OLD)).replace('\\', '/'): sha(p)
                         for p in OLD.rglob('*') if p.is_file()}
    print('Prepared 6 plans and isolated runner. Old package unchanged; no model executed.')


if __name__ == '__main__':
    main()
