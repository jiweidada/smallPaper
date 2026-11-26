"""Read-only endpoint identity, initialization and terminal-state requirements."""
ARMS = ('backbone_silu', 'p2_skip', 'backbone_attention')
EXPECTED = {
    'backbone_silu': dict(parameters=10231997, extra_parameters=0, exact_shared_tensors=794,
                         new_state_tensors=0, removed_native_tensors=0, state_tensors=794),
    'p2_skip': dict(parameters=10249437, extra_parameters=17440, exact_shared_tensors=794,
                   new_state_tensors=7, removed_native_tensors=0, state_tensors=801),
    'backbone_attention': dict(parameters=14030003, extra_parameters=3798006, exact_shared_tensors=714,
                              new_state_tensors=90, removed_native_tensors=80, state_tensors=804),
}

def identity_plan(identity, plan, initialization, arm, seed):
    assert arm in ARMS and seed in (1, 2)
    assert identity['schema'] == plan['schema'] == 'user_backbone12_training_v1'
    assert identity['arm'] == plan['arm'] == initialization['user_arm'] == arm
    assert identity['seed'] == plan['seed'] == seed
    assert plan['max_epochs'] == 12 and plan['recipe_epochs'] == 132 and plan['fixed_endpoint'] == 'native_epoch12_EMA'
    expected = EXPECTED[arm]
    assert plan['expected_parameters'] == initialization['parameters'] == expected['parameters']
    assert identity['native_parameter_count'] == 10231997
    assert identity['added_parameter_count'] == plan['extra_parameters'] == expected['extra_parameters']
    assert all(initialization[k] == v for k, v in expected.items())
    assert initialization['strict_load'] and initialization['all_shared_tensors_exact'] and initialization['new_state_unchanged']
    assert identity['baseline_training_epochs_added'] == identity['baseline_model_forwards_added'] == 0

def positive_terminal(identity, done, controller, candidate):
    assert done['identity'] == identity and done['status'] == 'NATIVE_FIT_FINISHED_EXACT12'
    assert done['completed_epochs'] == 12 and done['entered_epochs'] == list(range(12))
    assert not done['automatic_retry']
    assert controller['status'] == 'COMPLETE_FIXED12'
    assert candidate['exit_code'] == 0 and not candidate['timed_out'] and not candidate['automatic_retry']

def owners(trainer, controller, frozen):
    assert trainer['child_pid'] == frozen['pid'] and trainer['child_start_ticks'] == frozen['start_ticks']
    assert trainer['controller_pid'] == controller['pid'] == frozen['controller_pid']
    assert controller['start_ticks'] == frozen['controller_start_ticks']
