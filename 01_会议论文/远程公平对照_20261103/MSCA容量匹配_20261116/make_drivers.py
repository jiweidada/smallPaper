"""Derive isolated capacity-pair drivers from the verified 320-round FRESH driver."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FRESH = HERE.parent / '公平主对照FRESH_20261112_v1' / 'driver'
OUTPUT = HERE / 'driver'
ARMS = "('FULL_MSCA', 'NO_MSCA_MATCHED')"


def change(text, old, new, count=1):
    assert text.count(old) == count, old
    return text.replace(old, new)


def establish(name, edits):
    text = (FRESH / name).read_text(encoding='utf-8')
    for edit in edits:
        text = change(text, *edit)
    compile(text, name, 'exec')
    path = OUTPUT / name
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, name
    else:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)


OUTPUT.mkdir(exist_ok=True)
establish('stage_arm.py', [
    ("'backup_protocol.py','launch_arm.py','preflight.py']", "'backup_protocol.py','launch_arm.py','preflight.py','capacity_match.py']"),
    ("('BASE_FRESH','FULL_FRESH')", ARMS),
    ("['BASE_FRESH','FULL_FRESH']", "list("+ARMS+")", 2),
    ("args.arm.removesuffix('_FRESH')", "'FULL'"),
    ("staged['arm'].removesuffix('_FRESH')", "'FULL'"),
    ("assert sha(arm_dir/'inputs/initial.pt')==staged['initial_sha256']==pre['initial_file_sha256']",
     "assert sha(arm_dir/'inputs/initial.pt') == pre['initial_file_sha256']\n"
     "    assert sha(arm_dir/'frozen_parent/preflight/FULL.initial.pt') == staged['initial_sha256'] == pre['initialization']['reference_initial_sha256']\n"
     "    assert pre['trainable_parameters'] == 7540942"),
    ("'evaluated_and_next_roles':", "'comparison':'FULL versus parameter-matched ordinary convolution replacement of MSCA',\n        'evaluated_and_next_roles':"),
])
establish('fair_helpers.py', [
    ("arm = arm.removesuffix('_FRESH')", "variant = arm\n    assert variant in "+ARMS+"\n    arm = 'FULL'"),
    ("    model = cfg.model\n", "    model = cfg.model\n    if variant == 'NO_MSCA_MATCHED':\n        from capacity_match import apply\n        apply(model)\n"),
])
establish('preflight.py', [
    ("['BASE_FRESH','FULL_FRESH']", "list("+ARMS+")"),
    ("        original_arm=args.arm.removesuffix('_FRESH')", "        original_arm='FULL'"),
    ("        initial=torch.load(root/'inputs/initial.pt',map_location='cpu',weights_only=True)\n"
     "        model.load_state_dict(initial,strict=True)\n"
     "        initial_hash=fair.tensor_hash(model.state_dict())\n"
     "        assert initial_hash==archived['initial_tensor_sha256']\n"
     "        assert report['initial_file_sha256']==archived['initial_file_sha256']",
     "        from capacity_match import initialize, verify_gradients\n"
     "        assert report['initial_file_sha256'] == archived['initial_file_sha256']\n"
     "        initialization = initialize(fair, model, root/'frozen_parent/preflight/FULL.initial.pt')\n"
     "        initial_hash = fair.tensor_hash(model.state_dict())\n"
     "        if args.arm == 'FULL_MSCA':\n"
     "            assert initial_hash == archived['initial_tensor_sha256']\n"
     "        else:\n"
     "            fair.savept(root/'inputs/initial.pt', model.state_dict())\n"
     "        recovered = torch.load(root/'inputs/initial.pt',map_location='cpu',weights_only=True)\n"
     "        model.load_state_dict(recovered,strict=True)\n"
     "        assert fair.tensor_hash(model.state_dict()) == initial_hash\n"
     "        report['initial_file_sha256'] = fair.sha(root/'inputs/initial.pt')"),
    ("        assert [g['lr'] for g in optimizer.param_groups]==archived['initial_executed_lrs']",
     "        assert [g['lr'] for g in optimizer.param_groups]==archived['initial_executed_lrs']\n"
     "        groups = [p for group in optimizer.param_groups for p in group['params']]\n"
     "        assert len(groups) == len({id(p) for p in groups})\n"
     "        assert {id(p) for p in groups} == {id(p) for p in model.parameters() if p.requires_grad}\n"
     "        assert not optimizer.state\n"
     "        report['optimizer_groups'] = [{'parameters':sum(p.numel() for p in g['params']), 'lr':g['lr'],\n"
     "                                       'weight_decay':g['weight_decay']} for g in optimizer.param_groups]"),
    ("        loss.backward()", "        loss.backward()\n        if args.arm == 'NO_MSCA_MATCHED':\n            report['replacement_gradient_check'] = verify_gradients(model)"),
    ("initialization={'arm':args.arm,'source':'frozen archived initial tensors, strict loading; shared554 verified',\n"
     "                                      'initialization_after_constructor_seed_reset':0},",
     "initialization=initialization,"),
])
for script in ('launch_arm.py', 'train_runner.py'):
    establish(script, [("('BASE_FRESH', 'FULL_FRESH')", ARMS)])
establish('backup_protocol.py', [])
establish('backup_monitor.py', [
    ("{'BASE_FRESH':45 * 1024**3, 'FULL_FRESH':75 * 1024**3}[protocol.ARM]", "100 * 1024**3"),
    ("choices=[30715, 30048]", "choices=[30737, 30959]"),
])
establish('checkpoint_validator.py', [])
print('Capacity-pair drivers ready.')
