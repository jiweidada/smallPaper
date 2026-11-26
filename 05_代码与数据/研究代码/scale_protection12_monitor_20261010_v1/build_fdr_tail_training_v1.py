"""Create a separate FDR screen package; never mutate frozen experiments."""
from pathlib import Path
import ast
import hashlib
import json

WORK = Path(__file__).resolve().parents[3]
CODE = WORK / '英文小论文/code'
OLD = CODE / 'scale_protection12_20261010_v2'
NEW = CODE / 'fdr_tail12_20261011_v1'


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new, 1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not NEW.exists(), 'No overwrite of a previous candidate package'
    NEW.mkdir()
    for name in ('geometry.py', 'residual.py', 'stage_trace.py', 'evaluator_binding.py',
                 'evaluator_sources.json', 'source_binding.json', 'reuse_baseline.py'):
        (NEW / name).write_bytes((OLD / name).read_bytes())
    (NEW / 'tail_grid.py').write_bytes((CODE / 'fdr_tail_decoder_20261010_v2/tail_grid.py').read_bytes())
    results = (OLD / 'results.py').read_text(encoding='utf8')
    results = replace_once(results,
        "ARMS = ('baseline', 'fine', 'degraded', 'target_protect', 'random_protect')",
        "ARMS = ('baseline', 'endpoint2')")
    (NEW / 'results.py').write_text(results, encoding='utf8')
    audit = (OLD / 'training_audit.py').read_text(encoding='utf8')
    audit = replace_once(audit, 'Native first100 inputs/RNG differ before scale protection is active',
        'Native first100 inputs/pre-forward RNG differ in the isolated FDR candidate')
    audit = replace_once(audit, 'scale_protection_active=epoch >= 4,',
        'FDR_endpoint_factor=2.0, candidate_active_from_epoch_zero=True,')
    (NEW / 'training_audit.py').write_text(audit, encoding='utf8')
    exp = (OLD / 'experiment.py').read_text(encoding='utf8')
    exp = exp.replace('ScaleProtectionSolver', 'FDRTailSolver')
    exp = replace_once(exp, '    register_model()\n',
        '    register_model()\n    from tail_grid import TailGridBinding\n    binding = TailGridBinding(plan[\'endpoint_factor\']).install()\n')
    begin = exp.index('            from scale_guard import install\n')
    end = exp.index('\n        def _train_one_epoch_with_amp_rollback', begin)
    exp = exp[:begin] + '''            write(self.output_dir/'runtime_start.json', dict(identity=receipt['identity'],
                initialization=self.initialization_receipt, optimizer_coverage=coverage,
                tail_grid=binding.metadata(), evaluator_runtime=evaluator_runtime,
                torch=str(torch.__version__), cuda=torch.version.cuda,
                resume=None, automatic_retry=False, added_parameters=0,
                native_augmentation_and_supervision_weights_unchanged=True,
                caution='All ordinary and DN FGL labels use the same expanded prediction grid; no efficacy claim.'))
''' + exp[end:]
    exp = exp.replace("'scale_protection12_identity'", "'fdr_tail12_identity'")
    exp = replace_once(exp,
        "        if hasattr(solver,'scale_guard'):\n            solver.scale_guard.close()\n", '')
    exp = replace_once(exp, '        dist_utils.cleanup()\n',
        '        binding.close()\n        dist_utils.cleanup()\n')
    (NEW / 'experiment.py').write_text(exp, encoding='utf8')
    supervise = (OLD / 'supervise.py').read_text(encoding='utf8')
    supervise = replace_once(supervise,
        "        write(root/'controller_completion.json',dict(status='COMPLETE_FIXED12', automatic_extension=False,\n            automatic_retry=False, automatic_control_launch=False))",
        "        write(root/'controller_completion.json',dict(status='COMPLETE_FIXED12', automatic_extension=False,\n            automatic_retry=False, automatic_control_launch=False))\n"
        "        from verify_candidate_reload import supervise as verify_saved_candidate\n"
        "        verify_saved_candidate(Path(a.bundle).resolve())\n"
        "        from collect_terminal import collect\n"
        "        collect(Path(a.bundle).resolve())")
    (NEW / 'supervise.py').write_text(supervise, encoding='utf8')
    reload = (CODE / 'scale_protection12_monitor_20261010_v1/verify_candidate_reload.py').read_text(encoding='utf8')
    reload = reload.replace("'scale_protection12_closeout_v1'", "'fdr_tail12_closeout_v1'")
    reload = replace_once(reload, "code = bundle.parent.parent/'code_v2'", "code = bundle.parent.parent/'code'")
    reload = replace_once(reload, "assert plan['arm'] in ('target_protect','random_protect')", "assert plan['arm'] == 'endpoint2'")
    reload = replace_once(reload, '    register_model()\n',
        "    register_model()\n    from tail_grid import TailGridBinding\n    binding = TailGridBinding(plan['endpoint_factor']).install()\n")
    reload = reload.replace("'scale_protection12_identity'", "'fdr_tail12_identity'")
    reload = replace_once(reload, '        handle.remove(); trace.close(); dist_utils.cleanup()\n',
        '        handle.remove(); trace.close(); binding.close(); dist_utils.cleanup()\n')
    (NEW / 'verify_candidate_reload.py').write_text(reload, encoding='utf8')
    protocol = json.loads((OLD / 'training_protocol.json').read_text(encoding='utf8'))
    for key in ('experiment_protocol_sha256', 'random_control_additional_gate', 'data_only_report_sha256',
                'training_profile_sha256', 'data_only_physical_gate_verified', 'inference_architecture_unchanged',
                'expected_protected_training_image_count', 'mechanism_scope'):
        protocol.pop(key)
    protocol.update(schema='fdr_tail12_training_v1', execution_allowlist={'endpoint2':[1,2]},
        endpoint_factor=2.0, activation_epoch_zero_based=0, native_recipe_epochs=132,
        fixed_endpoint='native_epoch12_EMA', resume=False, automatic_retry=False,
        baseline_training_or_forward_replay=False,
        primary_gate='Both same-seed APs > BASE12 and AP >= BASE12; fixed12 only.',
        mechanism_scope='Fixed twofold extension of FDR extreme supports; central31, references, losses and augmentation retained. Physical-domain hypothesis, not a novelty claim.',
        tail_core_sha256=sha(NEW / 'tail_grid.py'),
        proposal_analysis_manifest_sha256='9b23cff617afa22c85c91ceaeec5c6881b581460a4eb91b6717ff25c58894e1b',
        CPU_engineering_remote_verification_sha256='ec48bc6517412c1494c170db3a90277c0df08fbbe04f33ec1c3fe239cfde1452',
        no_best_epoch_or_posthoc_factor_search=True,
        limitation='Range ablations already exist in D-FINE. A wider domain can damage previously correct geometry; fixed-head counterfactual and full training remain separate evidence.')
    (NEW / 'training_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    copied = {name:sha(NEW/name) for name in ('geometry.py','residual.py','stage_trace.py','evaluator_binding.py',
                                           'evaluator_sources.json','source_binding.json','reuse_baseline.py')}
    for p in NEW.glob('*.py'):
        ast.parse(p.read_text(encoding='utf8'))
    (NEW / 'construction_provenance.json').write_text(json.dumps(dict(
        status='PARTIAL_PACKAGE_NEW_RUNTIME_AND_PREPARATION_PENDING',
        unchanged_helpers=copied, source_package=str(OLD),
        frozen_source_package_not_modified=True, tail_core_sha256=sha(NEW/'tail_grid.py'),
        changed_files=['experiment.py','training_audit.py','results.py','supervise.py','verify_candidate_reload.py'],
        new_model_forwards=0,new_training_epochs=0),indent=2),encoding='utf8')
    print(json.dumps(dict(status='ISOLATED_PACKAGE_CONSTRUCTED_PARTIAL',path=str(NEW),files=len(list(NEW.iterdir())))))


if __name__ == '__main__':
    main()
