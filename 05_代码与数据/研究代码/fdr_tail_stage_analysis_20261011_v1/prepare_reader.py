"""Adapt the already exercised three-boundary reader to verified FDR packets.

Preserve its coordinate arithmetic and GT identity joins. Do not certify a
new experimental result or change the immutable training package.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / '英文小论文/code/scale_protection12_monitor_20261010_v1'
MONITOR = ROOT / '英文小论文/code/fdr_tail12_monitor_20261011_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert sha(OLD / 'analyze_stages.py') == 'b92a59d11df6f3de3cfe72843a974667afc642a4d38ffae99f2501ff492746a4'
    assert sha(OLD / 'geometry_core.py') == '0c0419f9f5d2b162600789cd62530d0826908034203f76978c61a66c2ef6df58'
    reader_protocol = json.loads((MONITOR / 'local_reader_protocol.json').read_text(encoding='utf8'))
    assert sha(MONITOR / 'verify_terminal_local.py') == reader_protocol['reader_files_sha256']['verify_terminal_local.py']
    original = (OLD / 'analyze_stages.py').read_text(encoding='utf8')
    source = replace_once(original, 'def analyze(packet, output):', 'def analyze(packet, output, verification_report):')
    source = replace_once(source, "    assert not output.exists()\n", """    assert not output.exists()
    verification = read(verification_report)
    monitor = Path(__file__).resolve().parents[3] / '英文小论文/code/fdr_tail12_monitor_20261011_v1'
    assert verification['status'] == 'PASS_FDR_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_EMA_RELOAD'
    assert verification['local_reader_protocol_sha256'] == sha(monitor/'local_reader_protocol.json')
    assert verification['verification_script_sha256'] == sha(monitor/'verify_terminal_local.py')
    assert verification['terminal_manifest_sha256'] == sha(packet/'terminal_manifest.json')
""")
    source = replace_once(source, "packet/'prepared/receipt.json'", "packet/'bundle/receipt.json'")
    source = replace_once(source, "    assert arm in ('target_protect','random_protect') and seed in (1,2)",
        "    assert arm == 'endpoint2' and seed in (1,2)\n    assert identity == verification['identity'] and identity['endpoint_factor'] == 2.0")
    source = replace_once(source, "Path(__file__).parent/'closeout_protocol.json'", "Path(__file__).parent/'analysis_protocol.json'")
    source = replace_once(source, "        native_fixed12_delta_AP_metrics_pp=paired['delta_AP_metrics_pp'],native_gate_passed=saved['native_gate_passed'],",
        "        native_fixed12_delta_AP_metrics_pp=paired['delta_AP_metrics_pp'],native_gate_passed=saved['native_gate_passed'],\n        source_verification_report_sha256=sha(verification_report),analysis_protocol_sha256=sha(Path(__file__).parent/'analysis_protocol.json'),")
    source = replace_once(source, "p.add_argument('--output',required=True)", "p.add_argument('--output',required=True);p.add_argument('--verification-report',required=True)")
    source = replace_once(source, 'r = analyze(a.packet,a.output)', 'r = analyze(a.packet,a.output,a.verification_report)')
    ast.parse(source)
    with (HERE / 'analyze_stages.py').open('x', encoding='utf8', newline='\n') as f:
        f.write(source)
    with (HERE / 'geometry_core.py').open('xb') as f:
        f.write((OLD / 'geometry_core.py').read_bytes())
    legacy_protocol = json.loads((OLD / 'closeout_protocol.json').read_text(encoding='utf8'))
    value = dict(schema='fdr_verified_saved_stage_analysis_v1',
        files={'geometry_core.py':sha(HERE / 'geometry_core.py'), 'analyze_stages.py':sha(HERE / 'analyze_stages.py')},
        prepare_reader_sha256=sha(__file__), source_analysis_sha256=sha(OLD / 'analyze_stages.py'),
        source_geometry_sha256=sha(OLD / 'geometry_core.py'),
        required_terminal_reader_protocol_sha256=sha(MONITOR / 'local_reader_protocol.json'),
        stage_boundary_locations=legacy_protocol['stage_boundary_locations'],
        stage_analysis_scope=legacy_protocol['stage_analysis_scope'],
        stages=['encoder_initial_boxes','pre_after_first_layer_boxes','final_boxes'],
        thresholds=[0.5,0.75],within_run_query_order_only=True,cross_run_query_identity_assumed=False,
        fixed_GT_cohorts=['all','COCO_small','input_under4','input_4_to_under8','input_8_to_under16','input_ge16'],
        geometry_is_any_query_support_not_one_to_one_native_TP=True,
        classification_argmax_is_proxy_not_native_selection=True,
        FDR_training_may_change_earlier_stage_outputs=True,
        qualification_loss_between_observed_boundaries_is_not_a_unique_module_cause=True,
        new_FDR_terminal_packets_analyzed=0,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
    write(HERE / 'analysis_protocol.json', value)
    assert 'torch' not in sys.modules
    print('FDR_SAVED_STAGE_READER_PREPARED_NO_NEW_EXPERIMENT_RESULT', sha(HERE / 'analysis_protocol.json'), flush=True)


if __name__ == '__main__':
    main()
