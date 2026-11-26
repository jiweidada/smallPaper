"""Repair only annotation lookup for the new terminal archive's reuse contract."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fdr_tail_stage_analysis_20261011_v1'
ROOT = HERE.parents[2]
MONITOR = ROOT / '英文小论文/code/fdr_tail12_monitor_20261011_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def main():
    assert sha(OLD/'analysis_protocol.json') == '505c999701c9f91f404525954e87ef85113bed95191d5319b56aa8c73c105250'
    protocol = json.loads((OLD/'analysis_protocol.json').read_text(encoding='utf8'))
    for name, digest in protocol['files'].items():
        assert sha(OLD/name) == digest, name
    source = (OLD/'analyze_stages.py').read_text(encoding='utf8')
    source = replace_once(source, "    ann_path = packet/'reference/val_annotations.json'", """    assert sha(__file__) == protocol['files']['analyze_stages.py']
    annotation_source = Path(verification['baseline_source_packet']).resolve()
    terminal_protocol = read(monitor/'local_reader_protocol.json')
    expected_source = (monitor.parents[2] / terminal_protocol['reused_local_evidence'][str(seed)]['packet_relative_path']).resolve()
    assert annotation_source == expected_source
    ann_path = annotation_source/'reference/val_annotations.json'""")
    source = replace_once(source,
        "        boundary_locations=protocol['stage_boundary_locations'],native_TP_geometry_disagreements=conflicts,",
        "        annotation_path=str(ann_path),annotation_source_packet=str(annotation_source),annotation_sha256=sha(ann_path),\n        boundary_locations=protocol['stage_boundary_locations'],native_TP_geometry_disagreements=conflicts,")
    ast.parse(source)
    with (HERE/'analyze_stages.py').open('x',encoding='utf8',newline='\n') as file:
        file.write(source)
    with (HERE/'geometry_core.py').open('xb') as file:
        file.write((OLD/'geometry_core.py').read_bytes())
    protocol.update(schema='fdr_verified_saved_stage_analysis_v2',
        files={name:sha(HERE/name) for name in ('geometry_core.py','analyze_stages.py')},
        previous_protocol_sha256=sha(OLD/'analysis_protocol.json'),
        preparation_script_sha256=sha(__file__),
        correction='New terminal archives omit reference files and reuse the verified old local packet. Resolve validation annotations from the accepted evidence report and require the pinned source packet and annotation SHA. Geometry, cohorts, native gates and training are unchanged.',
        annotation_origin='Accepted terminal evidence report baseline_source_packet/reference/val_annotations.json',
        new_FDR_terminal_packets_analyzed=0)
    with (HERE/'analysis_protocol.json').open('x',encoding='utf8') as file:
        json.dump(protocol,file,ensure_ascii=False,indent=2,allow_nan=False)
    old_sources = json.loads((MONITOR/'local_reader_protocol.json').read_text(encoding='utf8'))
    counts = {}
    for seed, item in old_sources['reused_local_evidence'].items():
        old_packet = ROOT/item['packet_relative_path']
        old_report = ROOT/item['report_relative_path']
        assert sha(old_report) == item['report_sha256']
        report = json.loads(old_report.read_text(encoding='utf8'))
        annotation = old_packet/'reference/val_annotations.json'
        assert sha(annotation) == report['identity']['val_annotation_sha256']
        values = json.loads(annotation.read_text(encoding='utf8'))
        assert len(values['images']) == 548 and len(values['annotations']) == 38759
        counts[seed] = dict(images=548,GT=38759,annotation_sha256=sha(annotation))
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='ANNOTATION_REUSE_PATH_REPAIRED_PRE_EXECUTION',
        protocol_sha256=sha(HERE/'analysis_protocol.json'),actual_pinned_annotation_checks=counts,
        geometry_arithmetic_unchanged=True,new_FDR_terminal_packets_analyzed=0,
        model_forwards=0,evaluator_calls=0,optimizer_updates=0,epochs_added=0),ensure_ascii=False))


if __name__ == '__main__':
    main()
