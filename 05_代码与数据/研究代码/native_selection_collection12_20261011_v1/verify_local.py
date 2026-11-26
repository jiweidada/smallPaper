"""Validate and extract an exact downloaded selection packet, without a model."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, zipfile
from collector import HERE, names_valid, protocol, read, sha, validate_metadata, write


def verify(archive, metadata, destination, report):
    p = protocol(); archive, destination, report = Path(archive), Path(destination).resolve(), Path(report)
    meta = read(metadata)
    assert meta['status'] == 'PASS_SELECTION_DIAGNOSTIC_READ_ONLY_COLLECTION'
    assert archive.stat().st_size == meta['bytes'] and sha(archive) == meta['sha256']
    assert meta['model_forwards_added'] == meta['training_epochs_added'] == meta['existing_process_signals'] == 0
    assert not report.exists()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist(); names_valid(names)
        assert all(not x.is_dir() for x in z.infolist()) and z.testzip() is None
        raw = z.read('packet_manifest.json'); assert hashlib.sha256(raw).hexdigest() == meta['manifest_sha256']
        m = json.loads(raw); assert m['schema'] == 'native_selection_diagnostic_packet_v1'
        assert m['collection_protocol_sha256'] == sha(HERE / 'protocol.json') and m['collector_source_sha256'] == p['files']['collector.py']
        assert m['all_observed_replay_handles_stopped'] is True
        assert m['model_forwards_added'] == m['evaluator_calls_added'] == m['optimizer_updates_added'] == m['training_epochs_added'] == m['existing_process_signals'] == 0
        assert set(names) == set(m['files']) | {'packet_manifest.json'} and len(names) == meta['entries']
        assert all((destination / n).resolve().is_relative_to(destination) for n in names)
        for n, item in m['files'].items():
            h = hashlib.sha256()
            with z.open(n) as f:
                for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
            assert z.getinfo(n).file_size == item['bytes'] and h.hexdigest() == item['sha256'], n
        if destination.exists(): assert set(names) == {x.relative_to(destination).as_posix() for x in destination.rglob('*') if x.is_file()}
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert shutil.disk_usage(destination.parent).free > sum(x.file_size for x in z.infolist()) + 64 * 1024**2
            destination.mkdir(); z.extractall(destination)
    for n, item in m['files'].items(): assert (destination / n).stat().st_size == item['bytes'] and sha(destination / n) == item['sha256']
    for n, digest in p['diagnostic_source_sha256'].items(): assert sha(destination / 'diagnostic_source' / n) == digest
    binding = destination / 'binding'; diag = destination / 'diagnostic'
    identity = read(binding / 'receipt.json')['identity']; plan = read(binding / 'plan.json')
    trained = read(binding / 'native_training_completion.json'); v = read(diag / 'verification.json')
    validate_metadata(identity, plan, trained, v, read(diag / 'completion.json'), p, meta['arm'], meta['seed'], m['checkpoint_sha256'], sha(binding / 'native_training_completion.json'))
    assert identity == m['identity'] and sha(binding / 'native_terminal_manifest.json') == m['native_terminal_manifest_sha256']
    assert sha(diag / 'verification.json') == m['diagnostic_verification_sha256']
    assert sha(diag / 'selection_epoch12/manifest.json') == v['selection_manifest_sha256']
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    result = dict(status='PASS_DOWNLOADED_SELECTION_DIAGNOSTIC_PACKET_HASHES_AND_BINDINGS', identity=identity,
        archive_sha256=meta['sha256'], packet_manifest_sha256=meta['manifest_sha256'], files_verified=len(names),
        diagnostic_folder=str(diag), diagnostic_verification_sha256=sha(diag / 'verification.json'),
        selection_manifest_sha256=v['selection_manifest_sha256'], native_terminal_manifest_sha256=m['native_terminal_manifest_sha256'],
        verification_script_sha256=sha(__file__), collection_protocol_sha256=sha(HERE / 'protocol.json'),
        formal_native_terminal_packet_still_required=True, actual_model_efficacy_inferred=False,
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0)
    write(report, result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for n in ('archive', 'metadata', 'destination', 'report'): parser.add_argument('--' + n, required=True)
    args = parser.parse_args(); print(verify(args.archive, args.metadata, args.destination, args.report)['status'])
