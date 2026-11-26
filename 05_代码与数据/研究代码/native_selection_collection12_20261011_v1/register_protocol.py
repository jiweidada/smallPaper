from pathlib import Path
import hashlib, json
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
diag = ROOT / '英文小论文/code/native_selection_diagnostic12_20261011_v2'
source = {x.name: sha(x) for x in diag.iterdir() if x.is_file()}
p = dict(schema='native_selection_collection12_v1',
    files={n: sha(HERE / n) for n in ('collector.py', 'verify_local.py', 'validate_cpu.py', 'register_protocol.py')},
    diagnostic_source_sha256=source, diagnostic_protocol_sha256=source['diagnostic_protocol.json'],
    arms=['original_stem', 'native_stem_scratch', 'highfreq_rgb'], seeds=[1,2], wait_timeout_seconds=172800,
    model_forwards_added=0, evaluator_calls_added=0, optimizer_updates_added=0, training_epochs_added=0,
    existing_process_signals=0, automatic_model_retry=False,
    scope='Package only completed, exact saved-EMA diagnostics from existing producer v2. '
          'Observe actual existing queue/replay handles; require all observed replay handles stopped and positive completion. '
          'Checksum all emitted arrays, source files and binding records; native checkpoints are not copied again. '
          'Partial positive packets may be retained if a later diagnostic fails. No model/runtime import, '
          'diagnostic launch, optimizer, trainer, process signal, baseline forward or extension.')
with (HERE / 'protocol.json').open('x', encoding='utf8') as f: json.dump(p, f, ensure_ascii=False, indent=2, allow_nan=False)
print('READ_ONLY_COLLECTION_PROTOCOL_SEALED')
