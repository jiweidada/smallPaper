"""Sequential read-only packaging of two already-stopped seed2 native12 controls."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import traceback

ROOT = Path('/24085403014/DEIM/experiments/stem_component_primary_collection_20261011_v2')
SOURCES = ROOT / 'code'
ARMS = ('attention_spatial_mean', 'early_final_downsample')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def write(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    protocol = read(SOURCES/'protocol.json')
    assert sha(SOURCES/'collect.py') == protocol['collector_sha256']
    spec = importlib.util.spec_from_file_location('exact_component_primary_collector', SOURCES/'collect.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
        source_sha256=sha(__file__), collector_sha256=sha(SOURCES/'collect.py'),
        collector_protocol_sha256=sha(SOURCES/'protocol.json'), arms=ARMS,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), new_model_forwards=0,
        new_evaluator_calls=0, new_training_epochs=0, automatic_retry=False)
    write(ROOT/'new_seed2_controls_collect_v1_start.json', receipt)
    results = []
    try:
        for arm in ARMS:
            directory = ROOT/f'seed2_{arm}_v1'
            if directory.exists():
                # Reuse only a positively completed package, never a partial attempt.
                value = read(directory/'metadata.json')
                assert value['status']=='COLLECTED_STOPPED_EXACT12_PRIMARY_ARTIFACTS'
                assert value['arm']==arm and value['seed']==2 and value['collector_sha256']==receipt['collector_sha256']
                assert value['collector_protocol_sha256']==receipt['collector_protocol_sha256']
                assert sha(value['archive'])==value['sha256']
            else:
                value = module.collect('/24085403014', 2, arm)
            results.append(value)
            write(ROOT/f'new_seed2_controls_{arm}_collection_execution_v1.json',dict(exit_code=0,metadata=value))
        assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
        write(ROOT/'new_seed2_controls_collect_v1_completion.json',dict(
            status='BOTH_STOPPED_SEED2_NATIVE12_CONTROL_PRIMARY_PACKAGES_COLLECTED',
            results=results, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,automatic_retry=False))
    except BaseException:
        write(ROOT/'new_seed2_controls_collect_v1_failure.json',dict(traceback=traceback.format_exc(),
            completed_arms=[v['arm'] for v in results],automatic_retry=False,new_model_forwards=0,new_training_epochs=0))
        raise


if __name__=='__main__':
    main()
