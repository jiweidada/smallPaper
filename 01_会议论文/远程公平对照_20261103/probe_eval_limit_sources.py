"""Read evaluator/package sources and metadata only; no evaluator import or execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import importlib.metadata as metadata
import json

root = Path('/25085400120/mpsma_fair_20261103_v1')
distribution = metadata.distribution('faster-coco-eval')
package_root = Path(distribution.locate_file('faster_coco_eval'))
paths = [package_root / 'core/cocoeval.py', package_root / 'core/faster_eval_api.py']
paths += list((root / 'source_snapshot').rglob('coco_eval.py'))
paths += list((root / 'source_snapshot').rglob('dfine_postprocessor.py'))
paths += list((root / 'source_snapshot').rglob('rtdetr_postprocessor.py'))
paths += list((root / 'source_snapshot').rglob('postprocessor.py'))
sources = []
for path in sorted(set(paths)):
    assert path.is_file(), str(path)
    content = path.read_bytes()
    sources.append({'path': str(path), 'sha256': sha256(content).hexdigest(),
                    'text': content.decode('utf-8-sig')})
print(json.dumps({'status': 'EVALUATOR_SOURCE_FILES_READ_ONLY',
                  'observed_remote_utc': datetime.now(timezone.utc).isoformat(),
                  'package': 'faster-coco-eval', 'version': distribution.version,
                  'sources': sources, 'model_imports': 0, 'model_executions': 0,
                  'evaluator_imports': 0, 'evaluator_executions': 0}))
