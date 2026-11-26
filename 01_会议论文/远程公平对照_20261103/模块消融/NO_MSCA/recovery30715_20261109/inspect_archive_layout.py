import json
from pathlib import Path
import zipfile

base = Path('/25085400120/mpsma_fair_20261103_v1')
archive = base / 'verified_dataset.zip'
with zipfile.ZipFile(archive) as bundle:
    names = bundle.infolist()
    result = {'archive': str(archive), 'members': len(names),
        'first_entries': [{'name': v.filename, 'bytes': v.file_size} for v in names[:8]],
        'annotation_entries': [v.filename for v in names if v.filename.endswith('.json')],
        'cache_root_exists': Path('/tmp/mpsma_module_ablation_20261107_data').exists()}
manifest = json.loads((base / 'dataset_bundle_manifest.json').read_text())
result['manifest_keys'] = list(manifest)
result['manifest_first_entries'] = manifest['files'][:3]
result['manifest_count'] = len(manifest['files'])
print(json.dumps(result))
