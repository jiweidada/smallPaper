"""Add byte-identical native augmentation modules for the query-only pair."""
from prepare_native_snapshot import HERE, SOURCE, sha
import json

files = ['data/_misc.py', 'data/dataset/coco_dataset.py', 'data/dataset/_dataset.py',
         'data/transforms/_transforms.py', 'data/transforms/container.py',
         'data/transforms/mosaic.py', 'data/transforms/functional.py']
manifest = json.loads((HERE / 'native_source_manifest.json').read_text(encoding='utf8'))
for rel in files:
    source = SOURCE / 'engine' / rel
    data = source.read_bytes()
    dest = HERE / 'native/engine' / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise RuntimeError(f'Already exists: {dest}')
    dest.write_bytes(data)
    manifest['files'].append(dict(source=str(source), local=dest.relative_to(HERE).as_posix(),
                                  sha256=sha(data), bytes=len(data)))
for p in ['data/dataset', 'data/transforms']:
    (HERE / 'native/engine' / p / '__init__.py').write_text('', encoding='utf8')
    manifest['package_initializers'][p] = ''
(HERE / 'native_source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
print('Native query augmentation snapshot added; shared source unchanged.')

