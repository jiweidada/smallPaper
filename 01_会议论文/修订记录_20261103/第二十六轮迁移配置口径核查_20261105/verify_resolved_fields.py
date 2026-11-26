"""Independently trace selected YAML field overlays and evaluator defaults."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import ast
import json
import posixpath
import shutil
import yaml

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'archived_config_resolution.json'
OUT = HERE / 'independent_field_provenance.json'
PREFIX = 'DEIM_1572862469/'
FIELDS = [('val_dataloader', 'dataset', 'img_folder'), ('val_dataloader', 'dataset', 'ann_file'),
          ('evaluator', 'type'), ('evaluator', 'using_tiny_metrice'), ('evaluator', 'max_dets')]
MISSING = '<absent>'


def file_digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def path_value(value, keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def applied_documents(archive, member, active):
    assert member.startswith(PREFIX) and member not in active
    data = archive.read(member)
    document = yaml.safe_load(data.decode('utf-8-sig'))
    assert isinstance(document, dict)
    active.add(member)
    docs = []
    for item in document.get('__include__', []):
        child = posixpath.normpath(posixpath.join(posixpath.dirname(member), item))
        docs.extend(applied_documents(archive, child, active))
    active.remove(member)
    docs.append((member, document))
    return docs


def main():
    assert not OUT.exists()
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    assert report['status'] == 'ARCHIVED_INCLUDE_CHAINS_RESOLVED_AS_INERT_DATA'
    assert not report['yaml_parser']['python_name_tags_preserved_inert']
    archive_path = Path(report['archive']['path'])
    assert file_digest(archive_path) == report['archive']['sha256']
    summary = {}
    with ZipFile(archive_path) as archive:
        for label, expected in report['configurations'].items():
            docs = applied_documents(archive, expected['entry'], set())
            resolved = {'.'.join(field): {'value': MISSING, 'last_source': None} for field in FIELDS}
            for name, doc in docs:
                if 'evaluator' in doc:
                    assert isinstance(doc['evaluator'], dict)
                if 'val_dataloader' in doc:
                    assert isinstance(doc['val_dataloader'], dict)
                    if 'dataset' in doc['val_dataloader']:
                        assert isinstance(doc['val_dataloader']['dataset'], dict)
                for field in FIELDS:
                    value = path_value(doc, field)
                    if value != MISSING:
                        resolved['.'.join(field)] = {'value': value, 'last_source': name}
            assert resolved['val_dataloader.dataset.ann_file']['value'] == expected['val_ann_file']
            assert resolved['val_dataloader.dataset.img_folder']['value'] == expected['val_img_folder']
            assert resolved['evaluator.type']['value'] == expected['evaluator']['type']
            for field in ['using_tiny_metrice', 'max_dets']:
                actual = resolved['evaluator.' + field]['value']
                assert actual == expected['evaluator'].get(field, MISSING)
            for visit in expected['include_visits']:
                data = archive.read(visit['member'])
                assert sha256(data).hexdigest() == visit['sha256']
                assert (HERE / 'source_snapshot' / visit['member'].removeprefix(PREFIX)).read_bytes() == data
            summary[label] = {'applied_document_count': len(docs), 'fields': resolved}
        eval_member = PREFIX + 'engine/data/dataset/coco_eval.py'
        eval_source = archive.read(eval_member).decode('utf-8')
        evaluator = next(node for node in ast.parse(eval_source).body if isinstance(node, ast.ClassDef) and node.name == 'CocoEvaluator')
        init = next(node for node in evaluator.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        names = [item.arg for item in init.args.args]
        defaults = {key: ast.literal_eval(value) for key, value in zip(names[-len(init.args.defaults):], init.args.defaults)}
        assert defaults['using_tiny_metrice'] is False and defaults['max_dets'] == 100
        shared = next(ast.literal_eval(node.value) for node in evaluator.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == '__share__' for target in node.targets))
        assert 'using_tiny_metrice' not in shared and 'max_dets' not in shared
        factory_member = PREFIX + 'engine/core/workspace.py'
        yaml_member = PREFIX + 'engine/core/yaml_config.py'
        factory_bytes, yaml_bytes = archive.read(factory_member), archive.read(yaml_member)
        factory, config = factory_bytes.decode('utf-8'), yaml_bytes.decode('utf-8')
        ast.parse(factory)
        ast.parse(config)
        assert "_cfg.update(_cfg['_kwargs'])" in factory and '_cfg.update(cfg)' in factory
        assert "create('evaluator', self.global_cfg, coco_gt=base_ds)" in config
        for member, data in [(factory_member, factory_bytes), (yaml_member, yaml_bytes)]:
            path = HERE / 'source_snapshot' / member.removeprefix(PREFIX)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert path.read_bytes() == data
            else:
                path.write_bytes(data)
    companion = {}
    for name, item in report['migration_companion_files'].items():
        path = Path(item['path'])
        assert file_digest(path) == item['sha256']
        dest = HERE / 'migration_records' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert file_digest(dest) == item['sha256']
        else:
            shutil.copy2(path, dest)
        companion[name] = {'path': str(path), 'snapshot': str(dest), 'sha256': item['sha256']}
    result = {'status': 'FIELD_OVERLAY_AND_EVALUATOR_DEFAULTS_INDEPENDENTLY_VERIFIED',
              'resolution_report_sha256': file_digest(REPORT), 'archive_sha256': file_digest(archive_path),
              'fields': summary, 'evaluator_constructor_defaults': defaults, 'evaluator_shared_fields': shared,
              'factory_and_config_source_sha256': {'workspace': sha256(factory_bytes).hexdigest(),
                                                    'yaml_config': sha256(yaml_bytes).hexdigest()},
              'migration_records': companion,
              'limits': ['Field overlays describe archived entry configurations without command-line or runtime overrides.',
                         'The default interpretation relies on the inspected factory and registered evaluator path.',
                         'No historical execution, source/data identity or accuracy comparison is established.',
                         'No model, evaluation or training was run.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'configs_verified': len(summary),
                      'sha256': file_digest(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
