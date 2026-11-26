"""Resolve selected archived YAML includes as inert data; do not import project code."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import ast
import json
import posixpath
import yaml

HERE = Path(__file__).resolve().parent
REC = HERE.parent
ROOT = REC.parent.parent
ARCHIVE = ROOT / '90_历史与工具/旧期刊投稿材料/《计算机应用与软件》投稿/_work/upload/DEIM_migrated.zip'
WORK = ARCHIVE.parent.parent
OUT = HERE / 'archived_config_resolution.json'
PREFIX = 'DEIM_1572862469/'
TARGETS = {
    'baseline_n': 'configs/test/deim_hgnetv2_n_visdrone.yml',
    'baseline_s': 'configs/test/deim_hgnetv2_s_visdrone.yml',
    'full_n': 'configs/test/deim_hgnetv2_EnhancedStem_n_visdrone.yml',
    'full_s': 'configs/test/deim_hgnetv2_EnhancedStem_s_visdrone.yml',
}


def digest(data):
    return sha256(data).hexdigest()


def file_digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class InertLoader(yaml.SafeLoader):
    """Reject executable tags; preserve Python-name tags as inert strings."""


TAGS = set()


def inert_python_name(loader, suffix, node):
    assert isinstance(node, yaml.ScalarNode)
    TAGS.add(suffix)
    return '<inert-python-name:' + suffix + '>'


InertLoader.add_multi_constructor('tag:yaml.org,2002:python/name:', inert_python_name)


def merge(target, other):
    for key, value in other.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge(target[key], value)
        else:
            target[key] = value
    return target


def load(archive, member, cfg, active, visits):
    assert member.startswith(PREFIX) and member not in active
    info = archive.getinfo(member)
    assert info.file_size < 2_000_000
    data = archive.read(info)
    doc = yaml.load(data.decode('utf-8-sig'), Loader=InertLoader)
    assert isinstance(doc, dict)
    visits.append({'member': member, 'sha256': digest(data), 'bytes': len(data)})
    active.add(member)
    includes = doc.get('__include__', [])
    assert isinstance(includes, list)
    for item in includes:
        assert isinstance(item, str) and not item.startswith(('/', '~'))
        child = posixpath.normpath(posixpath.join(posixpath.dirname(member), item))
        assert child.startswith(PREFIX)
        load(archive, child, cfg, active, visits)
    active.remove(member)
    return merge(cfg, doc)


def defaults_from_source(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CocoEvaluator')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    names = [arg.arg for arg in init.args.args]
    return {name: ast.literal_eval(value) for name, value in zip(names[-len(init.args.defaults):], init.args.defaults)}


def main():
    assert not OUT.exists()
    results, snapshots = {}, {}
    with ZipFile(ARCHIVE) as archive:
        source_member = PREFIX + 'engine/data/dataset/coco_eval.py'
        source = archive.read(source_member)
        defaults = defaults_from_source(source.decode('utf-8'))
        assert defaults['using_tiny_metrice'] is False and defaults['max_dets'] == 100
        snapshots[source_member] = source
        for label, rel in TARGETS.items():
            visits = []
            cfg = load(archive, PREFIX + rel, {}, set(), visits)
            evaluator = cfg['evaluator']
            val = cfg['val_dataloader']['dataset']
            result = {'entry': PREFIX + rel, 'include_visits': visits,
                      'val_img_folder': val['img_folder'], 'val_ann_file': val['ann_file'],
                      'evaluator': evaluator, 'num_classes': cfg['num_classes'],
                      'effective_constructor_defaults_if_unoverridden': {
                          'using_tiny_metrice': evaluator.get('using_tiny_metrice', defaults['using_tiny_metrice']),
                          'max_dets': evaluator.get('max_dets', defaults['max_dets'])}}
            results[label] = result
            for visit in visits:
                snapshots[visit['member']] = archive.read(visit['member'])
    for label in ['baseline_n', 'baseline_s']:
        result = results[label]
        assert 'VisDrone2019-DET-test-dev' in result['val_ann_file']
        assert result['evaluator']['using_tiny_metrice'] is True and result['evaluator']['max_dets'] == 300
    for label in ['full_n', 'full_s']:
        result = results[label]
        assert 'VisDrone2019-DET-val' in result['val_ann_file']
        assert result['effective_constructor_defaults_if_unoverridden'] == {'using_tiny_metrice': False, 'max_dets': 100}
    for member, data in snapshots.items():
        path = HERE / 'source_snapshot' / member.removeprefix(PREFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data
        else:
            path.write_bytes(data)
    companion = {}
    for name in ['MIGRATION_CHECKLIST.md', 'migrate_to_new_deim.ps1', 'make_upload_pack.py']:
        path = WORK / name
        companion[name] = {'path': str(path), 'sha256': file_digest(path)}
    output = {'status': 'ARCHIVED_INCLUDE_CHAINS_RESOLVED_AS_INERT_DATA',
              'checked_utc': datetime.now(timezone.utc).isoformat(),
              'archive': {'path': str(ARCHIVE), 'sha256': file_digest(ARCHIVE)},
              'yaml_parser': {'library': 'PyYAML', 'version': yaml.__version__,
                              'python_name_tags_preserved_inert': sorted(TAGS)},
              'method': 'Read YAML with a SafeLoader subclass; preserve Python-name tags as inert strings. Follow archived relative includes in order and recursively merge dictionaries as in its yaml_utils.py. No project code is imported.',
              'evaluator_constructor_source': {'member': source_member, 'sha256': digest(source), 'defaults': defaults},
              'configurations': results, 'migration_companion_files': companion,
              'limits': ['These are archived configuration resolutions without CLI overrides, not proof of what a historical process loaded or which dataset bytes it read.',
                         'Constructor defaults apply only if the inspected archive factory creates this CocoEvaluator without later overrides.',
                         'The baseline and full model use distinct splits/settings under these archived entry configurations; no measured AP or benchmark is recomputed.',
                         'Migration scripts describe intended/reported copying but are not an authenticated historical training-source binding.',
                         'No model, prediction, evaluator or training was run.'],
              'historical_execution_authenticated': False, 'model_instantiations': 0,
              'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': output['status'], 'configs': len(results), 'unique_archived_files': len(snapshots),
                      'inert_tag_count': len(TAGS), 'effective_settings': {key: {'val_ann_file': value['val_ann_file'],
                          **value['effective_constructor_defaults_if_unoverridden']} for key, value in results.items()},
                      'sha256': file_digest(OUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
