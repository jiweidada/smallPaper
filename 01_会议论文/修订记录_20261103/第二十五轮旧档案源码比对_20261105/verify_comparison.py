"""Independently verify selected ZIP members, snapshots and source facts."""
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import ast
import json

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'migrated_archive_comparison.json'
OUT = HERE / 'independent_archive_verification.json'
PREFIX = 'DEIM_1572862469/'


def file_sha(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_method(text, class_name, method_name):
    tree = ast.parse(text)
    return next(method for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
                for method in item.body if isinstance(method, ast.FunctionDef) and method.name == method_name)


def main():
    assert not OUT.exists()
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    assert report['status'] == 'MIGRATED_ARCHIVE_AND_CURRENT_SOURCE_COMPARED'
    archive_path = Path(report['archive']['path'])
    assert file_sha(archive_path) == report['archive']['sha256']
    code = {}
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        assert len(members) == report['archive']['members'] == 2737
        assert not [name for name in members if PurePosixPath(name).name in ('args.json', 'log.txt')]
        assert not [name for name in members if name.startswith(PREFIX + 'outputs/')]
        for item in report['file_comparisons']:
            archived_bytes = archive.read(item['archive_member'])
            current_bytes = Path(item['current_path']).read_bytes()
            assert sha256(archived_bytes).hexdigest() == item['archive_sha256']
            assert sha256(current_bytes).hexdigest() == item['current_sha256']
            archive_rel = item['archive_member'].removeprefix(PREFIX)
            assert (HERE / 'source_snapshot/archive' / archive_rel).read_bytes() == archived_bytes
            assert item['byte_identical'] == (archived_bytes == current_bytes)
            code[archive_rel] = (archived_bytes, current_bytes)
    for path in ['engine/backbone/hgnetv2_MSCA_EnhancedStem.py', 'engine/extre_module/custom_nn/attention/MSCA.py',
                 'configs/test/visdrone2019_mpsma.yml']:
        assert code[path][0] == code[path][1]
    for path, class_name, method_name in [('engine/extre_module/custom_nn/attention/simam.py', 'SimAM', 'forward'),
                                          ('engine/optim/ema.py', 'ModelEMA', 'update'),
                                          ('engine/core/yaml_config.py', 'YAMLConfig', 'ema')]:
        old, new = (data.decode('utf-8') for data in code[path])
        assert ast.dump(find_method(old, class_name, method_name), include_attributes=False) == ast.dump(find_method(new, class_name, method_name), include_attributes=False)
    old, new = (data.decode('utf-8') for data in code['engine/solver/det_solver.py'])
    def saves(text):
        fn = find_method(text, 'DetSolver', 'fit')
        return [node for node in ast.walk(fn) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.save']
    assert len(saves(old)) == 0 and len(saves(new)) == 1
    assert "coco_evaluator.coco_eval['bbox'].eval" in ast.unparse(saves(new)[0].args[0])
    for path in ['configs/test/deim_hgnetv2_EnhancedStem_n_visdrone.yml', 'configs/test/deim_hgnetv2_EnhancedStem_s_visdrone.yml']:
        old, new = (data.decode('utf-8').replace('\r\n', '\n') for data in code[path])
        assert old.replace('visdrone2019_mpsma.yml', 'visdrone2019.yml') == new
    result = {'status': 'ARCHIVE_MEMBERS_AND_SELECTED_SEMANTICS_INDEPENDENTLY_VERIFIED',
              'source_report_sha256': file_sha(REPORT), 'archive_sha256': file_sha(archive_path),
              'file_pairs_verified': len(code), 'method_checks': ['SimAM.forward', 'ModelEMA.update', 'YAMLConfig.ema'],
              'archive_fit_direct_torch_save_calls': 0, 'current_fit_direct_torch_save_calls': 1,
              'historical_source_authenticated': False,
              'limits': ['Selected source checks do not prove complete program equivalence or historical use.',
                         'Archive contents may share a common origin with the current tree.',
                         'No model, evaluator or training was run.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'file_pairs_verified': len(code), 'sha256': file_sha(OUT)}))


if __name__ == '__main__':
    main()
