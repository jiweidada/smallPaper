"""Compare selected archived and current source files without importing them."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import ast
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
ROOT = PAPER.parent
ARCHIVE = ROOT / '90_历史与工具/旧期刊投稿材料/《计算机应用与软件》投稿/_work/upload/DEIM_migrated.zip'
CURRENT = Path(r'C:\Users\jiwei\Desktop\DEIM')
OUTPUT = HERE / 'migrated_archive_comparison.json'
PREFIX = 'DEIM_1572862469/'
PAIRS = [
    ('engine/backbone/hgnetv2_MSCA_EnhancedStem.py', 'engine/backbone/hgnetv2_MSCA_EnhancedStem.py'),
    ('engine/extre_module/custom_nn/attention/MSCA.py', 'engine/extre_module/custom_nn/attention/MSCA.py'),
    ('engine/extre_module/custom_nn/attention/simam.py', 'engine/extre_module/custom_nn/attention/simam.py'),
    ('engine/backbone/hgnetv2.py', 'engine/backbone/hgnetv2.py'),
    ('engine/solver/det_solver.py', 'engine/solver/det_solver.py'),
    ('engine/solver/_solver.py', 'engine/solver/_solver.py'),
    ('engine/solver/det_engine.py', 'engine/solver/det_engine.py'),
    ('engine/core/yaml_config.py', 'engine/core/yaml_config.py'),
    ('engine/optim/ema.py', 'engine/optim/ema.py'),
    ('engine/optim/lr_scheduler.py', 'engine/optim/lr_scheduler.py'),
    ('configs/test/deim_hgnetv2_EnhancedStem_n_visdrone.yml', 'configs/test/deim_hgnetv2_EnhancedStem_n_visdrone.yml'),
    ('configs/test/deim_hgnetv2_EnhancedStem_s_visdrone.yml', 'configs/test/deim_hgnetv2_EnhancedStem_s_visdrone.yml'),
    ('configs/test/visdrone2019_mpsma.yml', 'configs/test/visdrone2019.yml'),
]


def digest(data):
    return sha256(data).hexdigest()


def file_digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class StripDocstrings(ast.NodeTransformer):
    def _body(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node
    visit_FunctionDef = _body
    visit_AsyncFunctionDef = _body
    visit_ClassDef = _body


def normal(node):
    return ast.dump(StripDocstrings().visit(deepcopy(node)), include_attributes=False)


def cls(text, name):
    return next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == name)


def method(text, class_name, method_name):
    return next(node for node in cls(text, class_name).body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def agg_branch(text, value):
    method_node = method(text, 'HG_Block', '__init__')
    hits = [node for node in ast.walk(method_node) if isinstance(node, ast.If) and
            isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and
            node.test.left.id == 'agg' and len(node.test.comparators) == 1 and
            isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == value]
    assert len(hits) == 1
    return normal(ast.Module(body=hits[0].body, type_ignores=[]))


def torch_save_calls(fn):
    return [{'line': node.lineno, 'first_argument': ast.unparse(node.args[0]) if node.args else None}
            for node in ast.walk(fn) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.save']


def restore_conditions(fn):
    result = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If) or 'epoch' not in ast.unparse(node.test):
            continue
        if any(isinstance(call, ast.Call) and ('load_resume_state' in ast.unparse(call.func) or
               '_load_best_stg1_if_available' in ast.unparse(call.func))
               for statement in node.body for call in ast.walk(statement)):
            result.append({'line': node.lineno, 'condition': ast.unparse(node.test)})
    return result


def main():
    assert not OUTPUT.exists()
    comparisons, contents = [], {}
    with ZipFile(ARCHIVE) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        assert len(names) == len(set(names)) and len(entries) == 2737
        args_and_logs = [name for name in names if PurePosixPath(name).name in ('args.json', 'log.txt')]
        output_members = [name for name in names if name.startswith(PREFIX + 'outputs/')]
        checkpoints = [name for name in names if name.lower().endswith(('.pt', '.pth'))]
        assert not args_and_logs and not output_members and len(checkpoints) == 2
        for archived_rel, current_rel in PAIRS:
            name = PREFIX + archived_rel
            info = archive.getinfo(name)
            assert info.file_size < 2_000_000 and not info.is_dir()
            archived = archive.read(name)
            current = (CURRENT / current_rel).read_bytes()
            contents[archived_rel] = (archived, current)
            comparisons.append({'archive_member': name, 'archive_sha256': digest(archived),
                                'archive_member_datetime_unverified': list(info.date_time),
                                'current_path': str(CURRENT / current_rel), 'current_sha256': digest(current),
                                'byte_identical': archived == current})
    def pair_text(name):
        left, right = contents[name]
        return left.decode('utf-8'), right.decode('utf-8')
    for name in ['engine/backbone/hgnetv2_MSCA_EnhancedStem.py', 'engine/extre_module/custom_nn/attention/MSCA.py']:
        assert contents[name][0] == contents[name][1]
    for name in ['configs/test/deim_hgnetv2_EnhancedStem_n_visdrone.yml', 'configs/test/deim_hgnetv2_EnhancedStem_s_visdrone.yml']:
        a, b = pair_text(name)
        assert a.replace('\r\n', '\n').replace('visdrone2019_mpsma.yml', 'visdrone2019.yml') == b.replace('\r\n', '\n')
    assert contents['configs/test/visdrone2019_mpsma.yml'][0] == contents['configs/test/visdrone2019_mpsma.yml'][1]
    semantic = {}
    simam_a, simam_b = pair_text('engine/extre_module/custom_nn/attention/simam.py')
    semantic['SimAM_class_without_docstrings'] = normal(cls(simam_a, 'SimAM')) == normal(cls(simam_b, 'SimAM'))
    ema_a, ema_b = pair_text('engine/optim/ema.py')
    semantic['ModelEMA_class_without_docstrings'] = normal(cls(ema_a, 'ModelEMA')) == normal(cls(ema_b, 'ModelEMA'))
    yaml_a, yaml_b = pair_text('engine/core/yaml_config.py')
    for method_name in ['ema', 'global_cfg', 'get_optim_params']:
        semantic['YAMLConfig.' + method_name] = normal(method(yaml_a, 'YAMLConfig', method_name)) == normal(method(yaml_b, 'YAMLConfig', method_name))
    base_a, base_b = pair_text('engine/backbone/hgnetv2.py')
    for class_name, method_name in [('StemBlock', '__init__'), ('StemBlock', 'forward'),
                                     ('HG_Stage', '__init__'), ('HG_Stage', 'forward'), ('HG_Block', 'forward')]:
        semantic[class_name + '.' + method_name] = normal(method(base_a, class_name, method_name)) == normal(method(base_b, class_name, method_name))
    for branch in ['se', 'ese', 'simam']:
        semantic['HG_Block.agg=' + branch] = agg_branch(base_a, branch) == agg_branch(base_b, branch)
    assert all(semantic.values()), semantic
    solver_a, solver_b = pair_text('engine/solver/det_solver.py')
    archive_fit, current_fit = method(solver_a, 'DetSolver', 'fit'), method(solver_b, 'DetSolver', 'fit')
    archive_saves, current_saves = torch_save_calls(archive_fit), torch_save_calls(current_fit)
    assert not archive_saves and len(current_saves) == 1 and "coco_evaluator.coco_eval['bbox'].eval" in current_saves[0]['first_argument']
    archive_restore = restore_conditions(method(solver_a, 'DetSolver', 'save_best_model'))
    current_restore = restore_conditions(current_fit)
    assert len(archive_restore) == 1 and 'epoch ==' in archive_restore[0]['condition']
    assert any('epoch >=' in entry['condition'] for entry in current_restore)
    for archived_rel, current_rel in PAIRS:
        archived, current = contents[archived_rel]
        for group, rel, data in [('archive', archived_rel, archived), ('current', current_rel, current)]:
            path = HERE / 'source_snapshot' / group / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert path.read_bytes() == data
            else:
                path.write_bytes(data)
    result = {'status': 'MIGRATED_ARCHIVE_AND_CURRENT_SOURCE_COMPARED',
              'checked_utc': datetime.now(timezone.utc).isoformat(),
              'archive': {'path': str(ARCHIVE), 'sha256': file_digest(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
                          'members': len(entries), 'args_or_log_members': args_and_logs,
                          'outputs_members': len(output_members), 'checkpoint_members': checkpoints},
              'file_comparisons': comparisons, 'selected_ast_comparisons_without_docstrings': semantic,
              'config_alias': 'Archived n/s config text differs only in included filename. The archived visdrone2019_mpsma.yml bytes equal the current visdrone2019.yml bytes.',
              'solver_difference': {'archive_fit_torch_save_calls': archive_saves,
                                    'current_fit_torch_save_calls': current_saves,
                                    'archive_nonbest_restore_conditions': archive_restore,
                                    'current_fit_restore_conditions': current_restore},
              'limits': ['The archive and current tree may share a source; matching files are not independent replication.',
                         'The archive contains neither the original run logs nor their trained checkpoints, and no cryptographic binding to historical runs.',
                         'ZIP member dates and file mtimes do not authenticate the version used during historical training.',
                         'AST equality covers only explicitly selected classes, methods or branches; it does not establish complete run equivalence.',
                         'The archive fit has no explicit torch.save call for periodic evaluation arrays, but another code path or process could create such files.',
                         'No model, prediction, evaluator or training was run.'],
              'historical_training_source_authenticated': False, 'model_instantiations': 0,
              'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'archive_members': len(entries), 'file_pairs': len(comparisons),
                      'byte_identical_pairs': sum(x['byte_identical'] for x in comparisons),
                      'selected_ast_checks': len(semantic), 'archive_torch_saves_in_fit': len(archive_saves),
                      'current_torch_saves_in_fit': len(current_saves), 'sha256': file_digest(OUTPUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
