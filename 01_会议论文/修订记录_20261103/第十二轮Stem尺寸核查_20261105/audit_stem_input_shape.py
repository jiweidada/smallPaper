"""Statically audit the documented EnhancedStem input dimensions; no model execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import json
import zipfile

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PAPER = REV.parent
SOURCE = REV / 'source_snapshot/current_code/engine/backbone/hgnetv2_MSCA_EnhancedStem.py'
MANUSCRIPT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
OUTPUT = HERE / 'stem_shape_static_audit.json'
SOURCE_SHA = '8d229219cce4ff8213e38b9f4fc3e6d72f227061b9ecca6a7a322e15d30195bc'
DOC_SHA = '06c51c534c640fe35348bd88ed6c9a2a45c540b35521e378f008fe5a93126809'
FIG_SHA = '4845bfab42ecfaac89ebbfc8b814287fd225bea2b607f25f001420cbc62ac168'

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def find_method(cls, name):
    return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)

def keyword(call, name):
    return next(x.value for x in call.keywords if x.arg == name)

def number(node):
    assert isinstance(node, ast.Constant) and isinstance(node.value, int)
    return node.value

def first_call_named(node, dotted):
    return next(x for x in ast.walk(node) if isinstance(x, ast.Call) and ast.unparse(x.func) == dotted)

def main():
    assert not OUTPUT.exists() and digest(SOURCE) == SOURCE_SHA and digest(MANUSCRIPT) == DOC_SHA
    source = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source)
    stem = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'EnhancedStemBlock')
    stem_init, forward = find_method(stem, '__init__'), find_method(stem, 'forward')
    assignments = {ast.unparse(x.targets[0]): x.value for x in stem_init.body
                   if isinstance(x, ast.Assign) and len(x.targets) == 1}
    main_calls = [x for x in assignments['self.main_branch'].args
                  if isinstance(x, ast.Call) and ast.unparse(x.func) == 'ConvBNAct']
    detail_calls = [x for x in assignments['self.detail_branch'].args
                    if isinstance(x, ast.Call) and ast.unparse(x.func) == 'ConvBNAct']
    assert [number(keyword(x, 'stride')) for x in main_calls] == [2, 1]
    assert [number(keyword(x, 'stride')) for x in detail_calls] == [1, 1]
    assert all(number(keyword(x, 'kernel_size')) == 3 for x in main_calls + detail_calls)
    assert all(not any(k.arg == 'padding' for k in x.keywords) for x in main_calls + detail_calls)
    conv = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ConvBNAct')
    conv_init = find_method(conv, '__init__')
    defaults = dict(zip([x.arg for x in conv_init.args.args[-len(conv_init.args.defaults):]], conv_init.args.defaults))
    assert ast.literal_eval(defaults['padding']) == ''
    conv2d = [x for x in ast.walk(conv_init) if isinstance(x, ast.Call) and ast.unparse(x.func) == 'nn.Conv2d']
    assert any(ast.unparse(keyword(x, 'padding')) == '(kernel_size - 1) // 2' for x in conv2d
               if any(k.arg == 'padding' for k in x.keywords))
    pools = [x for x in ast.walk(forward) if isinstance(x, ast.Call) and ast.unparse(x.func) == 'F.avg_pool2d']
    pool = next(x for x in pools if ast.unparse(x.args[0]) == 'detail')
    assert number(keyword(pool, 'kernel_size')) == 2 and number(keyword(pool, 'stride')) == 2
    assert not any(k.arg in ['padding', 'ceil_mode'] for k in pool.keywords)
    interpolate = first_call_named(forward, 'F.interpolate')
    assert ast.unparse(keyword(interpolate, 'size')) == '(main.shape[2], main.shape[3])'
    cat = first_call_named(forward, 'torch.cat')
    assert ast.unparse(cat.args[0]) == '[main, detail, high_freq]' and number(keyword(cat, 'dim')) == 1
    down = assignments['self.final_downsample']
    assert number(keyword(down, 'kernel_size')) == 3 and number(keyword(down, 'stride')) == 2
    assert not any(k.arg == 'padding' for k in down.keywords)

    configs = []
    for name in ['N1', 'N4', 'S1']:
        path = REV / 'source_snapshot' / name / 'args.json'
        cfg = json.loads(path.read_text(encoding='utf-8'))['yaml_cfg']
        assert cfg['DEIM']['backbone'] == 'HGNetv2_MSCA_EnhancedStem'
        sizes = {}
        for split in ['train', 'val']:
            ops = cfg[split + '_dataloader']['dataset']['transforms']['ops']
            resize = [op['size'] for op in ops if op.get('type') == 'Resize']
            assert resize == [[640, 640]]
            sizes[split] = resize[0]
        assert cfg.get('BatchImageCollateFunction', {}).get('base_size_repeat') is None
        configs.append({'run': name, 'args_sha256': digest(path), 'resizes': sizes,
                        'base_size_repeat': None})
    with zipfile.ZipFile(MANUSCRIPT) as z:
        figure = z.read('word/media/image9.png')
        assert sha256(figure).hexdigest() == FIG_SHA
    cases = []
    for dimension in [640, 641, 642, 644]:
        main_size = (dimension + 1) // 2
        detail_size = dimension // 2
        compatible = main_size == detail_size
        cases.append({'input_dimension': dimension, 'main_size': main_size, 'detail_size': detail_size,
                      'can_concatenate': compatible,
                      'final_size_if_compatible': (main_size + 1) // 2 if compatible else None})
    assert cases == [
        {'input_dimension': 640, 'main_size': 320, 'detail_size': 320, 'can_concatenate': True, 'final_size_if_compatible': 160},
        {'input_dimension': 641, 'main_size': 321, 'detail_size': 320, 'can_concatenate': False, 'final_size_if_compatible': None},
        {'input_dimension': 642, 'main_size': 321, 'detail_size': 321, 'can_concatenate': True, 'final_size_if_compatible': 161},
        {'input_dimension': 644, 'main_size': 322, 'detail_size': 322, 'can_concatenate': True, 'final_size_if_compatible': 161}]
    result = {'status': 'STEM_INPUT_SHAPE_STATICALLY_CHECKED_NO_MODEL_EXECUTION',
              'checked_utc': datetime.now(timezone.utc).isoformat(),
              'manuscript_sha256': DOC_SHA, 'figure_png_sha256': FIG_SHA,
              'source_path': str(SOURCE), 'source_sha256': SOURCE_SHA,
              'source_lines': {'convbnact': [conv.lineno, conv.end_lineno],
                               'enhanced_stem': [stem.lineno, stem.end_lineno],
                               'detail_pool': pool.lineno, 'concat': cat.lineno},
              'current_configs': configs, 'one_dimensional_cases': cases,
              'inference': 'With the inspected default padding, stride-two convolution gives ceil(D/2); detail pooling gives floor(D/2). Odd dimensions cannot concatenate. At 640 both give 320 and the final stride-two convolution gives 160.',
              'scope': 'Static source, saved configuration and integer shape arithmetic only. Does not establish an immutable link to historical training source; no model, evaluation or training executed.'}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'cases': cases}, ensure_ascii=False))

if __name__ == '__main__':
    main()
