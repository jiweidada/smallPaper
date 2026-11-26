"""Read-only inputs and exclusive-write evidence for the 342 CPU review run."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import sys, json, hashlib
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
SPEC = WORK.parents[1] / '持续寻找研究主线_20261020_v2'
CORE = SPEC / '311_unit_instance_core_v2' / 'core.py'
GCAL = SPEC / '320_geometry_calibrated_core_v2' / 'gcal.py'
for p in (WORK, CORE.parent, GCAL.parent):
    if str(p) not in sys.path:
        sys.path.append(str(p))

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(name, obj):
    assert name.startswith('342_')
    with open(HERE / name, 'x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def write_text(name, value):
    assert name.startswith('342_')
    with open(HERE / name, 'x', encoding='utf-8', newline='\n') as f:
        f.write(value)

def inputs():
    index = json.loads((SPEC / '327_训练实施输入索引.json').read_text(encoding='utf-8'))
    ids = sorted(index['sample_plan']['development_ids'])[:32]
    assert len(ids) == 32 and len(set(ids)) == 32
    return index, ids

def source_shas():
    paths = list(HERE.glob('*.py')) + [CORE, GCAL]
    paths += [p for p in WORK.glob('*.py')]
    paths += [SPEC / n for n in (
        '342_总指令_修几何链路至开训前审查包.txt',
        '317_A_单位解释残余补全_四臂有限验证方案_v3.txt',
        '317v4_G臂kappa_c学习率修订.txt',
        '296_A_单位实例残余补全_具体实现提案.txt',
        '327_训练实施输入索引.json')]
    return {str(p): sha(p) for p in paths}

def bootstrap():
    import torch, numpy, PIL
    index, ids = inputs()
    before = source_shas()
    # Specification files are hashed as bytes only; no heldout images, predictions,
    # annotations, or sample-plan heldout list is opened by this run.
    protected = list(SPEC.rglob('*.py')) + list(SPEC.glob('*.txt')) + list(SPEC.glob('*.json'))
    before.update({str(p): sha(p) for p in protected})
    write_json('342_输入快照.json', {
        'source_sha256': before, 'development_ids': ids,
        'environment': {'python': sys.executable, 'python_version': sys.version,
                        'torch': torch.__version__, 'numpy': numpy.__version__, 'pillow': PIL.__version__,
                        'device': 'cpu', 'threads': 2},
        'legacy_geom_is_preserved': str(WORK / 'geom342.py'),
        'output_directory': str(HERE),
        'authorization': '342 CPU implementation/unit tests/20-step smoke only; stop after stage 2',
        '已执行': {'source_hash_passes': 1},
        '未执行': ['真实训练', 'GPU', 'SSH', '队列', 'heldout3236读取', '阶段3至5'],
        '未核': ['科学问题12项方向结果']})

if __name__ == '__main__':
    bootstrap()
