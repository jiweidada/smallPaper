"""Preserve frozen readers and route a new reader to the four-job reference."""
from pathlib import Path
import ast, shutil
code = Path('英文小论文/code')
out = Path('outputs/native_stem_scratch12_20261011_v3')
src = code/'native_stem_scratch12_monitor_20261011_v1'
dst = code/'native_stem_scratch12_monitor_20261011_v2'
assert not dst.exists(); dst.mkdir()
for p in src.glob('*.py'):
    value = p.read_bytes().replace(b'native_stem_scratch12_monitor_20261011_v1',b'native_stem_scratch12_monitor_20261011_v2')
    value = value.replace(b'native_stem_scratch12_20261011_v2',b'native_stem_scratch12_20261011_v3')
    (dst/p.name).write_bytes(value); ast.parse(value)
src = code/'stem_init_stage_analysis_20261011_v1'
dst = code/'stem_init_stage_analysis_20261011_v2'
assert not dst.exists(); dst.mkdir()
for p in src.glob('*.py'):
    value = p.read_bytes().replace(b'native_stem_scratch12_monitor_20261011_v1',b'native_stem_scratch12_monitor_20261011_v2')
    value = value.replace(b'stem_init_stage_analysis_v1',b'stem_init_stage_analysis_v2')
    (dst/p.name).write_bytes(value); ast.parse(value)
print('NEW_READERS_CREATED_OLD_READERS_UNCHANGED')
