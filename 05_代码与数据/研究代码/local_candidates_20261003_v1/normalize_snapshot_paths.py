"""Normalize local manifest paths for copying toLinux; raw source bytes stay unchanged."""
import json
from pathlib import Path
root = Path(__file__).resolve().parent
p = root / 'native_source_manifest.json'
with (root / 'native_source_manifest_windows_v1.json').open('xb') as f: f.write(p.read_bytes())
m = json.loads(p.read_text(encoding='utf8'))
for row in m['files']: row['local'] = row['local'].replace('\\', '/')
p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf8')
readme = root / 'README.md'
s = readme.read_text(encoding='utf8').replace('cpu_integration_v3.json', 'cpu_integration_v4.json').replace('当前以v3为准', '当前以v4为准')
s = s.replace('parameter_counts_actual.json', 'parameter_counts_actual_v2.json').replace('local_delivery_manifest.json', 'local_delivery_manifest_v2.json')
s += '\n复制到Linux时排除`local_deps`和`__pycache__`，使用目标运行时自己的依赖；本地依赖中有Windows编译包。源码路径已规范为可移植相对路径，GPU入口不会加载Windows依赖。\n'
readme.write_text(s, encoding='utf8')
