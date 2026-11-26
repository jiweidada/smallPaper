"""Validate source receipts before the frozen local parameter-free audit."""
from pathlib import Path
import json
from a10_dn_geometry import sha, main
root=Path(__file__).resolve().parent
out=root/'A10_跨实例负去噪参考框几何审计_v1'
eq=json.loads((out/'author_operator_equivalence.json').read_text(encoding='utf-8'))
assert eq['status']=='PASS_PARAMETER_FREE_AUTHOR_OPERATOR_EQUIVALENCE'
assert eq['wrapper_sha256']==sha(root/'a10_author_operator_equivalence_compat_v1.py')
assert eq['numpy_operator_sha256']==sha(root/'a10_dn_geometry.py')
assert eq['device']=='CPU' and eq['learned_parameters']==0
for record in eq['source_files']:
    assert record['sha256']==sha(root/'official_assets_v1'/'Dome_DETR_2dde3bc_v1'/'source'/record['relative_path'])
main()
