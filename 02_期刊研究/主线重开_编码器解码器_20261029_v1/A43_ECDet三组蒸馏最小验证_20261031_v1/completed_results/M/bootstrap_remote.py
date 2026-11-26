"""Copy only already-authorized verified weights into the isolated A43 run."""
import hashlib
import json
from pathlib import Path
import shutil

R = Path(__file__).resolve().parent
PREVIOUS = R.parent / 'a42_ecdet_three_arm_20261031_v1'
assets = [
    ('common_init_10class.pth', PREVIOUS / 'common_init_10class.pth',
     '2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6'),
    ('teacher_B_epoch12.pth', PREVIOUS / 'arm_B/epoch12.pth',
     'bbd5712fe42bc7b11e3b01d46e930f7baa3992dd76832a913b367a58626a6ad6'),
]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for name, source, expected in assets:
    assert sha(source) == expected
    dest = R / name
    if dest.exists():
        assert sha(dest) == expected
    else:
        shutil.copyfile(source, dest)
    assert sha(dest) == expected
    rows.append({'name': name, 'reused_source': str(source), 'bytes': dest.stat().st_size,
                 'sha256': expected, 'source_modified': False})
(R / 'remote_asset_preparation.json').write_text(json.dumps({'status': 'PASS', 'assets': rows,
    'new_model_forwards': 0, 'new_optimizer_updates': 0}, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'verified_assets': len(rows), 'training_started': False}))
