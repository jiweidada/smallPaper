"""Replay only random draws/constructors from the completed CPU check; no forward."""
from pathlib import Path
import sys
import json
import hashlib
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent.parent / 'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
import numpy as np
from regional_range_attention_v1 import RegionalRangeContext, COUNTS

cpu = json.loads((P / 'CPU_component_result_v1.json').read_text('utf-8'))
assert cpu['status'] == 'PASS_SYNTHETIC_OPERATOR_AND_GRADIENT_CHECKS'
assert hashlib.sha256((P / 'regional_range_attention_v1.py').read_bytes()).hexdigest() == cpu['source_sha256']
torch.random.default_generator.manual_seed(20261102)
q = torch.randn(2, 2, 5, 3, 7, dtype=torch.float64)
k = torch.randn(2, 2, 6, 4, 7, dtype=torch.float64)
x = torch.randn(2, 6, 5, 7, dtype=torch.float64)
unused_B = RegionalRangeContext(6, 2, 2, 2, 'B').double()
m = RegionalRangeContext(6, 2, 2, 2, 'M').double()
with torch.no_grad():
    m.gate.fill_(.3)
values = {'x': x.numpy(), 'route_q': q.numpy(), 'route_k': k.numpy()}
values.update({'state::' + key: tensor.detach().numpy() for key, tensor in m.state_dict().items()})
target = P / 'synthetic_GPU_fixture_v1.npz'
assert not target.exists()
np.savez_compressed(target, **values)
assert COUNTS == {'module_forwards': 0, 'M_route_calls': 0, 'Q_route_calls': 0}
assert not torch.cuda.is_initialized()
record = {'status': 'EXPORTED_EXISTING_CPU_CHECK_FIXTURE_WITHOUT_FORWARD',
          'fixture_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'fixture_bytes': target.stat().st_size, 'CPU_result_sha256': hashlib.sha256((P / 'CPU_component_result_v1.json').read_bytes()).hexdigest(),
          'CPU_component_constructors': 2, 'additional_component_forwards': 0,
          'additional_full_model_forwards': 0, 'optimizer_updates': 0,
          'real_detector_checkpoint_included': False}
with (P / 'synthetic_GPU_fixture_export_v1.json').open('x', encoding='utf-8') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
