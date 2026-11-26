"""CPU-only check: candidate instrumentation reproduces native profiling RNG."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
import p2_candidate
from engine.core import YAMLConfig
from engine.misc import stats
from screen_runtime import strict_common_init, install_runtime, NEW_KEY


def reset():
    random.seed(727);np.random.seed(727);torch.manual_seed(727)


def capture():
    return (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())


def main():
    root=Path(sys.argv[1]);init=sys.argv[2];torch.set_num_threads(6)
    b=YAMLConfig(str(root/'baseline132_reference.yml'));p=YAMLConfig(str(root/'p2_full132.yml'))
    b.device=p.device='cpu';b.yaml_cfg['device']=p.yaml_cfg['device']='cpu'
    strict_common_init(b.model,init);strict_common_init(p.model,init)
    assert all(torch.equal(v,p.model.state_dict()[k]) for k,v in b.model.state_dict().items())
    reset();stats(b);native=capture()
    install_runtime(p,root/'cpu_profile_rng_fixture')
    from engine.solver import det_solver
    reset();det_solver.stats(p);replayed=capture()
    assert torch.equal(native[0],replayed[0])
    assert native[1]==replayed[1]
    assert native[2][0]==replayed[2][0] and np.array_equal(native[2][1],replayed[2][1]) and native[2][2:]==replayed[2][2:]
    assert p.model.__class__.__name__=='P2CandidateDEIM'
    assert p.model.decoder._get_decoder_input.__self__ is p.model.decoder
    assert torch.count_nonzero(p.model.state_dict()[NEW_KEY])==0
    receipt={'passed':True,'CPU_torch_RNG_after_profile_matches_native':True,
             'python_numpy_RNG_after_profile_matches_native':True,
             'public_weights_and_zero_branch_unchanged':True,
             'CPU_rng_digest':hashlib.sha256(native[0].numpy().tobytes()).hexdigest(),
             'scope':'CPU-only native profiling replay; full132 GPU has not started'}
    (root/'full132_cpu_checks.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
