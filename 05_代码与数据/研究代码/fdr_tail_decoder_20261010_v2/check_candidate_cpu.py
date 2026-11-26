"""Actual CPU construction/strict-load and native FDR target consistency checks."""
import argparse
import json
from pathlib import Path
import sys
from tail_grid import TailGridBinding,sha


def main(a):
    sys.path.insert(0,str(Path(a.frozen_runtime).resolve()))
    from runtime import verify_bundle,activate,register_model,load_common,read,write
    receipt,plan,paths=verify_bundle(Path(a.bundle).resolve())
    assert plan['arm']=='target_protect'
    activate(paths['native_root']);register_model()
    import torch
    import torch.nn.functional as F
    from engine.core import YAMLConfig
    from engine.deim import dfine_utils as utils
    from engine.deim.dfine_decoder import Integral
    cfg=YAMLConfig(str(Path(a.bundle)/'resolved.yml'))
    model=cfg.model
    assert not torch.cuda.is_initialized() and all(p.device.type=='cpu' for p in model.parameters())
    common=load_common(model,paths['common_init'])
    assert common['strict_keys']==794 and common['parameters']==10231997
    keys=tuple(model.state_dict())
    before={name:value.detach().clone() for name,value in model.state_dict().items()}
    numeric=[]
    base_function=utils.weighting_function
    integral=Integral(32)
    for dtype in (torch.float32,torch.float64):
        up=torch.tensor([.5],dtype=dtype)
        scale=torch.tensor([4.],dtype=dtype)
        old=base_function(32,up,scale)
        assert old.shape==(33,) and old[0]==-4 and old[-1]==4
        with TailGridBinding(1.):
            assert torch.equal(utils.weighting_function(32,up,scale),old)
        with TailGridBinding(2.):
            new=utils.weighting_function(32,up,scale)
            assert new[0]==-8 and new[-1]==8 and torch.equal(new[1:-1],old[1:-1])
            assert torch.equal(utils.weighting_function(32,up,scale,deploy=True)[1:-1],base_function(32,up,scale,deploy=True)[1:-1])
            assert (new[1:]>new[:-1]).all()
            # Finite positive probability construction: all four GT offsets
            # outside old support but strictly inside new support. Not a model.
            desired=torch.tensor([5.,5.5,6.,6.5],dtype=dtype)
            floor=1e-9 if dtype==torch.float32 else 1e-12
            adjusted=(desired-floor*new.sum())/(1.-33*floor)
            left=torch.searchsorted(new,adjusted,right=True)-1
            left=left.clamp(0,31)
            alpha=(adjusted-new[left])/(new[left+1]-new[left])
            probabilities=torch.full((4,33),floor,dtype=dtype)
            rr=torch.arange(4)
            probabilities[rr,left]+=(1.-33*floor)*(1.-alpha)
            probabilities[rr,left+1]+=(1.-33*floor)*alpha
            corners=probabilities.log().reshape(1,1,132).detach().requires_grad_(True)
            distances=integral(corners,new)
            error=float((distances.flatten()-desired).abs().max())
            assert error < (2e-6 if dtype==torch.float32 else 1e-12)
            # Actual native box encoder resolves the exact expanded grid.
            refs=torch.tensor([[[.5,.5,.01,.01]]],dtype=dtype)
            predicted=utils.distance2bbox(refs,distances,scale)
            gt_xyxy=torch.cat((predicted[:,:,:2]-predicted[:,:,2:]*.5,predicted[:,:,:2]+predicted[:,:,2:]*.5),-1)
            labels,wr,wl=utils.bbox2distance(refs.reshape(-1,4).detach().float(),gt_xyxy.reshape(-1,4).detach().float(),32,scale.float(),up.float())
            label_indices=labels.long()
            encoded_grid=utils.weighting_function(32,up.float(),scale.float())
            decoded_target=(wl*encoded_grid[label_indices]+wr*encoded_grid[label_indices+1])
            target_error=float((decoded_target-desired.float()).abs().max())
            assert target_error<1e-4
            # Native local loss interface encodes targets in FP32. This is a
            # synthetic four-edge leaf CE check, not a detector backward.
            loss=F.cross_entropy(corners.reshape(-1,33),label_indices,reduction='none')*wl.to(dtype)
            loss+=F.cross_entropy(corners.reshape(-1,33),label_indices+1,reduction='none')*wr.to(dtype)
            gradient=torch.autograd.grad(loss.sum(),corners)[0]
            assert torch.isfinite(loss).all() and torch.isfinite(gradient).all()
            numeric.append(dict(dtype=str(dtype),continuous_expectation_max_abs_error=error,
                native_FP32_target_encoding_max_abs_error=target_error,finite_synthetic_leaf_gradient=True,
                maximum_abs_leaf_gradient=float(gradient.abs().max())))
        assert utils.weighting_function is base_function
    assert tuple(model.state_dict())==keys and all(torch.equal(model.state_dict()[name],value) for name,value in before.items())
    assert not any(parameter.grad is not None for parameter in model.parameters())
    assert not torch.cuda.is_initialized()
    out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(status='PASS_ACTUAL_CPU_CANDIDATE_STRICT_LOAD_AND_GRID_ENCODING',seed=plan['seed'],
        common=common,numeric_checks=numeric,central31_values_bitwise_preserved=True,
        prediction_and_FGL_target_grid_binding_same=True,identity_factor1_grid_exact=True,
        native_module_global_bindings_restored=True,model_state_unchanged=True,
        parameters=10231997,extra_parameters=0,strict_keys=794,
        candidate_detector_forwards=0,baseline_detector_forwards=0,detector_backwards=0,
        synthetic_four_edge_leaf_gradient_calls=2,CUDA_initialized=False,optimizer_steps=0,epochs_added=0,
        full_training_integration_not_yet_executed=True,GPU_saved_prediction_damage_check_pending=True,
        actual_AP_improvement_not_tested=True,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    write(out/'verification.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True)
    parser.add_argument('--frozen-runtime',required=True);parser.add_argument('--destination',required=True)
    main(parser.parse_args())
