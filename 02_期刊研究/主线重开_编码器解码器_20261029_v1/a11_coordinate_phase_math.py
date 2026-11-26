"""CPU source-operator/math audit. No detector inference,optimizer or fitted model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import ast
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def spread(p, grid):
    mean=np.dot(p,grid)
    variance=np.dot(p,(grid-mean)**2)
    left=int(np.clip(np.searchsorted(grid,mean,side='right')-1,0,len(grid)-2))
    floor=(mean-grid[left])*(grid[left+1]-mean)
    return float(mean),float(variance),float(floor),float(variance-floor)


def main():
    parser=argparse.ArgumentParser()
    for name in ('source-root','checkpoint','protocol','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    protocol=json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    output=Path(args.output)
    output.mkdir()
    assert sha(Path(args.checkpoint))==protocol['checkpoint_sha256']
    source=Path(args.source_root)/'src/zoo/dome/dome_utils.py'
    assert sha(source)==protocol['source_hashes']['src/zoo/dome/dome_utils.py']
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model']
    up_values={k:v.tolist() for k,v in state.items() if k.endswith('.up')}
    scale_values={k:v.tolist() for k,v in state.items() if k.endswith('.reg_scale')}
    assert up_values and scale_values
    assert all(v==[.5] for v in up_values.values())
    assert all(v==[4.] for v in scale_values.values())
    namespace={'torch':torch,'Tensor':torch.Tensor}
    names=['weighting_function','translate_gt']
    body=[n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(body)==2
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),namespace)
    up=torch.tensor([.5],dtype=torch.float32)
    scale=torch.tensor([4.],dtype=torch.float32)
    grid_t=namespace['weighting_function'](32,up,scale)
    grid=grid_t.numpy().astype(np.float64)
    assert len(grid)==33 and np.all(np.diff(grid)>0)
    rows=[]
    for i in range(32):
        for alpha in protocol['alphas']:
            value=np.float32(grid[i]+alpha*(grid[i+1]-grid[i]))
            indices,wr,wl=namespace['translate_gt'](torch.tensor([value]),32,scale,up)
            left=int(indices[0].item())
            p=np.zeros(33,np.float64)
            p[left]=float(wl[0]);p[left+1]=float(wr[0])
            p=p/p.sum()
            mean,var,floor,excess=spread(p,grid)
            entropy=-float(np.sum(p[p>0]*np.log(p[p>0])))
            # Independent exact two-active-bin softmax derivatives. No network parameters.
            target=torch.tensor([1-alpha,alpha],dtype=torch.float64)
            logits=torch.log(target).requires_grad_(True)
            probs=logits.softmax(0)
            ce=-(target*probs.log()).sum()
            entropy_t=-(probs*probs.log()).sum()
            mean_t=(probs*torch.tensor(grid[i:i+2],dtype=torch.float64)).sum()
            grad_ce=torch.autograd.grad(ce,logits,retain_graph=True)[0]
            grad_h=torch.autograd.grad(entropy_t,logits,retain_graph=True)[0]
            grad_mean=torch.autograd.grad(mean_t,logits)[0]
            mean_descent_derivative=-float(torch.dot(grad_h,grad_mean))
            analytic=-2*alpha**2*(1-alpha)**2*(grid[i+1]-grid[i])*np.log((1-alpha)/alpha)
            assert abs(mean_descent_derivative-analytic)<1e-12
            assert float(grad_ce.abs().max())<1e-12
            assert abs(mean-float(value))<=2e-6
            assert abs(excess)<1e-12
            rows.append({'interval':i,'alpha':alpha,'target':float(value),'decoded_mean':mean,'reconstruction_abs_error':abs(mean-float(value)),'entropy':entropy,'variance':var,'minimum_variance_for_mean':floor,'excess_variance':excess,'entropy_descent_mean_derivative':mean_descent_derivative,'CE_logit_gradient_max':float(grad_ce.abs().max())})
    rng=np.random.Generator(np.random.PCG64(protocol['seed']))
    random_p=rng.dirichlet(np.full(33,.25),size=128)
    random_spread=np.asarray([spread(p,grid) for p in random_p])
    assert random_spread[:,-1].min()>=-1e-12
    # Entropy subtraction is not a valid nonnegative excess-spread measure.
    toy_grid=np.asarray([0.,1.,100.])
    ideal=np.asarray([.5,.5,0.])
    distant=np.asarray([.995,0.,.005])
    h_ideal=-np.sum(ideal[ideal>0]*np.log(ideal[ideal>0]))
    h_distant=-np.sum(distant[distant>0]*np.log(distant[distant>0]))
    assert np.isclose(np.dot(ideal,toy_grid),np.dot(distant,toy_grid))
    assert h_distant<h_ideal and spread(distant,toy_grid)[-1]>0
    np.savez_compressed(output/'numeric_cases.npz',grid=grid,random_probabilities=random_p,random_spread=random_spread)
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_SOURCE_INTERPOLATION_AND_ANALYTICAL_GRADIENT_AUDIT','checkpoint_up':up_values,'checkpoint_reg_scale':scale_values,'source_commit':protocol['source_commit'],'source_hashes':protocol['source_hashes'],'grid':grid.tolist(),'adjacent_cases':len(rows),'random_distributions':128,'max_source_reconstruction_error':max(r['reconstruction_abs_error'] for r in rows),'max_ideal_excess_variance_abs':max(abs(r['excess_variance']) for r in rows),'random_excess_variance_min':float(random_spread[:,-1].min()),'nonmidpoint_entropy_drift_cases':sum(abs(r['entropy_descent_mean_derivative'])>1e-12 for r in rows),'rows':rows,'entropy_subtraction_counterexample':{'grid':toy_grid.tolist(),'same_mean':.5,'ideal_entropy':float(h_ideal),'distant_entropy':float(h_distant),'distant_excess_variance':spread(distant,toy_grid)[-1]},'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','full_model_forwards':0,'fitted_models':0,'optimizer_updates':0,'mainline_validated':False,'not_proven':['Existing model distributions actually exhibit this as a limiting mechanism.','Any proposed measure is calibrated predictive uncertainty.','Any detection accuracy or training benefit.','Novelty of the mathematical bound or algorithm.'],'interpretation':'Basic interpolation/variance identities and entropy-gradient tradeoff only. Source target encoding may have nonzero entropy at zero coordinate error. Not a model-performance experiment or reproduction/refutation of complete UGS.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    completed={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED','protocol_sha256':sha(Path(args.protocol)),'script_sha256':sha(Path(__file__)),'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()]}
    (output/'completed.json').write_text(json.dumps(completed,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','adjacent_cases','random_distributions','max_source_reconstruction_error','max_ideal_excess_variance_abs','nonmidpoint_entropy_drift_cases','full_model_forwards','fitted_models')}))


if __name__=='__main__':
    main()
