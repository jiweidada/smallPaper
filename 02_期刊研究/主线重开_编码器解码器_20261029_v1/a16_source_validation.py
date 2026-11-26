"""Bounded CPU-double tests of the exact source sampling core; no model load."""
from pathlib import Path
import ast,hashlib,json,sys
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from a16_spatial_operator import project,pack,pooled

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent/'dome_assets_v1'
OLD=ROOT.parent/'a15_sampling_center_v1'

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    phase=sys.argv[1];assert phase in ['before_capture','after_capture']
    output=ROOT/('source_validation_'+phase+'.json');assert not output.exists()
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    src=ASSETS/'source'/'src/zoo/dome/utils.py'
    assert hashlib.sha256(src.read_bytes()).hexdigest()==p['source_hashes']['src/zoo/dome/utils.py']
    node=next(n for n in ast.parse(src.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='deformable_attention_core_func_v2')
    ns={'torch':torch,'F':F,'List':List};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
    core=ns[node.name];torch.set_num_threads(2)
    with np.load(ROOT/'final_sampler_state.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    existing=[json.loads(s) for s in (ROOT/'source_operator_calls.jsonl').read_text(encoding='utf-8').splitlines()] if (ROOT/'source_operator_calls.jsonl').exists() else []
    assert not any(r['phase']==phase for r in existing),'No automatic source-validation retry.'
    core_calls=linear_calls=0;tests=[]
    def journal(kind):
        already=sum(r['kind']==kind for r in existing)
        cap=p['operator_caps'][kind]
        assert already<cap
        existing.append({'phase':phase,'kind':kind})
        with (ROOT/'source_operator_calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'phase':phase,'kind':kind})+'\n')
    def test(name,q,r,values,shapes):
        nonlocal core_calls,linear_calls
        data=pack(q,r,values,shapes,state);offset,a,loc=project(q,r,state)
        qt=torch.tensor(q,dtype=torch.float64)
        journal('CPU_DOUBLE_OFFSET_LINEAR');linear_calls+=1
        oo=F.linear(qt,torch.tensor(state['sampling_offsets_weight'],dtype=torch.float64),torch.tensor(state['sampling_offsets_bias'],dtype=torch.float64)).numpy().reshape(-1,8,16,2)
        journal('CPU_DOUBLE_ATTENTION_LINEAR');linear_calls+=1
        aa=F.linear(qt,torch.tensor(state['attention_weights_weight'],dtype=torch.float64),torch.tensor(state['attention_weights_bias'],dtype=torch.float64)).reshape(-1,8,16).softmax(-1).numpy()
        assert np.allclose(oo,offset,atol=1e-10,rtol=1e-12) and np.allclose(aa,a,atol=1e-12,rtol=1e-12)
        v=[torch.tensor(x[None],dtype=torch.float64) for x in values]
        errors=[]
        for dx,dy in [(0.,0.),(.125,0.),(-.125,0.),(0.,.125),(0.,-.125)]:
            core_calls+=1;journal('AUTHOR_CPU_DOUBLE_SAMPLING_CORE')
            actual=core(v,[list(map(int,s)) for s in shapes],torch.tensor((loc+np.asarray([dx,dy])/800)[None],dtype=torch.float64),torch.tensor(a[None],dtype=torch.float64),[4,4,4,4])
            expected=pooled(data,(dx,dy));got=actual[0].numpy()
            errors.append(float(np.max(np.abs(got-expected))))
            assert np.allclose(got,expected,atol=1e-8,rtol=1e-10),(name,dx,dy,errors[-1])
        tests.append({'name':name,'max_absolute_output_error':max(errors),'queries':len(q),'points_per_query':128})
    if phase=='before_capture':
        rng=np.random.default_rng(20261030160)
        shapes=np.asarray([[7,11],[4,6],[3,5],[2,4]])
        for kind in ['random_rectangular','constant_with_boundary','coordinate_ramp_with_boundary']:
            q=rng.normal(size=(5,256)).astype(np.float32)*.1
            r=np.asarray([[.5,.5,.2,.2],[.001,.001,.2,.2],[.999,.999,.2,.2],[-.2,.5,.2,.2],[1.2,.5,.2,.2]],dtype=np.float32)[:,None,:]
            values=[]
            for h,w in shapes:
                if kind=='random_rectangular':v=rng.normal(size=(8,32,h*w)).astype(np.float32)
                elif kind=='constant_with_boundary':v=np.full((8,32,h*w),2.5,dtype=np.float32)
                else:
                    yy,xx=np.mgrid[:h,:w];v=np.broadcast_to((xx+2*yy)[None,None,:,:],(8,32,h,w)).copy().astype(np.float32).reshape(8,32,-1)
                values.append(v)
            test(kind,q,r,values,shapes)
        with np.load(OLD/'captures'/'case00.npz',allow_pickle=False) as c:
            qs=c['gt_query_indices'];test('old_case00_actual_native',c['ca_query'][qs],c['reference'][qs],tuple(c['value_%d'%l] for l in range(4)),c['spatial_shapes'])
        assert core_calls==20 and linear_calls==8
    else:
        with np.load(ROOT/'source_validation_case00.npz',allow_pickle=False) as c:
            test('new_case00_actual_native',c['ca_query'],c['reference'],tuple(c['value_%d'%l] for l in range(4)),c['spatial_shapes'])
        assert core_calls==5 and linear_calls==2
    write(output,{'status':'PASS_SOURCE_CORE_CPU_DOUBLE_AGREEMENT','phase':phase,'tests':tests,
        'core_calls':core_calls,'linear_calls':linear_calls,'new_complete_model_forwards':0,'suffix_forwards':0,
        'scope':'NumPy3x3patch interpolation agrees with exact author core in CPU FP64. Not bitwise GPUFP32 reproduction or a full image-shift Jacobian. Projection checks use fixed native checkpoint parameters.',
        'response_metrics_computed':False})
    print(json.dumps({'status':'PASS_SOURCE_CORE_CPU_DOUBLE_AGREEMENT','phase':phase,'core_calls':core_calls,'linear_calls':linear_calls,'maximum_error':max(t['max_absolute_output_error'] for t in tests)}))

if __name__=='__main__':main()
