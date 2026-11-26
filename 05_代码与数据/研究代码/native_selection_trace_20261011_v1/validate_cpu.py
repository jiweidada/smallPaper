"""Test actual source-extracted native selection on CPU, never a detector forward."""
from pathlib import Path
from types import MethodType
import argparse, ast, hashlib, json
import torch
from selection_trace import NativeSelectionObserver

EXPECTED_SOURCE_SHA='b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(source, output):
    torch.set_num_threads(1)
    assert sha(source)==EXPECTED_SOURCE_SHA
    tree=ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DFINETransformer')
    nodes=[next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ('_generate_anchors','_select_topk')]
    ns={'torch':torch}; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    class Fixture(torch.nn.Module):
        _generate_anchors=ns['_generate_anchors']
        _select_topk=ns['_select_topk']
        def __init__(self):
            super().__init__(); self.query_select_method='default'; self.mgsa_groups=0; self.o2m_t=False
            self.eval_spatial_size=[640,640]; self.feat_strides=[8,16,32]; self.eps=0.01
    d=Fixture(); anchors, valid=d._generate_anchors()
    assert anchors.shape==(1,8400,4) and valid.shape==(1,8400,1) and (~valid).any()
    cases=[]
    for dtype in (torch.float32,torch.float16):
        for batch in (1,4):
            for tie in (False,True):
                for training in (False,True):
                    d.train(training); g=torch.Generator().manual_seed(41)
                    memory=torch.randn(batch,8400,8,generator=g).to(dtype)
                    logits=torch.randn(batch,8400,10,generator=g).to(dtype)
                    if tie: logits.zero_()
                    priors=anchors.repeat(batch,1,1)
                    inputs=[x.clone() for x in (memory,logits,priors)]
                    rng=torch.get_rng_state().clone()
                    baseline=d._select_topk(memory,logits,priors,300)
                    witness={}
                    def counted(self,*args):
                        witness['calls']=witness.get('calls',0)+1
                        witness['result']=ns['_select_topk'](self,*args)
                        return witness['result']
                    original_method=MethodType(counted,d); d._select_topk=original_method
                    with NativeSelectionObserver(d,mode='synthetic_engineering') as observer:
                        actual=d._select_topk(memory,logits,priors,300)
                        assert actual is witness['result'] and witness['calls']==1 and observer.calls==1
                        record=observer.take_record(); ids=record['selected_indices']
                        assert ids.shape==(batch,300) and ids.dtype==torch.int64
                        assert torch.equal(actual[0],memory.gather(1,ids[...,None].expand(-1,-1,8)))
                        assert torch.equal(actual[2],priors.gather(1,ids[...,None].expand(-1,-1,4)))
                        assert torch.equal(record['dense_class_logits'],logits) and torch.equal(record['dense_anchor_logits'],priors)
                        assert torch.equal(record['selected_anchor_logits'],actual[2])
                        for a,b in zip(actual,baseline):
                            assert (a is b is None) or (a is not None and b is not None and torch.equal(a,b))
                    assert d._select_topk is original_method
                    del d._select_topk
                    assert torch.equal(torch.get_rng_state(),rng)
                    assert all(torch.equal(a,b) for a,b in zip(inputs,(memory,logits,priors)))
                    assert not torch.cuda.is_initialized()
                    cases.append(dict(dtype=str(dtype),batch=batch,tied_logits=tie,training=training,exact_native_return=True,actual_indices_checked=True,RNG_unchanged=True,inputs_unchanged=True,one_native_topk=True))
    # A deliberately selected invalid anchor stays invalid; the observer must
    # preserve this behavior rather than silently altering native candidates.
    d.eval(); invalid_ids=(~valid[0,:,0]).nonzero().flatten()[:300]
    logits=torch.zeros(1,8400,10); logits[:,invalid_ids,:]=100
    memory=torch.arange(8400,dtype=torch.float32)[None,:,None]
    with NativeSelectionObserver(d) as observer:
        result=d._select_topk(memory,logits,anchors,300); record=observer.take_record()
        assert torch.isinf(result[2]).all() and torch.isinf(record['selected_anchor_logits']).all()
    assert '_select_topk' not in d.__dict__
    # Exceptions must also restore class lookup and the TorchFunctionMode stack.
    error_observer=None
    try:
        with NativeSelectionObserver(d) as error_observer:
            d._select_topk(memory,logits,anchors,8401)
    except RuntimeError:
        pass
    else: raise AssertionError('Expected native out-of-range Top-K error')
    assert '_select_topk' not in d.__dict__ and not error_observer.entered and error_observer.record is None
    torch.topk(torch.ones(4),2)
    d.train()
    try: NativeSelectionObserver(d).__enter__()
    except ValueError: pass
    else: raise AssertionError('Training diagnostic accepted')
    assert '_select_topk' not in d.__dict__
    assert not torch.cuda.is_initialized()
    report=dict(status='PASS_SOURCE_EXTRACTED_NATIVE_SELECTION_CPU_FIXTURES',torch_version=torch.__version__,
        source_sha256=sha(source),code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        cases=cases,invalid_anchor_behavior_preserved=True,exception_teardown_passed=True,training_diagnostic_rejected=True,
        dense_positions=8400,selected_queries=300,additional_native_topk_calls_per_observed_call=0,
        detector_forwards=0,bbox_head_calls=0,optimizer_updates=0,CUDA_initialized=False,
        claim_boundary='CPU actual-selector engineering equivalence only. No full-detector GPU replay, image IDs, trained feature diagnosis or efficacy validation has been completed by this test.')
    with Path(output).open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(status=report['status'],cases=len(cases),source_sha256=report['source_sha256'],output_sha256=sha(output))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
