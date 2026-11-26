"""Actual native producer methods in a synthetic CPU fixture; no detector replay."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
import torch
from torch import nn
from torch.nn import functional as F

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(native_source, native_code, out):
    torch.set_num_threads(1)
    assert sha(native_source)=='b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5'
    assert sha(Path(native_code)/'stage_trace.py')=='d9668777da440da708ecb3779e6cefc4810abfe65480b6b11520463ac7b896a5'
    sys.path.insert(0,str(native_code))
    from selection_stage_trace import SelectionStageTrace
    tree=ast.parse(Path(native_source).read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DFINETransformer')
    names=('_generate_anchors','_select_topk','_get_decoder_input')
    nodes=[next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in names]
    ns={'torch':torch,'F':F};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(native_source),'exec'),ns)
    class Layer(nn.Module):
        def forward(self,content,reference):return content
    class Inner(nn.Module):
        def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer()])
        def forward(self,content,reference):
            content=self.layers[0](content,reference[:,:,None,:]);logits=content[:,:,:1].repeat(1,1,10)
            return reference[None],logits[None],None,None,reference,logits
    class Producer(nn.Module):
        _generate_anchors=ns['_generate_anchors'];_select_topk=ns['_select_topk'];_get_decoder_input=ns['_get_decoder_input']
        def __init__(self):
            super().__init__();self.query_select_method='default';self.mgsa_groups=0;self.o2m_t=False;self.learn_query_content=False
            self.eval_spatial_size=[640,640];self.feat_strides=[8,16,32];self.eps=0.01;self.num_queries=300
            self.enc_output=nn.Identity();self.enc_score_head=nn.Linear(8,10);self.enc_bbox_head=nn.Linear(8,4);self.decoder=Inner()
            with torch.no_grad():
                self.enc_score_head.weight.fill_(0.001);self.enc_score_head.bias.zero_();self.enc_bbox_head.weight.fill_(0.0001);self.enc_bbox_head.bias.zero_()
            a,v=self._generate_anchors();self.register_buffer('anchors',a);self.register_buffer('valid_mask',v)
    class Model(nn.Module):
        def __init__(self):super().__init__();self.decoder=Producer()
        def forward(self,samples):
            memory=torch.arange(8400*8,dtype=torch.float32).reshape(1,8400,8).repeat(len(samples),1,1)/10000
            content,raw,*_=self.decoder._get_decoder_input(memory,[[80,80],[40,40],[20,20]])
            result=self.decoder.decoder(content,raw.sigmoid())
            return dict(pred_boxes=result[0][-1],pred_logits=result[1][-1])
    out=Path(out);out.mkdir(exist_ok=False);cases=[]
    for batch in (1,4):
        model=Model().eval();samples=torch.zeros(batch,3,640,640)
        images=[dict(id=i,width=1000+i,height=500+i) for i in range(batch)]
        ann=out/f'annotation_batch{batch}.json';ann.write_text(json.dumps(dict(images=images)))
        identity=dict(val_annotation_sha256=sha(ann));targets=[dict(image_id=torch.tensor(i),orig_size=torch.tensor([x['width'],x['height']])) for i,x in enumerate(images)]
        with torch.no_grad():baseline=model(samples)
        counters=dict(score=0,bbox=0);witness={};handles=[]
        def count_score(module,args,result):counters['score']+=1;witness['dense']=result.detach().clone()
        def count_bbox(module,args,result):counters['bbox']+=1;witness['offsets']=result.detach().clone()
        handles.extend([model.decoder.enc_score_head.register_forward_hook(count_score),model.decoder.enc_bbox_head.register_forward_hook(count_bbox)])
        rng=torch.get_rng_state().clone();trace=SelectionStageTrace(model,identity,ann,mode='synthetic_engineering')
        try:
            trace.begin_batch((samples,targets))
            with torch.no_grad():actual=model(samples)
            trace.end_batch();assert not trace.errors
            assert counters==dict(score=1,bbox=1) and trace.selection_observer.calls==1
            assert all(torch.equal(actual[k],baseline[k]) for k in actual) and torch.equal(torch.get_rng_state(),rng)
            trace.finish(out/f'stages_batch{batch}');trace.finish_selection(out/f'selection_batch{batch}')
            assert '_select_topk' not in model.decoder.__dict__
            for i in range(batch):
                v=trace.selection_records[i]
                assert (v['dense_class_logits']==witness['dense'][i].numpy()).all()
                assert (v['selected_bbox_offsets']==witness['offsets'][i].numpy()).all()
                assert v['image_id'].item()==i and v['original_hw'].tolist()==[images[i]['height'],images[i]['width']]
                assert v['selected_indices'].shape==(300,) and v['dense_anchor_logits'].shape==(8400,4)
            assert not torch.cuda.is_initialized()
            cases.append(dict(batch=batch,actual_native_methods=list(names),exact_fixture_outputs=True,source_head_calls=counters,
                additional_head_calls=0,actual_topk_calls=1,RNG_unchanged=True,image_ID_and_size_join_checked=True,method_restored=True))
        finally:
            trace.close()
            for h in handles:h.remove()
    report=dict(status='PASS_NATIVE_PRODUCER_AND_STAGE_JOIN_CPU_FIXTURES',cases=cases,torch_version=torch.__version__,
        diagnostic_code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},native_source_sha256=sha(native_source),
        synthetic_fixture_forwards=4,actual_detector_forwards=0,optimizer_updates=0,CUDA_initialized=False,
        limitation='Source-extracted native input producer and real StageTrace hooks tested in synthetic fixtures only. Real checkpoint/GPU/native-evaluator equivalence is pending; these are not model performance or feature diagnosis results.')
    p=out/'verification.json'
    with p.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(status=report['status'],cases=len(cases),report_sha256=sha(p))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native-source',required=True);p.add_argument('--native-code',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.native_source,a.native_code,a.output)
