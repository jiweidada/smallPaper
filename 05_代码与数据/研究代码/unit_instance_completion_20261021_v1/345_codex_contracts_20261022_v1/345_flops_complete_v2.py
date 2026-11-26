"""CPU registered FLOPs plus explicit shape-based supplemental arithmetic."""
import importlib,math,collections,time,traceback
C=importlib.import_module('345_context')
import torch
from torch.utils.flop_counter import FlopCounterMode
T=importlib.import_module('train344');torch.set_num_threads(2)

ZERO={'_local_scalar_dense','_to_copy','_unsafe_view','all','any','arange','bitwise_and','bitwise_not','cat','clone','detach','eq','equal','expand','full_like','gather','ge','gt','index','index_copy','index_select','le','lift_fresh','lt','masked_fill_','max','maximum','min','ne','nonzero','ones_like','permute','scalar_tensor','scatter_','select','slice','split_with_sizes','squeeze','stack','t','transpose','unsqueeze','view','where','zeros','zeros_like','clamp_min','clamp','minimum','relu','copy_','empty','empty_like','empty_strided','reshape','_reshape_alias','contiguous','unbind'}
ONE={'abs','neg','mul','div','true_divide','pow','exp','log','log1p','sqrt','rsqrt','reciprocal','sin','cos','tanh','erf','expm1'}

def tensors(x):
    if torch.is_tensor(x):return [x]
    if isinstance(x,(tuple,list)):return [v for item in x for v in tensors(item)]
    if isinstance(x,dict):return [v for item in x.values() for v in tensors(item)]
    return []
def count_output(out):return sum(t.numel() for t in tensors(out) if t.is_floating_point())
def shape(x):
    if torch.is_tensor(x):return {'shape':list(x.shape),'dtype':str(x.dtype)}
    if isinstance(x,(tuple,list)):return [shape(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return 'nan' if math.isnan(x) else ('inf' if x>0 else '-inf')
    if isinstance(x,(bool,int,float,str)) or x is None:return x
    return str(type(x))

def manual(name,args,kw,out,registered):
    op=name.split('.')[-1];N=count_output(out);first=args[0] if args else None
    if registered:
        if op in ('convolution','_convolution'):
            assert not args[6], 'transposed convolution not part of this head'
            ref=2*out.numel()*math.prod(args[1].shape[1:])
            extra=out.numel() if args[2] is not None else 0
            return extra,'convolution bias addition',ref
        if op in ('mm','bmm','addmm','baddbmm'):
            a,b=(args[1],args[2]) if op in ('addmm','baddbmm') else (args[0],args[1])
            ref=2*math.prod(a.shape[:-1])*a.shape[-1]*b.shape[-1]
            extra=out.numel() if op in ('addmm','baddbmm') and kw.get('beta',1)!=0 else 0
            if kw.get('alpha',1)!=1:extra+=out.numel()
            if kw.get('beta',1) not in (0,1):extra+=out.numel()
            return extra,'matrix product registered; beta/add bias supplement',ref
        return 0,'registered forward operator',None
    if op in ZERO:return 0,'data movement/indexing/construction, boolean/integer, or comparison/selection: zero arithmetic FLOPs',None
    if op in ('add','sub'):
        return N*(1+(kw.get('alpha',1)!=1)),'elementwise add/sub; nonunit alpha adds a multiply',None
    if op in ONE:return N,'one arithmetic/transcendental primitive per floating output element',None
    if op in ('sum','mean'):
        if not torch.is_tensor(first) or not first.is_floating_point():return 0,'integer/bool reduction',None
        base=max(0,first.numel()-N)
        return base+(N if op=='mean' else 0),'reduction adds N_in-N_out; mean divides N_out',None
    if op in ('_softmax','softmax','_safe_softmax','_log_softmax','log_softmax'):
        dim=args[1] if len(args)>1 else kw.get('dim',-1);width=first.shape[dim];rows=first.numel()//max(width,1)
        return (4*first.numel()-rows if 'log_softmax' not in op else 4*first.numel()),'stable softmax: subtract/exp/sum/div; comparisons excluded',None
    if op=='logsumexp':return 3*first.numel()+N,'stable logsumexp: subtract/exp/sum/log/add; max comparisons excluded',None
    if op in ('native_layer_norm','layer_norm'):
        n=first.numel();width=math.prod(args[1]);rows=n//width;affine=(args[2] is not None)+(args[3] is not None)
        return 5*n+2*rows+affine*n,'normalization: mean D, centered variance 3D, scale D, eps+rsqrt2, optional affine D each',None
    if op in ('native_group_norm','group_norm'):
        n=first.numel()
        if op=='native_group_norm':rows=int(args[3])*int(args[6]);weight,bias=args[1],args[2]
        else:rows=int(first.shape[0])*int(args[1]);weight,bias=args[2],args[3]
        return 5*n+2*rows+n*((weight is not None)+(bias is not None)),'groupwise same normalization formula as LayerNorm',None
    if op=='silu':return 5*N,'silu: neg/exp/add/reciprocal/multiply',None
    if op=='sigmoid':return 4*N,'sigmoid: neg/exp/add/reciprocal',None
    if op=='gelu':return N*(9 if kw.get('approximate','none')=='tanh' else 5),'GELU erf form 5 primitives; tanh approximation9',None
    if op=='softplus':
        beta=args[1] if len(args)>1 else kw.get('beta',1)
        return N*(2 if beta==1 else 4),'softplus exp+log1p; optional nonunit beta multiply/divide, threshold compare excluded',None
    if op=='_scaled_dot_product_flash_attention_for_cpu':
        q,k,v=args[:3];B=math.prod(q.shape[:-2]);Q,D=q.shape[-2:];K=k.shape[-2];V=v.shape[-1];scores=B*Q*K;rows=B*Q
        fl=2*scores*(D+V)+scores+(4*scores-rows)+2*rows
        if kw.get('attn_mask') is not None and kw['attn_mask'].is_floating_point():fl+=scores
        return fl,'CPU fused attention: QK+PV MACs, scale, stable softmax, returned logsumexp; optional float mask add',None
    if op=='_native_multi_head_attention':
        q,k,v=args[:3];B,Q,D=q.shape;K=k.shape[1];L=v.shape[1];H=int(args[4]);scores=B*H*Q*K;rows=B*H*Q
        projection=2*B*(Q+K+L)*D*D+2*B*Q*D*D
        bias=B*(Q+K+L)*D*(args[6] is not None)+B*Q*D*(args[8] is not None)
        fl=projection+bias+4*B*Q*K*D+scores+(4*scores-rows)
        if len(args)>9 and torch.is_tensor(args[9]) and args[9].is_floating_point():fl+=scores
        return fl,'native fused MHA: Q/K/V/out projections+bias, QK/PV, scale, softmax; optional float mask',None
    return None,'UNCLASSIFIED',None

class Counter(FlopCounterMode):
    def __init__(self):super().__init__(display=False);self.events=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kw={} if kwargs is None else kwargs
        before=self.get_total_flops();out=super().__torch_dispatch__(func,types,args,kw);registered=self.get_total_flops()-before
        name=str(func._overloadpacket);isreg=func._overloadpacket in self.flop_registry
        extra,rule,reference=manual(name,args,kw,out,isreg)
        if reference is not None:assert reference==registered,(name,reference,registered)
        self.events.append({'op':name,'args':shape(args),'kwargs':shape(kw),'out':shape(out),'registered':registered,
            'manual':extra,'rule':rule,'independent_registered_formula':reference})
        return out

def main():
    assert len(C.read(C.HERE/'345_result_A合同总表_v2.json')['contracts'])==12
    policy=C.HERE/'345_FLOPs手算预定口径.json'
    assert policy.is_file()
    previous=C.read(C.HERE/'345_FLOPs部分标准输出登记.json')['rows']
    expected={(r['seed'],r['arm']):(r['registered'],r['manual'],r['sum']) for r in previous}
    C.write('345_FLOPs序列化修订口径_v2.json',{
        'authorization':str(C.SPEC/'345c_补充裁定_FLOPs写盘修正.txt'),
        'inherited_policy':{'path':str(policy),'sha256':C.sha(policy)},
        'metadata_encoding':'non-finite Python float constants only: inf/-inf/nan strings; tensor shape/dtype only; FLOPs integers unchanged',
        'expected_previous_stdout':previous,'source_sha256':C.sources()})
    rows=[];calls=0
    try:
        for seed in (1,2):
            e=T.load_entry(seed,1);b=T.collate([e])
            for arm in 'NSGR':
                m=T.make_common_init(arm,seed).eval()
                for p in m.parameters():p.requires_grad_(False)
                with torch.no_grad(),Counter() as counter:
                    out=m.forward_features(m.trunk(b['rgb']),b['P_boxes'],b['P_cls'],b['P_scores'],b['P_valid'])
                calls+=1
                assert torch.isfinite(out['pred_boxes']).all() and torch.isfinite(out['pred_logits']).all()
                unknown=[x for x in counter.events if x['manual'] is None];assert not unknown,[x['op'] for x in unknown]
                registered=counter.get_total_flops();extra=sum(x['manual'] for x in counter.events)
                totals=collections.defaultdict(lambda:{'calls':0,'registered':0,'manual':0})
                for x in counter.events:
                    d=totals[x['op']];d['calls']+=1;d['registered']+=x['registered'];d['manual']+=x['manual']
                assert (registered,extra,registered+extra)==expected[(seed,arm)], ('345c total mismatch',seed,arm,registered,extra)
                rows.append({'seed':seed,'arm':arm,'image_id':1,'P_W':int(b['P_valid'].sum()),'registered_FLOPs':registered,
                    'manual_FLOPs':extra,'registered_plus_manual_FLOPs':registered+extra,'op_totals':dict(totals),'events':counter.events})
                print(f'FLOPs seed={seed} arm={arm} registered={registered} manual={extra} total={registered+extra}',flush=True)
        C.write('345_result_FLOPs覆盖补足_v2.json',{'status':'已验证（明确算法原语口径）','rows':rows,'scope':'current train344 heads, both seeds image1, B1 RGB128², aux retained; no loss/backward/global. Fused kernel arithmetic is represented by its mathematical algorithm, not a hardware-instruction claim.',
            'metadata_encoding':'inf/-inf/nan strings for non-finite scalar metadata; numeric FLOPs unchanged','all_8_match_previous_stdout':True,'policy_sha256':C.sha(policy),'GPU':'NA','已执行':{'head_forwards':calls,'global_forwards':0,'optimizer_steps':0},
            '未执行':['真实训练','GPU','hardware instruction counting'],'未核':['主控独立复算'],'source_sha256':C.sources()})
    except Exception as ex:
        C.write('345_result_FLOPs实现失败_停止_v2.json',{'status':'实现失败，B未执行','exception':repr(ex),'traceback':traceback.format_exc(),
            '已执行':{'head_forwards_completed':calls,'optimizer_steps':0},'未执行':['B','真实训练','GPU'],'未核':['FLOPs补足'],'source_sha256':C.sources()});raise

if __name__=='__main__':main()
