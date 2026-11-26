"""Bounded, no-gradient current-head cost accounting; no optimizer/global model."""
import sys,importlib,time,statistics,hashlib,platform,collections
sys.stdout.reconfigure(encoding='utf-8')
C=importlib.import_module('342_stage5_context');T=importlib.import_module('train344')
import torch
from torch.utils.flop_counter import FlopCounterMode
import torch.utils.flop_counter as fc
torch.set_num_threads(2)

class AuditedCounter(FlopCounterMode):
    def __init__(self):
        super().__init__(display=False);self.calls=collections.Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.calls[str(func._overloadpacket)]+=1
        return super().__torch_dispatch__(func,types,args,kwargs)

def state_hash(m):
    h=hashlib.sha256()
    for name,v in m.state_dict().items():h.update(name.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    rows=[];counts={'bound_cache_loads':0,'head_constructions':0,'head_forwards':0,'warmup_forwards':0,
        'timed_forwards':0,'FLOP_count_forwards':0,'global_model_constructions':0,'global_forwards':0,'optimizer_steps':0}
    for seed in (1,2):
        e=T.load_entry(seed,1);counts['bound_cache_loads']+=1;b=T.collate([e])
        for arm in 'NSGR':
            m=T.make_common_init(arm,seed).eval();counts['head_constructions']+=1
            params=sum(p.numel() for p in m.parameters());trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
            before=state_hash(m)
            def forward():
                counts['head_forwards']+=1
                return m.forward_features(m.trunk(b['rgb']),b['P_boxes'],b['P_cls'],b['P_scores'],b['P_valid'])
            with torch.no_grad():
                for _ in range(2):forward();counts['warmup_forwards']+=1
                times=[]
                for _ in range(10):
                    t=time.perf_counter();out=forward();times.append((time.perf_counter()-t)*1000);counts['timed_forwards']+=1
                    assert torch.isfinite(out['pred_boxes']).all() and torch.isfinite(out['pred_logits']).all()
                with AuditedCounter() as counter:forward();counts['FLOP_count_forwards']+=1
            assert before==state_hash(m) and all(p.grad is None for p in m.parameters())
            registered={str(k) for k in counter.flop_registry}
            by_op={str(k):int(v) for k,v in counter.get_flop_counts()['Global'].items()}
            uncounted={k:v for k,v in counter.calls.items() if k not in registered}
            rows.append({'seed':seed,'image_id':1,'arm':arm,'P_W':int(b['P_valid'].sum()),'batch':1,'RGB_shape':list(b['rgb'].shape),
                'head_parameters':params,'trainable_parameters':trainable,'FLOPs_registered_ops':counter.get_total_flops(),
                'FLOPs_by_op':by_op,'unregistered_op_calls':uncounted,'latency_ms_samples':times,
                'latency_ms_median':statistics.median(times),'latency_ms_min':min(times),'latency_ms_max':max(times),
                'state_sha256_before':before,'state_sha256_after':state_hash(m),'grad_all_None':True})
            print(f'CPU cost seed={seed} arm={arm} params={params} countedFLOPs={counter.get_total_flops()} median_ms={statistics.median(times):.4f}',flush=True)
    assert counts['head_forwards']==104 and counts['optimizer_steps']==0
    C.write('342_result_CPU单窗资源实测.json',{'status':'已验证（所列计量口径）；完整算子FLOPs未核','rows':rows,
        'scope':'当前train344真实头，包含trunk/mass/可学习aux/generation；不裁剪部署aux。仅forward_features无损失，无新global前向。两seed各development#1单窗，不代表全数据平均。',
        'FLOPs_scope':'PyTorch FlopCounterMode注册算子的前向计数（乘加按2 FLOPs），逐算子计数与未注册调用全部披露；elementwise、归一化、softmax及CPU融合attention等未必计入，不冒称完整FLOPs。',
        'latency_scope':'CPU FP32，torch两线程，batch1；每case预热2次、计时10次、FLOP计量1次；不含缓存加载/collate/损失/反向/优化器/全局检测器/评价。',
        'runtime':{'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'processor':platform.processor(),
                   'torch_threads':torch.get_num_threads(),'torch_FLOP_counter_path':fc.__file__,'torch_FLOP_counter_sha256':C.sha(fc.__file__)},
        'GPU_memory':'NA','GPU_hours':'NA','GPU_wall_clock':'NA','已执行':counts,
        '未执行':['真实训练','GPU','完整逐算子FLOPs计量','新global前向','全数据CPU时延平均','部署aux裁剪计量'],
        '未核':['完整算子计数覆盖','主控独立复跑'],'source_sha256':{**C.sources(),str(C.S344/'train344.py'):C.sha(C.S344/'train344.py')}})

if __name__=='__main__':main()
