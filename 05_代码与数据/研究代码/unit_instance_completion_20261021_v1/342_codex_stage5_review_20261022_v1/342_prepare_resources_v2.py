"""CPU FLOP tracker requires its no-gradient inputs to carry no grad flags."""
import importlib,difflib
C=importlib.import_module('342_stage5_context')
old=(C.HERE/'342_cpu_resources.py').read_text(encoding='utf-8')
new=old.replace("            before=state_hash(m)","            # Cost-only temporary model: avoid module_tracker grad-hook registration.\n            for p in m.parameters():p.requires_grad_(False)\n            before=state_hash(m)")
new=new.replace('342_result_CPU单窗资源实测.json','342_result_CPU单窗资源实测_v2.json')
new=new.replace("'latency_scope':", "'temporary_model_grad_flags':'先记录训练参数量，再仅在计量用临时模型关闭requires_grad；权重/state不变，无优化器。', 'latency_scope':")
C.text('342_cpu_resources_v2.py',new)
C.text('342_CPU计量接口修正.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='342_cpu_resources.py',tofile='342_cpu_resources_v2.py')))
C.write('342_result_CPU计量首次实现失败.json',{'status':'实现失败',
    'exception':'PyTorch module_tracker.register_multi_grad_hook: _get_grad_fn_or_grad_acc assertion under no_grad with a learned query still requires_grad=True.',
    '已执行':{'bound_cache_loads':1,'head_constructions':1,'head_forward_attempts':13,'head_forwards_completed':12,
        'warmup_forwards':2,'timed_forwards':10,'FLOP_forward_attempts':1,'FLOP_forward_completions':0,'global_forwards':0,'optimizer_steps':0},
    'repair':'另存v2；只对临时计量模型关闭requires_grad。原模型、权重、科学配方均不改。',
    '未执行':['真实训练','GPU'],'未核':['完整FLOPs覆盖'],'source_sha256':C.sources()})
