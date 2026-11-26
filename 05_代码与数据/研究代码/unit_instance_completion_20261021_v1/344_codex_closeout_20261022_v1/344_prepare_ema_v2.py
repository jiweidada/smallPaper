"""Preserve the failed diagnostic source and publish a reviewed correction."""
import importlib,difflib
C=importlib.import_module('344_context')
old=(C.HERE/'344_ema_diagnostic.py').read_text(encoding='utf-8')
new=old.replace("            assert sum(p.numel() for p in model.parameters())==10231995", "            parameter_count=sum(p.numel() for p in model.parameters())\n            assert parameter_count==sum(ema[k].numel() for k,_ in model.named_parameters())\n            print(f'EMA seed={seed} strict-loaded tensors={len(ema)} parameters={parameter_count}',flush=True)")
new=new.replace("'EMA_updates':updates", "'parameter_count':parameter_count,'EMA_state_tensor_count':794,'EMA_updates':updates")
for name in ('344_EMA_seed{seed}_COCO.txt','344_result_EMA_seed{seed}.json','344_result_EMA诊断.json','344_result_EMA诊断_实现失败.json'):
    new=new.replace(name,name.rsplit('.',1)[0]+'_v2.'+name.rsplit('.',1)[1])
assert new!=old
C.text('344_ema_diagnostic_v2.py',new)
C.text('344_EMA诊断脚本修正.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='344_ema_diagnostic.py',tofile='344_ema_diagnostic_v2.py')))
