import json
from pathlib import Path
import yaml
root=Path(CONFIG['root'])/'code/configs/experiments/uavdt'
configs={arm:yaml.safe_load((root/(arm+'_full96.yml')).read_text()) for arm in ['baseline','c','c_mean']}
def differences(a,b,prefix=''):
 if isinstance(a,dict) and isinstance(b,dict):
  result=[]
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:result.append({'key':prefix+k,'left':a.get(k),'right':b.get(k)})
   else:result+=differences(a[k],b[k],prefix+k+'.')
  return result
 return [] if a==b else [{'key':prefix.rstrip('.'),'left':a,'right':b}]
print(json.dumps({'account':CONFIG['account'],'baseline_to_c':differences(configs['baseline'],configs['c']),
                  'c_to_cmean':differences(configs['c'],configs['c_mean'])}),flush=True)
