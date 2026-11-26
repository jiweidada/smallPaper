"""Pure protocol/implementation parity, no change to the completed run."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,csv
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REP=ROOT/'raw_EMA逐GT对照_20261026_v1/同后端CPU重放_20261026_v1';O=HERE/'argmax_parity_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir();declared={r['path']:r['sha256'] for n in ('preflight.json','complete.json') for r in read(REP/n)['entries']};rows=[];mismatch=[];inputs={str(Path(__file__)):sha(Path(__file__))}
 for state in ('raw','EMA'):
  for seed in (1,2):
   for stage in ('pre','final'):
    count=0;bad=0;logties=0;sigties=0
    for p in sorted((REP/f'seed{seed}_{state}').glob('image_*.npz')):
     h=sha(p);assert h==declared[str(p)];inputs[str(p)]=h
     with np.load(p,allow_pickle=False) as z:l=torch.from_numpy(z[stage+'_logits']);iid=int(z['image_id'])
     s=l.sigmoid();la=l.argmax(1);sa=s.argmax(1);different=(la!=sa).numpy();count+=l.shape[0];bad+=int(different.sum());logties+=int(((l==l.max(1,keepdim=True).values).sum(1)>1).sum());sigties+=int(((s==s.max(1,keepdim=True).values).sum(1)>1).sum())
     for q in np.flatnonzero(different):mismatch.append({'state':state,'seed':seed,'stage':stage,'image_id':iid,'query':int(q),'logit_argmax':int(la[q]),'sigmoid_argmax':int(sa[q])})
    assert count==548*300;rows.append({'state':state,'seed':seed,'stage':stage,'images':548,'queries':count,'argmax_mismatches':bad,'logit_top_ties':logties,'sigmoid_top_ties':sigties})
 assert all(sha(p)==h for p,h in inputs.items())
 with open(O/'per_condition.csv','x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={'status':'PASS' if not mismatch else 'PROTOCOL_DEVIATION_PRESERVED','scope':'argmax logits vs nativeFP32 sigmoid argmax, first-index ties, all2192 caches/two stages; audit only','conditions':rows,'mismatches':mismatch,'input_sha256':inputs,'queries_checked_total':sum(r['queries'] for r in rows),'operations':{'model_forwards':0,'evaluator_calls':0,'training':0}}
 (O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},ensure_ascii=False))
if __name__=='__main__':main()
