import json,pathlib,hashlib,itertools
B=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proto=B/'09_质量正样本计量语义_冻结证伪协议.json';data=B/'gradient_sources_attempt01/results.json';d=read(data)
result={'protocol_sha256':sha(proto),'data_sha256':sha(data),'source_code_sha256':sha(B/'existing_methods_code/eqfl.py'),'seeds':{},'new_forwards':0,'training':0}
def clip(a,b):return min(1,max(0,a/b)) if b>0 else None
for s,x in d['seeds'].items():
 rows=[]
 for c in range(10):
  v=x['groups']['class'+str(c)];S=v['support'];Ds=v['self_down'];Dn=v['cross_down']+v['background_down']
  rows.append({'class':c,'S':S,'D_self':Ds,'D_other':Dn,'label_clipped':clip(S+Ds,Dn),'direction_clipped':clip(S,Ds+Dn),'self_down_fraction_within_label_positive':Ds/(S+Ds) if S+Ds else None})
 pairs=[]
 for a,b in itertools.combinations(range(10),2):
  va,vb=rows[a],rows[b];la,lb=va['label_clipped'],vb['label_clipped'];da,db=va['direction_clipped'],vb['direction_clipped']
  ls=(la>lb)-(la<lb) if la is not None and lb is not None else None;ds=(da>db)-(da<db) if da is not None and db is not None else None
  pairs.append({'classes':[a,b],'label_order':ls,'direction_order':ds,'strict_inversion':ls is not None and ds is not None and ls*ds==-1})
 result['seeds'][s]={'classes':rows,'pairs':pairs,'strict_inversions':sum(p['strict_inversion'] for p in pairs),'label_tied_pairs':sum(p['label_order']==0 for p in pairs),'direction_tied_pairs':sum(p['direction_order']==0 for p in pairs)}
pairs=[]
for a,b in zip(result['seeds']['1']['pairs'],result['seeds']['2']['pairs']):
 assert a['classes']==b['classes']
 if a['strict_inversion'] and b['strict_inversion'] and a['label_order']==b['label_order'] and a['direction_order']==b['direction_order']:pairs.append(a['classes'])
result['consistent_strict_inversions']=pairs;result['fixed_screen_supported']=bool(pairs)
(B/'09_质量正样本计量语义_结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'per_seed':{s:{k:v for k,v in x.items() if k not in ['classes','pairs']} for s,x in result['seeds'].items()},'consistent_inversions':pairs,'fixed_screen_supported':bool(pairs)}))
