"""Check whether saved evaluation structures explicitly name checkpoint/hash fields."""
from hashlib import sha256
from pathlib import Path
import json
import re
from inspect_saved_arrays import InertArray, InertParams, read_saved

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'saved_array_log_matching.json'
OUTPUT=HERE/'saved_evaluation_binding_fields.json'
KEY=re.compile(r'checkpoint|weights?|state_dict|sha(?:256)?|hash',re.I)

def scan(value,path,found):
    if isinstance(value,InertArray):
        return
    if isinstance(value,InertParams):
        scan(vars(value),path+'.params',found)
    elif isinstance(value,dict):
        for k,v in value.items():
            name=str(k)
            if KEY.search(name):
                found.append(path+'/'+name)
            if isinstance(v,(dict,list,tuple,InertParams)):
                scan(v,path+'/'+name,found)
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):
            if isinstance(v,(dict,list,tuple,InertParams)):
                scan(v,path+'/'+str(i),found)

def main():
    assert not OUTPUT.exists()
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    rows=[]
    for record in source['saved_evaluations']:
        path=Path(record['path'])
        assert sha256(path.read_bytes()).hexdigest()==record['sha256']
        data=read_saved(path)
        found=[]
        scan(data,'$',found)
        rows.append({'run':record['run'],'path':str(path),'sha256':record['sha256'],
                     'top_level_keys':sorted(data),'params_keys':sorted(vars(data['params'])),
                     'checkpoint_or_hash_named_fields':found})
    assert len(rows)==49
    result={'status':'SAVED_EVALUATION_EXPLICIT_BINDING_FIELDS_INSPECTED',
            'source_sha256':sha256(SOURCE.read_bytes()).hexdigest(), 'files':len(rows),
            'files_with_checkpoint_or_hash_named_fields':[r['path'] for r in rows if r['checkpoint_or_hash_named_fields']],
            'records':rows,
            'limits':['Checks field names in saved metadata; cannot exclude hidden encodings or establish any historical checkpoint identity.',
                      'No model, evaluator, prediction generation or optimizer was run.']}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'status':result['status'],'files':result['files'],
                      'files_with_named_fields':len(result['files_with_checkpoint_or_hash_named_fields'])}))

if __name__=='__main__':
    main()
