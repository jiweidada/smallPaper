"""Separate evidence of gradient-driven movement from AdamW weight decay."""
from pathlib import Path
import json,re,hashlib

read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(root):
    root=Path(root);cfg=read(root/'prepared/resolved.yml');runtime=read(root/'run/runtime_start.json')
    assert runtime['resume'] is None and cfg['resume'] is None and cfg['optimizer']['type']=='AdamW'
    identity=read(root/'prepared/receipt.json')['identity'];assert runtime['identity']==identity
    for n in ('geometry_memory.py','small_object_model.py','training_audit.py','runtime.py','experiment.py'):
        assert sha(root/'code'/n)==identity['implementation'][n]
    initializer=(root/'code/geometry_memory.py').read_text(encoding='utf8')
    assert 'nn.init.zeros_(self.output.weight)' in initializer and 'nn.init.zeros_(self.output.bias)' in initializer
    epoch=read_epoch0(root/'run/training_audit_epochs.jsonl');changed=epoch['added_changed_elements_from_initial']
    witnesses={}
    for prefix,expected in [('encoder.geometry_field.',6),('decoder.decoder.geometry_layers.',18)]:
        rows=[]
        for name,count in changed.items():
            if not name.startswith(prefix) or count<=0:continue
            wd=cfg['optimizer']['weight_decay']
            for g in cfg['optimizer']['params']:
                if re.search(g['params'],name):wd=g.get('weight_decay',wd)
            if wd==0:rows.append({'name':name,'changed_elements':count,'weight_decay':0})
        assert len(rows)==expected
        witnesses[prefix]={'reason':'Zero weight decay; movement after fresh initialization cannot be explained by AdamW decay alone.',
                           'tensor_count':len(rows),'changed_elements':sum(r['changed_elements'] for r in rows),'parameters':rows}
    for prefix,expected in [('decoder.enc_bbox_head.geometry_proposal.',2),('decoder.decoder.geometry_attention.',6)]:
        rows=[{'name':n,'changed_elements':v,'epoch1_norm64':epoch['new_parameters'][n]['norm64']}
              for n,v in changed.items() if n.startswith(prefix) and '.output.' in n and v>0]
        assert len(rows)==expected and all(r['epoch1_norm64']>0 for r in rows)
        witnesses[prefix]={'reason':'Output weight/bias initialized to zero. Pure weight decay leaves zero unchanged; nonzero movement requires another update source.',
                           'tensor_count':len(rows),'changed_elements':sum(r['changed_elements'] for r in rows),'parameters':rows}
    return {'status':'PASS_GROUP_MOVEMENT_NOT_EXPLAINED_BY_WEIGHT_DECAY_ALONE','seed':identity['seed'],
        'epoch_zero_based':0,'witnesses':witnesses,'source_epoch_audit_sha256':sha(root/'run/training_audit_epochs.jsonl'),
        'fresh_optimizer_and_no_resume':True,'extra_model_forwards':0,'extra_optimizer_updates':0,
        'inference':'Together with the frozen optimizer-only update path and initialization guards, these observations support active gradient updates in all four added groups.',
        'not_proven':['Useful representation learned','Improved fixed12 APs','Eliminated classification-localization gradient conflict']}


def read_epoch0(path):
    rows=[json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines()]
    matches=[r for r in rows if r['epoch_zero_based']==0];assert len(matches)==1
    return matches[0]
