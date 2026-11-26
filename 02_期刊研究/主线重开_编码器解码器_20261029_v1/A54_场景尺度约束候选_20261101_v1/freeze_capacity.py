from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=P/'capacity_protocol_FROZEN_v1.json';assert not out.exists()
    old=json.loads((P.parent/'A52_不同机制候选有界筛选_20261101_v1/protocol_FROZEN_v1.json').read_text(encoding='utf-8'))
    own=['scene_scale_prior.py','runtime_A54.py','capacity_A54.py','scale_statistics_FROZEN_v1.json','pilot_design_FROZEN_v1.json']
    for p in own:
        if p.endswith('.py'):ast.parse((P/p).read_text(encoding='utf-8'))
    external={p:h for p,h in old['external_sha256'].items() if not p.startswith('/root/uav_a52_')}
    obj={'phase':'A54','scope':'NATIVE_INTEGRATION_AND_CAPACITY_ONLY','formal_training_queued':False,'maximum_whole_model_forwards':9,
        'maximum_capacity_only_optimizer_updates':6,'initial_FP32_prediction_absolute_tolerance':1e-6,
        'batches':'actual epoch0 first, max per-image GT, max distinct batch-total GT, as fixed in capacity_A54.py',
        'temporary_state':'discard all capacity model/optimizer states; formal arms must start from common init',
        'files_sha256':{p:sha(P/p) for p in own},'external_sha256':external}
    out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'capacity_protocol_sha256':sha(out),'own_files':len(own),'external_files':len(external),'limits':[9,6]}))
if __name__=='__main__':main()
