from pathlib import Path
import json,math
import numpy as np
from audit_scene_scale import P,load_rows,sha,MAN
def main():
    out=P/'scale_statistics_FROZEN_v1.json';assert not out.exists()
    a=json.loads((P/'02_场景尺度统计核查_v1.json').read_text(encoding='utf-8'))
    mu=np.asarray(a['canonical_class_log_offsets']);rows=load_rows()
    residual=np.concatenate([r['a'][:,3]-math.log(640)-mu[r['a'][:,0].astype(int)] for r in rows if r['calibration'] and len(r['a'])])
    obj={'status':'FROZEN_TRAIN_CALIBRATION_STATISTICS','class_offsets':mu.tolist(),'initial_log_scale':float(np.median(residual)),
        'calibration_images':sum(r['calibration'] for r in rows),'calibration_objects':len(residual),'definition':'class-relative log of normalized geometric-mean width/height; not metric GSD',
        'validation_or_test_used':False,'same_M_Q':True,'train_manifest_sha256':sha(MAN),'derivation_result_sha256':sha(P/'02_场景尺度统计核查_v1.json')}
    out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(obj,ensure_ascii=False))
if __name__=='__main__':main()
