from pathlib import Path
import json, time, torch, torchvision, scipy, numpy
from runtime_A48 import ROOT, ASSETS, DATA, native, plan_check

if __name__ == '__main__':
    with (ROOT/'prepare_once.lock').open('x') as f:
        f.write(str(time.time()))
    plan = plan_check()
    checks={}
    for split in ['train','val']:
        manifest=json.loads((ROOT/(split+'_data_manifest.json')).read_text())
        images=DATA/('VisDrone2019-DET-'+split)/'images'
        annotations=images.parent/'annotations'
        assert len(list(images.glob('*.jpg'))) == manifest['images']
        for row in manifest['rows']:
            assert native.digest(images/row['file']) == row['image_sha256'],row['file']
            assert native.digest(annotations/(row['file'][:-4]+'.txt')) == row['annotation_sha256'],row['file']
        checks[split]={'images':manifest['images'],'image_hashes_verified':len(manifest['rows']),
                       'annotation_hashes_verified':len(manifest['rows'])}
        if split=='val':
            view=ROOT/'val_inputs';view.mkdir(exist_ok=False)
            for row in manifest['rows']:
                (view/row['file']).symlink_to(images/row['file'])
                ann=row['file'][:-4]+'.txt'
                (view/ann).symlink_to(annotations/ann)
    result={'status':'PASS','unix':time.time(),'data':checks,'native_sources_verified':True,
            'common_init_verified':True,'old_files_modified':False,
            'environment':{'torch':torch.__version__,'torchvision':torchvision.__version__,
                           'scipy':scipy.__version__,'numpy':numpy.__version__},
            'new_model_forwards':0,'new_optimizer_updates':0,
            'protocol_sha256':native.digest(ROOT/'protocol_FROZEN.json')}
    native.save_json(ROOT/'preparation_result.json',result)
    print(json.dumps(result),flush=True)
