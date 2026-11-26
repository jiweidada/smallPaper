"""Collect the completed geometry diagnostic once and check independent scalar math."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import zipfile
import numpy as np

P=Path(__file__).resolve().parent
DL=P.parent/'A54_场景尺度约束候选_20261101_v1/A60_downloads'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text('utf-8'))
def main():
    ready=load(DL/'A60_geometry_D0_ready_v1.json')
    z=DL/'A60_geometry_D0_bundle_v1.zip'
    assert z.stat().st_size==ready['zip_bytes'] and sha(z)==ready['zip_sha256']
    out=P/'geometry_D0_remote_v1';out.mkdir(exist_ok=False)
    with zipfile.ZipFile(z) as a:
        manifest=json.loads(a.read('manifest.json'))
        assert len(manifest)==ready['files']==10
        assert set(a.namelist())==set(manifest)|{'manifest.json'}
        for name,row in manifest.items():
            rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
            target=out/Path(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
            raw=a.read(name)
            assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
            with target.open('xb') as f:f.write(raw)
    assert sha(out/'frozen/geometry_scope_FROZEN_v1.json')==sha(P/'geometry_scope_FROZEN_v1.json')
    assert sha(out/'frozen/geometry_exposure_D0_v1.py')==sha(P/'geometry_exposure_D0_v1.py')
    r=load(out/'result.json');assert load(out/'exit.json')=={'exit':0}
    assert r['status']=='GEOMETRY_D0_COMPLETE_NOT_A_PERFORMANCE_TEST'
    assert r['source_train_images']==6471 and r['selected_image_headers']==128
    assert r['model_forwards']==r['optimizer_updates']==r['checkpoint_loads']==0 and not r['torch_imported']
    arr=np.load(out/'selected_training_GT_geometry.npz')['rows']
    assert arr.shape==(r['selected_clipped_GT'],7) and np.isfinite(arr).all()
    nodes=(np.arange(65536)+.5)/65536
    errors=[]
    # Independent scalar derivation evaluated against a dense time grid.
    for d in [1.,10.,100.,1000.]:
        physical=nodes
        normalized=1.-(((1.+d)**(1.-nodes)-1.)/d)
        empirical=[np.mean((1.-physical)*d<=.1),np.mean((1.-normalized)*d<=.1)]
        exact=[min(1.,.1/d),min(1.,math.log(1.1)/math.log(1.+d))]
        for a,b in zip(empirical,exact):errors.append(abs(a-b))
    assert max(errors)<=2/65536
    strata={}
    for prior in ['specified_truncated_gaussian','uniform']:
        pair=np.load(out/(prior+'_derived_pairs.npz'))
        gt=pair['GT'];corner=pair['prior_corners'];d0=pair['d0']
        wh=gt[:,2:4]-gt[:,:2]
        reference=np.max(np.abs(corner-gt[:,:4])/np.tile(wh,(1,2)),axis=1)
        assert np.allclose(d0,reference,rtol=1e-13,atol=1e-13)
        mask=(gt[:,5]<=1024)&(gt[:,4]<=4)
        assert int(mask.sum()/4)==r['priors'][prior]['strata']['small_short_le_4']['GT_count']
        strata[prior]=r['priors'][prior]['strata']['small_short_le_4']
    receipt=dict(status='PASS_TRANSFER_AND_DERIVED_GEOMETRY_CONSISTENCY',payload_files=10,
        independent_scalar_dense_grid_max_fraction_error=max(errors),selected_GT=len(arr),
        tiny_strata=strata,source_annotation_passes_added=0,model_forwards=0,optimizer_updates=0,
        limits='Rechecks transferred derived pairs and scalar formulas; not a second annotation pass or learned detector test.')
    with (P/'geometry_D0_local_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
