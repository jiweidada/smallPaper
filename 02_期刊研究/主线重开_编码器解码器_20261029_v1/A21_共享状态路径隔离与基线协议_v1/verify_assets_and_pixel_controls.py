"""Remote CPU-only data integrity and frozen input-control preparation; no model."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
ROOT=Path(__file__).resolve().parent
A20=ROOT.parent/'a20_vssd_host_capacity_20261030_v1'
sys.path.insert(0,str(A20/'deps'))
import torch
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
    assert not (ROOT/'asset_verification_started.json').exists(),'Already started: inspect, do not rerun'
    write('asset_verification_started.json',{'rules_sha256':sha(ROOT/'cohort_pixel_control_rules_FROZEN.json'),'time':time.time()})
    manifest=json.loads((ROOT/'train_asset_manifest.json').read_text(encoding='utf-8'))
    train=ROOT.parents[1]/'dataset/train'
    assert len(list((train/'images').glob('*.jpg')))==manifest['image_count']==6471
    assert len(list((train/'annotations').glob('*.txt')))==6471
    aggregate=hashlib.sha256();total=0
    for row in manifest['files']:
        im=train/'images'/row['image'];ann=train/'annotations'/row['annotation']
        ih,ah=sha(im),sha(ann)
        assert ih==row['image_sha256'] and ah==row['annotation_sha256'],row['image']
        assert im.stat().st_size==row['image_bytes'];total+=im.stat().st_size
        aggregate.update(row['image'].encode()+b'\0'+bytes.fromhex(ih)+bytes.fromhex(ah))
    assert total==manifest['image_bytes'] and aggregate.hexdigest()==manifest['aggregate_sha256']
    write('remote_training_data_integrity.json',{'status':'ALL6471_ORIGINAL_IMAGES_AND_ANNOTATIONS_MATCH_LOCAL_SHA256',
       'aggregate_sha256':aggregate.hexdigest(),'images':6471,'annotations':6471,'image_bytes':total,
       'valid_GT':manifest['valid_GT_count'],'new_model_forwards':0,'training_launched':False})
    # Use only the old loader's exact author transform extraction, never its load_model().
    p=A20.parent/'dome_assets_v1/dome_probe_loader.py'
    spec=importlib.util.spec_from_file_location('a21_transform_only',p);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    loader.verified_source();transform,preprocessing=loader.input_transform()
    rules=json.loads((ROOT/'cohort_pixel_control_rules_FROZEN.json').read_text(encoding='utf-8'))
    assert rules['status']=='FROZEN_BEFORE_PIXEL_ENERGY_INSPECTION'
    cohort=json.loads((ROOT/'scientific_cohort_asset_manifest.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4);records=[]
    for c in cohort['cases']:
        p=ROOT/'cohort_inputs'/c['image'];a=p.with_suffix('.txt')
        assert sha(p)==c['image_sha256'] and sha(a)==c['annotation_sha256']
        with Image.open(p) as im:x=transform(im.convert('RGB')).squeeze(0)
        mean=x.double().mean(dim=(1,2),keepdim=True)
        masks=[c['neighbour_mask'],c['background_mask']]
        changes=[];norms=[]
        for x1,y1,x2,y2 in masks:
            delta=mean-x[:,y1:y2,x1:x2].double();changes.append(delta);norms.append(float(torch.linalg.vector_norm(delta)))
        energy=min(norms);edited=[];actual=[];alphas=[]
        tx1,ty1,tx2,ty2=c['target_guard']
        for mask,delta,norm in zip(masks,changes,norms):
            x1,y1,x2,y2=mask;alpha=energy/norm if norm else 0.0;y=x.clone()
            y[:,y1:y2,x1:x2]=(x[:,y1:y2,x1:x2].double()+alpha*delta).float()
            assert torch.equal(y[:,ty1:ty2,tx1:tx2],x[:,ty1:ty2,tx1:tx2])
            diff=(y-x).double();outside=diff.clone();outside[:,y1:y2,x1:x2]=0;assert not bool(torch.any(outside))
            actual.append(float(torch.linalg.vector_norm(diff)));alphas.append(alpha);edited.append(y)
        assert abs(actual[0]-actual[1])<=max(1e-7,1e-5*energy),(c['image'],actual,energy)
        assert masks[0][2]-masks[0][0]==masks[1][2]-masks[1][0]
        assert masks[0][3]-masks[0][1]==masks[1][3]-masks[1][1]
        records.append({'image':c['image'],'prefix':c['prefix'],'primary':not c['filename_prefix_also_in_train'],
          'original_resized_tensor_sha256':tensor_sha(x),'neighbour_edited_tensor_sha256':tensor_sha(edited[0]),
          'background_edited_tensor_sha256':tensor_sha(edited[1]),'mean_RGB':mean.flatten().tolist(),
          'raw_patch_norms':norms,'common_energy':energy,'alphas':alphas,'actual_perturbation_norms':actual,
          'target_pixels_unchanged':True,'outside_masks_unchanged':True,'zero_energy':energy==0})
    assert len(records)==33 and sum(r['primary'] for r in records)==27
    assert not torch.cuda.is_initialized()
    write('cohort_pixel_control_preparation.json',{'status':'FROZEN_PIXEL_CONTROLS_PREPARED_CPU_NO_OUTCOMES',
       'rules_sha256':sha(ROOT/'cohort_pixel_control_rules_FROZEN.json'),'cases':records,
       'primary_cases':27,'secondary_cases':6,'zero_energy_cases':sum(r['zero_energy'] for r in records),
       'source_preprocessing':preprocessing,'new_model_forwards':0,'optimizer_updates':0,
       'scientific_outcomes_estimated':False,'CUDA_initialized':False})
    print(json.dumps({'status':'ALL_ASSETS_AND_FROZEN_CONTROLS_VERIFIED','train_images':6471,'valid_GT':manifest['valid_GT_count'],
          'primary_cases':27,'secondary_cases':6,'zero_energy_cases':sum(r['zero_energy'] for r in records),'model_forwards':0}))
if __name__=='__main__':main()
