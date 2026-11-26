"""Exactly 16 source-separated native augmentation batches. CPU only; no model."""
import argparse,copy,hashlib,json,pathlib,random,time
from probe_runtime import ROOT,verify_package,sha
import numpy as np
import torch
from engine.data.transforms.container import Compose
from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.dataloader import BatchImageCollateFunction

def main(data):
    verify_package();plan=json.loads((ROOT/'sample_plan.json').read_text());y=json.loads((ROOT/'baseline_recipes.json').read_text())['1']
    ann=data/'annotations/train.json';assert sha(ann)==plan['annotation_sha256']
    output=ROOT/'inputs';output.mkdir(exist_ok=False);torch.set_num_threads(4)
    manifest=[];source_sets={};started=time.monotonic()
    class Logged(CocoDetection):
        def load_item(self,idx):
            image,target=super().load_item(idx);iid=int(target['image_id'].item());info=self.coco.imgs[iid]
            path=data/'images'/info['file_name'];self.reads.append({'image_id':iid,'file_name':info['file_name'],'sha256':sha(path)})
            return image,target
    for group,pool in plan['source_pools'].items():
        tc=copy.deepcopy(y['train_dataloader']['dataset']['transforms']);tc.pop('type')
        ds=Logged(str(data/'images'),str(ann),transforms=Compose(**tc),return_masks=False,remap_mscoco_category=False,cache_imgsz=y.get('cache_imgsz'),ram_cache=False,num_classes=10)
        assert not ds.cache_flag;ds.ids=list(pool);ds.set_epoch(plan['augmentation_epoch']);source_sets[group]=set()
        cc=copy.deepcopy(y['train_dataloader']['collate_fn']);cc.pop('type');collate=BatchImageCollateFunction(**cc);collate.set_epoch(plan['augmentation_epoch'])
        indices={iid:i for i,iid in enumerate(ds.ids)}
        for spec in [s for s in plan['batches'] if s['group']==group]:
            seed=spec['augmentation_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);ds.reads=[]
            batch=[ds[indices[i]] for i in spec['main_image_ids']];images,targets=collate(batch)
            images=images.detach().clone().float();targets=[{k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
            assert images.ndim==4 and images.shape[0]==4 and torch.isfinite(images).all()
            for t in targets:
                assert t['boxes'].shape==(len(t['labels']),4) and torch.isfinite(t['boxes']).all()
                assert ((t['labels']>=0)&(t['labels']<10)).all()
            sources={r['image_id'] for r in ds.reads};assert sources.issubset(set(pool));source_sets[group].update(sources)
            payload={'images':images,'targets':targets,'spec':spec,'source_reads':ds.reads,'historical_replay':False}
            path=output/f"batch{spec['batch']:02d}.pt";torch.save(payload,path)
            record={'batch':spec['batch'],'group':group,'file':path.name,'sha256':sha(path),'float_input_sha256':hashlib.sha256(images.numpy().tobytes()).hexdigest(),'shape':list(images.shape),'GT_counts':[len(t['labels']) for t in targets],'source_ids':sorted(sources),'source_reads':ds.reads}
            manifest.append(record);(output/'progress.json').write_text(json.dumps({'completed_batches':len(manifest),'inputs':4*len(manifest)}))
            print(json.dumps({'phase':'inputs','batch':spec['batch'],'shape':record['shape'],'GT':record['GT_counts'],'elapsed':time.monotonic()-started}),flush=True)
    assert len(manifest)==16 and source_sets['development'].isdisjoint(source_sets['heldout'])
    (output/'manifest.json').write_text(json.dumps({'batches':manifest,'formal_inputs':64,'source_groups_disjoint':True,'plan_sha256':sha(ROOT/'sample_plan.json'),'source_sets':{g:sorted(v) for g,v in source_sets.items()}},indent=2))
    (output/'complete.json').write_text(json.dumps({'state':'complete','inputs':64,'elapsed_seconds':time.monotonic()-started,'model_forwards':0,'optimizer_steps':0}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);main(p.parse_args().data)
