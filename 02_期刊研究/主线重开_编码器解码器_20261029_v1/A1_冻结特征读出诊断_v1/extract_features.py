"""A1:16 untouched author-model forwards,read-only feature hooks,no optimizer."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as VF
from tinyformer_probe_loader import ROOT, load_official_model, state_digest
from readout_common import sample_points, max_iou, cxcywh_to_xyxy, selfcheck

BASE = ROOT.parent
DATA = BASE.parent / 'dataset'

def write_json(path, obj):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temporary.replace(path)

def image_tensor(path):
    with Image.open(path) as im:
        im = VF.resize(im.convert('RGB'), [640, 640], antialias=True)
        x = VF.pil_to_tensor(im).float() / 255.
        x = VF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return x.unsqueeze(0).cuda()

def annotation(file_name):
    path = DATA/'train/images'/file_name
    ann = DATA/'train/annotations'/(Path(file_name).stem+'.txt')
    expected = json.loads((BASE/'diagnostic100_asset_manifest.json').read_text())
    row = next(r for r in expected['files'] if r['file_name']==file_name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['images']['sha256']
    assert hashlib.sha256(ann.read_bytes()).hexdigest() == row['annotations']['sha256']
    with Image.open(path) as im:
        width,height = im.size
    boxes,labels,uids,ignore = [],[],[],[]
    for line_no,line in enumerate(ann.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():
            continue
        fields = [float(v) for v in line.rstrip(',').split(',')]
        x,y,w,h,score,cat = fields[:6]
        x1,y1 = np.clip(x,0,width),np.clip(y,0,height)
        x2,y2 = np.clip(x+w,0,width),np.clip(y+h,0,height)
        if x2 <= x1 or y2 <= y1:
            continue
        box = [x1/width*640,y1/height*640,x2/width*640,y2/height*640]
        if score == 1 and 1 <= int(cat) <= 10:
            boxes.append(box); labels.append(int(cat)-1); uids.append(file_name+':'+str(line_no))
        else:
            ignore.append(box)
    return path,np.asarray(boxes,np.float64).reshape(-1,4),np.asarray(labels),np.asarray(uids),np.asarray(ignore,np.float64).reshape(-1,4)

def gather(feat, xy):
    assert feat.shape[0] == 1 and feat.shape[1] in (1,192)
    grid = torch.as_tensor(xy,device=feat.device,dtype=feat.dtype)/640.*2.-1.
    sampled = F.grid_sample(feat,grid.reshape(1,-1,1,2),mode='bilinear',
                            padding_mode='zeros',align_corners=False)
    return sampled[0,:,:,0].transpose(0,1).contiguous().cpu().numpy()

def main():
    frozen = json.loads((ROOT/'freeze_manifest.json').read_text())
    for row in frozen['files']:
        assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest() == row['sha256']
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert protocol['status'] == 'FROZEN_BEFORE_EXTRACTION_AND_PROBE_FITS'
    assert len(protocol['images']) == 16
    receipts,features = ROOT/'receipts',ROOT/'features'
    receipts.mkdir(exist_ok=True); features.mkdir(exist_ok=True)
    with (receipts/'extraction_started.json').open('x') as f:
        json.dump({'pid':os.getpid(),'started_unix':time.time(),'phase_cap':16},f)
    print(json.dumps({'pid':os.getpid(),'status':'EXTRACTION_STARTED','selfcheck':selfcheck()}),flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model,pp,loaded = load_official_model()
    model = model.cuda()
    assert model.decoder.decoder.__class__.__module__.endswith('.deim.deim_decoder')
    assert model.backbone.proj_c1[0].in_channels == 16
    before = state_digest(model)
    assert before == 'fc5c4f1c3e52ea3e21cf3e431d4b2d34fb5cbb17cc1de9cea2afbc7997b4d82a'
    budget_path = BASE/'receipts/cumulative_forward_budget.json'
    budget = json.loads(budget_path.read_text())
    assert budget['attempted_full_model_forwards'] == 169 and budget['cap'] == 200
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    comparisons = []
    with torch.inference_mode():
        for index,spec in enumerate(protocol['images']):
            name = spec['file_name']
            path,boxes,labels,uids,ignored = annotation(name)
            with np.load(BASE/'traces'/(Path(name).stem+'.npz'),allow_pickle=False) as old:
                previous = {k:old[k].copy() for k in old.files}
            assert np.array_equal(uids,previous['gt_uids'])
            assert np.array_equal(labels,previous['gt_labels'])
            assert np.allclose(boxes,previous['gt_boxes'],rtol=0,atol=1e-8)
            sample = sample_points(boxes,ignored,name)
            captured,counts = {},{}
            def backbone_hook(module,args,out):
                counts['backbone'] = counts.get('backbone',0)+1
                captured['P2'],captured['P3_before'] = out[0].detach(),out[1].detach()
            def down_hook(module,args,out):
                counts['P2_down'] = counts.get('P2_down',0)+1
                assert args[0].shape == (1,192,160,160)
                captured['P2_down'] = out.detach()
            def encoder_hook(module,args,out):
                counts['encoder'] = counts.get('encoder',0)+1
                captured['P3_after'] = out[0].detach()
            def score_hook(module,args,out):
                counts['native_score'] = counts.get('native_score',0)+1
                captured['native_score'] = out[:,:6400].amax(-1).reshape(1,1,80,80).detach()
            hooks = [model.backbone.register_forward_hook(backbone_hook),
                     model.encoder.fpn_downsample_convs[1].register_forward_hook(down_hook),
                     model.encoder.register_forward_hook(encoder_hook),
                     model.decoder.enc_score_head.register_forward_hook(score_hook)]
            budget['attempted_full_model_forwards'] += 1
            assert budget['attempted_full_model_forwards'] <= 185
            budget['events'].append({'index':budget['attempted_full_model_forwards'],'phase':'A1_frozen_feature_readout16'})
            write_json(budget_path,budget)
            try:
                out = model(image_tensor(path))
            finally:
                for hook in hooks:
                    hook.remove()
            assert counts == {'backbone':1,'P2_down':1,'encoder':1,'native_score':1}
            assert captured['P2'].shape == (1,192,160,160)
            assert all(captured[key].shape == (1,192,80,80) for key in ['P2_down','P3_before','P3_after'])
            bd = float(np.max(np.abs(out['pred_boxes'][0].cpu().numpy()-previous['decoder_3_boxes'])))
            old_scores = torch.as_tensor(previous['decoder_3_logits']).sigmoid().numpy()
            sd = float(np.max(np.abs(out['pred_logits'][0].sigmoid().cpu().numpy()-old_scores)))
            flat = torch.topk(out['pred_logits'].sigmoid().flatten(1),300,dim=-1).indices[0].cpu().numpy()
            assert bd <= 1e-6 and sd <= 1e-6
            assert np.array_equal(flat//10,previous['native_pp_query_ids'])
            assert np.array_equal(flat%10,previous['native_pp_labels'])
            arrays = dict(sample,gt_boxes=boxes,gt_labels=labels,gt_uids=uids,
                          gt_sqrt_area=previous['gt_sqrt_area'],gt_crowd=previous['gt_crowd'],
                          file_name=np.array(name),group=np.array(spec['group']),fold=np.array(spec['fold']),
                          gt_all_encoder_max_iou=max_iou(boxes,cxcywh_to_xyxy(previous['encoder_all_valid_boxes'])),
                          gt_topk_max_iou=max_iou(boxes,cxcywh_to_xyxy(previous['encoder_topk300_boxes'])))
            for key in ('P2','P2_down','P3_before','P3_after'):
                arrays['features_'+key] = gather(captured[key],sample['xy'])
                assert np.isfinite(arrays['features_'+key]).all()
            arrays['native_score'] = gather(captured['native_score'],sample['xy'])
            target_file = features/(Path(name).stem+'.npz')
            np.savez_compressed(target_file,**arrays)
            comparisons.append({'file_name':name,'box_max_abs':bd,'score_max_abs':sd,'flat_query_and_class_ids_exact':True,
                                'points':len(sample['target']),'gt':len(boxes),'probe_eligible_gt':int(sample['gt_probe_eligible'].sum()),
                                'sha256':hashlib.sha256(target_file.read_bytes()).hexdigest(),'bytes':target_file.stat().st_size})
            write_json(receipts/'extraction_progress.json',{'status':'RUNNING','images':index+1,'forwards':budget['attempted_full_model_forwards']})
            print(json.dumps({'status':'IMAGE_EXTRACTED','image':index+1,'points':len(sample['target'])}),flush=True)
            del captured,out
    assert state_digest(model) == before
    result = dict(status='FROZEN_FEATURE_EXTRACTION_COMPLETE',images=16,comparisons=comparisons,
                  full_model_forwards_cumulative=budget['attempted_full_model_forwards'],new_full_model_forwards=16,
                  full_model_training_updates=0,main_model_state_unchanged=True,state_digest=before,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,
                  scope='Train-seen author checkpoint,first16 diagnostic images; no architecture change,no validation AP,no probe fits yet')
    write_json(receipts/'extraction_complete.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}),flush=True)

if __name__ == '__main__':
    main()
