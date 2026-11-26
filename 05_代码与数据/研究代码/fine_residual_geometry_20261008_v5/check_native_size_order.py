"""Reproduce the old WH/HW bug and check native rectangular synthetic samples."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from experiment import HERE, verify_bundle
from runtime import activate_training_snapshot, sha
from evaluator_binding import activate
from stage_trace import StageTrace, TracedLoader
from test_stage_trace import ToyModel, Loader

sys.path.insert(0, str(HERE/'native_import_deps'))
import torch
torch.set_num_threads(4)
attempts = []
def forbidden(name):
    def block(*args, **kwargs):
        attempts.append(name)
        raise RuntimeError('Size-order check forbids: ' + name)
    return block
torch.load = forbidden('checkpoint_load')
torch.optim.Optimizer.__init__ = forbidden('optimizer_construction')
torch.nn.Parameter.__new__ = staticmethod(forbidden('parameter_construction'))
torch.cuda._lazy_init = forbidden('CUDA_initialization')


def main():
    out = HERE/'engineering_native_sizes'
    out.mkdir(exist_ok=False)
    receipt, plan, paths = verify_bundle(HERE/'prepared_local/baseline_seed1')
    activate_training_snapshot(paths['native_root'])
    activate(paths['evaluator_root'], receipt['identity']['evaluator_binding'])
    from engine.data.dataset.coco_dataset import CocoDetection
    from engine.data.transforms.container import Compose
    from engine.deim.postprocessor import scale_boxes_to_original_image
    checks=[]
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    sizes_wh = [(480,360),(240,400),(512,512)]
    images = out/'synthetic_images'
    images.mkdir()
    annotation = {'images':[], 'annotations':[],
                  'categories':[{'id':c,'name':str(c)} for c in range(10)]}
    for iid,(w,h) in enumerate(sizes_wh):
        name=f'{iid}.png'
        Image.new('RGB',(w,h),(20+iid,40,60)).save(images/name)
        annotation['images'].append({'id':iid,'width':w,'height':h,'file_name':name})
        annotation['annotations'].append({'id':iid+1,'image_id':iid,'category_id':0,
                                         'bbox':[20,10,40,20],'area':800,'iscrowd':0})
    ann_file=out/'synthetic_annotations.json'
    ann_file.write_text(json.dumps(annotation),encoding='utf-8')
    transform=deepcopy(plan['native_baseline_config']['val_dataloader']['dataset']['transforms'])
    assert transform.pop('type')=='Compose'
    dataset=CocoDetection(str(images),str(ann_file),Compose(**transform),num_classes=10,
                          cache_imgsz=640,ram_cache=False,return_masks=False)
    samples=[dataset[i] for i in range(3)]
    for iid,(sample,target) in enumerate(samples):
        check(f'{iid}: native dataset preserves WH original dimensions through validation resize',
              target['orig_size'].tolist()==list(sizes_wh[iid]) and tuple(sample.shape)==(3,640,640))
    batches=[(torch.stack([samples[i][0] for i in (0,1)]),[samples[i][1] for i in (0,1)]),
             (samples[2][0].unsqueeze(0),[samples[2][1]])]
    identity={'arm':'baseline','seed':1,'val_annotation_sha256':sha(ann_file)}
    old_path=HERE.parent/'fine_residual_geometry_20261008_v4'/'stage_trace.py'
    spec=importlib.util.spec_from_file_location('old_stage_trace_size_reproducer',old_path)
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_errors=[]
    for iid in (0,1):
        model=ToyModel().eval()
        observer=old.StageTrace(model,identity,ann_file,mode='synthetic_engineering')
        observer.begin_batch((samples[iid][0].unsqueeze(0),[samples[iid][1]]))
        old_errors.append({'image_id':iid,'errors':list(observer.errors)})
        check(f'{iid}: original v4 observer rejects actual native rectangular target',
              any('Original image size differs from annotation' in error for error in observer.errors))
        observer.close()
    old_model=ToyModel().eval()
    old_square=old.StageTrace(old_model,identity,ann_file,mode='synthetic_engineering')
    old_square.begin_batch(batches[1])
    check('square fixture conceals original order bug',not old_square.errors)
    old_square.close()
    (out/'old_failure_reproduction.json').write_text(json.dumps({
        'status':'REPRODUCED_OLD_SIZE_ORDER_BUG','old_stage_sha256':sha(old_path),
        'native_orig_size_order':'width_height','rectangular_failures':old_errors,
        'old_model_forwards':0},ensure_ascii=False,indent=2),encoding='utf-8')

    original=Loader(batches)
    targets_before=[{k:v.clone() if torch.is_tensor(v) else deepcopy(v) for k,v in target.items()}
                    for _,target in samples]
    model=ToyModel().eval()
    check('fixture has no trainable model parameters',not list(model.parameters()))
    with torch.inference_mode():
        expected=[model(batch[0]) for batch in batches]
    counts_before=(model.calls,model.decoder.decoder.calls,model.decoder.decoder.layers[0].calls)
    rng=torch.get_rng_state().clone()
    observer=StageTrace(model,identity,ann_file,mode='synthetic_engineering')
    for i,batch in enumerate(TracedLoader(original,observer)):
        check(f'batch {i}: wrapped input identity unchanged',batch is batches[i])
        with torch.inference_mode():
            actual=model(batch[0])
        check(f'batch {i}: recording leaves output bitwise unchanged',
              all(torch.equal(actual[k],expected[i][k]) for k in actual))
    manifest=observer.finish(out/'trace')
    counts_after=(model.calls,model.decoder.decoder.calls,model.decoder.decoder.layers[0].calls)
    check('no additional forwards from record keeping',
          tuple(a-b for a,b in zip(counts_after,counts_before))==(2,2,2))
    check('RNG unaffected by recording',torch.equal(rng,torch.get_rng_state()))
    check('all observation hooks removed',not model._forward_hooks and
          not model.decoder.decoder._forward_hooks and not model.decoder.decoder.layers[0]._forward_pre_hooks)
    for iid,(w,h) in enumerate(sizes_wh):
        with np.load(manifest.parent/f'image_{iid:06d}.npz',allow_pickle=False) as saved:
            check(f'{iid}: saved original_hw explicitly stores height then width',
                  saved['original_hw'].tolist()==[h,w])
            batch_index,row=(0,iid) if iid<2 else (1,0)
            check(f'{iid}: image identity and final predictions preserved',
                  np.array_equal(saved['final_boxes'],expected[batch_index]['pred_boxes'][row].numpy()))
        current=samples[iid][1]
        check(f'{iid}: native target values unmodified',
              all(torch.equal(current[k],v) if torch.is_tensor(v) else current[k]==v
                  for k,v in targets_before[iid].items()))
        native_box=scale_boxes_to_original_image(torch.tensor([[[.5,.5,.5,.5]]]),
                                                  current['orig_size'].unsqueeze(0))
        check(f'{iid}: native box restoration uses WH without swapping',
              torch.equal(native_box,torch.tensor([[[w*.25,h*.25,w*.75,h*.75]]])))
    for label,wrong in [('swapped_wh',[360,480]),('wrong_width',[479,360])]:
        bad_target=deepcopy(samples[0][1])
        bad_target['orig_size']=torch.tensor(wrong)
        wrong_observer=StageTrace(ToyModel().eval(),identity,ann_file,mode='synthetic_engineering')
        wrong_observer.begin_batch((samples[0][0].unsqueeze(0),[bad_target]))
        check(label+' still rejected',bool(wrong_observer.errors))
        try:
            wrong_observer.finish(out/label)
        except ValueError:
            pass
        else:
            raise AssertionError('Incorrect size accepted for completed archive')
        check(label+' cannot produce a completed archive',not (out/label).exists())
    summary=json.loads((manifest.parent/'summary.json').read_text(encoding='utf-8'))
    check('both input and stored dimension orders declared',
          summary['native_target_orig_size_order']=='width_height' and
          summary['stored_original_hw_order']=='height_width')
    from stage_trace import verify_trace
    try:
        verify_trace(manifest.parent,identity)
    except ValueError:
        check('synthetic trace cannot be accepted as a trained endpoint',True)
    else:
        raise AssertionError('Synthetic data accepted as formal evidence')
    real=json.loads(paths['val_ann'].read_text(encoding='utf-8'))
    dimension_counts={name:sum(1 for im in real['images'] if condition(im)) for name,condition in
        [('landscape',lambda im:im['width']>im['height']),('portrait',lambda im:im['width']<im['height']),
         ('square',lambda im:im['width']==im['height'])]}
    check('all checks stayed within CPU engineering scope',not attempts and not torch.cuda.is_initialized())
    report={'status':'PASS_NATIVE_RECTANGULAR_DATA_AND_STAGE_ARCHIVE_INTERFACE',
        'checks':checks,'check_count':len(checks),'old_failure_reproduced':True,
        'native_dataset':'CocoDetection with bound Resize/ConvertPILImage validation pipeline',
        'native_source_root':str(paths['native_root']),'synthetic_sizes_wh':sizes_wh,
        'real_annotation_metadata_only':{'path':str(paths['val_ann']), 'sha256':sha(paths['val_ann']),
                                         'dimension_counts':dimension_counts},
        'real_research_images_read':0,'parameter_free_toy_forwards':4,
        'detector_forwards':0,'optimizer_constructions':0,'parameter_updates':0,
        'checkpoint_loads':0,'training_started':False,'CUDA_initialized':False,
        'code_sha256':{name:sha(HERE/name) for name in ('stage_trace.py','check_native_size_order.py')},
        'old_stage_sha256':sha(old_path),'synthetic_trace_manifest_sha256':sha(manifest)}
    with (out/'report.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':report['status'],'checks':len(checks),
                     'old_failure_reproduced':True,'real_annotation_dimensions':dimension_counts}))


if __name__=='__main__':
    main()
