"""Reuse unchanged test-only predictions/evaluation arrays without GPU inference."""
import ast
import copy
import hashlib
import json
import os
import pathlib
import yaml


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_config(path):
    source = pathlib.Path.cwd()/'engine/core/yaml_utils.py'
    functions = [node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in ['load_config','merge_dict']]
    namespace = {'os':os,'copy':copy,'yaml':yaml,'Dict':dict,'INCLUDE_KEY':'__include__'}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['load_config'](str(path))


def import_export(args, cli_directory):
    import numpy as np
    import torch
    destination = pathlib.Path(args.out)
    destination.mkdir(parents=True,exist_ok=False)
    config = load_config(args.config)
    annotation = pathlib.Path(args.annotations or config['val_dataloader']['dataset']['ann_file'])
    gt = json.loads(annotation.read_text())
    images = {int(image['id']) for image in gt['images']}
    rows = json.loads((cli_directory/'pred_bbox.json').read_text())
    predictions = [{key:row[key] for key in ['image_id','category_id','bbox','score']} for row in rows]
    counts = {image:0 for image in images}
    for row in predictions:
        if int(row['image_id']) not in images:
            raise ValueError('CLI predictions contain an image outside the requested split')
        counts[int(row['image_id'])] += 1
    if any(count != 300 for count in counts.values()):
        raise ValueError('Expected exactly 300 CLI predictions for every annotated image')
    payload = torch.load(cli_directory/'eval.pth',map_location='cpu',weights_only=False)
    precision = np.asarray(payload['precision'])
    recall = np.asarray(payload['recall'])
    params = payload['params']
    def mean_ap(area='all', threshold=None):
        a=params.areaRngLbl.index(area);m=params.maxDets.index(100)
        values=precision[:,:,:,a,m]
        if threshold is not None:values=values[np.isclose(params.iouThrs,threshold)]
        valid=values[values>=0]
        return float(valid.mean()) if valid.size else -1.
    def mean_ar(area='all', maximum=100):
        values=recall[:,:,params.areaRngLbl.index(area),params.maxDets.index(maximum)]
        valid=values[values>=0]
        return float(valid.mean()) if valid.size else -1.
    metrics=[mean_ap(),mean_ap(threshold=.5),mean_ap(threshold=.75),mean_ap('small'),mean_ap('medium'),mean_ap('large'),
             mean_ar(maximum=1),mean_ar(maximum=10),mean_ar(),mean_ar('small'),mean_ar('medium'),mean_ar('large')]
    with (destination/'predictions.json').open('x') as output:json.dump(predictions,output,allow_nan=False)
    with (destination/'project_eval.json').open('x') as output:
        json.dump({'metrics_raw_0_to_1':metrics,'same_predictions_as_export':True,
                   'source':'unchanged test-only pred_bbox.json and eval.pth','small_AP50':mean_ap('small',.5),'small_AP75':mean_ap('small',.75)},output,indent=2)
    manifest={'checkpoint':str(pathlib.Path(args.checkpoint).resolve()),'checkpoint_sha256':digest(args.checkpoint),
              'weight_source':'test-only EMA module','config':str(pathlib.Path(args.config).resolve()),'config_sha256':digest(args.config),
              'annotations':str(annotation.resolve()),'annotations_sha256':digest(annotation),
              'predictions_sha256':digest(destination/'predictions.json'),'coordinate_space':'original_image',
              'shift_input_pixels':[0,0],'shift_reversed_in_export':True,'input_size':[640,640],
              'dtype':'FP32','amp':False,'deploy_fusion':True,'topk':300,'score_threshold':0,
              'image_count':len(images),'prediction_count':len(predictions),'complete':True,
              'reused_cli_directory':str(cli_directory),'CLI_prediction_sha256':digest(cli_directory/'pred_bbox.json'),
              'CLI_evaluation_arrays_sha256':digest(cli_directory/'eval.pth')}
    with (destination/'manifest.json').open('x') as output:json.dump(manifest,output,indent=2)
    print(json.dumps({'out':str(destination),'images':len(images),'reused_test_only_predictions':True,'GPU_inference_repeated':False}))
