from pathlib import Path
import collections, hashlib, importlib.util, json, time
import torch

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/root/uav_a47_20261101_v1/assets_A43_baseline')
DATA = Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    spec = importlib.util.spec_from_file_location('native_a43', ASSETS / 'runtime_common.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manifest = json.loads((ASSETS / 'train_data_manifest.json').read_text())
    counts = collections.Counter()
    total_area = 0.0
    input_area = collections.defaultdict(float)
    orig_counts = collections.Counter()
    samples = []
    for row in manifest['rows']:
        path = DATA / 'VisDrone2019-DET-train' / 'annotations' / (row['file'][:-4] + '.txt')
        assert sha(path) == row['annotation_sha256'], row['file']
        valid, _, _ = native.boxes_from_annotation(path, row['width'], row['height'])
        assert len(valid) == row['valid_gt']
        for x0, y0, x1, y1, cls in valid:
            area_original = (x1-x0)*(y1-y0)
            area_input = area_original * 640**2 / (row['width']*row['height'])
            group = 'lt8' if area_input < 8**2 else '8to16' if area_input < 16**2 else '16to32' if area_input < 32**2 else 'ge32'
            counts[group] += 1
            input_area[group] += area_input
            total_area += area_input
            orig_counts['small' if area_original < 32**2 else 'medium' if area_original < 96**2 else 'large'] += 1
        if len(samples) < 8:
            samples.append({'file':row['file'], 'boxes':valid[:5], 'valid_gt':len(valid)})
    assert sum(counts.values()) == manifest['valid_gt'] == 343204
    torch.set_num_threads(4)
    model, _, _ = native.create_components('B', role='cpu_check')
    sites = ['encoder.fpn_blocks.0.cv4.norm', 'encoder.fpn_blocks.1.cv4.norm',
             'encoder.pan_blocks.0.cv4.norm', 'encoder.pan_blocks.1.cv4.norm']
    site_info = {}
    for name in sites:
        layer = model.get_submodule(name)
        assert isinstance(layer, torch.nn.BatchNorm2d), (name, type(layer))
        site_info[name] = dict(type=type(layer).__name__, channels=layer.num_features, eps=layer.eps,
                               momentum=layer.momentum, affine=layer.affine, track_running_stats=layer.track_running_stats)
    result = dict(status='COMPLETE_CPU_ONLY', unix=time.time(), annotation_hashes_verified=len(manifest['rows']),
                  instances=sum(counts.values()), input_size=[640,640],
                  input_area_groups={g:{'count':counts[g], 'instance_fraction':counts[g]/sum(counts.values()),
                                        'summed_bbox_area':input_area[g], 'bbox_area_fraction':input_area[g]/total_area}
                                     for g in ['lt8','8to16','16to32','ge32']},
                  original_area_groups=dict(orig_counts),
                  summed_bbox_area_over_image_area=total_area/(6471*640**2),
                  interpretation='Summed box footprints count overlaps with multiplicity and include background/occlusions inside boxes. This is not a segmentation foreground ratio or evidence of normalization-caused detection errors.',
                  chosen_encoder_sites=site_info, model_parameters=sum(p.numel() for p in model.parameters()),
                  native_forward_counters=native.MODEL_COUNTER, training_updates=0, samples=samples,
                  script_sha256=sha(__file__), manifest_sha256=sha(ASSETS/'train_data_manifest.json'),
                  native_runtime_sha256=sha(ASSETS/'runtime_common.py'))
    (ROOT / 'data_and_site_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['samples']}), flush=True)

if __name__ == '__main__':
    main()
