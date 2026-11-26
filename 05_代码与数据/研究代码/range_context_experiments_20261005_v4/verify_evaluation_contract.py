"""Known-IoU fixtures verify the new small-object AP endpoints; not research data."""
import argparse
import json
from pathlib import Path
import torch
from evaluate import coco_evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    annotation = {'info': {}, 'images': [{'id': 1, 'width': 64, 'height': 64, 'file_name': 'fixture'}],
                  'categories': [{'id': i, 'name': str(i)} for i in range(10)],
                  'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [10, 10, 10, 10], 'area': 100, 'iscrowd': 0}]}
    annpath = args.output / 'synthetic_annotation.json'
    annpath.write_text(json.dumps(annotation))
    results = {}
    for name, dx in [('perfect', 0), ('IoU_two_thirds', 2)]:
        out = args.output / name
        out.mkdir()
        predictions = [{'image_id': 1, 'category_id': 0, 'bbox': [10 + dx, 10, 10, 10], 'score': .9}]
        results[name] = coco_evaluate(annpath, predictions, out)
    checks = {'perfect_APs': abs(results['perfect']['metrics']['APs'] - 1) < 1e-9,
              'perfect_small_AP75': abs(results['perfect']['mechanism_metrics']['small_AP75'] - 1) < 1e-9,
              'perfect_small_AP85': abs(results['perfect']['mechanism_metrics']['small_AP85'] - 1) < 1e-9,
              'two_thirds_APs_four_of_ten_IoU_thresholds': abs(results['IoU_two_thirds']['metrics']['APs'] - .4) < 1e-9,
              'two_thirds_small_AP75_zero': results['IoU_two_thirds']['mechanism_metrics']['small_AP75'] == 0,
              'two_thirds_small_AP85_zero': results['IoU_two_thirds']['mechanism_metrics']['small_AP85'] == 0,
              'CPU_only': not torch.cuda.is_initialized()}
    assert all(checks.values())
    (args.output / 'receipt.json').write_text(json.dumps({'status': 'synthetic_evaluation_contract_pass_not_research_result', 'checks': checks}, indent=2))
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
