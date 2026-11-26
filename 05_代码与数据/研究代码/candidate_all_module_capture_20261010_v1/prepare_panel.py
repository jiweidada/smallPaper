"""Freeze a source-only panel and code identity before any candidate module probe."""
import hashlib
import json
from pathlib import Path


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(annotation, destination):
    source = json.loads(Path(annotation).read_text(encoding='utf8'))
    digest = sha(annotation)
    assert digest == '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    images = {int(row['id']): row for row in source['images']}
    ids = sorted(images)
    assert ids == list(range(548))
    categories = {int(row['id']) for row in source['categories']}
    under4 = {iid: [] for iid in ids}
    for row in source['annotations']:
        w, h = row['bbox'][2:]
        if w <= 0 or h <= 0 or row.get('iscrowd', 0) or row['category_id'] not in categories:
            continue
        image = images[int(row['image_id'])]
        if min(w * 640 / image['width'], h * 640 / image['height']) < 4:
            under4[int(row['image_id'])].append(int(row['id']))
    candidates = []
    for index in range(137):
        batch = ids[4 * index:4 * index + 4]
        if not any(under4[iid] for iid in batch):
            continue
        token = 'module_output_panel_v1:' + digest + ':' + ','.join(map(str, batch))
        candidates.append((hashlib.sha256(token.encode()).hexdigest(), index, batch))
    chosen = sorted(candidates)[:2]
    batches = [dict(native_batch_index=index, image_ids=batch, selection_hash=token,
        source_float64_under4_annotation_ids={str(iid): under4[iid] for iid in batch})
        for token, index, batch in chosen]
    here = Path(__file__).resolve().parent
    value = dict(schema='candidate_all_module_panel_protocol_v1',
        annotation_sha256=digest, native_batch_size=4, validation_images=548,
        selected_batches=sorted(batches, key=lambda row: row['native_batch_index']),
        selection_rule='Choose two SHA-ranked sequential native validation batches containing at least one source-Float64 short-side<4px GT; no model output is used.',
        panel_scope='8 selected images only; not a validation population efficacy estimate.',
        output_precision='Native frozen evaluate calls model without autocast; do not infer eval AMP from use_amp training setting.',
        no_training_or_BASE_forward=True, candidate_forward_batches_per_seed=2,
        weights='target_protect saved fixed12 EMA; strict794 tensors',
        requires_original_random_training_reload_and_collection_stopped=True,
        no_automatic_retry=True, no_new_efficacy_gate=True,
        files={name: sha(here / name) for name in (
            'raw_module_recorder.py', 'check_raw_recorder.py', 'capture_candidate_panel.py')})
    write(destination, value)
    print(json.dumps(dict(status='SOURCE_ONLY_MODULE_PANEL_FROZEN',
        selected_batches=[dict(batch=row['native_batch_index'], ids=row['image_ids']) for row in value['selected_batches']],
        protocol_sha256=sha(destination), candidate_model_forwards=0, baseline_model_forwards=0)), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    prepare(args.annotation, args.destination)
