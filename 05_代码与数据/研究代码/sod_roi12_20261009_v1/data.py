"""Read-only native caches, native-PIL ROI inputs, and train-only A/B/C splits.

ROIDataset returns one complete image (no detection subsampling). Tensor/data
dependencies are imported lazily; make_splits itself uses only the stdlib and
does not open formal validation annotations. A/B/C mean refiner/gate/calibration.
Only optional binding artifacts and the requested split JSON are written.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import uuid


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _manifest(path):
    path = Path(path).resolve()
    if path.is_dir(): path = path / 'manifest.json'
    data = _read(path)
    if data.get('status') not in ('COMPLETE', 'ENGINEERING_SUBSET_COMPLETE'):
        raise ValueError(f'incomplete native cache: {path}')
    if not isinstance(data.get('images'), list) or not isinstance(data.get('files'), list):
        raise ValueError('native manifest requires images and full files records')
    return path, data


def _records(manifest):
    records = {}
    for r in manifest['files']:
        image_id = int(r['image_id'])
        if image_id in records: raise ValueError('duplicate image ID in manifest.files')
        records[image_id] = dict(r)
    result = {}
    for image in manifest['images']:
        image_id = int(image['image_id'])
        if image_id in result or image_id not in records:
            raise ValueError('manifest images/files disagree or duplicate image ID')
        record = records[image_id]
        for key in ('file_name', 'width', 'height', 'cache_file'):
            if image.get(key) != record.get(key):
                raise ValueError(f'manifest images/files disagree: {image_id}/{key}')
        for key in ('sha256', 'image_sha256'):
            value = record.get(key, '')
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
                raise ValueError(f'missing/invalid {key}: image {image_id}')
        result[image_id] = record
    if set(result) != set(records): raise ValueError('extra files outside manifest.images')
    return result


def _under(root, path):
    root = Path(root).resolve()
    p = (root / path).resolve()
    if p == root or root not in p.parents:
        raise ValueError(f'path escapes intended directory: {path}')
    return p


def _annotation_path(manifest_path, manifest):
    p = Path(manifest['annotation'])
    return p.resolve() if p.is_absolute() else (manifest_path.parent / p).resolve()


def _trainable_gt(row, width, height):
    """Native COCO preparation: remove crowd, clamp original edges, positive area.

    Category IDs remain 0..9; no COCO-category remapping and no fake GT labels.
    """
    if int(row.get('iscrowd', 0)) != 0: return None
    category = int(row['category_id'])
    if not 0 <= category < 10:
        raise ValueError('expected native VisDrone category IDs 0..9, no implicit remapping')
    xywh = row['bbox']
    if len(xywh) != 4 or not all(math.isfinite(float(x)) for x in xywh):
        raise ValueError('invalid GT box')
    x, y, w, h = map(float, xywh)
    box = [min(max(x, 0.), width), min(max(y, 0.), height),
           min(max(x+w, 0.), width), min(max(y+h, 0.), height)]
    if box[2] <= box[0] or box[3] <= box[1]: return None
    return box, category


class ROIDataset:
    """Map-style image dataset compatible with torch DataLoader.

    ROIDataset(cache_dir, image_ids, arm, bindings_cache_dir=None).
    Init hashes selected NPZ files and original images ONCE, and verifies the
    bound annotation SHA. __getitem__ does not rehash them every epoch. Source
    caches/images/annotations are never modified. Instantiate after caching
    has completed and keep those inputs immutable during training.

    Optional binding JSONs go outside the native cache, keyed by cache NPZ,
    annotation, and refiner source SHA. Otherwise a per-worker 128-image LRU
    avoids unbounded RAM. Set persistent_workers=True to retain worker LRUs.
    """
    def __init__(self, cache_dir, image_ids, arm, bindings_cache_dir=None):
        if arm not in ('raw', 'degraded'): raise ValueError('arm must be raw or degraded')
        self.manifest_path, self.manifest = _manifest(cache_dir)
        self.cache_dir = self.manifest_path.parent
        self.records = _records(self.manifest)
        self.image_ids = [int(x) for x in image_ids]
        if len(set(self.image_ids)) != len(self.image_ids): raise ValueError('duplicate requested image IDs')
        if not set(self.image_ids).issubset(self.records): raise ValueError('image IDs outside cache')
        self.arm = arm
        image_root = Path(self.manifest['image_root'])
        self.image_root = image_root.resolve() if image_root.is_absolute() else (self.cache_dir/image_root).resolve()
        annotation = _annotation_path(self.manifest_path, self.manifest)
        self.annotation_sha256 = _sha(annotation)
        if self.annotation_sha256 != self.manifest['annotation_sha256']:
            raise ValueError('annotation SHA differs from frozen cache manifest')
        annotation_data = _read(annotation)
        infos = {int(x['id']): x for x in annotation_data['images']}
        self.gt = {image_id: [] for image_id in self.image_ids}
        self.gt_raw_counts = Counter()
        for a in annotation_data['annotations']:
            image_id = int(a['image_id'])
            if image_id not in self.gt: continue
            self.gt_raw_counts[image_id] += 1
            info = infos[image_id]
            kept = _trainable_gt(a, int(info['width']), int(info['height']))
            if kept is not None: self.gt[image_id].append(kept)
        self.npz_paths, self.image_paths = {}, {}
        for image_id in self.image_ids:
            record = self.records[image_id]
            info = infos.get(image_id)
            if info is None or any(info[k] != record[k] for k in ('file_name','width','height')):
                raise ValueError(f'annotation/cache image metadata mismatch: {image_id}')
            npz_path = _under(self.cache_dir, record['cache_file'])
            image_path = _under(self.image_root, record['file_name'])
            if _sha(npz_path) != record['sha256']: raise ValueError(f'NPZ SHA mismatch: {image_id}')
            if _sha(image_path) != record['image_sha256']: raise ValueError(f'image SHA mismatch: {image_id}')
            self.npz_paths[image_id], self.image_paths[image_id] = npz_path, image_path
        self.bindings_cache_dir = None
        if bindings_cache_dir is not None:
            p = Path(bindings_cache_dir).resolve()
            if p == self.cache_dir or self.cache_dir in p.parents:
                raise ValueError('bindings_cache_dir must be OUTSIDE immutable native cache')
            p.mkdir(parents=True, exist_ok=True)
            self.bindings_cache_dir = p
        self.refiner_source_sha256 = _sha(Path(__file__).with_name('refiner.py'))
        self._binding_lru = OrderedDict()
        self.verification = {'verified_selected_images': len(self.image_ids),
                             'manifest_sha256': _sha(self.manifest_path),
                             'annotation_sha256': self.annotation_sha256,
                             'NPZ_and_image_sha_verified_at_init': True,
                             'arm': arm, 'GT_count': sum(map(len,self.gt.values()))}

    def __len__(self): return len(self.image_ids)

    def _bindings(self, image_id, boxes, labels, scores, gt_boxes, gt_labels):
        import torch
        from refiner import FixedBindings, fixed_bindings
        if image_id in self._binding_lru:
            value = self._binding_lru.pop(image_id)
            self._binding_lru[image_id] = value
            return value
        key_text = '|'.join(('native075_then050_v1', str(image_id),
                            self.records[image_id]['sha256'], self.annotation_sha256,
                            self.refiner_source_sha256))
        key = hashlib.sha256(key_text.encode('utf-8')).hexdigest()
        path = self.bindings_cache_dir / f'{image_id:06d}_{key}.json' if self.bindings_cache_dir else None
        if path is not None and path.exists():
            obj = _read(path)
            if obj.get('source_key') != key: raise ValueError('binding cache provenance mismatch')
            ids = torch.tensor(obj['det_to_gt'], dtype=torch.long)
            old = torch.tensor(obj['old_success'], dtype=torch.bool)
            if ids.shape != (len(boxes),) or old.shape != ids.shape:
                raise ValueError('binding cache shape mismatch')
            if bool(((ids < -1) | (ids >= len(gt_boxes))).any()) or bool((old & (ids < 0)).any()):
                raise ValueError('binding cache invalid GT reference')
            value = FixedBindings(ids,old)
        else:
            value = fixed_bindings(boxes,labels,scores,gt_boxes,gt_labels)
            if path is not None:
                _json_write(path, {'source_key':key, 'image_id':image_id,
                                  'det_to_gt':value.det_to_gt.tolist(),
                                  'old_success':value.old_success.tolist()})
        self._binding_lru[image_id] = value
        if len(self._binding_lru) > 128: self._binding_lru.popitem(last=False)
        return value

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image
        from refiner import prepare_rois
        image_id = self.image_ids[index]
        record = self.records[image_id]
        with np.load(self.npz_paths[image_id], allow_pickle=False) as a:
            required = ('boxes','labels','scores','query_ids','features','output_index')
            arrays = {k: a[k].copy() for k in required}
            if int(a['image_id']) != image_id or str(a['file_name']) != record['file_name']:
                raise ValueError('NPZ image identity mismatch')
            if int(a['width']) != record['width'] or int(a['height']) != record['height']:
                raise ValueError('NPZ original size mismatch')
        n = int(record['outputs'])
        shapes = {'boxes':(n,4), 'labels':(n,), 'scores':(n,), 'query_ids':(n,),
                  'features':(n,256), 'output_index':(n,)}
        for k, a in arrays.items():
            expected = np.dtype('float32') if k in ('boxes','scores','features') else np.dtype('int64')
            if a.shape != shapes[k] or a.dtype != expected or not np.isfinite(a).all():
                raise ValueError(f'invalid native {k} tensor for image {image_id}')
        if not np.array_equal(arrays['output_index'],np.arange(n,dtype=np.int64)):
            raise ValueError('native output order/index changed')
        if ((arrays['labels']<0)|(arrays['labels']>=10)).any(): raise ValueError('non-native labels')
        if ((arrays['query_ids']<0)|(arrays['query_ids']>=300)).any(): raise ValueError('non-native query IDs')
        tensors = {k:torch.from_numpy(a) for k,a in arrays.items()}
        with Image.open(self.image_paths[image_id]) as source:
            pil = source.convert('RGB')
            if pil.size != (record['width'],record['height']): raise ValueError('image dimensions changed')
            image = torch.from_numpy(np.array(pil,dtype=np.uint8,copy=True)).permute(2,0,1).float()/255.
            degraded = None
            if self.arm == 'degraded':
                low = pil.resize((640,640), resample=Image.Resampling.BILINEAR)
                degraded = torch.from_numpy(np.array(low,dtype=np.uint8,copy=True)).permute(2,0,1).float()/255.
        gt_rows = self.gt[image_id]
        gt_boxes = torch.tensor([r[0] for r in gt_rows],dtype=torch.float32).reshape(-1,4)
        gt_labels = torch.tensor([r[1] for r in gt_rows],dtype=torch.long)
        bindings = self._bindings(image_id,tensors['boxes'],tensors['labels'],tensors['scores'],gt_boxes,gt_labels)
        rois = prepare_rois(image,tensors['boxes'],tensors['query_ids'],arm=self.arm,
                            input_hw=(640,640),degraded_image=degraded)
        return {'image_id':image_id, 'rois':rois, 'boxes0':tensors['boxes'],
                'labels':tensors['labels'], 'scores':tensors['scores'],
                'query_ids':tensors['query_ids'], 'features':tensors['features'],
                'gt_boxes':gt_boxes, 'gt_labels':gt_labels, 'bindings':bindings,
                'output_index':tensors['output_index']}


def single_image_collate(items):
    """Picklable DataLoader collate_fn; explicitly requires batch_size=1."""
    if len(items) != 1: raise ValueError('ROI experiment requires image batch_size=1')
    return items[0]


def worker_init(worker_id):
    """Picklable CPU worker hook; sampling itself has no random subset/augment."""
    import torch
    torch.set_num_threads(1)


def _filename(row):
    return str(PurePosixPath(str(row['file_name']).replace('\\','/'))).casefold()


def _block(row):
    # A filename proxy only: not verified camera/flight/true sequence identity.
    return PurePosixPath(_filename(row)).stem.split('_',1)[0]


def make_splits(train_manifest, val_manifest, output, seed=20261009):
    """Stable filename-first-block hash A/B/C=70/20/10; no proportion fitting.

    IDs are split-local and may overlap train vs val. Filenames/basenames or
    image SHA overlap with val is forbidden. Validation GT is never opened.
    Training GT is opened only AFTER assignments to count actual populations.
    Returns dict; output may be a JSON filename or directory (splits.json).
    No retry/reassignment is performed to improve ratios or GT composition.
    """
    train_path, train = _manifest(train_manifest)
    val_path, val = _manifest(val_manifest)
    tr, va = _records(train), _records(val)
    if train.get('engineering_subset') or val.get('engineering_subset'):
        raise ValueError('scientific train splits require complete caches, not engineering subsets')
    train_names, val_names = {_filename(r) for r in tr.values()}, {_filename(r) for r in va.values()}
    train_base = {PurePosixPath(n).name for n in train_names}
    val_base = {PurePosixPath(n).name for n in val_names}
    train_sha, val_sha = {r['image_sha256'] for r in tr.values()}, {r['image_sha256'] for r in va.values()}
    overlap = {'file_names':sorted(train_names & val_names),
               'basenames':sorted(train_base & val_base), 'image_sha256':sorted(train_sha & val_sha)}
    if any(overlap.values()): raise ValueError('train/val image leakage: '+json.dumps(overlap))
    if len(train_names) != len(tr): raise ValueError('duplicate train filename')
    groups = defaultdict(list)
    for image_id,r in tr.items(): groups[_block(r)].append(image_id)
    # Content-identical train copies must not be spread across A/B/C by filename.
    sha_blocks = defaultdict(set)
    for r in tr.values(): sha_blocks[r['image_sha256']].add(_block(r))
    if any(len(blocks)>1 for blocks in sha_blocks.values()):
        raise ValueError('identical train image SHA appears in multiple filename blocks')
    splits = {key:{'purpose':purpose,'image_ids':[],'groups':[]} for key,purpose in
              (('A','refiner'),('B','gate'),('C','calibration'))}
    group_records = []
    for block,ids in sorted(groups.items()):
        digest = hashlib.sha256(f'{int(seed)}:{block}'.encode('utf-8')).hexdigest()
        fraction = int(digest,16)/(1<<256)
        name = 'A' if fraction < .7 else ('B' if fraction < .9 else 'C')
        splits[name]['image_ids'].extend(sorted(ids));splits[name]['groups'].append(block)
        group_records.append({'block':block,'split':name,'hash_sha256':digest,
                              'image_ids':sorted(ids),'image_count':len(ids)})
    # No access to val['annotation']; do not use formal-val GT to choose splits.
    annotation_path = _annotation_path(train_path,train)
    annotation_sha = _sha(annotation_path)
    if annotation_sha != train['annotation_sha256']: raise ValueError('train annotation SHA mismatch')
    annotation = _read(annotation_path)
    infos = {int(x['id']):x for x in annotation['images']}
    raw_counts,gt_counts = Counter(),Counter()
    for a in annotation['annotations']:
        image_id = int(a['image_id'])
        if image_id not in tr: continue
        raw_counts[image_id]+=1
        info=infos[image_id]
        if _trainable_gt(a,int(info['width']),int(info['height'])) is not None: gt_counts[image_id]+=1
    for record in group_records:
        record['GT_count']=sum(gt_counts[i] for i in record['image_ids'])
        record['annotation_row_count']=sum(raw_counts[i] for i in record['image_ids'])
    for name,s in splits.items():
        s['image_ids'].sort()
        s.update(group_count=len(s['groups']),image_count=len(s['image_ids']),
                 GT_count=sum(gt_counts[i] for i in s['image_ids']),
                 annotation_row_count=sum(raw_counts[i] for i in s['image_ids']))
        s['image_fraction']=len(s['image_ids'])/max(len(tr),1)
    if any(s['image_count']==0 for s in splits.values()):
        raise ValueError('hash split has an empty partition; do not silently retry a seed')
    result={'schema':'sod_roi12_train_abc_split_v1','seed':int(seed),
            'rule':'sha256(seed:casefolded_filename_stem_first_underscore_block), <.7 A, <.9 B, else C',
            'block_identity_caveat':'Filename first segment is a grouping proxy; true video/flight sequence identity has NOT been verified.',
            'image_id_scope':'Split-local; overlapping numeric train/val IDs are allowed.',
            'train_manifest':str(train_path),'train_manifest_sha256':_sha(train_path),
            'val_manifest':str(val_path),'val_manifest_sha256':_sha(val_path),
            'train_annotation_sha256':annotation_sha,'formal_val_GT_opened':False,
            'leakage_checks':{'train_val_overlap':overlap,'cross_partition_train_sha_overlap':False},
            'splits':splits,'groups':group_records,
            'totals':{'groups':len(groups),'images':len(tr),'GT_count':sum(gt_counts.values()),
                      'annotation_row_count':sum(raw_counts.values())},
            'GT_count_definition':'Native trainable GT: non-crowd, original xywh to xyxy, clamped to image, positive width/height; zero-based category IDs unchanged.'}
    out=Path(output)
    if out.suffix.lower() != '.json': out=out/'splits.json'
    if out.exists():
        if _read(out) != result: raise ValueError('existing split differs; refusing silent reassignment')
    else: _json_write(out,result)
    return result
