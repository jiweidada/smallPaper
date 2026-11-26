"""341 val548 cache builder. Same coordinate/window/M0/K_g/crop pipeline as 340 (frozen),
plus COCO fields (annotation_id) and native640 GT short side for tiny judgement."""
import os, sys, json, time, hashlib
import numpy as np, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v3 as S3
import stage1_cache_v4 as S4
from PIL import Image
torch.set_num_threads(2)
OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
VAL_ROOT = r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val"
VAL_ANN = os.path.join(VAL_ROOT, "annotations", "val.json")
VAL_IMG = os.path.join(VAL_ROOT, "images")
CKPT = {
  1: r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth",
  2: r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed2\run\last.pth",
}

def load_val_meta():
    ann = json.load(open(VAL_ANN, encoding='utf-8'))
    imgs = {im['id']: im for im in ann['images']}
    by = {}
    for a in ann['annotations']: by.setdefault(a['image_id'], []).append(a)
    order = [im['id'] for im in ann['images']]
    return imgs, by, order

def build_entry(model, im, items):
    pid = im['id']; ow, oh = im['width'], im['height']
    pil = Image.open(os.path.join(VAL_IMG, im['file_name'])).convert('RGB')
    pb, logits = S3.global_forward_640(model, pil)
    p300 = S3.norm_cxcywh_to_640_xyxy(pb)
    fullp = S3.make_fullp(pb, logits)
    rc, cnt = S3.select_window(p300)
    W = S3.window_xyxy(rc)
    if items:
        xywh = torch.tensor([a['bbox'] for a in items], dtype=torch.float32)
        labs = torch.tensor([a['category_id'] for a in items], dtype=torch.long)
        gt = S3.orig_xywh_to_640_xyxy(xywh, ow, oh)
        aids = [a['id'] for a in items]
    else:
        gt = torch.zeros(0, 4); labs = torch.zeros(0, dtype=torch.long); aids = []
    tg = S3.build_targets_v2(fullp, gt, labs, W)
    K = torch.zeros(gt.shape[0], 256)
    for i in range(gt.shape[0]):
        if bool(tg['G_W'][i]):
            x0, y0, x1, y1 = [float(v) for v in gt[i]]
            K[i] = torch.from_numpy(S4.bilinear_scatter_window([(x0+x1)/2, (y0+y1)/2, 1, 1], W)).float()
    crop = S4.crop128_from_window(pil, W, (ow, oh))
    gw = gt[tg['G_W']] if tg['G_W'].any() else torch.zeros(0,4)
    short = torch.minimum(gw[:,2]-gw[:,0], gw[:,3]-gw[:,1]) if gw.numel() else torch.zeros(0)
    return {'image_id': pid, 'orig_wh': [ow, oh], 'file_name': im['file_name'],
            'final300_boxes': pb, 'final300_logits': logits,
            'fullp_boxes': fullp['boxes'], 'fullp_scores': fullp['scores'],
            'fullp_labels': fullp['labels'], 'fullp_query_index': fullp['query_index'],
            'fullp_logits_row': fullp['logits_row'],
            'window_cell': list(rc), 'window_xyxy_640': W,
            'G_W': tg['G_W'], 'U_W': tg['U_W'],
            'm0_gt_explained': tg['m0']['gt_explained'], 'm0_cand_owner': tg['m0']['cand_owner'],
            'm0_coverage': int(tg['m0']['coverage']),
            'K_g': K, 'gt_boxes_640': gt, 'gt_labels': labs, 'gt_annotation_ids': aids,
            'gt_short_side_640': short,
            'n_gt': int(gt.shape[0]), 'n_G_W': int(tg['G_W'].sum()), 'n_U_W': int(tg['U_W'].sum()),
            'crop128': crop}

def entry_path(seed, iid):
    d = os.path.join(OUT, f'341_cache_val_seed{seed}')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{iid:06d}.pt')