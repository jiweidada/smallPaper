# PUD-DETR: Positive-Unlabeled Deformable DETR for Object Detection under Incomplete Annotations

**Junhyung Kim\*, Jiseok Son\*, Taehyung Kim** (\*equal contribution)

AIS Lab, Hongik University

> **Abstract:** Object detection models are typically trained under the assumption of complete and accurate annotations, whereas label omission frequently occurs in real-world scenarios — particularly in expert-driven domains where exhaustive bounding box labeling is impractical. Treating unlabeled positive instances as negatives induces incorrect classification losses that significantly degrade model performance. We propose **PUD-DETR**, which integrates non-negative Positive-Unlabeled (nnPU) learning into Deformable DETR by treating Hungarian-unmatched queries as unlabeled rather than definite negatives, effectively mitigating false-negative supervision caused by missing labels.

---

## Results

### PASCAL VOC 2007 — Increasing Label Omission (Table 2)

| Drop Ratio | Method | AP | AP50 | AP75 | AR@1 |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0.35 | Baseline | 0.5236 | 0.7317 | 0.5823 | 0.4369 |
| 0.35 | **PUD-DETR** | **0.5293** | **0.7426** | **0.5923** | **0.4526** |
| 0.48 | Baseline | 0.5120 | 0.7108 | 0.5722 | 0.4326 |
| 0.48 | **PUD-DETR** | **0.5265** | **0.7408** | **0.5890** | **0.4472** |
| 0.67 | Baseline | 0.4267 | 0.5974 | 0.4768 | 0.4092 |
| 0.67 | **PUD-DETR** | **0.4882** | **0.6811** | **0.5420** | **0.4366** |

PUD-DETR consistently outperforms the baseline across all drop ratios. The gain widens as label omission increases (+6.15 AP at drop=0.67).

### Diatom Dataset — Inherently Incomplete Annotation (Table 3)

| Method | AP | AP50 | AP75 | AR@1 | AR@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | 0.4182 | 0.4310 | 0.4309 | 0.7866 | 0.8560 |
| **PUD-DETR** | **0.4761** | **0.4925** | **0.4923** | **0.7926** | **0.8728** |

---

## Installation

```bash
pip install torch torchvision
pip install pytorch-lightning
pip install transformers
pip install pycocotools
pip install scipy
```

The code is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and uses the HuggingFace `transformers` implementation (`SenseTime/deformable-detr`), which is downloaded automatically on first run.

---

## Data Preparation

### PASCAL VOC 2007

Download the PASCAL VOC 2007 dataset from the [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and place images under `data/voc/JPEGImages/`.

Pre-split COCO-format annotation files for training, validation, and test sets — along with randomly dropped variants for incomplete annotation experiments — are provided in the `annotations/` directory:

```
annotations/
├── pascal_train.json                   # full training set
├── pascal_train_drop_{0.1–0.7}.json    # training sets with labels randomly removed
├── pascal_val.json                     # validation set
└── pascal_test.json                    # test set
```

The effective omission ratios corresponding to each drop probability are approximately:

| File | Drop prob. | Effective omission ratio |
|:---|:---:|:---:|
| `pascal_train_drop_0.1.json` | 0.1 | ~0.10 |
| `pascal_train_drop_0.3.json` | 0.3 | ~0.35 |
| `pascal_train_drop_0.4.json` | 0.4 | ~0.43 |
| `pascal_train_drop_0.5.json` | 0.5 | ~0.48 |
| `pascal_train_drop_0.6.json` | 0.6 | ~0.62 |
| `pascal_train_drop_0.7.json` | 0.7 | ~0.67 |

### Diatom Dataset

The diatom dataset (Gunduz et al., 2022) consists of 2,184 microscopy images across 68 diatom species with inherently incomplete annotations. Place images under `data/diatom/images/` and prepare COCO-format annotation files (`diatom_train.json`, `diatom_val.json`) following the split used in the paper (1,520 / 327 / 337 for train / val / test).

The dataset is available at: [DOI: 10.55730/1300-0632.3938](https://doi.org/10.55730/1300-0632.3938)

---

## Training

All models are initialized from `SenseTime/deformable-detr` (pre-trained on COCO) and fine-tuned for 20 epochs. Evaluation uses COCO-style mAP@[IoU=0.50:0.95]; the checkpoint with the best validation mAP is reported.

### Baseline (PN learning)

```bash
python train_base.py \
    --train_json    annotations/pascal_train_drop_0.7.json \
    --val_json      annotations/pascal_val.json \
    --train_img_dir data/voc/JPEGImages \
    --val_img_dir   data/voc/JPEGImages \
    --log_dir       logs/voc_drop_0.7 \
    --log_name      baseline \
    --devices 0
```

### PUD-DETR (nnPU learning)

```bash
python train_pu.py \
    --train_json    annotations/pascal_train_drop_0.7.json \
    --val_json      annotations/pascal_val.json \
    --train_img_dir data/voc/JPEGImages \
    --val_img_dir   data/voc/JPEGImages \
    --log_dir       logs/voc_drop_0.7 \
    --log_name      pud_detr \
    --alpha         5.0 \
    --devices 0
```

**Key argument — `--alpha`:** The class-prior scaling factor α (Eq. 21 in the paper). It compensates for the underestimation of the true positive ratio caused by missing labels. α=5.0 yields the best performance on both PASCAL VOC 2007 and the diatom dataset (see sensitivity analysis below).

### Common Training Arguments

| Argument | Default | Description |
|:---|:---:|:---|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 1e-4 | Learning rate for transformer modules |
| `--lr_backbone` | 1e-5 | Learning rate for backbone |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--gradient_clip_val` | 0.1 | Gradient clipping |
| `--warmup_steps` | 2000 | Linear warm-up steps (followed by cosine annealing) |
| `--devices` | [0] | GPU device ID(s) |

---

## Reproducing Paper Results

`run_experiments.sh` contains commands to reproduce all tables and ablations in the paper. Edit the `VOC_IMG` and `DIATOM_IMG` variables at the top of the script to match your data paths, then run:

```bash
bash run_experiments.sh
```

The script covers:
- **Table 1** — VOC 2007 under complete annotation
- **Table 2** — VOC 2007 under increasing label omission (drop 0.1–0.7)
- **Table 3** — Diatom dataset with inherently incomplete annotation
- **Table 4 & 5** — Alpha sensitivity analysis (α ∈ {1, ..., 10})
- **Table 6** — Non-negative correction order ablation (global / element-wise / query-wise)

Training logs and checkpoints are saved under `logs/<experiment>/<run_name>/`.

---

## Alpha Sensitivity (Table 4 — VOC drop=0.67)

| α | 1.0 | 2.0 | 3.0 | 4.0 | **5.0** | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AP | 0.448 | 0.471 | 0.486 | 0.489 | **0.504** | 0.501 | 0.496 | 0.496 | 0.494 | 0.488 |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{kim2025puddetr,
  title   = {{PUD-DETR}: Positive-Unlabeled Deformable {DETR} for Object Detection under Incomplete Annotations},
  author  = {Kim, Junhyung and Son, Jiseok and Kim, Taehyung},
  journal = {Preprint},
  year    = {2025}
}
```

---

## Acknowledgements

This implementation builds on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) (Zhu et al., 2020) and the nnPU risk estimator proposed by [Kiryo et al. (NeurIPS 2017)](https://arxiv.org/abs/1703.00593).
