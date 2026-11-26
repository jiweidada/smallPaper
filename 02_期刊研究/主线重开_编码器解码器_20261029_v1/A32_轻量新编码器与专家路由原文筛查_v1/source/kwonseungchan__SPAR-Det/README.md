# SPAR-Det

## Introduction

**Abstract**: Small Object Detection (SOD) is essential for real-world applications, including satellite image analysis and drone-based surveillance, where target objects typically exhibit limited spatial extent, high density, and visual similarity to complex backgrounds. These factors substantially hinder conventional object detection methods, resulting in weak feature representations and poor detection accuracy. To overcome these challenges, we introduce Segmentation-guided and Prior-Aided Routing for Small Object Detection (SPAR-Det), a unified framework integrating segmentation-guided attention, geometric prior supervision, and adaptive feature routing. At its core, SPAR-Det employs a Cross-Attention Heterogeneous Feature Fusion (CAHF) module that leverages pretrained segmentation backbones to enhance foreground object features while effectively suppressing background. Additionally, we propose a Geometric Prior Supervision Loss that combines Gaussian bounding box maps with segmentation feature maps, providing crucial geometric context and semantic cues to address the limited self-representation capability of small objects. Furthermore, our framework includes a Mixture-of-Experts (MoE) detection head, dynamically allocating specialized classifiers according to varying scene characteristics, thereby significantly improving generalization across diverse environments. Extensive evaluations conducted on two benchmark datasets, AI-TOD and VisDrone, demonstrate that SPAR-Det achieves state-of-the-art performance, verifying its robustness and applicability for challenging small object detection scenarios.


<img width="1529" height="798" alt="Image" src="https://github.com/user-attachments/assets/2af294a4-7c53-4c99-9c01-f19d2135ede4" />

## Installation and Get Started 
Required enviroment
- Linux
- Python 3.9+
- Pytorch 2.0
- CUDA 11.8+
- GCC 5+
- MMCV
- cocoapi-aitod

Install:
Note that this repository is based on the MMDetection. Assume that your enviroment has satisfied the above requirements, please follow the following steps for installation.

Please follow the instructions in [`mmdetection_install.ipynb`](./mmdetection_install.ipynb) to set up all required dependencies and environments.

This repository requires a pretrained segmentation model.

You can download it from Google Drive:

👉 [segmentation_aitod.pth – Pretrained Model (G-Drive)](https://drive.google.com/file/d/1-QPhbeCt6XFGpwq97XQa0Q593SW_2jkx/view?usp=sharing)

```
After downloading, place the file inside the pretrained/ directory:

SPAR-Det/
├── pretrained/
│   └── segmentation_aitod.pth

```

## Optional: Train Your Own Segmentation Model

If you do not wish to use the provided pretrained model,
you can train your own segmentation model using the script and dataloader in the segmentation/ directory.

Directory structure:
```
segmentation/
├── train.py
├── dataloader/
│   └── ...

```
To start training, simply run:

```
python segmentation/train.py

```
Once training is complete, place the trained weight file inside the pretrained/ directory
just like the provided model mentioned above:
```
SPAR-Det/
├── pretrained/
│   └── your_trained_model.pth
```
Make sure to update the model path in your code accordingly if you use a different filename.



## Performance on AI-TOD Dataset
| Method        | AP<sub>vt</sub> | AP<sub>t</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP       | mAP      |
| ------------- | --------------- | -------------- | -------------- | -------------- | -------- | -------- |
| RetinaNet     | 0.8             | 2.8            | 10.8           | 19.2           | 2.4      | 5.4      |
| CenterNet     | 1.3             | 6.9            | 12.9           | 22.3           | 8.2      | 24.3     |
| PANet         | 0.0             | 5.4            | 21.6           | 32.5           | 9.7      | 22.2     |
| Faster R-CNN  | 0.0             | 5.2            | 21.7           | 32.5           | 9.7      | 22.1     |
| Cascade R-CNN | 0.1             | 7.9            | 23.0           | 34.8           | 11.8     | 25.6     |
| AutoAssign    | 3.5             | 13.5           | 17.0           | 24.9           | 13.4     | 36.2     |
| KLDNet        | 8.4             | 20.6           | 22.7           | 26.4           | 19.6     | -        |
| DetectoRS     | 0.1             | 10.6           | 27.7           | 37.7           | 14.5     | 32.2     |
| DINO          | 9.9             | 23.1           | 29.3           | 37.6           | 23.2     | 56.6     |
| RFLA          | 9.3             | 24.8           | 30.3           | 38.2           | 24.8     | 55.2     |
| BRSTD         | 11.6            | 26.0           | 32.3           | 38.5           | 26.1     | 58.0     |
| SR-TOD        | 4.8             | 24.8           | 29.3           | -              | 24.0     | 54.6     |
| ORFENet       | 9.7             | 24.4           | 28.7           | 35.1           | 24.8     | 55.4     |
| BAFNet        | **16.6**        | 31.3           | 35.1           | 40.5           | 30.5     | 59.8     |
| **Ours**      | 16.0            | **31.5**       | **35.7**       | **41.0**       | **31.2** | **61.2** |
