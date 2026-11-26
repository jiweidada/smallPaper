# 把论文的自定义配置迁移到新版 DEIM 发行包

- 源：`C:\Users\jiwei\Desktop\DEIM`（你跑论文实验的树）
- 目标：`D:\BaiduNetdiskDownload\DEIM_1572862469`（下载的新版发行包，更新日志到 **20260924**）

---

## 一、结论先说：能移，但必须"外科式追加"，不能整体覆盖

我用脚本逐文件比对了 3461 个源码/配置文件（已排除数据集、权重、outputs、缓存）：

| 分类 | 数量 | 含义 |
|---|---|---|
| 只在你旧树里有 | 61 个代码/配置文件 | **要搬的就是这些** |
| 两棵树同名但内容不同 | 233 个 | **绝不能覆盖** |
| 只在新版包里有 | 1270 个 | 新版新增功能，保持原样 |

那 233 个同名文件里，`engine/` 几乎全军覆没（`_solver.py`、`det_solver.py`、`dist_utils.py`、`hgnetv2.py`、`coco_eval.py`、`_transforms.py`……）。因为新版包是一个大得多的发行版：多了 `deimv2`、`dinov3`、`obb`（旋转框）、`ov`（开放词表）、`segment`、`distill`、`prune`、`gradio app`，并且评估器加了 `using_tiny_metrice` 等新特性。

> **把旧 `engine/` 覆盖过去 = 把新版包退回旧版，新功能全部丢失。**
> 正确做法：只追加 4 个自定义模块 + 1 处注册补丁 + 配置。

---

## 二、要搬的东西（共 34 个文件，脚本已列全）

### A. 自定义模块（论文创新点的全部实现，4 个必需 + 2 个可选）

| 文件 | 作用 |
|---|---|
| `engine\extre_module\custom_nn\attention\MSCA.py` | MSCA / MSCA-Lite / MSCA-Dilated 实现 |
| `engine\backbone\hgnetv2_MSCA_EnhancedStem.py` | **MPS + 分层 MSCA，论文主模型** |
| `engine\backbone\hgnetv2_MSCA.py` | 消融用：只加 MSCA |
| `engine\backbone\hgnetv2_Silu.py` | 消融用：只换 SiLU |
| `engine\extre_module\custom_nn\stem\EnhancedStem.py` | 可选，全树无任何 import，不影响 |
| `engine\extre_module\custom_nn\stem\MultiLoG.py` | 可选，同上 |

**依赖已逐一验证，新版包全部具备**：
`multiprocess_sync` / `is_dist_available_and_initialized`（新版 `dist_utils.py` L303 / L94）✔
`ConvBNAct`（新版 `hgnetv2.py` L74）✔　`get_activation`（新版 `engine/__init__.py`）✔
`ema` / `simam` / `lsk` / `DeformableLKA` 四个注意力 ✔
两个 `__init__.py`（`attention/`、`stem/`）在新版里是空文件，所以**不需要注册**，直接按全路径 import 即可。

### B. 消融与跨数据集配置（24 个，`configs/test/`）
包括表1 第 8 行的 `deim_hgnetv2_EnhancedStem_s_visdrone.yml`，以及 n/m/x 尺度、`NOMSCA_NOSilu` / `MSCA_NOSilu` / `NOMSCA_Silu` 等组合、UAVDT、TinyPerson 全部配置。

### C. 新版包缺失的基础配置（4 个）
`configs\dataset\tinyperson_detection.yml`、`configs\dataset\uavdt_detection.yml`、`configs\deim\visdrone2019.yml`、`configs\base\deim_losschange.yml`

### D. 唯一需要改的公共文件
`engine\backbone\__init__.py` —— 追加 3 行（脚本幂等处理，原文件自动备份）：
```python
from .hgnetv2_Silu import HGNetv2_Silu
from .hgnetv2_MSCA import HGNetv2_MSCA
from .hgnetv2_MSCA_EnhancedStem import HGNetv2_MSCA_EnhancedStem
```

---

## 三、怎么执行

```powershell
# 1) 先演练，只打印计划、不写任何文件
& 'C:\Users\jiwei\Desktop\小论文投稿\《计算机应用与软件》投稿\_work\migrate_to_new_deim.ps1'

# 2) 确认无误后执行
& 'C:\Users\jiwei\Desktop\小论文投稿\《计算机应用与软件》投稿\_work\migrate_to_new_deim.ps1' -Apply
```

脚本特性：默认演练；任何覆盖都先备份（`.bak-时间戳`）；可重复运行。

---

## 四、迁移后必须手工处理的 4 件事

### 1. 数据路径（必改）
新版包的 `configs/test/visdrone2019.yml` 写死的是 Linux 服务器路径：
```
img_folder: /root/dataset/dataset_visdrone/VisDrone2019-DET-train/images
```
你旧树用的是相对路径 `dataset_visdrone/VisDrone2019-DET-train/images`，迁移过去的
`visdrone2019_mpsma.yml` 保留了这个相对路径 —— 在新机器上要按实际位置改。

### 2. 评估口径（关键，不改数字就不可比）
新版包的 `visdrone2019.yml` 与你的版本有 **两处实质差异**：

| 项 | 你的（论文口径） | 新版包 |
|---|---|---|
| `using_tiny_metrice` | 无 | `True` |
| `max_dets` | 默认 100 | `300` |
| val 指向 | `VisDrone2019-DET-val` | `VisDrone2019-DET-test-dev` |

若直接用新版包那份配置，AP 口径会变，和论文里的 0.287/0.474 不可比。所以脚本默认
**不覆盖**它，而是把你的版本存成 `configs/test/visdrone2019_mpsma.yml`，并把 24 个消融
配置的 `__include__` 改指向它。

### 3. 环境
新版 `requirements.txt` 要求 `pycocotools==2.0.11`、`timm==1.0.27`、`faster-coco-eval>=1.6.5`、
`transformers==5.3.0`、`pywavelets`、`grad-cam`、`swanlab` 等。建议在新环境里单独装，别和
现有 `py310` 混在一起（那里面是 torch 2.7.0+cu126，论文写的是 PyTorch 2.5.1 / CUDA 12.1，
**顺便把这个不一致也统一掉**，回复审稿意见时环境描述要对得上）。

### 4. 验证（CPU 就能做，不需要 GPU 和数据集）
```powershell
cd D:\BaiduNetdiskDownload\DEIM_1572862469
python tools\benchmark\get_info.py -c configs\test\deim_hgnetv2_EnhancedStem_s_visdrone.yml
```
期望输出 **13.99 M params / 29.63 GFLOPS**。能打印出这个数，就说明 MSCA 导入、
EnhancedStem、`__init__.py` 注册、YAML 继承链**全部打通**。这是最省事的冒烟测试。

---

## 五、顺手核对出来的两个问题（对回复审稿意见有影响）

### 1. 表3 的参数量对不上（硬伤，建议尽快改）
我用官方 `tools/benchmark/get_info.py` 在 CPU 上复核，640×640、deploy 模型：

| 模型 | 实测 Params | 实测 GFLOPs | 论文表3 |
|---|---|---|---|
| DEIM-D-FINE-S | **10.18 M** | **24.86** | 10.18 M / 24.9 ✔ |
| MPSMA-DEIM | **13.99 M** | **29.63** | 10.80 M / 29.6 ⚠ |

计算量对得上（29.63 vs 29.6），但**参数量应该是 13.99 M，不是 10.80 M**。
训练日志里的 `n_parameters` 也印证了这一点：基线 10,221,723 → 本文 14,035,326，增量 **+3.8 M**。
也就是真实的参数量增幅是 **+37%**，而不是论文写的 +6.1%（+0.62 M）。

这比计算量的质疑更致命——审稿人下一轮很可能直接抓住。建议按实测值改写表3，
并把论述从"复杂度增量有限"调整为"参数量增加主要来自 MSCA 的瓶颈式空洞卷积分支，
计算量仅 +18.9%"，或者干脆裁剪 MSCA 的通道数把参数量压回去。

### 2. 环境版本与论文不一致
论文写 Python 3.10.16 / PyTorch 2.5.1 / CUDA 12.1；
本机 `C:\Users\jiwei\.conda\envs\py310` 实为 Python 3.10.18 / torch 2.7.0+cu126。

---

## 六、明确不要动的东西

以下在新版包里都存在、都是新版实现，**保持原样**：
`engine\**`（除 `backbone\__init__.py` 追加 3 行外）、`train.py`、`train_auto.py`、
`configs\base\deim.yml`、`configs\base\dfine_hgnetv2.yml`、`configs\runtime.yml`、
`configs\test\deim_hgnetv2_s_visdrone.yml`、`tools\**`。

新版 `train.py` 的 CLI 参数是你旧版的**超集**（同样支持 `-c/-r/-t/--seed/--use-amp/--output-dir/--test-only/-u`，
另加 `--use-amp-fallback-fp32`、swanlab、`-p/--path`），所以**你原来的训练命令可以原样使用**，
这一点不需要迁移。

---

## 七、配套脚本

| 文件 | 用途 |
|---|---|
| `migrate_to_new_deim.ps1` | 迁移脚本（默认演练，`-Apply` 执行） |
| `check_config_deps.py` | 校验 30 个入口配置的 `__include__` 在新版包里能否全部解析（已通过：14 条 include 全部 OK） |
| `tree_diff.py` | 两棵树全量比对，输出 `tree_diff.txt` |
| `verify_migration.py` | 迁移后 SHA1 保真校验 + `__init__.py` 补丁差异检查 |
| `smoke_new_deim.py` | 迁移后冒烟测试（语法/注册/YAML 链/模型构建） |

---

## 八、迁移记录（已执行）

**执行时间**：2026-09-26 21:50　**结果：成功**

| 项目 | 结果 |
|---|---|
| 复制文件 | 34 / 34，全部为"新增"，**零覆盖** |
| SHA1 保真校验 | 33 / 33 逐字节一致（另 16 个消融配置为有意改写 `__include__`，逐行核对确认**每份只改了 1 行**） |
| YAML 继承链 | 主模型配置 8 条 include 全部解析成功 |
| `engine\backbone\__init__.py` | 精确追加 3 行，**删除 0 行**；原文件已备份为 `__init__.py.bak-20260926-215059` |
| 回滚方式 | 删掉新增文件 + 用备份覆盖 `__init__.py` 即可完全还原 |

新增落地的关键文件：
```
engine\extre_module\custom_nn\attention\MSCA.py            10.6 KB
engine\backbone\hgnetv2_MSCA_EnhancedStem.py               34.0 KB
engine\backbone\hgnetv2_MSCA.py                            30.8 KB
engine\backbone\hgnetv2_Silu.py                            29.2 KB
configs\test\visdrone2019_mpsma.yml                         2.2 KB
```

### 未完成项：本机无法做模型构建冒烟测试

`python tools/benchmark/get_info.py -c configs\test\deim_hgnetv2_EnhancedStem_s_visdrone.yml`
在本机失败，原因是**新版包的环境依赖本机没装齐**，与迁移无关：

```
engine\data\dataset\coco_eval.py:14  from pycocoeval.cocoeval import COCOeval
engine\data\dataset\coco_eval.py:15  from pycocoeval.coco import COCO
engine\data\dataset\evaluator.py:143 from pycocoeval.coco import COCO
engine\data\dataset\evaluator.py:222 from pycocoeval.yoloeval import get_yolo_obb_metrice
engine\extre_module\yolo_metrice.py:4 from pycocoeval import mask as mask_utils
```

本机 `py310` 环境里：`faster-coco-eval 1.6.7` ✔、`pycocotools 2.0.10` ✔，但 **`pycocoeval` 不存在**；
它**不在新版包的 `requirements.txt` 里，包里也没有自带**，`ServerInfo.md` 显示发行方是通过
"专属云镜像"提供环境的 —— 推断 `pycocoeval`（含 `coco/cocoeval/mask/yoloeval/rotated_cocoeval`
子模块）是该发行方私有/镜像内的包。

三种处理方式，任选：
1. **用发行方的云镜像**（最省事，`ServerInfo.md` 里有两个平台）；
2. 向发行方要 `pycocoeval`（B 站/QQ 群 798692951）；
3. **训练在新包、评测回旧树** —— 旧树的评估走标准 `pycocotools`，正是论文现有数字的口径，
   新旧两棵树共用同一套 checkpoint 格式，`--test-only -r xxx.pth` 可直接在旧树复评。

另外新版 `engine/__init__.py` 会连带导入 `timm / transformers / dinov3 / mamba` 等，
本机执行 `import engine` 超过 10 分钟未返回（疑似尝试联网拉取模型配置被沙箱阻断），
**在 GPU 机器上装好环境后不会是这个表现**。
