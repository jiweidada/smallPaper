# Repository Guidelines

## 项目结构与模块组织

`train.py` 是训练与评估的主入口。核心 Python 代码位于 `engine/`：配置与注册机制在 `engine/core`，数据管线在 `engine/data`，模型实现在 `engine/deim` 和 `engine/backbone`，训练循环在 `engine/solver`。实验 YAML 文件位于 `configs/`；通用设置放入 `configs/base`，数据集定义放入 `configs/dataset`。CUDA/C++ 扩展集中在 `compile_module/`。`tools/` 按用途存放推理、部署、性能测试、可视化和数据转换脚本。`research_lowlight/` 是独立的低光照研究流程。

将 `dataset*/`、`UAVDT_Datasets/`、`weight/`、`outputs/` 及生成的可视化或结果目录视为本地制品，不要作为源代码提交。

## 构建、测试与开发命令

- `pip install -r requirements.txt`：安装公共 Python 依赖；应先安装与本机 CUDA 匹配的 PyTorch。
- `python train.py -c configs/deim_dfine/deim_hgnetv2_n_coco.yml --seed 0`：启动单进程实验。
- `python train.py -c <config.yml> --test-only -r <checkpoint.pth>`：使用结构匹配的配置与权重进行评估。
- `torchrun --nproc_per_node=4 train.py -c <config.yml> --use-amp --seed 0`：启动四进程分布式训练。
- `python tools/inference/torch_inf.py -c <config.yml> -r <checkpoint.pth> --input <image>`：执行推理。
- `python research_lowlight/smoke_test.py`：验证低光照脚本与样本生成流程。

自定义算子应通过对应的 `compile_module/<operator>/setup.py` 构建，并确保编译器、PyTorch 与 CUDA 工具链版本兼容。

## 编码风格与命名约定

使用四空格缩进并遵循常规 Python 约定：函数、变量和模块采用 `snake_case`，类采用 `PascalCase`，常量采用 `UPPER_CASE`。导入顺序依次为标准库、第三方库和本地模块。项目未配置统一格式化或检查工具，因此修改应与相邻代码保持一致。实验配置使用描述性名称，例如 `deim_hgnetv2_n_coco.yml`。

## 测试规范

编译算子的 Pytest 测试与模块放在一起，例如 `compile_module/selective_scan/test_selective_scan.py`。在受支持的 GPU 环境中使用 `python -m pytest <path>` 运行针对性测试。新增测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。项目没有全局覆盖率门槛；每项修改都应附带聚焦的回归测试，或记录可复现的冒烟测试、评估命令。

## 提交与合并请求规范

当前检出内容不包含 Git 历史，无法确认既有提交格式。提交标题应简短并使用祈使语气，例如 `Fix COCO category remapping`；无关修改应拆分提交。合并请求需说明修改动机，列出验证所用配置和命令，关联相关问题；模型或可视化变更还应提供指标对比或截图。禁止提交数据集、凭据、检查点及生成结果。
