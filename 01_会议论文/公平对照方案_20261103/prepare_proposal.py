"""Prepare a reviewable two-arm training proposal using static local evidence.

Standard library only: no model imports, weight loading, inference, training,
network access, process launching, or changes to the manuscript/source tree.
"""
from pathlib import Path
import ast
import copy
import hashlib
import itertools
import json

OUT = Path(__file__).resolve().parent
PAPER = OUT.parent
AUDIT = PAPER / '修订记录_20261103'
CPU = PAPER / '本地CPU核验_20261103'
CODE = CPU / 'source_snapshot'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
readj = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))


def writej(name, value):
    path = OUT / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return sha(path)


def flat(value, prefix=''):
    result = {}
    for key, val in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(val, dict):
            result.update(flat(val, path))
        else:
            result[path] = val
    return result


args = {rid: readj(AUDIT / 'source_snapshot' / rid / 'args.json') for rid in ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']}
source_records = []
for rid in args:
    path = AUDIT / 'source_snapshot' / rid / 'args.json'
    source_records.append({'path': str(path), 'sha256': sha(path)})

training_keys = ['train_dataloader', 'val_dataloader', 'optimizer', 'lr_scheduler', 'lr_warmup_scheduler',
                 'lrsheduler', 'lr_gamma', 'warmup_iter', 'flat_epoch', 'no_aug_epoch', 'epoches',
                 'use_amp', 'use_ema', 'ema', 'seed', 'resume', 'tuning', 'clip_max_norm']
training_views = {}
for rid, a in args.items():
    cfg = a['yaml_cfg']
    training_views[rid] = flat({k: cfg.get(k, a.get(k)) for k in training_keys})
pair_checks = []
for left, right in itertools.combinations(['N0', 'N1', 'N2', 'N3', 'N4'], 2):
    a, b = training_views[left], training_views[right]
    diff = {k: {left: a.get(k), right: b.get(k)} for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}
    pair_checks.append({'pair': [left, right], 'declared_training_field_differences': diff,
                        'historical_source_runtime_identity_established': False})
writej('existing_pair_audit.json', {
    'scope': 'Static comparison of archived declared fields, not proof of actual execution equality.',
    'source_records': source_records, 'pairs': pair_checks,
    'finding': 'No historical N0/full-N4 pair has matched batches. N1/N2/N4 have the same selected declared training fields but change combinations of architectural factors; none is the original host baseline. N3 also records resume to its own last checkpoint. Source/runtime linkage remains missing for all historical pairs.',
    'new_experiments': False
})

# Build both configs from one declared N4 recipe, then substitute only the
# baseline backbone. They are proposals, not launch authorization.
common = copy.deepcopy(args['N4']['yaml_cfg'])
for key in list(common):
    if key.startswith('HGNetv2') or key in ['__include__', 'config', 'output_dir', 'summary_dir']:
        common.pop(key)
common.update({'seed': 0, 'last_epoch': -1, 'resume': None, 'tuning': None, 'test_only': False,
               'device': 'cuda:0', 'epoches': 320, 'use_amp': False})
dataset_audit = readj(AUDIT / 'dataset_eval_audit/annotation_audit.json')
for split in ['train', 'val']:
    loader = common[f'{split}_dataloader']
    loader['total_batch_size'] = 4
    annotation = Path(dataset_audit[split]['annotation_path'])
    assert sha(annotation) == dataset_audit[split]['annotation_sha256']
    loader['dataset']['ann_file'] = str(annotation)
    loader['dataset']['img_folder'] = str(annotation.parent.parent / 'images')
baseline = copy.deepcopy(common)
full = copy.deepcopy(common)
baseline['DEIM']['backbone'] = 'HGNetv2'
baseline['HGNetv2'] = copy.deepcopy(args['N0']['yaml_cfg']['HGNetv2'])
full['DEIM']['backbone'] = 'HGNetv2_MSCA_EnhancedStem'
full['HGNetv2_MSCA_EnhancedStem'] = copy.deepcopy(args['N4']['yaml_cfg']['HGNetv2_MSCA_EnhancedStem'])
baseline['output_dir'] = './outputs/fair_pair_20261103/BASE'
full['output_dir'] = './outputs/fair_pair_20261103/FULL'
for config in [baseline, full]:
    assert config['train_dataloader']['total_batch_size'] == config['val_dataloader']['total_batch_size'] == 4
    assert config['resume'] is None and config['tuning'] is None and config['seed'] == 0
    assert config[config['DEIM']['backbone']]['pretrained'] is False
    assert config['epoches'] == 320 and config['use_amp'] is False
config_hashes = {name: writej(name, value) for name, value in [('baseline.proposed.yml', baseline), ('full.proposed.yml', full)]}
differences = {k: {'BASE': flat(baseline).get(k), 'FULL': flat(full).get(k)} for k in sorted(flat(baseline).keys() | flat(full).keys())
               if flat(baseline).get(k) != flat(full).get(k)}
assert all(k == 'DEIM.backbone' or k == 'output_dir' or k.startswith('HGNetv2') for k in differences)
writej('configuration_difference.json', differences)

code_paths = ['engine/solver/det_solver.py', 'engine/solver/_solver.py', 'engine/optim/ema.py',
              'engine/optim/lr_scheduler.py', 'engine/data/transforms/container.py', 'engine/data/dataloader.py']
source_evidence = []
for rel in code_paths:
    path = CODE / rel
    text = path.read_text(encoding='utf-8-sig')
    ast.parse(text)
    source_evidence.append({'path': str(path), 'sha256': sha(path)})
writej('current_training_semantics.json', {
    'scope': 'Read-only current-source analysis; not a claim that the historical runs used these exact source bytes.',
    'sources': source_evidence,
    'findings': [
        {'source': 'engine/solver/det_solver.py', 'lines': [79, 82],
         'finding': 'At loop epoch 296 the current solver restores best_stg1 and assigns ema.decay.'},
        {'source': 'engine/solver/det_solver.py', 'lines': [148, 162],
         'finding': 'At a late epoch without a new global best AP, it restores best_stg1 again after validation. This can make post-branch state differ from the just-evaluated state.'},
        {'source': 'engine/solver/_solver.py', 'lines': [166, 176],
         'finding': 'Resume restores last_epoch and stateful attributes present in the checkpoint, including the optimizer and EMA when recorded. Loop epoch and restored last_epoch must be logged separately.'},
        {'source': 'engine/solver/det_solver.py', 'lines': [112, 119],
         'finding': 'last.pth is written only while loop epoch is below stop_epoch; it is not a reliable terminal-epoch artifact in this solver.'},
        {'source': 'engine/optim/ema.py', 'lines': [43, 65],
         'finding': 'decay_fn captures the constructor argument decay, not self.decay. Later assignments to self.decay do not change that captured multiplier; restoring EMA updates can still change its warmup factor.'},
        {'source': 'engine/optim/lr_scheduler.py', 'lines': [25, 43],
         'finding': 'With the retained lr_gamma=1.0, min_lr equals init_lr. The selected flatcosine function warms up quadratically for 2000 iterations and is constant afterward. The inactive MultiStepLR declaration is not the effective decay schedule.'}
    ],
    'effect_on_manuscript': 'Current prose describes configured rates and an available-solver restoration limitation; it does not claim cosine LR decay or an effective adaptive EMA decay. Keep historical identity caveat.',
    'effect_on_proposal': 'Preserve the same effective solver behavior in both arms; instrument evaluation BEFORE any rollback. Do not silently fix training algorithms during the comparison.',
    'model_execution': False
})

protocol = {
    'status': 'PROPOSED_NOT_AUTHORIZED_NOT_RUN', 'date': '2026-11-03',
    'question': 'Does the complete n-scale MPSMA combination improve validation detection over the original n-scale host under one shared training/evaluation protocol?',
    'minimum_groups': [{'id': 'BASE', 'backbone': 'HGNetv2 B0 with original stem, ReLU and original se-path aggregation'},
                       {'id': 'FULL', 'backbone': 'HGNetv2 B0 with enhanced stem, SiLU and hierarchical ESE/SimAM/MSCA gates'}],
    'scope_limit': 'Only total-combination evidence. This does not isolate stem, attention, activation or aggregation effects and does not establish multi-seed stability.',
    'why_two_fresh_runs': 'The historical N0/N4 batch sizes differ, and exact historical source/data/runtime binding is missing. CPU re-evaluation does not alter their training trajectories. Reusing old N4 as if it were trained under this frozen proposal would be unjustified.',
    'configuration_files_sha256': config_hashes, 'single_shared_recipe_source': source_records[4],
    'budget': {'groups': 2, 'seeds': [0], 'epochs_per_group': 320, 'epoch_indices': [0, 319],
               'batch': 4, 'world_size': 1, 'precision': 'FP32, AMP disabled',
               'training_images': 6471, 'drop_last': True, 'batches_per_epoch': 1617,
               'optimizer_step_attempts_per_group': 517440, 'validation_images_per_epoch': 548,
               'stop_on_nonfinite_or_error': True, 'automatic_retry_or_extension': False},
    'data': {split: {k: dataset_audit[split][k] for k in ['annotation_path', 'annotation_sha256', 'images', 'annotations']} for split in ['train', 'val']},
    'evaluation': {'name': 'Local converted COCO-style bbox protocol, not official VisDrone',
                   'max_dets': [1, 10, 100], 'postprocessor_topk': 300, 'val_batch': 4, 'input': [640, 640],
                   'weight_source': 'EMA', 'primary_endpoint': 'Actual loop epoch 319, evaluated EMA state BEFORE any late-stage rollback',
                   'primary_metrics': ['AP', 'APs'],
                   'secondary_endpoint': 'Highest AP among all 320 recorded epochs; earliest epoch wins a tie; retain all 12 metrics from that one row',
                   'no_independent_test_claim': True},
    'initialization_and_randomness': [
        'Both arms start fresh without pretrained, tuning or historical checkpoint resume.',
        'Before training, create an explicit semantically matched tensor map: shared encoder/decoder and unchanged backbone blocks only. Check keys, shapes and tensor hashes; copy the common random initialization for those roles. Do not copy stem/aggregation/gate tensors merely because shapes happen to agree.',
        'Record deterministic initialization for arm-specific modules, complete initial-state hashes and the mapping manifest. This has not yet been instantiated or runtime-validated.',
        'Reset RNG after model construction and use identical explicit sampler/worker seeds; verify image IDs, transformed inputs and targets for the first 100 training batches before accepting a matched-data claim. An unrelated import/profiling call must not perturb one arm only.'
    ],
    'required_instrumentation_before_execution': [
        'Freeze actual training source/dependencies/device and all training/validation image hashes; proposed local paths must be remapped and reverified if another device is authorized.',
        'Preserve current solver/augmentation/EMA behavior equally in both arms, including late rollback. Exclude incidental FLOP profiling from both runs.',
        'Record actual loop epoch, monotonic optimizer-step attempts, restored last_epoch, restore source hash and executed LR for each parameter group.',
        'At validation, bind EMA tensor hash, configuration/source/data/environment hashes and all 12 metrics BEFORE conditional restoring of best_stg1. Keep actual epoch-319 and selected-best checkpoint files, predictions and evaluation arrays.',
        'No automatic restart, resume from unrelated experiments, tuning or extra seeds. Stop and report a failure without consuming an unapproved replacement run.'
    ],
    'resource_status': {'device': 'Not selected or authorized', 'remote_access': False, 'paid_compute': False,
                        'training_runtime_verified': False, 'duration_estimate': None,
                        'note': 'CPU evaluation duration is not a valid estimate of full training time. No GPU/remote probing was performed for this proposal.'},
    'acceptance': [
        'Both arms complete their fixed budget with matched declared conditions and validated common initialization/data controls.',
        'Report the signed FULL-minus-BASE AP/APs differences at the preselected endpoints, including negative or mixed findings. No post-hoc endpoint switching or extra tuning.',
        'A positive single-seed difference is evidence about this pair only, not statistical significance or isolated component novelty.',
        'These results could address the fair primary comparison; historical reproducibility and accuracy/timing linkage remain separate limitations.'
    ],
    'execution_authorization': None, 'launcher_created': False,
    'not_ready_to_launch': 'Runtime location, common-initialization mapping and instrumentation must be completed and validated within a separately authorized experiment scope.'
}
training_manifest = OUT / 'training_image_manifest.json'
if training_manifest.exists():
    manifest = readj(training_manifest)
    assert manifest['count'] == 6471 and manifest['annotation_sha256'] == protocol['data']['train']['annotation_sha256']
    protocol['data']['train']['current_image_manifest'] = str(training_manifest)
    protocol['data']['train']['current_image_manifest_sha256'] = sha(training_manifest)
    protocol['data']['val']['current_image_manifest'] = str(CPU / 'validation_image_manifest.json')
    protocol['data']['val']['current_image_manifest_sha256'] = sha(CPU / 'validation_image_manifest.json')
writej('protocol.json', protocol)
summary = '''公平主对照：最小补实验方案（2026-11-03）

状态：仅完成现有材料核查、配置准备和协议整理。未获得新训练授权；未导入模型、加载权重、运行前向/训练、访问远程或新增付费资源。此前授权只覆盖已完成的三份权重CPU重评。

要回答的问题
完整n配置在相同训练条件下，是否比原始n宿主基线更好？本方案只检验完整组合，不检验MPS、MSCA或SiLU的独立收益。

为什么现有材料不足
历史N0为batch16，N4为batch4；CPU重评不会改变这些权重的训练过程。N1/N2/N4部分声明训练字段相同，但它们都不是原始宿主基线，且比较同时改变多个结构因素。N3另有恢复自身last的记录。历史训练源码/环境的不可变关联仍缺失，不能默认保留N4、只补训一条基线就完成公平对照。

最少两组
BASE：原始HGNetv2 B0宿主（原stem、ReLU、原se双卷积聚合）。
FULL：完整MPSMA n配置（增强stem、SiLU、分阶段注意力）。
两组同seed0、train/val batch4、FP32、640输入、320轮、AdamW、相同增强、学习率规则及EMA。均从新初始化开始，无旧权重resume或tuning。
6471张训练图、drop_last=true，按单设备每轮1617批计算，每组最多517440次更新尝试。每轮固定548张验证图；不加seed、不调参、不自动延长或重启。
共享部分按显式语义映射使用可核验的共同初始化；新增结构使用固定规则。模型构造后的随机流和数据顺序也要核对，不能只凭seed相同就声称训练输入相同。

评价与验收
采用已明确命名的本地COCO-style协议，maxDets=[1,10,100]；不冒称官方VisDrone结果。
主端点固定为实际循环epoch319、条件回滚前评价的EMA，报告AP和APs。辅助表报告320轮中最高AP的同一完整12指标行；同值取较早轮，不逐指标挑峰值。
如结果为负或有取舍，也如实报告；两组单seed不能支持统计显著性或模块独立归因。

本次发现的实施注意点
当前solver在296轮恢复best_stg1，后期未刷新全局最佳AP时还会再次恢复；last.pth只在296前写入。必须在每次评价时、任何回滚前保存实际权重/指标的关联，不能再仅凭last_epoch或文件名判断归属。
当前lr_gamma=1.0使flatcosine的最低LR等于初始LR，2000步二次预热后实际不衰减；模型EMA的decay_fn捕获构造参数，后续改self.decay不会直接改闭包系数。两组必须保留同一有效规则，不能只修一组，也不能把当前源码行为当成已证实的历史行为。

算力与授权边界
未选择训练设备；不由CPU推理时长估算完整训练耗时。没有访问或探测远程设备，也没有授权租用资源。需要用户决定是否安排这两组训练，并提供允许使用的GPU/设备范围；届时才完善实际运行环境与记录代码。当前配置是可审阅提案，没有启动器，也未做运行时兼容性验证。
仅同意这两组训练不会授权其他消融、额外seed、官方协议重评或测速；这些仍另行说明。

文件
baseline.proposed.yml、full.proposed.yml：两个完整展开配置，以JSON语法保存（YAML的子集）。
configuration_difference.json：两组差异，仅模型结构及输出位置。
existing_pair_audit.json：历史配置对的选定训练字段比较及限制。
current_training_semantics.json：本次静态代码证据和行号。
protocol.json：预算、端点、初始化、数据、记录要求及未授权状态。

论文当前状态：继续使用第三轮英文修订稿；本方案不是实验结果，未据此更改论文数字或宣称可以投稿。
'''
if training_manifest.exists():
    summary += '\n补充：已只读核对本机6471张训练图片，并保存逐文件SHA-256和大小清单 training_image_manifest.json。图片总计1547923454字节；未解码或用于训练。这证明当前文件可读取，不证明历史训练时使用的字节相同。验证集使用已完成核验的548图清单。\n'
(OUT / '公平主对照方案.txt').write_text(summary, encoding='utf-8')
checks = {'status': 'STATIC_PROPOSAL_CHECKS_PASSED_NOT_RUN', 'proposal_only': True,
          'historical_N1_N4_selected_training_fields_equal': all(training_views['N4'].get(k) == training_views['N1'].get(k) for k in training_views['N4']),
          'paired_config_differences_only_architecture_and_output': True,
          'no_experiment_or_model_execution': True,
          'config_sha256': config_hashes, 'protocol_sha256': sha(OUT / 'protocol.json'),
          'manuscript_sha256_unchanged': sha(PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx')}
writej('preparation_check.json', checks)
print(json.dumps({'status': checks['status'], 'proposal': str(OUT / '公平主对照方案.txt'),
                  'historical_pairs_checked': len(pair_checks), 'config_diff_paths': len(differences)}, ensure_ascii=False))
