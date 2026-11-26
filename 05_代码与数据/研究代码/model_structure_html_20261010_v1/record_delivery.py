"""Archive the verified standalone diagram and preserve the active research state."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/model_structure_html_20261010_v1'
STATE = ROOT / 'outputs/sod_roi12_20261009/execution_state.json'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

coverage = read(OUT / 'diagram_coverage.json')
browser = read(OUT / 'browser_verification.json')
complete_path = OUT / 'collection_complete/captured_complete_graph_v3/completion.json'
complete = read(complete_path)
receipt = read(OUT / 'download_complete_receipt.json')
assert browser['status'] == 'PASS_STANDALONE_HTML_RENDER_AND_INTERACTIONS'
assert complete['status'] == 'PASS_NATIVE_OPERATION_GRAPHS_REQUESTED_MODES'
assert receipt['archive_verified'] and receipt['immutable_files_verified'] == 10
for mode in ('train', 'eval'):
    assert coverage[mode]['all_native_nodes_retained']
    assert coverage[mode]['all_leaf_invocations_represented']
    assert browser[mode]['nodes'] == coverage[mode]['display_nodes']
assert sha(Path(__file__).parent / 'capture_native_graph.py') == complete['code_sha256']

artifact = OUT / 'deim-dfine-s-full-network.html'
now = datetime.now(timezone.utc).isoformat()
delivery = {
    'status': 'COMPLETE_UNMERGED_STANDALONE_HTML_VERIFIED',
    'observed_utc': now,
    'artifact': str(artifact),
    'artifact_bytes': artifact.stat().st_size,
    'artifact_sha256': sha(artifact),
    'model': 'Actual native DEIM-D-FINE-S / HGNetv2-B0; BASE132 seed1 EMA',
    'checkpoint_sha256': complete['checkpoint_sha256'],
    'capture_code_sha256': complete['code_sha256'],
    'scope': 'All registered modules listed; every executed tensor operation and leaf invocation represented separately for native model.forward eval/train fixture at batch1 640x640. Loss criterion, optimizer and backward are outside this model-forward diagram.',
    'coverage': coverage,
    'browser_verification': browser,
    'native_graph_completion': str(complete_path),
    'download_receipt': str(OUT / 'download_complete_receipt.json'),
    'authoritative_v3_model_forward_calls': 4,
    'total_engineering_model_forward_calls_across_v1_v2_v3': 10,
    'training_epochs_added': 0,
    'optimizer_steps': 0,
    'native_model_outputs_equal': all(x['output_equal'] for x in complete['results']),
    'native_model_parameters_unchanged': complete['model_parameters_unchanged'],
    'private_train_BN_updates_equal': complete['private_train_BN_updates_equal'],
    'native_sources_unchanged': complete['native_sources_unchanged'],
    'attention': 'Fused SDPA mathematical steps explicitly marked; not separately dispatched native kernels. Original native operator IDs retained.',
    'no_ops': 'Explicit metadata-only nodes for executed Identity and p=0 Dropout leaf calls; no tensor computation changes.',
    'preserved_engineering_failures': [
        'collection: v1 train graph JSON serialization rejected native attention-mask -inf after native output parity passed; partial file and failure retained. v2 serializer repair preserves nonfinite scalar as metadata.',
        'browser_check_failure.json: initial native-source button absent for builtin Conv2d; UI now explicitly resolves the closest native parent class. Subsequent browser verification passed.'
    ],
    'root_cause_identified': False,
    'method_AP_evidence': False,
    'assignment_diagnostic_status': 'Prepared and deployed, not launched; diagram request took priority.',
    'overall_research_goal': 'active; diagram subtask complete'
}
write(OUT / 'delivery.json', delivery)

readme = '''完整网络结构图交付说明

打开 deim-dfine-s-full-network.html 即可使用；HTML 为离线单文件，不需要联网或安装库。
图对应当前真实原生 DEIM-D-FINE-S / HGNetv2-B0，BASE132 seed1 EMA，输入 batch=1、640×640。

展开范围
683个注册模块全部逐项列出。训练路径1719个节点、1926条连接；推理路径1326个节点、1583条连接。
原生张量操作、卷积/归一化/激活、残差加法、拼接、上采样、TopK、查询构造、注意力采样及各预测头均保留。
所有重复的实际调用独立绘制，不使用“×N”概括，也没有折叠模块。
训练图含去噪查询和辅助预测头；此固定样本有81个GT，原生前向产生300个普通查询与162个去噪查询。
这里的“训练路径”指 model.train() 下的 model.forward；损失函数、优化器和反向传播不属于本图范围。
注册但没有独立 forward 调用的模块仍列在左侧；“未独立调用”不能解释成“参数未使用”。

使用方式
鼠标拖动平移、滚轮缩放。顶部“跳转”可定位Stem、P3/P4/P5、FPN、PAN与三层Decoder。
左侧683个模块逐项可点击，搜索可按路径或算子定位；搜索仅高亮，不会隐藏其余节点。
点击节点后，右侧显示完整模块路径、所有输出张量的尺寸、实际调用参数、输入/输出连接及原生源码入口。
源码入口遇到PyTorch内置类时明确展示最近的原生父模块源码，不能将父模块源码误称为该内置算子的实现。
“全图”显示整体走线；因为所有张量操作逐项展开，整体会很小，阅读时请使用缩放、跳转和搜索。
“导出SVG”导出当前完整训练/推理图，保留全部节点，不只导出当前视口。

图中两类特殊节点
Identity和p=0 Dropout确实被调用，但没有原生张量计算；以恒等节点显式保留。
融合SDPA注意力的QKᵀ、缩放、mask、softmax和AV等步骤以虚线数学展开，区别于实际独立执行的原生算子。
这些展示节点没有更改模型运算，也没有用另一种注意力替换当前模型。

核验结果
远端采集完成、10个不可变下载文件SHA256核验通过。
训练459次、推理450次实际叶子调用均有图节点；所有原生节点保留。
无记录与有记录前向输出完全一致；参数、随机状态及原生源码保持一致，训练分支的私有BN更新逐项一致并恢复。
浏览器检查训练/推理节点数量、683模块清单、搜索、点击、源码、缩放、完整SVG导出及375px移动界面均通过，无JS错误。
工程采集v1/v2/v3累计10次前向，没有优化器更新，新增训练轮数为0。

保留的问题与修复
v1训练图的JSON序列化遇到原生attention mask的负无穷值，前向一致核验已经通过；失败与部分文件保留。
v2仅修复非有限标量的元数据编码，v3再补齐没有aten算子的恒等叶子调用。
浏览器初次检查发现内置Conv2d没有独立原生源码按钮；现在明确引用最近的原生父模块，随后检查通过。

证据文件
diagram_coverage.json：各路径覆盖统计。
browser_verification.json：浏览器操作核验。
download_complete_receipt.json：下载归档核验。
collection_complete/captured_complete_graph_v3/：原生运行图、配置、源码快照、完成回执。
delivery.json：交付身份、SHA256、范围与已保留失败记录。

这张图用于沿实际连接定位模块与张量，不能单独证明小目标错误的根因或某个改进的AP收益。
原有研究目标仍处于进行中，已关闭的旧实验路线不恢复，新的匹配诊断尚未启动。
'''
(OUT / '使用说明与核验.txt').write_text(readme, encoding='utf-8')

state = read(STATE)
backup = STATE.with_name('execution_state_before_network_structure_html_delivery.json')
if not backup.exists():
    backup.write_bytes(STATE.read_bytes())
state['status'] = 'PROBLEM_DRIVEN_RESEARCH_ACTIVE_FULL_NETWORK_HTML_DELIVERED'
state['goal_status'] = 'active'
state['latest_user_steering'] = '通过HTML完整画出当前网络结构，不合并重复结构。'
state['network_structure_html'] = delivery
history = state.setdefault('research_iteration_history', [])
assert isinstance(history, list)
history.append({
    'utc': now,
    'status': 'COMPLETE_HTML_SUBTASK_RESEARCH_GOAL_ACTIVE',
    'evidence': str(OUT / 'delivery.json'),
    'detector_epochs_added': 0,
    'optimizer_steps': 0,
    'root_cause_identified': False,
    'next': 'User can inspect full HTML. Resume problem-mechanism diagnosis from preserved native supervision evidence; assignment panel prepared, not launched. No old closed-route revival.'
})
write(STATE, state)
with STATE.with_name('continuation_state.txt').open('a', encoding='utf-8') as stream:
    stream.write(f'\n\n{now} PROGRESS: User requested complete unmerged HTML network structure. Delivered offline {artifact}; 683 registered modules, train1719/eval1326 display nodes, all native nodes and executed leaf calls retained. Native code/outputs/parameters parity, downloaded10 immutable files verified, browser search/click/source/zoom/mode/fullSVG/mobile passed. Engineering capture v1/v2/v3 total10 forwards, 0 optimizer steps/epochs; serializer and initial source-button failures preserved and repaired. Diagram subtask complete, overall goal active, no root/AP claim. Assignment diagnostic prepared/deployed not launched, no old-route restart.\n')
print(json.dumps({'status': delivery['status'], 'html_bytes': delivery['artifact_bytes'], 'html_sha256': delivery['artifact_sha256'], 'overall_goal': 'active'}, ensure_ascii=False))
