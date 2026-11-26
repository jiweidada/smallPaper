from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[3]
out = root / 'outputs/model_structure_html_20261010_v1'
code = Path(__file__).resolve().parent
backup = out / 'ui_before_tree_v1'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def embedded(p):
    html = p.read_text(encoding='utf-8')
    return json.loads(re.search(r'<script type="application/json" id="network-data">(.*?)</script>', html, re.S).group(1))

artifact = out / 'deim-dfine-s-full-network.html'
data = read(out / 'diagram_data_laid_out.json')
assert embedded(backup / artifact.name) == data == embedded(artifact), 'Native graph data changed'
browser = read(out / 'browser_verification.json')
assert browser['status'] == 'PASS_FOLDABLE_GRAPH_AND_MODULE_TREE' and not browser['errors']
assert browser['allTrain']['nodes'] == 1719 and browser['allEval']['nodes'] == 1326
exports = {}
for filename, expected in [('verified-export-current-train.svg', 7), ('verified-export-train.svg', 1719), ('verified-export-eval.svg', 1326)]:
    p = out / filename
    xml = ET.parse(p).getroot()
    count = sum('data-node' in n.attrib for n in xml.iter())
    assert count == expected
    exports[filename] = {'nodes': count, 'xml_valid': True, 'sha256': sha(p)}

now = datetime.now(timezone.utc).isoformat()
delivery = read(out / 'delivery.json')
state_path = root / 'outputs/sod_roi12_20261009/execution_state.json'
if not (backup / 'execution_state_before_ui_revision.json').exists():
    shutil.copy2(state_path, backup / 'execution_state_before_ui_revision.json')
if not (backup / '使用说明与核验.txt').exists():
    shutil.copy2(out / '使用说明与核验.txt', backup / '使用说明与核验.txt')
receipt = {
    'status': 'COMPLETE_FOLDABLE_SINGLE_FILE_HTML_VERIFIED',
    'observed_utc': now,
    'user_request': 'Large structures collapsed by default, expand using selected-node right pane or node button; left module inventory becomes a collapsible folder tree.',
    'artifact': str(artifact), 'artifact_bytes': artifact.stat().st_size,
    'previous_artifact_sha256': sha(backup / artifact.name), 'artifact_sha256': sha(artifact),
    'native_embedded_data_exactly_equal_to_previous_artifact': True,
    'native_data_file_sha256': sha(out / 'diagram_data_laid_out.json'),
    'default_display_nodes': {'train': 7, 'eval': 4},
    'complete_display_nodes': {'train': 1719, 'eval': 1326},
    'registered_modules_preserved': 683,
    'folded_edge_accounting': 'Every full edge is either internal to a folded representative or included in a displayed aggregated edge. Every full node maps to a visible node. Tested at 10 different view states.',
    'offline_runtime': 'Pinned vendored dagre1.1.5 embedded into standalone HTML; no requests or external dependencies required.',
    'vendor_sha256': sha(code / 'vendor/dagre.min.js'),
    'browser_verification': browser, 'verified_svg_exports': exports,
    'source_files': {name: sha(code / name) for name in ['network-template.html', 'network-style.css', 'network-browser.js', 'build_html.cjs', 'render_from_layout.cjs', 'layout_graphs.js', 'check_folded.cjs']},
    'previous_ui_and_receipts_preserved': str(backup),
    'model_computation_changed': False, 'model_forward_calls_added': 0,
    'training_epochs_added': 0, 'optimizer_steps': 0,
    'overall_research_goal_status': 'active'
}
write(out / 'foldable_tree_revision.json', receipt)
delivery.update(status=receipt['status'], observed_utc=now, artifact_bytes=receipt['artifact_bytes'], artifact_sha256=receipt['artifact_sha256'], browser_verification=browser, foldable_tree_revision=str(out / 'foldable_tree_revision.json'))
delivery['scope'] = 'All native model.forward nodes, repeated calls and registered modules retained. Large structures and direct-operation groups collapse by default; recursive expansion and full SVG export restore every individual node. Loss, backward and optimizer are outside the model-forward diagram.'
delivery['assignment_diagnostic_status'] = 'Completed and independently verified in assignment_mechanism_diagnostic_20261010_v1; this UI revision performs no model work.'
write(out / 'delivery.json', delivery)
state = read(state_path)
state['network_structure_html'] = delivery
state['latest_user_steering'] = '用户要求网络图默认收起大结构、右侧按钮展开，左侧改为可折叠模块树；已完成并通过浏览器核验。'
state['research_iteration_history'].append({'utc': now, 'status': receipt['status'], 'evidence': str(out / 'foldable_tree_revision.json'), 'detector_epochs_added': 0, 'optimizer_steps': 0})
state['updated_utc'] = now
write(state_path, state)
with state_path.with_name('continuation_state.txt').open('a', encoding='utf-8') as stream:
    stream.write(f'\n{now} FOLDABLE NETWORK UI COMPLETE: Default train7/eval4 visible nodes; right-pane/node expansion, independent VS Code-style module tree, search ancestor reveal, current/full SVG exports pass. Full train1719/eval1326, all683 modules and embedded native data unchanged. HTML SHA{receipt["artifact_sha256"]}. No remote/model/optimizer/training activity added. Overall research goal remains active; previously verified assignment diagnostic and next post-Mosaic/MixUp coverage inquiry preserved.\n')
(out / '使用说明与核验.txt').write_text('''可折叠网络结构图使用说明

打开 deim-dfine-s-full-network.html；如果浏览器已经打开，按 F5 刷新。
这是离线单文件，无需联网。直接打开 code 中的 network-template.html 会跳转到完整文件。

查看方式
1. 初始只显示主要结构。训练默认7个节点，推理默认4个节点。
2. 点击一个大结构，右侧出现“展开内部结构”。节点右侧也有“展开”按钮。
3. 每次展开一层，下一层结构仍收起。大量本级张量操作作为独立折叠项保留。
4. 右侧可以收起当前结构或上级结构。顶部“全部展开”和“收起大结构”控制整张图。
5. 左侧683个模块按真实父子关系组织成文件夹树；箭头只控制目录，点击名称查看与定位。
6. 搜索模块或算子，按回车或“下一处”，会自动展开目标所在路径并定位到实际算子。
7. 鼠标拖动平移，滚轮缩放；“适应当前图”显示当前展开范围。“输入端”定位图像输入。
8. “导出当前SVG”保留当前折叠状态；“导出完整SVG”保留当前训练/推理路径的所有节点。

模型与覆盖
真实原生 DEIM-D-FINE-S / HGNetv2-B0，BASE132 seed1 EMA，batch1、640×640。
训练完整1719个节点/1926条连接；推理完整1326个节点/1583条连接；683个注册模块均保留。
所有原生节点、重复调用、恒等模块及融合注意力的数学展示步骤均未删除。
折叠只改变显示：跨结构多条连接会合并显示，每条原始连接都有归属，展开后恢复。
节点详情显示完整路径、已记录张量尺寸、相邻连接和原生源码；内置模块可明确查看最近的原生父模块源码。
训练路径指 model.train() 下的 model.forward，包含去噪与辅助头；损失、反向传播和优化器不在本图内。
注意力内部的虚线数学展示不同于实际独立执行的算子。

核验
browser_verification.json：默认折叠、逐层展开、目录独立折叠、深层跳转、隐藏节点搜索、源码、完整展开、两种SVG导出与移动界面均通过。
foldable_tree_revision.json：本次交互修改的身份与核验。完整嵌入数据与上一版逐项相同，未新增模型前向、训练或优化器操作。
diagram_coverage.json 与 collection_complete/captured_complete_graph_v3/：原始覆盖和采集证据。
ui_before_tree_v1/：上一版页面、说明和核验回执保留。

图用于查看真实结构，不能单独证明小目标错误根因或AP收益。研究目标仍进行中；既有双seed匹配诊断已完成核验。
''', encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'artifact_sha256': receipt['artifact_sha256'], 'native_data_unchanged': True, 'goal_status': state['goal_status']}))
