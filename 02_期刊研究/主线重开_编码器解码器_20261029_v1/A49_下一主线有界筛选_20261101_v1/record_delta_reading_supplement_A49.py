"""Append selected-method reading evidence without editing frozen/signed startup files."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[1] / '文献_相关原始论文_20261004'
SRC = HERE.parent / 'A43_ECDet三组蒸馏最小验证_20261031_v1' / 'source' / 'EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23' / 'ecdetseg'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def once(p, x):
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)

acquisition = json.loads((HERE / 'delta_visual_source_acquisition_v1.json').read_text(encoding='utf-8-sig'))
assert len(acquisition) == 3 and all(x['status'] == 'PDF_ACQUIRED_NOT_FULL_READ' for x in acquisition)
for r in acquisition:
    assert sha(Path(r['path'])) == r['sha256']
    assert sha(Path(r['text_path'])) == r['text_sha256']

original = json.loads((HERE / 'source_acquisition_v1.json').read_text(encoding='utf-8-sig'))
gdn = next(x for x in original if x.get('title','').startswith('Gated DeltaNet-2:'))
records = [{
    'title': gdn['title'], 'authors': 'Ali Hatamizadeh; Yejin Choi; Jan Kautz',
    'version': 'arXiv2605.22791v1 technical report', 'url': 'https://arxiv.org/abs/2605.22791v1',
    'reading_scope': 'Previously abstract-only. Now selected preliminary2.1-2.2, method3.1 and opening3.2 read from archived PDF text; PDFp3 eq7-12 visually checked. No full paper, chunkwise-kernel analysis or experiment reproduction.',
    'original_method': 'Channel-wise key-side erase and value-side write gates are independent, with channel-wise decay. Tying both edit gates to one scalar recovers KDA; tying decay further recovers Gated DeltaNet.',
    'our_inference': 'Decoupling erase/write is already the source method, so importing that operation into a detector is a known-method control, not by itself our new operation.',
    'unverified_hypothesis': 'A distinct task-specific spatial operator might be useful for UAV detection, but none is selected or tested here.',
    'path': gdn['path'], 'sha256': gdn['sha256'], 'text_path': gdn['text_path'], 'text_sha256': gdn['text_sha256']
}]
details = [
    ('GDKVM:', 'Rui Wang; Yimu Sun; Jingxing Guo; Huisi Wu; Jing Qin', 'ICCV2025 publisher PDF',
     'Selected3.1-3.3 on PDFp3-4 and text continuation of3.4 onp5. PDFp4 Figure3 and eq3-5 visually checked. No full experimental reading, code audit or reproduction.',
     'Linear key-value temporal memory uses delta correction plus learned update/decay projected from the previous state. KPFF adds spatial feature fusion.',
     'Visual delta memory and selective forgetting are existing operations. Do not equate its notation or state-derived gates with all details of Gated DeltaNet-2; no source-code dimensional equivalence was established.'),
    ('OSA:', 'Rui Wang; Huisi Wu; Jing Qin', 'CVPR2026 publisher PDF',
     'Selected method overview and3.1 on PDFp3-5;3.2 text seen without a full APFE formula/figure audit. PDFp4 Figure4/5 and eq1-5 visually checked. No full experiments or code audit.',
     'OSU formulates state orthogonalization on a Stiefel manifold and describes a normalized finite Newton-Schulz implementation. APFE is a separate feature module.',
     'State-spectrum protection already has a visual prior. The theoretical orthogonality formulation and finite iterative implementation must not be treated as independently verified exact equality. Medical temporal benefits do not establish UAV image-detection benefits.'),
    ('DeltaMIL:', 'Yueting Zhu; Yuehao Song; Shuai Zhang; Wenyu Liu; Xinggang Wang', 'arXiv2512.19331v1,2025-12-22',
     'Original arXiv abstract only; PDF/text acquired, not methods-read.',
     'Abstract proposes gated delta memory integration together with local pattern mixing for whole-slide image analysis.',
     'Generic visual delta-memory plus local feature preservation already has a primary abstract precedent; implementation-level equivalence cannot be claimed from an abstract.')
]
for prefix, authors, version, scope, method, inference in details:
    a = next(x for x in acquisition if x['title'].startswith(prefix))
    records.append({
        'title': a['title'], 'authors': authors, 'version': version,
        'url': a['url'], 'reading_scope': scope,
        'original_method_or_abstract_claim': method, 'our_inference': inference,
        'unverified_hypothesis': 'No UAV effectiveness, native ECDet failure cause or novel task-specific operator demonstrated by this reading.',
        **{k:a[k] for k in ['path','sha256','text_path','text_sha256']}
    })
files = [
    SRC / 'engine/edgecrafter/hybrid_encoder.py',
    SRC / 'configs/ecdet/ecdet_s.yml'
]
host = {
    'commit': 'd8752b75c9beaecfcdd0e5cd207bae3b6946ae23',
    'files_sha256': {str(p):sha(p) for p in files},
    'selected_source_fact': 'HybridEncoder TransformerEncoderLayer uses nn.MultiheadAttention at258 and a whole-map call at276-294/414-430. This inspected path is not a tokenwise delta-state recurrence.',
    'scope': 'Selected HybridEncoder path/config only; not a claim that every backbone or all detector attention paths were audited.',
    'inference': 'A recurrent-state pathology from the LM/medical papers is not automatically a demonstrated failure of this native path. Replacing the path would introduce a different architecture requiring a specific task reason and proper known-method control.'
}
views = {str(HERE/'source_views'/n): sha(HERE/'source_views'/n) for n in [
    'A49_Gated_DeltaNet2_p3.png','A49_GDKVM_ICCV2025_Publisher_p4.png','A49_OSA_CVPR2026_Publisher_p4.png']}
newlogs = sorted((HERE/'search_read_logs').glob('A49_delta*.json'))
assert len(newlogs) == 4
registry = {
    'scope': 'Bounded architecture-reserve method reading during an already running pilot. Additive record; original startup reading snapshots are preserved.',
    'records': records, 'host_source_check': host, 'visual_checks': views,
    'supplementary_search_logs_sha256': {str(p):sha(p) for p in newlogs},
    'search_coverage': 'Six explicit discovery queries and primary entry-point attempts are preserved. This is not exhaustive; a negative search result is not novelty proof.',
    'access_note': 'Web tool returned403 for the two CVF PDF/HTML entry points. Direct primary PDF acquisition succeeded and is hash-archived; selected reading above uses those actual files.',
    'decision': 'A49_A_HELD_NOT_SELECTED_FOR_PILOT',
    'not_a_training_negative': True,
    'old_route_boundary': 'The broad noncausal shared-state route N/A18-A27 and generic encoder route L are retained with their original scopes. No old frozen diagnostic is reclassified as training failure, and no host/threshold-only rerun is authorized by this reading.',
    'new_training_queue': False, 'new_model_forwards': 0, 'new_optimizer_updates': 0,
    'current_training': 'Only frozen A49 B12/M12/Q12 is active; no change to protocol, source, initialization or gates.'
}
once(HERE/'09_架构备选补读与阅读范围_v1.json', registry)
once(LIB/'A49_追加架构候选阅读范围_20261101_v1.json', registry)
note = """A49 架构备选补读结论（新增阅读记录，不覆盖已冻结训练方案）

结论：Delta方向继续保留为已有方法控制/架构灵感，本轮未选出新的UAV操作，不加入训练队列。不是训练失败，也没有否定整个Delta视觉方向。

1. 本次补读
Gated DeltaNet-2从“只读摘要”推进到选定2.1—3.1及3.2开头，PDF第3页式7—12已核对。独立的key侧擦除、value侧写入和退化到KDA等关系均为原方法已有设计，直接复用不属于我们的新操作。
GDKVM已读选定3.1—3.3及3.4正文，PDF第4页图3、式3—5核对；视觉时序key-value记忆、Delta校正、选择遗忘已有先例。未核查代码，不对论文门矩阵符号作实现等价断言。
OSA已读方法概览和3.1及后续说明，PDF第4页图4/5、式1—5核对；状态谱保护及Newton-Schulz正交化已有视觉先例。有限迭代与理论正交投影不在本次视作已独立证明相等。
DeltaMIL只读原始摘要，PDF已存；它已经把门控Delta记忆与局部模式结合用于整幅病理图像分析，不能依据摘要推断全部实现。

2. 对主线选择的意义
上述原文支持“存在新编码机制可作比较”，并不支持“原生ECDet已经因同类状态遗忘而漏检”。
所核查ECDet HybridEncoder路径使用完整特征图的MultiheadAttention，不是沿token递推的Delta状态；不能先换成循环状态，再把该状态自身可能产生的秩问题当作原生检测器问题。
未来若选择这一类架构，需要一项具体、可区分的任务操作和相应简单控制；无需先证明所有原方法无效。当前没有选定该操作，因此不给它新增GPU训练额度。

3. 当前实验
A49已知忽略区监督的B/M/Q各12轮按原协议继续。这里不改变当前假设、门槛或训练队列，也不根据原版早期分数提前淘汰或延长。
主线仍未确认，目标保持active。所有阅读范围、原始入口、PDF/text SHA、选定源码SHA见09_架构备选补读与阅读范围_v1.json。
"""
with (HERE/'10_架构备选补读结论.txt').open('x',encoding='utf-8') as f:
    f.write(note)
print(json.dumps({'status':'PUBLISHED','records':len(records),'newly_archived_papers':3,
                  'delta_reserve_status':registry['decision'],'new_training_queue':False,
                  'model_forwards_added':0,'optimizer_updates_added':0},ensure_ascii=False))
