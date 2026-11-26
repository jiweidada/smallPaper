"""Cross-check the current manuscript tables against retained source records.

Read-only analysis of existing artifacts: no training, evaluation, or manuscript edits.
"""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from docx import Document
import json
import re

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PAPER = REV.parent
CPU = PAPER / '本地CPU核验_20261103'
REMOTE = PAPER / '远程公平对照_20261103'
MANUSCRIPT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
EXPECTED_DOC = 'a494f6b7ca2b1c200232ab49fa7f3d3831b161e2552990da3b88109f7b599de2'
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cells(row):
    return [' '.join(cell.text.split()) for cell in row.cells]


def locate(path, needle):
    lines = path.read_text(encoding='utf-8').splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if needle in line]
    assert len(hits) == 1, (path, needle, hits)
    return hits[0]


assert digest(MANUSCRIPT) == EXPECTED_DOC, 'The manuscript changed; update and review this index first.'
doc = Document(MANUSCRIPT)
assert len(doc.tables) == 6
source_manifest = {Path(x['snapshot']).resolve(): x['sha256'] for x in read_json(REV / 'source_manifest.json')}
audited = read_json(REV / 'run_audit.json')
run_ids = ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']
sources = {}
runs = {}
for run_id in run_ids:
    folder = REV / 'source_snapshot' / run_id
    log, args = folder / 'log.txt', folder / 'args.json'
    assert digest(log) == source_manifest[log.resolve()]
    assert digest(args) == source_manifest[args.resolve()]
    lines = log.read_text(encoding='utf-8').splitlines()
    rows = [(i + 1, json.loads(line)) for i, line in enumerate(lines) if line.strip()]
    assert len(rows) == 320
    best_line, best = max(rows, key=lambda pair: pair[1]['test_coco_eval_bbox'][0])
    info = audited[run_id]
    assert best_line == info['best_log_line'] and best['epoch'] == info['best_epoch']
    assert best['n_parameters'] == info['n_parameters']
    assert dict(zip(METRICS, best['test_coco_eval_bbox'])) == info['metrics']
    runs[run_id] = {'log': str(log), 'log_sha256': digest(log), 'args': str(args), 'args_sha256': digest(args),
                    'best_line_1based': best_line, 'best_epoch_0based': best['epoch'],
                    'metrics': info['metrics'], 'train_batch': info['train_batch'],
                    'val_batch': info['val_batch'], 'trainable_parameters': info['n_parameters']}

# Table 1: only directly check ID, train batch, and logged parameter count.
for index, run_id in enumerate(run_ids[:5], 1):
    row = cells(doc.tables[0].rows[index]); run = runs[run_id]
    assert row[0] == run_id and int(row[4]) == run['train_batch']
    assert row[5] == f"{run['trainable_parameters'] / 1e6:.4f}"

# Table 2: each complete displayed row comes from the same best-AP log row.
for index, run_id in enumerate(run_ids[:5], 1):
    row = cells(doc.tables[1].rows[index]); run = runs[run_id]
    expected = [run_id, str(run['best_epoch_0based'])] + [f"{run['metrics'][m]:.4f}" for m in ['AP', 'AP50', 'AP75', 'APs', 'AR100']]
    assert row == expected, (run_id, row, expected)

# Table 3: N4 and S1 are scale-specific records, not a matched scale ablation.
for index, (label, run_id) in enumerate([('n', 'N4'), ('s', 'S1')], 1):
    row = cells(doc.tables[2].rows[index]); run = runs[run_id]
    expected = [label, str(run['best_epoch_0based'])] + [f"{run['metrics'][m]:.4f}" for m in ['AP', 'AP50', 'AP75', 'APs']]
    expected += [f"{run['trainable_parameters'] / 1e6:.4f}"]
    assert row == expected, (run_id, row, expected)

# Tables 4 and 5: retain their original static-benchmark limitations.
bench = REV / 'source_snapshot/archived_benchmark'
summary = (bench / '零训练数据汇总.md').read_text(encoding='utf-8')
summary_rows = summary.splitlines()
assert any(all(term in line for term in ['DEIM-D-FINE-S', '24.86 G', '25.37 G']) for line in summary_rows)
assert any(all(term in line for term in ['MPSMA-DEIM', '29.63 G', '30.14 G']) for line in summary_rows)
records = [r for r in read_json(bench / 'latency_FP32.json') if r['precision'] == 'FP32']
assert len(records) == 2 and all((r['size'], r['batch'], r['warmup'], r['iters']) == (640, 1, 50, 200) for r in records)
records = {r['tag']: r for r in records}
for i, (doc_label, tag, gflops) in enumerate([('DEIM-D-FINE-S', 'DEIM-D-FINE-S', '24.86'), ('MPSMA-DEIM (s)', 'MPSMA-DEIM', '29.63')], 1):
    r = records[tag]
    assert cells(doc.tables[3].rows[i]) == [doc_label, gflops, f"{r['params_M']:.2f}"]
    assert cells(doc.tables[4].rows[i]) == [doc_label, f"{r['net']['mean_ms']:.3f}", f"{r['e2e']['mean_ms']:.3f}", f"{r['e2e']['fps']:.2f}"]

# Table 6: historic best-log and separate CPU checkpoint evaluations.
cpu_runs = {}
for i, run_id in enumerate(['N0', 'N4', 'S1'], 1):
    folder = CPU / 'results' / run_id
    metric_path, receipt_path = folder / 'metrics.json', folder / 'receipt.json'
    measurement, receipt = read_json(metric_path), read_json(receipt_path)
    historical = runs[run_id]['metrics']
    assert receipt['status'] == 'COMPLETED' and receipt['completed_images'] == 548
    assert measurement['checkpoint_sha256'] == receipt['checkpoint_sha256']
    current = measurement['metrics']
    expected = [run_id, f"{historical['AP']:.6f}", f"{current['AP']:.6f}",
                f"{100 * (current['AP'] - historical['AP']):+.6f}",
                f"{historical['APs']:.6f}", f"{current['APs']:.6f}",
                f"{100 * (current['APs'] - historical['APs']):+.6f}"]
    actual = cells(doc.tables[5].rows[i])
    assert actual == expected, (run_id, actual, expected)
    cpu_runs[run_id] = {'metrics_path': str(metric_path), 'metrics_sha256': digest(metric_path),
                        'receipt_path': str(receipt_path), 'receipt_sha256': digest(receipt_path),
                        'checkpoint_sha256': measurement['checkpoint_sha256']}

backbone = REV / 'source_snapshot/current_code/engine/backbone/hgnetv2_MSCA_EnhancedStem.py'
msca = REV / 'source_snapshot/current_code/engine/extre_module/custom_nn/attention/MSCA.py'
simam = REV / 'source_snapshot/current_code/engine/extre_module/custom_nn/attention/simam.py'
method_sources = [
    {'topic': 'MPS stem and Fig. 2', 'path': str(backbone), 'line': locate(backbone, 'class EnhancedStemBlock('), 'sha256': digest(backbone)},
    {'topic': 'MSCA-Dilated, Fig. 5 and Eqs. 3–5', 'path': str(msca), 'line': locate(msca, 'class MSCA_Dilated('), 'sha256': digest(msca)},
    {'topic': 'MSCA-Lite, Fig. 4 and Eqs. 1–2', 'path': str(msca), 'line': locate(msca, 'class MSCA_Lite('), 'sha256': digest(msca)},
    {'topic': 'stage-2 SimAM implementation', 'path': str(simam), 'line': locate(simam, 'class SimAM('), 'sha256': digest(simam)},
]
other_paths = [REV / 'configuration_differences.json', REV / 'dataset_eval_audit/annotation_audit.json',
               REV / 'dataset_eval_audit/annotation_correspondence.json', REV / 'dataset_eval_audit/saved_eval_metadata.json',
               REV / 'dataset_eval_audit/figure6_source_manifest.json', REV / '引用核查记录.txt', REV / 'citation_archive.json',
               bench / 'latency_FP32.json', bench / 'latency.py', bench / '零训练数据汇总.md',
               CPU / 'comparison_report.json', REMOTE / 'execution_protocol.json']
for path in other_paths:
    assert path.is_file(), path
    sources[str(path)] = digest(path)
references = [p.text for p in doc.paragraphs if re.match(r'^\[\d+\]', p.text)]
assert len(references) == 11 and [int(re.match(r'^\[(\d+)\]', r).group(1)) for r in references] == list(range(1, 12))
report = {'status': 'CURRENT_DRAFT_TRACE_INDEX_NUMERIC_CHECKS_PASSED',
          'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': str(MANUSCRIPT),
          'manuscript_sha256': EXPECTED_DOC, 'tables_checked': [1, 2, 3, 4, 5, 6],
          'table_1_scope': 'ID, training batch and logged parameter count; structural labels remain a code/config review.',
          'historical_runs': runs, 'cpu_reevaluations': cpu_runs, 'method_sources': method_sources,
          'other_source_sha256': sources, 'numbered_references_found': len(references),
          'live_training_ledger_path': str(REMOTE / 'training_execution_ledger.json'),
          'limits': ['Historical logs and current code do not certify the original immutable training environment.',
                     'Historical N0 versus N1–N4 uses different batch sizes and cannot establish isolated effects.',
                     'The archived benchmark does not prove an exact trained-model checkpoint load.',
                     'The BASE/FULL pair is pending; no interim result is used in this index.',
                     'Code-to-figure and citation assessments use the existing targeted review records; numeric table checks are not full independent peer review.']}
(HERE / '主要结论证据索引_第七轮.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

lines = ['主要结论证据索引（第七轮稿件）', f'稿件SHA256：{EXPECTED_DOC}',
         '本索引只校验现有记录，不构成新训练、重评或投稿验收。路径均为本机绝对路径。', '',
         '一、历史配置与日志（表1—3）']
for run_id, r in runs.items():
    lines += [f"{run_id}：最佳AP所在日志第{r['best_line_1based']}行，epoch={r['best_epoch_0based']}，训练/验证batch={r['train_batch']}/{r['val_batch']}，训练参数={r['trainable_parameters']}。",
              f"  日志：{r['log']}（SHA256 {r['log_sha256']}）", f"  配置：{r['args']}（SHA256 {r['args_sha256']}）"]
lines += ['', '表1逐项核对ID、训练batch和参数；表2、表3数值逐格对回各自最佳AP同一行。结构标签的代码与配置依据见 configuration_differences.json。',
          'N0与其余组batch不同，部分变体的聚合路径也不同；历史行不能作为匹配对照或单因素因果归因。', '',
          '二、CPU独立权重重评（表6）']
for run_id, r in cpu_runs.items():
    lines += [f"{run_id}：指标 {r['metrics_path']}；执行回执 {r['receipt_path']}；权重SHA256 {r['checkpoint_sha256']}。"]
lines += ['表6逐格核对原始日志、当前权重评价和百分点差值；这些当前重评不能证明历史版本或12项指标完全一致。', '',
          '三、方法、图和公式（第3节）']
for item in method_sources:
    lines.append(f"{item['topic']}：{item['path']}:{item['line']}（SHA256 {item['sha256']}）")
lines += ['第五轮式(4)偏置修订见第五轮公式核查；第六轮MPS图与正文的LAB路径修订见第六轮结构核查。现存源码与历史运行时源码的不可变绑定仍缺失。', '',
          '四、数据、评价、计时与文献',
          f"数据与评价：{REV / 'dataset_eval_audit'}；图6来源见该目录figure6_source_manifest.json。",
          f"旧复杂度与结构计时（表4—5）：{bench / 'latency_FP32.json'}、{bench / 'latency.py'}、{bench / '零训练数据汇总.md'}。复杂度表为本地旧树归档，时延为服务器包归档；脚本的加载链不能证明准确率与时延来自同一正确加载模型。",
          f"引用核查范围：{REV / '引用核查记录.txt'}；正文11条文献条目均存在，但本索引不把有限范围核查表述为全文通读。", '',
          '五、未完成的主要实证证据',
          f"新BASE/FULL固定预算和最终端点审计尚未完成。协议：{REMOTE / 'execution_protocol.json'}；最新实时账本：{REMOTE / 'training_execution_ledger.json'}。",
          '不把早期AP或历史跨条件排名作为完整组合收益，不据当前状态宣布可投稿。', '',
          f"机器可读明细及源文件哈希：{HERE / '主要结论证据索引_第七轮.json'}"]
(HERE / '主要结论证据索引_第七轮.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'tables_checked': 6, 'historical_runs': len(runs), 'cpu_runs': len(cpu_runs)}, ensure_ascii=False))
