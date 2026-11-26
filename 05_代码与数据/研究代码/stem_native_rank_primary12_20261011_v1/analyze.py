"""Exact score-order accounting from saved native matches; never rematch predictions.

Retains all10 categories, all10 IoUs, both seeds, native stable ties, and GT joins.
This describes evaluation outputs, not a uniquely identified model defect.
"""
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/stem_native_rank_primary12_20261011_v1'
read = lambda p: json.loads(Path(p).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def median(values):
    return float(np.median(values)) if len(values) else None


def accounting(endpoint, expected, annotation, evaluator_sha):
    assert sha(endpoint / 'manifest.json') == expected['endpoint_manifest_sha256']
    manifest = read(endpoint / 'manifest.json')
    for n, h in manifest['files'].items():
        rel = Path(n); assert not rel.is_absolute() and '..' not in rel.parts and ':' not in n
        assert sha(endpoint / rel) == h, n
    summary = read(endpoint / 'summary.json')
    assert summary['epoch_zero_based'] == 11 and summary['evaluated_weights'] == 'EMA'
    assert summary['identity']['evaluator_binding']['files']['cocoeval.py'] == evaluator_sha
    assert summary['protocol']['category_ids'] == list(range(10)) and summary['protocol']['image_ids'] == list(range(548))
    assert summary['protocol']['max_dets'] == [1, 10, 100] and summary['protocol']['area_ranges'][0] == [0, 1e10]
    with gzip.open(endpoint / 'matching_all_area.json.gz', 'rt', encoding='utf8') as f: records = json.load(f)
    with np.load(endpoint / 'precision_recall.npz', allow_pickle=False) as z:
        native = z['precision'][:, :, :, 0, 2].copy(); native_recall = z['recall'][:, :, 0, 2].copy()
    with np.load(endpoint / 'GT_hits.npz', allow_pickle=False) as z:
        gt_ids = z['annotation_id'].copy(); gt_classes = z['category_id'].copy(); gt_flags = z['hits'].copy()
    gt = {int(a['id']): a for a in annotation['annotations']}; loc = {int(g): i for i, g in enumerate(gt_ids)}
    assert len(gt) == len(loc) == 38759 and set(gt) == set(loc)
    ranks = np.full((38759, 10), -1, np.int64)
    rank_min = np.full_like(ranks, -1); rank_max = np.full_like(ranks, -1)
    actual_scores = np.full(ranks.shape, np.nan, np.float64)
    precision = np.empty_like(native); classes = []; cells = set()
    for row in records:
        key = (row['image_id'], row['category_id']); assert key not in cells; cells.add(key)
        assert row['aRng'] == [0, 1e10] and row['maxDet'] == 100 and len(row['dtScores']) <= 100
    for cid in range(10):
        rows = sorted((r for r in records if r['category_id'] == cid), key=lambda r: r['image_id'])
        npig = sum(int((~np.asarray(r['gtIgnore'], bool)).sum()) for r in rows)
        assert npig == sum(a['category_id'] == cid for a in annotation['annotations'])
        scores = np.concatenate([r['dtScores'] for r in rows]); assert np.isfinite(scores).all()
        order = np.argsort(-scores, kind='mergesort'); ordered_scores = scores[order]
        matches = np.concatenate([np.asarray(r['dtMatches']).reshape(10, -1) for r in rows], 1)[:, order]
        ignored = np.concatenate([np.asarray(r['dtIgnore'], bool).reshape(10, -1) for r in rows], 1)[:, order]
        tps, fps = (matches != 0) & ~ignored, (matches == 0) & ~ignored
        tp = np.cumsum(tps, 1).astype(float); fp = np.cumsum(fps, 1).astype(float)
        rc = tp / npig; raw_pr = tp / (tp + fp + np.spacing(1))
        envelope = np.maximum.accumulate(raw_pr[:, ::-1], 1)[:, ::-1]
        q = np.zeros((10, 101))
        threshold_rows = []
        for ti, threshold in enumerate(summary['protocol']['iou_thresholds']):
            indices = np.searchsorted(rc[ti], summary['protocol']['recall_thresholds'], side='left')
            valid = indices < len(order); q[ti, valid] = envelope[ti, indices[valid]]
            positions = np.flatnonzero(tps[ti]); matched_ids = matches[ti, positions].astype(np.int64)
            assert len(matched_ids) == len(set(matched_ids.tolist()))
            assert all(int(g) in loc and gt[int(g)]['category_id'] == cid for g in matched_ids)
            joined_positions = np.array([loc[int(g)] for g in matched_ids], np.int64)
            ranks[joined_positions, ti] = positions + 1
            actual_scores[joined_positions, ti] = ordered_scores[positions]
            rank_min[joined_positions, ti] = np.searchsorted(-ordered_scores, -ordered_scores[positions], side='left') + 1
            rank_max[joined_positions, ti] = np.searchsorted(-ordered_scores, -ordered_scores[positions], side='right')
            prefix = []
            for n in sorted({n for n in (100, 500, 1000, 2000, 5000, 10000, 20000, len(order)) if 0 < n <= len(order)}):
                prefix.append(dict(rank=n, TP=int(tp[ti, n - 1]), FP=int(fp[ti, n - 1]),
                    ignored=n - int(tp[ti, n - 1] + fp[ti, n - 1]), raw_precision=float(raw_pr[ti, n - 1]),
                    recall=float(rc[ti, n - 1]), native_score_at_prefix=float(ordered_scores[n - 1])))
            threshold_rows.append(dict(IoU=float(threshold), prefixes=prefix, final_TP=int(tp[ti, -1]), final_FP=int(fp[ti, -1]),
                native_interpolated_precision_by_recall={str(r): float(q[ti, int(round(r * 100))]) for r in (0., .1, .2, .3, .4, .5, .75, 1.)}))
        assert np.array_equal(q, native[:, :, cid]), 'Native PR order reconstruction differs'
        assert np.array_equal(rc[:, -1], native_recall[:, cid]), 'Native final recall differs'
        precision[:, :, cid] = q
        classes.append(dict(class_id=cid, name=summary['categories_percent'][cid]['name'], GT=npig,
            predictions=len(order), equal_score_adjacent_pairs=int((np.diff(ordered_scores) == 0).sum()), by_IoU=threshold_rows))
    assert np.array_equal(ranks > 0, gt_flags), 'Native GT-to-score-order join differs'
    assert np.array_equal(precision, native)
    assert ((rank_min[ranks > 0] <= ranks[ranks > 0]) & (ranks[ranks > 0] <= rank_max[ranks > 0])).all()
    return dict(description=dict(classes=classes, native_all_area_PR_cells_reproduced=10100,
            native_PR_max_abs_difference=0., all_native_GT_flags_exact=True, actual_native_stable_ties_preserved=True,
            cross_run_detection_or_query_identity_assumed=False), precision=precision, arrays=dict(annotation_id=gt_ids,
                category_id=gt_classes, native_true_match_score_rank=ranks, score_tie_rank_min=rank_min, score_tie_rank_max=rank_max,
                native_true_match_score=actual_scores, native_TP=gt_flags))


def analyze():
    assert not OUT.exists()
    source_folder = ROOT / 'outputs/stem_AP_primary_breakdown12_20261011_v1'
    source = read(source_folder / 'report.json')
    for n, h in read(source_folder / 'manifest.json')['files'].items(): assert sha(source_folder / n) == h
    assert source['status'] == 'ACTUAL_PRIMARY12_AP_ARITHMETIC_AND_CATEGORY_STAGE_DESCRIPTION_REPLAY_PENDING'
    assert not source['saved_EMA_independent_replay_verified'] and not source['formal_terminal_reader_passed']
    assert sha(ROOT / '英文小论文/code/stem_AP_primary_breakdown12_20261011_v1/analyze.py') == source['source_sha256']
    p = read(ROOT / '英文小论文/code/original_stem12_monitor_20261011_v1/local_reader_protocol.json')
    base = ROOT / p['reused_local_evidence']['1']['packet_relative_path']
    annotation = read(base / 'reference/val_annotations.json')
    assert sha(base / 'reference/val_annotations.json') == source['val_annotation_sha256']
    evaluator = base / 'evaluator/pycocoeval/cocoeval.py'; evaluator_sha = sha(evaluator)
    outputs = {}; comparisons = []
    for seed in (1, 2):
        for arm, folder in (
            ('full', ROOT / f'outputs/stem_primary_collection_20261011_v1/seed{seed}_original_stem_v1'),
            ('RGB', ROOT / f'outputs/stem_component_primary_collection_20261011_v2/seed{seed}_highfreq_rgb_v1')):
            outputs[(seed, arm)] = accounting(folder / 'extracted/run/endpoint_epoch12', source['source_primary_packets'][str(seed)][arm], annotation, evaluator_sha)
        left, right = outputs[(seed, 'RGB')], outputs[(seed, 'full')]
        for k in ('annotation_id', 'category_id'): assert np.array_equal(left['arrays'][k], right['arrays'][k])
        classes = []
        for cid in range(10):
            m = left['arrays']['category_id'] == cid; thresholds = []
            for ti, t in enumerate(np.linspace(.5, .95, 10)):
                lr, rr = left['arrays']['native_true_match_score_rank'][m, ti], right['arrays']['native_true_match_score_rank'][m, ti]
                gained, lost, common = (lr < 0) & (rr > 0), (lr > 0) & (rr < 0), (lr > 0) & (rr > 0)
                thresholds.append(dict(IoU=float(t), gained_TP=int(gained.sum()), lost_TP=int(lost.sum()), common_TP=int(common.sum()),
                    full_gained_TP_median_native_rank=median(rr[gained]), RGB_lost_TP_median_native_rank=median(lr[lost]),
                    common_TP_median_full_minus_RGB_native_rank=median(rr[common] - lr[common]),
                    positive_rank_delta_means_later_within_own_native_class_order=True))
            classes.append(dict(class_id=cid, name=right['description']['classes'][cid]['name'], by_IoU=thresholds,
                native_interpolated_AP_delta_points_by_IoU=(100 * (right['precision'][:, :, cid] - left['precision'][:, :, cid]).mean(1)).tolist()))
        comparisons.append(dict(seed=seed, classes=classes))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    OUT.mkdir()
    for (seed, arm), item in outputs.items():
        write(OUT / f'seed{seed}_{arm}_order.json', item['description'])
        np.savez_compressed(OUT / f'seed{seed}_{arm}_GT_order.npz', **item['arrays'])
    report = dict(status='ALL_FOUR_PRIMARY_NATIVE_SCORE_ORDERS_AND10100_PR_CELLS_REPRODUCED_EXACTLY',
        source_primary_AP_breakdown_sha256=sha(source_folder / 'report.json'), source_sha256=sha(__file__),
        evaluator_source_sha256=evaluator_sha, source_primary_packets=source['source_primary_packets'],
        native_all_area_PR_cells_per_record=10100, total_native_PR_cells_reproduced=40400, maximum_abs_PR_difference=0.,
        comparisons=comparisons, all10_categories_and_all10_IoUs_and_both_seeds_retained=True,
        actual_native_tie_order_preserved=True, native_zero_GT_ID_match_sentinel_preserved=True,
        same_GT_join_only_no_cross_run_query_alignment=True, independent_saved_EMA_verification_pending=True,
        causal_root_proven=False, novelty_proven=False, new_model_forwards=0, new_evaluator_calls=0,
        new_matching_calls=0, new_optimizer_updates=0, new_training_epochs=0,
        scope='Pure accounting of already saved native matches and score order at all-area maxDet100. '
              'No predictions are rematched or rescored; no oracle reordering. Prefixes describe the joint effect of '
              'geometry, classification, candidate retention and native matching. Same-GT true-match rank joins '
              'do not align detections/queries or isolate a classification-head/Stem cause. '
              'Ranks are within each native class prediction list; list lengths differ between runs. '
              'Primary outputs only, independent EMA replay pending; BASE12 gate unchanged.')
    write(OUT / 'report.json', report)
    lines = ['第12轮原生分数顺序与PR曲线复原（增强Stem对RGB分支，主记录）', '',
        '四组已下载记录，每组10类×10个IoU×101个召回采样格，全尺寸PR共40400格精确复原，最大差异0。没有重新匹配框、重排分数或运行模型/评估器。',
        '排序采用原生按图像顺序拼接、mergesort保留并列分数顺序。GT跨运行对齐，检测框/查询没有跨运行对齐。', '']
    for seed in (1, 2):
        full = outputs[(seed, 'full')]['description']['classes'][3]['by_IoU'][5]
        rgb = outputs[(seed, 'RGB')]['description']['classes'][3]['by_IoU'][5]
        lines += [f'seed{seed} car / IoU0.75：', '前N个按原生分数排序的预测 | 增强TP/FP | RGB对照TP/FP']
        for n in (100, 500, 1000, 2000, 5000, 10000):
            f = next(x for x in full['prefixes'] if x['rank'] == n); r = next(x for x in rgb['prefixes'] if x['rank'] == n)
            lines.append(f"{n} | {f['TP']}/{f['FP']} | {r['TP']}/{r['FP']}")
        pair = comparisons[seed - 1]['classes'][3]['by_IoU'][5]
        lines += [f"全部预测末端TP：增强 {full['final_TP']}，RGB {rgb['final_TP']}；末端FP：增强 {full['final_FP']}，RGB {rgb['final_FP']}。",
                  f"新增TP GT数 {pair['gained_TP']}，其增强分数顺序中位数 {pair['full_gained_TP_median_native_rank']}；丢失TP GT数 {pair['lost_TP']}，其RGB分数顺序中位数 {pair['RGB_lost_TP_median_native_rank']}。", '']
    lines += ['seed1中，增强的正确检出总量略增，但前段预测更差；这与AP75下降一致。seed2前段预测变好，类别AP75也提升。两种子方向不同，不支持稳定的单一排序收益。',
        '这是输出层面的位置、类别分数、候选保留和匹配共同结果，不能唯一解释为分类头故障。跨运行相同GT的分数顺序可比较，不能冒充同一查询的迁移。',
        '后续选点记录仍需通过独立EMA复核，区分初始选择位置与所选框回归变化；正式12轮基线规则不变。']
    (OUT / '原生排序与PR证据.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    write(OUT / 'manifest.json', dict(files={x.name: sha(x) for x in OUT.iterdir() if x.is_file()}))
    print(json.dumps(dict(status=report['status'], cells=40400, max_abs_error=0., car_pairs=[dict(seed=x['seed'], IoU75=x['classes'][3]['by_IoU'][5]) for x in comparisons])))


if __name__ == '__main__': analyze()
