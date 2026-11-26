"""Exploratory arithmetic of actual fixed12 PR tensors and saved GT stages.

All statistics are primary outputs; independent EMA replay remains pending.
AP-cell contributions are arithmetic, not a causal decomposition of a network.
"""
from pathlib import Path
import gzip, hashlib, importlib.util, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/stem_AP_primary_breakdown12_20261011_v1'
read = lambda p: json.loads(Path(p).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def mean_percent(values):
    values = np.asarray(values); valid = values[values >= 0]
    return float(valid.mean()) * 100 if valid.size else -100.


def checked_cache(folder):
    m = read(folder / 'manifest.json')
    for n, h in m['files'].items():
        path = Path(n); assert not path.is_absolute() and '..' not in path.parts and ':' not in n
        assert sha(folder / path) == h, n
    return sha(folder / 'manifest.json')


def primary(folder, kind, reader, annotation):
    verification = read(folder / 'primary_verification_v1.json')
    assert verification['status'] == {'full': 'PASS_DOWNLOADED_EXACT12_PRIMARY_ARTIFACTS_ONLY',
        'control': 'PASS_COMPONENT_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'}[kind]
    assert not verification['saved_EMA_independent_replay_verified'] and not verification['formal_terminal_reader_passed']
    assert verification['actual_stage_images'] == 548
    packet = Path(verification['downloaded_packet']); manifest = read(packet / 'primary_manifest.json')
    assert sha(packet / 'primary_manifest.json') == verification['primary_manifest_sha256']
    for n, item in manifest['files'].items():
        rel = Path(n); assert not rel.is_absolute() and '..' not in rel.parts and ':' not in n
        assert (packet / rel).stat().st_size == item['bytes'] and sha(packet / rel) == item['sha256'], n
    endpoint = packet / 'run/endpoint_epoch12'
    summary, flags, _ = reader.native_endpoint(endpoint, annotation)
    assert summary['identity'] == verification['identity'] and summary['native_stats'] == verification['native_stats']
    trained = read(packet / 'run/training_completion.json')
    assert trained['status'] == 'NATIVE_FIT_FINISHED_EXACT12' and trained['completed_epochs'] == 12
    assert trained['entered_epochs'] == list(range(12)) and trained['identity'] == summary['identity']
    assert sha(endpoint / 'manifest.json') == trained['endpoint_manifest_sha256']
    data = npz(endpoint / 'precision_recall.npz'); precision, recall = data['precision'], data['recall']
    assert precision.shape == (10, 101, 10, 4, 3) and recall.shape == (10, 10, 4, 3)
    assert np.isfinite(precision).all() and ((precision == -1) | ((precision >= 0) & (precision <= 1))).all()
    fp = np.zeros((10, 10), np.int64); tp = np.zeros_like(fp)
    with gzip.open(endpoint / 'matching_all_area.json.gz', 'rt', encoding='utf8') as f: matches = json.load(f)
    for row in matches:
        assert row['aRng'] == [0, 1e10] and row['maxDet'] == 100
        dt, ignore = np.asarray(row['dtMatches']), np.asarray(row['dtIgnore'], bool)
        k = int(row['category_id'])
        tp[:, k] += ((dt != 0) & ~ignore).sum(1); fp[:, k] += ((dt == 0) & ~ignore).sum(1)
    assert tp.sum(1).tolist() == summary['TP_by_IoU'] and fp.sum(1).tolist() == summary['FP_by_IoU']
    return dict(summary=summary, flags=flags, precision=precision, recall=recall, all_area_TP=tp, all_area_FP=fp,
        source=dict(primary_verification_sha256=sha(folder / 'primary_verification_v1.json'),
            primary_manifest_sha256=sha(packet / 'primary_manifest.json'), endpoint_manifest_sha256=sha(endpoint / 'manifest.json'),
            checkpoint_sha256=trained['last_checkpoint_sha256'], fixed_epoch_one_based=12, weights='EMA'))


def analyze():
    assert not OUT.exists()
    source_path = ROOT / '英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py'
    source = module('AP_breakdown_pinned_sources', source_path)
    stage_p, geometry, onset, rank, readers = source.prepared()
    caches = dict(full=ROOT / 'outputs/stem_primary_joint_analysis12_20261011_v1',
                  RGB=ROOT / 'outputs/stem_highfreq_primary_analysis12_20261011_v1')
    cache_manifests = {k: checked_cache(v) for k, v in caches.items()}
    assert read(caches['full'] / 'report.json')['status'] == 'ANALYZED_BOTH_PRIMARY_FIXED12_JOINT_PARTITION_REPLAY_PENDING'
    assert read(caches['RGB'] / 'report.json')['status'] == 'ANALYZED_BOTH_SEEDS_FULL_VS_RGB_PRIMARY12_SAME_GT_REPLAY_PENDING'
    results = []; native_sources = {}; annotation_hash = None
    for seed in (1, 2):
        base = source.cached_baseline(seed, stage_p, readers['full']); annotation = base['annotation']
        actual_hash = sha(base['packet'] / 'reference/val_annotations.json')
        if annotation_hash is None: annotation_hash = actual_hash
        assert annotation_hash == actual_hash
        assert all(not x.get('ignore', 0) and not x.get('iscrowd', 0) for x in annotation['annotations'])
        cats = sorted(annotation['categories'], key=lambda c: c['id']); assert [c['id'] for c in cats] == list(range(10))
        ranges = [(0, 1e10), (0, 1024), (1024, 9216), (9216, 1e10)]
        gt_counts = np.array([[sum(x['category_id'] == c['id'] and low <= x['area'] <= high for x in annotation['annotations'])
                               for low, high in ranges] for c in cats], np.int64)
        endpoints = {}
        for arm, kind, folder in (
            ('full', 'full', ROOT / f'outputs/stem_primary_collection_20261011_v1/seed{seed}_original_stem_v1'),
            ('RGB', 'control', ROOT / f'outputs/stem_component_primary_collection_20261011_v2/seed{seed}_highfreq_rgb_v1')):
            endpoints[arm] = primary(folder, kind, readers[kind], annotation)
            assert endpoints[arm]['summary']['identity']['seed'] == seed
            assert endpoints[arm]['summary']['identity']['val_annotation_sha256'] == annotation_hash
        full, rgb = endpoints['full'], endpoints['RGB']
        for k in ('common_init_sha256', 'baseline_checkpoint_sha256', 'baseline_first100_sha256', 'evaluator_binding'):
            assert full['summary']['identity'][k] == rgb['summary']['identity'][k], k
        pf, pr = full['precision'][:, :, :, :, 2], rgb['precision'][:, :, :, :, 2]
        assert np.array_equal(pf >= 0, pr >= 0), 'Different valid AP-cell populations'
        native_area_tp = {}
        for arm, endpoint in endpoints.items():
            rec = endpoint['recall'][:, :, :, 2]; raw = rec * gt_counts[None, :, :]
            assert np.max(np.abs(raw - np.rint(raw))) < 1e-7
            counts = np.rint(raw).astype(np.int64); assert (counts >= 0).all()
            assert counts[:, :, 0].sum(1).tolist() == endpoint['summary']['TP_by_IoU']
            native_area_tp[arm] = counts
        all_delta = {k: full['summary']['overall_percent'][k] - rgb['summary']['overall_percent'][k]
                     for k in full['summary']['overall_percent']}
        fields = {'AP': (slice(None), 0), 'AP50': (0, 0), 'AP75': (5, 0), 'APs': (slice(None), 1),
                  'APm': (slice(None), 2), 'APl': (slice(None), 3), 'small_AP75': (5, 1)}
        contribution_sum = {k: 0. for k in fields}; classes = []
        stage_arrays = dict(full=npz(caches['full'] / f'seed{seed}_original_stem/stage_per_GT.npz'),
                            RGB=npz(caches['RGB'] / f'seed{seed}/RGB_stage_per_GT.npz'))
        for arm, v in stage_arrays.items():
            flags = endpoints[arm]['flags']
            assert np.array_equal(v['annotation_id'], flags['annotation_id']) and np.array_equal(v['native_TP'], flags['hits'])
        for k in ('annotation_id', 'category_id', 'image_id', 'original_area'):
            assert np.array_equal(stage_arrays['full'][k], stage_arrays['RGB'][k]), k
        for cid, category in enumerate(cats):
            row = dict(class_id=cid, name=category['name'], GT_by_native_area=dict(zip(('all', 'small', 'medium', 'large'), gt_counts[cid].tolist())),
                full_AP_percent={}, RGB_AP_percent={}, delta_AP_points={}, contribution_to_aggregate_AP_points={},
                native_area_specific_TP_delta_by_IoU={area: (native_area_tp['full'][:, cid, ai] - native_area_tp['RGB'][:, cid, ai]).tolist()
                    for ai, area in enumerate(('all', 'small', 'medium', 'large'))},
                native_all_area_FP_delta_by_IoU=(full['all_area_FP'][:, cid] - rgb['all_area_FP'][:, cid]).tolist())
            for key, (ti, ai) in fields.items():
                f, r = pf[ti, :, cid, ai], pr[ti, :, cid, ai]
                fv, rv = mean_percent(f), mean_percent(r)
                assert abs(fv - full['summary']['categories_percent'][cid][key]) < 1e-9
                assert abs(rv - rgb['summary']['categories_percent'][cid][key]) < 1e-9
                denom = int((pf[ti, :, :, ai] >= 0).sum()); valid = f >= 0
                contribution = float((f[valid] - r[valid]).sum()) * 100 / denom
                row['full_AP_percent'][key], row['RGB_AP_percent'][key] = fv, rv
                row['delta_AP_points'][key] = fv - rv
                row['contribution_to_aggregate_AP_points'][key] = contribution; contribution_sum[key] += contribution
            row['saved_geometric_support_delta'] = {}
            for area, (low, high) in zip(('all', 'small', 'medium', 'large'), ranges):
                values = stage_arrays['full']; mask = (values['category_id'] == cid) & (values['original_area'] >= low) & (values['original_area'] <= high)
                row['saved_geometric_support_delta'][area] = {}
                for threshold in (.5, .75):
                    fs, rs = stage_arrays['full']['max_iou'][mask] >= threshold, stage_arrays['RGB']['max_iou'][mask] >= threshold
                    row['saved_geometric_support_delta'][area][str(threshold)] = dict(
                        GT=int(mask.sum()), delta_by_boundary=dict(zip(geometry.STAGES, (fs.sum(0) - rs.sum(0)).tolist())))
            classes.append(row)
        for key in fields: assert abs(contribution_sum[key] - all_delta[key]) < 1e-9, key
        assert ((pf[:, :, :, 0] >= 0).sum((0, 1)) == 1010).all()
        assert ((pf[:, :, :, 1] >= 0).sum((0, 1)) == 1010).all()
        iou_rows = [dict(IoU=float(t), all_area_AP_delta_points=mean_percent(pf[i, :, :, 0]) - mean_percent(pr[i, :, :, 0]),
            small_AP_delta_points=mean_percent(pf[i, :, :, 1]) - mean_percent(pr[i, :, :, 1]),
            native_all_area_TP_delta=int((full['all_area_TP'][i] - rgb['all_area_TP'][i]).sum()),
            native_all_area_FP_delta=int((full['all_area_FP'][i] - rgb['all_area_FP'][i]).sum()))
            for i, t in enumerate(full['summary']['protocol']['iou_thresholds'])]
        assert abs(np.mean([x['all_area_AP_delta_points'] for x in iou_rows]) - all_delta['AP']) < 1e-9
        assert abs(np.mean([x['small_AP_delta_points'] for x in iou_rows]) - all_delta['APs']) < 1e-9
        results.append(dict(seed=seed, full_AP_percent=full['summary']['overall_percent'], RGB_AP_percent=rgb['summary']['overall_percent'],
            full_minus_RGB_points=all_delta, exact_category_contribution_sums=contribution_sum, classes=classes, by_IoU=iou_rows,
            positive_class_AP_contribution=sum(c['contribution_to_aggregate_AP_points']['AP'] for c in classes if c['delta_AP_points']['AP'] > 0),
            negative_class_AP_contribution=sum(c['contribution_to_aggregate_AP_points']['AP'] for c in classes if c['delta_AP_points']['AP'] < 0)))
        native_sources[str(seed)] = {a: e['source'] for a, e in endpoints.items()}
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    OUT.mkdir()
    report = dict(status='ACTUAL_PRIMARY12_AP_ARITHMETIC_AND_CATEGORY_STAGE_DESCRIPTION_REPLAY_PENDING', seeds=results,
        source_sha256=sha(__file__), stage_source_sha256=sha(source_path), stage_protocol_sha256=sha(source.PROTOCOL),
        source_primary_packets=native_sources, derived_cache_manifests_sha256=cache_manifests, val_annotation_sha256=annotation_hash,
        saved_EMA_independent_replay_verified=False, formal_terminal_reader_passed=False, causal_root_proven=False,
        novelty_proven=False, new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0,
        scope='Both seeds and all10 categories retained. Exact contributions are sums of saved native precision cells; '
              'overall AP is not a GT-count-weighted mean of size APs. Native size-specific TP counts are separately reconstructed '
              'from saved native recall and verified GT denominators; saved stage support uses all-area GT joins and is not AP. '
              'Dense8400 boxes, feature-module causes and historical training assignments are unobserved. Full-versus-RGB '
              'changes both high-pass/low-frequency content and input statistics; no unique frequency mechanism inferred. '
              'Exploratory arithmetic based on known primary results, not a new confirmatory experiment or a passed BASE12 gate.')
    write(OUT / 'report.json', report)
    lines = ['增强Stem与RGB对照：第12轮AP差异的数值拆解（主记录，独立EMA复核待完成）', '',
        '这份分析只读取已经下载的原生PR张量、GT匹配记录和阶段缓存，没有重跑模型或评估器。',
        '总体AP是有效类别与IoU/召回采样点的平均；本数据总体AP和APs各10类均有1010个有效采样格，因此每类权重为1/10。目标数量不决定类别AP权重。', '']
    for r in results:
        lines += [f"seed{r['seed']}：增强−RGB AP={r['full_minus_RGB_points']['AP']:+.6f}，APs={r['full_minus_RGB_points']['APs']:+.6f} 点。",
                  '类别 | GT/小目标GT | 类别AP变化 | 对总体AP的贡献 | 类别APs变化']
        for c in r['classes']:
            lines.append(f"{c['name']} | {c['GT_by_native_area']['all']}/{c['GT_by_native_area']['small']} | {c['delta_AP_points']['AP']:+.6f} | {c['contribution_to_aggregate_AP_points']['AP']:+.6f} | {c['delta_AP_points']['APs']:+.6f}")
        lines += [f"正贡献合计 {r['positive_class_AP_contribution']:+.6f}；负贡献合计 {r['negative_class_AP_contribution']:+.6f}。",
                  f"AP50变化 {r['full_minus_RGB_points']['AP50']:+.6f}；AP75变化 {r['full_minus_RGB_points']['AP75']:+.6f}；APm变化 {r['full_minus_RGB_points']['APm']:+.6f}；APl变化 {r['full_minus_RGB_points']['APl']:+.6f}。", '']
    lines += ['seed1的下降来自car、awning-tricycle、bus三类，七类正贡献被三类负贡献抵消。seed2总体AP十类都提高，但bus的小目标AP下降。不能把APs平均改善写成每一种小目标都改善。',
        'seed1的全部10个IoU阈值，总TP数量都增加，但IoU0.60至0.90的AP下降。数量、类别分布和分数排序一起影响AP；数量增加不能单独证明AP提高，也不能凭此唯一归因分类头。',
        'seed1 car的IoU0.75初始框支持减少766个，其中中目标减少646、大目标减少164，小目标增加44；最终框支持反而增加46，原生TP增加36，但类别AP仍下降。初始支持损失与最终AP不是单向因果关系。',
        '这些初始框已经过TopK选择和框回归。后续实际选点诊断要区分位置选择变化与选中框回归变化；空间邻域只是代理，不代表目标归属。',
        '区域TP采用原生对应面积范围的recall重建，不拿all-area匹配的小目标子集直接冒充APs的匹配结果。',
        '直接增强Stem仍未通过与预训练BASE12比较的正式规则。以上是算术分解和观察证据，尚未证明根因、创新或稳定收益。']
    (OUT / 'AP差异与后续诊断方向.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    write(OUT / 'manifest.json', dict(files={x.name: sha(x) for x in OUT.iterdir() if x.is_file()}))
    print(json.dumps(dict(status=report['status'], seeds=[dict(seed=x['seed'], delta=x['full_minus_RGB_points'],
        positive_AP_contribution=x['positive_class_AP_contribution'], negative_AP_contribution=x['negative_class_AP_contribution']) for x in results])))


if __name__ == '__main__': analyze()
