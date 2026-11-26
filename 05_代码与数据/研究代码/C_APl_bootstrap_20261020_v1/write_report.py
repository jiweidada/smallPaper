"""Describe the completed, unchanged bootstrap; never modifies its analysis."""
import csv
import hashlib
import json
from pathlib import Path
from audit import OUT, V2, now, write_csv, dump


def main():
    summary=json.loads((OUT/'汇总与核验.json').read_text())
    protocol=json.loads((OUT/'冻结分析口径.json').read_text())
    for split,expected_missing in [('val',235),('testdev',104)]:
        records=[json.loads(p.read_text()) for p in (OUT/'models').glob(f'{split}_*.json')]
        assert len(records)==20
        assert all(r['replicates_missing_any_category']==expected_missing for r in records)
    rows=[]
    for split,ds in [('val','val'),('testdev','test-dev')]:
        ann=json.loads((V2/f'data/annotations/{ds}/converted.json').read_text(encoding='utf8'))
        gt=[g for g in ann['annotations'] if not g.get('iscrowd',0) and 96**2<=g['area']<=1e10]
        for cat in ann['categories']:
            gs=[g for g in gt if g['category_id']==cat['id']]
            rows.append({'split':split,'category_id':cat['id'],'category':cat['name'],
                         'large_GT':len(gs),'images_with_large_GT':len({g['image_id'] for g in gs})})
    write_csv(OUT/'大目标_GT及有效图像数.csv',rows)
    main=summary['main_comparisons']
    zero_included=sum(r['ci95_low_pp']<=0<=r['ci95_high_pp'] for r in main)
    lines=[
        'C 稿 APl 的图像抽样与训练种子核查（CPU，只使用既有预测）',
        'Material Passport: exploratory statistical audit; 40 archived prediction exports; ANALYZED with verified metric/accumulator parity; no new training/inference/remote access.',
        f"完成时间 UTC：{summary['completed_utc']}",
        '',
        '一、结论边界',
        '现有跨种子结果不足以把 C 的大目标损失写成普遍、确定的结构性代价；C-baseline 与 C-C-mean 均存在种子间方向反转。',
        f'本轮主比较的{zero_included}/{len(main)}个图像配对区间包含零（包含单seed比较及共同seed均值），在这一探索性分析中未确认稳定的APl下降方向；这不是等效或纯噪声的证明。',
        '同样不能把已观察到的负差值直接改称噪声：应查看下表的同图像配对区间。单模型区间宽或重叠，并不等于配对差值区间宽或包含零。',
        '本报告不估计两/三个种子所代表的训练种子总体效应，不做等效判断，不把抽样区间和种子极差相加。',
        '',
        '二、实际执行口径',
        'val 全部548图、test-dev全部1610图；按图像有放回抽样，各2000次；同一split的全部模型和种子使用同一份抽样次数矩阵。',
        '保持原图面积 large>=96²、原GT/忽略处理、原score0/top300预测，project COCO AP@[.50:.05:.95]、101召回点、类别等权、maxDets100。',
        '重复图像视作保留全GT与全预测的独立副本。相同分数保留稳定的图像/副本/检测顺序，未将重复imgIds交给会去重的COCO接口。',
        '区间为2.5/97.5百分位；先逐次求模型间差值，再求差值区间。跨种子均值使用共同种子的配对差，仍仅条件于已训练权重。',
        '类别在某次重采样中无有效GT时沿用COCO排除规则并记录；不当作AP=0。5项合成边界测试通过，包含同分预测、重复图像、忽略检测与无GT类别。',
        f"40份预测均复现CSV APl；最大原始0–1误差={summary['point_parity_max_abs_error_raw']:.3g}。每个模型另外以官方累积程序核对两次重复图像抽样；最大误差={summary['duplicate_parity_max_abs_error_raw']:.3g}。",
        '',
        '三、主比较（单位均为百分点；95%区间仅包含图像抽样不确定性）',
        'split | 比较 | seed | 差值 | 区间下限 | 区间上限',
    ]
    for r in main:
        label='共同种子均值' if r['kind']=='mean_of_common_seed_pairs' else '单种子'
        lines.append(f"{r['split']} | {r['candidate']} - {r['reference']} | {r['seeds']} ({label}) | {r['point_delta_pp']:+.6f} | {r['ci95_low_pp']:+.6f} | {r['ci95_high_pp']:+.6f}")
    lines += ['',
        '四、为什么总GT数仍不足以说明APl稳定',
        'val大目标1068个，出现于356图；test-dev大目标2409个，出现于734图。APl对有效类别等权，不按每类GT数加权。',
        'test-dev 的 people/bicycle/motor 大目标仅3/5/6个，分别分布于3/5/5图；val也有仅3–7个大目标的类别。',
        '本轮2000次抽样，val有235次（11.75%）、test-dev有104次（5.20%）缺少至少一个有效大目标类别；有效类别数分别为7–10、9–10。',
        '因此单模型区间明显不对称，不能简写为±1.5pp，也不能仅由2409个GT推断有效样本量足够。完整类别清单另存CSV。',
        '',
        '五、如何决定后续工作',
        '当前不把“消除尺度代价”确立为新方法主线，不启动路径诊断、门控、FDR新臂或新训练。',
        '若某个配对区间排除零，只说明该已训练模型对在图像抽样下的负差值比较稳定，不说明训练种子总体必然有负效应。',
        '若区间包含零，只说明本分析不能确认方向，不说明差值已被证明是噪声或两模型等效。',
        '保留原始负差值与所有种子，避免挑最差seed构造机制故事；需要描述性拆分时明确它仅解释特定权重对。',
        '',
        '六、稿件建议（本轮未直接编辑中英文稿）',
        '中文：大目标AP的差异随种子和参照模型变化。我们在若干配对中观察到下降，但当前种子数不足以确认普遍的尺度代价。图像bootstrap量化的是固定权重下的评测抽样不确定性，不能消除训练种子不确定性。',
        'English: Large-object AP differences vary with the training seed and comparator. Although decreases occur in individual pairs, the available seeds do not establish a consistent scale-dependent penalty. Image-bootstrap intervals quantify evaluation-sample uncertainty conditional on the selected checkpoints and do not capture population-level training-seed uncertainty.',
        '',
        '七、限制',
        '图像被视为独立抽样簇；若存在相关帧/场景，区间可能偏窄。本轮不声称完成场景/序列级bootstrap。',
        'val曾用于best_stg2选择；区间不含选模不确定性。test-dev也已被反复分析，本轮属于探索性核查。',
        '所有臂间/臂内区间完整导出，未经多重比较校正，不作筛选式显著性宣称。11/11项统计误用检查保存在汇总JSON。',
        '',
        '文件：主比较_同种子与共同种子均值区间.csv；各模型_APl图像抽样区间.csv；全部配对差值_图像抽样区间.csv；全部臂_种子描述统计.csv；冻结分析口径.json；汇总与核验.json。',
    ]
    target=OUT/'核查结果与结论边界.txt'
    target.write_text('\n'.join(lines)+'\n',encoding='utf8')
    dump(OUT/'交付清单.json',{'created_utc':now(),'report':str(target),
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()},
        'dependency_py_coco_2_0_11_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).parent/'runtime/pycocotools').glob('*') if p.is_file()},
        'new_training':False,'new_model_inference':False,'manuscript_edited':False})
    print(target)


if __name__=='__main__': main()
