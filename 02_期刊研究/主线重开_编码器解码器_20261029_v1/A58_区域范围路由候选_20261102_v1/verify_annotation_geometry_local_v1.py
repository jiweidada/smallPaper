"""Verify the single data-only pass and write a narrowly scoped report."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

P=Path(__file__).resolve().parent
R54=P.parent/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rpc=load(R54/'A58_annotation_geometry_poll_export_v1.json');assert rpc['exit']==0
remote=json.loads(rpc['stdout']);assert remote['status']=='PASS_SINGLE_ANNOTATION_HEADER_PASS' and remote['source_pid_absent']
download=load(R54/'A58_annotation_geometry_download_v1.json');assert download['status']=='TRANSFERRED'
z=R54/'A57_downloads/A58_annotation_geometry_terminal_v1.zip'
assert z.stat().st_size==remote['export']['bytes'] and sha(z)==remote['export']['sha256']
target=P/'annotation_geometry_terminal_v1.zip';assert not target.exists();shutil.copyfile(z,target)
d=P/'annotation_geometry_remote_v1';d.mkdir(exist_ok=False)
with zipfile.ZipFile(z) as f:
    assert all(Path(n).name==n for n in f.namelist())
    for n in f.namelist():(d/n).write_bytes(f.read(n))
r=load(d/'result.json')
assert r['status']=='PASS_SINGLE_ANNOTATION_HEADER_PASS'
assert r['model_constructors']==r['model_forwards']==r['optimizer_updates']==r['AP_evaluations']==0
assert r['script_sha256']==sha(d/'audit_annotation_geometry_v1.py')==sha(P/'audit_annotation_geometry_v1.py')
assert r['scope_sha256']==sha(d/'annotation_geometry_scope_FROZEN_v1.json')==sha(P/'annotation_geometry_scope_FROZEN_v1.json')
rows=[]
for split,images,valid,scored in [('train',6471,343204,343204),('val',548,38759,38602)]:
    s=r['splits'][split];assert s['images']==images and s['valid_boxes_before_ignore']==valid and s['scored_or_training_boxes']==scored
    small=s['small_original_area'];n=small['objects']
    assert n==sum(v['objects'] for v in s['small_by_class'].values())
    assert n==sum(small['same_region_bbox_center_count'].values())
    assert scored==s['all_sizes']['objects']
    quantiles=list(small['own_bbox_occupancy_quantiles'].values())
    assert all(0<=v<=1 for v in quantiles) and quantiles==sorted(quantiles)
    for t,x in small['own_bbox_fraction_at_most'].items():
        assert x['count']==sum(v['own_bbox_fraction_at_most'][t]['count'] for v in s['small_by_class'].values())
        assert abs(x['fraction']-x['count']/n)<1e-15
    rows.append(dict(split=split,scored_or_training_boxes=scored,small_boxes=n,small_fraction=n/scored,
        own_bbox_center_region_median=small['own_bbox_occupancy_quantiles']['0.5'],
        own_bbox_center_region_at_most_quarter=small['own_bbox_fraction_at_most']['0.25'],
        same_region_has_multiple_bbox_centers_fraction=1-small['same_region_bbox_center_count'].get('1',0)/n))
out=dict(status='PASS_SOURCE_BYTES_AND_AGGREGATE_CONSISTENCY',summary=rows,
         new_audit_detector_forwards=0,new_optimizer_updates=0,scope=r['scope'],
         remote_monotonic_seconds=r['monotonic_seconds'])
with (P/'annotation_geometry_local_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
train,val=rows
report=f'''A58单次标注几何统计：小目标在固定区域内的自身框占比

范围：对VisDrone2019-DET训练6471张、验证548张各读一次标注和图像头尺寸。没有解码像素张量、构造模型、检测器前向、优化器更新或AP评价；没有改动正式B/M/Q代码、参数或门槛。单次统计正常退出，源码、范围及归档字节核对通过。

定义
沿用原实验640×640伸缩输入、P3 stride8和4×4 token区域，空间区域为32×32输入坐标。
先找每个GT框中心所在的32×32区域，然后统计“该GT框与此区域的相交面积/1024”。小目标按裁剪后的原图框面积≤1024确定，与当前评价口径一致。
训练使用实际有效10类框343204个；验证先按固定class0矩形并集IoA≥0.5去掉157个框，剩38602个。不是VisDrone所有官方标注的通用统计。

数据事实
训练：小目标{train['small_boxes']}个，占有效框{100*train['small_fraction']:.2f}%；自身框与中心区域相交面积比例的中位数为{100*train['own_bbox_center_region_median']:.2f}%；其中{train['own_bbox_center_region_at_most_quarter']['count']}个（{100*train['own_bbox_center_region_at_most_quarter']['fraction']:.2f}%）不超过区域面积1/4。
验证：小目标{val['small_boxes']}个，占计分框{100*val['small_fraction']:.2f}%；自身框与中心区域相交面积比例的中位数为{100*val['own_bbox_center_region_median']:.2f}%；其中{val['own_bbox_center_region_at_most_quarter']['count']}个（{100*val['own_bbox_center_region_at_most_quarter']['fraction']:.2f}%）不超过区域面积1/4。
同一中心区域还包含其他有效/计分GT框中心的比例：训练{100*train['same_region_has_multiple_bbox_centers_fraction']:.2f}%，验证{100*val['same_region_has_multiple_bbox_centers_fraction']:.2f}%。这是框中心共区统计，不是可见前景覆盖率。

可作的推断
“单个小目标自身框只占一个固定区域的少数面积”在这个输入和分块方案下很常见，值得让M/Q对照检验其关系是否在区域表示中受到影响。

不能作的推断
框不是可见目标分割掩码；框外可能有其他目标；token感受野可以大于坐标格子；几何占比不等于query特征贡献或注意力质量。
本统计没有证明均值路由造成漏检、没有证明Q上界更紧或有效，也不能替代三组固定12轮终点比较。不因这个统计增加训练预算或更改原停止规则。

待验证假设
小目标query偏离区域代表值时，Q能否在同样top4区域预算下保留其有用关系，同时避免把高跨度背景区域选入；是否比M取得更好的APs和可接受延迟。

证据：annotation_geometry_remote_v1/result.json（含各类别、分位数、阈值计数及标注/尺寸流SHA256）；annotation_geometry_scope_FROZEN_v1.json；annotation_geometry_local_verification_v1.json。
'''
with (P/'06_标注几何事实与机制边界.txt').open('x',encoding='utf-8') as f:f.write(report)
print(json.dumps(dict(status=out['status'],summary=rows,new_model_forwards=0),ensure_ascii=False))
