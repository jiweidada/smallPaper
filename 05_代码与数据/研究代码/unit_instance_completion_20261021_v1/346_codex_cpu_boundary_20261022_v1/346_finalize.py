"""Read-only preservation audit and exclusive boundary acceptance delivery."""
import importlib,json,time
from pathlib import Path
C=importlib.import_module('346_context')

def link(name,label=None):
    p=Path(name)
    if not p.is_absolute():p=C.HERE/p
    return f'[{label or p.name}]({p.as_posix()})'

def main():
    result=C.read(C.HERE/'346_边界验收_result.json')
    assert len(result['case_rows'])==16 and len(result['negative_rows'])==27
    assert result['counts']['optimizer_steps']==40 and result['counts']['cache_loads']==432
    assert not (C.HERE/'346_实现失败_停止.json').exists()
    assert not (C.HERE/'346_worker_实现失败_停止.json').exists()
    start=time.perf_counter();C.verify_sources()
    old_seal=C.S345/'345_交付封存SHA256_v3.json'
    old=C.read(old_seal)
    for p,item in old['artifacts'].items():
        assert C.sha(p)==item['sha256'] and Path(p).stat().st_size==item['bytes'],p
    active_manifest=C.BASE.S5/'342_活动缓存7566_逐条SHA总清单.json'
    active=C.read(active_manifest);cache_bytes=0
    for entry in active['entries']:
        p=Path(entry['path'])
        assert p.stat().st_size==entry['bytes'] and C.sha(p)==entry['file_sha256'],str(p)
        cache_bytes+=entry['bytes']
    assert len(active['entries'])==7566
    assert C.E.CURRENT_CPU_CAP=={'formal':2,'mini_acceptance':6}
    timings=[]
    for row in result['case_rows']:
        d=C.read(Path(row['run_dir'])/'346_epoch_times.json')
        seconds=sum(r['seconds'] for r in d['completed_epochs'])+d['partial_seconds']
        timings.append({'arm':row['arm'],'seed':row['seed'],'case':row['case'],
                        'actual_update_seconds':seconds,'actual_optimizer_updates':row['actual_optimizer_steps']})
    step_seconds=sum(r['actual_update_seconds'] for r in timings)
    C.write('346_封存与缓存结束核验.json',{'status':'已验证',
            'old_sealed_files_verified':len(old['artifacts']),'old_seal_path':str(old_seal),
            'old_seal_sha256':C.sha(old_seal),'active_cache_files_verified':7566,
            'active_cache_bytes_verified':cache_bytes,'cache_manifest_sha256':C.sha(active_manifest),
            'old_entry_caps_unchanged':C.E.CURRENT_CPU_CAP,'new_source_bundle_sha256':C.verify_sources(),
            'wall_seconds':time.perf_counter()-start,'head_forwards':0,'optimizer_steps':0,
            'source_sha256':C.source_sha()})
    C.write('346_本轮CPU计时账本.json',{'status':'已执行，限定于构造短验收','rows':timings,
            'total_actual_update_seconds':step_seconds,'total_actual_updates':40,
            'harness_wall_seconds':result['wall_seconds'],
            'scope':'update seconds include cache reading/collate/head forward/loss/backward/clip/optimizer and batch fingerprint; exclude model setup/checkpoint writing. Harness wall includes fixture construction and 16 new processes.',
            'sampling_limitation':'40步里有16个3图末批，不能用本轮均值直接外推完整1218步run。完整训练耗时未核。',
            'real_training_steps':0})
    C.text('346_一页中文汇总.md',f'''# 346 本机CPU边界验收汇总

状态：两seed、四臂共16组边界短验收全部通过，27个反向用例被拒绝。完成后停下，真实训练0步。

| 构造起点→终点 | 每个arm/seed的实际更新 | 实际batch大小 | 最终位置 |
|---|---:|---|---|
| 201→204 | 3 | 16、3、16 | epoch1 / batch1 / 图偏移16 |
| 1216→1218 | 2 | 16、3 | epoch6 / batch0 / 偏移0，终止 |

两组都使用原345物化的完整3235×6顺序，不用mini文件。独立枚举全轮顺序，对照每步图ID/epoch/batch/offset；同seed四臂真实collate张量SHA相同。每个case的参数确有更新，optimizer state的实际计数最大仅2或3。检测/质量/关联损失和梯度范数均有限，学习率逐值匹配已冻结1218点曲线，G额外组保持预定曲线。

反向用例：8次拒绝205这个未授权终点、8次拒绝1219、8次拒绝从终止1218继续恢复；原345入口拒绝两类346验收断点；正式顺序加载器拒绝mini文件，共27例。负例没有前向或优化步。

本轮实际40次CPU head前向/反向/优化步，432次缓存加载、864次native loss调用、16个边界worker进程；global前向0、真实训练0。所有构造断点都来自未训练公共初态，只改变显式标记的游标。未执行的201/1216步前缀不填造历史日志或耗时，不能把1218这个终止游标写成已训练1218步。

旧345封存的{len(old['artifacts'])}个文件与7566活动缓存SHA结束核对一致。数值循环AST在去除epoch耗时字段重命名后完全相同，损失/几何/优化器仍直接调用旧模块。只另存train346和验收文件，原train345上限仍为正式2步/mini6步。

整体验收实测{result['wall_seconds']:.3f}秒；40次更新段合计{step_seconds:.3f}秒。后者不含初始化与断点写盘，并且末批3图占比高，不能直接外推8个完整run耗时。

本轮补齐CPU路线的轮次/预算边界自测。A/FLOPs/B已由用户消息放行，本轮结果待用户独立核对。正式8run、val评估、G Δ、bootstrap和12项科学判定均未执行；真实训练需另行明确授权，届时另存并封存正式CPU启动版本。当前入口继续保留短验收限制。

{link('346_边界验收_result.json')} · {link('346_交接索引.md')}。
''')
    C.text('346_交接索引.md',f'''# 346 本机CPU边界验收交接索引

当前状态：16组通过，27反例拒绝；真实训练0步，停止等待用户核对。依据是用户本轮直接指令“你自己设定目标 本机验证看看”，未自行扩大到真实训练。

- {link('346_用户指令与预定验收口径.json')}：执行前固定的范围、40实际优化步预算和停止规则。
- {link('346_边界验收_result.json')}：16组逐项独立对照、每步数值、图ID与张量SHA、27个反例、实际调用数及源码SHA。
- {link('346_一页中文汇总.md')}：中文汇总与构造游标限制。
- {link('train346.py')}、{link('346_train345_to_train346.diff')}、{link('346_数值循环继承证明.json')}：另存入口及AST继承证明；原训练循环仅调整耗时元数据，未改损失/优化器/几何。
- {link('346_构造断点清单.json')}、{link('346_fixtures')}：16份未训练公共初态＋显式虚拟游标断点。不是训练产物，旧345 binding会拒绝。
- {link('346_boundary_runs')}：16个实际短验收目录，24份完整执行断点及收据、逐步日志和实际耗时。两个旧入口拒绝测试的隔离目录见346_original345_rejection；其中初态文件名由未改动的旧入口生成。
- {link('346_冻结源码与旧封存绑定.json')}、{link('346_bindings_anchor.py')}：本轮源码及旧封存绑定。
- {link('346_旧封存起始核验.json')}、{link('346_封存与缓存结束核验.json')}：旧文件及7566缓存未变。
- {link('346_本轮CPU计时账本.json')}：计时范围与不能外推的采样限制。
- {link('346_交付封存SHA256.json')}：本目录交付SHA清单。

旧审查包v2：{link(C.S345/'342_开训前审查包_v2.md')}。旧文件保留，不回写状态；用户已确认FLOPs/B通过的消息记录在本轮口径文件中。

本轮构造起点201/1216不代表执行过那些前缀；每个模型仅实际更新3或2步。末批3图、跨epoch和1218终止已在真实顺序上验证；完整1218步实际run仍未执行。CPU训练若获单独授权，应另存正式CPU启动版本，沿用已验证数值循环、配方、数据和1218硬上限，并封存SHA。当前不提供训练启动命令，不连接SSH，不使用GPU，不读heldout图像或缓存。
''')
    files=[p for p in sorted(C.HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    artifacts={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in files}
    C.write('346_交付封存SHA256.json',{'status':'346边界验收完成，待用户核对；真实训练0',
            'artifacts':artifacts,'files':len(artifacts),'bytes':sum(i['bytes'] for i in artifacts.values()),
            'active_index':str(C.HERE/'346_交接索引.md'),'active_result':str(C.HERE/'346_边界验收_result.json'),
            'actual_optimizer_steps_CPU_acceptance':40,'real_training_steps':0,'source_sha256':C.source_sha()})
    print(f'346 delivery written; old files={len(old["artifacts"])}, caches=7566 unchanged; acceptance wall={result["wall_seconds"]:.3f}s',flush=True)

if __name__=='__main__':main()
