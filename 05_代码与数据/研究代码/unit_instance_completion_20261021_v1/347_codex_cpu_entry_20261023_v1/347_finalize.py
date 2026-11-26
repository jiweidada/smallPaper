"""Exclusive347 handoff, with read-only postchecks of prior seals and caches."""
import importlib,time
from pathlib import Path
C=importlib.import_module('347_context')

def link(name,label=None):
    path=Path(name)
    if not path.is_absolute():path=C.HERE/path
    return f'[{label or path.name}]({path.as_posix()})'

def main():
    r=C.read(C.HERE/'347_验收_result.json')
    assert r['subject_steps']==r['reference_steps']==10
    assert r['counts']['optimizer_steps']==20 and len(r['negative_rows'])==10
    assert not (C.HERE/'347_实现失败_停止.json').exists()
    assert not (C.HERE/'347_隔离参考实现失败_停止.json').exists()
    start=time.perf_counter();bundle_sha=C.verify_sources();old_rows=[]
    for seal in (C.S345/'345_交付封存SHA256_v3.json',C.S346/'346_交付封存SHA256.json'):
        d=C.read(seal)
        for p,item in d['artifacts'].items():
            assert C.sha(p)==item['sha256'] and Path(p).stat().st_size==item['bytes'],p
        old_rows.append({'path':str(seal),'sha256':C.sha(seal),'verified_files':len(d['artifacts'])})
    cache_manifest=C.BASE.S5/'342_活动缓存7566_逐条SHA总清单.json'
    caches=C.read(cache_manifest)['entries'];total_bytes=0
    for item in caches:
        p=Path(item['path'])
        assert p.stat().st_size==item['bytes'] and C.sha(p)==item['file_sha256'],str(p)
        total_bytes+=item['bytes']
    assert len(caches)==7566
    F=importlib.import_module('train347')
    assert F.CURRENT_CPU_CAP=={'formal':1218,'mini_acceptance':6}
    assert C.E.CURRENT_CPU_CAP=={'formal':2,'mini_acceptance':6}
    C.write('347_旧封存与缓存结束核验.json',{'status':'已验证','old_seals':old_rows,
            'active_caches_verified':7566,'active_cache_bytes':total_bytes,
            'cache_manifest_sha256':C.sha(cache_manifest),'new_source_bundle_sha256':bundle_sha,
            'new_caps':F.CURRENT_CPU_CAP,'old_caps_unchanged':C.E.CURRENT_CPU_CAP,
            'wall_seconds':time.perf_counter()-start,'head_forwards':0,'optimizer_steps':0,
            'source_sha256':C.sources()})
    times=[]
    for name,folder,filename in (
        ('new347',C.HERE/'347_subject_G_seed1_10','347_epoch_times.json'),
        ('reference345',C.HERE/'347_reference345_G_seed1_10','345_epoch_times.json')):
        doc=C.read(folder/filename)
        seconds=sum(v['seconds'] for v in doc['completed_epochs'])+doc['partial_seconds']
        times.append({'entry':name,'optimizer_steps':10,'update_segment_seconds':seconds,
                      'mean_seconds_per_step':seconds/10})
    C.write('347_CPU验收计时.json',{'status':'实测，仅十步样本','rows':times,
            'full_acceptance_harness_wall_seconds':r['wall_seconds'],
            'scope':'update-segment timing includes input loading/collate/head forward/loss/backward/clip/optimizer and batch fingerprint; excludes setup/checkpoint writing. Timing is not part of bit-equality assertions.',
            '未核':['完整1218步耗时','八run与评估总墙钟']})
    C.text('347_一页中文汇总.md',f'''# 347 正式CPU入口与G_seed1十步验收

日期：2026-10-23。状态：实现与验收通过，封存后停止。八个完整run仍未授权，实际完整run数0。

另存train347.py，正式模式许可从2步改为1218；1218硬预算、CPU FP32、mini上限6、配置/数据/曲线/初态/断点SHA、既有目录拒绝覆盖等守卫保持。新入口还将自身源码SHA加入原有绑定。旧345/346及所有冻结数学文件均未修改。

完整数值while循环AST逐项相同，所有execute中的require调用AST相同；原绑定检查保持原顺序。独立diff列明新入口bootstrap、源码绑定、347文件名/schema及中性执行状态。旧入口硬编码“真实训练未执行”的状态不用于将来的正式执行；新入口只报告实际物理步数和完整预算是否完成，由本轮验收报告明确当前授权范围。

| 对照 | 实际结果 |
|---|---|
| 新347的前2步 vs 既有345正式日志 | 所有字段逐位一致，序列化UTF-8字节一致 |
| 新347完整10步 vs 隔离原345参考10步 | model/optimizer/RNG/各组LR/逐步日志逐位一致，模型及优化器最大差0 |
| 图ID和学习率 | 逐步匹配真实物化顺序和原1218点曲线，G的κ路由保留 |
| 保留守卫 | 9个本入口反例＋参考进程拒绝第11步，共10例全部拒绝 |

隔离参考仅在其子进程RAM里将原345的formal许可改成10，源文件SHA前后相同，退出后父进程原345仍为2。两份对照都从冻结公共初态开始，没有使用346构造游标。

本轮两份正向工作进程分别实际更新10步，合计20次CPU前向/反向/优化步、320次缓存加载、640次native loss调用；global前向0、真实科学训练0。末态step10/epoch0/batch10/图偏移160，未完成1218步预算。实际耗时单独记账，不声称逐位相同；整体验收实测{r['wall_seconds']:.3f}秒。

结束核对旧两份封存合计{sum(v['verified_files'] for v in old_rows)}个条目与7566活动缓存SHA均一致。通过的反例包含配置/清单/shuffle篡改、1219步、既有run目录、正式模式误用mini、两类346构造断点、CLI学习率覆写、参考进程第11步。

当前交付只完成347授权。八个完整run、val548评估、G Δ、bootstrap、12项科学判定均未执行；不使用GPU或SSH，不读heldout图像/缓存。不提供真实训练启动命令。正式训练须用户另行授权。

{link('train347.py')} · {link('347_验收_result.json')} · {link('347_交接索引.md')}。
''')
    C.text('347_交接索引.md',f'''# 347 交接索引

状态：正式CPU入口另存完成，G_seed1十步验收通过；已停止，八个完整run未授权。

- {link('train347.py')}：实际新CPU入口，formal上限1218，mini6，硬预算1218；仅五个CLI参数，配方读旧冻结配置。
- {link('347_train345_to_train347.diff')}、{link('347_静态继承证明.json')}：原循环和守卫AST相同，新增源码SHA绑定及输出元数据变更逐项披露。
- {link('347_用户授权与预定验收口径.json')}：用户本轮明确授权与执行前固定的比较范围和20实际优化步预算。
- {link('347_验收_result.json')}：主结果、实际调用数、10个反例及所有源码SHA。
- {link('347_前2步与既有345日志对照.json')}：旧2步原始日志与新入口前缀逐位对照。
- {link('347_完整10步逐位对照.json')}：完整模型/优化器/RNG/step日志比较，耗时另列。
- {link('347_隔离参考进程证明.json')}：原345文件未改、RAM许可10、拒绝11、进程PID和实际调用数。
- {link('347_原有守卫反向验收.json')}：本入口9个反向用例；另1个参考反例见上一项。
- {link('347_subject_G_seed1_10')}、{link('347_reference345_G_seed1_10')}：两份10步实际输出、完整断点和日志；参考目录内345文件名由未改动的原函数生成。
- {link('347_入口源码SHA绑定.json')}、{link('347_bindings_anchor.py')}：旧数据配置绑定保留，额外封存新入口及支持脚本源码。
- {link('347_旧封存起始核验.json')}、{link('347_旧封存与缓存结束核验.json')}：旧345/346封存与7566缓存结束复核。
- {link('347_CPU验收计时.json')}、{link('347_一页中文汇总.md')}：实际计时范围与中文摘要。
- {link('347_交付封存SHA256.json')}：最终交付SHA清单。

本轮完整10步参考是新运行的隔离原345进程，不把旧345仅有的2步日志冒称成10步证据。两份10步checkpoint在model/optimizer/RNG/确定性step日志上逐位相同；binding中新347入口身份/SHA、schema与实际墙钟元数据有明确差别，不声称整个checkpoint文件字节相同。

已放行的346只作CPU边界验证，构造断点仍被新347拒绝。此前审查包v2与346结果保持只读，本轮交付作为新入口验收附录。当前没有创建八个正式科学run，没有执行val/G Δ/bootstrap/12项判定；用户另行授权后才进入正式运行阶段。本索引不提供训练启动命令。
''')
    paths=[p for p in sorted(C.HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    items={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in paths}
    C.write('347_交付封存SHA256.json',{'status':'347实现与验收通过，封存后停止；八个完整run未授权',
            'artifacts':items,'files':len(items),'bytes':sum(v['bytes'] for v in items.values()),
            'active_entry':str(C.HERE/'train347.py'),'active_index':str(C.HERE/'347_交接索引.md'),
            'actual_optimizer_steps_CPU_acceptance':20,'scientific_full_runs':0,'real_training_steps':0,
            'source_sha256':C.sources()})
    print('347 delivery sealed: entry1218, subject10 + reference10 matched; old seals/caches unchanged; stopped',flush=True)

if __name__=='__main__':main()
