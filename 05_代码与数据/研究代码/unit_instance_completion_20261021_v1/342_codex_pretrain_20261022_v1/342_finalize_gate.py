"""Package existing stage 1/2 evidence at the gate; no optimizer or training."""
import importlib, importlib.util, json, re, ast
C=importlib.import_module('342_context')
import torch
import numpy as np
from PIL import Image,ImageDraw
import geom342 as G
import g_effect_342 as E
S=importlib.import_module('342_stage1')
torch.set_num_threads(2)

def main():
    stage1=json.loads((C.HERE/'342_阶段1_result.json').read_text(encoding='utf-8'))
    stage2=json.loads((C.HERE/'342_阶段2_result.json').read_text(encoding='utf-8'))
    assert stage1['status']=='已验证' and stage2['status']=='已验证（CPU工程项）；待主控'
    ids,entries=S.load_entries()
    boxes,classes,valid=S.padded(entries)
    reverse={}
    for name,bad in (
        ('cxcywh_instead_of_xyxy',G.xyxy_to_cxcywh(boxes)),
        ('pixel_instead_of_normalized',boxes*64)):
        try:
            E.measure_initial(bad,classes,valid,torch.full((10,),.67))
        except AssertionError as ex:
            reverse[name]={'assert_rejected':True,'message':str(ex)}
        else:
            raise AssertionError(name+' was accepted by G-effect metric')
    # Empty P windows must contribute a zero delta instead of being omitted.
    empty=E.measure_initial(torch.empty(1,0,4),torch.empty(1,0,dtype=torch.long),
                            torch.empty(1,0,dtype=torch.bool),torch.full((10,),.67))
    assert empty['windows']==1 and empty['delta']==0.
    # Reproduce the pre-existing module's axis bug without running its old
    # result-writing test or changing any file.
    spec=importlib.util.spec_from_file_location('_342_preserved_legacy_geom',C.WORK/'geom342.py')
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    legacy={}
    e=entries[0]; keep=G.intersecting_candidates(e['fullp_boxes'],e['window_xyxy_640'])
    loc=G.win_norm_xyxy(e['fullp_boxes'][keep],e['window_xyxy_640'])
    try: old.geometry_cost(loc)
    except RuntimeError as ex:
        legacy={'observed':'实现失败（已有geom342广播轴）','exception':str(ex),'input_shape':list(loc.shape)}
    else: raise AssertionError('legacy axis bug did not reproduce')
    # Draw candidate edges, frozen grid centers and K_g on the SAME crop.
    # Only the display is clipped. All box arithmetic uses geom342.
    pngs=[]
    for iid,e in list(zip(ids,entries))[:6]:
        w=e['window_xyxy_640']; keep=G.intersecting_candidates(e['fullp_boxes'],w)
        edges=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['fullp_boxes'][keep],w),w,(512,512)).numpy()
        rgb=(e['crop128'].permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')
        crop=Image.fromarray(rgb).resize((512,512),Image.NEAREST)
        canvas=Image.new('RGB',(1056,584),'white')
        canvas.paste(crop,(8,48));canvas.paste(crop,(536,48))
        d=ImageDraw.Draw(canvas)
        d.text((8,8),f'id={iid}; left=candidates + grid; right=candidates + grid + K_g',fill='black')
        d.text((8,24),f'W={w.tolist()}; G_W={int(e["G_W"].sum())}; U_W={int(e["U_W"].sum())}',fill='black')
        kg=e['K_g'][e['G_W']].sum(0).numpy()
        for xoff in (8,536):
            for edge in edges:
                bounded=np.clip(edge,0,511)+[xoff,48,xoff,48]
                d.rectangle(tuple(bounded),outline='#46f0f0',width=1)
            for (x,y),weight in zip(G.grid_xy().numpy()*512,kg):
                x,y=x+xoff,y+48
                d.ellipse((x-1,y-1,x+1,y+1),fill='white')
                if xoff==536 and weight>0:
                    radius=2+6*np.sqrt(float(weight)/max(float(kg.max()),1e-12))
                    d.ellipse((x-radius,y-radius,x+radius,y+radius),outline='#ff3030',width=2)
        d.text((8,566),'Cyan: candidate edges. White: grid centers. Red circles: K_g (display radius only).',fill='black')
        name=f'342_阶段2_叠加复核_{iid:06d}.png'
        with open(C.HERE/name,'xb') as f: canvas.save(f,format='PNG')
        pngs.append(str(C.HERE/name))
    # Refresh audit by creating a new result; keep the stage-1 audit intact.
    rows=[]
    paths=list(C.WORK.glob('*.py'))+list(C.HERE.glob('*.py'))
    paths+=list((C.SPEC/'311_unit_instance_core_v2').glob('*.py'))+list((C.SPEC/'320_geometry_calibrated_core_v2').glob('*.py'))
    pattern=re.compile(r'\b(independent_cap|d_out|geometry|window_xyxy(?:_640)?|K_g)\b')
    for p in sorted(paths):
        source=p.read_text(encoding='utf-8-sig');ast.parse(source)
        if p.parent==C.HERE and p.name=='geom342.py': route='共享入口'
        elif p.parent==C.HERE and p.name=='342_independent.py': route='独立算法对照，按规格不调用共享入口'
        elif p.parent==C.HERE and p.name in ('342_head.py','train342.py','g_effect_342.py'): route='活动实现经geom342；详见代码行'
        elif p.parent==C.HERE: route='测试/证据/审计；历史实现仅作指定对照'
        else: route='未接共享入口的历史文件/冻结模块；不作为修复后的活动几何实现'
        for n,line in enumerate(source.splitlines(),1):
            if pattern.search(line): rows.append({'file':str(p),'line':n,'text':line.strip(),'route':route})
    metadata={'source_sha256':C.source_shas(),'已执行':{'cache_loads':32,'metric_reverse_calls':2,
        'metric_empty_P_calls':1,'legacy_geometry_calls':1,'PNG生成':6,'optimizer_steps':0,
        'AST解析文件数':len(paths)},'未执行':['阶段3至5','真实训练','GPU','heldout3236读取'],
        '未核':['主控独立复核'],'待主控':['阶段2确认；未收到确认前不得进入val548全量缓存']}
    C.write_json('342_result_geom_audit_阶段2.json',dict(metadata,entries=rows))
    C.write_json('342_result_闸门补充核对.json',dict(metadata,g_effect_reverse=reverse,
        empty_P_metric=empty,legacy_geometry_observation=legacy,overlay_png_paths=pngs,
        visual_QA={'codex_inspected_existing_PNGs':12,'observation':'六个真实development窗中，共享与独立框叠加位置一致；未见明显行列转置。K_g空窗保留空图。',
                   'scope':'单执行会话目视检查，不冒充主控独立审核'},
        limits=['归一化坐标的数值范围断言是合理范围检查；不能仅凭四个数证明其坐标单位。反向通过范围限于记录的实际错误输入。',
                '原生loss内部框格式转换保留于native box_ops；640/窗口空间换算和活动d_out走geom342。',
                '两个seed的真实训练与训练后G有效性均未执行。']))
    # Full source provenance of actually imported local Python modules, including
    # native criterion and loss adapter, without importing extra experiment code.
    imported={}
    for mod in list(__import__('sys').modules.values()):
        p=getattr(mod,'__file__',None)
        if p and p.endswith('.py'):
            path=C.Path(p)
            if path.exists() and str(C.WORK.parents[1]) in str(path): imported[str(path)]=C.sha(path)
    C.write_json('342_实际导入源码SHA256.json',dict(metadata,imported_source_sha256=imported))
    explanation=f'''# 342 阶段1/2交接索引（阶段2闸门）

阶段1已完成；阶段2 CPU工程项已验证，现等主控。阶段3至5未执行。真实训练、GPU、SSH、队列、heldout3236读取均未执行；`342_开训前审查包.md`尚未生成。

活动版本目录：{C.HERE}

由于工作目录已有geom342.py和test_342_seven.py，本次独占新建上述342目录。所有原文件保留；434项输入/源码SHA复核无变化。活动入口是本目录geom342.py、train342.py、342_head.py、g_effect_342.py；原train340/train341仅作指定对照。

| 证据 | 文件 |
|---|---|
| 阶段1总结果、七点、逐窗独立比较及反向测试 | 342_阶段1_result.json / 342_result_geom_unit.json |
| G初值合成与非均匀mass输入 | 342_result_g_effect_synthetic.json |
| 阶段2总结果、源/缓存SHA、调用计数 | 342_阶段2_result.json |
| 四臂CPU各20步详细loss与G每步10维梯度 | 342_result_四臂CPU冒烟.json |
| 四臂2步与1+恢复+1的参数/优化器/RNG逐位对比 | 342_result_续跑逐位.json |
| 原候选像素范围与正确归一化范围、同初态geometry/plan/loss_assoc差异 | 342_result_训练尺度诊断.json（完整来源与调用计数继承阶段2总结果） |
| 全文件/行几何扫描 | 342_result_geom_audit.json / 342_result_geom_audit_阶段2.json |
| G入口反向测试、空P、已有geom轴错误实测、补充叠加图 | 342_result_闸门补充核对.json |
| 新旧代码差异 | 342_geom342_与已有版本.diff / 342_g_effect_与341.diff / 342_train342_与341.diff |
| 各阶段一页中文汇总 | 342_阶段1_中文汇总.md / 342_阶段2_中文汇总.md |

已实测：七点四位小数全部复现；独立geometry最大绝对差1.7870724562385476e-6。候选原像素范围[-22.363388,83.381836]，正确归一化[-0.349428,1.302841]。四臂loss有限，四臂续跑逐位一致。G十类均有19/20步非零梯度，初始第1步为零，已原样披露。

除候选尺度外，本版本修正296明文公式的三处实现偏差：空U窗不应贡献虚假检测分母、batch分母后不再额外平均、mass目标为零仍保留+m项；另修恢复时误把G的x10学习率覆盖为普通组学习率。直接native检测损失最大差2.384185791015625e-7，NumPy质量损失最大差9.068182578175765e-7。未改配方权重、标签、阈值或窗口。请主控一并审查这些独立列明的差异。

每阶段原有6张PNG保留；另有6张`342_阶段2_叠加复核_*.png`把候选、格点与K_g叠在同一裁片。PNG中的颜色缩放只用于显示。

本轮计数：阶段1的224次残余计划、32窗独立几何与224次NumPy计划对照；阶段2共96次optimizer.step（80次指定冒烟+16次四臂续跑合同），均为工程测试；阶段2一次CPU冻结global合成前向。全量训练未执行。补充核对只调用几何/度量，不做优化。

待主控：独立核对阶段1/2的数字、diff与PNG，确认后才可进入阶段3。最终包仍须逐条填写342第六节清单；不得把本阶段交接索引当成完整审查包或训练授权。
'''
    C.write_text('342_交接索引_阶段2闸门.md',explanation)
    artifacts={str(p):{'bytes':p.stat().st_size,'sha256':C.sha(p)} for p in sorted(C.HERE.iterdir()) if p.is_file()}
    C.write_json('342_交付清单_SHA256.json',dict(metadata,artifacts=artifacts,
        note='本清单不含自身；证据文件均独占新建，未覆盖。'))
    print(json.dumps({'artifacts':len(artifacts),'gate':'STOP_AFTER_STAGE2','overlay_pngs':6,'optimizer_steps':0}))

if __name__=='__main__': main()
