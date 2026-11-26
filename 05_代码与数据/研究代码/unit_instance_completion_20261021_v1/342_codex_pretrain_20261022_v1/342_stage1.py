"""Run only stage-1 checks; exclusive-create evidence, never train."""
import importlib, sys, json, traceback, difflib, re, ast
C = importlib.import_module('342_context')
import numpy as np
import torch
import geom342 as G
import core
import g_effect_342 as E
I = importlib.import_module('342_independent')
from PIL import Image, ImageDraw
torch.set_num_threads(2)
KS = [-6.7, -.67, -.067, 0., .067, .67, 6.7]
TARGET = [.9989, .7920, .7489, .7445, .7402, .7067, .6315]

def load_entries():
    _, ids = C.inputs()
    return ids, [torch.load(C.WORK/'340_cache_seed1'/f'{i:06d}.pt',
                            map_location='cpu', weights_only=True) for i in ids]

def padded(entries):
    per = []
    for e in entries:
        w, b = e['window_xyxy_640'], e['fullp_boxes']
        keep = G.intersecting_candidates(b, w)
        per.append((G.win_norm_xyxy(b[keep], w), e['fullp_labels'][keep]))
    n = max(len(b) for b, c in per)
    boxes = torch.tensor([0., 0., 1., 1.]).expand(len(per), n, 4).clone()
    classes = torch.zeros(len(per), n, dtype=torch.long)
    valid = torch.zeros(len(per), n, dtype=torch.bool)
    for i, (b, c) in enumerate(per):
        boxes[i, :len(b)] = b
        classes[i, :len(b)] = c
        valid[i, :len(b)] = True
    return boxes, classes, valid

def vis(entries, ids, prefix, include_kg=False):
    paths = []
    for iid, e in list(zip(ids, entries))[:6]:
        w, b = e['window_xyxy_640'], e['fullp_boxes']
        keep = G.intersecting_candidates(b, w)
        loc = G.win_norm_xyxy(b[keep], w)
        pixel = G.win_norm_to_window_pixels(loc, w, (512, 512)).numpy()
        independent = I.normalized_edges(b[keep].numpy(), w.numpy()) * np.array([512]*4)
        rgb = (e['crop128'].permute(1, 2, 0).clamp(0, 1).numpy()*255).round().astype('uint8')
        crop = Image.fromarray(rgb).resize((512, 512), resample=Image.NEAREST)
        canvas = Image.new('RGB', (1584, 600), 'white')
        d = ImageDraw.Draw(canvas)
        colors = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6','#bcf60c','#fabebe']
        for panel, edges in enumerate((pixel, independent)):
            xoff, yoff = 8+panel*528, 56
            canvas.paste(crop, (xoff, yoff))
            for xy in G.grid_xy().numpy()*512:
                x, y = xy+[xoff, yoff]
                d.ellipse((x-1, y-1, x+1, y+1), fill='white')
            for edge, cls in zip(edges, e['fullp_labels'][keep].tolist()):
                # Clip only the displayed line to the panel; data boxes stay unclipped.
                drawn = np.clip(edge, 0, 511) + [xoff, yoff, xoff, yoff]
                d.rectangle(tuple(drawn), outline=colors[cls], width=1)
            d.text((xoff, 32), 'geom342 normalized -> pixels' if panel == 0 else 'independent raw-plane projection', fill='black')
        if include_kg:
            grid = e['K_g'][e['G_W']].sum(0).reshape(16,16).numpy()
            title = 'K_g sum (display scaling only)'
        else:
            cost = G.geometry_cost(loc)[0].numpy()
            grid = cost.mean(1).reshape(16,16) if cost.shape[1] else np.zeros((16,16))
            title = 'mean d_out (display scaling only)'
        value = grid / max(float(grid.max()), 1e-12)
        rgbheat = np.stack((value*255, (1-value)*160, (1-value)*255), -1).astype('uint8')
        heat = Image.fromarray(rgbheat).resize((512,512), Image.NEAREST)
        canvas.paste(heat, (1064,56))
        d.text((1064,32), title, fill='black')
        d.text((8,8), f'id={iid}; W={w.tolist()}; P={int(keep.sum())}; 16x16 grid; white cell centers', fill='black')
        d.text((8,578), 'Cross-check visualization; overlaps are retained. PNG inspection by main controller is pending.', fill='black')
        name = f'{prefix}_{iid:06d}.png'
        with open(C.HERE/name, 'xb') as f:
            canvas.save(f, format='PNG')
        paths.append(str(C.HERE/name))
    return paths

def audit():
    rows = []
    patterns = re.compile(r'\b(independent_cap|d_out|geometry|window_xyxy(?:_640)?|K_g)\b')
    sources = list(C.WORK.glob('*.py')) + list(C.HERE.glob('*.py'))
    sources += list((C.SPEC/'311_unit_instance_core_v2').glob('*.py'))
    sources += list((C.SPEC/'320_geometry_calibrated_core_v2').glob('*.py'))
    for p in sorted(set(sources)):
        raw = p.read_text(encoding='utf-8-sig')
        ast.parse(raw, filename=str(p))
        for n, line in enumerate(raw.splitlines(), 1):
            if patterns.search(line):
                if p.name == 'geom342.py' and p.parent == C.HERE:
                    route = '共享实现'
                elif p.name == '342_independent.py':
                    route = '独立对照例外（不调用共享实现）'
                elif p.parent == C.HERE:
                    route = '已走共享入口/审计测试引用，见原行'
                else:
                    route = '未走共享入口；历史源码保留，阶段2处理活动链路'
                rows.append({'path': str(p), 'line': n, 'text': line.strip(), 'route': route})
    C.write_json('342_result_geom_audit.json', {'scope':'工作目录顶层/本342版本/311_v2/320_v2的全部Python源文件',
        'entries': rows, '已执行': {'AST解析文件数': len(set(sources)), '扫描行命中数': len(rows)},
        '未执行':['历史源码改写','阶段3至5'], '未核':['历史独立脚本是否会被外部调用'],
        'source_sha256': C.source_shas()})
    return len(rows)

def main():
    ids, entries = load_entries()
    result = {'status':'未核', 'development_ids':ids, 'seed':1, 'per_window':[],
              '已执行': {'cache_loads':len(entries),'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
              '未执行':['真实训练','训练后G有效性','阶段2至5'], '未核':['主控目视核对'], '待主控':[],
              'source_sha256': C.source_shas()}
    try:
        values = {k:[] for k in KS}
        counts = []
        cx_rejected = 0
        cx_different = 0
        pix_rejected = 0
        maxerr = 0.
        for iid, e in zip(ids, entries):
            w, b = e['window_xyxy_640'], e['fullp_boxes']
            keep = G.intersecting_candidates(b, w)
            np.testing.assert_array_equal(keep.numpy(), I.intersection(b.numpy(), w.numpy()))
            loc = G.win_norm_xyxy(b[keep], w)
            geo = G.geometry_cost(loc)
            oracle = I.geometry_raw(b[keep].numpy(), w.numpy())
            error = float(np.max(np.abs(geo[0].numpy()-oracle))) if oracle.size else 0.
            assert error < 1e-5, (iid, error)
            maxerr = max(maxerr, error)
            cls = e['fullp_labels'][keep]
            counts.append(len(cls))
            valid = torch.ones(1,len(cls),dtype=torch.bool)
            mass = E.initial_mass(1)
            plan_errors = {}
            for k in KS:
                p = core.independent_cap(mass, geo, torch.full((1,len(cls)),k), cls[None], valid)
                ratio = float(p['residual'].sum()/mass.sum())
                values[k].append(ratio)
                ref = I.plan(mass[0].numpy(), oracle, cls.numpy(), np.full(10,k))
                perr = max(float(np.max(np.abs(p[n][0].numpy()-ref[n]))) if ref[n].size else 0. for n in ('T','z','residual'))
                assert perr < 1e-5, (iid,k,perr)
                plan_errors[str(k)] = perr
            try:
                bad = G.geometry_cost(G.xyxy_to_cxcywh(loc))
                different = float((bad-geo).abs().max())
                assert different > 1e-3
                cx_different += 1
            except AssertionError as ex:
                # Only the geometry rejection qualifies; an absence of a numerical
                # difference must remain a failed test.
                if str(ex) == 'expected positive xyxy edges':
                    cx_rejected += 1
                else:
                    raise
            try:
                G.geometry_cost(G.win_norm_to_window_pixels(loc,w))
            except AssertionError:
                pix_rejected += 1
            else:
                raise AssertionError(f'pixel input not rejected for {iid}')
            result['per_window'].append({'image_id':iid,'candidates':len(cls),'geometry_max_abs_diff':error,
                                         'plan_max_abs_diff':plan_errors})
        got = [float(np.mean(values[k])) for k in KS]
        result.update({'kappa':KS,'target_4dp':TARGET,'actual_full_precision':got,
            'actual_4dp':[round(x,4) for x in got], 'mean_candidates':float(np.mean(counts)),
            'independent_geometry_max_abs_diff':maxerr,
            'reverse_tests':{'cxcywh_assert_rejected':cx_rejected,'cxcywh_numerically_different':cx_different,
                             'pixel_assert_rejected':pix_rejected}})
        assert [round(x,4) for x in got] == TARGET, '主控七点未复现：按342闸门停止，不改目标数值'
        # The rectangular/original-image case uses separate x/y scales, while
        # the edge case has width=40 and height=16, not 64.
        fixtures = []
        for name, w, boxes in [
            ('non_square_orig_1280x720', [128.,192.,192.,256.], [[136.,200.,160.,224.], [120.,180.,200.,260.]]),
            ('edge_truncated_40x16', [600.,624.,640.,640.], [[610.,628.,632.,638.], [590.,620.,650.,648.]])]:
            raw = torch.tensor(boxes)
            geo = G.geometry_cost(G.win_norm_xyxy(raw,w))[0].numpy()
            ref = I.geometry_raw(raw.numpy(), np.array(w))
            np.testing.assert_allclose(geo, ref, rtol=0, atol=1e-5)
            fixtures.append({'name':name,'window':w,'max_abs_diff':float(np.abs(geo-ref).max())})
        # Malformed and transposed grids must fail, independent of box tests.
        for bad in (torch.tensor([[0.,0.,0.,1.]]), torch.tensor([[0.,0.,float('nan'),1.]])):
            try: G.geometry_cost(bad)
            except AssertionError: pass
            else: raise AssertionError('malformed box accepted')
        try: G.geometry_cost(torch.tensor([[0.,0.,1.,1.]]),G.grid_xy().flip(1))
        except AssertionError: pass
        else: raise AssertionError('transposed grid accepted')
        result['fixtures'] = fixtures
        boxes, classes, valid = padded(entries)
        synthetic = {str(k): E.measure_initial(boxes,classes,valid,torch.full((10,),k)) for k in KS}
        assert synthetic['0.0']['delta'] == 0
        for sign in (-1, 1):
            seq = [synthetic[str(sign*k)]['delta'] for k in (.067,.67,6.7)]
            assert seq[0] < seq[1] < seq[2]
        for k, approximate in ((.67,.0378),(-.67,.0475)):
            assert abs(synthetic[str(k)]['delta']-approximate) < .0002
        # Nonuniform mass proves the metric consumes caller-supplied mass.
        supplied = E.initial_mass(32) * torch.linspace(.5,2.,256)[None,None,:]
        mass_changed = E.measure(supplied,boxes,classes,valid,torch.full((10,),.67))
        assert mass_changed['delta'] != synthetic['0.67']['delta']
        C.write_json('342_result_g_effect_synthetic.json', {'synthetic_initial_mode':synthetic,
            'nonuniform_mass_delta':mass_changed['delta'],
            'source_sha256':C.source_shas(), '已执行':{'initial_metric_calls':7,'nonuniform_metric_calls':1},
            '未执行':['真实模式/训练后mass','真实训练'],'未核':['训练后G有效性'],
            'note':'合成验证不对G是否有效校准下结论'})
        result['png_paths'] = vis(entries,ids,'342_阶段1_几何')
        result['audit_rows'] = audit()
        for old, new, name in [('geom342.py','geom342.py','342_geom342_与已有版本.diff'),
                                ('g_effect_341.py','g_effect_342.py','342_g_effect_与341.diff')]:
            a, b = C.WORK/old, C.HERE/new
            C.write_text(name,''.join(difflib.unified_diff(a.read_text(encoding='utf-8').splitlines(True),
                b.read_text(encoding='utf-8').splitlines(True),fromfile=str(a),tofile=str(b))))
        result['status'] = '已验证'
        result['已执行'].update({'independent_geometry_windows':32,'numpy_plan_comparisons':224,
            'initial_residual_plan_calls':224,'g_effect_metric_calls':8,'PNG生成':6})
    except Exception as ex:
        result['status'] = '实现失败'
        result['exception'] = repr(ex)
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], flush=True)
    C.write_json('342_result_geom_unit.json',result)
    C.write_json('342_阶段1_result.json',result)
    text = f'''# 342 阶段1中文汇总\n\n状态：{result['status']}。本机CPU，真实训练未执行。\n\n独占版本目录：{C.HERE}\n\n已有 geom342.py/test_342_seven.py 保留未覆盖；本目录 geom342.py 为活动共享入口。\n\n实际七点：{result.get('actual_4dp','未执行')}。主控固定目标：{TARGET}。\n\n独立实现从640平面逐行逐列做最近点投影，不复用主实现归一化或广播轴。最大绝对差：{result.get('independent_geometry_max_abs_diff','未执行')}。\n\n反向测试：{result.get('reverse_tests','未执行')}。6张PNG路径及逐窗证据见 342_result_geom_unit.json；主控目视核对未执行。\n\nG有效性真实模式要求调用方传入训练后mass输出，本轮未执行。无训练结论。\n\n源文件SHA256、调用计数、未执行/未核详见本阶段JSON。下一步：{'按342直接进入阶段2，阶段2后等主控' if result['status']=='已验证' else '按342停止排查，不进入阶段2'}。\n'''
    C.write_text('342_阶段1_中文汇总.md',text)
    print(json.dumps({k:result.get(k) for k in ('status','actual_4dp','mean_candidates','independent_geometry_max_abs_diff','exception')},ensure_ascii=False),flush=True)
    if result['status'] != '已验证':
        sys.exit(1)

if __name__ == '__main__': main()
