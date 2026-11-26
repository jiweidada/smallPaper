"""Inspect the recorded receipts without constructing models or rerunning tests."""
import importlib,json
C=importlib.import_module('342_stage5_context')
paths=[C.S2/n for n in ('342_result_geom_unit.json','342_result_g_effect_synthetic.json','342_result_闸门补充核对.json','342_阶段2_result.json','342_result_四臂CPU冒烟.json')]
paths += [C.S3/n for n in ('342_阶段3_result.json','342_result_val548独立验收.json','342_val548_逐条SHA清单.json')]
paths += [C.S343/n for n in ('343_阶段4_result_v2.json','343_result_阶段4映射与融合_v3.json','343_result_阶段4自然val配对bootstrap.json','343_result_阶段4方向规则.json','343_result_阶段4主B阳性构造_v2.json')]
paths += [C.WORK/n for n in ('340_result.json','340_config.json','335_result.json','332_result.json') if (C.WORK/n).is_file()]

def prune(x,depth=0):
    if isinstance(x,dict):
        if depth>=2:return {'keys':list(x)[:24]}
        return {k:prune(v,depth+1) for k,v in x.items() if k not in ('source_sha256','per_image','entries','rows','log','audit','png_paths','evidence')}
    if isinstance(x,list):return {'length':len(x),'first':prune(x[0],depth+1) if x else None}
    return x

out={}
for p in paths:
    d=C.read(p);out[str(p)]={'keys':list(d),'summary':prune(d)}
    print(json.dumps({'file':p.name,'keys':list(d),'summary':prune(d)},ensure_ascii=False),flush=True)
C.write('342_阶段5既有证据结构登记.json',out)
