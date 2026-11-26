"""Stage-3 input audit. No global forward, optimizer, or cache modification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import torch
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
SPEC=WORK.parents[1]/'持续寻找研究主线_20261020_v2'
PREVIOUS=WORK/'342_codex_pretrain_20261022_v1'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()

def write(name,obj):
    with open(HERE/name,'x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    sources=[WORK/'stage1_cache_v3.py',WORK/'stage1_cache_v4.py',WORK/'build_340.py',
             WORK/'build_341_val.py',WORK/'uic_m0.py',PREVIOUS/'geom342.py',
             PREVIOUS/'train342.py',PREVIOUS/'342_head.py',PREVIOUS/'g_effect_342.py',
             SPEC/'296_A_单位实例残余补全_具体实现提案.txt',
             SPEC/'317_A_单位解释残余补全_四臂有限验证方案_v3.txt',
             SPEC/'342_总指令_修几何链路至开训前审查包.txt',Path(__file__)]
    manifest={str(p):sha(p) for p in sources}
    write('342_放行与更正记录.json',{
        'authorization':'用户明确：阶段1、2独立核过，可以放行进入阶段3；阶段3完成后仍须停下等主控。',
        'corrections':[
            {'item':'候选范围适用域','corrected':'旧汇报[-0.349,1.303]仅为seed1前16窗，2168个坐标标量（非2168个框）；不应写成32窗或一般范围。',
             'user_reported':'用户独立报告seed1前32窗[-0.857,1.801]；seed2前16/32窗最大3.016。本记录尚未重新运行这些范围统计。'},
            {'item':'恢复学习率表述','corrected':'旧load_checkpoint临时把所有组LR设为首组；run_steps在每步开始重设，因而不会改变续跑第一步的实际更新。新写法为状态恢复清理，不计作修复实际训练更新错误。'}],
        'independence_scope':'用户重算了seed1前32窗几何与损失；用户未重跑四臂冒烟/续跑数字。',
        'stage4_reminder':'阶段4新增头输出为窗内归一化cxcywh；反变换必须经geom342并有独立/反向测试，两seed同样核对。本阶段不实现或运行阶段4。',
        'source_sha256':manifest,'已执行':{'记录用户放行与更正':1},
        '未执行':['真实训练','GPU','SSH','队列','heldout3236读取','阶段4至5'],
        '未核':['阶段3全量缓存']})
    ann=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
    idx=json.loads((SPEC/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
    expected=next(a['expected_SHA256_from_existing_audit'] for a in idx['annotations'] if a['split']=='val')
    actual=sha(ann)
    assert actual==expected,(actual,expected)
    meta=json.loads(ann.read_text(encoding='utf-8'))
    assert len(meta['images'])==548
    rows=[];mismatch=[];non_topclass_candidates=0
    oldfiles=sorted((WORK/'341_cache_val_seed1').glob('*.pt'))
    assert len(oldfiles)==32
    for path in oldfiles:
        e=torch.load(path,map_location='cpu',weights_only=True)
        owner=e['m0_cand_owner'];matched=(owner>=0).nonzero().flatten()
        top=e['fullp_logits_row'].argmax(-1)
        non_topclass_candidates+=int((top!=e['fullp_labels']).sum())
        wrong=[]
        for j in matched.tolist():
            g=int(owner[j]);pc=int(e['fullp_labels'][j]);gc=int(e['gt_labels'][g])
            if pc!=gc:
                wrong.append({'candidate_index':j,'query_index':int(e['fullp_query_index'][j]),
                              'candidate_label':pc,'logits_argmax_label':int(top[j]),
                              'owner_gt_index':g,'owner_gt_label':gc,
                              'annotation_id':int(e['gt_annotation_ids'][g])})
        row={'image_id':int(e['image_id']),'matched':len(matched),'wrong_class_owner_count':len(wrong),
             'wrong_class_owners':wrong,'cache_path':str(path),'cache_file_sha256':sha(path)}
        rows.append(row)
        if wrong:mismatch.append(row)
    report={'val_annotation_path':str(ann),'val_annotation_sha256':actual,'val_images':548,
            'existing_val_seed1_caches_checked':32,'non_argmax_fullP_candidates':non_topclass_candidates,
            'wrong_class_owner_total':sum(r['wrong_class_owner_count'] for r in rows),
            'images_with_wrong_class_owners':len(mismatch),'rows':rows,
            'source_sha256':manifest,'已执行':{'annotation_SHA_checks':1,'val_annotation_parses':1,'cache_loads':32,
                'global_forwards':0,'optimizer_steps':0},
            '未执行':['新建val548缓存','阶段4至5','真实训练','GPU','heldout3236读取'],
            '未核':['M0精确lex三级合同'],
            'basis':'317 §2要求每个fullP candidate实例按其自身预测类建立同类合法M0边。此处只读既有缓存核查owner类别，不改标签。'}
    write('342_result_阶段3输入预检.json',report)
    print(json.dumps({k:report[k] for k in ('val_images','existing_val_seed1_caches_checked','non_argmax_fullP_candidates','wrong_class_owner_total','images_with_wrong_class_owners')},ensure_ascii=False))

if __name__=='__main__':main()
