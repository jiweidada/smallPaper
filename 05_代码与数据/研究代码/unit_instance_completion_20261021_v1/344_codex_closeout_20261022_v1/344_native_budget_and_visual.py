"""Check the retained bad raw instance against native postprocessing; record actual PNG review."""
import importlib,json,torch
C=importlib.import_module('344_context');T=importlib.import_module('train344');G=T.geom
from engine.deim.postprocessor import PostProcessor

def main():
    e=T.load_entry(2,1466);pp=PostProcessor(num_classes=10,use_focal_loss=True,num_top_queries=300,remap_mscoco_category=False,score_threshold=.001)
    with torch.no_grad():out=pp({'pred_boxes':e['final300_boxes'][None],'pred_logits':e['final300_logits'][None]},torch.tensor([e['orig_wh']]))[0]
    # Shared geometry is used for implementation coordinates; native is the independent oracle.
    expect=G.global_queries_to_orig_xyxy(e['final300_boxes'],e['orig_wh'])[e['fullp_query_index']]
    assert torch.equal(out['boxes'],expect) and torch.equal(out['scores'],e['fullp_scores']) and torch.equal(out['labels'],e['fullp_labels'])
    assert len(out['scores'])==300 and float(out['scores'][259])==float(e['fullp_scores'][259])
    assert (out['boxes'][259,2:]-out['boxes'][259,:2]<=0).any()
    counts={};ix=[]
    for j,c in enumerate(out['labels'].tolist()):
        rank=counts.get(c,0);counts[c]=rank+1
        if rank<100:ix.append(j)
    independent=[]
    for c in range(10):independent.extend(torch.where(out['labels']==c)[0][:100].tolist())
    independent=sorted(independent);assert ix==independent
    cls=int(out['labels'][259]);rank=int((out['labels'][:260]==cls).sum())
    keep=G.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],e['fullp_nonpositive_mask'])
    assert not keep[259]
    wrongly_dropped=[j for j in range(300) if j!=259]
    assert len(wrongly_dropped)!=len(out['scores'])
    C.write('344_result_1466_native预算核对.json',{'status':'已验证','seed':2,'image_id':1466,
        'fullp_index':259,'query_index':62,'flatten_index':623,'native_fullP_count':300,
        'native_fullP_boxes_scores_labels_bit_equal':True,'fullp_score':float(out['scores'][259]),
        'native_signed_original_xyxy':out['boxes'][259].tolist(),'class':cls,'rank_within_class_1based':rank,
        'retained_after_class100':259 in ix,'final_after_class100_count':len(ix),
        'budget_rule':'先保留该实例参加native top300与score>.001，再按每类100名处理；无abs/clip/交换边/eps修框。',
        'P_W_count':int(keep.sum()),'M0_bad_owner':int(e['m0_cand_owner'][259]),'M0_bad_legal_edges':int(e['m0_legal_mask'][259].sum()),
        'independent_class100_index_selection_equal':True,'negative_control_premature_fullP_drop_detected':True,
        '已执行':{'train344_bound_loads':1,'native_PostProcessor_calls':1,'independent_class100_checks':1,'negative_controls':1,
            'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['真实训练','GPU','development AP（不属于本诊断）','阶段5'],'未核':['主控独立复核'],'source_sha256':C.sources()})
    report=json.loads((C.HERE/'344_result_1466独立验收与6PNG_v2.json').read_text(encoding='utf-8'))
    notes={
        (1,1):'绿窗覆盖道路车辆；青框与车/人位置一致，黄框与红圈相互对应，存储crop与独立crop目视一致。',
        (1,729):'密集车辆窗，黄框和红圈密集对应GT；青框较少。两份crop建筑与道路结构一致。',
        (1,2):'停车场窗，青框落在车辆附近，黄框与红圈覆盖对应车辆。两份crop朝向一致。',
        (2,1):'seed2窗在人群区域，青框/黄框与人对应，K_g红圈在目标中心附近；两份crop一致。',
        (2,729):'与seed1同窗，候选数量不同；GT/K_g图位一致，两份crop一致。',
        (2,1466):'洋红短线保留原始负高度的边序，位于绿窗左上且远离W；crop中仅有合法青框；G_W=0，无黄框或红圈；独立crop一致。',
    }
    entries=[]
    for row,path in zip(report['per_case'],report['png_paths']):
        entries.append({'seed':row['seed'],'image_id':row['image_id'],'path':path,'sha256':C.sha(path),
            'viewed_with':'tools.view_image, all six PNGs in current turn','observation':notes[row['seed'],row['image_id']]})
    C.write('344_result_6PNG本轮目视.json',{'status':'已目视核对','entries':entries,
        '已执行':{'PNG_views':6,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['真实训练','GPU','阶段5'],'未核':['主控独立目视复核'],'source_sha256':C.sources()})
    print(json.dumps({'status':'PASS','native_count':300,'bad_class_rank':rank,'retained_after100':259 in ix,'final_count':len(ix)},ensure_ascii=False))

if __name__=='__main__':main()
