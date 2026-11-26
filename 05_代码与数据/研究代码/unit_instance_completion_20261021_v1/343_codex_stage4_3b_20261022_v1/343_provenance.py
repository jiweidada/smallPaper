"""Read-only source gate for 3b. Never starts a model or relabels entries."""
import importlib,json,hashlib,time
C=importlib.import_module('343_context')
import torch,numpy as np
torch.set_num_threads(2)

def payload(e,keys):
    h=hashlib.sha256()
    for k in keys:h.update(k.encode());h.update(e[k].contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    start=time.perf_counter();ids,digest=C.development_ids()
    old=json.loads((C.S2/'342_阶段2_result.json').read_text(encoding='utf-8'))['cache_sha256']
    fields=['fullp_boxes','fullp_scores','fullp_logits_row','G_W','U_W','K_g','gt_boxes_640','crop128']
    rows=[];mismatch=[]
    for seed in (1,2):
        for n,iid in enumerate(ids,1):
            path=C.WORK/f'340_cache_seed{seed}'/f'{iid:06d}.pt'
            p339=C.WORK/f'339_cache_seed{seed}'/f'{iid:06d}.pt'
            a=torch.load(path,map_location='cpu',weights_only=True)
            b=torch.load(p339,map_location='cpu',weights_only=True)
            stored_ok=payload(a,fields)==a['sha256']
            independent_ok=payload(b,fields[:-1])==b['sha256']
            same={key:torch.equal(a[key],b[key]) for key in ('final300_boxes','final300_logits')}
            file_sha=C.sha(path);historical=str(path) in old and file_sha==old[str(path)]
            row={'seed':seed,'image_id':iid,'path':str(path),'stored_340_subset_SHA_match':stored_ok,
                'stored_339_subset_SHA_match':independent_ok,'raw_bit_equal_to_339':same,
                'current_340_file_sha256':file_sha,'historical_full_file_sha_bound':historical,
                'raw_tensor_sha256':{k:hashlib.sha256(a[k].contiguous().numpy().tobytes()).hexdigest() for k in same}}
            rows.append(row)
            if not stored_ok or not independent_ok or not all(same.values()):mismatch.append(row)
            if n%512==0 or n==3235:print(f'provenance seed={seed} {n}/3235',flush=True)
    assert not mismatch
    C.write('343_result_3b来源预检.json',{
        'status':'待主控（历史SHA对原始final300的覆盖范围不足）',
        'development_id_sha256':digest,'entries':rows,
        'summary':{'checked':len(rows),'stored_340_subset_SHA_match':sum(r['stored_340_subset_SHA_match'] for r in rows),
            'raw_equal_to_339':sum(all(r['raw_bit_equal_to_339'].values()) for r in rows),
            'historical_full_file_sha_bound':sum(r['historical_full_file_sha_bound'] for r in rows)},
        'SHA_scope':{'340_hashed_fields':fields,'339_hashed_fields':fields[:-1],
            'not_covered':['final300_boxes','final300_logits'],
            'note':'新计算的整文件/原始张量SHA用于本轮封存，不能冒称历史记录已覆盖这些张量。339逐位相等是独立缓存副本对照，不等于历史原始张量SHA。'},
        'source_sha256':C.sources()|{str(C.WORK/'build_340.py'):C.sha(C.WORK/'build_340.py'),str(C.WORK/'build_339.py'):C.sha(C.WORK/'build_339.py')},
        'elapsed_seconds':time.perf_counter()-start,
        '已执行':{'340_cache_loads':6470,'339_cache_loads':6470,'subset_SHA_checks':12940,
                   'raw_pairwise_comparisons':12940,'global_forwards':0,'optimizer_steps':0},
        '未执行':['3b重建/新标签冒烟（来源闸门待主控）','真实训练','GPU','阶段5'],
        '未核':['余下6438条原始final300的历史整文件或张量SHA来源'],
        '待主控':['343第二节1的原始来源核对：请指定覆盖final300的历史清单，或明确是否接受340有效子集SHA+339原始张量逐位对照作为本轮来源。']})
    print(json.dumps({'checked':len(rows),'historical_full_file_sha_bound':sum(r['historical_full_file_sha_bound'] for r in rows),'all_raw_equal_to339':True}))

if __name__=='__main__':main()
