"""Constructed-only bootstrap acceptance. No formal B counting policy adopted."""
import importlib,copy,json,time,hashlib
C=importlib.import_module('343_context');E=importlib.import_module('343_evaluator');B=importlib.import_module('343_bootstrap')
import numpy as np

def fixture():
    data={'images':[],'categories':[{'id':0,'name':'fixture'}],'annotations':[]};base=[];improved=[]
    for i in range(548):
        data['images'].append({'id':i,'width':640,'height':480})
        data['annotations'].append({'id':i+1,'image_id':i,'category_id':0,'bbox':[12.,12.,6.,6.],'area':36.,'iscrowd':0})
        score=.5+(i%20)/100
        fp={'image_id':i,'category_id':0,'bbox':[90.,90.,6.,6.],'score':score+.01}
        tp={'image_id':i,'category_id':0,'bbox':[12.,12.,6.,6.],'score':score}
        if i%3:base.append(copy.deepcopy(tp))
        if i%7==0:base.append(copy.deepcopy(fp))
        if i%11:improved.append(copy.deepcopy(tp))
        if i%13==0:improved.append(copy.deepcopy(fp))
    return data,base,improved

def independent_clone(data,preds,indices):
    # Rebuild a complete independent COCO dataset, including new GT/image IDs.
    images={im['id']:im for im in data['images']};anns={};dets={}
    for row in data['annotations']:anns.setdefault(row['image_id'],[]).append(row)
    for row in preds:dets.setdefault(row['image_id'],[]).append(row)
    out={'images':[],'categories':copy.deepcopy(data['categories']),'annotations':[]};p=[];aid=1
    for new_id,source in enumerate(indices,1):
        im=copy.deepcopy(images[int(source)]);im['id']=new_id;out['images'].append(im)
        for row in anns.get(int(source),[]):
            a=copy.deepcopy(row);a['id']=aid;a['image_id']=new_id;aid+=1;out['annotations'].append(a)
        for row in dets.get(int(source),[]):
            r=copy.deepcopy(row);r['image_id']=new_id;p.append(r)
    assert len(set(im['id'] for im in out['images']))==len(indices)
    return out,p

def main():
    start=time.perf_counter();data,base,improved=fixture()
    ev0,_,_,_=E.evaluate(data,base);ev1,_,_,_=E.evaluate(data,improved)
    rng=np.random.default_rng(2026102208)
    trials=[np.array([0,0,0,1,1,2,2,2]),rng.integers(0,548,548),np.arange(547,-1,-1),np.zeros(548,dtype=int)]
    comparisons=[]
    for version,ev,pred in [('base',ev0,base),('improved',ev1,improved)]:
        for t,ix in enumerate(trials):
            cached=B.sampled_evaluator(ev,ix)
            cloned,rows=independent_clone(data,pred,ix);fresh,stats,_,_=E.evaluate(cloned,rows)
            diff=float(np.max(np.abs(cached.eval['precision']-fresh.eval['precision'])))
            assert diff==0
            assert np.array_equal(cached.eval['recall'],fresh.eval['recall'])
            assert B.ap_aps(cached)==B.ap_aps(fresh)
            comparisons.append({'case':version,'trial':t,'replica_images':len(ix),'precision_max_abs_diff':diff,'AP_APs':B.ap_aps(fresh)})
    runs={f'seed{s}_{arm}':ev1 if arm=='R' else ev0 for s in (1,2) for arm in ('baseline','N','S','G','R')}
    # Counts are invented fixture values solely to test resampled summation.
    counts={name:[int((i+(1 if name.endswith('_R') else 0))%3==0) for i in range(548)] for name in runs}
    ix,output,calls=B.paired(runs,counts)
    assert all(output[f'seed1_{a}']==output[f'seed2_{a}'] for a in ('baseline','N','S','G','R'))
    for r in (0,499,999):
        assert output['seed1_R'][r][2]==sum(counts['seed1_R'][int(i)] for i in ix[r])
    with open(C.HERE/'343_bootstrap_构造1000次_共享抽样.npy','xb') as f:np.save(f,ix)
    C.write('343_bootstrap_构造1000次_逐次结果.json',output)
    C.write('343_result_阶段4bootstrap.json',{'status':'已验证（构造数据工程测试）','independent_clone_comparisons':comparisons,
        'sampling':{'seed':2026102103,'repetitions':1000,'images_per_replica':548,
            'shared_index_sha256':hashlib.sha256(ix.tobytes()).hexdigest(),'arms_and_seeds_share_one_matrix':True,
            'duplicate_image_id':'每次重采样赋予1..548的新image_id，annotation/detection ID同步分离'},
        'percentile95':B.intervals(output),'paired_difference_percentile95':B.paired_differences(output),
        'method':'复用每图匹配，逐重采样数据集调用绑定COCO accumulate重算precision/AP/APs；与独立克隆后完整evaluate比较，不是逐图AP平均。',
        'B_status':'仅构造整数求和合同；317正式B匹配/新增定义待主控，未采用口径。',
        'wall_seconds':time.perf_counter()-start,
        '已执行':{'source_COCO_evaluate_calls':2,'independent_full_clone_COCO_evaluate_calls':8,
            'independent_sampled_COCO_accumulate_calls':8,'bootstrap_COCO_accumulate_calls':calls,
            'shared_resample_rows':1000,'reported_run_replica_values':10000,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['训练后真实八run bootstrap','正式B重采样','真实训练','阶段5'],'未核':['主控独立核对'],
        'evaluator':E.evidence(),'source_sha256':C.sources()})

if __name__=='__main__':main()
