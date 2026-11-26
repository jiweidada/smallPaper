"""Same bounded 343 test, now exercising train344.load_entry and the #1466 batch."""
import importlib,difflib
C=importlib.import_module('344_context')
old=(C.S343/'343_smoke_development_v2.py').read_text(encoding='utf-8')
prefix=old[:old.index('def main():')]
prefix=prefix.replace("'343_context'","'344_context'").replace("T=importlib.import_module('train342');D=importlib.import_module('343_development')","T=importlib.import_module('train344')")
prefix=prefix.replace("f'343_CPU_smoke_resume_seed{seed}_{arm}.pt'","f'344_CPU_smoke_resume_seed{seed}_{arm}.pt'")
main='''def main():
    assert len(json.loads((C.HERE/'344_development6470_逐条SHA清单.json').read_text(encoding='utf-8'))['entries'])==6470
    ids=C.development_ids()[:32];results={};resumes={};comparisons=[];special=[]
    previous=json.loads((C.S343/'343_result_3b新标签四臂冒烟与续跑.json').read_text(encoding='utf-8'))['arms']
    try:
        for seed in (1,2):
            entries=[T.load_entry(seed,i) for i in ids];COUNTS['new_cache_loads']+=32
            assert [e['image_id'] for e in entries]==ids
            assert all(not e['fullp_nonpositive_mask'].any() for e in entries)
            batches=[T.collate(entries[:16]),T.collate(entries[16:])];results[str(seed)]={};resumes[str(seed)]={}
            for arm in 'NSGR':
                results[str(seed)][arm]=smoke(arm,seed,batches);resumes[str(seed)][arm]=resume(arm,seed,batches)
                now=results[str(seed)][arm];prior=previous[str(seed)][arm]
                # New entry SHA binding and all-false masks cannot alter these 64 windows.
                assert equal(now['log'],prior['log'])
                comparisons.append({'seed':seed,'arm':arm,'all20_step_logs_bit_equal_to_343':True,
                    'previous_path':str(C.S343/'343_result_3b新标签四臂冒烟与续跑.json')})
        extra=[T.load_entry(2,i) for i in ids[:15]]+[T.load_entry(2,1466)];COUNTS['new_cache_loads']+=16
        b=T.collate(extra);e=extra[-1]
        keep=G.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],e['fullp_nonpositive_mask'])
        assert not keep[259] and e['m0_cand_owner'][259]==-1 and e['n_G_W']==e['n_U_W']==0
        assert torch.isfinite(b['P_boxes']).all() and (b['P_boxes'][b['P_valid']][:,2:]>b['P_boxes'][b['P_valid']][:,:2]).all()
        for arm in 'NSGR':
            rng(2);m=T.make_common_init(arm,2);m.train()
            with torch.no_grad():loss,out=compute(arm,m,b)
            assert all(bool(torch.isfinite(v)) for v in loss.values())
            assert torch.isfinite(out['pred_boxes']).all() and torch.isfinite(out['pred_logits']).all()
            special.append({'arm':arm,'seed':2,'batch_image_ids':[x['image_id'] for x in extra],
                'bad_fullp_index':259,'bad_query_index':62,'excluded_from_P_W':True,'all_loss_finite':True,
                'losses':{k:float(v) for k,v in loss.items()},'P_W_1466':int(keep.sum()),'optimizer_steps':0})
        assert COUNTS['head_forwards']==198 and COUNTS['optimizer_steps']==192 and COUNTS['native_loss_adapter_calls']==6336
        C.write('344_result_train344四臂CPU冒烟与续跑.json',{'status':'已验证（有界CPU工程验证）',
            'same_development_ids_per_seed':ids,'arms':results,'resume':resumes,'previous343_comparisons':comparisons,
            'batch_including_1466':special,'input_binding_calls':T.INPUT_COUNTS,
            'global_contract':'本次未构造global模型；优化参数仅来自head。global零grad/state合同继承阶段2，本次不冒称重测。',
            '已执行':COUNTS,'未执行':['真实训练','完整1218步','global前向','GPU','阶段5'],
            '未核':['主控独立重跑','完整训练后G效果'],'source_sha256':C.sources()})
        print(json.dumps({'status':'PASS','counts':COUNTS,'input_counts':T.INPUT_COUNTS},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.write('344_result_train344冒烟_实现失败.json',{'status':'实现失败','exception':repr(ex),'traceback':traceback.format_exc(),
            '已执行':COUNTS,'未执行':['真实训练','GPU','阶段5'],'未核':['train344冒烟与续跑'],'source_sha256':C.sources()});raise

if __name__=='__main__':main()
'''
new=prefix+main
C.text('344_smoke_train344.py',new)
C.text('344_smoke_vs343.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='343_smoke_development_v2.py',tofile='344_smoke_train344.py')))
