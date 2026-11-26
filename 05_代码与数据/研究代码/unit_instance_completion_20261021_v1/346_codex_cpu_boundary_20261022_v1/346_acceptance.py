"""Independent full-order boundary oracle; stop at first failed acceptance."""
import importlib,json,math,os,subprocess,sys,traceback,time
from pathlib import Path
C=importlib.import_module('346_context')
import torch
torch.set_num_threads(2)
FIX=importlib.import_module('346_fixture')
RESULTS=[]

def load_checkpoint(reference):
    assert C.sha(reference['path'])==reference['sha256']
    r=C.read(reference['path']);assert C.sha(r['checkpoint_path'])==r['checkpoint_sha256']
    ck=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False)
    return r,ck

def cross_process_case(arm,seed,case,reference):
    run_dir=C.RUN_ROOT/f'346_{arm}_seed{seed}_{case}'
    job_name=f'346_job_{arm}_seed{seed}_{case}.json'
    C.write(job_name,{'arm':arm,'seed':seed,'case':case,'run_dir':str(run_dir),'resume':reference})
    command=[sys.executable,'-B',str(C.HERE/'346_worker.py'),str(C.HERE/job_name)]
    child=subprocess.run(command,cwd=str(C.HERE),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         encoding='utf-8',errors='strict',env=dict(os.environ),check=False)
    C.text(f'346_stdout_{arm}_seed{seed}_{case}.txt',child.stdout)
    C.text(f'346_stderr_{arm}_seed{seed}_{case}.txt',child.stderr)
    assert child.returncode==0,(arm,seed,case,child.returncode,child.stderr[-6000:])
    result=C.read(run_dir/'346_run_result.json');RESULTS.append(result)
    return run_dir,result

def check_case(arm,seed,case,reference,run_dir,result):
    spec=C.CASES[case]
    cfg,order,curve,original_binding=C.E.validate_bindings(arm,seed,'formal')
    rows=C.schedule(order,cfg['batch'])
    assert len(rows)==1218
    expected=rows[spec['start']:spec['end']]
    logs=C.read(run_dir/'346_steps.json')
    assert len(logs)==spec['maximum_actual_updates']
    for observed,gold in zip(logs,expected):
        for key in ('step','epoch','batch_index','position_in_epoch','image_ids'):
            assert observed[key]==gold[key],(arm,seed,case,key,observed[key],gold[key])
        assert all(math.isfinite(observed[k]) for k in ('loss_det','loss_mass','loss_assoc','loss_total','grad_norm_before_clip'))
        expected_lr=[curve['non_kappa'][gold['step']]]*2
        if arm=='G':expected_lr.append(curve['G_kappa'][gold['step']])
        assert observed['param_group_lr']==expected_lr
        if arm=='G':
            assert len(observed['kappa_before'])==len(observed['kappa_after'])==10
            gradient=observed['kappa_grad_after_clip']
            assert gradient is None or (len(gradient)==10 and all(math.isfinite(v) for v in gradient))
    assert [len(r['image_ids']) for r in logs]==spec['expected_batch_sizes']
    assert [result['epoch'],result['batch_index'],result['position_in_epoch']]==spec['end_cursor']
    assert result['step']==spec['end']
    counts=result['counts_this_process']
    exposures=sum(len(r['image_ids']) for r in expected)
    assert counts['head_forwards']==counts['backward_calls']==counts['optimizer_steps']==spec['maximum_actual_updates']
    assert counts['cache_loads']==exposures and counts['native_loss_calls']==2*exposures
    assert counts['global_forwards']==counts['real_training_steps']==0
    assert result['actual_step_log_records_since_fixture']==spec['maximum_actual_updates']
    receipt,ck=load_checkpoint(result['last_checkpoint'])
    fixture_receipt,initial=load_checkpoint(reference)
    assert ck['format']==initial['format']==C.KIND
    assert ck['synthetic_origin_step']==spec['start'] and len(initial['step_logs'])==0
    assert ck['actual_optimizer_updates_since_fixture']==len(logs)
    assert not initial['optimizer']['state']
    assert C.E.state_sha(ck['model'])!=initial['model_state_sha256'],'no parameter update occurred'
    optimizer_steps=[float(st['step']) for st in ck['optimizer']['state'].values() if 'step' in st]
    assert optimizer_steps and max(optimizer_steps)==spec['maximum_actual_updates']
    assert all(0<n<=spec['maximum_actual_updates'] for n in optimizer_steps)
    expected_checkpoint_steps=[203,204] if case=='epoch_201_204' else [1218]
    observed_checkpoint_steps=[]
    for ref in result['checkpoints_this_process']:
        rr,cc=load_checkpoint(ref);observed_checkpoint_steps.append(rr['step'])
        assert len(cc['step_logs'])==cc['step']-spec['start']
        assert cc['shuffle_sha256']==original_binding['shuffle_sha256']
        assert cc['param_group_lr']==[g['lr'] for g in cc['optimizer']['param_groups']]
        assert set(cc['rng'])=={'torch','numpy','python'}
    assert observed_checkpoint_steps==expected_checkpoint_steps
    assert len(ck['epoch_times'])==1
    assert ck['epoch_times'][0]['actual_batches_since_fixture']==2
    assert ck['epoch_times'][0]['actual_windows_since_fixture']==19
    assert ck['epoch_times'][0]['cursor_batches_at_epoch_end']==203
    assert ck['epoch_times'][0]['order_epoch_windows']==3235
    guards=C.read(run_dir/'346_worker_guards.json')['rows']
    row={'status':'已验证','arm':arm,'seed':seed,'case':case,
         'oracle':'full materialized order expanded by nested epoch/offset loops; no execution cursor helper',
         'synthetic_origin_step':spec['start'],'actual_optimizer_steps':counts['optimizer_steps'],
         'global_cursor_end':result['step'],'end_cursor':spec['end_cursor'],
         'batch_sizes':[len(r['image_ids']) for r in logs],
         'image_ids':[r['image_ids'] for r in logs],
         'batch_tensor_sha256':[r['batch_tensor_sha256'] for r in logs],
         'logs':logs,'checkpoints':result['checkpoints_this_process'],
         'optimizer_state_step_max':max(optimizer_steps),'guard_results':guards,
         'run_dir':str(run_dir),'fixture_receipt':reference,
         'counts_this_process':counts,'source_sha256':C.source_sha(),
         'unexecuted_prefix_steps':spec['start'],'real_training_steps':0}
    C.write(f'346_result_{arm}_seed{seed}_{case}.json',row)
    print(f'346 PASS {arm} seed{seed} {case}: batch_sizes={row["batch_sizes"]}; actual_updates={counts["optimizer_steps"]}; global_cursor={result["step"]}',flush=True)
    return row

def additional_negatives(records):
    negative=[]
    for case in C.CASES:
        target=C.HERE/'346_original345_rejection'/f'346_{case}'
        before=dict(C.T.INPUT_COUNTS)
        try:
            C.E.execute('N',1,target,'cpu',2,_resume=records[f'N_seed1/{case}'])
        except C.E.GuardError as ex:
            assert ex.code=='resume_receipt_binding',ex.code
            assert before==dict(C.T.INPUT_COUNTS)
            negative.append({'case':'original345_rejects_'+case,'guard':ex.code,
                             'models_constructed':1,'implicit_common_init_reference_models':1,
                             'head_forwards':0,'optimizer_steps':0,'run_dir':str(target)})
        else:raise AssertionError('original345 accepted an acceptance-only checkpoint')
    cfg,_,_,_=C.E.validate_bindings('N',1,'formal')
    bundle=C.read(C.S345/'345_B_冻结入口绑定.json')
    mini=C.read(bundle['shuffle']['1']['mini']['path'])
    try:C.E.validate_order(mini,'formal',cfg,C.T.load_order())
    except C.E.GuardError as ex:
        assert ex.code=='order_kind_mismatch'
        negative.append({'case':'formal_rejects_mini','guard':ex.code,'head_forwards':0,'optimizer_steps':0})
    else:raise AssertionError('formal loader accepted mini order')
    return negative

def main():
    started=time.perf_counter()
    try:
        C.verify_sources()
        records=FIX.build_all()
        C.RUN_ROOT.mkdir(exist_ok=False)
        rows=[];same_seed={}
        for seed in (1,2):
            for arm in 'NSGR':
                for case in C.CASES:
                    ref=records[f'{arm}_seed{seed}/{case}']
                    run_dir,result=cross_process_case(arm,seed,case,ref)
                    row=check_case(arm,seed,case,ref,run_dir,result)
                    pair=(row['image_ids'],row['batch_tensor_sha256'])
                    key=(seed,case)
                    if key in same_seed:assert pair==same_seed[key],('same-seed arm batch mismatch',seed,case,arm)
                    else:same_seed[key]=pair
                    rows.append(row)
        more=additional_negatives(records)
        guards=[g for r in rows for g in r['guard_results']]+more
        assert len(guards)==27
        counts={k:sum(r['counts_this_process'][k] for r in RESULTS) for k in RESULTS[0]['counts_this_process']}
        assert counts['head_forwards']==counts['backward_calls']==counts['optimizer_steps']==40
        assert counts['cache_loads']==432 and counts['native_loss_calls']==864
        C.write('346_边界验收_result.json',{'status':'两seed四臂16组真实顺序边界短验收全部通过；待主控',
            'case_rows':rows,'negative_rows':guards,'counts':counts,'boundary_workers':16,
            'fixture_models_constructed':16,'implicit_fixture_reference_models':16,
            'execution_models_constructed':16,'implicit_execution_reference_models':16,
            'negative_old_entry_models_constructed':2,'implicit_negative_reference_models':2,
            'actual_models_have_only_2_or_3_updates':True,'same_seed_arms_batch_SHA_equal':True,
            'wall_seconds':time.perf_counter()-started,
            '已执行':{'positive_boundary_cases':16,'negative_cases':27,**counts},
            '未执行':['真实训练','完整1218步run','val/G Delta/bootstrap','GPU','SSH','heldout图像或缓存'],
            '未核':['主控对本轮结果的独立复核','完整训练耗时及训练后效果'],
            'source_sha256':C.source_sha()})
        print('346 COMPLETE: boundary cases=16, rejected negatives=27, CPU optimizer updates=40, cache loads=432, real training=0',flush=True)
    except Exception as ex:
        C.write('346_实现失败_停止.json',{'status':'实现失败，停止，未启动真实训练',
            'exception':repr(ex),'traceback':traceback.format_exc(),
            'completed_worker_counts':{k:sum(r['counts_this_process'][k] for r in RESULTS) for k in RESULTS[0]['counts_this_process']} if RESULTS else {},
            'completed_workers':len(RESULTS),'real_training_steps':0,'source_sha256':C.source_sha()})
        raise

if __name__=='__main__':main()
