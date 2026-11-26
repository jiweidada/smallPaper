"""Independent launch/cursor audit; stop permanently at the first failed B check."""
import importlib, json, os, struct, subprocess, sys, traceback
from pathlib import Path
C = importlib.import_module('345_context')
import numpy as np
import torch
import train345 as E

ROOT = C.HERE/'345_B_CPU_acceptance_runs'
RUN_RESULTS = []


def bit_equal(a, b, path='root'):
    """No shared model math: recursively compare raw bytes, shapes and metadata."""
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape, path
        assert a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes(), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(), path
    elif isinstance(a, dict):
        assert isinstance(b,dict) and a.keys()==b.keys(), path
        for key in a:
            bit_equal(a[key],b[key],f'{path}.{key}')
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b), path
        for i,(x,y) in enumerate(zip(a,b)):
            bit_equal(x,y,f'{path}[{i}]')
    elif isinstance(a,float):
        assert isinstance(b,float) and struct.pack('!d',a)==struct.pack('!d',b), path
    else:
        assert type(a)==type(b) and a==b, (path,a,b)


def load_last(result):
    receipt_ref = result['last_checkpoint']
    assert C.sha(receipt_ref['path'])==receipt_ref['sha256']
    receipt = C.read(receipt_ref['path'])
    assert C.sha(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
    return torch.load(receipt['checkpoint_path'],map_location='cpu',weights_only=False)


def launch(arm, seed, label, endpoint, resume=None, formal=False):
    run_dir = ROOT/f'345_{arm}_seed{seed}_{label}'
    if formal:
        command = [sys.executable,'-B',str(C.HERE/'train345.py'),'--arm',arm,'--seed',str(seed),
                   '--run_dir',str(run_dir),'--device','cpu','--max_steps',str(endpoint)]
    else:
        job_name = f'345_B_job_{arm}_seed{seed}_{label}.json'
        C.write(job_name, {'arm':arm,'seed':seed,'run_dir':str(run_dir),'max_steps':endpoint,
                          'mode':'mini_acceptance','resume':resume})
        command = [sys.executable,'-B',str(C.HERE/'345_B_worker.py'),str(C.HERE/job_name)]
    child = subprocess.run(command,cwd=str(C.HERE),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           encoding='utf-8',errors='strict',env=dict(os.environ),check=False)
    C.text(f'345_B_stdout_{arm}_seed{seed}_{label}.txt',child.stdout)
    C.text(f'345_B_stderr_{arm}_seed{seed}_{label}.txt',child.stderr)
    assert child.returncode==0, (arm,seed,label,child.returncode,child.stderr[-6000:])
    result = C.read(run_dir/'345_run_result.json')
    RUN_RESULTS.append(result)
    print(f'B {arm} seed{seed} {label}: endpoint={result["step"]} local_optimizer_steps={result["counts_this_process"]["optimizer_steps"]}',flush=True)
    return run_dir,result


def expect_guard(label, expected, call):
    try:
        call()
    except E.GuardError as ex:
        assert ex.code==expected, (label,expected,ex.code)
        return {'test':label,'status':'已验证','expected_guard':expected,'observed':str(ex),
                'optimizer_steps':0,'head_forwards':0}
    raise AssertionError(('negative input was accepted',label))


def negative_tests():
    directory = C.HERE/'345_B_negative_fixtures'
    directory.mkdir(exist_ok=False)
    bundle = C.read(C.HERE/'345_B_冻结入口绑定.json')
    cfg_record = bundle['configs']['N_seed1']
    cfg = C.read(cfg_record['path'])['config']
    rows = []
    for label, key, source, expected in (
        ('configuration_SHA','config',cfg_record['path'],'config_SHA_mismatch'),
        ('manifest_SHA','manifest',cfg['development_manifest'],'manifest_SHA_mismatch'),
        ('shuffle_bytes','shuffle',bundle['shuffle']['1']['formal']['path'],'shuffle_SHA_mismatch')):
        target=directory/f'345_{label}.json'
        with open(target,'xb') as f:
            f.write(Path(source).read_bytes()+b' ')
        rows.append(expect_guard(label,expected,lambda k=key,p=target:E.validate_bindings('N',1,_test_paths={k:p})))
    existing=directory/'345_existing_run'
    existing.mkdir(exist_ok=False)
    rows.append(expect_guard('existing_run','run_directory_exists',lambda:E.execute('N',1,existing,'cpu',2)))
    rows.append(expect_guard('step_1219','max_steps_budget',lambda:E.execute('N',1,directory/'345_never_created','cpu',1219)))
    assert not (directory/'345_never_created').exists()
    mini=C.read(bundle['shuffle']['1']['mini']['path'])
    rows.append(expect_guard('formal_rejects_mini_order','order_kind_mismatch',
                            lambda:E.validate_order(mini,'formal',cfg,E.T.load_order())))
    parser_child=subprocess.run([sys.executable,'-B',str(C.HERE/'train345.py'),'--arm','N','--seed','1',
                '--run_dir',str(directory/'345_no_lr_override'),'--device','cpu','--max_steps','2','--lr','0.001'],
                cwd=str(C.HERE),stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',check=False)
    assert parser_child.returncode==2 and 'unrecognized arguments: --lr' in parser_child.stderr
    assert not (directory/'345_no_lr_override').exists()
    rows.append({'test':'unlisted_cli_hyperparameter','status':'已验证','exit_code':2,
                 'stderr':parser_child.stderr,'optimizer_steps':0,'head_forwards':0})
    C.write('345_result_B_五种篡改与附加守卫.json',{'status':'已验证','rows':rows,
            'mutation_scope':'new local copies only; original source/cache/config files unchanged',
            '已执行':{'negative_cases':len(rows),'optimizer_steps':0,'head_forwards':0},'source_sha256':C.sources()})
    print('B negative guards: 5 required + mini-kind + CLI override all rejected',flush=True)


def audit_orders():
    bundle=C.read(C.HERE/'345_B_冻结入口绑定.json')
    rows=[]
    ids=E.T.load_order()
    for seed in (1,2):
        cfg=C.read(bundle['configs'][f'N_seed{seed}']['path'])['config']
        formal=C.read(bundle['shuffle'][str(seed)]['formal']['path'])
        mini=C.read(bundle['shuffle'][str(seed)]['mini']['path'])
        for epoch,row in enumerate(formal['epochs']):
            # Independent API for the same specified generator: argsort is not used.
            generator=torch.Generator(device='cpu');generator.manual_seed(cfg['head_seed']+epoch)
            permutation=torch.randperm(3235,generator=generator).numpy()
            independent=np.asarray(ids,dtype=np.int64)[permutation].tolist()
            assert row==independent and sorted(row)==sorted(ids) and len(set(row))==3235
            batches=[row[start:start+16] for start in range(0,len(row),16)]
            assert len(batches)==203 and len(batches[-1])==3 and sum(map(len,batches))==3235
            assert mini['epochs'][epoch]==row[:32]
        rows.append({'seed':seed,'epochs':6,'all_permutations_exact':True,'steps_per_epoch':203,
                     'last_batch_size':3,'formal_sha256':bundle['shuffle'][str(seed)]['formal']['sha256'],
                     'mini_sha256':bundle['shuffle'][str(seed)]['mini']['sha256']})
    C.write('345_result_B_物化顺序独立验收.json',{'status':'已验证','rows':rows,'optimizer_steps':0,'source_sha256':C.sources()})


def main():
    try:
        audit_orders()
        negative_tests()
        ROOT.mkdir(exist_ok=False)
        resume_rows=[]
        for seed in (1,2):
            for arm in 'NSGR':
                full_dir,full=launch(arm,seed,'mini_continuous6',6)
                part_dir,part=launch(arm,seed,'mini_first3',3)
                assert (part['step'],part['epoch'],part['batch_index'],part['position_in_epoch'])==(3,1,1,16)
                resumed_dir,resumed=launch(arm,seed,'mini_newprocess_resume3',6,resume=part['last_checkpoint'])
                a,b=load_last(full),load_last(resumed)
                fields=('model','optimizer','rng','param_group_lr','step','epoch','batch_index',
                        'position_in_epoch','shuffle_sha256','initial_state_sha256','model_state_sha256','binding','step_logs')
                for field in fields:
                    bit_equal(a[field],b[field],field)
                assert (full_dir/'345_steps.json').read_bytes()==(resumed_dir/'345_steps.json').read_bytes()
                assert (resumed['step'],resumed['epoch'],resumed['batch_index'])==(6,3,0)
                assert [C.read(r['path'])['step'] for r in full['checkpoints_this_process']]==[2,4,6]
                assert [C.read(r['path'])['step'] for r in part['checkpoints_this_process']]==[2,3]
                assert [C.read(r['path'])['step'] for r in resumed['checkpoints_this_process']]==[4,6]
                row={'arm':arm,'seed':seed,'status':'已验证','comparison':'strict recursive bytes + exact step-log file bytes',
                     'compared_fields':list(fields),'max_model_difference':0.0,'max_optimizer_difference':0.0,
                     'rng_bit_identical':True,'step_log_bytes_identical':True,'resume_from':{'step':3,'epoch':1,'batch_index':1,'position':16},
                     'final_cursor':{'step':6,'epoch':3,'batch_index':0},
                     'run_dirs':[str(full_dir),str(part_dir),str(resumed_dir)],
                     'actual_this_case':{'head_forwards':12,'optimizer_steps':12,'fresh_worker_processes':3},
                     'wall_time_comparison':'not compared; actual timings separately recorded',
                     'source_sha256':C.sources()}
                C.write(f'345_result_B_连续恢复_{arm}_seed{seed}.json',row)
                resume_rows.append(row)
                print(f'B strict resume PASS {arm} seed{seed}: model/optimizer/RNG/step logs all byte identical',flush=True)
        formal_rows=[]
        bundle=C.read(C.HERE/'345_B_冻结入口绑定.json')
        for seed in (1,2):
            sequence=C.read(bundle['shuffle'][str(seed)]['formal']['path'])['epochs'][0]
            reference=None
            for arm in 'NSGR':
                directory,result=launch(arm,seed,'formal_order_first2',2,formal=True)
                logs=C.read(directory/'345_steps.json')
                ids=[r['image_ids'] for r in logs]
                digests=[r['batch_tensor_sha256'] for r in logs]
                assert ids==[sequence[:16],sequence[16:32]]
                if reference is None:
                    reference=(ids,digests)
                else:
                    bit_equal((ids,digests),reference,'formal batch equality')
                assert result['binding']['shuffle_sha256']==bundle['shuffle'][str(seed)]['formal']['sha256']
                row={'arm':arm,'seed':seed,'status':'已验证','image_ids':ids,'batch_tensor_sha256':digests,
                     'run_dir':str(directory),'optimizer_steps':2,'formal_epoch_cursor':{'epoch':0,'batch_index':2,'position':32}}
                formal_rows.append(row)
        C.write('345_result_B_正式顺序两步八组.json',{'status':'已验证','rows':formal_rows,
                'all_four_arms_same_seed_actual_tensor_bytes':True,'source_sha256':C.sources()})
        totals={key:sum(result['counts_this_process'][key] for result in RUN_RESULTS)
                for key in RUN_RESULTS[0]['counts_this_process']}
        assert totals['optimizer_steps']==112 and totals['head_forwards']==112 and totals['cache_loads']==1792
        C.write('345_result_B_入口与CPU验收总表.json',{'status':'已验证；待主控核对，真实训练未授权',
                'resume_cases':resume_rows,'formal_cases':formal_rows,'counts':totals,
                'worker_processes':len(RUN_RESULTS),'negative_cases':7,'maximum_model_or_optimizer_difference':0.0,
                'math_functions_inherited_by_identity':['make_common_init','compute_losses_340','build_optimizer','load_entry','collate','lr_at','lr_at_kappa','capture_rng','restore_rng'],
                'current_authorization_caps':E.CURRENT_CPU_CAP,'scientific_hard_cap':E.HARD_TOTAL_STEPS,
                '未执行':['真实训练','GPU','full 1218-step run','trained AP/TP effect'],'未核':['主控独立复核B'],
                'source_sha256':C.sources()})
        print('B COMPLETE: 8 strict resumes + 8 formal two-step runs; head/optimizer steps=112; real training=0',flush=True)
    except Exception as ex:
        totals={key:sum(result['counts_this_process'][key] for result in RUN_RESULTS)
                for key in RUN_RESULTS[0]['counts_this_process']} if RUN_RESULTS else {}
        C.write('345_result_B实现失败_停止.json',{'status':'实现失败，立即停止，未写通过版审查包',
                'exception':repr(ex),'traceback':traceback.format_exc(),'completed_worker_counts':totals,
                'completed_workers':len(RUN_RESULTS),'real_training_steps':0,'source_sha256':C.sources()})
        raise


if __name__=='__main__':
    main()
