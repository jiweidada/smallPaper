"""G_seed1 ten-step source equivalence and retained guards; stop on first failure."""
import ast,importlib,json,os,struct,subprocess,sys,time,traceback
from pathlib import Path
C=importlib.import_module('347_context')
import numpy as np
import torch
torch.set_num_threads(2)
RUN_RESULTS=[]

def same(a,b,path='root'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype,path
        assert a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes(),path
    elif isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],f'{path}.{k}')
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,f'{path}[{i}]')
    elif isinstance(a,float):
        assert isinstance(b,float) and struct.pack('!d',a)==struct.pack('!d',b),path
    else:assert type(a)==type(b) and a==b,(path,a,b)

def checkpoint(result):
    ref=result['last_checkpoint'];assert C.sha(ref['path'])==ref['sha256']
    r=C.read(ref['path']);assert C.sha(r['checkpoint_path'])==r['checkpoint_sha256']
    return torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False)

def launch(command,label):
    child=subprocess.run(command,cwd=str(C.HERE),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         encoding='utf-8',errors='strict',env=dict(os.environ),check=False)
    C.text(f'347_{label}_stdout.txt',child.stdout)
    C.text(f'347_{label}_stderr.txt',child.stderr)
    assert child.returncode==0,(label,child.returncode,child.stderr[-6000:])
    print(label+': process completed',flush=True)

def rejection(label,code,call):
    F=importlib.import_module('train347')
    before=dict(C.T.INPUT_COUNTS)
    try:call()
    except (C.E.GuardError,F.GuardError) as ex:
        assert ex.code==code,(label,code,ex.code)
        assert before==dict(C.T.INPUT_COUNTS),('guard reached data loader',label)
        return {'case':label,'status':'已验证','guard':ex.code,'head_forwards':0,'optimizer_steps':0}
    raise AssertionError(('negative was accepted',label))

def negative_checks(F,subject_dir):
    folder=C.HERE/'347_negative_fixtures';folder.mkdir(exist_ok=False)
    bundle=C.read(C.S345/'345_B_冻结入口绑定.json')
    cfg=C.read(bundle['configs']['G_seed1']['path'])['config']
    rows=[]
    for label,key,path,code in (
        ('config_SHA','config',bundle['configs']['G_seed1']['path'],'config_SHA_mismatch'),
        ('manifest_SHA','manifest',cfg['development_manifest'],'manifest_SHA_mismatch'),
        ('shuffle_SHA','shuffle',bundle['shuffle']['1']['formal']['path'],'shuffle_SHA_mismatch')):
        target=folder/f'347_{label}.json'
        with open(target,'xb') as f:f.write(Path(path).read_bytes()+b' ')
        rows.append(rejection(label,code,lambda k=key,p=target:F.validate_bindings('G',1,_test_paths={k:p})))
    rows.append(rejection('existing_run','run_directory_exists',lambda:F.execute('G',1,subject_dir,'cpu',1)))
    never=folder/'347_step1219_must_not_exist'
    rows.append(rejection('step1219','max_steps_budget',lambda:F.execute('G',1,never,'cpu',1219)))
    assert not never.exists()
    mini=C.read(bundle['shuffle']['1']['mini']['path'])
    rows.append(rejection('formal_rejects_mini','order_kind_mismatch',lambda:F.validate_order(mini,'formal',cfg,C.T.load_order())))
    fixture_map=C.read(C.S346/'346_构造断点清单.json')['records']
    for case in ('epoch_201_204','final_1216_1218'):
        rows.append(rejection('reject346_'+case,'resume_receipt_binding',lambda name=case:
            F.execute('G',1,folder/f'347_reject346_{name}','cpu',1,_resume=fixture_map[f'G_seed1/{name}'])))
    cli=subprocess.run([sys.executable,'-B',str(C.HERE/'train347.py'),'--arm','G','--seed','1',
                       '--run_dir',str(folder/'347_cli_override_must_not_exist'),'--device','cpu','--max_steps','10',
                       '--lr','0.001'],cwd=str(C.HERE),stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',check=False)
    assert cli.returncode==2 and 'unrecognized arguments: --lr' in cli.stderr
    assert not (folder/'347_cli_override_must_not_exist').exists()
    rows.append({'case':'CLI_rejects_lr_override','status':'已验证','exit_code':2,
                 'stderr':cli.stderr,'head_forwards':0,'optimizer_steps':0})
    assert len(rows)==9
    C.write('347_原有守卫反向验收.json',{'status':'已验证','rows':rows,
            'head_forwards':0,'optimizer_steps':0,'negative_models_constructed':2,
            'implicit_negative_common_init_reference_models':2,'source_sha256':C.sources()})
    return rows

def main():
    started=time.perf_counter()
    try:
        C.verify_sources()
        F=importlib.import_module('train347')
        assert F.CURRENT_CPU_CAP=={'formal':1218,'mini_acceptance':6}
        assert C.E.CURRENT_CPU_CAP=={'formal':2,'mini_acceptance':6}
        reference_dir=C.HERE/'347_reference345_G_seed1_10'
        subject_dir=C.HERE/'347_subject_G_seed1_10'
        C.write('347_reference_job.json',{'arm':'G','seed':1,'device':'cpu','max_steps':10,'run_dir':str(reference_dir)})
        launch([sys.executable,'-B',str(C.HERE/'347_reference_worker.py'),str(C.HERE/'347_reference_job.json')],'original345_reference10')
        ref=C.read(reference_dir/'345_run_result.json');RUN_RESULTS.append(ref)
        assert ref['counts_this_process']['optimizer_steps']==10
        launch([sys.executable,'-B',str(C.HERE/'train347.py'),'--arm','G','--seed','1','--run_dir',str(subject_dir),
                '--device','cpu','--max_steps','10'],'train347_formal_CLI10')
        subject=C.read(subject_dir/'347_run_result.json');RUN_RESULTS.append(subject)
        assert subject['step']==10 and not subject['full_budget_completed']
        assert subject['counts_this_process']['optimizer_steps']==10
        old2_path=C.S345/'345_B_CPU_acceptance_runs/345_G_seed1_formal_order_first2/345_steps.json'
        old2=C.read(old2_path);logs=C.read(subject_dir/'347_steps.json')
        same(logs[:2],old2,'existing345 first2 logs')
        encoded=json.dumps(logs[:2],ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        assert encoded.encode('utf-8')==old2_path.read_bytes(),'first2 log UTF-8 bytes changed'
        C.write('347_前2步与既有345日志对照.json',{'status':'已验证','steps':2,
            'all_log_fields_bit_identical':True,'serialized_UTF8_bytes_identical':True,
            'maximum_numeric_difference':0.0,'old_path':str(old2_path),'old_sha256':C.sha(old2_path),
            'subject_step_log':str(subject_dir/'347_steps.json'),'source_sha256':C.sources()})
        a=checkpoint(ref);b=checkpoint(subject)
        fields=('model','optimizer','rng','param_group_lr','step','epoch','batch_index','position_in_epoch',
                'shuffle_sha256','initial_state_sha256','model_state_sha256','step_logs')
        for name in fields:same(a[name],b[name],name)
        assert (reference_dir/'345_steps.json').read_bytes()==(subject_dir/'347_steps.json').read_bytes()
        for key,value in ref['binding'].items():same(value,subject['binding'][key],'binding.'+key)
        assert set(subject['binding'])-set(ref['binding'])=={'347_source_bundle_sha256','entry_version'}
        assert subject['binding']['347_source_bundle_sha256']==C.verify_sources()
        assert subject['binding']['entry_version']=='train347'
        assert (b['step'],b['epoch'],b['batch_index'],b['position_in_epoch'])==(10,0,10,160)
        assert b['format']=='347_full_state_v1'
        assert C.E.state_sha(b['model'])!=b['initial_state_sha256']
        optimizer_step_values=[float(v['step']) for v in b['optimizer']['state'].values() if 'step' in v]
        assert optimizer_step_values and max(optimizer_step_values)==10
        curve=C.read(C.BASE.S5/'342_预定学习率曲线1218点.json')
        order=C.read(C.S345/'345_shuffle_development_seed1.json')['epochs'][0]
        for step,r in enumerate(logs):
            assert r['step']==step and r['image_ids']==order[step*16:(step+1)*16]
            assert r['param_group_lr']==[curve['non_kappa'][step],curve['non_kappa'][step],curve['G_kappa'][step]]
        C.write('347_完整10步逐位对照.json',{'status':'已验证','fields':list(fields),
            'model_max_difference':0.0,'optimizer_max_difference':0.0,'rng_bit_identical':True,
            'step_log_file_bytes_identical':True,'reference_dir':str(reference_dir),'subject_dir':str(subject_dir),
            'new_binding_only':['347_source_bundle_sha256','entry_version'],
            'epoch_timing':'actual elapsed separately recorded, not compared for byte equality',
            'source_sha256':C.sources()})
        print('347 PASS: first2 existing logs and full10 model/optimizer/RNG/step logs all bit identical',flush=True)
        negatives=negative_checks(F,subject_dir)
        reference_proof=C.read(C.HERE/'347_隔离参考进程证明.json')
        negatives.append(reference_proof['negative']);assert len(negatives)==10
        keys=('head_forwards','loss_adapter_calls','native_loss_calls','backward_calls','optimizer_steps','cache_loads','global_forwards')
        totals={k:sum(r['counts_this_process'][k] for r in RUN_RESULTS) for k in keys}
        assert totals['optimizer_steps']==totals['head_forwards']==totals['backward_calls']==20
        assert totals['cache_loads']==320 and totals['native_loss_calls']==640
        assert C.E.CURRENT_CPU_CAP=={'formal':2,'mini_acceptance':6}
        C.write('347_验收_result.json',{'status':'347 G_seed1十步CPU验收通过，封存后停止；八个完整run未授权',
            'subject_steps':10,'reference_steps':10,'first2_existing_logs_bit_identical':True,
            'full10_model_optimizer_RNG_step_logs_bit_identical':True,'maximum_model_or_optimizer_difference':0.0,
            'subject_result':subject,'reference_result':ref,'negative_rows':negatives,'counts':totals,
            'subject_models_constructed':1,'reference_models_constructed':1,'negative_models_constructed':2,
            'implicit_common_init_reference_models':4,
            'wall_seconds':time.perf_counter()-started,
            '已执行':{'positive_worker_processes':2,'negative_cases':10,**totals},
            '未执行':['八个完整run','完整1218步run','val评估/G Delta/bootstrap/12项科学判定','GPU','SSH','heldout图像或缓存'],
            '未核':['用户独立核对本轮结果','完整训练耗时与效果'],
            'scientific_full_runs':0,'real_training_steps':0,'source_sha256':C.sources()})
        print('347 COMPLETE: subject10 + reference10 CPU updates; guards10 rejected; full scientific runs0',flush=True)
    except Exception as ex:
        C.write('347_实现失败_停止.json',{'status':'实现失败，停止，不启动完整run','exception':repr(ex),
            'traceback':traceback.format_exc(),'completed_workers':len(RUN_RESULTS),
            'completed_worker_optimizer_steps':sum(r['counts_this_process']['optimizer_steps'] for r in RUN_RESULTS),
            'scientific_full_runs':0,'source_sha256':C.sources()})
        raise

if __name__=='__main__':main()
