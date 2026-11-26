"""Bounded new-arm sequence; explicit root acceptance gate precedes formal runs."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)

def run(root):
    jobs=root/'jobs';code=root/'code'
    lock=(jobs/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=root/'bundle_manifest.json';manifest_hash=sha(manifest)
    runtime=json.loads((root/'runtime.json').read_text());account=runtime['account']
    def verify():
        assert sha(manifest)==manifest_hash
        for row in json.loads(manifest.read_text())['files']:assert sha(root/row['path'])==row['sha256'],row['path']
    verify()
    uuid=runtime['GPU_uuid'];python=runtime['python']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH',None)
    def child(name,command,gpu=True):
        verify()
        assert shutil.disk_usage(root).free>=40*2**30,'Insufficient storage; never delete old materials'
        if gpu:
            assert not subprocess.check_output(['nvidia-smi','--id',uuid,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy; never stop another job'
        with (jobs/(name+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            ticks=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            save(jobs/(name+'_started.json'),{'pid':p.pid,'start_ticks':ticks,'command':command,'time':time.time()})
            (jobs/'status.json').write_text(json.dumps({'status':'running','stage':name,'pid':p.pid,'time':time.time()}))
            rc=p.wait()
        save(jobs/(name+'_exit.json'),{'exit_code':rc,'time':time.time()})
        if rc:raise RuntimeError(f'{name} failed, no automatic retry or later stage')
    child('preflight',[python,'-u',str(code/'preflight.py'),str(root)],gpu=False)
    if account=='secondary':
        audit=json.loads((jobs/'baseline_audit.json').read_text());assert audit['all_training_checks_passed']
        variants=['fixed_range_global_context']
    else:variants=['adaptive_range_global_context','adaptive_range_local_context']
    common=['--common-init',runtime['init'],'--train-images',runtime['train_images'],'--train-annotations',runtime['train_ann'],
            '--val-images',runtime['val_images'],'--val-annotations',runtime['val_ann'],'--gpu-uuid',uuid]
    capacities={}
    for variant in variants:
        child(variant+'_capacity',[python,'-u',str(code/'run_candidate.py'),'capacity','--config',str(code/'configs'/f'{variant}.yml'),
              *common,'--output',str(root/'outputs'/('capacity_'+variant))])
        cap=json.loads((root/'outputs'/('capacity_'+variant)/'capacity.json').read_text())
        assert cap['status']=='passed_capacity_only'
        capacities[variant]={'receipt_sha256':sha(root/'outputs'/('capacity_'+variant)/'capacity.json'),
            'allocated_peak_GiB':max(p['allocated_peak_GiB'] for p in cap['phases'])}
    if account=='secondary':
        expected={}
        for label,record in [('selected',audit['selected']),('final',audit['final'])]:
            output=root/'outputs'/('baseline_'+label+'_separate_eval')
            child('baseline_'+label+'_eval',[python,'-u',str(code/'evaluate.py'),'--config',str(code/'configs/independent.yml'),
               '--checkpoint',record['checkpoint'],'--reuse-legacy-independent','--images',runtime['val_images'],
               '--annotations',runtime['val_ann'],'--output',str(output),'--device','cuda:0'])
            result=json.loads((output/'evaluation.json').read_text())
            assert result['images']==548 and result['forward_views']==3390
            assert all(abs(result['metrics'][k]-record['metrics'][k])<1e-8 for k in record['metrics'])
            expected[label]={'metrics':result['metrics'],'checkpoint_sha256':result['checkpoint_sha256'],
                             'evaluation_sha256':sha(output/'evaluation.json')}
        save(jobs/'baseline_reuse_eval_acceptance.json',{'status':'selected_and_final_all12_reproduced','records':expected,'time':time.time()})
    save(jobs/'capacity_stages_complete.json',{'status':'capacity_stages_passed_waiting_for_root_pair_acceptance','variants':capacities,'time':time.time()})
    (jobs/'status.json').write_text(json.dumps({'status':'waiting_for_explicit_root_acceptance','time':time.time()}))
    while not (jobs/'formal_authorization.json').exists():time.sleep(5)
    authorization=json.loads((jobs/'formal_authorization.json').read_text())
    assert authorization['bundle_manifest_sha256']==manifest_hash and authorization['status']=='paired_capacity_and_reused_baseline_accepted'
    for variant in variants:
        output=root/'outputs'/(variant+'_96');cfg=code/'configs'/f'{variant}.yml'
        child(variant+'_train96',[python,'-u',str(code/'run_candidate.py'),'train','--config',str(cfg),*common,
            '--capacity-receipt',str(root/'outputs'/('capacity_'+variant)/'capacity.json'),'--output',str(output)])
        complete=json.loads((output/'complete.json').read_text());records=[]
        for line in (output/'training_audit.jsonl').open():
            r=json.loads(line)
            assert r['batches']==808 and len(r['steps'])==808 and r['unique_images']==6464 and r['consumed_originals']==6464
            assert r['optimizer_steps']==sum(s['optimizer_step_calls'] for s in r['steps'])
            assert sha(r['checkpoint'])==r['checkpoint_sha256']
            records.append(r)
        assert [r['epoch'] for r in records]==list(range(96))
        best=max(records,key=lambda r:r['metrics']['AP']);assert complete['best_epoch']==best['epoch']
        accepted={}
        for label,r in [('selected',best),('final',records[-1])]:
            dest=root/'outputs'/(variant+'_'+label+'_separate_eval')
            child(variant+'_'+label+'_eval',[python,'-u',str(code/'evaluate.py'),'--config',str(cfg),
              '--checkpoint',r['checkpoint'],'--images',runtime['val_images'],'--annotations',runtime['val_ann'],
              '--output',str(dest),'--device','cuda:0'])
            e=json.loads((dest/'evaluation.json').read_text())
            assert e['images']==548 and e['forward_views']==3390
            assert all(abs(e['metrics'][k]-r['metrics'][k])<1e-8 for k in r['metrics'])
            accepted[label]={'metrics':e['metrics'],'checkpoint_sha256':e['checkpoint_sha256']}
        save(jobs/(variant+'_accepted.json'),{'status':'96_epochs_selected_and_final_separate_evaluation_verified','records':accepted,'time':time.time()})
    (jobs/'status.json').write_text(json.dumps({'status':'bounded_account_plan_finished','time':time.time(),'no_additional_experiments':True}))

if __name__=='__main__':
    root=Path(sys.argv[1])
    try:run(root)
    except BaseException as e:
        (root/'jobs/failed_preserved.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'no_automatic_retry':True,'time':time.time()}))
        raise
