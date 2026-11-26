"""One CPU boundary case per new process; never a full run."""
import importlib,json,sys,traceback,os
C=importlib.import_module('346_context')
F=importlib.import_module('train346')

def rejected(label,expected,call):
    before=dict(C.T.INPUT_COUNTS)
    try:call()
    except C.E.GuardError as ex:
        assert ex.code==expected,(label,expected,ex.code)
        assert before==dict(C.T.INPUT_COUNTS),'negative guard reached the data loader'
        return {'case':label,'status':'已验证','guard':ex.code,'head_forwards':0,'optimizer_steps':0}
    raise AssertionError(('negative input accepted',label))

if __name__=='__main__':
    job=C.read(sys.argv[1]);arm=job['arm'];seed=job['seed'];case=job['case'];spec=C.CASES[case]
    try:
        result=F.execute_boundary(arm,seed,job['run_dir'],'cpu',spec['end'],
                                  _resume=job['resume'],_case=case)
        guards=[]
        if case=='epoch_201_204':
            guards.append(rejected('endpoint205','346_endpoint_not_whitelisted',lambda:
                F.execute_boundary(arm,seed,C.RUN_ROOT/f'346_never205_{arm}_{seed}','cpu',205,
                                   _resume=job['resume'],_case=case)))
        else:
            guards.append(rejected('endpoint1219','max_steps_budget',lambda:
                F.execute_boundary(arm,seed,C.RUN_ROOT/f'346_never1219_{arm}_{seed}','cpu',1219,
                                   _resume=job['resume'],_case=case)))
            guards.append(rejected('resume_completed1218','resume_endpoint_not_after_checkpoint',lambda:
                F.execute_boundary(arm,seed,C.RUN_ROOT/f'346_never_after1218_{arm}_{seed}','cpu',1218,
                                   _resume=result['last_checkpoint'],_case=case)))
        C.E.write_json(C.RUN_ROOT/f'346_{arm}_seed{seed}_{case}'/'346_worker_guards.json',
                        {'rows':guards,'actual_optimizer_updates_in_guards':0,'worker_pid':os.getpid()})
        print(json.dumps({'arm':arm,'seed':seed,'case':case,'global_cursor':result['step'],
                          'actual_steps':result['counts_this_process']['optimizer_steps'],'guards_rejected':len(guards)},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.write('346_worker_实现失败_停止.json',{'status':'实现失败，立即停止','job':job,
                'exception':repr(ex),'traceback':traceback.format_exc(),
                'input_counts':dict(C.T.INPUT_COUNTS),'partial_optimizer_updates':'未核；不以游标当作实际更新数',
                'real_training_steps':0})
        raise
