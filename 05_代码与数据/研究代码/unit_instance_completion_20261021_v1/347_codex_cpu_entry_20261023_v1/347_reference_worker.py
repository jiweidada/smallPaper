"""Original sealed345 reference, fresh process with an absolute ten-step cap."""
import importlib,json,sys,traceback,os
C=importlib.import_module('347_context')

if __name__=='__main__':
    job=C.read(sys.argv[1])
    try:
        assert job['arm']=='G' and job['seed']==1 and job['max_steps']==10 and job['device']=='cpu'
        C.verify_sources()
        old_path=C.S345/'train345.py';before=C.sha(old_path)
        assert C.E.CURRENT_CPU_CAP=={'formal':2,'mini_acceptance':6}
        C.E.CURRENT_CPU_CAP={'formal':10,'mini_acceptance':6}
        assert C.E.HARD_TOTAL_STEPS==1218
        result=C.E.execute('G',1,job['run_dir'],'cpu',10)
        counts_before=dict(C.T.INPUT_COUNTS)
        try:C.E.execute('G',1,C.HERE/'347_reference_step11_must_not_exist','cpu',11)
        except C.E.GuardError as ex:
            assert ex.code=='acceptance_step_cap',ex.code
            guard={'case':'reference_rejects11','guard':ex.code,'head_forwards':0,'optimizer_steps':0}
        else:raise AssertionError('reference process accepted more than10 steps')
        assert counts_before==dict(C.T.INPUT_COUNTS)
        assert C.sha(old_path)==before
        C.write('347_隔离参考进程证明.json',{'status':'已验证','pid':os.getpid(),
                'original_source_path':str(old_path),'original_source_sha256_before':before,
                'original_source_sha256_after':C.sha(old_path),'in_memory_formal_cap':10,
                'scientific_budget_unchanged':1218,'actual_counts':result['counts_this_process'],
                'negative':guard,'source_sha256':C.sources()})
        print(json.dumps({'reference':'sealed345','arm':'G','seed':1,'steps':10,'max_allowed':10,
                          'pid':os.getpid(),'file_unchanged':True},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.write('347_隔离参考实现失败_停止.json',{'status':'实现失败，停止','exception':repr(ex),
                'traceback':traceback.format_exc(),'input_counts':dict(C.T.INPUT_COUNTS),
                'partial_optimizer_steps':'未核；不补造数字','scientific_full_runs':0})
        raise
