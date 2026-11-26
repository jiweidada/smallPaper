"""Independently inspect solver/EMA source and arithmetic behind the conditional N2 trace."""
from hashlib import sha256
from pathlib import Path
import ast
import json
import shutil

HERE=Path(__file__).resolve().parent
REC=HERE.parent
DEIM=Path(r'C:\Users\jiwei\Desktop\DEIM')
TIMELINE=HERE/'n2_conditional_timeline.json'
TRACE=HERE/'checkpoint_time_trace.json'
ARGS=REC/'source_snapshot/N2/args.json'
OUTPUT=HERE/'n2_source_semantics_verification.json'
SNAPSHOT=HERE/'source_snapshot'

def sha(path):
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def function_source(path,class_name,method):
    text=path.read_text(encoding='utf-8')
    tree=ast.parse(text)
    nodes=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==class_name]
    assert len(nodes)==1
    if isinstance(nodes[0],ast.ClassDef):
        methods=[x for x in nodes[0].body if isinstance(x,ast.FunctionDef) and x.name==method]
        assert len(methods)==1
        node=methods[0]
    else:
        assert method is None
        node=nodes[0]
    return ast.unparse(node)

def main():
    assert not OUTPUT.exists() and not SNAPSHOT.exists()
    timeline=json.loads(TIMELINE.read_text(encoding='utf-8'))
    trace=json.loads(TRACE.read_text(encoding='utf-8'))
    cfg=json.loads(ARGS.read_text(encoding='utf-8-sig'))['yaml_cfg']
    assert timeline['trace_sha256']==sha(TRACE) and timeline['args_sha256']==sha(ARGS)
    assert cfg['ema']['type']=='ModelEMA' and cfg['ema']['start']==0 and cfg['use_ema'] is True
    assert cfg['use_amp'] is False
    det=DEIM/'engine/solver/det_solver.py'
    base=DEIM/'engine/solver/_solver.py'
    engine=DEIM/'engine/solver/det_engine.py'
    ema=DEIM/'engine/optim/ema.py'
    fit=function_source(det,'DetSolver','fit')
    resume=function_source(base,'BaseSolver','load_resume_state')
    load=function_source(base,'BaseSolver','load_state_dict')
    update=function_source(ema,'ModelEMA','update')
    ema_load=function_source(ema,'ModelEMA','load_state_dict')
    train=function_source(engine,'train_one_epoch',None)
    checks={
        'restore_at_stop_epoch': 'epoch == self.train_dataloader.collate_fn.stop_epoch' in fit and "self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))" in fit,
        'increment_last_epoch_after_train': fit.index('train_one_epoch(') < fit.index('self.last_epoch += 1'),
        'resume_dispatches_state_load': 'self.load_state_dict(state)' in resume,
        'load_restores_last_epoch': "self.last_epoch = state['last_epoch']" in load,
        'load_dispatches_ema_state': 'v.load_state_dict(state[k])' in load,
        'ema_load_restores_updates': "self.updates = state['updates']" in ema_load,
        'ema_update_increments': 'self.updates += 1' in update,
        'training_loop_calls_ema_update': 'ema.update(model)' in train,
        'nonbest_late_epoch_restores_stage1': "elif epoch >= self.train_dataloader.collate_fn.stop_epoch:" in fit and fit.count("self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))")>=2}
    assert all(checks.values())
    path=[x for x in timeline['conditional_trajectory']]
    assert [x['loop_epoch'] for x in path]==[296,297,298,299,300]
    assert all(x['is_new_global_best'] for x in path)
    assert 472164 + 5*1617==480249 and 291+5==296
    assert path[-1]['assumed_ema_updates_after_train']==480249 and path[-1]['assumed_last_epoch_after_train']==296
    n0=next(x for x in trace['runs'] if x['run']=='N0')
    n2=next(x for x in trace['runs'] if x['run']=='N2')
    assert Path(n2['checkpoint']['path']).stat().st_mtime_ns==n2['checkpoint']['filesystem']['mtime_ns']
    assert Path(n2['best_log_evaluation_file']['path']).stat().st_mtime_ns==n2['best_log_evaluation_file']['filesystem']['mtime_ns']
    assert n2['best_log_evaluation_file']['mtime_delta_vs_checkpoint_ns']==36988300
    assert timeline['n0_saved_evaluation_embedded_date_span_seconds']>24*3600
    assert timeline['n0_saved_evaluation_filesystem_mtime_span_seconds']<10
    sources=[]
    for path in [det,base,engine,ema]:
        rel=path.relative_to(DEIM)
        target=SNAPSHOT/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        assert sha(path)==sha(target)
        sources.append({'path':str(path),'snapshot':str(target),'sha256':sha(path)})
    output={'status':'INSPECTED_CODE_AND_N2_COUNTER_ARITHMETIC_CONSISTENT',
            'timeline_sha256':sha(TIMELINE),'source_trace_sha256':sha(TRACE),'saved_args_sha256':sha(ARGS),
            'static_source_checks':checks,'source_files':sources,
            'config':{'ema_type':cfg['ema']['type'],'ema_start':cfg['ema']['start'],
                      'use_ema':cfg['use_ema'],'use_amp':cfg['use_amp']},
            'conditional_counter_arithmetic':'472164 + 5 * 1617 = 480249; 291 + 5 = 296',
            'n2_current_mtime_delta_ns':36988300,
            'n0_eval_embedded_date_span_hours':timeline['n0_saved_evaluation_embedded_date_span_seconds']/3600,
            'n0_eval_current_mtime_span_seconds':timeline['n0_saved_evaluation_filesystem_mtime_span_seconds'],
            'limits':['The inspected source version and the assumed historical batch count are not authenticated.',
                      'A conditional counter match and nearby file timestamps do not bind a checkpoint hash to an evaluation.',
                      'Current filesystem timestamps are not generally reliable historical event clocks.',
                      'No model, prediction, evaluator, training or optimizer was run.']}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'status':output['status'],'static_checks':checks,'counter_arithmetic':output['conditional_counter_arithmetic'],
                      'n2_mtime_delta_ms':output['n2_current_mtime_delta_ns']/1e6,
                      'n0_embedded_span_hours':output['n0_eval_embedded_date_span_hours'],
                      'n0_fs_span_seconds':output['n0_eval_current_mtime_span_seconds']},ensure_ascii=False))

if __name__=='__main__':
    main()
