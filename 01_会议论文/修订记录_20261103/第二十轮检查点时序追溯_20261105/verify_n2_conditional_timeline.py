"""Test the inspected solver's conditional bookkeeping against retained N2 metadata.

This is arithmetic on saved logs and metadata. No model or training loop is run.
"""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import ast
import json

HERE=Path(__file__).resolve().parent
REC=HERE.parent
PAPER=REC.parent
TRACE=HERE/'checkpoint_time_trace.json'
OUTPUT=HERE/'n2_conditional_timeline.json'
SOLVER=Path(r'C:\Users\jiwei\Desktop\DEIM\engine\solver\det_solver.py')
ARGS=REC/'source_snapshot/N2/args.json'
LOG=REC/'source_snapshot/N2/log.txt'
TRAIN_MANIFEST=PAPER/'公平对照方案_20261103/training_image_manifest.json'

def sha(path):
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    assert not OUTPUT.exists()
    trace=json.loads(TRACE.read_text(encoding='utf-8'))
    assert trace['status']=='SAVED_CHECKPOINT_AND_EVALUATION_TIMESTAMPS_INSPECTED'
    n2=next(x for x in trace['runs'] if x['run']=='N2')
    n0=next(x for x in trace['runs'] if x['run']=='N0')
    ckpt=n2['checkpoint']
    stage1=trace['N2_additional_checkpoints']['best_stg1.pth']
    for item in [ckpt,stage1,n2['best_log_evaluation_file']]:
        assert sha(Path(item['path']))==item['sha256']
    args=json.loads(ARGS.read_text(encoding='utf-8-sig'))
    cfg=args['yaml_cfg']
    stop=cfg['train_dataloader']['collate_fn']['stop_epoch']
    batch=cfg['train_dataloader']['total_batch_size']
    drop_last=cfg['train_dataloader']['drop_last']
    count=json.loads(TRAIN_MANIFEST.read_text(encoding='utf-8-sig'))['count']
    assert (stop,batch,drop_last,count)==(296,4,True,6471)
    assumed_steps=count//batch
    logs=[json.loads(line) for line in LOG.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    assert len(logs)==320 and [r['epoch'] for r in logs]==list(range(320))
    before=max(logs[:stop],key=lambda x:x['test_coco_eval_bbox'][0])
    assert before['epoch']==291 and before['epoch']==stage1['saved_last_epoch']
    code=SOLVER.read_text(encoding='utf-8')
    ast.parse(code)
    markers=["if epoch == self.train_dataloader.collate_fn.stop_epoch:",
             "self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))",
             'train_stats = train_one_epoch(', 'self.last_epoch += 1',
             "if best_stat['epoch'] == epoch and self.output_dir:",
             "elif epoch >= self.train_dataloader.collate_fn.stop_epoch:"]
    offsets=[code.index(mark) for mark in markers]
    assert offsets==sorted(offsets)
    meta_epoch=stage1['saved_last_epoch']
    updates=stage1['saved_ema_updates']
    best=before['test_coco_eval_bbox'][0]
    trajectory=[]
    for epoch in range(stop,n2['best_log_epoch']+1):
        meta_epoch+=1
        updates+=assumed_steps
        ap=logs[epoch]['test_coco_eval_bbox'][0]
        is_new_best=ap>best
        trajectory.append({'loop_epoch':epoch,'logged_AP':ap,'assumed_last_epoch_after_train':meta_epoch,
                           'assumed_ema_updates_after_train':updates,'is_new_global_best':is_new_best})
        if is_new_best:
            best=ap
        else:
            meta_epoch=stage1['saved_last_epoch']
            updates=stage1['saved_ema_updates']
    assert all(x['is_new_global_best'] for x in trajectory)
    assert meta_epoch==ckpt['saved_last_epoch']==296 and updates==ckpt['saved_ema_updates']==480249
    fs_delta=n2['best_log_evaluation_file']['filesystem']['mtime_ns']-ckpt['filesystem']['mtime_ns']
    assert fs_delta==n2['best_log_evaluation_file']['mtime_delta_vs_checkpoint_ns']
    assert Path(ckpt['path']).stat().st_mtime_ns==ckpt['filesystem']['mtime_ns']
    assert Path(n2['best_log_evaluation_file']['path']).stat().st_mtime_ns==n2['best_log_evaluation_file']['filesystem']['mtime_ns']
    n0_dates=[datetime.strptime(e['stored_date'],'%Y-%m-%d %H:%M') for e in n0['all_saved_evaluations']]
    n0_fs=[e['filesystem']['mtime_ns'] for e in n0['all_saved_evaluations']]
    n0_embedded_span=(max(n0_dates)-min(n0_dates)).total_seconds()
    n0_fs_span=(max(n0_fs)-min(n0_fs))/1e9
    assert n0_embedded_span>24*3600 and n0_fs_span<10
    output={'status':'N2_METADATA_CONSISTENT_WITH_CONDITIONAL_SOLVER_BOOKKEEPING',
            'trace_sha256':sha(TRACE),'args_sha256':sha(ARGS),'log_sha256':sha(LOG),
            'training_manifest_sha256':sha(TRAIN_MANIFEST),'inspected_solver_sha256':sha(SOLVER),
            'assumptions':{'current_solver_used_historically':'not verified',
                'one_ema_update_per_batch':'assumed', 'no_other_resume_or_skipped_updates':'assumed',
                'current_6471_images_and_batch4_used_historically':'not verified',
                'assumed_steps_per_epoch':assumed_steps},
            'stage1_saved_epoch':stage1['saved_last_epoch'],'stage1_saved_ema_updates':stage1['saved_ema_updates'],
            'stage1_best_logged_epoch':before['epoch'],'stage1_best_logged_AP':before['test_coco_eval_bbox'][0],
            'conditional_trajectory':trajectory,'saved_best_stage2_epoch_field':ckpt['saved_last_epoch'],
            'saved_best_stage2_ema_updates':ckpt['saved_ema_updates'],
            'n2_evaluation_minus_checkpoint_mtime_ns':fs_delta,
            'n2_checkpoint_saved_date':ckpt['saved_date'],
            'n2_evaluation_saved_date_minute':n2['best_log_evaluation_file']['stored_date'],
            'n0_saved_evaluation_embedded_date_span_seconds':n0_embedded_span,
            'n0_saved_evaluation_filesystem_mtime_span_seconds':n0_fs_span,
            'limits':['This conditional bookkeeping agreement does not prove the saved checkpoint generated the epoch-300 evaluation.',
                      'The historical code, full dataset bytes, clock origin, skipped updates and any additional resumes are not authenticated.',
                      'Filesystem timestamps can be reset or copied: the N0 embedded date span and current file mtime span differ sharply.',
                      'No training, model inference, evaluator execution or optimizer update occurred.'],
            'models_run':0,'forwards':0,'evaluator_runs':0,'optimizer_updates':0}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'status':output['status'],'assumed_steps':assumed_steps,'trajectory':trajectory,
        'n2_mtime_delta_ms':fs_delta/1e6,'n0_embedded_span_hours':n0_embedded_span/3600,
        'n0_filesystem_mtime_span_seconds':n0_fs_span},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
