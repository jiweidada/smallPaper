"""Audit the original two-arm recipe executed on two verified hosts.

Analysis only. Keeps separate source queues and actual runtime provenance.
The per-arm audit body is copied unchanged from the checked serial auditor.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, sys
from summarize_completed_pair import (AuditError, require, sha, read, read_rows,
    validate_epoch_rows, validate_restores, validate_learning_rates, endpoint_epochs,
    PROTOCOL_SHA, EPOCHS, BATCHES, METRICS)
SERIAL_AUDITOR_SHA='3dee0beb969c698efdf7a64b796155672ca29224dd2eb844be2e4d9c4c546615'

def audit_one_arm(root,run):
    report={'arms':{}}
    first_batches=[]
    arm = run['arm']
    folder = root / 'runs' / arm
    receipt = read(folder / 'receipt.json')
    require(run['status'] == 'COMPLETED' and run['exit_code'] == 0, f'{arm} did not finish successfully')
    require(run['final_receipt'] == receipt, f'{arm} final receipt mismatch')
    require(receipt['status'] == 'COMPLETED' and receipt['completed_epochs'] == EPOCHS
            and receipt['actual_loop_epoch'] == EPOCHS - 1
            and receipt['optimizer_step_attempts'] == EPOCHS * BATCHES, f'{arm} incomplete budget')
    require(receipt['arm'] == arm and receipt['protocol_sha256'] == PROTOCOL_SHA, f'{arm} receipt binding')
    config_path = root / 'execution_configs' / f'{arm}.yml'
    config_sha = sha(config_path)
    for key, path in [('config_sha256', config_path), ('runner_sha256', root / 'fair_runner.py'),
                      ('environment_sha256', root / 'environment.current.json'),
                      ('inputs_receipt_sha256', root / 'cache_verification.json'),
                      ('initialization_sha256', root / 'preflight' / f'{arm}.initial.pt')]:
        require(receipt[key] == sha(path), f'{arm} {key}')
    rows = read_rows(folder / 'epochs.jsonl')
    validate_epoch_rows(rows, config_sha)
    actual_batches = read_rows(folder / 'actual_first100.jsonl')
    require(len(actual_batches) == 100 and actual_batches == read(root / 'preflight' / f'{arm}.first100.json'),
            f'{arm} actual first 100 batches')
    first_batches.append(actual_batches)
    events = read_rows(folder / 'restore_events.jsonl')
    stage_one_best, resets = validate_restores(rows, events, sha(folder / 'best_stg1.pt'))
    preflight = read(root / 'preflight' / f'{arm}.json')
    lr_result = validate_learning_rates(folder / 'executed_learning_rates.csv.gz', rows,
                                       read(config_path), preflight, resets, stage_one_best)
    selected = endpoint_epochs(rows)
    require(receipt['best_epoch'] == selected['best'] and receipt['best_AP'] == rows[selected['best']]['metrics']['AP'],
            f'{arm} best endpoint selection')
    endpoints = {}
    for endpoint, epoch in selected.items():
        metric_path = folder / f'{endpoint}.metrics.json'
        metrics = read(metric_path)
        require({k: v for k, v in metrics.items() if k != 'checkpoint_sha256'} == rows[epoch],
                f'{arm}/{endpoint} mixed or wrong epoch row')
        audit_dir = root / 'evidence_audits' / f'{arm}_{endpoint}_epoch{epoch:04}'
        audit = read(audit_dir / 'receipt.json')
        require(audit['status'] == 'SAVED_ENDPOINT_BINDING_VERIFIED' and audit['arm'] == arm
                and audit['endpoint'] == endpoint and audit['actual_epoch'] == epoch, f'{arm}/{endpoint} audit identity')
        require(audit['protocol_sha256'] == PROTOCOL_SHA and audit['metrics'] == rows[epoch]['metrics']
                and audit['ema_tensor_sha256'] == rows[epoch]['ema_tensor_sha256'], f'{arm}/{endpoint} audit binding')
        require(audit['image_count'] == 548 and audit['prediction_count'] == 164400,
                f'{arm}/{endpoint} validation coverage')
        require(all(audit[key] == 0 for key in ['model_inference_passes', 'optimizer_updates', 'evaluator_reruns']),
                f'{arm}/{endpoint} unexpected audit execution scope')
        require(audit['metrics_recomputed_from_saved_arrays'] == audit['metrics']
                or all(abs(audit['metrics_recomputed_from_saved_arrays'][k] - audit['metrics'][k]) <= 1e-12 for k in METRICS),
                f'{arm}/{endpoint} saved-array summary')
        require(read(audit_dir / 'metrics.json') == metrics, f'{arm}/{endpoint} copied metric row')
        files = [('checkpoint.pt', 'best.pt' if endpoint == 'best' else 'terminal_epoch0319.pt', 'checkpoint_sha256'),
                 ('predictions.json', f'{endpoint}.predictions.json', 'prediction_sha256'),
                 ('evaluation_arrays.npz', f'{endpoint}.evaluation_arrays.npz', 'evaluation_arrays_sha256')]
        for copied, original, key in files:
            require(sha(audit_dir / copied) == sha(folder / original) == metrics[key] == audit[key],
                    f'{arm}/{endpoint} {key}')
        endpoints[endpoint] = {'actual_epoch': epoch, 'metrics': metrics['metrics'],
                               'checkpoint_sha256': metrics['checkpoint_sha256'],
                               'ema_tensor_sha256': metrics['ema_tensor_sha256'],
                               'metrics_sha256': sha(metric_path), 'audit_receipt_sha256': sha(audit_dir / 'receipt.json')}
    report['arms'][arm] = {'trainable_parameters': preflight['trainable_parameters'],
                          'completed_epochs': len(rows), 'optimizer_step_attempts': receipt['optimizer_step_attempts'],
                          'stage_one_best_epoch': stage_one_best, 'restore_events': len(events),
                          'learning_rates': lr_result, 'endpoints': endpoints,
                          'record_hashes': {name: sha(folder / name) for name in ['receipt.json', 'epochs.jsonl',
                              'restore_events.jsonl', 'actual_first100.jsonl', 'executed_learning_rates.csv.gz']}}
    return report['arms'][run['arm']], first_batches[0]

def audit_split(primary,secondary):
    require(primary.resolve()!=secondary.resolve(),'Separate host evidence roots required')
    require(sha(Path(__file__).with_name('summarize_completed_pair.py'))==SERIAL_AUDITOR_SHA,'Shared audit helper changed')
    if not (primary/'queue.json').is_file() or not (secondary/'secondary_queue.json').is_file():
        return {'status':'NOT_READY','reason':'One or both source queues are absent','result_generated':False}
    queues=[read(primary/'queue.json'),read(secondary/'secondary_queue.json')]
    require([v['arm'] for v in queues[0]['runs']]==['BASE'],'Primary must contain exactly the BASE run')
    require([v['arm'] for v in queues[1]['runs']]==['FULL'],'Secondary must contain exactly the FULL run')
    states=[q['runs'][0]['status'] for q in queues]
    if states!=['COMPLETED','COMPLETED']:
        return {'status':'NOT_READY','arm_states':dict(zip(['BASE','FULL'],states)),'result_generated':False}
    require(queues[1]['status']=='COMPLETED','Secondary supervisor did not finish')
    require(queues[0]['status']=='RUNNING','Unexpected primary scheduler state')
    require(not (primary/'runs/FULL').exists() and not (secondary/'runs/BASE').exists(),'Unexpected duplicate arm directory')
    primary_guard=read(primary/'primary_full_reservation.json')
    require(primary_guard==read(secondary/'primary_full_reservation.json'),'Reservation provenance differs')
    require(primary_guard['status']=='FULL_RESERVED_FOR_SECONDARY_HOST_DO_NOT_REMOVE','Wrong reservation')
    require(sha(primary/'FULL.train.log')==sha(secondary/'primary_FULL_reserved_log.json')==primary_guard['reserved_log_sha256'],'Original queue reservation changed')
    require(sha(primary/'parallel_execution_addendum.json')==sha(secondary/'parallel_execution_addendum.json')==primary_guard['addendum_sha256'],'Placement addendum differs')
    require(sha(primary/'secondary_preparation_verified.json')==sha(secondary/'secondary_preparation.json')==primary_guard['secondary_preparation_sha256'],'Secondary preparation provenance')
    log=(primary/'training_supervisor.log').read_text()
    require('FileExistsError' in log and 'FULL.train.log' in log,'Original scheduler retirement has not been observed')
    snapshot=read(primary/'primary_terminal_status.json')
    require(snapshot['host_port']==30166 and snapshot['reserved_log_matches'] and not snapshot['unexpected_other_arm_directory'],'Primary terminal identity')
    require(snapshot['hardware']=='NVIDIA GeForce RTX 3090, GPU-85eca9af-10f3-4279-d72a-05ff1ecdcd3a, 550.100, 24576','Primary GPU/driver provenance')
    require(snapshot['queue_sha256']==sha(primary/'queue.json'),'Primary terminal snapshot queue binding')
    supervisor=next(v for v in snapshot['processes'] if v['pid']==queues[0]['supervisor_pid'])
    require(not supervisor['live'],'Original scheduler still live')
    runtime=read(secondary/'secondary_runtime.json')
    prep=read(secondary/'secondary_preparation.json')
    require(prep['status']=='READY_FOR_AUTHORIZED_FULL_ONLY' and prep['runtime_matches_frozen_software'],'Secondary preparation incomplete')
    require(sha(secondary/'secondary_runtime.json')==prep['runtime_sha256']==queues[1]['actual_runtime_sha256'],'Actual secondary runtime binding')
    require(queues[1]['reservation_sha256']==sha(secondary/'primary_full_reservation.json'),'Supervisor reservation binding')
    require(queues[1]['placement_addendum_sha256']==sha(secondary/'parallel_execution_addendum.json'),'Supervisor placement binding')
    require(queues[1]['supervisor_script_sha256']==sha(secondary/'supervise_secondary_full.py'),'Secondary supervisor code binding')
    report={'status':'PAIR_RECORDS_VERIFIED_PENDING_MANUSCRIPT_REVIEW','protocol_sha256':PROTOCOL_SHA,
            'analysis_utc':datetime.now(timezone.utc).isoformat(),'analysis_script_sha256':sha(Path(__file__)),
            'model_executions':0,'evaluator_reruns':0,'submission_ready':False,'arms':{},'comparisons':{},
            'execution_placement':'BASE primary GPU; FULL secondary GPU; same RTX3090 and driver, frozen recipe and software',
            'source_queue_hashes':{'primary':sha(primary/'queue.json'),'secondary':sha(secondary/'secondary_queue.json')},
            'primary_scheduler_retirement':'Expected exclusive-log reservation after recorded successful BASE exit; not a model failure.',
            'actual_secondary_runtime_sha256':sha(secondary/'secondary_runtime.json')}
    batches=[]
    for root,arm,queue in [(primary,'BASE',queues[0]),(secondary,'FULL',queues[1])]:
        require(sha(root/'execution_protocol.json')==PROTOCOL_SHA,'Protocol changed')
        protocol=read(root/'execution_protocol.json')
        require(queue['protocol_sha256']==PROTOCOL_SHA and not queue['automatic_retry'] and queue['hard_timeout_seconds_per_arm']==604800,'Queue budget binding')
        for item in protocol['frozen_files']:
            require(sha(root/item['path'])==item['sha256'],'Frozen file changed: '+arm+'/'+item['path'])
        result,first=audit_one_arm(root,queue['runs'][0])
        report['arms'][arm]=result;batches.append(first)
    expected=read(primary/'execution_protocol.json')['runtime']
    normalize=lambda d:{k.lower().replace('_','-'):v for k,v in d.items()}
    require(normalize(runtime['packages'])==normalize(expected['packages']),'Cross-host package difference')
    for key in ['python','torch','cuda','cudnn','torch_build']:
        require(runtime[key]==expected[key],'Cross-host runtime difference: '+key)
    require(runtime['gpu']=='NVIDIA GeForce RTX 3090, GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a, 550.100, 24576','Unexpected secondary GPU')
    require(batches[0]==batches[1],'Cross-host actual first100 batches differ')
    for endpoint in ['terminal','best']:
        a=report['arms']['BASE']['endpoints'][endpoint];b=report['arms']['FULL']['endpoints'][endpoint]
        report['comparisons'][endpoint]={'role':'primary' if endpoint=='terminal' else 'secondary',
          'BASE_epoch':a['actual_epoch'],'FULL_epoch':b['actual_epoch'],
          'FULL_minus_BASE':{k:b['metrics'][k]-a['metrics'][k] for k in METRICS},
          'FULL_minus_BASE_percentage_points':{k:100*(b['metrics'][k]-a['metrics'][k]) for k in METRICS}}
    report['conclusion_scope']='One fixed-protocol pair on two same-model GPUs; total combination only. No isolated-module, multi-seed significance, bitwise cross-GPU reproducibility, independent-test, official VisDrone ranking or deployment-speed claim.'
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--secondary',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    try:
        report=audit_split(a.primary.resolve(),a.secondary.resolve())
        if report['status']=='NOT_READY':print(json.dumps(report));return 2
        if a.output:
            require(a.output.resolve().is_relative_to(a.primary.resolve().parent),'Output outside evidence parent')
            with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report,allow_nan=False));return 0
    except (AuditError,OSError,KeyError,ValueError,TypeError,StopIteration) as e:
        print(json.dumps({'status':'FINAL_SPLIT_AUDIT_FAILED','result_generated':False,'error':type(e).__name__+': '+str(e)}));return 1
if __name__=='__main__':sys.exit(main())
