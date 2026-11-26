from __future__ import annotations
import datetime,json,os,pathlib,signal,subprocess,sys,time,traceback
from runtime_common import R,digest,save_json

def ticks(pid):
    try:return int(pathlib.Path('/proc/%s/stat'%pid).read_text().split(') ',1)[1].split()[19])
    except FileNotFoundError:return None

def summarize():
    rows={a:json.loads((R/('arm_'+a)/'result.json').read_text())for a in ['B','S','L']}
    assert all(v['status']=='COMPLETED'and v['epochs']==12 for v in rows.values())
    flow=[]
    for epoch in range(1,13):
        x={a:json.loads((R/('arm_'+a)/('epoch%02d_input_flow.json'%epoch)).read_text())for a in rows}
        for a in ['S','L']:
            assert x['B']['order_sha256']==x[a]['order_sha256']
            assert x['B']['first_eight_actual_input_sha256']==x[a]['first_eight_actual_input_sha256']
        flow.append({'epoch':epoch,'sampler_order_and_first_eight_actual_batches_match':True})
    ini={a:json.loads((R/('arm_'+a)/'initialization.json').read_text())for a in rows}
    assert len({v['common_init_sha256']for v in ini.values()})==1
    assert ini['S']['branch_initial_tensor_digest']==ini['L']['branch_initial_tensor_digest']
    metrics={a:rows[a]['endpoint_metrics']for a in rows}
    db=metrics['L']['APs75']-metrics['B']['APs75'];ds=metrics['L']['APs75']-metrics['S']['APs75']
    overhead=rows['L']['latency']['median_ms']/rows['B']['latency']['median_ms']-1
    overall_change=metrics['L']['AP']-metrics['B']['AP']
    gates={'L_minus_B_small_AP75_at_least_0_5pp':db>=.5,'L_minus_S_small_AP75_at_least_0_3pp':ds>=.3,'same_device_latency_overhead_at_most_10pct':overhead<=.1,'overall_AP_drop_no_more_than_0_2pp':overall_change>=-.2}
    passed=all(gates.values())
    result={'status':'COMPLETED','arms':metrics,'L_minus_B_small_AP75_pp':db,'L_minus_S_small_AP75_pp':ds,'L_minus_B_AP_pp':overall_change,'L_latency_fraction_over_B':overhead,'practical_gates':gates,'pilot_pass':passed,'decision':'WORTH_NEXT_VALIDATION_NOT_ESTABLISHED_INNOVATION'if passed else 'STOP_THIS_LOCALIZATION_ONLY_MAINLINE_NO_RESCUE_SWEEP','interpretation':'Single-seed 12-epoch resource-priority pilot. No significance/generalization claim. AP gain alone does not prove a localization mechanism.','input_flow_verification':flow,'optimizer_updates':sum(v['optimizer_updates']for v in rows.values()),'full_model_forwards':sum(v['full_model_forwards']for v in rows.values()),'elapsed_arm_seconds':{a:rows[a]['elapsed_seconds']for a in rows},'protocol_sha256':digest(R/'execution_protocol_FROZEN.json'),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save_json(R/'three_arm_result.json',result)
    lines=['VisDrone + ECDet-S 三组最小验证结果','状态：三组均完成12轮；单seed；COCO-style评估，不是官方榜单分数。','组别\tAP\tAPs\t小目标AP75']
    for a,m in metrics.items():lines.append('%s\t%.4f\t%.4f\t%.4f'%(a,m['AP'],m['APs'],m['APs75']))
    lines+=['L-B 小目标AP75：%+.4f 个百分点'%db,'L-S 小目标AP75：%+.4f 个百分点'%ds,'L/B 同机原生推理延时变化：%+.2f%%'%(overhead*100),'判断：'+('达到下一阶段投入门槛，尚未证实论文贡献。'if passed else '未达到预定投入门槛，停止定位专用读出主张，不继续扫模块/层数/权重。'),'训练数据顺序与每轮前8个实际输入批次在三组间一致；共同主干初始化一致，S/L读出初始化一致。','本结果不证明所有相似方法无效，也不能据此保证充分训练后的表现。']
    (R/'三组结果摘要.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

def main():
    with (R/'supervisor_once.lock').open('x')as f:f.write(str(os.getpid()))
    plan=json.loads((R/'execution_protocol_FROZEN.json').read_text())
    save_json(R/'supervisor_started.json',{'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'started_unix':time.time(),'protocol_sha256':digest(R/'execution_protocol_FROZEN.json'),'queue':['B','S','L'],'max_epochs_each':12,'timeout_seconds_per_arm':20*3600,'auto_retry':False,'new_direction_training_queued':False})
    try:
        for arm in ['B','S','L']:
            output=R/('arm_'+arm);output.mkdir(exist_ok=True)
            with (output/'stdout.log').open('w')as stdout,(output/'stderr.log').open('w')as stderr:
                process=subprocess.Popen([sys.executable,'-u',str(R/'train_arm.py'),'--arm',arm],cwd=str(R),stdout=stdout,stderr=stderr,start_new_session=True)
                started=time.monotonic();identity={'pid':process.pid,'start_ticks':ticks(process.pid),'arm':arm,'started_unix':time.time()}
                save_json(output/'child_identity.json',identity)
                while process.poll()is None:
                    if time.monotonic()-started>20*3600:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                        raise TimeoutError('Own arm hard timeout: '+arm)
                    progress={}
                    try:progress=json.loads((output/'progress.json').read_text())
                    except FileNotFoundError:pass
                    save_json(R/'queue_progress.json',{'status':'RUNNING','active_arm':arm,'worker':identity,'worker_progress':progress,'supervisor_pid':os.getpid(),'alive_checked_unix':time.time()})
                    time.sleep(30)
                save_json(output/'exit_receipt.json',{'exit_code':process.returncode,'pid':process.pid,'start_ticks':identity['start_ticks'],'elapsed_seconds':time.monotonic()-started})
                if process.returncode!=0:raise RuntimeError('Arm '+arm+' exited '+str(process.returncode)+'; remaining queue not started')
        result=summarize()
        save_json(R/'queue_progress.json',{'status':'COMPLETED','pilot_pass':result['pilot_pass'],'next':result['decision']})
        save_json(R/'supervisor_result.json',{'status':'COMPLETED','exit_code':0,'pid':os.getpid(),'result':str(R/'three_arm_result.json')})
    except BaseException as exc:
        save_json(R/'supervisor_result.json',{'status':'FAILED','pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),'auto_retry':False,'remaining_queue_started':False})
        save_json(R/'queue_progress.json',{'status':'FAILED','error':repr(exc),'remaining_queue_started':False})
        raise

if __name__=='__main__':main()
