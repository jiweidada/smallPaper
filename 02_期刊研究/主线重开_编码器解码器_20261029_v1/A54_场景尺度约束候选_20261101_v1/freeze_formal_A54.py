from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=P/'protocol_FROZEN_v1.json';assert not dest.exists()
    capacity=json.loads((P/'capacity_v2/capacity_result.json').read_text());assert capacity['status']=='PASS' and capacity['cumulative_A54_capacity_forwards']==9 and capacity['capacity_only_optimizer_updates']==6
    design=json.loads((P/'pilot_design_FROZEN_v1.json').read_text());cp=json.loads((P/'capacity_v2/capacity_protocol_FROZEN_v2.json').read_text())
    files=['scene_scale_prior.py','runtime_A54.py','scale_statistics_FROZEN_v1.json','pilot_design_FROZEN_v1.json','train_A54.py','supervise_A54.py','launch_formal_A54.py','observe_A54.py','capacity_v2/capacity_protocol_FROZEN_v2.json','capacity_v2/capacity_result.json']
    for f in files:
        if f.endswith('.py'):ast.parse((P/f).read_text())
    assert sha(P/'scene_scale_prior.py')==cp['files_sha256']['scene_scale_prior.py']
    plan=dict(design);plan.update(stage='FORMAL_FIXED12_FROZEN_BEFORE_LAUNCH',current_training_queued=True,
        files_sha256={f:sha(P/f) for f in files},external_sha256=cp['external_sha256'],
        capacity_protocol_sha256=sha(P/'capacity_v2/capacity_protocol_FROZEN_v2.json'),
        capacity_result_sha256=sha(P/'capacity_v2/capacity_result.json'),
        capacity_only_total_forwards=9,capacity_only_total_optimizer_updates=6,
        expected_native_initial_sha256='2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6',
        expected_M_Q_new_initial_tensors_sha256=capacity['results'][0]['new_initial_tensors_sha256'],
        post_Q_queue='none',formal_optimizer_updates_each=9708,whole_model_forwards_each_including_eval_and_latency=10159)
    dest.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'FROZEN','protocol_sha256':sha(dest),'own_files':len(files),'external_files':len(cp['external_sha256']),'epochs_each':12,'arms':['M','Q']}))
if __name__=='__main__':main()
