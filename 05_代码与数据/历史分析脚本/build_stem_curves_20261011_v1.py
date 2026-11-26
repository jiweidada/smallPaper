"""Prepare the actual epoch curves using byte-verified observations and cached BASE."""
from pathlib import Path
import base64,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stem_curves_20261011_v1';OUT.mkdir(exist_ok=False)
MAIN=ROOT/'outputs/stem_component12_20261011_v1';snapshot=json.loads((MAIN/'training_observation_v23.json').read_bytes())
verified=json.loads((MAIN/'training_observation_v23_transport_verification.json').read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert verified['verified_snapshot_sha256']==sha(MAIN/'training_observation_v23.json')
sys.path.insert(0,str(ROOT/'英文小论文/code/original_stem12_monitor_20261011_v1'))
from verify_terminal_local import local_protocol,read
p=local_protocol();payloads=json.loads("{\"1\":{\"base64\":\"eyJzZWVkIjogMSwgIm9ic2VydmVkX3JlbW90ZV9VVEMiOiAiMjAyNi0xMC0xMVQxMTo1NTowMy43NjY2MzArMDA6MDAiLCAicm93cyI6IFt7ImVwb2NoIjogMCwgInRlc3RfY29jb19ldmFsX2Jib3giOiBbMC4wMDU4MDQwMTMwNDY1NDg3MzcsIDAuMDE0NzEyNjEwOTQ5OTczNTIxLCAwLjAwMzcxMjM1ODk0MzUxNzAxNzQsIDAuMDAzMTc4NjEwMDk0NDQ4MDM5OCwgMC4wMDg4ODIyNTMxOTU4MjY0MTIsIDAuMDE1OTQzNTk0NjY0MzI3OTczLCAwLjAwNTY5OTQxNTAwOTgzMzAxNiwgMC4wMjAzMzM3OTIwNDk2NTY0NzMsIDAuMDM5NTg3ODkxOTY0MDAzNDk1LCAwLjAxMTMyMzEyMDExMzMwNDY1NSwgMC4wNzYzOTY5MTkwNjQ1NjQzMiwgMC4wODAyNjIyOTk4NTEzNTgwMV19LCB7ImVwb2NoIjogMSwgInRlc3RfY29jb19ldmFsX2Jib3giOiBbMC4wMzYxNDczMjEwMzA5MDIxNCwgMC4wNjY5MjYyMDQ0MDI4NDMyNSwgMC4wMzYzNjg1Mzc4Njg4MDIwNCwgMC4wMTY1ODU2MDcyNTcwODA2NjgsIDAuMDUzMDI4NjQwMjA5MDYzNjQsIDAuMDk1OTU2NDg2NzkwNDA2NTgsIDAuMDI1NzUzNDA2MjMxMDIwNzM3LCAwLjA3OTYxNTkxODc0ODI0MzU2LCAwLjEyNTE2NTU1ODg0MzU0ODM4LCAwLjA3MDQwMzA1NTk0ODQ1NjAzLCAwLjE5ODMyNDA1ODA2MjI5ODc1LCAwLjI0NjYyMDk0MjIyODA4ODI1XX0sIHsiZXBvY2giOiAyLCAidGVzdF9jb2NvX2V2YWxfYmJveCI6IFswLjA1OTkwMTg4ODA3MDMzOTk3LCAwLjExMDUwMjc5MjkwNTkxNDkzLCAwLjA1NzM2MDY4NTIzMTg5MDQ0LCAwLjAyODcxNTMxMTE5NTA2OTgzNiwgMC4wOTA1MzQ2NDMxMDY4Mzg2NywgMC4xNDcyODA5ODA5MTIxNjQwNywgMC4wNDAwMTkzNzg0MjA4NTAyNywgMC4xMjEyNjI0NDQ0MTYxODk5MSwgMC4xNzY4ODcwMjc5MzkwOTY0NCwgMC4xMDY5OTYzNzM3MDg5ODQ5OSwgMC4yNjY2Nzg0NDQzMTQ2NzMxNCwgMC4yOTk4Nzc2MTAyOTYyNjM5XX1dLCAiY29kZV9zaGEyNTZfdmVyaWZpZWQiOiB0cnVlLCAibWF0cGxvdGxpYl9hdmFpbGFibGUiOiB0cnVlfQ==\",\"sha256\":\"e6b2a4c443e2bd1fb4f4840bb945eb688d71b3f9352328e0dfea156338496dab\",\"bytes\":1027},\"2\":{\"base64\":\"eyJzZWVkIjogMiwgIm9ic2VydmVkX3JlbW90ZV9VVEMiOiAiMjAyNi0xMC0xMVQxMTo1NTowNC4xNzM5MTQrMDA6MDAiLCAicm93cyI6IFt7ImVwb2NoIjogMCwgInRlc3RfY29jb19ldmFsX2Jib3giOiBbMC4wMDU3MzUzMzEwOTUwNjI1NTY1LCAwLjAxMzA1MTUyNzU0OTgzMDAwNSwgMC4wMDQ4NDk4MTQxMDM1NDg3MzMsIDAuMDAzNDczMzAxMjY5NTIxMDU2MywgMC4wMDkzMjg5MzczNTY5NTcxNjYsIDAuMDMxNzkzMDMwNDEzMDcwMzUsIDAuMDA1NzkwMjY3ODMxNTkwNjcsIDAuMDE4Mjk3OTMwMzQxNTI1MzQ3LCAwLjAzNzA3MjExMjEzNTQ2MjIsIDAuMDE0MDQzMTM3NzA4MzAwNTk0LCAwLjA3MTE5Njk0MTQyNjA1OTc3LCAwLjA5MjU5ODQ1NjU0ODM2MjUzXX0sIHsiZXBvY2giOiAxLCAidGVzdF9jb2NvX2V2YWxfYmJveCI6IFswLjAyNDk0MDA2OTY5MDIyNDUsIDAuMDQ3ODUwMjU3MzgzNTQ1MTk2LCAwLjAyMzM3OTQ5OTgwNjYyMjMxLCAwLjAxMjk3Mzg2ODk1ODQwNzM2NSwgMC4wMzgyODY0NzU3ODQ4NjA4OCwgMC4wNjA3NTQ3NzQ5ODAwMDgwNiwgMC4wMTk0MjM1ODA0NjUwNzAyNjYsIDAuMDYxNjk2ODIxNjYzMTYxMzksIDAuMDk4MzIxMzQ5ODMwMTcwOCwgMC4wNDk5NDIwMjkxOTAxOTIxMSwgMC4xNTg4NDMxMTE1NjY0OTgyLCAwLjE3Mjk1MjgyOTU0NjM5NDk2XX0sIHsiZXBvY2giOiAyLCAidGVzdF9jb2NvX2V2YWxfYmJveCI6IFswLjA0Mjc0MTExODMwNzMwMDc4LCAwLjA3OTk5MjU3NDAwMzk3NjgzLCAwLjA0MDQ2OTIwMjU4MjI1MDY1LCAwLjAyMTc0Nzg1MzM3ODY1NTg4NSwgMC4wNjQ1NTg3MjQ0MjI0Mzk1NCwgMC4xMDUxNjg5ODk2MjY4MTkwNSwgMC4wMzE1NTYzNzQwODM1NDA3MSwgMC4xMDA1MjQwNTA3ODUzMzU1NSwgMC4xNDk3ODQ5NjEyMzk0NzE2LCAwLjA5MDcxODExNjE3MDgyMTE4LCAwLjIyOTc3Mjk5MTMxMzcxNzI2LCAwLjI1NTQ1MjM0MjAyNTM5Mjk2XX0sIHsiZXBvY2giOiAzLCAidGVzdF9jb2NvX2V2YWxfYmJveCI6IFswLjA1NzkwNzY5ODQ5MjQzMjYyLCAwLjEwNzY5OTcyMjMwOTk2NTA4LCAwLjA1NDk0MTg0ODU1NzAyNDcsIDAuMDMwOTgzMTYxMTIyMTgzMzM1LCAwLjA4NTcyNjEzMzc3MTAzMjU3LCAwLjEzODYzNTcxNzA3NDQ2OTIzLCAwLjAzOTMxNTg1Mjg4OTk1NTU2NiwgMC4xMjA3NDM3MzAyOTQ0MDgwOSwgMC4xNzY1MzA0ODY3MjAxMDI0NiwgMC4xMTMyNjE3MzQ0NTkwMzA2OCwgMC4yNjU3MzQ0MzEwNDg1MjY4LCAwLjI4ODgyMDc5MjA2NzA2MDJdfV0sICJjb2RlX3NoYTI1Nl92ZXJpZmllZCI6IHRydWUsICJtYXRwbG90bGliX2F2YWlsYWJsZSI6IHRydWV9\",\"sha256\":\"fd9d14771621243b9d1a1ab091062629304d6efc2a982ce4f0923c0d872346b1\",\"bytes\":1311}}");transport={};rows=[];sources={}
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
for seed in (1,2):
    o=snapshot['seed'+str(seed)];assert not o['errors']
    d=payloads[str(seed)];raw=base64.b64decode(d['base64'],validate=True)
    assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256']
    name=f'reference_history_seed{seed}_v1.json'
    with (OUT/name).open('xb') as f:f.write(raw)
    ref=json.loads(raw);assert ref['seed']==seed and ref['code_sha256_verified']
    transport[str(seed)]={k:v for k,v in d.items() if k!='base64'};transport[str(seed)]['file']=name
    assert [r['epoch'] for r in ref['rows']]==list(range(len(ref['rows']))) and len(ref['rows'])<=12
    ev=p['reused_local_evidence'][str(seed)];packet=ROOT/ev['packet_relative_path'];report=ROOT/ev['report_relative_path']
    assert sha(report)==ev['report_sha256'] and sha(packet/'manifest.json')==ev['manifest_sha256']
    log=packet/'reference/baseline_log.txt';item=read(packet/'manifest.json')['files']['reference/baseline_log.txt']
    assert log.stat().st_size==item['bytes'] and sha(log)==item['sha256']
    base=[json.loads(x) for x in log.read_text().splitlines() if x];assert [r['epoch'] for r in base]==list(range(24))
    groups={'pretrained_BASE':base[:12],'native_stem_scratch':ref['rows']}
    groups.update({arm:o['jobs'][arm]['metrics'] for arm in ('original_stem','detail_lowres','highfreq_rgb')})
    for arm,metrics in groups.items():
        assert [r['epoch'] for r in metrics]==list(range(len(metrics))) and len(metrics)<=12
        for r in metrics:
            stats=r['test_coco_eval_bbox'];assert len(stats)==12
            rows.append(dict(seed=seed,arm=arm,epoch=r['epoch']+1,AP=stats[0]*100,APs=stats[3]*100))
    sources[str(seed)]=dict(full_and_controls_remote_UTC=o['observed_remote_UTC'],reference_remote_UTC=ref['observed_remote_UTC'],
        baseline_log_sha256=sha(log),reference_payload_sha256=d['sha256'],completed_epochs={a:len(m) for a,m in groups.items()})
write(OUT/'history_transport_v1.json',transport)
data=dict(status='VERIFIED_SAVED_LOG_CURVES_FIXED12_PENDING',rows=rows,sources=sources,
    snapshot_sha256=sha(MAIN/'training_observation_v23.json'),new_model_forwards=0,new_optimizer_updates=0,
    fixed12_saved_EMA_verified=False,queued_controls_without_results=['attention_spatial_mean','early_final_downsample'],
    scope='All available completed epochs, preserving both seeds. Pretrained BASE is cached existing evidence; all current Stem arms drop Stem pretraining only. No unequal-epoch efficacy claim, endpoint selection, extrapolation, or causal conclusion.')
write(OUT/'plot_data_v1.json',data)
write(OUT/'execution_state_v1.json',dict(status='VERIFIED_CURVE_DATA_READY_RENDER_PENDING',goal_complete=False,previous_goal_turn='progress',
    current_goal_turn='progress',data_sha256=sha(OUT/'plot_data_v1.json'),source_rows=len(rows),real_model_forwards=0))
print(json.dumps(dict(data=data,data_sha256=sha(OUT/'plot_data_v1.json'),plot_code_sha256=sha(ROOT/'英文小论文/code/stem_curves_20261011_v1/plot.py')),ensure_ascii=False))

