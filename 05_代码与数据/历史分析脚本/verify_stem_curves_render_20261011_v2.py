"""Validate the updated scientific figure against all87 verified source points."""
from pathlib import Path
import base64,hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stem_curves_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads("{\"base64\":\"eyJleGl0X2NvZGUiOiAwLCAib3V0cHV0IjogIntcInN0YXR1c1wiOiBcIlJFTkRFUkVEX1ZFUklGSUVEX1NBVkVEX0xPR19DVVJWRVNcIiwgXCJzb3VyY2Vfcm93c1wiOiA4NywgXCJwbG90X2NvZGVfc2hhMjU2XCI6IFwiY2VmYjY0ZjIwYWU5MWU3YWMyYTVmOTBjOGFkODU3NWYzMTFlODJlNmI1OWFjODNmMWYyM2YxMmM4MzQzODk3ZVwifVxuIiwgInBhdGhzIjogeyJyZW5kZXJlZF92Mi9wbG90X3JlcG9ydC5qc29uIjogeyJieXRlcyI6IDEwNDEwLCAic2hhMjU2IjogIjVkNzUxM2ZlOGZlMzQ4MGFhMTMzYmM3OWU5ZDIwZDU4MDNmMWFlODJmNTU4YjYyM2E1MTdkMmM3MTU0MTNkNDkifSwgInJlbmRlcmVkX3YyL3N0ZW1fY3VydmVzX3YxLnBuZyI6IHsiYnl0ZXMiOiAyNTgyMTksICJzaGEyNTYiOiAiYTY2Mjg5NWY0NGE1MDgxZTJlOGY2OTJiZTM0ZmEzZGIwZDQxYWJlYzZkMDIxMDNlYWU0NDcxNDQwODJjM2M4NSJ9LCAicmVuZGVyZWRfdjIvc3RlbV9jdXJ2ZXNfdjEuc3ZnIjogeyJieXRlcyI6IDc5OTkzLCAic2hhMjU2IjogIjY1ZjE1NDc2ZjNjMjFlYTE1NmVlYzU1ZmEzMDBlZTQzZDNlNjExMjEzNDBjNjI4ZDU2ODk5ZGZlNWVjMGU5NDYifSwgInJlbmRlcmVkX3YyL3N0ZW1fY3VydmVzX3YxLnBkZiI6IHsiYnl0ZXMiOiA0MDU1MywgInNoYTI1NiI6ICJkNzllYjkyNThmZTBmMDAxNzQ0NDUyNGZhZDA2Mjk4NDZiNWY1N2Y3ZDQ0YzE1NzI3ZjEzOWM2ZjkyNDUyYmUwIn19fQ==\",\"sha256\":\"5735b64a47a3aa08e9e46ede581a8e705800b328ebc3cbe5b91bb5b9de9d1dcd\",\"bytes\":727}");b=base64.b64decode(p['base64'],validate=True)
assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256']
with (OUT/'remote_render_payload_v2.json').open('xb') as f:f.write(b)
remote=json.loads(b);assert remote['exit_code']==0
assert set(remote['paths'])=={'rendered_v2/plot_report.json'}|{f'rendered_v2/stem_curves_v1.{s}' for s in ('png','svg','pdf')}
for name,m in remote['paths'].items():
    path=OUT/name;assert path.stat().st_size==m['bytes'] and sha(path)==m['sha256'],name
data=read(OUT/'plot_data_v2.json');r=read(OUT/'rendered_v2/plot_report.json')
assert r['status']=='RENDERED_VERIFIED_SAVED_LOG_CURVES' and r['data_sha256']==sha(OUT/'plot_data_v2.json')
assert r['plot_code_sha256']==sha(ROOT/'英文小论文/code/stem_curves_20261011_v1/plot.py')
assert r['source_rows']==len(data['rows'])==87 and len(r['plotted_series'])==20
assert r['model_forwards']==r['epochs_added']==0
assert len({(s['seed'],s['arm'],s['metric']) for s in r['plotted_series']})==20
for s in r['plotted_series']:
    expected=sorted([row for row in data['rows'] if row['seed']==s['seed'] and row['arm']==s['arm']],key=lambda row:row['epoch'])
    assert s['epochs']==[row['epoch'] for row in expected] and s['values']==[row[s['metric']] for row in expected]
assert sum(len(s['epochs']) for s in r['plotted_series'])==174
v=dict(status='PASS_RENDER_TRANSFER_AND_ALL_ACTUAL_SERIES_VALUES',actual_source_rows=87,plotted_series=20,
    all_points_exact_to_source_data=True,data_sha256=sha(OUT/'plot_data_v2.json'),plot_report_sha256=sha(OUT/'rendered_v2/plot_report.json'),
    remote_payload_sha256=p['sha256'],visual_QA_pending=True,model_forwards=0,evaluator_calls=0,optimizer_updates=0,
    fixed12_candidate_EMA_verified=False,scope=data['scope'])
with (OUT/'render_verification_v2.json').open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
print(v['status'])

