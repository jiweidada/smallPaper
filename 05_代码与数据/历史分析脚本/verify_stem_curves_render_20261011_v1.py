"""Check downloaded plots against exact remote bytes and the86 real source rows."""
from pathlib import Path
import base64,hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stem_curves_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
payload=json.loads("{\"base64\":\"eyJleGl0X2NvZGUiOiAwLCAib3V0cHV0IjogIntcInN0YXR1c1wiOiBcIlJFTkRFUkVEX1ZFUklGSUVEX1NBVkVEX0xPR19DVVJWRVNcIiwgXCJzb3VyY2Vfcm93c1wiOiA4NiwgXCJwbG90X2NvZGVfc2hhMjU2XCI6IFwiY2VmYjY0ZjIwYWU5MWU3YWMyYTVmOTBjOGFkODU3NWYzMTFlODJlNmI1OWFjODNmMWYyM2YxMmM4MzQzODk3ZVwifVxuIiwgInBhdGhzIjogeyJyZW5kZXJlZC9wbG90X3JlcG9ydC5qc29uIjogeyJieXRlcyI6IDEwMzM0LCAic2hhMjU2IjogImQ5OGE5ZDRkZjQ4YThiOWY2NTJkM2ZmNzQ3NzRiMmNjYTY1MTQ5NjJlODg3NGIzZjY0MGZhYWM0MmUyOTM1YzgifSwgInJlbmRlcmVkL3N0ZW1fY3VydmVzX3YxLnBuZyI6IHsiYnl0ZXMiOiAyNTU5ODksICJzaGEyNTYiOiAiZTVhYzI5MzBhYjRiYzdkYmI2NTM4YTRkODMzY2QxOTkwODMxYzE0ZTgzYTQ5OTUxNWZmZDcwOTkwOTg0NzhhNiJ9LCAicmVuZGVyZWQvc3RlbV9jdXJ2ZXNfdjEuc3ZnIjogeyJieXRlcyI6IDc5NjQ1LCAic2hhMjU2IjogImUyNzhkNTdmODI0YzAyOWQ5NTM2NTFlYmZlMjM5Yjk0ZTg0ZjA2OGFmYTk3OWM2MmE5MmIxMTdhZTdhMzA1ZTcifSwgInJlbmRlcmVkL3N0ZW1fY3VydmVzX3YxLnBkZiI6IHsiYnl0ZXMiOiA0MDUxNiwgInNoYTI1NiI6ICI5MDI2NTdmNDMxZWJlNjJhYjIxMmRiNWE5ZWVkYzYwNmZhN2RiNzM5MjY3MzUxMTZkYzgxZmM1ZGJiNGI0MWU4In19fQ==\",\"sha256\":\"9e75f2dd912dcf7fd97ec95a2a489b49db7e2f85b1e0a2b2f813ff5e4abc1ecd\",\"bytes\":715}");b=base64.b64decode(payload['base64'],validate=True)
assert len(b)==payload['bytes'] and hashlib.sha256(b).hexdigest()==payload['sha256']
with (OUT/'remote_render_payload_v1.json').open('xb') as f:f.write(b)
remote=json.loads(b);assert remote['exit_code']==0
assert set(remote['paths'])=={'rendered/plot_report.json'}|{f'rendered/stem_curves_v1.{s}' for s in ('png','svg','pdf')}
for name,m in remote['paths'].items():
    path=OUT/name;assert path.stat().st_size==m['bytes'] and sha(path)==m['sha256'],name
data=read(OUT/'plot_data_v1.json');report=read(OUT/'rendered/plot_report.json')
assert report['status']=='RENDERED_VERIFIED_SAVED_LOG_CURVES'
assert report['data_sha256']==sha(OUT/'plot_data_v1.json')
assert report['plot_code_sha256']==sha(ROOT/'英文小论文/code/stem_curves_20261011_v1/plot.py')
assert report['source_rows']==len(data['rows'])==86
assert report['model_forwards']==report['epochs_added']==0
assert len(report['plotted_series'])==20
unique=set()
for series in report['plotted_series']:
    key=(series['seed'],series['arm'],series['metric']);assert key not in unique;unique.add(key)
    expected=sorted([r for r in data['rows'] if r['seed']==series['seed'] and r['arm']==series['arm']],key=lambda r:r['epoch'])
    assert series['epochs']==[r['epoch'] for r in expected]
    assert series['values']==[r[series['metric']] for r in expected]
assert sum(len(s['epochs']) for s in report['plotted_series'])==2*len(data['rows'])
verification=dict(status='PASS_RENDER_TRANSFER_AND_ALL_ACTUAL_SERIES_VALUES',
    data_sha256=sha(OUT/'plot_data_v1.json'),plot_report_sha256=sha(OUT/'rendered/plot_report.json'),
    remote_payload_sha256=payload['sha256'],plotted_series=20,actual_source_rows=86,
    all_points_exact_to_source_data=True,visual_QA_pending=True,
    new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
    fixed12_candidate_EMA_verified=False,scope=data['scope'])
with (OUT/'render_verification_v1.json').open('x',encoding='utf8') as f:json.dump(verification,f,ensure_ascii=False,indent=2)
print(verification['status'])

