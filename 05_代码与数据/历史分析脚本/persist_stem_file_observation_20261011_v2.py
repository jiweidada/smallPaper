from pathlib import Path
import base64,hashlib,json
ROOT=Path.cwd();OUT=ROOT/'outputs/stem_observation_file_transport_20261011_v1'
source=json.loads(r'''{"1":{"base64":"eyJzZWVkIjogMSwgImZpbGUiOiAiLzI1MDg1NDAwMTIwL0RFSU0vZXhwZXJpbWVudHMvc3RlbV9vYnNlcnZhdGlvbl9maWxlX3RyYW5zcG9ydF8yMDI2MTAxMV92MS9zZWVkMV9vYnNlcnZhdGlvbl92Mi5wYXlsb2FkLmpzb24iLCAicGF5bG9hZF9ieXRlcyI6IDM3MDE1MSwgInBheWxvYWRfc2hhMjU2IjogIjlmNDY4NWVmNDYwM2NiNmFkYjBiNDYyYjc0NDc5NTUyZmVhNTRkNjc4NWY0MDhkYzU2NTgzY2YwMjcwNjMxYTIiLCAib2JzZXJ2ZWRfcmVtb3RlX1VUQyI6ICIyMDI2LTEwLTExVDEyOjQzOjA0Ljg2Mjg4MiswMDowMCIsICJzYW1lX29ic2VydmVyX3NvdXJjZV9zaGEyNTYiOiAiZjNjYTFjNzk4NjUyYmE3YjhmMjMzMDI1NmYzMjFmODZiNDU4MDZiNmYzNWNjYWQyOTVhY2ZlYTBiMTJiZmZhNiIsICJlcG9jaHMiOiB7Im9yaWdpbmFsX3N0ZW0iOiAxMiwgImRldGFpbF9sb3dyZXMiOiAxMCwgImhpZ2hmcmVxX3JnYiI6IDEwLCAiYXR0ZW50aW9uX3NwYXRpYWxfbWVhbiI6IDAsICJlYXJseV9maW5hbF9kb3duc2FtcGxlIjogMH0sICJub19tb2RlbF9mb3J3YXJkcyI6IHRydWUsICJub190cmFpbmluZ19yZXN0YXJ0cyI6IHRydWV9","expected_sha256":"df7fdac552b34ea7f8cecfe559fcab9f33bd46f0d57c360e5dfaa4efe6b4a6b8","expected_bytes":588},"2":{"base64":"eyJzZWVkIjogMiwgImZpbGUiOiAiLzI0MDg1NDAzMDE0L0RFSU0vZXhwZXJpbWVudHMvc3RlbV9vYnNlcnZhdGlvbl9maWxlX3RyYW5zcG9ydF8yMDI2MTAxMV92MS9zZWVkMl9vYnNlcnZhdGlvbl92Mi5wYXlsb2FkLmpzb24iLCAicGF5bG9hZF9ieXRlcyI6IDk3NzI4MiwgInBheWxvYWRfc2hhMjU2IjogImQyNTQzMDI1ZTI2MWMwMjQ4NDRmMGMyOTljN2U2ZjE3YWM2Y2IyMWFlMGVkYjY0N2FmZjYxYjdhNzVkMjAxNWUiLCAib2JzZXJ2ZWRfcmVtb3RlX1VUQyI6ICIyMDI2LTEwLTExVDEyOjQzOjA0LjgyNzk4MCswMDowMCIsICJzYW1lX29ic2VydmVyX3NvdXJjZV9zaGEyNTYiOiAiZjNjYTFjNzk4NjUyYmE3YjhmMjMzMDI1NmYzMjFmODZiNDU4MDZiNmYzNWNjYWQyOTVhY2ZlYTBiMTJiZmZhNiIsICJlcG9jaHMiOiB7Im9yaWdpbmFsX3N0ZW0iOiAxMiwgImRldGFpbF9sb3dyZXMiOiAxMiwgImhpZ2hmcmVxX3JnYiI6IDEyLCAiYXR0ZW50aW9uX3NwYXRpYWxfbWVhbiI6IDAsICJlYXJseV9maW5hbF9kb3duc2FtcGxlIjogMH0sICJub19tb2RlbF9mb3J3YXJkcyI6IHRydWUsICJub190cmFpbmluZ19yZXN0YXJ0cyI6IHRydWV9","expected_sha256":"ab0d1b54b1f17deba9edf9d183689627cdc1d0c9845c49f4dcc950182bdcc191","expected_bytes":588}}''')
payloads={}
for seed in (1,2):
    s=source[str(seed)];raw=base64.b64decode(s['base64'],validate=True)
    assert len(raw)==s['expected_bytes'] and hashlib.sha256(raw).hexdigest()==s['expected_sha256']
    meta=json.loads(raw);assert meta['seed']==seed and meta['no_model_forwards'] and meta['no_training_restarts']
    assert meta['same_observer_source_sha256']==hashlib.sha256((ROOT/'outputs/stem_component12_20261011_v1/observe_payload64_v1.py').read_bytes()).hexdigest()
    path=OUT/f'seed{seed}_observation_v2.payload.json';b=path.read_bytes()
    assert len(b)==meta['payload_bytes'] and hashlib.sha256(b).hexdigest()==meta['payload_sha256']
    with (OUT/f'seed{seed}_metadata_v2.json').open('xb') as f:f.write(raw)
    payloads[str(seed)]=dict(file=path.name,bytes=len(b),sha256=meta['payload_sha256'])
with (OUT/'observation_transport_v2.json').open('x',encoding='utf8') as f:
    json.dump(dict(schema='stem_live_observation_transport_v1',observer_source='outputs/stem_component12_20261011_v1/observe_payload64_v1.py',payloads=payloads,
      scope='Unchanged read-only observer, exact raw JSON transferred via SCP instead of PTY text.'),f,indent=2)
print('PASS_EXACT_REMOTE_FILE_OBSERVATION_METADATA_AND_PAYLOADS')

