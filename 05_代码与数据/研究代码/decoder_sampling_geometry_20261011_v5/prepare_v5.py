"""Repair the diagnostic's constant reconstruction using actual saved EMA buffers."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))

old=HERE.parent/'decoder_sampling_geometry_20261011_v4'
audit=ROOT/'outputs/sampling_audit_20261011_v1'
p=read(old/'protocol.json')
assert sha(old/'analyze_saved_sampling.py')==p['script_sha256']
buffers=read(audit/'actual_checkpoint_buffers.json')
assert buffers['schema']=='verified_checkpoint_actual_constant_buffers_v1'
assert sha(audit/'read_checkpoint_buffers.py')==buffers['reader_sha256']
src=(old/'analyze_saved_sampling.py').read_text(encoding='utf8')
src=src.replace('saved_native_decoder_sampling_geometry_v4','saved_native_decoder_sampling_geometry_v5')
src=src.replace('outputs/decoder_sampling_geometry_20261011_v4','outputs/decoder_sampling_geometry_20261011_v5')
needle="    rows=[];gt_rows=[];controls=[];derived={};consumed={}"
assert src.count(needle)==1
src=src.replace(needle,"""    checkpoint=old/'extracted/run/last.pth'
    assert sha(checkpoint)==binding['checkpoint_sha256']
    actual_scales=[]
    with zipfile.ZipFile(checkpoint) as saved:
        for layer,record in enumerate(p['actual_EMA_sampling_buffers'][str(seed)]):
            assert record['state']=='ema.module'
            assert record['key']==f'decoder.decoder.layers.{layer}.cross_attn.num_points_scale'
            raw=saved.read(record['storage_member'])
            assert hashlib.sha256(raw).hexdigest()==record['storage_sha256']
            descriptor=record['descriptor']
            assert descriptor['offset']==0 and descriptor['size']==[12] and descriptor['stride']==[1]
            assert descriptor['storage']['dtype']=='<f4' and len(raw)==48
            scale=np.frombuffer(raw,dtype='<f4').copy()
            assert np.array_equal(scale.view(np.uint32),np.array(record['uint32_bits'],dtype=np.uint32))
            actual_scales.append(scale)
    rows=[];gt_rows=[];controls=[];derived={};consumed={}""")
needle="                    scales=np.repeat(np.float32(1)/np.array(p['points_per_level'],dtype=np.float32),p['points_per_level'])"
assert src.count(needle)==1
src=src.replace(needle,"                    scales=actual_scales[layer]")
needle="            protocol_sha256=sha(HERE/'protocol.json'),script_sha256=sha(__file__),sources=binding,"
assert src.count(needle)==1
src=src.replace(needle,needle+"\n            actual_EMA_sampling_buffers=p['actual_EMA_sampling_buffers'][str(seed)],")
ast.parse(src);compile(src,'analyze_saved_sampling.py','exec')
with (HERE/'analyze_saved_sampling.py').open('x',encoding='utf8',newline='\n') as f:f.write(src)
p['schema']='saved_native_decoder_sampling_geometry_v5'
p['script_sha256']=sha(HERE/'analyze_saved_sampling.py')
p['registration_script_sha256']=sha(__file__)
p['parent_protocol_sha256']=sha(old/'protocol.json')
p['repair']='Use exact EMA num_points_scale storage from each saved checkpoint instead of reconstructing nominal 1/n constants. No metric, threshold, image, model or training change.'
p['actual_EMA_sampling_buffers']={str(s['seed']):[r for r in s['rows'] if r['state']=='ema.module' and 'num_points_scale' in r['key']] for s in buffers['seeds']}
for seed in ('1','2'):assert len(p['actual_EMA_sampling_buffers'][seed])==3
p['actual_buffer_source_report_sha256']=sha(audit/'actual_checkpoint_buffers.json')
p['actual_buffer_reader_sha256']=sha(audit/'read_checkpoint_buffers.py')
p['reconstruction_control']['arithmetic']+=' v5: all three sampling scale buffers read as exact float32 from the saved EMA checkpoint; not reconstructed from nominal recipe.'
with (HERE/'protocol.json').open('x',encoding='utf8') as f:json.dump(p,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(status='REGISTERED_V5_ACTUAL_EMA_BUFFER_REPAIR',protocol_sha256=sha(HERE/'protocol.json'),script_sha256=p['script_sha256'],threshold_changes=False,new_model_forwards=0)))
