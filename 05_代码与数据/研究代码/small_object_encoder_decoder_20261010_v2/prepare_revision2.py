"""Explicit engineering correction; failed revision1 stays frozen/preserved."""
from pathlib import Path
import ast,datetime,difflib,hashlib,io,json,zipfile,base64
HERE=Path(__file__).resolve().parent;PREV=HERE.with_name('small_object_encoder_decoder_20261010_v1');ROOT=HERE.parents[2]
OUT=ROOT/'outputs/small_object_encoder_decoder_20261010_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    OUT.mkdir(exist_ok=False);files=['small_object_model.py','geometry_memory.py','preflight.py','controller.py']
    old=json.loads((PREV/'preflight_protocol.json').read_text())
    assert all(sha(PREV/n)==v for n,v in old['implementation_sha256'].items())
    assert sha(PREV/'geometry_memory.py')==sha(HERE/'geometry_memory.py')
    diffs=[]
    for n in files:
        ast.parse((HERE/n).read_text(),filename=n)
        diffs.extend(difflib.unified_diff((PREV/n).read_text().splitlines(True),(HERE/n).read_text().splitlines(True),fromfile='revision1/'+n,tofile='revision2/'+n))
    (OUT/'engineering_correction.diff').write_text(''.join(diffs),encoding='utf8')
    correction={'kind':'ENGINEERING_PREFLIGHT_CORRECTION_ONLY','reason':'CPU checkpoint tensors were compared directly against CUDA model tensors during repeated initialization verification. Failed before any candidate forward; native fixture5 forwards0 backwards0 optimizer updates.','change':'Cache loaded state once, compare native tensor on its original device, assert dtype unchanged. Isolated revision2 root and output. Architecture, tensor contract, inputs, checks and exact equality tolerances unchanged.','failed_revision1_protocol_sha256':sha(PREV/'preflight_protocol.json'),'failed_revision1_remote_root':'/25085400120/DEIM/experiments/small_object_encoder_decoder_20261010_v1','failed_records_preserved':True,'automatic_retry':False,'architecture_change':False,'comparison_tolerance_change':False,'diff_sha256':sha(OUT/'engineering_correction.diff')}
    save(OUT/'engineering_correction.json',correction)
    protocol=dict(old);protocol['status']='FROZEN_ENGINEERING_REVISION2_BEFORE_EXECUTION';protocol['frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();protocol['implementation_sha256']={n:sha(HERE/n) for n in files};protocol['engineering_correction_sha256']=sha(OUT/'engineering_correction.json');protocol['prior_failed_forwards']=5;protocol['prior_failed_backwards']=0;protocol['cumulative_max_forwards_this_design']=22;protocol['cumulative_max_backwards_this_design']=6
    save(HERE/'preflight_protocol.json',protocol);save(OUT/'preflight_protocol.json',protocol)
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for n in files+['preflight_protocol.json']:z.writestr(n,(HERE/n).read_bytes())
    data=b.getvalue();(OUT/'deploy_bundle.zip').write_bytes(data);(OUT/'deploy_bundle.zip.b64').write_text(base64.b64encode(data).decode())
    print(json.dumps({'status':'FROZEN_REVISION2','protocol_sha256':sha(HERE/'preflight_protocol.json'),'zip_size':len(data),'zip_sha256':hashlib.sha256(data).hexdigest()}))
    print(base64.b64encode(data).decode())
if __name__=='__main__':main()
