"""Verify existing pinned Ultralytics sources against public Git tree; no imports."""
from pathlib import Path
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;E=R.parent.parent
PIN='2004cba021c68a46429be460e26954ab171b2862'
LOCAL=E/'前期调研_码率受限航拍小目标_20261028_v1'/'深入核查_20261028_v1'/'source_code'/'v8.4.191_2004cba021c6'
RECEIPTS=E/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'/'ssh_operation_receipts.jsonl'
FILES=['ultralytics/nn/modules/head.py','ultralytics/utils/loss.py','ultralytics/utils/tal.py','ultralytics/nn/tasks.py','ultralytics/engine/trainer.py','ultralytics/cfg/models/26/yolo26.yaml','ultralytics/utils/ops.py']
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-fixed-head-source-audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=25) as f:return f.read()
def save(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
def jsave(p,x):save(p,(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def main():
    assert not (R/'pinned_source_manifest.json').exists()
    raw=get('https://api.github.com/repos/ultralytics/ultralytics/git/commits/'+PIN);save(R/'public_commit.json',raw);commit=json.loads(raw);assert commit['sha']==PIN
    raw=get('https://api.github.com/repos/ultralytics/ultralytics/git/trees/'+commit['tree']['sha']+'?recursive=1');save(R/'public_tree.json',raw);tree=json.loads(raw);assert not tree.get('truncated');index={x['path']:x for x in tree['tree'] if x['type']=='blob'}
    nms='ultralytics/utils/nms.py';data=get('https://raw.githubusercontent.com/ultralytics/ultralytics/'+PIN+'/'+nms);save(R/'source'/nms,data)
    files=[(x,LOCAL/x) for x in FILES]+[(nms,R/'source'/nms)];rows=[]
    for name,path in files:
        data=path.read_bytes();git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert git==index[name]['sha'],name
        rows.append({'path':name,'local':str(path),'commit':PIN,'git_blob_sha1':git,'sha256':hashlib.sha256(data).hexdigest(),'url':'https://github.com/ultralytics/ultralytics/blob/'+PIN+'/'+name,'execution':'static audit;no fetched code imported by this script'})
    calls=[json.loads(x) for x in RECEIPTS.read_text(encoding='utf-8').splitlines() if 'A30_host_preflight_readonly_v1' in x]
    call=next(x for x in reversed(calls) if x.get('action')=='exec');assert call['exit_code']==0
    host=json.loads(call['stdout']);assert host['ultralytics']=='8.4.191'
    for name,h in host['files'].items():assert h==next(x['sha256'] for x in rows if x['path']=='ultralytics/'+name),name
    jsave(R/'host_preflight_readonly.json',{'call':call,'decoded':host})
    jsave(R/'pinned_source_manifest.json',{'commit':PIN,'files':rows,'host5_sources_match_pin':True,'source_read_scope':'Selected Detect full-sequence forward/decode/postprocess,DetectionModel criterion selection,E2ELoss,v8DetectionLoss score target and TAL geometry/ranking/conflict/normalization;not full repository or training reproduction.'})
    print(json.dumps({'status':'EIGHT_PINNED_GIT_BLOBS_VERIFIED_AND_FIVE_HOST_FILES_MATCH','source_files':len(rows),'new_model_calls':0}))
if __name__=='__main__':main()
