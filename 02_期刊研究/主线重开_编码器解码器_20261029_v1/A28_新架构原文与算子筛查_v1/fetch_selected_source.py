"""Fetch pinned author-source audit subsets and verify exact Git blob bytes."""
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
FILES={
 'CQU-ADHRI-Lab/MI-DETR':[
  'README.md','requirements.txt','projects/midetr/modeling/midetr_transformer.py',
  'projects/midetr/modeling/dino.py','projects/midetr/configs/models/midetr_r50.py',
  'projects/midetr/configs/midetr-resnet/midetr_r50_4scale_12ep.py',
  'detrex/layers/attention.py','detrex/layers/multi_scale_deform_attn.py'],
 'schnurrd/ICL-Architectures':[
  'README.md','pyproject.toml','PFNs/configs/linear_attention/linear_attention_config.py',
  'PFNs/pfns/model/linear_attention.py','PFNs/pfns/model/fla_patches.py',
  'PFNs/pfns/model/bidirectional_fla.py','PFNs/pfns/model/fla_state_passing.py',
  'PFNs/notebooks/minimal_linear_attention_seq_len_generalization.py']}

def main():
    out=[]
    for repo,names in FILES.items():
        meta=ROOT/'repo_metadata'/repo.replace('/','__')
        commit=json.loads((meta/'commit.json').read_text(encoding='utf-8'))['sha']
        tree=json.loads((meta/'tree.json').read_text(encoding='utf-8'))
        blobs={r['path']:r for r in tree['tree'] if r['type']=='blob'}
        source=ROOT/'source'/repo.replace('/','__');source.mkdir(parents=True,exist_ok=False)
        rows=[]
        for name in names:
            assert name in blobs
            url='https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+urllib.parse.quote(name,safe='/')
            req=urllib.request.Request(url,headers={'User-Agent':'UAV-literature-source-audit'})
            with urllib.request.urlopen(req,timeout=25) as r:data=r.read()
            gh=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert gh==blobs[name]['sha'],name
            dest=source/name;dest.resolve().relative_to(source.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            rows.append({'path':name,'url':url,'bytes':len(data),'git_blob_sha1':gh,'sha256':hashlib.sha256(data).hexdigest()})
        row={'repo':repo,'commit':commit,'source_root':str(source),'scope':'Selected operator/configuration/dependency subset, not an executable full clone or reproduction. Fetched code not imported or executed.',
             'verified_git_blobs':len(rows),'files':rows}
        out.append(row);print(json.dumps({k:v for k,v in row.items() if k!='files'}))
    (ROOT/'selected_source_manifest.json').write_text(json.dumps({'rows':out},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
