"""Verify provenance and package only isolated CPU construction assets, not a GPU job."""
from pathlib import Path
import ast, hashlib, json, shutil, sys, zipfile
sys.stdout.reconfigure(encoding='utf-8')
O = Path(__file__).resolve().parent
R = O.parent
S = R / 'source_audit_v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    assert not (O/'cpu_preparation_bundle.zip').exists(), 'Do not recreate a previously deployed bundle'
    old=json.loads((R/'49_A19方法与宿主资产核验回执.json').read_text(encoding='utf-8'))
    for x in old['files']: assert sha(R/x['path'])==x['sha256']
    tree=json.loads((S/'A18_VSSD_tree.json').read_text(encoding='utf-8'))
    files=[('classification/models/mamba2.py','A18_VSSD_mamba2.py','mamba2.py'),
           ('classification/models/mamba_util.py','A18_VSSD_mamba_util.py','mamba_util.py'),
           ('classification/config.py','A20_VSSD_classification_config.py','config.py'),
           ('classification/models/__init__.py','A20_VSSD_models_init.py','models_init.py'),
           ('classification/configs/vssd_iccv_version/vssd_tiny.yaml','A18_VSSD_classification_iccv_tiny.yaml','vssd_tiny.yaml')]
    evidence=[]
    (O/'vssd_source').mkdir(exist_ok=True)
    for source, local, name in files:
        b=(S/local).read_bytes()
        row=next(x for x in tree['tree'] if x['path']==source)
        blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert blob==row['sha'] and len(b)==row['size']
        (O/'vssd_source'/name).write_bytes(b)
        evidence.append({'author_path':source,'git_blob':blob,'sha256':hashlib.sha256(b).hexdigest()})
    # Verify all newly read provenance sources, including preprocessing and author requirements.
    for source, local in [('requirements.txt','A20_VSSD_requirements.txt'),('classification/data/build.py','A20_VSSD_data_build.py'),
                          ('detection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py','A20_VSSD_swin_detection_base.py')]:
        b=(S/local).read_bytes(); row=next(x for x in tree['tree'] if x['path']==source)
        blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert blob==row['sha'] and len(b)==row['size']
        evidence.append({'author_path':source,'git_blob':blob,'sha256':hashlib.sha256(b).hexdigest()})
    wheels=[]
    for p in sorted((O/'wheels').glob('*.whl')):
        package=p.name.split('-')[0]
        metadata=json.loads((O/(package+'_pypi.json')).read_text(encoding='utf-8'))
        item=next(x for x in metadata['urls'] if x['filename']==p.name)
        assert sha(p)==item['digests']['sha256'] and p.stat().st_size==item['size']
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                dest=(O/'deps'/n).resolve()
                assert dest.is_relative_to((O/'deps').resolve()) and not n.startswith('/')
                if not n.endswith('/'):
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
        wheels.append({'filename':p.name,'sha256':sha(p),'bytes':p.stat().st_size,
                       'official_PyPI_url':item['url'],'requires_python':metadata['info']['requires_python']})
    assert len(wheels)==3
    ast.parse((O/'a20_host.py').read_text(encoding='utf-8'))
    packed=[O/'a20_host.py']+sorted((O/'vssd_source').glob('*'))+sorted((O/'deps').rglob('*'))
    packed=[p for p in packed if p.is_file()]
    manifest={'status':'CPU_CONSTRUCTION_ONLY_NO_FORWARD_JOB','source_commit':'c10de2f8b717a04459b09cbd651de2e768f302c8',
              'source_blobs':evidence,'wheels':wheels,
              'dependency_isolation':'ZIP wheel files extracted only into task/deps; no existing environment changed',
              'source_compatibility':'Unused fvcore import omitted in AST only; original files preserved, no profiler calls',
              'files':[{'path':str(p.relative_to(O)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p in packed]}
    wr(O/'bundle_manifest.json',manifest)
    with zipfile.ZipFile(O/'cpu_preparation_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in packed+[O/'bundle_manifest.json']:z.write(p,str(p.relative_to(O)).replace('\\','/'))
    wr(O/'bundle_download_upload_manifest.json',{'zip_sha256':sha(O/'cpu_preparation_bundle.zip'),
       'bytes':(O/'cpu_preparation_bundle.zip').stat().st_size,'member_count':len(packed)+1,'author_sources_verified':len(evidence),
       'wheels':wheels,'no_model_forward_or_GPU_invocation':True})
    print(json.dumps({'bundle_sha256':sha(O/'cpu_preparation_bundle.zip'),'bytes':(O/'cpu_preparation_bundle.zip').stat().st_size,
          'members':len(packed)+1,'source_blobs':len(evidence),'wheel_count':len(wheels),'model_forwards':0}))

if __name__=='__main__':main()
