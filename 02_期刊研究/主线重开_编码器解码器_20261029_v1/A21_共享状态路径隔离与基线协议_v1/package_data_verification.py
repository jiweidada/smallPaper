"""Package local manifests, original cohort files and CPU-only checks for upload."""
from pathlib import Path
import ast,hashlib,json,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;DATA=Path(r'D:\数据集\visdrone2019\VisDrone2019-DET-val')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (O/'data_and_controls.zip').exists(),'Do not overwrite a deployed package'
    cohort=json.loads((O/'scientific_cohort_asset_manifest.json').read_text(encoding='utf-8'))
    assert len(cohort['cases'])==33 and cohort['train_exact_image_hash_overlap']==0 and cohort['train_filename_prefix_overlap_count']==6
    names=['train_asset_manifest.json','scientific_cohort_asset_manifest.json','cohort_pixel_control_rules_FROZEN.json','verify_assets_and_pixel_controls.py']
    ast.parse((O/'verify_assets_and_pixel_controls.py').read_text(encoding='utf-8'))
    evidence=[]
    with zipfile.ZipFile(O/'data_and_controls.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(O/n,n);evidence.append({'path':n,'sha256':sha(O/n)})
        for c in cohort['cases']:
            p=DATA/'images'/c['image'];a=DATA/'annotations'/(p.stem+'.txt')
            assert sha(p)==c['image_sha256'] and sha(a)==c['annotation_sha256']
            for f in [p,a]:z.write(f,'cohort_inputs/'+f.name);evidence.append({'path':'cohort_inputs/'+f.name,'sha256':sha(f)})
    value={'sha256':sha(O/'data_and_controls.zip'),'bytes':(O/'data_and_controls.zip').stat().st_size,'member_count':len(evidence),
           'files':evidence,'purpose':'CPU data/control verification only; no training/forward launch'}
    (O/'data_and_controls_package_receipt.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:value[k] for k in ['sha256','bytes','member_count']}))
if __name__=='__main__':main()
