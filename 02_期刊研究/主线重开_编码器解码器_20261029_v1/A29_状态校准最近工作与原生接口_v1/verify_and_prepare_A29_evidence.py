"""Verify archived primary sources,previous evidence,and six raw D0 files against tar."""
from pathlib import Path
import datetime,hashlib,json,sys,tarfile
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;N=R.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert not (R/'source_and_cache_verification.json').exists()
    prior=read(N/'67_A28原文源码与候选范围核验回执.json')
    for p,h in prior['files_sha256'].items():assert sha(p)==h,p
    rows=read(R/'new_core_records.json')['rows']
    for row in rows:
        pdf=Path(row['local_pdf']);side=Path(row['access_record']);meta=read(side)
        assert pdf.read_bytes()[:5]==b'%PDF-'
        assert sha(pdf)==meta['sha256']
        text=pdf.parent/'_text'/(row['stem']+'.txt');assert text.exists()
        if 'text_sha256' in meta:assert sha(text)==meta['text_sha256']
        meta.update(title=row['title'],authors=row['authors'],publication_status=row['tier'],text_file=str(text),text_sha256=sha(text),read_scope=row['read_scope'],read_by='assistant',human_read_attested=False)
        write(side,meta);row.update(pdf_sha256=sha(pdf),text_sha256=sha(text))
    write(R/'new_core_records.json',{'rows':rows})
    blobs=read(R/'selected_source_manifest.json')['rows'];assert len(blobs)==9
    for x in blobs:
        data=Path(x['local']).read_bytes();assert hashlib.sha256(data).hexdigest()==x['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==x['git_blob_sha1']
        meta=R/'repo_metadata'/x['repo'].replace('/','__');commit=read(meta/'commit.json');tree=read(meta/'tree.json')
        assert commit['sha']==x['commit'];node=next(t for t in tree['tree'] if t['path']==x['path']);assert node['sha']==x['git_blob_sha1']
    cfg=read(R/'D0_cache_audit_protocol_FROZEN.json');files=cfg['files'];raw=Path(files['protocol']['path']).parent
    wanted={str(Path(v['path']).relative_to(raw)).replace('\\','/'):v for k,v in files.items() if k!='archive'}
    checked=[]
    with tarfile.open(files['archive']['path'],'r|gz') as tar:
        for member in tar:
            if member.name not in wanted:continue
            assert member.isfile();h=hashlib.sha256();f=tar.extractfile(member)
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==wanted[member.name]['sha256'],member.name;checked.append(member.name)
    assert set(checked)==set(wanted)
    cache_verify=read(R/'D0_cache_independent_verification.json');assert cache_verify['status'].startswith('PASS_') and cache_verify['GT_rows_checked']==1718
    assert cache_verify['result_sha256']==sha(R/'D0_cache_candidate_audit_result.json')
    math=read(R/'formula_and_geometry_result.json');assert math['protocol_sha256']==sha(R/'formula_and_geometry_protocol_FROZEN.json')
    assert len(math['affine_cases'])==12 and len(math['geometry_cases'])==4
    metrics=read(files['metrics']['path']);thresholds=[{k:x[k] for k in ['confidence','iou','small_TP','small_FN','FP']} for x in metrics['matches'] if x['iou']==.5]
    out={'status':'PASS_PRIMARY_BYTE_SCOPE_AND_ALL1718_CACHED_MISS_BINS','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_A28_files_verified':len(prior['files_sha256']),'new_primary_PDFs':len(rows),'verified_author_git_blobs':len(blobs),'D0_raw_files_matched_original_archive':checked,'cached_small_miss_rows_independently_recomputed':1718,'historical_D0_threshold_table':thresholds,'new_native_model_or_scan_calls':0,'new_optimizer_updates':0,'new_GPU_model_calls':0,'mainline_validated':False,'not_proven':['UAV scan failure caused by resolution scaling','P novelty or AP benefit','D0 encoder feature loss','D0 score correction at matched FP','a new architecture contribution','heldout generalization']}
    write(R/'source_and_cache_verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['not_proven','historical_D0_threshold_table']},ensure_ascii=False))
if __name__=='__main__':main()
