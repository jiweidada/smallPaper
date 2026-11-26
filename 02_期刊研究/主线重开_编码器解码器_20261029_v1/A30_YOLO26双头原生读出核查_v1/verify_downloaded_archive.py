"""Verify the terminal remote job and downloaded archive before reading metrics."""
from pathlib import Path
import hashlib,json,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    complete=read(R/'remote_complete.json');archive=R/'remote_evidence.zip';assert complete['status']=='COMPLETED' and complete['exit_code']==0
    assert archive.stat().st_size==complete['archive_bytes'] and sha(archive)==complete['archive_sha256']
    receiptfile=R.parent.parent/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'/'ssh_operation_receipts.jsonl'
    calls=[json.loads(x) for x in receiptfile.read_text(encoding='utf-8').splitlines() if 'A30_poll_native_job_05' in x]
    call=next(x for x in reversed(calls) if x.get('action')=='exec');poll=json.loads(call['stdout']);assert poll['process_state'] is None and poll['complete']['exit_code']==0 and poll['failure'] is None
    dest=R/'downloaded_evidence';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p=Path(info.filename);assert not p.is_absolute() and '..' not in p.parts
            assert (dest/p).resolve().is_relative_to(dest.resolve())
        assert z.testzip() is None;z.extractall(dest)
    manifest=read(dest/'evidence_manifest.json');assert len(manifest)==complete['manifest_files']
    for name,h in manifest.items():assert sha(dest/name)==h,name
    inputs=read(R/'execution_inputs_manifest.json')
    for name,h in inputs.items():assert sha(R/name)==sha(dest/name)==h,name
    result=read(dest/'result.json');assert result['counts']==complete['counts']
    assert result['counts']=={'whole_detector_forwards':101,'standalone_native_TAL_forward_calls':200,'extra_native_head_inference_decodes':200,'all_native_head_inference_decodes':301,'optimizer_updates':0}
    assert result['fidelity_all100_passed'] and result['native_end2end_repeat_bitwise_equal'] and result['weights_source_images_unchanged']
    assert len(read(dest/'cached_prediction_fidelity.json'))==100
    assert len(list((dest/'captures').glob('case_*.npz')))==100
    out={'status':'PASS_TERMINAL_IDENTITY_ARCHIVE119_FILES_AND_FROZEN_INPUTS','archive_sha256':complete['archive_sha256'],'archive_bytes':complete['archive_bytes'],'manifest_files':len(manifest),'pid':poll['pid'],'terminal_process_absent':True,'remote_exit_code':0,'counts':result['counts'],'new_algorithm_or_mainline_validated':False}
    with (R/'archive_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(out,ensure_ascii=False))
    print(json.dumps({'native_O_minus_M_small_AP':result['native_O_minus_M_input_small_AP'],'native_paired95':result['native_O_minus_M_paired_image_bootstrap95'],'metrics':{k:{'AP':v['input_area_AP'],'FP1200':next(x for x in v['same_FP'] if x['FP_budget']==1200)} for k,v in result['metrics'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
