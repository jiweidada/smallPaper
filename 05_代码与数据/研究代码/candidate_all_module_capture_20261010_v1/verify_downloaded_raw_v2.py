"""Verify every downloaded raw file and module invocation without new forwards."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def tensors(value):
    if isinstance(value,dict):
        if value.get('kind')=='tensor':yield value
        for v in value.values():yield from tensors(v)
    elif isinstance(value,list):
        for v in value:yield from tensors(v)


def main(seed):
    DEST=ROOT/f'outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1'
    target=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1'
    target_report=json.loads((target/'verification_report_v1.json').read_text())
    assert target_report['status']=='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    assert target_report['identity']['seed']==seed and target_report['identity']['arm']=='target_protect'
    stagepath=target/'extracted/run/stages_epoch12'
    stage_manifest=json.loads((stagepath/'manifest.json').read_text())
    archive=DEST/f'module_candidate_panel_seed{seed}_raw_v1.zip'
    meta=json.loads((DEST/f'module_candidate_panel_seed{seed}_raw_v1_metadata.json').read_text())
    assert len(meta['sha256'])==64 and meta['actual_capture_exit_code']==0
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    reports=[]; total_bytes=0
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        raw=z.read('capture/manifest.json')
        assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256']
        manifest=json.loads(raw)
        for name,key in [('raw_module_recorder.py','recorder_source_sha256'),('capture_candidate_panel.py','capture_source_sha256')]:
            assert hashlib.sha256(z.read('source/'+name)).hexdigest()==manifest[key]
        for name,v in manifest['files'].items():
            h=hashlib.sha256();size=0
            with z.open('capture/'+name) as stream:
                for b in iter(lambda:stream.read(8*1024**2),b''):
                    h.update(b);size+=len(b)
            assert h.hexdigest()==v['sha256'] and size==v['bytes'],name
            total_bytes+=size
        assert total_bytes==manifest['total_bytes']==meta['raw_capture_bytes']
        completion=json.loads(z.read(f'control/module_candidate_completion_seed{seed}_v1.json'))
        assert completion['exit_code']==0 and not completion['timed_out']
        assert completion['status']=='CANDIDATE_MODULE_PANEL_COMPLETE'
        verification=json.loads(z.read('capture/verification.json'))
        assert verification==completion['verification']
        assert verification['seed']==seed and verification['identity']==target_report['identity']
        expected={1:'e195f5a3d126bee9e0f7a373f5eb5fdebb78ada9fc891780e251be9dafc2ae94',2:'8113eb35c4ca0519e9757a101327cb95e709ebd3609559f55c240e1a3a472269'}
        assert verification['checkpoint_sha256']==expected[seed]
        assert verification['protocol_sha256']=='f251e2a4c8ebd4c4737d46a76aa71aede595d346d2e19b011bf60437de9a3fc9'
        assert verification['candidate_model_forwards']==2 and verification['baseline_model_forwards']==verification['epochs_added']==0
        assert verification['status']=='PASS_ALL_CALLED_MODULE_RAW_OUTPUTS_SELECTED_PANEL'
        for bi in (25,105):
            base=f'capture/batch_{bi:03d}/'
            record=json.loads(z.read(base+'record.json')); events=record['invocation_events']
            inventory=record['registered_modules']; counts={}
            assert len(events)==record['total_module_calls']==648
            assert len(inventory)==record['registered_module_count']==683
            for index,event in enumerate(events):
                assert event['call_id']==index and event['module'] in inventory and 'output' in event
                parent=event['parent_call_id']; assert parent is None or 0<=parent<index
                assert (parent is None)==(index==0)
                name=event['module']; assert event['invocation']==counts.get(name,0)
                counts[name]=counts.get(name,0)+1
            assert len(counts)==record['called_module_count']==633
            assert all(row['actual_calls']==counts.get(name,0) for name,row in inventory.items())
            descriptors=list(tensors([record['normalized_model_input'],record['native_targets'],events]))
            unique={}
            for desc in descriptors:
                digest=desc['blob_sha256'];assert desc['blob_file']=='blobs/'+digest+'.npy'
                if digest in unique:
                    assert unique[digest]==(desc['shape'],desc['storage_numpy_dtype'])
                    continue
                blob=z.read(base+desc['blob_file']);assert hashlib.sha256(blob).hexdigest()==digest
                array=np.load(io.BytesIO(blob),allow_pickle=False)
                assert list(array.shape)==desc['shape'] and str(array.dtype)==desc['storage_numpy_dtype']
                assert array.nbytes==desc['tensor_bytes'] and not array.dtype.hasobject
                if desc['encoding']=='bfloat16_raw_bits_as_uint16':
                    assert str(array.dtype)=='uint16' and desc['original_dtype']=='torch.bfloat16'
                else:
                    assert desc['encoding']=='numpy_exact_dtype'
                unique[digest]=(desc['shape'],desc['storage_numpy_dtype'])
            assert set(unique)==set(record['unique_physical_blobs'])
            model=events[0];assert model['module']=='model' and model['output']['kind']=='dict'
            out={item['key']['value']:item['value'] for item in model['output']['entries']}
            outputs={key:np.load(io.BytesIO(z.read(base+out[key]['blob_file'])),allow_pickle=False)
                     for key in ('pred_boxes','pred_logits')}
            for i,iid in enumerate(record['metadata']['image_ids']):
                file=stagepath/f'image_{iid:06d}.npz'
                assert sha(file)==stage_manifest['files'][file.name]
                with np.load(file,allow_pickle=False) as stage:
                    assert np.array_equal(outputs['pred_boxes'][i],stage['final_boxes'])
                    assert np.array_equal(outputs['pred_logits'][i],stage['final_logits'])
            reports.append(dict(batch=bi,images=record['metadata']['image_ids'],module_calls=648,
                actually_called_modules=633,registered_modules=683,unique_blobs=len(unique),
                every_shape_dtype_and_blob_hash_verified=True,final_boxes_and_logits_exact=True))
    report=dict(status='PASS_DOWNLOADED_ALL_CALLED_MODULE_RAW_OUTPUTS',seed=seed,archive_sha256=meta['sha256'],
        verified_capture_files=len(manifest['files']),verified_raw_bytes=total_bytes,batches=reports,
        original_capture_exit_code=0,new_model_forwards=0,epochs_added=0,
        functional_operations_outside_module_hooks_not_claimed=True,whole_dataset_coverage_not_claimed=True)
    with (DEST/'raw_verification_report_v1.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(1,2),required=True)
    main(parser.parse_args().seed)
