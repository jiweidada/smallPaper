"""Verify every downloaded raw file and module invocation without new forwards."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'outputs/candidate_all_module_capture_20261010_v1/seed2_download_v1'


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


def main():
    archive=DEST/'module_candidate_panel_seed2_raw_v1.zip'
    meta=json.loads((DEST/'module_candidate_panel_seed2_raw_v1_metadata.json').read_text())
    assert meta['sha256']=='4d25184e2615ad01ebb36fbb6da1ef6241fafc28c953787666da011cb2aeddfa'
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    reports=[]; total_bytes=0
    with zipfile.ZipFile(archive) as z,zipfile.ZipFile(DEST/'module_candidate_FDR_subset_seed2_v1.zip') as subset:
        names=z.namelist();assert len(names)==len(set(names))
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        raw=z.read('capture/manifest.json')
        assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256']=='38d9a79310ee73e75d4406b76f85a9017d2afb12e27507b0c0d724eecec3720d'
        manifest=json.loads(raw)
        for name,v in manifest['files'].items():
            h=hashlib.sha256();size=0
            with z.open('capture/'+name) as stream:
                for b in iter(lambda:stream.read(8*1024**2),b''):
                    h.update(b);size+=len(b)
            assert h.hexdigest()==v['sha256'] and size==v['bytes'],name
            total_bytes+=size
        assert total_bytes==manifest['total_bytes']==meta['raw_capture_bytes']
        completion=json.loads(z.read('control/module_candidate_completion_seed2_v1.json'))
        assert completion['exit_code']==0 and not completion['timed_out']
        assert completion['status']=='CANDIDATE_MODULE_PANEL_COMPLETE'
        verification=json.loads(z.read('capture/verification.json'))
        assert verification==completion['verification']
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
                with np.load(io.BytesIO(subset.read(f'stages/image_{iid:06d}.npz')),allow_pickle=False) as stage:
                    assert np.array_equal(outputs['pred_boxes'][i],stage['final_boxes'])
                    assert np.array_equal(outputs['pred_logits'][i],stage['final_logits'])
            reports.append(dict(batch=bi,images=record['metadata']['image_ids'],module_calls=648,
                actually_called_modules=633,registered_modules=683,unique_blobs=len(unique),
                every_shape_dtype_and_blob_hash_verified=True,final_boxes_and_logits_exact=True))
    report=dict(status='PASS_DOWNLOADED_ALL_CALLED_MODULE_RAW_OUTPUTS',seed=2,archive_sha256=meta['sha256'],
        verified_capture_files=len(manifest['files']),verified_raw_bytes=total_bytes,batches=reports,
        original_capture_exit_code=0,new_model_forwards=0,epochs_added=0,
        functional_operations_outside_module_hooks_not_claimed=True,whole_dataset_coverage_not_claimed=True)
    with (DEST/'raw_verification_report_v1.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
