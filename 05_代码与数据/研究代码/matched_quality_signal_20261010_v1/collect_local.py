"""Assemble complete archives and independently compare native leaf gradients."""
from pathlib import Path
import base64, hashlib, json, tarfile, numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/matched_quality_signal_20261010_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=read(OUT/'protocol.json');descriptors=read(OUT/'transfer_descriptors.json');receipts=[];comparisons=[];coordinates=0;maximum_gradient_error=0.;maximum_iou_error=0.
    with np.load(OUT/'numpy_analysis/native_comparison_expected.npz',allow_pickle=False) as expected:
      for desc in descriptors:
        seed=desc['seed'];assert desc['next']==desc['bytes'];archive=OUT/f'collection_seed{seed}.tar.gz'
        with archive.open('xb') as f:
          for part in desc['parts']:
            encoded=(OUT/f'transfer_seed{seed}_parts'/part['name']).read_text().strip();assert len(encoded)==part['encoded_length'];decoded=base64.b64decode(encoded,validate=True);assert len(decoded)==part['count'];f.write(decoded)
        assert archive.stat().st_size==desc['bytes'] and sha(archive)==desc['sha256']
        target=OUT/f'collection_seed{seed}';target.mkdir(exist_ok=False)
        with tarfile.open(archive) as tf:tf.extractall(target,filter='data')
        manifest=read(target/'manifest.json')['files']
        for name,h in manifest.items():assert sha(target/name)==h,name
        assert (target/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes();assert sha(target/'verify_native.py')==protocol['code_sha256']['verify_native.py']
        report=read(target/'native_verification/verification.json');assert report['status']=='PASS_NATIVE_CPU_MAL_LEAF_AUTOGRAD' and report['leaf_logit_backward_calls']==160 and report['model_forwards']==report['model_backwards']==report['optimizer_steps']==report['training_epochs_added']==0
        native_path=target/'native_verification/matched_native_gradients.npz';assert sha(native_path)==report['output_npz_sha256']
        with np.load(native_path,allow_pickle=False) as native:
          keys=[k for k in expected.files if k.startswith(f'seed{seed}.')];assert set(keys)==set(native.files)
          for key in keys:
            actual=native[key];want=expected[key];assert actual.shape==want.shape and np.isfinite(actual).all()
            if key.endswith(('.GT_row','.query','.max_other_class')):assert np.array_equal(actual,want)
            elif key.endswith('.IoU'):
              error=float(np.max(np.abs(actual.astype(np.float64)-want)));maximum_iou_error=max(maximum_iou_error,error);assert error<=protocol['verification']['IoU_abs_tolerance']
            else:
              error=float(np.max(np.abs(actual.astype(np.float64)-want)));maximum_gradient_error=max(maximum_gradient_error,error);coordinates+=actual.size;assert np.allclose(actual,want,atol=protocol['verification']['native_leaf_gradient_atol'],rtol=protocol['verification']['native_leaf_gradient_rtol']),key
          comparisons.append({'seed':seed,'native_arrays_compared':len(keys),'matched_exposures':sum(native[k].size for k in native.files if k.endswith('.GT_grad')),'all_native_autograd_coordinates_compared_remotely':report['all_gradient_coordinates_compared'],'native_CPU_to_archived_CUDA_max_loss_error':report['maximum_CPU_to_archived_CUDA_loss_error'],'native_autograd_to_analytic_max_gradient_error':report['maximum_native_autograd_to_detached_analytic_gradient_error']})
        receipts.append({'seed':seed,'archive_bytes':desc['bytes'],'archive_sha256':desc['sha256'],'all_manifest_files_verified':len(manifest),'same_frozen_protocol':True,'same_frozen_code':True,'native_source_unchanged':report['native_source_sha_unchanged'],'native_source_files':report['native_files'],'process':read(target/'process.json')})
    result={'status':'PASS_BOTH_ARCHIVES_NATIVE_LOSS_AUTOGRAD_AND_INDEPENDENT_NUMPY_COMPARISON','seeds':receipts,'comparisons':comparisons,'gradient_coordinates_independently_compared':coordinates,'max_native_numpy_gradient_abs_error':maximum_gradient_error,'max_native_numpy_IoU_abs_error':maximum_iou_error,'total_native_MAL_leaf_backward_calls':320,'total_model_forward_calls_added':0,'total_model_backward_calls_added':0,'optimizer_steps':0,'training_epochs_added':0,'protocol_sha256':sha(OUT/'protocol.json'),'collector_sha256':sha(Path(__file__)),'AP_evidence':False,'root_cause_established':False}
    (OUT/'download_and_gradient_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
if __name__=='__main__':main()
