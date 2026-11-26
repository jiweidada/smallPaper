"""Observe existing tensors in the final native validation pass; never rerun a head."""
from pathlib import Path
from results import read, write, sha


class StageTrace:
    def __init__(self, model, identity, annotation_path, mode='native_runtime'):
        if mode not in ('native_runtime', 'synthetic_engineering'):
            raise ValueError('Unknown trace mode')
        if sha(annotation_path) != identity['val_annotation_sha256']:
            raise ValueError('Trace annotation identity differs')
        annotation = read(annotation_path)
        self.images = {int(x['id']): x for x in annotation['images']}
        if len(self.images) != len(annotation['images']):
            raise ValueError('Duplicate image identifiers')
        self.identity, self.mode = identity, mode
        self.model, self.pending = model, None
        self.records, self.errors, self.handles = {}, [], []
        self.closed = False
        try:
            decoder = model.decoder.decoder
            self.handles.append(decoder.layers[0].register_forward_pre_hook(self._initial_hook))
            self.handles.append(decoder.register_forward_hook(self._decoder_hook))
            self.handles.append(model.register_forward_hook(self._model_hook))
        except Exception:
            self.close()
            raise

    def _observe(self, operation):
        # Observation errors must not alter inference or prevent native last.pth saving.
        if self.errors:
            return
        try:
            operation()
        except Exception as exc:
            self.errors.append(type(exc).__name__+': '+str(exc))

    @staticmethod
    def _array(value):
        # Conversion is on a detached observation, never on live model storage.
        return value.detach().float().cpu().numpy().copy()

    def begin_batch(self, batch):
        def collect():
            if self.closed or self.pending is not None:
                raise ValueError('Closed trace or overlapping validation batches')
            samples, targets = batch
            if len(targets) != samples.shape[0]:
                raise ValueError('Target and sample batch sizes differ')
            if self.mode == 'native_runtime' and tuple(samples.shape[-2:]) != (640,640):
                raise ValueError('Unexpected native validation input size')
            ids, sizes = [], []
            for target in targets:
                iid = int(target['image_id'].item())
                size_wh = list(map(int,target['orig_size'].tolist()))
                if iid not in self.images or iid in self.records or iid in ids:
                    raise ValueError('Foreign or duplicate validation image')
                original = self.images[iid]
                # Bound native CocoDetection emits orig_size as [width, height],
                # also consumed in that order by the native PostProcessor.
                if size_wh != [original['width'], original['height']]:
                    raise ValueError('Original image size differs from annotation')
                ids.append(iid); sizes.append([size_wh[1], size_wh[0]])
            self.pending = {'ids':ids, 'sizes':sizes, 'data':{}}
        self._observe(collect)

    def _store(self, module, name, value, tail):
        import numpy as np
        if module.training or self.model.training or self.pending is None:
            raise ValueError('Trace hook called outside the bound evaluation batch')
        if name in self.pending['data']:
            raise ValueError('Duplicate stage observation')
        array = self._array(value)
        if array.shape != (len(self.pending['ids']),*tail) or not np.isfinite(array).all():
            raise ValueError('Unexpected shape or nonfinite '+name)
        self.pending['data'][name] = array

    def _initial_hook(self, module, inputs):
        def collect():
            reference = inputs[1]
            if tuple(reference.shape[1:]) != (300,1,4):
                raise ValueError('Expected native initial reference points')
            # This is the sigmoid already computed by native TransformerDecoder.
            self._store(module,'encoder_initial_boxes',reference[:,:,0,:],(300,4))
        self._observe(collect)
        return None

    def _decoder_hook(self, module, inputs, output):
        def collect():
            if len(output) != 6 or output[0].shape[0] != 1 or output[1].shape[0] != 1:
                raise ValueError('Expected final-only native evaluation outputs')
            self._store(module,'pre_after_first_layer_boxes',output[4],(300,4))
            self._store(module,'pre_after_first_layer_logits',output[5],(300,10))
            self._store(module,'decoder_final_boxes',output[0][-1],(300,4))
            self._store(module,'decoder_final_logits',output[1][-1],(300,10))
        self._observe(collect)
        return None

    def _model_hook(self, module, inputs, output):
        def collect():
            import numpy as np
            self._store(module,'final_boxes',output['pred_boxes'],(300,4))
            self._store(module,'final_logits',output['pred_logits'],(300,10))
            data = self.pending['data']
            for field in ('boxes','logits'):
                if not np.array_equal(data.pop('decoder_final_'+field),data['final_'+field]):
                    raise ValueError('Model output differs from the observed decoder output')
        self._observe(collect)
        return None

    def end_batch(self):
        def collect():
            expected = {'encoder_initial_boxes','pre_after_first_layer_boxes',
                        'pre_after_first_layer_logits','final_boxes','final_logits'}
            if self.pending is None or set(self.pending['data']) != expected:
                raise ValueError('Missing stage observation')
            for i,iid in enumerate(self.pending['ids']):
                self.records[iid] = {k:v[i].copy() for k,v in self.pending['data'].items()}
                self.records[iid]['original_hw'] = self.pending['sizes'][i]
            self.pending = None
        self._observe(collect)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True

    def finish(self, output):
        import numpy as np
        self.close()
        if self.errors or self.pending is not None or set(self.records) != set(self.images):
            raise ValueError('Incomplete native stage trace: '+repr(self.errors)+
                             f'; images={len(self.records)}/{len(self.images)}')
        output = Path(output)
        output.mkdir(parents=True,exist_ok=False)
        for iid,data in sorted(self.records.items()):
            np.savez_compressed(output/f'image_{iid:06d}.npz',**data)
        write(output/'summary.json',dict(status='EXISTING_FORWARD_STAGES_CAPTURED',
            mode=self.mode,identity=self.identity,epoch_zero_based=131,evaluated_weights='EMA',
            image_ids=sorted(self.images),images=len(self.images),query_count=300,
            stage_names=['encoder_initial','pre_after_first_layer','final'],
            output_copy_dtype='float32',same_run_query_order_preserved=True,
            native_target_orig_size_order='width_height',
            stored_original_hw_order='height_width',
            cross_run_query_identity_assumed=False,model_decoder_final_equal=True,
            additional_model_or_head_calls=0,additional_IoU_or_matcher_calls=0,
            timing_includes_observation_overhead=True,
            claim_boundary='Passive stages only; not a causal intervention, training history or efficacy result'))
        write(output/'manifest.json',dict(mode=self.mode,trace_code_sha256=sha(__file__),
            files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        return output/'manifest.json'


class TracedLoader:
    def __init__(self, original, trace):
        self.original, self.trace = original, trace

    def __len__(self): return len(self.original)

    def __getattr__(self, name): return getattr(self.original,name)

    def __iter__(self):
        for batch in self.original:
            self.trace.begin_batch(batch)
            try:
                yield batch
            finally:
                self.trace.end_batch()


def verify_trace(path, identity):
    """Integrity check only. Does not compute geometry or run any network."""
    path=Path(path)
    manifest=read(path/'manifest.json')
    summary=read(path/'summary.json')
    if (manifest['mode']!='native_runtime' or summary['mode']!='native_runtime'
        or manifest['trace_code_sha256']!=sha(__file__) or summary['identity']!=identity
        or summary['epoch_zero_based']!=131 or summary['evaluated_weights']!='EMA'
        or summary['status']!='EXISTING_FORWARD_STAGES_CAPTURED'
        or summary.get('native_target_orig_size_order')!='width_height'
        or summary.get('stored_original_hw_order')!='height_width'
        or summary['image_ids']!=list(range(548))):
        raise ValueError('Trace does not belong to the bound final native experiment')
    expected={'summary.json'}|{f'image_{i:06d}.npz' for i in range(548)}
    if set(manifest['files'])!=expected or any(sha(path/k)!=v for k,v in manifest['files'].items()):
        raise ValueError('Stage trace files are incomplete or changed')
    return summary
