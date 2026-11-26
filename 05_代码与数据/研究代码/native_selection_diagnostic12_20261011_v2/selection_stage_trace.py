"""Join native selector observations to the existing native StageTrace by batch ID."""
from pathlib import Path
import numpy as np
from stage_trace import StageTrace
from results import write, sha
from selection_trace import NativeSelectionObserver


class SelectionStageTrace(StageTrace):
    def __init__(self, model, identity, annotation_path, mode='native_runtime'):
        self.selection_observer = None
        self.selection_records = {}
        self.selected_offsets = None
        self.selected_bbox_calls = 0
        super().__init__(model, identity, annotation_path, mode=mode)
        try:
            observer_mode = 'synthetic_engineering' if mode == 'synthetic_engineering' else 'post_training_diagnostic'
            self.selection_observer = NativeSelectionObserver(model.decoder, mode=observer_mode)
            self.selection_observer.__enter__()
            self.handles.append(model.decoder.enc_bbox_head.register_forward_hook(self._selected_bbox_hook))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _native_array(value):
        return value.detach().cpu().numpy().copy()

    def begin_batch(self, batch):
        self.selected_offsets = None
        super().begin_batch(batch)

    def _selected_bbox_hook(self, module, inputs, output):
        def collect():
            if self.pending is None or self.selected_offsets is not None:
                raise ValueError('Missing batch or repeated selected bbox head')
            if tuple(output.shape) != (len(self.pending['ids']), 300, 4):
                raise ValueError('Bbox head is not the native selected300 readout')
            self.selected_offsets = self._native_array(output)
            self.selected_bbox_calls += 1
        self._observe(collect)
        return None

    def _model_hook(self, module, inputs, output):
        super()._model_hook(module, inputs, output)
        def collect():
            record = self.selection_observer.take_record()
            ids, sizes = self.pending['ids'], self.pending['sizes']
            arrays = {k:self._native_array(record[k]) for k in (
                'dense_class_logits','dense_anchor_logits','selected_indices','selected_anchor_logits')}
            shapes = dict(dense_class_logits=(8400,10), dense_anchor_logits=(8400,4),
                          selected_indices=(300,), selected_anchor_logits=(300,4))
            if record['selector_training'] or record['native_topk_calls'] != 1:
                raise ValueError('Expected one native eval selection per image batch')
            if self.selected_offsets is None:
                raise ValueError('Missing existing selected-only bbox regression output')
            for k, tail in shapes.items():
                if arrays[k].shape != (len(ids),*tail):
                    raise ValueError('Unexpected selection shape '+k)
            if arrays['selected_indices'].dtype != np.int64 or not np.isfinite(arrays['dense_class_logits']).all():
                raise ValueError('Bad selected IDs or class logits')
            if not ((arrays['selected_indices'] >= 0) & (arrays['selected_indices'] < 8400)).all():
                raise ValueError('Selected location outside native8400')
            for i, iid in enumerate(ids):
                if iid in self.selection_records:
                    raise ValueError('Duplicate selection image ID')
                self.selection_records[iid] = {k:v[i].copy() for k,v in arrays.items()}
                self.selection_records[iid].update(selected_bbox_offsets=self.selected_offsets[i].copy(),
                    original_hw=np.asarray(sizes[i],dtype=np.int64), input_hw=np.array([640,640],dtype=np.int64),
                    image_id=np.array(iid,dtype=np.int64))
        self._observe(collect)
        return None

    def close(self):
        if self.selection_observer is not None and self.selection_observer.entered:
            self.selection_observer.__exit__(None,None,None)
        super().close()

    def finish_selection(self, output):
        self.close()
        if self.errors or self.pending is not None or set(self.selection_records) != set(self.images) or set(self.records) != set(self.images):
            raise ValueError('Selection/stage image join incomplete: '+repr(self.errors))
        output = Path(output); output.mkdir(exist_ok=False)
        for iid, values in sorted(self.selection_records.items()):
            np.savez_compressed(output/f'image_{iid:06d}.npz',**values)
        write(output/'summary.json',dict(status='NATIVE_SELECTION_AND_EXISTING_STAGE_BATCH_IDS_JOINED',
            identity=self.identity,mode=self.mode,images=len(self.images),image_ids=sorted(self.images),
            epoch_zero_based=11,evaluated_weights='EMA',dense_locations=8400,selected_queries=300,
            original_hw_order='height_width',input_hw=[640,640],stored_copy_dtype='native dtype preserved',
            selected_indices_source='Actual original torch.topk return; never recomputed',
            dense_anchors_role='Unregressed anchor logits, possibly inf when invalid; not8400 predicted boxes',
            selected_bbox_offsets_source='Existing one selected-only enc_bbox_head output',
            additional_head_calls=0,additional_topk_calls=0,cross_run_query_identity_assumed=False,
            claim_boundary='Passive post-training recording. Selection locations, selected initial regression and subsequent stages are joined; coverage/causal efficacy claims require separate analysis and exact saved-stage replay validation.'))
        write(output/'manifest.json',dict(trace_code_sha256=sha(__file__),files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        return output/'manifest.json'
