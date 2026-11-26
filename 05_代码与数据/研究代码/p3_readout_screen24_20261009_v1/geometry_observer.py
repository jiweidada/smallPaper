"""Passive capture on the existing final EMA pass, no extra model/head forward."""
from stage_trace import StageTrace
from readout import sample_locations,sampling_diagnostics

class GeometryTrace(StageTrace):
    fields=('readout_z0_boxes','readout_delta_logits','readout_z1_boxes',
            'readout_locations','outside_normalized_image_points','border_clamped_points',
            'duplicate_after_clamp_points')
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.handles.append(self.model.decoder.local_readout.register_forward_hook(self._readout_hook))
    def _readout_hook(self,module,inputs,output):
        def collect():
            semantic,p3,z0=inputs
            self._store(module,'readout_z0_boxes',z0.detach().sigmoid(),(300,4))
            self._store(module,'readout_delta_logits',output,(300,4))
            self._store(module,'readout_z1_boxes',(z0.detach()+output.detach()).sigmoid(),(300,4))
            self._store(module,'readout_locations',sample_locations(z0,module.arm),(300,9,2))
            for name,value in sampling_diagnostics(z0,module.arm,*p3.shape[-2:]).items():
                self._store(module,name,value,(300,))
        self._observe(collect)
    def end_batch(self):
        if self.errors or self.pending is None:
            return super().end_batch()
        import numpy as np
        try:
            ids=list(self.pending['ids'])
            extra={name:self.pending['data'].pop(name) for name in self.fields}
            if not np.array_equal(extra['readout_z1_boxes'],self.pending['data']['encoder_initial_boxes']):
                raise ValueError('Observed residual reference differs from native decoder input')
        except Exception as exc:
            self.errors.append(type(exc).__name__+': '+str(exc));return
        super().end_batch()
        if not self.errors:
            for i,iid in enumerate(ids):
                self.records[iid].update({name:value[i].copy() for name,value in extra.items()})
