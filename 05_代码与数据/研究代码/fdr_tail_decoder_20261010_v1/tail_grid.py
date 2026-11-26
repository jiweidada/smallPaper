"""Isolated FDR endpoint-only change; never edit native engine source files."""
import hashlib
from pathlib import Path

UTILS_SHA = 'ad87ec5a50b1970e76b06f7242298ffa332beaf78c828ac3e8d41ac656ec5f6c'
DECODER_SHA = 'b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TailGridBinding:
    """Fresh-process experimental binding shared by Integral and GT encoding.

    It keeps native references, reg_scale, bins, probability residuals, DDF and
    IoU weights. Only grid endpoints are multiplied. No new parameters/state.
    """
    def __init__(self, factor=2.):
        assert float(factor) in (1.,2.), 'Only the registered identity and factor2 diagnostic configurations'
        self.factor=float(factor)
        self.installed=False

    def install(self):
        assert not self.installed, 'Do not stack bindings'
        from engine.deim import dfine_utils as utils, dfine_decoder as decoder
        assert sha(utils.__file__)==UTILS_SHA and sha(decoder.__file__)==DECODER_SHA
        assert utils.weighting_function is decoder.weighting_function
        original=utils.weighting_function
        assert original.__module__=='engine.deim.dfine_utils' and original.__name__=='weighting_function'
        factor=self.factor
        def expanded(reg_max, up, reg_scale, deploy=False):
            assert reg_max==32, 'Candidate requires exactly33 native bins'
            grid=original(reg_max,up,reg_scale,deploy=deploy)
            assert grid.ndim==1 and grid.numel()==33
            if factor==1.:
                return grid
            import torch
            return torch.cat((grid[:1]*factor,grid[1:-1],grid[-1:]*factor),dim=0)
        self.utils,self.decoder,self.original,self.expanded=utils,decoder,original,expanded
        # translate_gt/bbox2distance resolve the utility module global at call.
        # DFINE decoder holds an imported alias, so both bindings must change.
        utils.weighting_function=expanded
        decoder.weighting_function=expanded
        assert utils.translate_gt.__globals__['weighting_function'] is expanded
        self.installed=True
        return self

    def close(self):
        if not self.installed:
            return
        assert self.utils.weighting_function is self.expanded
        assert self.decoder.weighting_function is self.expanded
        self.utils.weighting_function=self.original
        self.decoder.weighting_function=self.original
        self.installed=False

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.close()

    def metadata(self):
        return dict(candidate='FDR_ENDPOINTS_ONLY_FACTOR2',factor=self.factor,
            ordinary_bins=33,extra_parameters=0,new_state_dict_keys=0,
            central_supports_unchanged=31,native_reference_and_reg_scale_unchanged=True,
            FGL_encoder_and_prediction_decoder_share_same_grid=True,
            new_architecture_not_claimed=True,novelty_or_efficacy_not_claimed=True)
