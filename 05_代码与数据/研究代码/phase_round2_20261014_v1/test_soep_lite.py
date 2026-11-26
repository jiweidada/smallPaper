import pathlib
import sys
import unittest
import torch

sys.path.insert(0, str(pathlib.Path.cwd()))
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.soep_lite_round2 import SOEPLiteFusion, HybridEncoder_SOEP_Lite


class SOEPLiteChecks(unittest.TestCase):
    def test_budget_and_zero_identity(self):
        model = SOEPLiteFusion().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 15524)
        p2, p3 = torch.randn(2, 64, 16, 16), torch.randn(2, 256, 8, 8)
        self.assertTrue(torch.equal(model(p2, p3), p3))
        self.assertEqual(model.spatial.conv.kernel_size, (3, 3))
        self.assertEqual(model.spatial.conv.stride, (1, 1))
        self.assertIsInstance(model.mix.act, torch.nn.ReLU)

    def test_nonzero_restore_makes_branch_depend_on_neighboring_cells(self):
        torch.manual_seed(13)
        model = SOEPLiteFusion().eval()
        torch.nn.init.normal_(model.restore.weight, std=.1)
        p2, p3 = torch.randn(2, 64, 16, 16), torch.randn(2, 256, 8, 8)
        changed = p2.clone(); changed[:, :, 6:8, 8:10] += 5
        first, second = model(p2, p3), model(changed, p3)
        # A neighboring output cell is affected: impossible for O8's pointwise post-SPD mapping.
        self.assertGreater((first[:, :, 3, 3]-second[:, :, 3, 3]).abs().max().item(), 1e-6)

    def test_zero_branch_preserves_all_encoder_pyramid_outputs(self):
        options = dict(in_channels=[16, 32, 64], hidden_dim=16, nhead=4,
                       dim_feedforward=32, use_encoder_idx=[2], num_encoder_layers=1,
                       expansion=.5, depth_mult=.34, eval_spatial_size=None)
        torch.manual_seed(2); baseline = HybridEncoder(**options).eval()
        torch.manual_seed(2); candidate = HybridEncoder_SOEP_Lite(**options, p2_channels=4, bridge_width=4).eval()
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        pyramid = [torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4), torch.randn(2, 64, 2, 2)]
        with torch.no_grad():
            original = baseline(pyramid)
            actual = candidate([torch.randn(2, 4, 16, 16)]+pyramid)
        for first, second in zip(original, actual):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_gradient_gate_and_later_upstream_updates(self):
        model = SOEPLiteFusion(p2_channels=4, hidden_dim=16, width=4).train()
        p2, p3 = torch.randn(2, 4, 16, 16), torch.randn(2, 16, 8, 8)
        model(p2, p3).square().mean().backward()
        self.assertGreater(model.restore.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.spatial.conv.weight.grad.abs().sum().item(), 0)
        torch.nn.init.normal_(model.restore.weight, std=.1); model.zero_grad()
        model(p2, p3).square().mean().backward()
        self.assertGreater(model.spatial.conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.mix.conv.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
