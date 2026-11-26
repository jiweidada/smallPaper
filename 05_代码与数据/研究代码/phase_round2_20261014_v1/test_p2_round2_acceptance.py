"""Additional checks for expensive round-two training; uses the received tests."""
import unittest
import torch
from torch.nn import functional as F
import test_p2_phase_controls as ref


class RoundTwoAcceptance(unittest.TestCase):
    def test_zero_init_blocks_upstream_gradients_in_all_new_arms(self):
        for model in [ref.CMean(), ref.LowRank(), ref.LowRank(phase_mean=True)]:
            model.train()
            p2, p3 = ref._inputs()
            model(p2, p3).square().mean().backward()
            self.assertGreater(model.restore.weight.grad.abs().sum().item(), 0)
            for name, parameter in model.named_parameters():
                if name != 'restore.weight':
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, name)

    def test_nonzero_restore_reaches_reduce_and_mix(self):
        for model in [ref.CMean(), ref.LowRank(), ref.LowRank(phase_mean=True)]:
            torch.nn.init.normal_(model.restore.weight, std=0.05)
            p2, p3 = ref._inputs()
            model.train()(p2, p3).square().mean().backward()
            self.assertGreater(model.reduce.conv.weight.grad.abs().sum().item(), 0)
            if hasattr(model, 'mix'):
                self.assertGreater(model.mix.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_all_mean_phase_columns_have_equal_gradients(self):
        for model in [ref.CMean(), ref.LowRank(phase_mean=True)]:
            torch.nn.init.normal_(model.restore.weight, std=0.05)
            p2, p3 = ref._inputs()
            model.train()(p2, p3).square().mean().backward()
            weight = model.mix.weight if hasattr(model, 'mix') else model.restore.weight
            gradient = weight.grad.reshape(weight.shape[0], 16, 4)
            for phase in range(1, 4):
                torch.testing.assert_close(gradient[:, :, phase], gradient[:, :, 0], rtol=0, atol=1e-7)

    def test_all_inference_only_modes_reject_training(self):
        for mode in ['mean', 'shuffle', 'zero']:
            with self.assertRaises(RuntimeError):
                ref.Ablate(phase_eval=mode).train()(*ref._inputs())
        ref.Ablate(phase_eval='none').train()(*ref._inputs())

    def test_invalid_eval_options_cannot_hide_behind_disabled_bridge(self):
        for options in [dict(use_p2_skip=False, p2_skip_mode='c_mean', p2_phase_eval='mean'),
                        dict(use_p2_skip=False, p2_skip_mode='bad'),
                        dict(use_p2_skip=False, p2_phase_eval='bad'),
                        dict(use_p2_skip=False, p2_phase_eval='mean')]:
            with self.assertRaises(ValueError):
                ref.Backbone('B0', **options)

    def test_odd_size_mean_includes_explicit_zero_padding(self):
        model = ref.CMean().eval()
        torch.nn.init.normal_(model.restore.weight, std=0.05)
        p2, p3 = ref._inputs(15, 17)
        with torch.no_grad():
            reduced = F.pad(model.reduce(p2), (0, 1, 0, 1))
            pooled = F.avg_pool2d(reduced, 2, 2)
            expected = p3 + model.restore(pooled.repeat_interleave(4, dim=1))
            torch.testing.assert_close(model(p2, p3), expected, rtol=1e-5, atol=1e-6)

    def test_phase_mean_preserves_phase_permutations(self):
        torch.manual_seed(91)
        detail = torch.randn(2, 16, 4, 7, 9)
        fn = ref.M['hgnetv2_light_phase']._phase_mean
        expected = fn(detail.reshape(2, 64, 7, 9), 16)
        actual = fn(detail[:, :, [3, 0, 2, 1]].reshape(2, 64, 7, 9), 16)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
