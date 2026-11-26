"""Selection-window and resume-state tests; no model updates or GPU use."""
import unittest
from tiny_stage2 import choose_stage2, stage2_snapshot, preserve_clock_during_refresh
from types import SimpleNamespace


class Stage2Test(unittest.TestCase):
    def test_earlier_peak_cannot_remove_stage2_checkpoint(self):
        self.assertIsNone(choose_stage2(None, {'tiny_eval_bbox': [.9]}, 83))
        selected = choose_stage2(None, {'tiny_eval_bbox': [.4]}, 84)
        self.assertEqual(selected['epoch'], 84)
        self.assertEqual(selected['avg_metric'], .4)

    def test_primary_metric_and_first_tie(self):
        first = choose_stage2(None, {'tiny_eval_bbox': [.4,.99]}, 84)
        self.assertIs(choose_stage2(first, {'tiny_eval_bbox': [.4,1.0]}, 85), first)
        self.assertEqual(choose_stage2(first, {'tiny_eval_bbox': [.45,.1]}, 86)['epoch'], 86)

    def test_snapshot_preserves_full_training_state_and_resume_selection(self):
        selected = choose_stage2(None, {'tiny_eval_bbox': [.4]}, 93)
        original = {'last_epoch':93,'model':{'x':1},'optimizer':{'step':100},'ema':{'updates':100},
                    'scaler':{'scale':2048},'best_stat':{'epoch':30,'avg_metric':.9}}
        snapshot = stage2_snapshot(original, selected, 93)
        self.assertEqual(snapshot['optimizer'], original['optimizer'])
        self.assertEqual(snapshot['ema'], original['ema'])
        self.assertEqual(snapshot['scaler'], original['scaler'])
        self.assertEqual(original['best_stat']['epoch'], 30)
        self.assertEqual(snapshot['best_stat']['epoch'], 93)
        self.assertIs(choose_stage2(snapshot['tiny_stage2_best_stat'], {'tiny_eval_bbox':[.39]},94), snapshot['tiny_stage2_best_stat'])

    def test_rejects_wrong_epoch_binding(self):
        with self.assertRaises(AssertionError):stage2_snapshot({'last_epoch':94}, {'epoch':93},94)

    def test_native_refresh_keeps_true_epoch_and_stage2_selection(self):
        selected = choose_stage2(None, {'tiny_eval_bbox':[.4]},84)
        solver=SimpleNamespace(last_epoch=84,_tiny_stage2_best_stat=selected,ema='evaluated')
        def refresh(obj):
            obj.last_epoch=30;obj._tiny_stage2_best_stat=None;obj.ema='native_stage1_refresh';return True
        self.assertTrue(preserve_clock_during_refresh(solver,refresh))
        self.assertEqual(solver.last_epoch,84)
        self.assertEqual(solver._tiny_stage2_best_stat,selected)
        self.assertEqual(solver.ema,'native_stage1_refresh')


if __name__ == '__main__':unittest.main(verbosity=2)
