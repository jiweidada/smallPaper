import copy
import unittest
import numpy as np
from PIL import Image
from uavdt_protocol import NativeScorer, contained_mask, curve, erase_ignored, match_frame, voc_ap
from uavdt_stage2 import choose_stage2, stage2_snapshot, preserve_clock_during_refresh
from types import SimpleNamespace

class ProtocolTest(unittest.TestCase):
    def test_exact_point7_continuous_iou_and_duplicates(self):
        _,matched,_=match_frame([[0,0,10,10]],[[0,0,7,10,.9],[0,0,10,10,.8],[0,0,6,10,.7]])
        self.assertEqual(matched.tolist(),[0,-1,-1])
    def test_score_order_and_last_gt_tie(self):
        scores,matched,order=match_frame([[0,0,10,10],[0,0,10,10]],[[0,0,10,10,.5],[0,0,10,10,.9]])
        self.assertEqual(matched.tolist(),[1,0]);self.assertEqual(order.tolist(),[1,0])
    def test_ignore_is_strict_containment_not_iod(self):
        d=[[1,1,8,8,.9],[0,0,10,10,.8],[0,0,7,10,.7],[-1,-1,12,12,.6]]
        self.assertEqual(contained_mask(d,[[0,0,10,10]]).tolist(),[True,False,False,False])
    def test_voc_envelope(self):self.assertEqual(voc_ap(np.array([.5,1]),np.array([1.,.5])),.75)
    def test_empty_frames_still_count_false_positives(self):
        a={'images':[{'id':1,'uav_sequence':'M0001','uav_frame':1,'uav_apply_native_ignore':True,'uav_ignore_boxes':[]},
                     {'id':2,'uav_sequence':'M0001','uav_frame':2,'uav_apply_native_ignore':True,'uav_ignore_boxes':[]}],
           'annotations':[{'image_id':1,'bbox':[0,0,10,10]}]}
        s=NativeScorer(a);s.update(1,[[0,0,10,10,.8]]);s.update(2,[[0,0,10,10,.9]])
        self.assertEqual(s.finish()[0]['AP70'],.5)
    def test_sequence_ignore_switch_and_coverage(self):
        image={'id':1,'uav_sequence':'M0209','uav_frame':1,'uav_apply_native_ignore':False,'uav_ignore_boxes':[[0,0,20,20]]}
        s=NativeScorer({'images':[image],'annotations':[]})
        with self.assertRaises(RuntimeError):s.finish()
        s.update(1,[[1,1,10,10,.9]]);self.assertEqual(s.ignored,0)
        with self.assertRaises(RuntimeError):s.update(1,[])
        image=copy.deepcopy(image);image['uav_apply_native_ignore']=True
        s=NativeScorer({'images':[image],'annotations':[]});s.update(1,[[1,1,10,10,.9]]);self.assertEqual(s.ignored,1)
    def test_training_mask_preserves_positive_pixels_and_input(self):
        raw=np.arange(12*12*3,dtype=np.uint8).reshape(12,12,3);image=Image.fromarray(raw)
        masked=np.asarray(erase_ignored(image,[[0,0,12,12]],[[2,2,4,4]]))
        np.testing.assert_array_equal(masked[2:6,2:6],raw[2:6,2:6]);np.testing.assert_array_equal(np.asarray(image),raw)
        self.assertFalse(np.array_equal(masked[0,0],raw[0,0]))
    def test_stage2_not_global_peak_and_first_tie(self):
        self.assertIsNone(choose_stage2(None,{'uavdt_eval_bbox':[.9]},83))
        first=choose_stage2(None,{'uavdt_eval_bbox':[0.]},84);self.assertEqual(first['epoch'],84)
        self.assertIs(choose_stage2(first,{'uavdt_eval_bbox':[0.]},85),first)
        self.assertEqual(choose_stage2(first,{'uavdt_eval_bbox':[.4]},86)['epoch'],86)
    def test_stage2_full_state_and_epoch_binding(self):
        selected=choose_stage2(None,{'uavdt_eval_bbox':[.4]},93)
        old={'last_epoch':93,'optimizer':{'step':100},'ema':{'updates':100},'scaler':{'scale':1024},'best_stat':{'epoch':30,'avg_metric':.9}}
        state=stage2_snapshot(old,selected,93)
        self.assertEqual(state['optimizer'],old['optimizer']);self.assertEqual(state['ema'],old['ema'])
        self.assertEqual(old['best_stat']['epoch'],30)
        self.assertIs(choose_stage2(state['uavdt_stage2_best_stat'],{'uavdt_eval_bbox':[.39]},94),state['uavdt_stage2_best_stat'])
        with self.assertRaises(AssertionError):stage2_snapshot({'last_epoch':94},selected,93)
    def test_stage_refresh_preserves_selection_clock(self):
        s=SimpleNamespace(last_epoch=84,_uavdt_stage2_best_stat={'epoch':84},ema='old')
        def refresh(obj):obj.last_epoch=30;obj._uavdt_stage2_best_stat=None;obj.ema='refreshed'
        preserve_clock_during_refresh(s,refresh)
        self.assertEqual(s.last_epoch,84);self.assertEqual(s._uavdt_stage2_best_stat,{'epoch':84});self.assertEqual(s.ema,'refreshed')

if __name__=='__main__':unittest.main(verbosity=2)
