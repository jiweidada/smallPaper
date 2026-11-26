import copy
import importlib.util
import unittest
import analyze_preds as analysis


def fixture():
    return {'images': [{'id': 1, 'width': 1280, 'height': 640}],
            'categories': [{'id': 0, 'name': 'object'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [100, 100, 16, 8], 'area': 128, 'iscrowd': 0}]}


class PredictionAnalysis(unittest.TestCase):
    def test_non_square_resize_scales_axes_separately(self):
        data, predictions, _ = analysis.transform_data(fixture(), [])
        self.assertEqual(data['annotations'][0]['bbox'], [50, 100, 8, 8])
        self.assertEqual(data['annotations'][0]['area'], 64)
        self.assertEqual(analysis.bin_name(data['annotations'][0]['bbox']), '8to16')

    def test_size_boundaries_are_half_open(self):
        self.assertEqual([analysis.bin_name([0, 0, x, x]) for x in [3.999, 4, 8, 16, 32]], ['lt4', '4to8', '8to16', '16to32', 'ge32'])

    def test_one_prediction_never_matches_two_ground_truths(self):
        data = fixture()
        duplicate = copy.deepcopy(data['annotations'][0]); duplicate['id'] = 2
        data['annotations'].append(duplicate)
        pred = dict(image_id=1, category_id=0, bbox=[100, 100, 16, 8], score=.9)
        matched = analysis.match_gt(data, [pred], .3, .5)
        self.assertEqual(len(matched), 1)
        self.assertIn(1, matched)

    def test_wrong_class_and_low_score_never_match(self):
        data = fixture()
        for category, score in [(1, .9), (0, .299)]:
            self.assertFalse(analysis.match_gt(data, [dict(image_id=1, category_id=category, bbox=[100, 100, 16, 8], score=score)], .3, .5))

    def test_new_and_lost_are_excluded_from_error_distributions(self):
        data = fixture()
        pred = dict(image_id=1, category_id=0, bbox=[100, 100, 16, 8], score=.9)
        result, rows = analysis.compare(data, [], [pred], skip_ap=True)
        group = result['localization']['0.5']['8to16']
        self.assertEqual(group['new'], 1)
        self.assertEqual(group['flip_rate_all_gt'], 1)
        self.assertFalse(group['error_distributions_common_gt'])

    def test_identical_predictions_have_zero_errors_and_no_flips(self):
        data = fixture()
        pred = dict(image_id=1, category_id=0, bbox=[100, 100, 16, 8], score=.9)
        result, rows = analysis.compare(data, [pred], [pred], skip_ap=True)
        group = result['localization']['0.75']['8to16']
        self.assertEqual(group['common'], 1)
        self.assertEqual(group['flip_rate_all_gt'], 0)
        self.assertEqual(group['error_distributions_common_gt']['candidate_center_error']['mean'], 0)

    def test_common_roi_clips_both_prediction_and_gt_without_resizing(self):
        data = fixture(); data['images'][0].update(width=640, height=640)
        data['annotations'][0]['bbox'] = [30, 100, 10, 10]
        pred = dict(image_id=1, category_id=0, bbox=[30, 100, 10, 10], score=.9)
        transformed, predictions, clipped = analysis.transform_data(data, [pred], border=32)
        self.assertEqual(transformed['images'][0]['width'], 576)
        self.assertEqual(transformed['annotations'][0]['bbox'], [0, 68, 8, 10])
        self.assertEqual(predictions[0]['bbox'], transformed['annotations'][0]['bbox'])
        self.assertEqual(clipped, 1)

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'), 'COCO dependency unavailable in this local Python')
    def test_perfect_detection_has_perfect_ap_in_correct_size_group(self):
        data, _, _ = analysis.transform_data(fixture(), [])
        pred = dict(image_id=1, category_id=0, bbox=data['annotations'][0]['bbox'], score=.9)
        result = analysis.coco_ap(data, [pred], custom_bins=True)
        self.assertAlmostEqual(result['8to16']['AP'], 1.)
        self.assertIsNone(result['4to8']['AP'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
