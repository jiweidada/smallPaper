"""CPU tests for official ignore matching, scale ranges, metric naming and empty results."""
import unittest

from tiny_protocol import METRIC_NAMES, evaluate_records


def annotation(extra=None):
    anns = [{'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [10, 10, 10, 10],
             'area': 100, 'iscrowd': 0, 'ignore': False, 'uncertain': False}]
    anns += extra or []
    return {'images': [{'id': 1, 'width': 200, 'height': 200}],
            'categories': [{'id': 0, 'name': 'person'}], 'annotations': anns}


def detection(box, score=0.9):
    return {'image_id': 1, 'category_id': 0, 'bbox': box, 'score': score}


class TinyProtocolTest(unittest.TestCase):
    def test_primary_metric_is_official_tiny_AP50(self):
        metrics, values, evaluator = evaluate_records(annotation(), [detection([10, 10, 10, 10])], verbose=False)
        self.assertEqual(METRIC_NAMES[0], 'AP50_tiny')
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(metrics['AP50_tiny'], values[0])
        self.assertEqual(evaluator.params.maxDets, [200])
        self.assertEqual(evaluator.params.areaRng[1], [1, 400])

    def test_unmatched_detection_is_false_positive(self):
        metrics, _, _ = evaluate_records(annotation(), [detection([80, 80, 10, 10], 0.99), detection([10, 10, 10, 10], 0.8)], verbose=False)
        self.assertLess(metrics['AP50_tiny'], 0.6)

    def test_ignore_region_uses_intersection_over_detection(self):
        region = {'id': 2, 'image_id': 1, 'category_id': 0, 'bbox': [50, 50, 100, 100],
                  'area': 10000, 'iscrowd': 0, 'ignore': True, 'uncertain': False}
        metrics, _, _ = evaluate_records(annotation([region]), [detection([60, 60, 10, 10], 0.99), detection([10, 10, 10, 10], 0.8)], verbose=False)
        self.assertAlmostEqual(metrics['AP50_tiny'], 1.0)

    def test_uncertain_region_is_ignored(self):
        region = {'id': 2, 'image_id': 1, 'category_id': 0, 'bbox': [50, 50, 100, 100],
                  'area': 10000, 'iscrowd': 0, 'ignore': False, 'uncertain': True}
        metrics, _, _ = evaluate_records(annotation([region]), [detection([60, 60, 10, 10], 0.99), detection([10, 10, 10, 10], 0.8)], verbose=False)
        self.assertAlmostEqual(metrics['AP50_tiny'], 1.0)

    def test_empty_predictions_return_zero_AP(self):
        metrics, values, _ = evaluate_records(annotation(), [], verbose=False)
        self.assertEqual(metrics['AP50_tiny'], 0)
        self.assertEqual(len(values), 42)

    def test_tiny2_size_bin(self):
        metrics, _, _ = evaluate_records(annotation(), [detection([10, 10, 10, 10])], verbose=False)
        self.assertAlmostEqual(metrics['AP50_tiny2'], 1.0)
        self.assertEqual(metrics['AP50_tiny1'], -1)

    def test_max_200_detections_is_enforced(self):
        predictions = [detection([80, 80, 10, 10], 1.0-i/1000) for i in range(200)]
        predictions.append(detection([10, 10, 10, 10], 0.1))
        metrics, _, _ = evaluate_records(annotation(), predictions, verbose=False)
        self.assertEqual(metrics['AP50_tiny'], 0)


if __name__ == '__main__': unittest.main(verbosity=2)
