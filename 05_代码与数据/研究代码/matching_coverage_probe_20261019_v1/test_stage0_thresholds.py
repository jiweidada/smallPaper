"""Synthetic counts only; never opens any development or heldout output."""
import unittest
from stage0_thresholds import decide

def two(a,b=None):return {1:dict(a),2:dict(a if b is None else b)}

class ThresholdTests(unittest.TestCase):
    def test_exact_boundaries_do_not_trigger(self):
        r=decide(two({'M':10,'T':2000,'D':20,'B':2000}))
        self.assertFalse(r['stop'])
        self.assertEqual(r['rules']['missing']['state'],'neither_below')
        self.assertEqual(r['rules']['different']['state'],'neither_below')

    def test_zero_denominator_is_NA(self):
        r=decide(two({'M':0,'T':0,'D':0,'B':0}))
        self.assertIsNone(r['rules']['missing']['per_seed'][1]['ratio'])
        self.assertIsNone(r['rules']['missing']['stop'])
        self.assertIsNone(r['rules']['different']['stop'])
        self.assertFalse(r['stop'])
        self.assertEqual(r['status'],'insufficient_applicable_assets')

    def test_one_seed_below_is_mixed(self):
        r=decide(two({'M':9,'T':2000,'D':19,'B':2000},
                     {'M':10,'T':2000,'D':20,'B':2000}))
        self.assertFalse(r['stop'])
        self.assertEqual(r['status'],'mixed_seed_results')
        self.assertTrue(all(v['state']=='mixed' for v in r['rules'].values()))

    def test_missing_rule_alone_stops(self):
        r=decide(two({'M':9,'T':2000,'D':20,'B':2000}))
        self.assertTrue(r['stop']);self.assertEqual(r['stop_reasons'],['missing'])

    def test_difference_rule_alone_stops(self):
        r=decide(two({'M':10,'T':2000,'D':19,'B':2000}))
        self.assertTrue(r['stop']);self.assertEqual(r['stop_reasons'],['different'])

    def test_missing_seed_NA_is_not_zero(self):
        r=decide(two({'M':0,'T':0,'D':20,'B':2000},
                     {'M':9,'T':2000,'D':20,'B':2000}))
        self.assertFalse(r['stop']);self.assertEqual(r['rules']['missing']['state'],'NA')

    def test_passing_counts_never_auto_advance(self):
        r=decide(two({'M':11,'T':2000,'D':21,'B':2000}))
        self.assertFalse(r['stop']);self.assertFalse(r['automatic_progression'])

    def test_invalid_counts_rejected(self):
        with self.assertRaises(ValueError):decide(two({'M':1,'T':0,'D':20,'B':2000}))

if __name__=='__main__':unittest.main()
