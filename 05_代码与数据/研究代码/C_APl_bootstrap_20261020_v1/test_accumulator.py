"""Synthetic exact-score-tie and missing-category checks, no real data."""
import unittest
import numpy as np
from audit import category_cache, cached_ap, official_repeated_ap


def entry(scores, tp, ignored=None, ngt=1):
    a=np.tile(np.asarray(tp)[None,:],(10,1))
    ig=np.zeros_like(a,dtype=bool) if ignored is None else np.tile(np.asarray(ignored)[None,:],(10,1))
    return {'dtScores':scores,'dtMatches':a,'dtIgnore':ig,'gtIgnore':np.zeros(ngt,dtype=bool)}


class AccumulatorTests(unittest.TestCase):
    def compare(self, es, counts):
        cache=[category_cache(e,len(counts)) for e in es]
        actual,_=cached_ap(cache,np.asarray(counts,dtype=np.int32)[None,:])
        expected=official_repeated_ap(es,np.asarray(counts,dtype=np.int32))
        self.assertAlmostEqual(actual[0],expected,places=12)

    def test_tied_fp_then_tp_repeated_image(self):
        self.compare([[entry([.5,.5],[0,1]),entry([.5],[1])]], [3,2])

    def test_tied_tp_then_fp_repeated_image(self):
        self.compare([[entry([.5,.5],[1,0]),entry([.5],[1])]], [3,2])

    def test_ignored_and_empty(self):
        self.compare([[entry([.9,.8,.7],[0,1,0],[True,False,False]),None]], [2,1])

    def test_zero_gt_category_excluded(self):
        self.compare([[entry([.5],[1]),None],[None,entry([.5],[0],ngt=0)]], [2,1])

    def test_all_zero_gt_is_na(self):
        c=category_cache([None,entry([.5],[0],ngt=0)],2)
        actual,present=cached_ap([c],np.asarray([[1,1]],dtype=np.int32))
        self.assertTrue(np.isnan(actual[0]));self.assertEqual(present[0],0)


if __name__=='__main__': unittest.main()
