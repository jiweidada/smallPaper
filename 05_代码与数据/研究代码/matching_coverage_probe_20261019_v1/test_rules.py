import itertools,random,unittest
from rules import frequencies,solve_cover,anchor_fill,objective

class RulesTest(unittest.TestCase):
    def verify_bruteforce(self,branches,n):
        votes=frequencies(branches);qs=sorted({q for q,g in votes})
        native={q:min((g for qq,g in votes if qq==q),key=lambda g:(-votes[q,g],g)) for q in qs}
        actual,meta=solve_cover(votes,native,n,n_queries=len(qs))
        choices=[sorted(g for qq,g in votes if qq==q) for q in qs]
        valid=[]
        for values in itertools.product(*choices):
            if set(values)==set(range(n)):valid.append(dict(zip(qs,values)))
        expected=min(valid,key=lambda m:objective(m,votes,native))
        self.assertFalse(meta['fallback']);self.assertEqual(actual,expected)
        last=dict(branches[-1]);anchored,_=anchor_fill(native,last,votes,n,len(qs))
        self.assertEqual(set(anchored),set(native));self.assertEqual(set(anchored.values()),set(range(n)))
    def test_known_no_tie_counterexample(self):
        self.verify_bruteforce([[(0,0),(1,1),(2,3),(5,2)],[(0,0),(1,3),(2,2),(4,1)],[(0,3),(1,1),(2,2),(3,0)]],4)
    def test_exhaustive_small_random_graphs(self):
        rng=random.Random(20261019)
        for _ in range(80):
            q=rng.randint(3,6);n=rng.randint(1,min(q,4));layers=rng.randint(2,5)
            branches=[list(zip(rng.sample(range(q),n),rng.sample(range(n),n))) for _ in range(layers)]
            self.verify_bruteforce(branches,n)
    def test_empty_and_overcapacity(self):
        self.assertEqual(solve_cover({}, {},0)[0],{})
        votes={(0,0):3};native={0:0}
        self.assertTrue(solve_cover(votes,native,301)[1]['fallback'])
    def test_infeasible_coverage_falls_back(self):
        votes={(0,0):2,(1,0):1};native={0:0,1:0}
        actual,meta=solve_cover(votes,native,2,n_queries=2)
        self.assertEqual(actual,native);self.assertTrue(meta['fallback'])

if __name__=='__main__':unittest.main()
