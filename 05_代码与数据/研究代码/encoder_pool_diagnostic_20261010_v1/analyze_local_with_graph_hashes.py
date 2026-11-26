"""Instrument the unchanged original analysis with graph hashes, not SciPy.

This adds evidence only; it uses the exact same matcher and derived statistics.
Local graph hashes and valid-witness counts must equal the remote independent
SciPy optimality certificates before the scientific result is delivered.
"""
import hashlib
import numpy as np
import analyze_pool

original=analyze_pool.match_all
def instrumented(edge,tiny,scipy_check):
    assert not scipy_check
    covered,witness,unused=original(edge,tiny,False)
    evidence={'all_GT_cardinality':int(covered.sum()),'tiny_cardinality':int((covered&tiny).sum()),'edge_sha256':hashlib.sha256(np.ascontiguousarray(edge).tobytes()).hexdigest()}
    return covered,witness,evidence

if __name__=='__main__':
    analyze_pool.match_all=instrumented
    analyze_pool.main()
