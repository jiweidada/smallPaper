"""Exact bounded-graph allocation; no detector, training, or GT-at-inference code."""
from collections import Counter
import networkx as nx

def frequencies(branches):
    return Counter((int(q),int(g)) for pairs in branches for q,g in pairs)

def solve_cover(votes,native,n_gt,n_queries=300):
    """Max votes, min changed identities, then lexicographically smallest GT vector.

    Integer costs encode all three priorities exactly, not floating perturbations.
    Each query has unit supply. Each GT has unit demand plus an edge to a sink,
    giving a lower bound of one while keeping exactly the original K pairs.
    """
    queries=sorted(native);k=len(queries)
    if n_gt==0 or not k or n_gt>n_queries:return dict(native),{'fallback':True,'reason':'empty_or_over_capacity'}
    if n_gt>k:return dict(native),{'fallback':True,'reason':'not_enough_selected_queries'}
    assert set(q for q,g in votes)==set(queries)
    assert all((q,g) in votes and 0<=g<n_gt for q,g in native.items())
    graph=nx.DiGraph();sink=('sink',0);graph.add_node(sink,demand=k-n_gt)
    base=n_gt+1;lex_scale=base**k;vote_scale=(k+1)*lex_scale
    for g in range(n_gt):
        gn=('g',g);graph.add_node(gn,demand=1);graph.add_edge(gn,sink,capacity=k,weight=0)
    for i,q in enumerate(queries):
        qn=('q',q);graph.add_node(qn,demand=-1);lex_place=base**(k-1-i)
        for (qq,g),count in sorted(votes.items()):
            if qq==q:
                cost=-int(count)*vote_scale-int(native[q]==g)*lex_scale+g*lex_place
                graph.add_edge(qn,('g',g),capacity=1,weight=cost)
    try:cost,flow=nx.network_simplex(graph)
    except nx.NetworkXUnfeasible:return dict(native),{'fallback':True,'reason':'no_full_matching'}
    chosen={q:next(node[1] for node,v in flow[('q',q)].items() if v) for q in queries}
    assert set(chosen)==set(native) and set(chosen.values())==set(range(n_gt))
    return chosen,{'fallback':False,'vote_total':sum(votes[q,g] for q,g in chosen.items()),'changed_queries':sum(chosen[q]!=native[q] for q in queries),'lex_exact':True}

def anchor_fill(native,last,votes,n_gt,n_queries=300):
    chosen=dict(native)
    if n_gt==0 or n_gt>n_queries or len(last)!=n_gt or len(set(last.values()))!=n_gt:
        return chosen,{'fallback':True}
    assert set(last).issubset(native) and all((q,g) in votes for q,g in last.items())
    chosen.update(last)
    assert set(chosen)==set(native) and set(chosen.values())==set(range(n_gt))
    return chosen,{'fallback':False}

def objective(mapping,votes,native):
    return (-sum(votes[q,g] for q,g in mapping.items()),sum(mapping[q]!=native[q] for q in mapping),tuple(mapping[q] for q in sorted(mapping)))
