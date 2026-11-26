"""Actual candidate counts: global-only baseline has zero new candidates."""
import importlib
V=importlib.import_module('343_formal_counts_v2')
C=V.C;F=V.F;G=V.G;match=V.match;independent_match=V.independent_match
def metrics(out,owner,baseout,baseowner,groups,aids,pre=None,num_new_instances=40):
    assert num_new_instances in (0,40)
    r=V.metrics(out,owner,baseout,baseowner,groups,aids,pre)
    selected=r['new_candidate_instances_final_selected']
    assert selected<=num_new_instances
    r['new_candidate_instances_input']=num_new_instances
    r['new_candidate_instances_final_unselected']=num_new_instances-selected
    if pre is not None:
        entered=r['new_candidate_instances_top300_selected'];assert entered<=num_new_instances
        r['new_candidate_instances_top300_unselected']=num_new_instances-entered
    return r
