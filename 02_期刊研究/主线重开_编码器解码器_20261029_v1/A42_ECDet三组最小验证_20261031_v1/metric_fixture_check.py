import pathlib,torch
from runtime_common import R,save_json
from evaluate import metrics_from_predictions,union_overlap_fraction

assert union_overlap_fraction([0,0,10,10],[[0,0,6,10],[4,0,8,10]])==.8
assert union_overlap_fraction([0,0,10,10],[[0,0,2,10],[8,0,10,10]])==.4
assert union_overlap_fraction([0,0,10,10],[[0,0,5,10]])==.5
class Fixture:
    rows=[{'id':1,'width':100,'height':100,'file':'synthetic','valid':[[10,10,20,20,0],[30,30,40,40,0]],'ignore':[[30,30,35,40]]}]
d=R/'metric_fixture';d.mkdir(exist_ok=True)
p=[{'image_id':1,'category_id':1,'bbox':[10,10,10,10],'score':.9}]
a=metrics_from_predictions(Fixture(),p,d,'perfect_small')
assert abs(a['AP']-100)<1e-8 and abs(a['APs75']-100)<1e-8
assert a['ignored_valid_GT']==1
p=[{'image_id':1,'category_id':1,'bbox':[14,10,10,10],'score':.9}]
b=metrics_from_predictions(Fixture(),p,d,'miss')
assert b['AP']==0 and b['APs75']==0
p=[{'image_id':1,'category_id':1,'bbox':[10,10,10,10],'score':.9}]
Fixture.rows[0]['valid']=[[10,10,90,90,0]]
c=metrics_from_predictions(Fixture(),p,d,'large_gt')
assert c['APs75'] is None
save_json(R/'metric_fixture_check.json',{'status':'PASS','independent_known_geometry_cases':['overlapping union .8','disjoint union .4','threshold union .5','perfect original small AP/APs75 100','shifted IoU below .5 gives AP0','large original GT not counted as small'],'not_detector_results':True})
print('Metric fixture PASS; synthetic values are not model accuracy.',flush=True)
