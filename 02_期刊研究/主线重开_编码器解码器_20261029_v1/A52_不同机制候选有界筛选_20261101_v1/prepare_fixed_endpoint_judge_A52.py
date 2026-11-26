from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parent
A51=ROOT.parent/'A51_跨分辨率守恒交换_20261101_v1'
source=(A51/'judge_fixed_endpoint_A51.py').read_text(encoding='utf-8')
source=source.replace("'phase':'A51'","'phase':'A52'")
source=source.replace('a301ec5fb1f6a5c4675ef129f11e8359fa46c5d49640f636ccbeb343703cb8e8','c13b61bdbc09d81a051108f7a28516f14ae3a3521a81d8247574129ca413d027')
source=source.replace("ROOT/'capacity_result.json'","ROOT/'capacity_v2/capacity_result.json'")
anchor="    assert capacity['capacity_only_optimizer_updates']==6\n"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+"    guard_failure=load(ROOT/'capacity_v1_failed_archive/capacity_result.json')\n    assert guard_failure['whole_model_forwards']==1 and guard_failure['capacity_only_optimizer_updates']==0\n")
source=source.replace('9876858','9809592').replace('73746','6480')
source=source.replace('165284','185612').replace('136481','155903')
source=source.replace("'cumulative_formal_epochs':168","'cumulative_formal_epochs':192")
source=source.replace("'new_model_forwards_this_phase':9+sum","'new_model_forwards_this_phase':10+sum")
source=source.replace("fixed_endpoint_judgment_A51.json","fixed_endpoint_judgment_A52.json")
start=source.index("      'limits':[")
stop=source.index("      'next_extra_training_queued'",start)
source=source[:start]+"""      'limits':['One seed fixed12 is not statistical significance or cross-dataset generalization',
         'B and val548 were visible in multiple previous selections; this is not an untouched test',
         'M/Q differ in categorical-versus-Gaussian marginal embedding, not calibrated joint-box probability',
         'Support-grid encoding and diagonal per-query quality may contribute; inter-object discrimination is not isolated',
         'Fourier expectations, integrated encoding and position-relation attention have prior foundations',
         'Relation-DETR+ and Exact-NeRF have only abstract-level reading in the current audit'],
"""+source[stop:]
ast.parse(source)
p=ROOT/'judge_fixed_endpoint_A52.py';assert not p.exists();p.write_text(source,encoding='utf-8')
print(json.dumps({'status':'FIXED_ENDPOINT_JUDGE_PREPARED_SYNTAX_ONLY','executed_against_results':False,'new_model_or_optimizer_operations':0}))

