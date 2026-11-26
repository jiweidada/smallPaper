"""Independent arithmetic checks and source archiving; no model/evaluator access."""
import csv,json,hashlib
from pathlib import Path
PAIR=Path(__file__).resolve().parent;ROOT=PAIR.parent;B=PAIR/'边界与匹配机制_20261026_v1'
DISC=ROOT/'讨论伙伴_错误诊断文献_20261026_v1';LIB=ROOT/'文献_相关原始论文_20261004'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):
 with open(p,'x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 rows=list(csv.DictReader(open(B/'per_GT_bounds.csv',encoding='utf-8-sig')))
 assert len(rows)==77518 and len({(r['seed'],r['annotation_id']) for r in rows})==77518
 result=read(B/'results.json');known=0
 for r in rows:
  bits=r['EMA_threshold_hit_bits'];assert len(bits)==10 and set(bits)<=set('01')
  assert all(int(bits[i])>=int(bits[i+1]) for i in range(9))
  assert int(bits[0])==int(r['EMA_hit50'])
  if '1' in bits:
   known+=1;h=float(r['EMA_highest_matching_threshold']);ix=bits.rfind('1')
   assert abs(h-(.5+.05*ix))<2e-16
   assert float(r['IoU_improvement_lower_bound'])==h-float(r['raw_maxIoU'])
  else:assert not r['IoU_improvement_lower_bound'] and not r['EMA_highest_matching_threshold']
 for s in ('1','2'):
  for name,pred in [('rawG_EMA_hit50',lambda r:r['raw_state50']=='G' and r['EMA_hit50']=='1'),('rawK_EMA_hit50',lambda r:r['raw_state50']=='K' and r['EMA_hit50']=='1'),('rawT_EMA_hit50',lambda r:r['raw_state50']=='T' and r['EMA_hit50']=='1')]:
   rr=[r for r in rows if r['seed']==s and pred(r)];ref=result['populations'][s][name]
   assert len(rr)==ref['n_population']
   for t in (0,.01,.05,.1,.25):assert sum(float(r['IoU_improvement_lower_bound'])>t for r in rr)==ref['bound_above_markers'][str(t)]
 write(B/'independent_arithmetic_checks.json',{'status':'PASS','allGT_rows':len(rows),'known_bounds':known,'unknown_bounds':len(rows)-known,
  'formula_and_bit_threshold_and_population_counts':True,'source_SHA256':{str(p):sha(p) for p in (B/'per_GT_bounds.csv',B/'results.json')},'forwards':0,'evaluator_calls':0})
 text=['基于原始缓存的几何变化下界｜2026-10-26','',
 '这是原raw CPU与已捕获原生EMA的端点比较，执行上下文仍不同。同后端CPU重放是另一个后续检验。',
 '对每GT，r=raw全部300框的最高IoU；h=EMA实际匹配所达到的最高记录阈值。只要h存在，EMA最高IoU−r至少为h−r。高阈值未匹配不能给最高IoU上界，故不能把它写成[h,h+0.05]。',
 '该下界比较整个候选集合的最高重叠，不是同一query的坐标移动，也没有指定发生变化的层。','']
 for s in ('1','2'):
  text.append('seed'+s+'（全GT38759）：')
  for name,label in [('rawG_EMA_hit50','rawG后EMA恢复'),('rawK_EMA_hit50','rawK后EMA恢复'),('rawT_EMA_hit50','rawT且EMA命中参照')]:
   q=result['populations'][s][name];n=q['n_population'];med=q['lower_bound']['quantiles'][3];c=q['bound_above_markers']
   text.append(f'  {label}：n={n}，下界中位数={med:.9f}；下界>0为{c["0"]}/{n}，>0.10为{c["0.1"]}/{n}。')
 text.extend(['','观察：rawG恢复组的最高重叠不只是多数贴着0.50跨过一点；相当部分的保守下界已经>0.10。rawK恢复组也存在严格正下界，故不能把这些个体全解释为固定raw几何上的最后读出变化。',
 '限制：rawG恢复组“下界>0”本身由分组与命中条件保证，不能将其当独立机制证据；下界的幅度才提供额外约束。下界≤0不证明几何变差；未命中组没有h时记未知，不记0。rawT参照并非因果对照。尚无EMA坐标逐项误差界，不能把较大下界直接用作排除所有后端放大的证据。',
 '', '匹配事件解释修订：两个seed的“.5漏检而较高阈值命中”均0/38759。源码归纳证明当前规则强制命中GT集合嵌套，因此0不具鉴别力。详见解释修订_匹配单调性.txt；该假说停止，其他问题继续。',
 '', '复核：77518行公式、阈值bit串、群组计数已独立重算；原12个输入文件SHA不变。没有模型forward、训练或评价器调用。冻结protocol SHA256='+result['protocol_sha256'],
 '', 'Material Passport：数据=548张验证图/38759 GT×两seed；模型=同epoch24 checkpoint model与捕获EMA；评价=原绑定COCO风格协议，GT0以gtMatches识别；不是官方VisDrone挑战成绩。',
 '统计边界：全量有限数据计数，无iid-GT显著性检验；两个seed只报告分别结果；群组条件选择、负下界、结构性零事件及AP非线性均不能混同为因果结论。'])
 (B/'00_下界结果与解释修订.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
 # Preserve partner's originals, archive exact copies in the project's original-paper library.
 ledger=read(DISC/'sources_and_reading.json');archive=[]
 for src in ledger['sources']:
  origin=Path(src['local_path']);assert sha(origin)==src['sha256']
  if src['id']=='tide_author_updated':dest=origin;txt='tide_full.txt'
  elif src['id']=='hoiem2012':dest=LIB/'问题优先诊断_Hoiem_ECCV2012_author.pdf';txt='hoiem2012_full.txt'
  else:dest=LIB/'问题优先诊断_Hoiem_2014_analysis_v2_author.pdf';txt='hoiem_v2_full.txt'
  if not dest.exists():
   with open(dest,'xb') as f:f.write(origin.read_bytes())
  assert sha(dest)==src['sha256']
  ts=DISC/'text'/txt;td=LIB/'_text'/('问题优先诊断_20261026_'+txt)
  if not td.exists():
   with open(td,'xb') as f:f.write(ts.read_bytes())
  assert sha(td)==sha(ts)
  archive.append({**src,'archived_original_path':str(dest),'archived_text_path':str(td),'original_unchanged':True,'root_reading_scope':'TIDE pages6,16 local extracted text only; Hoiem read by discussion partner, root has not read its body.'})
 write(LIB/'_sources_问题优先诊断讨论_20261026_v1.json',{'client_date':'2026-10-26','sources':archive,'discussion_reading_record':str(DISC/'sources_and_reading.json'),'archiving_only_no_new_reading_claim':True})
 print('Independent bounds checks and primary-source archiving complete')
if __name__=='__main__':main()
