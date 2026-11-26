"""Independent counting formula and human-readable result tables, no inference."""
import json,csv,hashlib,collections,gzip
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;RDIR=OUT/'attempt02';ROOT=OUT.parent
PREV=ROOT/'问题优先诊断_20261026_v1'
BASE=ROOT/'data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,j):
    with open(OUT/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(n,s):
    with open(OUT/n,'x',encoding='utf-8') as f:f.write(s)
def tab(h,rr):return '\n'.join([' | '.join(h),' | '.join(['---']*len(h))]+[' | '.join(map(str,r)) for r in rr])
R=read(RDIR/'results.json');rows=csvread(RDIR/'per_GT_transition.csv');groups=csvread(RDIR/'summary_by_group.csv')
old={(int(r['seed']),int(r['annotation_id'])):r for r in csvread(PREV/'per_GT.csv')}
derived={};GT0=[]
for seed in (1,2):
    z=np.load(BASE/f'seed{seed}/run/endpoint_epoch24/GT_hits.npz',allow_pickle=False)
    nz={int(a):j for j,a in enumerate(z['annotation_id'])}
    rr=[r for r in rows if int(r['seed'])==seed];assert len(rr)==38759
    for th,t in ((50,0),(75,5)):
        a=[];b=[]
        for r in rr:
            aid=int(r['annotation_id']);raw=int(r[f'raw_hit{th}']);ema=int(r[f'EMA_hit{th}'])
            assert raw==int(old[seed,aid][f'hit{th}'])
            if aid!=0:assert ema==int(z['hits'][nz[aid],t])
            else:GT0.append({'seed':seed,'IoU':th/100,'primary_raw':raw,'primary_EMA':ema,'native_EMA':int(z['hits'][nz[aid],t])})
            a.append(raw);b.append(ema)
        code=np.asarray(a)*2+np.asarray(b);c=np.bincount(code,minlength=4).tolist()
        expected=R['all_GT_primary'][str(seed)][str(th)]
        labels=['persistent_miss','recovered','reverse_loss','both_hit']
        assert c==[expected[k] for k in labels]
        for kind in ('size','category','size_category','area','density32','occlusion'):
            gg=[g for g in groups if int(g['seed'])==seed and float(g['IoU'])==th/100 and g['group_type']==kind]
            assert sum(int(g['n_gt']) for g in gg)==38759
            assert all(sum(int(g[k]) for g in gg)==expected[k] for k in labels)
        derived[f'{seed}_{th}']=dict(zip(labels,c))
assert sha(OUT/'per_GT_transition.csv')==sha(RDIR/'per_GT_transition.csv')
inputs=read(RDIR/'input_manifest.json')['files'];assert all(sha(p)==v['sha256'] for p,v in inputs.items())
assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text('utf-8').strip()==R['protocol_sha256']
# Explicit list of the GT missed by all four archived branch/seed outputs.
by={s:{int(r['annotation_id']):r for r in rows if int(r['seed'])==s} for s in (1,2)}
common=[]
for aid,r in by[1].items():
    if r['transition50']=='persistent_miss' and by[2][aid]['transition50']=='persistent_miss':
        common.append({k:r[k] for k in ['image_id','annotation_id','category_id','short_side_640','size_group','density32','occlusion']}|
            {'raw_state_seed1':r['raw_state50'],'raw_state_seed2':by[2][aid]['raw_state50']})
assert len(common)==R['cross_seed']['50']['both_seeds_persistent_miss']
with open(OUT/'persistent_four_snapshots_GT.csv','x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(common[0]));w.writeheader();w.writerows(common)
write('independent_checks.json',{'bincount_recounts':derived,'all_disjoint_strata_sum_to_population':True,
    'old_new_pairing_verified_rowwise':True,'legacy_EMA_hits_verified_for_all_nonzero_GT':True,'GT0_separate':GT0,
    'failed_attempt_and_fixed_attempt_CSV_identical_SHA256':sha(RDIR/'per_GT_transition.csv'),
    'inputs_rehashed':len(inputs),'changed_inputs':[],'protocol_unchanged':True,'four_snapshot_persistent_GT':len(common),
    'scope':'same assistant, second counting formula; not an independent blind review or new evaluation'})

def alltable(th):
    rr=[]
    for s in ('1','2'):
        x=R['all_GT_primary'][s][str(th)]
        rr.append([s,x['raw_hit'],x['raw_miss'],x['recovered'],x['persistent_miss'],x['reverse_loss'],x['both_hit'],x['EMA_hit'],x['net_hit_gain']])
    return tab(['seed','raw命中','raw漏检','恢复','持续漏检','反向丢失','两边都命中','EMA命中','净增'],rr)
def group(s,kind,label):return next(g for g in groups if int(g['seed'])==s and g['IoU']=='0.5' and g['group_type']==kind and g['group']==label)
sizes=['[0,4)','[4,8)','[8,16)','[16,32)','[32,inf)']
size_table=tab(['640短边','共同GT分母','seed1 恢复/持续/反向','seed2 恢复/持续/反向'],[[a,group(1,'size',a)['n_gt'],
    '/'.join(group(1,'size',a)[k] for k in ('recovered','persistent_miss','reverse_loss')),
    '/'.join(group(2,'size',a)[k] for k in ('recovered','persistent_miss','reverse_loss'))] for a in sizes])
names=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
class_table=tab(['类别','共同GT分母','seed1 恢复/持续/反向/净增','seed2 恢复/持续/反向/净增'],[[n,group(1,'category',str(i))['n_gt'],
    '/'.join(group(1,'category',str(i))[k] for k in ('recovered','persistent_miss','reverse_loss','net_hit_gain')),
    '/'.join(group(2,'category',str(i))[k] for k in ('recovered','persistent_miss','reverse_loss','net_hit_gain'))] for i,n in enumerate(names)])
flows=tab(['原raw状态','seed1分母→EMA命中/仍漏','seed2分母→EMA命中/仍漏'],[[st,
    next(f"{f['n_raw_state']}→{f['EMA_hit']}/{f['EMA_miss']}" for f in R['raw_state_flows'] if f['seed']==1 and f['IoU']==.5 and f['raw_state']==st),
    next(f"{f['n_raw_state']}→{f['EMA_hit']}/{f['EMA_miss']}" for f in R['raw_state_flows'] if f['seed']==2 and f['IoU']==.5 and f['raw_state']==st)] for st in 'GSKCMT'])
legacy=tab(['seed','原生零哨兵：恢复/持续/反向/两边命中','逐GT恢复GT0：恢复/持续/反向/两边命中'],[[s,
    '/'.join(str(R['legacy_zero_sentinel_sensitivity'][s]['50'][k]) for k in ('recovered','persistent_miss','reverse_loss','both_hit')),
    '/'.join(str(R['all_GT_primary'][s]['50'][k]) for k in ('recovered','persistent_miss','reverse_loss','both_hit'))] for s in ('1','2')])
report=f'''raw与EMA逐GT对照｜2026-10-26

Material Passport：academic-research-suite / experiment-agent validate；ANALYZED。
本轮完成现有档案的身份连接、匹配记录复核与计数。无模型加载/前向/评价器调用/训练/GPU/远端/test读取。

一、对象与操作定义
同一BASE24的两个seed、同一last.pth、last_epoch=23：raw使用model分支的现有CPU预测缓存；EMA使用当时原生评估捕获的ema.module结果。548图、38759个GT；以annotation_id连接，同时核对image_id/category_id。不把同序号query当跨权重的对象身份。
恢复=raw未命中→EMA命中；持续漏检=raw未命中→EMA未命中；反向丢失=raw命中→EMA未命中；两边命中=raw命中→EMA命中。四格互斥且和为38759。
主阈值IoU≥0.5，参照IoU≥0.75；均沿用完整图像、类别感知的一对一匹配、top300、score>0.001、maxDets=[1,10,100]。分组在完整匹配之后进行，没有对子集重新匹配。

二、两个seed分别的完整结果
IoU≥0.5，每行分母均38759：
{alltable(50)}

恢复率分母是raw漏检：seed1 3132/15283=20.493%，seed2 2385/14577=16.361%。
持续漏检比例同用raw漏检分母：12151/15283=79.507%，12192/14577=83.639%。
反向丢失率分母是raw命中：1530/23476=6.517%，1642/24182=6.790%。
净增=恢复−反向丢失，不能只用恢复数概括EMA变化。

IoU≥0.75，每行分母仍38759：
{alltable(75)}

三、原raw失败状态的去向（IoU≥0.5）
G=所有末端raw框均无合格几何；S=有几何但真类分数均不过0.001；K=合格真类项未入top300；C=被每类100上限清空；M=保留合格候选但评价未命中；T=raw实际命中。
{flows}
G恢复率17.996%/12.912%，K恢复率28.953%/29.431%；这是不同raw状态群体的描述，不能直接称为两个环节的因果可改善率。
按尺寸5组×类别10类共同层、合并G+K GT数加权：seed1 G/K恢复率18.350%/30.901%（47层，G12463/K2359，排除23个缺乏K参照的G）；seed2为13.186%/33.017%（48层，G11809/K2392，排除17个G）。共同层以外样本保留在全体表中。粗分层、遮挡/图像差异及按raw状态筛选的偏差仍在。

四、尺寸与类别参照（IoU≥0.5）
{size_table}
小于4px组：seed1 raw漏3875，恢复249、持续3626；raw命中1077，反向丢失194；seed2 raw漏3879，恢复249、持续3630；raw命中1073，反向丢失178。两seed净增仅55/71，不能把249个恢复写成净收益。

{class_table}
seed2 truck为恢复45、丢失70、净−25；bus为16进16出、净0。类别净方向并不由总体正净增保证。完整尺寸×类别、密度与遮挡参照及分母见attempt02/summary_by_group.csv、raw_state_to_EMA_by_group.csv。

五、跨seed与图像参照
10504/38759个GT在两个seed的raw与EMA四份现有输出中均未检出。588/38759在两seed中都恢复，245/38759在两seed中都反向丢失。IoU≥0.75时四份均未检出20496/38759。
seed1有501/548图出现恢复、431/548图出现反向丢失，416/548图两种变化并存；seed2对应484/548、431/548、409/548。不能把变化简化为一组图整体变好、另一组图整体变差。
四份持续漏检GT的精确ID清单保存于persistent_four_snapshots_GT.csv。持续的是终端未检出状态，尚不能说四份输出都没有几何候选或共享同一原因。

六、当前可以与不能解释什么
可确认：raw-G→EMA命中的2247/1527个GT，EMA端点必有合格几何且完成真类检出，该GT的末端几何覆盖缺口已消失。不能确认是哪一个query、哪一层或哪条内部路径改变。
raw-K→EMA命中可能来自原类别项排名变化，也可能来自另一合格框/query；没有跨状态框/身份对应，不能说原被压低的类别项获释放。
raw-G/K→EMA仍漏检，只证明终端未检出延续；EMA可能已从几何缺失转为筛选失败，也可能反向迁移。当前没有EMA完整框池的同口径G/K图。
四份共同漏检削弱“换到这个EMA档案后所有困难即消失”的解释；但这些都是同架构、同训练轮次的相关预测，不能称结构上不可解、信息必定不存在或成熟模型也如此。
raw为现有CPU推理/Windows评价，EMA为原生GPU评估捕获；本轮未做同后端同batch前向对照。权重平均、执行上下文与阈值临界变化仍是竞争解释，不把全部逐GT变化单变量归因于EMA。EMA24也不是训练成熟度或完整96/132轮终点。
命中计数不直接推出AP变化；本轮未调用评价器重算AP，也没有按GT独立同分布作总体推断。

七、ID0与证据链
EMA存档把dtMatches=0当未匹配，因此GT id0有歧义。已从gtMatches中的真实检测ID，回查dtIds/dtMatches与ignore标志；两个seed的GT0均在IoU.5/.55实际匹配，IoU.75未匹配。它在raw .5也匹配，所以主表GT0为“两边命中”。
{legacy}
历史EMA TP50 25077/24924，逐GT匹配语义恢复GT0后为25078/24925。历史AP与源文件完全不改；净增与恢复/反向丢失均不受该GT影响，持续漏检与两边命中各差1。
源EMA manifest的4文件SHA、manifest与training_completion绑定、completion与last.pth SHA、标注SHA、identity均相符；全部10个IoU列的原生GT_hits都从匹配记录精确重建。
24个输入收口重哈希无变更；逐GT CSV另一种bincount计数与主计数一致，各尺寸/类别等互斥分组之和回到38759。它们是同一助手的公式核对，不是独立盲审。

八、执行记录与权威文件
首轮统计/CSV完成，JSON导出因NumPy bool序列化中断。根目录results.json是不完整残留，不作权威输入。原文件保留，analyze_attempt01.py保留首轮代码。
attempt02只修正导出布尔值类型并另建输出目录，冻结协议字节不变。两轮完整per_GT_transition.csv的SHA相同。修复未改变样本、阈值、计数或匹配。
权威结果：attempt02/results.json；attempt02/per_GT_transition.csv；attempt02/summary_by_group.csv；attempt02/raw_state_to_EMA.csv；attempt02/per_image_transition.csv。
校验：attempt02/verification.json、input_manifest.json；independent_checks.json。
协议：protocol.json，SHA {R['protocol_sha256']}。
脚本：analyze.py、check_and_report.py；运行日志run_attempt02.log。
与新对话的原因讨论将另存；截至本数值报告，只形成以上观察与解释边界，没有提出模块、损失函数或启动后续实验。
'''
text('01_逐GT结果与证据边界.txt',report)
text('README_权威输出.txt','本轮权威输出在attempt02目录。根目录results.json为保留的首轮导出失败残留。请先读01_逐GT结果与证据边界.txt；原因讨论另存02文件。原数据/缓存/历史报告未修改。')
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10.6,4.7))
 for ax,seed in zip(axes,('1','2')):
    x=R['all_GT_primary'][seed]['50'];mat=np.array([[x['persistent_miss'],x['recovered']],[x['reverse_loss'],x['both_hit']]])
    ax.imshow(mat,cmap='Blues',vmin=0,vmax=24000)
    for i in range(2):
     for j in range(2):
      n=mat[i,j];den=x['raw_miss'] if i==0 else x['raw_hit']
      ax.text(j,i,f'{n:,}\n{n/den*100:.2f}% of row',ha='center',va='center',fontsize=12,color='white' if n>12000 else '#132f4c')
    ax.set_xticks([0,1],['EMA miss','EMA hit']);ax.set_yticks([0,1],[f"Raw miss (n={x['raw_miss']:,})",f"Raw hit (n={x['raw_hit']:,})"])
    ax.set_title(f'Seed{seed} | IoU >= .50',fontweight='bold');ax.tick_params(length=0)
 fig.suptitle('Paired GT outcomes | BASE24 raw / EMA | All 38,759 GT',fontsize=12)
 fig.subplots_adjust(left=.17,right=.98,wspace=.55,bottom=.10,top=.85)
 fig.savefig(OUT/'GT_transition_map.png',dpi=150,facecolor='white');plt.close(fig)
except ImportError:pass
print('Independent counts and source checks passed; human report created.')
