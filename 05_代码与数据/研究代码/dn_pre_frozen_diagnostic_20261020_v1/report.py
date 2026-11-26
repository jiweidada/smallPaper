"""Collect the completed frozen diagnostic, without any new model execution."""
import csv,datetime,zoneinfo
from common import *

def main():
 resource_limits()
 result=json.loads((OUT/'03_全部局部反事实原始结果.json').read_text(encoding='utf8'))
 split=json.loads((OUT/'05_裁剪联动分解.json').read_text(encoding='utf8'))
 protocol=json.loads((OUT/'01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
 preflight=json.loads((OUT/'02_实现与运行前核验.json').read_text(encoding='utf8'))
 assert result['state']=='complete' and result['new_training_steps']==0
 primary=[r for r in result['rows'] if r['cross_batch_primary']]
 assert len(primary)==4 and len(result['rows'])==8
 rows=[]
 for r in result['rows']:
  for group,g in r['groups'].items():
   row={'seed':r['seed'],'generator_batch':r['generator_batch'],'evaluation_batch':r['evaluation_batch'],'primary_cross_batch':r['cross_batch_primary'],'group':group,'pairs':g['before']['pairs'],'loss_before':g['before']['loss_contribution'],'loss_include_DN':g['include_DN']['loss_contribution'],'loss_exclude_DN_last':g['exclude_DN_last']['loss_contribution'],'DN_extra_loss_positive_harmful':g['DN_effect'],'mean_IoU_before':g['before']['mean_IoU'],'mean_IoU_include_DN':g['include_DN']['mean_IoU'],'mean_IoU_exclude_DN_last':g['exclude_DN_last']['mean_IoU'],'SGD_cross_gradient_dot_DN':g['eval_ordinary_dot_generator_DN'],'SGD_cross_gradient_dot_ordinary':g['eval_ordinary_dot_generator_ordinary'],'Adam_decrease_without_current_DN':g['decrease_excluding_DN_with_historical_Adam_momentum'],'clip_include_DN':r['clip_include_DN'],'clip_exclude_DN_last':r['clip_exclude_DN_last']}
   rows.append(row)
 for filename,selected in [('06_跨批主结果.csv',[v for v in rows if v['primary_cross_batch']]),('07_全部局部反事实明细.csv',rows)]:
  with (OUT/filename).open('x',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
 checks={'native_sources_unchanged':True,'checkpoints_unchanged':True,'preflight_scripts_unchanged':True,'same_CPU_DN_box_noise_across_model_seeds':{},'cached_head_parity':{},'new_training_steps':0,'virtual_updates_written_to_model':False}
 for p,h in preflight['scripts'].items():assert sha(CODE/p)==h
 for item in protocol['matching_and_native_source_identity']:assert sha(CODE/item['relative'])==item['sha256']
 for seed in [1,2]:
  assert sha(protocol['checkpoints'][str(seed)]['path'])==protocol['checkpoints'][str(seed)]['sha256']
  receipt=json.loads((OUT/f'seed{seed}/capture_complete.json').read_text(encoding='utf8'))
  assert receipt['model_parameters_unchanged'] and receipt['BN_buffers_restored'] and receipt['optimizer_step_calls']==0
  for file in receipt['capture_files']:assert sha(OUT/f'seed{seed}'/file['file'])==file['sha256']
 for batch in [0,1]:
  caches=[torch.load(OUT/f'seed{s}/batch{batch:02d}_frozen_head.pt',map_location='cpu',weights_only=False) for s in [1,2]]
  assert caches[0]['dn_split']==caches[1]['dn_split']
  diff=float((caches[0]['anchors'][:,:caches[0]['dn_split']]-caches[1]['anchors'][:,:caches[1]['dn_split']]).abs().max())
  checks['same_CPU_DN_box_noise_across_model_seeds'][str(batch)]={'max_abs_difference':diff,'same':diff==0}
  assert diff==0
  for c in caches:checks['cached_head_parity'][f"seed{c['seed']}_batch{batch}"]=c['verification']
 save_json(OUT/'09_数值核验与资产完整性.json',checks)
 completed=datetime.datetime.fromisoformat(result['completed_utc']).astimezone(zoneinfo.ZoneInfo('America/Los_Angeles'))
 text=[
 'Material Passport',
 'Origin Skill: academic-research-suite / experiment-agent',
 'Origin Mode: run；本地冻结诊断及描述性解释',
 f'Origin Date: {completed.isoformat()}',
 'Verification Status: UNVERIFIED（没有独立第三方复核；本地原损失/梯度与资产完整性校验通过）',
 'Version Label: dn_pre_frozen_v1',
 '',
 '结果：存在局部干扰，但方向依赖批次和种子；当前不支持把DN稳定有害作为新训练主线。',
 '建议：按计算前已写下的跨批/跨seed一致性口径，停止优先投入“DN稳定损害普通初始化→固定末层阻断”的训练解释。保留原始数据，不新增训练。这个建议不是证明所有阶段或所有条件下都不存在干扰。',
 '',
 '实际执行与对照口径',
 '两份VisDrone BASE24 last.pth，在线原模型而非EMA，last_epoch=23，旧132轮配方第24轮。原模型和criterion均严格加载；末层AdamW参数ID、形状、组别和状态映射核验通过。',
 '每seed使用同一组development batch00/batch01，各4张增强输入：GT数分别[516,365,209,360]与[110,41,182,251]。不是随机抽取的新验证集，不能当独立留出。',
 'ordinary pre boxes沿原indices_go/num_boxes_go；DN pre沿indices_dn/(num_boxes*dn_num_group)，损失为5L1+2GIoU。不去重、不改匹配、不改变归一化。虚拟比较冻结本次原模型产生的对应。',
 '原生全模型总损失求导用于计算clip_max_norm=.1的全局裁剪；只在克隆的末层W/b上，从同一历史AdamW状态计算包含/删除当前DN pre末层梯度的局部虚拟一步。评估时其他参数、输入特征、参考输入及匹配保持固定。',
 'DN额外损失 = 含DN更新后的普通pre损失 - 去掉DN末层更新后的普通pre损失。正值表示DN相对于该对照增加局部损失；不一定表示含DN更新后的总损失比更新前升高。',
 '这是原损失差，不是AP或百分点。模型前向及原参数梯度为FP32；所得FP32虚拟权重的缓存末层损失用FP64求值，避免小差相减的FP32舍入。',
 '',
 '跨批主表（全部ordinary匹配；以10^-4为单位显示，即原始损失差乘10^4）',
 'seed | 更新批次→评估批次 | DN额外损失×10^4 | 局部作用 | 普通配对曝光数',
 ]
 for r in primary:
  g=r['groups']['all_ordinary'];effect=g['DN_effect']
  text.append(f"{r['seed']} | {r['generator_batch']:02d}→{r['evaluation_batch']:02d} | {effect*1e4:+.6f} | {'不利' if effect>0 else '有利' if effect<0 else '无差'} | {g['before']['pairs']}")
 text += ['', '输入锚点IoU<.5组（仍沿原完整num_boxes_go计算该组损失贡献）',
 'seed | 更新批次→评估批次 | DN额外损失×10^4 | 配对曝光数']
 for r in primary:
  g=r['groups']['input_anchor_IoU_lt_0_5']
  text.append(f"{r['seed']} | {r['generator_batch']:02d}→{r['evaluation_batch']:02d} | {g['DN_effect']*1e4:+.6f} | {g['before']['pairs']}")
 text += ['',
 '解释：seed1两个方向相反，seed2两个方向都不利。困难组同样没有消除seed1的方向反转。不能用3/4多数票认定稳定有害，也不能把相反结果平均掉。',
 '单步损失差绝对值虽小，但它们与部分去DN更新本身的单步下降量相当，不能据绝对数值小就宣布“没有作用”。seed2的00→01中，两种Adam局部更新均使全组损失相对原点升高；该行的正差表示DN额外增加损失，不代表删DN就必然实现下降。',
 '',
 '补充：全局裁剪联动（看到主结果后登记的探索性分解，不替代主表）',
 '固定原含DN裁剪系数，计算直接删除DN梯度的差；再计算删除梯度后重新求裁剪系数带来的差。两项严格相加等于主差值。这是指定顺序的局部分解，不是唯一的训练因果占比。',
 'seed | 更新→评估 | 直接项×10^4 | 裁剪联动项×10^4 | 含DN/去DN裁剪系数比',
 ]
 for r in split['rows']:
  g=r['groups']['all_ordinary']
  text.append(f"{r['seed']} | {r['generator_batch']:02d}→{r['evaluation_batch']:02d} | {g['direct_given_original_clip']*1e4:+.6f} | {g['clip_coupling']*1e4:+.6f} | {g['include_clip_divided_by_exclude_clip']:.6f}")
 r=next(r for r in split['rows'] if r['seed']==2 and r['generator_batch']==0)
 g=r['groups']['all_ordinary'];share=g['clip_coupling']/g['total_primary_effect']
 text += [f'例如seed2的00→01，裁剪联动项占该顺序分解总差的{share*100:.2f}%。不能把全部差值归结为DN梯度方向与普通梯度相反。',
 '另一个描述事实：seed2生成批01的同批普通/DN末层梯度余弦为正（约+.06393），但01→00的跨批DN作用不利。只看生成批同批余弦，不一定识别跨批局部影响；这不是新的优化算法或检测收益证据。',
 '',
 '核验与资源',
 '本机torch2.5.1+cpu，2线程、CPU亲和[13,15]、Below Normal；未使用GPU，未连接远程或操作其他任务/监测。四次原模型前向均完成；原生求导后不调用optimizer.step，不写回参数。',
 'checkpoint SHA前后保持一致；模型参数和BN缓冲前后哈希一致；缓存末层重建pre框与原框最大差为0；原pre损失差为0；缓存末层参数梯度与原生梯度最大差低于1e-6。',
 'CPU使用原存档的整数DN种子，但不是历史CUDA随机流的逐位回放。两模型本次DN输入框噪声逐位相同已核验。两批原增强输入的文件SHA、浮点输入SHA核验通过，没有打开heldout预测。',
 '只执行局部克隆参数反事实，非完整模型同时更新的训练反事实；固定上游时，删DN pre损失与仅阻断末层梯度在这个检查里等价，不能证明保留上游梯度的价值。',
 '',
 '不能据此声称',
 '不能证明DN长期训练有害，不能证明关闭DN会提升AP，不能证明冲突只在pre末层，不能证明有限FDR支持域或最终漏检已由该路径造成，也不构成跨数据集泛化。',
 '没有p值、置信区间或等效性结论：只有两个模型seed和两个重复使用的开发批次，双向比较相关；配对曝光不是独立GT。未按最终失败者筛选，困难组由pre之前锚点定义。没有采用10%等事后数值门槛。',
 '',
 '文件与复现',
 '01：计算前锁定资产与口径；02：实现SHA、合成检查及运行前核验；03：原始全部反事实；04/05：事后明确登记的裁剪分解；06/07：CSV主表及完整明细；09：数值和资产核验；seed1/seed2：缓存activation、锚点、生产匹配、梯度、末层Adam状态及完成回执。',
 '运行命令：D:\\Anaconda\\python.exe -s -X utf8 -B 英文小论文\\code\\dn_pre_frozen_diagnostic_20261020_v1\\run.py（已成功执行；输出目录为追加创建，不能覆盖重跑）。',
 '补充分解命令：同Python执行decompose_clip.py，仅读取缓存，无新模型执行。',
 ]
 with (OUT/'08_结果解释与去留建议.txt').open('x',encoding='utf8') as f:f.write('\n'.join(text)+'\n')
 files=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file():files.append({'relative':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 save_json(OUT/'10_结果证据SHA清单.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command_exit_code':0,'primary_cases':4,'frozen_model_forwards':4,'training_steps':0,'scripts':{p.name:sha(p) for p in sorted(CODE.glob('*.py'))},'files':files})
 print(json.dumps({'report':'08_结果解释与去留建议.txt','primary_cases':4,'assets_verified':True,'recommendation':'do not initiate training for stable-harm hypothesis; local effects are mixed'}))

if __name__=='__main__':main()
