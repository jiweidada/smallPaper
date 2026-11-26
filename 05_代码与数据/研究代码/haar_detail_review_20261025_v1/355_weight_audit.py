"""Weight-only Haar audit; no model construction, detector forward, or optimizer."""
import os,sys
sys.dont_write_bytecode=True
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
import json,hashlib,datetime,gc
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent
EN=OUT.parent.parent
items=[
 ("VisDrone_C_seed0",EN/"data/01_C主线_已完成/weights/p2_skip_main_seed0_best_stg2_remote.pth"),
 ("UAVDT_C_seed0",EN/"data/07_第二数据集_20261017/训练产物回收_20261025_v2/A/files/outputs/uavdt_c_full96_seed0/best_stg2.pth"),
 ("UAVDT_C_seed1",EN/"data/07_第二数据集_20261017/训练产物回收_20261025_v2/B/files/outputs/uavdt_c_full96_seed1/best_stg2.pth")]
source=EN/"data/01_C主线_已完成/C实际代码_远端_hgnetv2_light.py"
paired=EN/"相位机制论文_20261018/data/paired_full12_deltas_pp.json"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
def write(name,data):
 with (OUT/name).open("x",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write("\n")
basis=torch.tensor([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]],dtype=torch.float64)/2
assert torch.equal(basis@basis.T,torch.eye(4,dtype=torch.float64))
rows=[]
for name,path in items:
 c=torch.load(path,map_location="cpu",weights_only=False)
 w=c["ema"]["module"]["backbone.p2_skip.restore.weight"].double().reshape(256,16,4)
 q=torch.einsum("ocp,kp->ock",w,basis);energy=q.square().sum((0,1));frac=energy/energy.sum()
 reconstructed=torch.einsum("ock,kp->ocp",q,basis)
 assert torch.equal(reconstructed,w)
 assert float(energy.sum())==float(w.square().sum())
 rows.append({"model":name,"checkpoint":str(path),"checkpoint_sha256":sha(path),"checkpoint_last_epoch0":c["last_epoch"],
              "weights":"ema.module","tensor":"backbone.p2_skip.restore.weight","layout":"F.pixel_unshuffle: per reduced channel [00,01,10,11]",
              "analysis_dtype":"float64","energy_Frobenius_squared":energy.tolist(),"fractions_DC_Hx_Hy_HH":frac.tolist(),
              "DC_divided_by_average_single_detail_energy":float(energy[0]/energy[1:].mean()),
              "reconstruction_max_abs":float((reconstructed-w).abs().max()),
              "Parseval_absolute_difference":float(abs(energy.sum()-w.square().sum()))})
 del c,w,q;gc.collect()
d=json.loads(paired.read_text(encoding="utf-8-sig"));mean_deltas=[]
for seed in [0,1]:
 for split in ["val","testdev"]:
  a=next(r for r in d if r["seed"]==seed and r["split"]==split and r["arm"]=="C" and r["reference"]=="baseline")
  b=next(r for r in d if r["seed"]==seed and r["split"]==split and r["arm"]=="C" and r["reference"]=="C-mean")
  mean_deltas.append({"seed":seed,"split":split,"Cmean_minus_baseline_APs_pp":a["APs"]-b["APs"],
                      "Cmean_minus_baseline_AP_pp":a["AP"]-b["AP"]})
try:timestamp=datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
except Exception:timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-7))).isoformat()
result={"checked_client_time":timestamp,"numerical_verification":"VERIFIED weight decomposition; detector dependence and proposed method untested",
 "Haar_basis_rows_DC_Hx_Hy_HH":basis.tolist(),"rows":rows,"VisDrone_Cmean_baseline_deltas":mean_deltas,
 "parameter_accounting":{"reduction_conv":64*16,"BN_trainable":32,"original_restore":256*64,
 "current_branch":17440,"detail_only_restore":256*48,"detail_only_branch":13344,
 "saved":4096,"branch_saved_fraction":4096/17440,
 "whole_VisDrone_C_saved_fraction":4096/10249437,"existing_rank16_branch":6176,
 "assumption":"Preserve current reduce(Conv-BN-ReLU) before Haar. Moving Haar before nonlinearity changes the intervention."},
 "inference_limits":["Weight energy is not activation/output energy, task utility, or conditional information.",
 "Three detail subspaces versus one DC subspace; for iid isotropic weights expected detail fraction is .75, not .50. This is dimensional accounting, not a fitted statistical null.",
 "Cmean trained separately is not DC-only intervention of a trained C; joint DC-detail utility can differ.",
 "P3 is learned nonlinear downsampling; it is not guaranteed to contain a recoverable copy of P2 cell means.",
 "Only three checkpoints were decomposed. Five existing dataset-seed C/Cmean pairs are not five tests of the untrained detail-only method.",
 "No global novelty conclusion follows from this bounded search."],
 "counts":{"checkpoint_loads":3,"model_builds":0,"detector_forward_calls":0,"optimizer_steps":0,"remote_server_calls":0},
 "sources":[{"path":str(p),"sha256":sha(p)} for p in [Path(__file__),source,paired]+[p for _,p in items]],
 "fallacy_scan":{"Simpson":"Dataset, seed and split kept separate; positive Cmean seed1 retained.",
 "ecological":"Weight aggregate not identified with GT-level improvement.",
 "Berkson":"No successful detection subset used.",
 "collider":"No common-detected conditioning.",
 "base_rate":"Three detail bands compared with one mean band; no 50:50 expectation.",
 "regression_to_mean":"Existing fixed best checkpoints; no checkpoint reselection.",
 "survivorship":"All available specified three weight decompositions and four mean-baseline contrasts retained.",
 "look_elsewhere":"Analysis is exploratory; no p values or confirmatory novelty claim.",
 "forking_paths":"New dependence test is a draft; no detector inference performed.",
 "causation":"Weight norm does not prove redundancy, necessity or benefit after retraining.",
 "reverse_causality":"No assertion that a large weight norm caused the paired AP gain."}}
write("355_weight_only_audit.json",result)
lines=["355｜Haar细节支路：权重与既有均值对照复核","时间："+timestamp,"",
"结论：分解数值复现；“均值基本多余”尚无依据。新方法未训练，也未执行模型前向。","",
"模型                         DC       Hx       Hy       HH      DC/单个细节均值"]
for r in rows:
 f=r["fractions_DC_Hx_Hy_HH"]
 lines.append(r["model"]+"   "+" ".join(f"{v:.6f}" for v in f)+"   "+f'{r["DC_divided_by_average_single_detail_energy"]:.4f}')
lines+=["","DC是每份权重中能量最大的单个子带；细节有3个子空间，共约2/3不是偏好细节的证明。",
"按3:1维数划分，iid各向同性权重的细节期望份额为3/4；这只作维数解释，不能把训练后权重当成该随机模型的独立样本。",
"没有输入激活协方差，不能从W的Frobenius能量推断Wz输出贡献；输出能量还含子带协方差与交叉项。","",
"VisDrone C-mean−baseline（百分点，全部已有共同seed）："]
for r in mean_deltas:lines.append(f'seed{r["seed"]} {r["split"]}: APs {r["Cmean_minus_baseline_APs_pp"]:+.6f}; AP {r["Cmean_minus_baseline_AP_pp"]:+.6f}')
lines+=["","仅引用seed0 test-dev APs≈0会漏掉seed1的+0.505pp，也漏掉两个val的正增益。",
"既有C/C-mean配对只支持在受测条件下加入差异信息有增量用途，不支持联合C中的均值分量没有用途。","",
"保留当前16通道Conv-BN-ReLU，再只传48维Haar细节：支路17440→13344，少4096（23.486%）；全VisDrone C模型少约0.03996%。",
"已有rank16支路仅6176参数，不能仅凭比C少参数就宣称新设计为更优成本选择。","",
"进一步测试见355_修订预检草案.txt；它检验冻结C的可裁剪性，不预先把裁剪通过当成冗余证明或方法新颖性。",
"本次仅加载3份checkpoint作FP64权重分解；模型构建/前向/优化步数均0。"]
with (OUT/"355_数值核查与推论边界.txt").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
print(json.dumps({"status":"weight_numbers_reproduced_inference_not_established","models":len(rows),
 "Cmean_seed1_testdev_APs_gain_pp":mean_deltas[-1]["Cmean_minus_baseline_APs_pp"],
 "detector_forwards":0,"output":str(OUT)},ensure_ascii=False),flush=True)
