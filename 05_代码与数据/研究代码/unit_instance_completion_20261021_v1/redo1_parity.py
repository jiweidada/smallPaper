import sys, os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
sys.dont_write_bytecode=True
NATIVE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native"
EXP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
sys.path.insert(0,NATIVE); sys.path.insert(0,EXP)
import torch; torch.set_num_threads(2)
from geometry import ResidualGeometryDEIM
from engine.deim.postprocessor import PostProcessor
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
base=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1"
pj=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo1_real_model.json"
out=json.load(open(pj,encoding="utf-8")); out.setdefault("tests",{})
cfg=json.load(open(os.path.join(base,"seed1","bundle","resolved.yml"),encoding="utf-8"))
m=ResidualGeometryDEIM(cfg["ResidualGeometryDEIM"]["recipe"],arm="baseline")
ck=torch.load(base+r"\seed1\run\last.pth",map_location="cpu",weights_only=True)
m.load_state_dict(ck["model"],strict=True); m.eval()
for p in m.parameters(): p.requires_grad_(False)
torch.manual_seed(0); x=torch.rand(1,3,640,640)
with torch.no_grad(): o=m(x)
pl,pb=o["pred_logits"],o["pred_boxes"]
orig=torch.tensor([[640.,640.]])

# manual native reimplementation (as in uic_fusion) vs PostProcessor
def manual(pl,pb,orig,C=10,topq=300):
    cx,cy,w,h=pb.unbind(-1)
    b=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)*orig.repeat(1,2).unsqueeze(1)
    s=torch.sigmoid(pl); sv,idx=torch.topk(s.flatten(1),topq,dim=-1)
    lab=idx%C; bi=idx//C
    sel=b.gather(1,bi.unsqueeze(-1).repeat(1,1,4))
    return lab,sel,sv
lab2,box2,sco2=manual(pl,pb,orig)
pp=PostProcessor(num_classes=10,num_top_queries=300,use_focal_loss=True)
r=pp({"pred_logits":pl,"pred_boxes":pb},orig)
diff_lab=not torch.equal(lab2[0],r[0]["labels"]); diff_sco=float((sco2[0]-r[0]["scores"]).abs().max()); diff_box=float((box2[0]-r[0]["boxes"]).abs().max())
out["tests"]["raw_native_parity"]={"labels_identical":not diff_lab,"score_max_abs_diff":diff_sco,"box_max_abs_diff":diff_box,
 "note":"手动原生实现(与uic_fusion.native_postprocess同式) vs DEIM PostProcessor，同一真实输出"}
# eval prepare: convert_to_xywh consistency (does not filter)
xywh=convert_to_xywh(r[0]["boxes"])
recon=torch.stack([xywh[:,0],xywh[:,1],xywh[:,0]+xywh[:,2],xywh[:,1]+xywh[:,3]],-1)
out["tests"]["eval_prepare_consistency"]={"max_recon_diff":float((recon-r[0]["boxes"]).abs().max()),
 "num_boxes":int(xywh.shape[0]),"note":"prepare_for_coco_detection 的 convert_to_xywh 可逆、不过滤 score"}
out["tests"]["score_threshold_effect_on_real"]={
 "postprocessor_scores_min":float(r[0]["scores"].min()),
 "num_above_0p001":int((r[0]["scores"]>0.001).sum()),
 "note":"真实输出 300 query 分数均 >0.001，故 317 的 >0.001 阈值在真实预测上不产生过滤效果（与自建口径的一致性含义需绑定评估库确认）"}
out["not_executed"].append("raw↔绑定评估库(custom_pycocoeval) parity：本地无该包且后端为Linux .so")
json.dump(out,open(pj,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({k:out["tests"][k] for k in ["raw_native_parity","eval_prepare_consistency","score_threshold_effect_on_real"]},ensure_ascii=False,indent=2))