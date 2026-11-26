import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native")
import stage1_cache as S
from PIL import Image
torch.set_num_threads(2)
from engine.deim.postprocessor import PostProcessor
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
model=S.load_real_model(CKPT)
pp=PostProcessor(num_classes=10,num_top_queries=300,use_focal_loss=True)
out={"item":"336 阶段1 raw/eval 一致性","checks":[]}
for pid in [269,5429,3536,4113]:
    im=imgs[pid]; pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    b,l=S.global_forward_640(model,pil)
    fullp=S.make_fullp(b,l)
    r=pp({"pred_logits":l.unsqueeze(0),"pred_boxes":b.unsqueeze(0)}, torch.tensor([[640.,640.]]))
    # PostProcessor top300 (before >0.001): compare its score-sorted set to fullP's
    pp_scores=r[0]["scores"]; pp_labels=r[0]["labels"]; pp_boxes=r[0]["boxes"]
    # fullP is subset of top300 after threshold; verify every fullP element appears in PP top300 with same score
    ok=all(any(abs(float(s)-float(ps))<1e-6 for ps in pp_scores) for s in fullp["scores"])
    out["checks"].append({"image_id":pid,"n_fullp":int(fullp["boxes"].shape[0]),
      "n_pp":int(pp_scores.shape[0]),"fullp_subset_of_pp_top300":bool(ok),
      "pp_score_min":float(pp_scores.min()),"pp_score_max":float(pp_scores.max())})
json.dump(out,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\336_result_stage1_raweval.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
