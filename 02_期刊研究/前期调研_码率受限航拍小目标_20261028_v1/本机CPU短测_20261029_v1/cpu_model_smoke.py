"""YOLO26s CPU interface test with random weights and synthetic residuals only."""
from pathlib import Path
from io import BytesIO
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
import ultralytics
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

root = Path(__file__).resolve().parent
package = Path(ultralytics.__file__).resolve().parent
pinned = root.parent / "深入核查_20261028_v1/source_code/v8.4.191_2004cba021c6/ultralytics"
assert ultralytics.__version__ == "8.4.191", ultralytics.__version__
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
assert not torch.cuda.is_available(), "This smoke test is CPU-only."
source_checks = []
for rel in ("cfg/models/26/yolo26.yaml", "cfg/default.yaml", "nn/modules/head.py", "nn/tasks.py", "utils/loss.py", "utils/tal.py"):
    installed, archive = package / rel, pinned / rel
    first = hashlib.sha256(installed.read_bytes()).hexdigest()
    second = hashlib.sha256(archive.read_bytes()).hexdigest()
    # Normalize CRLF only for source comparison; retain both exact file hashes in receipt.
    normalized_equal = installed.read_bytes().replace(b"\r\n", b"\n") == archive.read_bytes().replace(b"\r\n", b"\n")
    source_checks.append({"file": rel,"installed_sha256":first,"archive_sha256":second,
                          "identical_after_newline_normalization":normalized_equal})
    assert normalized_equal, f"Pinned source mismatch: {rel}"

cfg = yaml.safe_load((package / "cfg/models/26/yolo26.yaml").read_text(encoding="utf-8"))
cfg.update(nc=10, scale="s")
(root / "yolo26s_smoke_10class.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
counts = {"construction":0,"explicit":0}
stage = ["construction"]
def count_head(module, args):
    if isinstance(module, Detect):
        counts[stage[0]] += 1
hook = torch.nn.modules.module.register_module_forward_pre_hook(count_head)
started = time.perf_counter()
model = DetectionModel(cfg=cfg, ch=3, nc=10, verbose=False).to("cpu")
construction_seconds = time.perf_counter()-started
model.args = get_cfg()
model.eval().requires_grad_(False)
head = model.model[-1]
head.end2end = False
assert head.cv2 is not None and head.one2one_cv2 is not None
assert head.nc == 10 and head.stride.tolist() == [8.,16.,32.]
criterion = v8DetectionLoss(model, tal_topk=10)
stage[0] = "explicit"

# Native STAL candidate geometry, with artificial GT at a known position on the 640 input grid.
test_feats=[torch.zeros(1,1,side,side) for side in (80,40,20)]
grid, stride_tensor=make_anchors(test_feats,head.stride)
centers=grid*stride_tensor
tiny_gt=torch.tensor([[[29.,29.,31.,31.]]])
def raw_in_box(box):
    x1,y1,x2,y2=box[0,0]
    return (centers[:,0]>x1)&(centers[:,0]<x2)&(centers[:,1]>y1)&(centers[:,1]<y2)
raw_count=int(raw_in_box(tiny_gt).sum())
expanded=criterion.assigner.select_candidates_in_gts(centers,tiny_gt,torch.ones(1,1,1))
big_gt=torch.tensor([[[300.,300.,340.,340.]]])
big_native=criterion.assigner.select_candidates_in_gts(centers,big_gt,torch.ones(1,1,1))
assert criterion.assigner.stride_val==16 and raw_count==0 and int(expanded.sum())==5
assert torch.equal(big_native[0,0],raw_in_box(big_gt))
assert torch.equal(tiny_gt,torch.tensor([[[29.,29.,31.,31.]]]))
stal_fixture={"kind":"synthetic GT and native candidate geometry only", "tiny_gt_xyxy":[29,29,31,31],
              "original_box_geometric_candidates":raw_count,"native_stal_geometric_candidates":int(expanded.sum()),
              "minimum_candidate_side_pixels":criterion.assigner.stride_val,"large_box_geometry_unchanged":True,
              "final_positive_assignment_tested":False,"input_gt_unchanged":True}

def state_hash():
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
before_hash = state_hash()
bn_before = {name: (m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone())
             for name,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)}
images = json.loads((root/"image_checks.json").read_text(encoding="utf-8"))["images"]
timings = []
shape_record = None

def prepare(record):
    payload = next(p for p in record["payloads"] if p["quality"] == 20)
    raw = Path(payload["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == payload["sha256"]
    with Image.open(BytesIO(raw)) as im:
        pixels = np.asarray(im.convert("RGB"))
    info = record["letterbox_dry_run"]
    rw,rh = info["resized_wh"]
    l,t,r,b = info["pad_ltrb"]
    pixels = cv2.resize(pixels,(rw,rh),interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(pixels,t,b,l,r,cv2.BORDER_CONSTANT,value=(114,114,114))
    assert padded.shape == (640,640,3)
    inp = torch.from_numpy(np.ascontiguousarray(padded.transpose(2,0,1))).unsqueeze(0).float()/255
    gt=record["gt"]
    batch={"batch_idx":torch.zeros(len(gt),dtype=torch.long),
           "cls":torch.tensor([g["class"] for g in gt],dtype=torch.float32).reshape(-1,1),
           "bboxes":torch.tensor([g["xywh_normalized"] for g in gt],dtype=torch.float32).reshape(-1,4)}
    return inp,batch,(l,t,l+rw,t+rh),payload

def forward_loss(inp,batch,label):
    global shape_record
    begin=time.perf_counter()
    with torch.no_grad():
        output=model(inp)
        assert isinstance(output,tuple) and len(output)==2
        decoded,raw=output
        assert set(raw)=={"one2many","one2one"}
        selected=raw["one2many"]
        assert set(selected)=={"boxes","scores","feats"}
        assert selected["scores"].shape == (1,10,8400)
        assert selected["boxes"].shape == (1,4,8400)
        vector,_=criterion.loss(selected,batch)
        scalar=vector.sum()
        assert vector.shape == (3,) and torch.isfinite(vector).all() and torch.isfinite(decoded).all()
        # Only shape/finite/repeatability evidence; these random-weight loss values are not research findings.
        elapsed=time.perf_counter()-begin
        timings.append({"label":label,"seconds":elapsed,"loss_finite":True,"loss_scalar":float(scalar),
                        "loss_vector":vector.tolist()})
        if shape_record is None:
            shape_record={"decoded":list(decoded.shape),"boxes":list(selected["boxes"].shape),
                          "scores":list(selected["scores"].shape),"feats":[list(t.shape) for t in selected["feats"]]}
        print(json.dumps({"forward":len(timings),"label":label,"seconds":round(elapsed,4),"finite":True}),flush=True)
        return scalar.detach().clone(),decoded.detach().clone()

inp,batch,valid,payload=prepare(images[0])
l,t,r,b=valid
yy,xx=torch.meshgrid(torch.arange(640,dtype=torch.float32)+.5,torch.arange(640,dtype=torch.float32)+.5,indexing="ij")
inside=(xx>=l)&(xx<r)&(yy>=t)&(yy<b)
rx=((xx-(l+r)/2)/16+.5).clamp(0,1)
ry=((yy-(t+b)/2)/16+.5).clamp(0,1)
weights=torch.stack([(1-rx)*(1-ry),rx*(1-ry),(1-rx)*ry,rx*ry])*inside
assert torch.allclose(weights.sum(0)[inside],torch.ones_like(xx[inside]),atol=1e-7)
assert weights.min() >= 0 and weights.max() <= 1
candidate=inp.clone()
candidate[:,:,inside]=inp[:,:,inside]*.97+.015
residual=candidate-inp
assert (residual[:,:,~inside]==0).all()
losses=[]
first_output=None
for bits in range(16):
    gate=sum((weights[i] for i in range(4) if bits & (1<<i)),torch.zeros_like(xx))
    mixed=inp if bits==0 else candidate if bits==15 else inp+gate*residual
    assert (mixed[:,:,~inside]==inp[:,:,~inside]).all()
    assert mixed.min()>=0 and mixed.max()<=1
    scalar,decoded=forward_loss(mixed,batch,f"synthetic_residual_subset_{bits:04b}")
    losses.append(float(scalar))
    if bits==0:
        first_output=(scalar,decoded)
repeat_loss,repeat_decoded=forward_loss(inp,batch,"repeat_first_degraded")
assert torch.equal(repeat_loss,first_output[0]) and torch.equal(repeat_decoded,first_output[1])
for record in images[1:]:
    x,gt,_,_=prepare(record)
    forward_loss(x,gt,"q20_"+record["file_name"])
hook.remove()
after_hash=state_hash()
assert before_hash==after_hash
for name,m in model.named_modules():
    if name in bn_before:
        assert all(torch.equal(x,y) for x,y in zip(bn_before[name],(m.running_mean,m.running_var,m.num_batches_tracked)))
assert counts["explicit"]==19==len(timings)
assert not model.training and all(not m.training for m in model.modules())
versions={d.metadata["Name"]:importlib.metadata.version(d.metadata["Name"]) for d in importlib.metadata.distributions()}
result={"status":"PASS","scope":"CPU functional smoke only; random weights; artificial brightness residual; no trained detector/restorer",
        "python":sys.version,"executable":sys.executable,"package":str(package),"versions":versions,
        "torch_version":torch.__version__,"cuda_available":torch.cuda.is_available(),"torch_threads":torch.get_num_threads(),
        "initialization":"random seed=0; no checkpoint download or load", "classes":10,"scale":"s","input_shape":[1,3,640,640],
        "parameters":sum(p.numel() for p in model.parameters()),"strides":head.stride.tolist(),
        "unfused_dual_head_retained":True,"fixed_loss_branch":"one2many; v8DetectionLoss(tal_topk=10).loss(raw,batch)[0].sum()",
        "source_checks":source_checks,"native_stal_fixture":stal_fixture,"construction_seconds":construction_seconds,
        "construction_detect_head_calls":counts["construction"],"explicit_detector_forwards":counts["explicit"],
        "actual_output_shapes":shape_record,"input_payload_sha256":payload["sha256"],
        "bn_modules_checked":len(bn_before),"batchnorm_unchanged_during_explicit_forwards":True,
        "state_before_sha256":before_hash,"state_after_sha256":after_hash,"repeat_exact":True,
        "subset_forward_count":16,"additional_repeat_count":1,"additional_train_image_forwards":2,
        "synthetic_candidate_is_learned_restoration":False,"restorer_calls":0,"optimizer_steps":0,"backward_calls":0,
        "training_epochs":0,"ap_computed":False,"scientific_hypothesis_validated":False,
        "timings_not_a_benchmark":timings,"elapsed_seconds_since_model_construction":time.perf_counter()-started}
(root/"cpu_model_smoke.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
(root/"environment_versions.txt").write_text("\n".join(f"{k}=={v}" for k,v in sorted(versions.items(),key=lambda x:x[0].lower()))+"\n",encoding="utf-8")
print(json.dumps({"status":"PASS","explicit_forwards":counts["explicit"],"construction_head_calls":counts["construction"],
                  "repeat_exact":True,"state_unchanged":True,"training_epochs":0,"hypothesis_validated":False},ensure_ascii=True),flush=True)
