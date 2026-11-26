import sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
res={"item":"337 坐标换算单元测试","tests":{}}
# 1) normalized cxcywh -> 640 xyxy roundtrip
b=torch.tensor([[0.5,0.5,0.2,0.1],[0.25,0.75,0.5,0.25]])
xy=S.norm_cxcywh_to_640_xyxy(b)
exp=torch.tensor([[320-64,320-32,320+64,320+32],[160-160,480-80,160+160,480+80]],dtype=torch.float32)
res["tests"]["norm_to_640"]={"got":xy.tolist(),"expected":exp.tolist(),"allclose":bool(torch.allclose(xy,exp,atol=1e-4))}
# 2) orig xywh -> 640 for non-square 1400x788
g=torch.tensor([[700.,394.,140.,78.8]])  # full-image-half box
xy2=S.orig_xywh_to_640_xyxy(g,1400,788)
# expect x*640/1400, y*640/788
ex2=torch.tensor([[700*640/1400,394*640/788,(700+140)*640/1400,(394+78.8)*640/788]])
res["tests"]["orig_to_640_nonsquare"]={"got":xy2.tolist(),"expected":ex2.tolist(),"allclose":bool(torch.allclose(xy2,ex2,atol=1e-4))}
# 3) boundary: full image
g3=torch.tensor([[0.,0.,1400.,788.]])
xy3=S.orig_xywh_to_640_xyxy(g3,1400,788)
res["tests"]["boundary_full_image"]={"got":xy3.tolist(),"is_0_640":bool(torch.allclose(xy3,torch.tensor([[0.,0.,640.,640.]]),atol=1e-4))}
# 4) window 640 -> orig
W=torch.tensor([320.,320.,330.,330.])
o=S.window_640_to_orig_xyxy(W,1400,788)
ex4=torch.tensor([320*1400/640,320*788/640,330*1400/640,330*788/640])
res["tests"]["window_to_orig"]={"got":o.tolist(),"expected":ex4.tolist(),"allclose":bool(torch.allclose(o,ex4,atol=1e-4))}
# 5) roundtrip: orig->640->orig
g5=torch.tensor([[123.,45.,67.,89.]])
a=S.orig_xywh_to_640_xyxy(g5,1400,788)
back=torch.stack([a[0,0]*1400/640,a[0,1]*788/640,a[0,2]*1400/640,a[0,3]*788/640])
ex5=torch.tensor([123.,45.,190.,134.])
res["tests"]["roundtrip_orig640"]={"got":back.tolist(),"expected":ex5.tolist(),"allclose":bool(torch.allclose(back,ex5,atol=1e-3))}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\337_result_coord_unit_tests.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
