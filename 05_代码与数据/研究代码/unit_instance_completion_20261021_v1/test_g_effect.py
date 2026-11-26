import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
sys.path.insert(0, r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\320_geometry_calibrated_core_v2')
import g_effect_341 as G
from gcal import GeometryCalibratedHead
torch.set_num_threads(2)
m = GeometryCalibratedHead(); m.eval()
W=torch.tensor([128.,128.,192.,192.])
b=torch.tensor([[140.,140.,160.,170.],[150.,150.,170.,180.],[160.,130.,190.,160.],[130.,160.,150.,185.],[170.,170.,195.,195.]])
cls=torch.tensor([0,1,2,0,1])
e={'window_xyxy_640':W,'fullp_boxes':b,'fullp_labels':cls}
out={}
for name,k in [('zero',torch.zeros(10)),('0.5',torch.full((10,),0.5)),('mix',torch.tensor([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))]:
    d,n = G.delta_g_effect(m,[e],kappa_override=k)
    out[name]={'delta':round(d,6),'windows':n,'verdict':G.classify(d)}
print(json.dumps(out,indent=1))
json.dump({'item':'341 G有效性度量脚本 synthetic test','results':out},open(os.path.join(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1','341_result_g_effect_synthetic.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)