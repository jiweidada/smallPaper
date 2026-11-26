import sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
res={"item":"验证 select_window 行列转置 bug"}
# 1) construct an ASYMMETRIC case: most centers in x-cell 7, y-cell 2  => correct window (row=2,col=7) -> x[448,512), y[128,192)
boxes=torch.zeros(100,4)
# put 90 centers at x=7*64+32=480, y=2*64+32=160
boxes[:90,0]=480; boxes[:90,1]=160; boxes[:90,2]=480+10; boxes[:90,3]=160+10
# 10 at x=1*64+32=96, y=8*64+32=544
boxes[90:,0]=96; boxes[90:,1]=544; boxes[90:,2]=106; boxes[90:,3]=554
rc,cnt=S.select_window(boxes)
W=S.window_xyxy(rc)
print("returned cell:",rc,"window:",[float(x) for x in W.tolist()])
print("EXPECTED (row=2 y, col=7 x): window [448,128,512,192]")
res["asym"]={"returned_cell":list(rc),"returned_window":[float(x) for x in W.tolist()],
  "expected_window":[448.0,128.0,512.0,192.0],
  "bug_confirmed": not (abs(float(W[0])-448)<1e-3 and abs(float(W[1])-128)<1e-3)}
# 2) decode: cnt is indexed [x,y]; check density location
print("cnt argmax (first index=x?):", np.unravel_index(int(np.argmax(cnt)),cnt.shape), "cnt[7,2]=",int(cnt[7,2]),"cnt[2,7]=",int(cnt[2,7]))
res["cnt_indexing"]={"cnt_shape":list(cnt.shape),"cnt[7,2]":int(cnt[7,2]),"cnt[2,7]":int(cnt[2,7]),
  "note":"cnt[i,j] 中 i=x格，j=y格；而 window_xyxy 把 r 当 y、c 当 x -> 转置"}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\339_verify_transpose.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=1))
