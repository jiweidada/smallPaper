import sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v3 as S3
res={"item":"339 窗口单元测试与独立复核","tests":{}}

def boxes_from_centers(centers):
    b=[]
    for x,y in centers: b.append([x-5,y-5,x+5,y+5])
    return torch.tensor(b,dtype=torch.float32)

# 1) asymmetric: 90 at (x=480,y=160) -> cell (row=2,col=7); 10 at (x=96,y=544)-> (row=8,col=1)
cents=[(480,160)]*90+[(96,544)]*10
rc,cnt=S3.select_window(boxes_from_centers(cents))
W=S3.window_xyxy(rc)
res["tests"]["asym"]={"returned":list(rc),"window":[float(x) for x in W.tolist()],
  "expected_cell":[2,7],"expected_window":[448.,128.,512.,192.],
  "pass": list(rc)==[2,7] and abs(float(W[0])-448)<1e-6 and abs(float(W[1])-128)<1e-6}

# 2) independent pure-numpy implementation straight from 262.make_grid
def indep_select(boxes640, win=64, size=640):
    cx=((boxes640[:,0]+boxes640[:,2])/2).numpy(); cy=((boxes640[:,1]+boxes640[:,3])/2).numpy()
    # 262: grid=[(x,y,min(x+64,w),min(y+64,h)) for y in range(0,h,64) for x in range(0,w,64)]
    best=None; best_n=-1
    for y in range(0,size,win):
        for x in range(0,size,win):
            n=0
            for px,py in zip(cx,cy):
                if x<=px<x+win and y<=py<y+win: n+=1
            if n>best_n: best_n=n; best=(y//win, x//win)   # (row,col), first max in y-then-x order
    return best
N=8
rng=np.random.default_rng(7)
allok=True; mism=[]
for k in range(N):
    pts=[(float(rng.uniform(0,639)),float(rng.uniform(0,639))) for _ in range(200)]
    b=boxes_from_centers(pts)
    rc,_=S3.select_window(b); ref=indep_select(b)
    if tuple(rc)!=tuple(ref): allok=False; mism.append((k,list(rc),list(ref)))
res["tests"]["independent_compare_random"]={"cases":N,"all_equal":allok,"mismatch":mism}

# 3) reverse test: a transposed implementation must fail
def transposed_select(boxes640, win=64, size=640):
    cx=((boxes640[:,0]+boxes640[:,2])/2).numpy(); cy=((boxes640[:,1]+boxes640[:,3])/2).numpy()
    cnt=np.zeros((10,10),dtype=np.int64)
    for x,y in zip(cx,cy):
        if 0<=x<=640 and 0<=y<=640: cnt[min(int(x//win),9),min(int(y//win),9)]+=1
    idx=int(np.argmax(cnt.reshape(-1))) if cnt.max()>0 else 0
    return (idx//10, idx%10)
rc_t=transposed_select(boxes_from_centers(cents))
res["tests"]["reverse_transposed_fails"]={"transposed_result":list(rc_t),
  "correct":[2,7],"detected_as_wrong": list(rc_t)!=[2,7]}

# 4) boundaries
pts=[(0,0)]*50+[(639.9,639.9)]*10
rc4,_=S3.select_window(boxes_from_centers(pts))
res["tests"]["boundary_0"]={"cell":list(rc4),"pass":list(rc4)==[0,0]}
pts2=[(640,640)]*50+[(0,0)]*5  # 640 is edge -> excluded by <=, so (0,0) wins
rc5,_=S3.select_window(boxes_from_centers(pts2))
res["tests"]["boundary_640_excluded"]={"cell":list(rc5),"note":"x=640 落在边界，按 0<=x<=640 计入 col 9 边界格"}
# NaN / out of range
b6=torch.tensor([[float('nan'),0,10,10],[700,700,710,710],[100,100,110,110]])
rc6,_=S3.select_window(b6)
res["tests"]["nan_out_of_range"]={"cell":list(rc6),"note":"NaN/越界被排除，只剩 (100,100)->(row1,col1)"}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\339_result_window_unit_tests.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2)[:3000])
