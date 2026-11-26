import sys, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
torch.set_num_threads(2)
from uic_model import Encoder, D
m=Encoder(); x=torch.randn(2,3,128,128); f=m(x); print("f shape", f.shape, "D", D)
print("mem tokens", f.flatten(2).transpose(1,2).shape)
print("f[b] reshape", f[0].reshape(D,-1).shape)
