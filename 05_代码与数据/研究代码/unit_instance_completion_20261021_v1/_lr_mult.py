import sys, math
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
total=1218
base=[T.lr_at(s,total) for s in range(total)]
s=sum(base)
print("baseline LR sum:", round(s,6), "-> max kappa displacement ~", round(s,4))
for mult in (10, 100, 1000):
    print(f"x{mult}: LR sum={round(s*mult,3)} -> max displacement ~{round(s*mult,2)}")
# what lr sum is needed for D>=0.1 with realistic efficiency (~20% of max)?
print("=== to reach D=0.1 at 20% efficiency, need LR sum:", round(0.1/0.2,3))
print("=== baseline gives D ~", round(s*0.2,4), "at 20% efficiency")