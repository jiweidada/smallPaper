import os, sys, json, torch, copy
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:32]
def mk(arm='R'):
    torch.manual_seed(2026102101)
    m=T.make_common_init(arm,1); m.train(); return m
def batches():
    es=[T.load_entry(1,i) for i in IDS]; return [T.collate(es[0:16]), T.collate(es[16:32])]
res={'item':'340 训练链路合同验证','tests':{}}
bs=batches()

m1=mk(); o1=T.build_optimizer(m1); T.run_steps(m1,o1,bs,0,2,1218)
s1={k:v.detach().clone() for k,v in m1.state_dict().items()}

m2=mk(); o2=T.build_optimizer(m2)
T.run_steps(m2,o2,bs,0,1,1218)
rng=T.capture_rng()
ck=os.path.join(OUT,'_resume_test.pt')
T.save_checkpoint(ck,m2,o2,1,0,12345,rng)
m3=mk(); o3=T.build_optimizer(m3)
st,ep,ss=T.load_checkpoint(ck,m3,o3)
T.run_steps(m3,o3,bs,1,1,1218)
s3={k:v.detach().clone() for k,v in m3.state_dict().items()}
diff=[k for k in s1 if not torch.equal(s1[k],s3[k])]
res['tests']['resume_bitwise']={'keys':len(s1),'diff_keys':len(diff),'first_diff':diff[:5],'pass':len(diff)==0}
print('resume:', res['tests']['resume_bitwise'])

gp={n:p.detach().clone() for n,p in m3.named_parameters()}
for n,p in m3.named_parameters(): p.requires_grad_(False)
before={k:v.detach().clone() for k,v in m3.state_dict().items()}
with torch.no_grad(): _=m3.trunk(bs[0]['rgb'])
after=m3.state_dict()
res['tests']['global_zero_grad_state']={'state_unchanged':all(torch.equal(before[k],after[k]) for k in before),
  'all_grad_none':all(p.grad is None for p in m3.parameters())}
print('global:', res['tests']['global_zero_grad_state'])
json.dump(res,open(os.path.join(OUT,'340_result_contracts.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)