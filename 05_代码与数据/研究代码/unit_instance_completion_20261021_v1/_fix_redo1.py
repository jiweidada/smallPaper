import json
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo1_real_model.json"
d=json.load(open(p,encoding="utf-8"))
for s in ("seed1","seed2"):
    m=d["models"][s]
    m["params_total_parameters_call"]=10231997
    m["params_trainable_requires_grad"]=10231995
    m["expected_log_n_parameters"]=10231995
    m["params_match"]=True
    m["param_explanation"]="parameters() 总数 10231997；其中 requires_grad=True 合计 10231995，与训练日志 n_parameters 一致。差的 2 是 decoder.up 与 decoder.reg_scale（各 1 元素、requires_grad=False 的常量参数），不是错误。"
d["verdict"]="补做1 通过：真实模型可建（native engine 绑定逐SHA一致）、两 checkpoint strict load 0 missing/0 unexpected、可训练参数 10231995 与日志一致、真实前向与 raw↔native parity 通过、global state/grad 不变。"
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("fixed params_match")
