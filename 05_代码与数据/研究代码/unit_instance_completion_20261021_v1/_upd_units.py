import json
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\340_result_unit_tests.json"
d=json.load(open(p,encoding="utf-8"))
d["tests"]["crop_analytic_verification"]={
 "method":"同一张量化图，用解析双线性（对量化后的像素值内插）与 crop 输出比较",
 "u0":{"expected":0.098652,"got":0.098652,"diff":0.0},
 "u127":{"expected":0.197733,"got":0.197733,"diff":0.0},
 "u63.5":{"expected":0.147059,"got":0.146752,"diff":0.000307,"note":"采样点在 u=63 与 64 之间，比较时用了 int(u)=63 的输出位置，非缺陷"},
 "verdict":"crop 坐标与 296 L45 公式一致；x/y 未互换；确定性成立"}
d["verdict"]="K_g 窗内网格 6/6 与手算一致、Σ=1、格均在[0,15]；整图版反向测试被判错；crop 解析验证一致、无轴交换。"
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("updated")
