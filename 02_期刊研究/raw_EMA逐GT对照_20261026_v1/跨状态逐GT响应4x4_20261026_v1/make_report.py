"""Render frozen joined counts as a human-readable report; no new analysis/evaluation."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "attempt01"
result = json.loads((OUT / "all_GT_results.json").read_text("utf-8"))
stage = json.loads((OUT / "fixed_raw_A_stage_results.json").read_text("utf-8"))
verify = json.loads((OUT / "verification.json").read_text("utf-8"))
assert verify["status"] == "PASS"
lines = []

def emit(text=""):
    lines.append(text)

def fraction(n, d):
    return f"{n}/{d} ({100*n/d:.2f}%)" if d else "N/A（分母为0）"

emit("同GT raw×EMA读出响应4×4：全GT与固定raw A阶段")
emit("主要数据连接于2026-10-26，独立复核与讨论跨至2026-10-27（America/Los_Angeles）；seed与IoU阈值分别列出。")
emit("协议SHA256：a1ea55445363e9db94d4902b4afd434c07d82c568f8099f86150ded2e766a275。")
emit("每seed为全部548张验证图、38759个GT；原GT id 0有效。raw/EMA是两个训练来源各自的两个保存状态。")
emit("C−B为主，二元状态按(B,C)；B−A参照按(A,B)。00两臂漏、01新增、10丢失、11两臂命中。")
emit("下列矩阵行=raw响应，列=EMA响应；每格是同一GT在两种保存状态中的状态交叉，四种状态与完整分母均保留。")
emit("A为原top300；B为只从A已使用query身份取top600；候选臂C为原有全query中取top600。")
emit("固定阶段中的C是原A的‘合格导出候选未过每类cap’，不是候选臂C。阶段来自既有raw A记录，并按表的IoU阈值分别固定。")

emit("\n一、先看全部GT的4×4表")
for contrast, label in (("C_minus_B", "C−B 主对照"), ("B_minus_A", "B−A 参照")):
    for th in ("50", "75"):
        for seed in ("1", "2"):
            v = result["summaries"][seed][th][contrast]
            emit(f"\n{label}；IoU .{th}；seed {seed}；N={v['GT']}")
            emit("raw\\EMA       00       01       10       11      raw合计")
            for code, row, margin in zip(result["state_order"], v["matrix"], v["raw_margins"]):
                emit(f"{code:>7}  " + " ".join(f"{n:>8}" for n in row) + f"  {margin:>10}")
            emit("EMA合计 " + " ".join(f"{n:>8}" for n in v["EMA_margins"]) + f"  {v['GT']:>10}")
            emit("共享新增01→01：相对raw新增 " + fraction(v["shared_gain"], v["raw_gain"]) +
                 "；相对EMA新增 " + fraction(v["shared_gain"], v["EMA_gain"]))
            emit("共享丢失10→10：相对raw丢失 " + fraction(v["shared_loss"], v["raw_loss"]) +
                 "；相对EMA丢失 " + fraction(v["shared_loss"], v["EMA_loss"]))
emit("\n解读：共同新增表示同一GT在两种状态下均由对应读出变化找回，并不表示同一query、框、排序或原因。")
emit("00→00是两状态中该对比的两臂都漏，并非原A持续漏检的完整集合。01→11或11→01等也有明确状态含义，不应被归成‘无效收益’。")
emit("C−B中，raw新增GT在EMA中的去向如下；四格之和是该raw新增分母：")
for th in ("50", "75"):
    for seed in ("1", "2"):
        v = result["summaries"][seed][th]["C_minus_B"]
        miss, gain, loss, hit = v["matrix"][1]
        emit(f"IoU .{th} seed{seed} raw新增{v['raw_gain']}：EMA 00={miss}、01={gain}、10={loss}、11={hit}。")
emit("其中EMA 11是两臂已经命中，EMA 00是两臂都漏；不能把未共享新增一概称为EMA失效。")

emit("\n二、固定raw A阶段后，在同一批GT内分层")
emit("G：原池无达标几何；S：真类对分数未达门槛；K：没有达标真类对被原A导出；C：有导出但未过cap；M：过cap仍未命中；T：原A命中。")
emit("每格沿用同一份raw A阶段标签，不另选EMA的A-K。S组本轮为0仍保留；后两列为同GT跨状态的共享新增、共享丢失。")
for contrast, label in (("C_minus_B", "C−B 主对照"), ("B_minus_A", "B−A 参照")):
    for th in ("50", "75"):
        for seed in ("1", "2"):
            emit(f"\n{label}；IoU .{th}；seed {seed}")
            emit("raw A阶段       组N    raw新增  raw丢失  EMA新增  EMA丢失  共同新增  共同丢失")
            for code in stage["stage_order"]:
                v = stage["summaries"][seed][th][contrast][code]
                emit(f"{code:>9} {v['GT']:>9} {v['raw_gain']:>9} {v['raw_loss']:>9} "
                     f"{v['EMA_gain']:>9} {v['EMA_loss']:>9} {v['shared_gain']:>9} {v['shared_loss']:>9}")
emit("\n结构性限制：raw A-G表示原raw完整300框池无达标几何；因此该GT在原raw B与候选臂C的响应均为00。")
emit("raw A-T在本次B−A对照中为raw 11，是本次已验证无B−A反向丢失与阶段定义共同产生的约束。它们不能作为新的机制阴性证据。")
emit("raw A-G组中，EMA新增GT在EMA臂C、EMA丢失GT在EMA臂B已有实际命中，因此各自的EMA池已有达标匹配几何；这不等于读出规则修复了原raw池的G缺口。")
emit("两种保存状态的框、分数和相互竞争可能变化；该交叉表不能把这些因素拆开，也不能定位几何变化的上游来源。")
emit("raw A-K是原A的首缺阶段；前轮操作地图中的B-K是对B读出定义的阶段，两者不应混作同一分母或同一事件。")

for th in ("50", "75"):
    for seed in ("1", "2"):
        g = stage["summaries"][seed][th]["C_minus_B"]["G"]
        k = stage["summaries"][seed][th]["C_minus_B"]["K"]
        whole = result["summaries"][seed][th]["C_minus_B"]
        emit(f"C−B .{th} seed{seed}：raw A-G组EMA新增 {g['EMA_gain']}/{g['GT']}（该组内），"
             f"也是 {g['EMA_gain']}/{whole['EMA_gain']}（EMA全部新增中的占比）；"
             f"共享新增 {whole['shared_gain']}，其中raw A-K组 {k['shared_gain']}。")
        emit("  固定raw A-K组内共享新增：相对raw新增 " + fraction(k["shared_gain"], k["raw_gain"]) +
             "；相对EMA新增 " + fraction(k["shared_gain"], k["EMA_gain"]) + "。")
emit("固定raw A-K组的两个分母不同；.75下EMA新增仅26/16例，不能用单侧比例宣称对象稳定。")

emit("\n三、先前讨论中关心的固定类和尺度参照")
with (OUT / "fixed_class_area_size_4x4.csv").open(encoding="utf-8-sig", newline="") as stream:
    grouped = list(csv.DictReader(stream))
for typ, group, th in (("class", "7", "0.5"), ("size", "[0,4)", "0.75")):
    for seed in ("1", "2"):
        subset = [r for r in grouped if r["seed"] == seed and r["iou"] == th and r["contrast"] == "C_minus_B"
                  and r["group_type"] == typ and r["group"] == group and r["fixed_raw_A_stage"] == "all"]
        assert len(subset) == 16
        get = {(r["raw_response"], r["EMA_response"]): int(r["count"]) for r in subset}
        r_gain = sum(get["01", e] for e in result["state_order"])
        e_gain = sum(get[r, "01"] for r in result["state_order"])
        r_loss = sum(get["10", e] for e in result["state_order"])
        e_loss = sum(get[r, "10"] for r in result["state_order"])
        emit(f"{typ}={group}, IoU {th}, seed{seed}, GT={subset[0]['group_GT']}："
             f"raw新增/丢失 {r_gain}/{r_loss}，EMA新增/丢失 {e_gain}/{e_loss}，"
             f"共同新增/共同丢失 {get['01','01']}/{get['10','10']}。")
emit("类7的负净变化不表示是同一批GT受损，也不等于该类AP下降。极小组seed2的EMA新增1、丢失1：净变化为0，但两个GT不同。")
emit("全部10类、3档原面积、5档短边及其与固定raw A阶段交叉的完整4×4均已输出；此处两组是前轮讨论提出的参照，不据此选择新门槛。")

emit("\n四、执行核验和边界")
emit(f"连接{verify['GT_rows']}行、每seed {verify['images_per_seed']}图和{verify['GT_per_seed']}个GT；"
     f"三表的完整GT key相同，GT0存在，raw/EMA的A命中逐GT匹配既有阶段记录。")
emit(f"已精确复现既有{verify['prior_exact_ID_sets_replayed']}个新增/丢失GT ID集；"
     f"{verify['partitions_reconstruct_cellwise']}项固定分层逐格重构全表检查通过；"
     f"{verify['input_files_rehashed']}个输入的分析前后SHA不变。")
independent_path = ROOT / "independent_crosscheck.json"
if independent_path.is_file():
    independent = json.loads(independent_path.read_text("utf-8"))
    assert independent["status"] == "PASS" and independent["cells_checked"] == 128
    emit("又以原始两张hit表按GT身份集合交集独立复核全部128个全GT格子，均一致；详见independent_crosscheck.json。")
independent_stage_path = ROOT / "independent_stage_check.json"
if independent_stage_path.is_file():
    independent_stage = json.loads(independent_stage_path.read_text("utf-8"))
    assert independent_stage["status"] == "PASS" and independent_stage["fixed_raw_A_stage_cells_checked"] == 768
    emit("再用原始阶段表及两张hit表的GT身份集合交集，独立复核全部768个固定raw A阶段格子，均一致；详见independent_stage_check.json。")
emit("面积只在raw和EMA两张hit表之间核验；短边由三张表核验，按固定边界分箱后与另外两表的既有尺寸组核对。")
emit("原raw A阶段采用既有记录的edge几何；先前raw操作地图在.50/.75的edge与实际评价bbox阈值跨越均为0，本轮没有重新计算几何。")
emit("只读取已保存CSV/JSON并计算描述性计数；0新增前向、0模型构建/权重加载、0新增评价、0训练。没有新bootstrap、显著性检验、AP分解或原因贡献。")
emit("固定GT上的交集反映本次两个保存状态的响应对象；它不单独解释原因为评分、几何或匹配，也不推断完整训练终点、其他模型或数据域。")
emit("\n权威数据文件：attempt01/all_GT_4x4.csv、all_GT_results.json、fixed_raw_A_stage_4x4.csv、fixed_raw_A_stage_results.json、")
emit("fixed_class_area_size_4x4.csv、joined_GT.csv、verification.json、input_manifest.json；结果后两对话讨论见01号记录。")
with (ROOT / "00_逐GT响应表_全体与固定rawA阶段.txt").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
print(f"Rendered {len(lines)} lines; original computed JSON/CSV unchanged")
