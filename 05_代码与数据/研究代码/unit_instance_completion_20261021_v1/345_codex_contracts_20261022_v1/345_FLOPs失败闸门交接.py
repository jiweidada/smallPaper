"""No rerun: seal all 12 passed contracts and the failed FLOPs receipt writer."""
import importlib,json
C=importlib.import_module('345_context')

def link(p):
    p=C.Path(p);return f'[{p.name}]({p.as_posix()})'

def main():
    passed=C.read(C.HERE/'345_result_A合同总表_v2.json');assert len(passed['contracts'])==12
    failed=C.read(C.HERE/'345_result_FLOPs实现失败_停止.json');partial=C.HERE/'345_result_FLOPs覆盖补足.json'
    assert failed['已执行']['head_forwards_completed']==8
    try:json.loads(partial.read_text(encoding='utf-8'))
    except json.JSONDecodeError:incomplete=True
    else:raise AssertionError('Expected partial invalid receipt')
    verified=[]
    for p,h in failed['source_sha256'].items():assert C.sha(p)==h;verified.append({'path':p,'sha256':h,'unchanged':True})
    assert not (C.HERE/'train345.py').exists() and not (C.HERE/'342_开训前审查包_v2.md').exists()
    rows=[]
    for r in passed['contracts']:
        i=r['contract'];p=C.HERE/(f'345_result_A{i:02d}.json' if i<5 else f'345_result_A{i:02d}_v2.json')
        rows.append({'contract':i,'status':r['status'],'independent_or_negative_method':r['independent_or_negative_method'],
            'details_path':str(p),'details_sha256':C.sha(p),'call_counts':r['call_counts'],'source_sha256':r['source_sha256']})
    C.write('345_A逐项状态表_v2.json',{'status':'12项合同已验证；FLOPs报告实现失败，停止于B之前','contracts':rows,
        'scope':'A1..4继承原通过记录；A5..12依据345b正式重跑；主控抽查不替代执行结果。',
        '已执行':passed['已执行'],'未执行':['B','FLOPs完整报告交付','审查包v2','真实训练'],
        '未核':['主控独立复核','FLOPs完整审计'],'source_sha256':C.sources()})
    # Numbers below are copied only from the eight actual stdout records.
    preliminary=[(1,'N',211834880,4923390,216758270),(1,'S',216031744,5068600,221100344),
        (1,'G',212121600,5082870,217204470),(1,'R',211834880,4923390,216758270),
        (2,'N',210991104,4891438,215882542),(2,'S',214663680,5017448,219681128),
        (2,'G',211236864,5029310,216266174),(2,'R',210991104,4891438,215882542)]
    C.write('345_FLOPs部分标准输出登记.json',{'status':'未验证（完整审计报告未交付）；以下仅为实际已打印计量值',
        'rows':[{'seed':s,'arm':a,'registered':r,'manual':m,'sum':n} for s,a,r,m,n in preliminary],
        'data_source':'345_flops_complete.py本轮工具返回的八条stdout；不补造丢失的逐算子事件。',
        'partial_receipt':{'path':str(partial),'sha256':C.sha(partial),'JSON_complete':False,'must_not_use_as_pass_receipt':True},
        'failure_location':'首个不能序列化的参数在aten.ne的args：isfinite检查会与+inf比较；其他位置也可能含-inf掩码常量。shape()把这些标量原样放入元数据，allow_nan=False拒绝。',
        'model_output_scope':'八次前向后的pred_boxes/pred_logits有限性和未分类算子检查已完成；错误发生在最后C.write/json.dump。未进行模型或计数公式修复。',
        '已执行':{'head_forwards':8,'complete_result_writes':0,'optimizer_steps':0},
        '未执行':['FLOPs重跑','B','真实训练'],'未核':['完整FLOPs审计'],'source_sha256':C.sources()})
    C.write('345_B未执行清单_v2.json',{'status':'未执行；停在FLOPs报告序列化失败闸门',
        'rows':[{'item':i,'status':'未验证','execution':'未执行','calls':0} for i in range(1,10)],
        'train345_exists':False,'review_package_v2_exists':False,
        '已执行':{'B_source_writes':0,'B_validation_calls':0,'optimizer_steps':0},
        '未执行':['B全部实现与验收','审查包v2','真实训练','GPU','SSH','队列'],
        '未核':['主控对FLOPs输出故障的裁定'],'source_sha256':C.sources()})
    text='''# 345 续跑一页中文汇总（FLOPs写盘失败闸门）

状态：A的12项合同全部通过。FLOPs八个单窗计算完成，但完整报告因JSON元数据序列化失败而未交付；按345停下，B与审查包v2未执行，真实训练0步。

A1–A4沿用原通过记录，函数AST未变，无重复运行。A5按345b修正：框/类/分数只改有效槽位，排列和数量变化同步处理四字段；N的boxes/logits/query_states逐字节不变。旧seed1前16窗的错误构造独立复现348个非正有效框，被断言拒绝，记为反向测试通过。A7中同类fixture也已修正，模型和冻结模块未改。

A6：四臂对三种真实损失分别求梯度，L_det/L_assoc的mass_head梯度为0或None，L_mass非零。A7：非零S路径的排列、count和空行skip通过。A8：G/R初态的两层hidden、query_states、预测和generation residual逐字节相同。A9：改变G的aux只改变aux，生成计划和预测不变，κ只接L_det。A10：R确实读取residual，临时以mass替换后等于同权重N。A11：错误坐标格式被拒绝或数值明显不同；两seed R/G共4张PNG已逐张目视。A12：一次真实raw global前向，794项state逐字节不变、grad None、输出detach。

续跑实测：64条缓存加载、158次head尝试/149次完成（9次为故意错误输入的预期拒绝）、95次autograd；1次global构造/前向，优化步0。FLOPs另8次head前向；不计作训练。

FLOPs失败：shape()把算子参数中的+inf原样写到JSON（首个在aten.ne/isfinite检查；掩码还可含-inf），allow_nan=False拒绝，生成了不完整文件。实际模型boxes/logits有限性检查已过。不能把部分输出当完整FLOPs验收通过，原失败报告及坏文件SHA保留。

未执行：FLOPs输出修复/重跑、train345.py、B所有验收、审查包v2、真实训练、GPU/SSH/队列。建议的最小后续修复只涉及元数据：将非有限常量写成有类型标签的字符串，并在创建文件前完成严格JSON序列化验证；尚未实施。

按345及345b“任一合同或验收失败就停下”等待主控，不越过闸门进入B。
'''
    C.text('345_一页中文汇总_v2_停止.md',text)
    C.text('345_交接索引_v2.md','\n'.join([
        '# 345 当前交接：12合同通过，FLOPs报告实现失败，B未执行','',
        '本页为活动交接，旧A5停止索引保留。12合同通过后，在FLOPs报告写盘处停下，未创建train345或审查包v2。','',
        '- '+link(C.HERE/'345_一页中文汇总_v2_停止.md'),
        '- '+link(C.HERE/'345_A逐项状态表_v2.json'),
        '- '+link(C.HERE/'345_result_A合同总表_v2.json'),
        '- '+link(C.HERE/'345_result_A05_v2.json')+'：含348个错误有效槽位的独立复现与预期拒绝。',
        '- '+link(C.HERE/'345_result_A11四PNG目视.json'),
        '- '+link(C.HERE/'345_A5及A7测试构造修正.diff'),
        '- '+link(C.HERE/'345_result_FLOPs实现失败_停止.json')+'：完整traceback，失败在JSON写出。',
        '- '+link(C.HERE/'345_FLOPs部分标准输出登记.json')+'：仅登记实际stdout，不冒称完整审计。',
        '- '+link(C.HERE/'345_result_FLOPs覆盖补足.json')+'：已知不完整JSON，禁止作为通过结果。',
        '- '+link(C.HERE/'345_B未执行清单_v2.json'),
        '- '+link(C.HERE/'345_停止交付封存SHA256_v2.json'),'',
        '停止依据：'+link(C.SPEC/'345_授权_继承合同重跑与正式入口验收.txt')+'及'+link(C.SPEC/'345b_补充裁定_A5测试构造修正.txt')+'。本轮未自行修复/重跑FLOPs，未进入B。','']))
    C.write('345_result_FLOPs失败后只读核验.json',{'status':'已验证','source_entries':verified,
        '已执行':{'source_SHA_checks':len(verified),'model_forwards_after_failure':0,'optimizer_steps':0,'B_source_writes':0},
        '未执行':['FLOPs重跑','B','真实训练'],'未核':['主控裁定'],'source_sha256':C.sources()})
    files={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in sorted(C.HERE.rglob('*')) if p.is_file()}
    C.write('345_停止交付封存SHA256_v2.json',{'status':'12合同通过；FLOPs报告失败，B未执行','artifacts':files,
        '已执行':{'resumed_A':passed['已执行'],'FLOPs_head_forwards':8,'optimizer_steps':0,'real_training_steps':0},
        '未执行':['B','审查包v2','真实训练','GPU'],'未核':['FLOPs完整报告','主控独立核对'],'source_sha256':C.sources()})
    print('STOP handoff complete; 12 A contracts passed; FLOPs serialization failed; B unexecuted')

if __name__=='__main__':main()
