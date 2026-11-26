"""Record the required stop. No model/test calls, repairs, retry, or B work."""
import importlib,json
C=importlib.import_module('345_context')

def link(path):
    p=C.Path(path);return f'[{p.name}]({p.as_posix()})'

def main():
    failed=C.read(C.HERE/'345_result_A实现失败_停止.json');assert failed['failed']['contract']==5
    checks=[]
    for path,h in failed['source_sha256'].items():
        assert C.sha(path)==h,path;checks.append({'path':path,'sha256':h,'unchanged':True})
    assert not (C.HERE/'train345.py').exists()
    assert not (C.HERE/'342_开训前审查包_v2.md').exists()
    names=['适配器等价性','独立计划算子','极小u','empty与overflow','N不读P','sg(m)','S排列与count','G/R初态hidden等价','G的aux隔离','R残余接口','坐标反向与PNG','global冻结']
    rows=[]
    for i,name in enumerate(names,1):
        if i<5:
            r=C.read(C.HERE/f'345_result_A{i:02d}.json')
            details=r['details'];differences=[]
            for x in details.get('rows',[]):
                for k in ('max_abs_diff','NumPy_max_abs_diff','independent_geometry_max_abs_diff','analytic_grad_abs_diff','loss_det_abs_diff'):
                    if k in x:differences.append(x[k])
            rows.append({'contract':i,'item':name,'status':'已验证','independent_or_negative_method':r['independent_or_negative_method'],
                'max_reported_abs_difference':max(differences,default=0.),'call_counts':r['call_counts'],
                'evidence_path':str(C.HERE/f'345_result_A{i:02d}.json'),'source_sha256':r['source_sha256']})
        elif i==5:
            rows.append({'contract':i,'item':name,'status':'实现失败','scope':'测试输入构造在有效框正宽高断言处失败，N输出比较未完成。',
                'diagnosis':'variants(boxes)只flip P_boxes，不flip P_valid；real16 batch有零padding框，反序后可能被旧valid标成有效。静态代码及调用计数支持该定位，未修复或重跑。',
                'independent_or_negative_method':'计划用非零读出固定N参数，分别扰动box/class/score/count/order/mask并比较输出字节；本项完整比较未完成。',
                'max_reported_abs_difference':'未核：失败发生在输出之前','evidence_path':str(C.HERE/'345_result_A实现失败_停止.json'),
                'call_counts_since_A04':{k:failed['已执行'][k]-sum(C.read(C.HERE/f'345_result_A{j:02d}.json')['call_counts'].get(k,0) for j in range(1,5)) for k in ('head_attempts','head_completions','negative_controls')},
                'source_sha256':failed['source_sha256']})
        else:rows.append({'contract':i,'item':name,'status':'未验证','reason':'未执行；A5失败后按345停止。','model_calls':0,'source_sha256':C.sources()})
    C.write('345_A逐项状态表_停止.json',{'status':'A5实现失败，停止；A1至A4通过','rows':rows,
        '已执行':failed['已执行'],'未执行':failed['未执行'],'未核':failed['未核'],'source_sha256':C.sources()})
    bitems=['train345.py另存与diff','配置只读及SHA绑定','6轮共享shuffle物化','初态SHA对照','任意epoch内完整状态恢复','SHA/run目录/1218上限守卫','逐步/逐轮日志','8组迷你epoch新进程续跑与正式2步及5种篡改拒绝','训练说明边界']
    C.write('345_B未执行清单.json',{'status':'未执行；A5失败闸门','rows':[{'item':i,'name':x,'status':'未验证','execution':'未执行','calls':0} for i,x in enumerate(bitems,1)],
        'FLOPs_supplement':'未执行；只读查看过既有资源记录的未注册算子名称，未进行新的计量或手算加回。',
        'review_package_v2':'未执行；A/B未完成，不把部分验收写成新通过包。',
        '已执行':{'B_source_writes':0,'B_validation_calls':0,'optimizer_steps':0,'global_forwards':0},
        '未执行':['B全部实现与验收','FLOPs补足','审查包v2','真实训练','GPU','SSH','队列'],
        '未核':['A5后续处理的主控裁定'],'source_sha256':C.sources()})
    summary='''# 345 一页中文汇总（失败闸门）

状态：A5测试输入构造实现失败，已按345停止。A6–A12、FLOPs补足、B正式入口与审查包v2均未执行。模型、配方、旧文件和冻结模块未修改，没有自行重跑。

已通过A1–A4：

- A1：四臂、合成输入及两个seed各前32窗，当前适配器与冻结原模型加载同一非零测试权重；预测、query_states、计划各字段和两层hidden逐字节相同，最大差0。
- A2：独立纯NumPy逐候选质量归一化、逐类逐格cap和退回null，对照当前计划；另逐行列独立计算geometry。列守恒、行容量、退回null、无供应、零质量、同cost质量拆分通过。
- A3：u约8.76e-27，四臂计划前向/反向有限，并对照解析导数；故意使用旧m/u代数的反向测试检出非有限梯度。
- A4：J=0、全padding的受控输入前向与loss有限；6个GT的overflow例匹配4个、2个未匹配，分母仍为6，错误分母4反向对照被检出。

A5在candidate_inputs的有效框宽高断言处停止。代码中只把P_boxes反序而未同步处理P_valid，真实batch的padding零框可能因此被标成有效；这一测试构造需要修正。N对合法P扰动是否严格不变尚未完整验证，不能从此断言作科学判断。未改模型或冻结模块。

实际调用：64条缓存加载；110次head前向尝试、109次完成；44次loss调用；8次autograd；24次显式计划调用、20次NumPy计划对照、16次独立geometry对照。global模型/前向0、优化步0、PNG0，真实训练0。A5后未再执行模型或合同测试。

停下依据：345授权“先做A。A里任一合同失败就停下，不进入B”，以及“任何合同或验收失败：报告实现失败，先停下等主控”。当前等待主控核对与后续裁定。
'''
    C.text('345_一页中文汇总_停止.md',summary)
    C.text('345_交接索引.md','\n'.join([
        '# 345 交接索引：A5实现失败，已停止','',
        'A1–A4已验证；A5未完成；A6–A12/FLOPs/B未执行。没有train345.py或审查包v2，真实训练0步。','',
        '- '+link(C.HERE/'345_一页中文汇总_停止.md'),
        '- '+link(C.HERE/'345_result_A实现失败_停止.json')+'：完整traceback、原始计数、源SHA。',
        '- '+link(C.HERE/'345_A逐项状态表_停止.json')+'：12项逐项状态、方法、最大差和调用数。',
        '- '+link(C.HERE/'345_B未执行清单.json'),
        '- '+link(C.HERE/'345_A_预定验收口径.json')+'：运行前固定的数值/反向测试口径。',
        *['- '+link(C.HERE/f'345_result_A{i:02d}.json') for i in range(1,5)],
        '- '+link(C.HERE/'345_contracts_A.py')+'：失败源保留，variants只翻转boxes的测试构造未修复。',
        '- '+link(C.HERE/'345_numpy_oracle.py')+'：无torch/geom342/冻结代码导入的独立算法。',
        '- '+link(C.HERE/'345_停止交付封存SHA256.json'),'',
        '按 '+link(C.SPEC/'345_授权_继承合同重跑与正式入口验收.txt')+' 的失败闸门等待主控；不得跳过A5进入B。','']))
    C.write('345_result_停止后源文件核验.json',{'status':'已验证（只读SHA核验）','entries':checks,
        '已执行':{'source_SHA_checks':len(checks),'model_forwards_after_failure':0,'optimizer_steps':0,'B_source_writes':0},
        '未执行':['A重跑','B','模型/配方修复','真实训练'],'未核':['主控对失败处理的裁定'],'source_sha256':C.sources()})
    files={str(p):{'sha256':C.sha(p),'bytes':p.stat().st_size} for p in sorted(C.HERE.rglob('*')) if p.is_file()}
    C.write('345_停止交付封存SHA256.json',{'status':'A5实现失败，按345停下等主控','artifacts':files,
        '已执行':failed['已执行'],'未执行':failed['未执行'],'未核':failed['未核'],'source_sha256':C.sources()})
    print(json.dumps({'status':'STOPPED','A_passed':[1,2,3,4],'A_failed':5,'B_executed':False,'calls':failed['已执行'],'sealed_files':len(files)},ensure_ascii=False))

if __name__=='__main__':main()
