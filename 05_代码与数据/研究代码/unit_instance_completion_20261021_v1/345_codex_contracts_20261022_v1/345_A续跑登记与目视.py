"""Metadata only: record 345b correction scope and four already-viewed PNGs."""
import importlib,ast,difflib
C=importlib.import_module('345_context')
old=(C.HERE/'345_contracts_A.py').read_text(encoding='utf-8');new=(C.HERE/'345_contracts_A_v2.py').read_text(encoding='utf-8')
aa={n.name:ast.dump(n) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
bb={n.name:ast.dump(n) for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)}
assert all(aa[k]==bb[k] for k in ('A1','A2','A3','A4'))
C.text('345_A5及A7测试构造修正.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='345_contracts_A.py',tofile='345_contracts_A_v2.py')))
C.write('345_续跑代码范围与元数据输出修复.json',{'status':'已验证；仅测试构造及续跑入口改变','A1_A4_AST_unchanged':True,
    'changed_functions':['variants','A5','A7的同类无效fixture修正','main(继承A1..4，续跑A5..12)'],
    'metadata_preparer_failure':'345b_prepare_A_v2.py在成功写出并AST核对v2源码后，写345b_前缀diff被345_context只允许345_前缀的断言拒绝；未发生模型/合同失败，A5..12已独立完成。现用345_前缀另存元数据。',
    'authorization_path':str(C.SPEC/'345b_补充裁定_A5测试构造修正.txt'),'authorization_sha256':C.sha(C.SPEC/'345b_补充裁定_A5测试构造修正.txt'),
    '已执行':{'AST_identity_checks':4,'diff_writes':1,'model_forwards':0,'optimizer_steps':0},
    '未执行':['A1..4重跑','模型/配方修改','真实训练'],'未核':['主控独立复核'],'source_sha256':C.sources()})
paths=C.read(C.HERE/'345_result_A11_v2.json')['details']['png_paths']
assert len(paths)==4
C.write('345_result_A11四PNG目视.json',{'status':'已目视','entries':[{'path':p,'sha256':C.sha(p),'viewed':True,
    'observation':'候选青框与实际车/人位置对应；格点按列x/行y排列；右侧generation residual热图和左侧正残余标记均来自初态，不作训练效果。相同seed的R/G初态图一致，无轴转置。'} for p in paths],
    '已执行':{'PNG_views':4,'model_forwards':0,'optimizer_steps':0},'未执行':['训练后残余可视化','真实训练'],
    '未核':['主控独立目视'],'source_sha256':C.sources()})
